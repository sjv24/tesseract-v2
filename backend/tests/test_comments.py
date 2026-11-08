"""Tests for counting questions about the comments and for the comments agent. No Ollama or Weaviate needed."""
import pytest

from app.agent import graph as g
from app.agent.comments_query import CommentsQuery, run_stats
from app.retrieval import comments_stats as stats

ROWS = [
    {"text": "The soundtrack by Hans Zimmer is amazing", "sentiment": ["Positive"], "aspects": ["soundtrack", "Zimmer", "movie"]},
    {"text": "Hans Zimmer ruined the sound mixing", "sentiment": ["Negative"], "aspects": ["Zimmer", "mixing"]},
    {"text": "The ending was confusing", "sentiment": ["Negative"], "aspects": ["ending"]},
    {"text": "I love the ending, pending my second viewing", "sentiment": ["Positive", "Positive"], "aspects": ["Ending", "ending"]},
    {"text": "Watched it yesterday", "sentiment": [], "aspects": []},
]


def test_parse_row_turns_the_stored_text_back_into_lists():
    row = stats.parse_row({"text": "t", "sentiment": "['Negative', 'Neutral']", "aspects": "['ending']"})
    assert row == {"text": "t", "sentiment": ["Negative", "Neutral"], "aspects": ["ending"]}
    assert stats.parse_row({"text": "t", "sentiment": "[]", "aspects": "[]"})["sentiment"] == []


def test_keyword_matches_whole_words_only_and_ignores_case():
    assert stats.count(ROWS, keyword="ending") == 2  # "pending" must not count as "ending"
    assert stats.count(ROWS, keyword="hans zimmer") == 2


def test_count_filters_by_sentiment_and_keyword_together():
    assert stats.count(ROWS, sentiment="Negative") == 2
    assert stats.count(ROWS, sentiment="Negative", keyword="zimmer") == 1
    assert stats.count(ROWS) == 5


def test_share_states_its_base():
    everything = stats.share(ROWS, keyword="zimmer")
    assert (everything["matching"], everything["base"], everything["base_name"]) == (2, 5, "all comments")
    negative = stats.share(ROWS, sentiment="Negative")
    assert negative["base"] == 4  # only the comments that carry a sentiment label
    assert negative["share"] == 0.5
    assert stats.share([], sentiment="Negative")["share"] == 0.0


def test_ratio_and_division_by_zero():
    assert stats.ratio(ROWS, "Positive", "Negative")["ratio"] == 1.0
    assert stats.ratio(ROWS, "Positive", "Neutral")["ratio"] is None


def test_top_aspects_merge_case_skip_generic_words_and_count_a_comment_once():
    top = dict(stats.top_aspects(ROWS, n=10))
    assert top["ending"] == 2  # "Ending" and "ending" in one comment count once, plus the other comment
    assert top["zimmer"] == 2
    assert "movie" not in top  # generic
    assert [w for w, _ in stats.top_aspects(ROWS, sentiment="Negative", n=10)] != []


@pytest.mark.parametrize("query,expected", [
    (CommentsQuery(kind="stats", metric="count", keyword="ending"), "2 comments that mention 'ending' (out of 5 comments in all)."),
    (CommentsQuery(kind="stats", metric="share", sentiment="Negative"), "50.0% of comments with a sentiment label are negative comments: 2 of 4"),
    (CommentsQuery(kind="stats", metric="ratio", sentiment="Positive", versus="Negative"), "ratio of positive to negative comments is 1.00 to 1 (2 against 2)"),
])
def test_run_stats_phrases_the_result_and_states_what_was_counted(query, expected):
    assert expected in run_stats(query, ROWS)


def test_run_stats_top_aspects_lists_words_with_counts():
    answer = run_stats(CommentsQuery(kind="stats", metric="top_aspects"), ROWS)
    assert "ending (2)" in answer and "zimmer (2)" in answer


# --- the agent ----------------------------------------------------------------

def hit(n=1):
    return {"text": f"comment {n}", "distance": 0.2, "score": None, "chunk_id": f"c{n}", "section": None, "page": None}


def test_a_stats_question_is_computed_not_searched(monkeypatch):
    monkeypatch.setattr(g, "understand", lambda q: CommentsQuery(kind="stats", metric="count", keyword="ending"))
    monkeypatch.setattr(g, "load_comments", lambda: ROWS)
    monkeypatch.setattr(g, "search", lambda *a, **kw: pytest.fail("a counting question must not use vector search"))
    monkeypatch.setattr(g, "generate", lambda *a, **kw: pytest.fail("the model must not rephrase computed numbers"))
    result = g.comments_agent({"question": "How many comments mention the ending?"})
    assert "2 comments" in result["answer"] and result["citations"] == []


def test_an_opinion_question_with_a_tone_searches_only_comments_with_that_tone(monkeypatch):
    seen = {}
    monkeypatch.setattr(g, "understand", lambda q: CommentsQuery(kind="opinion", sentiment="Negative"))
    monkeypatch.setattr(g, "search", lambda collection, q, **kw: seen.update(kw) or [hit(1), hit(2)])
    monkeypatch.setattr(g, "generate", lambda question, chunks, prompt: f"{len(chunks)} comments summarised")
    result = g.comments_agent({"question": "What do viewers dislike?"})
    assert seen["sentiment"] == "Negative"
    assert result["answer"] == "2 comments summarised" and len(result["citations"]) == 2


def test_an_unparseable_query_falls_back_to_plain_search(monkeypatch):
    def broken(question):
        raise ValueError("not json")

    seen = {}
    monkeypatch.setattr(g, "understand", broken)
    monkeypatch.setattr(g, "search", lambda collection, q, **kw: seen.update(kw) or [hit()])
    monkeypatch.setattr(g, "generate", lambda *a, **kw: "ok")
    assert g.comments_agent({"question": "What do viewers think?"})["answer"] == "ok"
    assert seen["sentiment"] is None


def test_the_model_output_is_converted_so_any_and_none_become_real_nulls(monkeypatch):
    from app.agent import comments_query as cq

    reply = '{"kind": "opinion", "metric": "count", "sentiment": "Any", "versus": "Any", "keyword": "  pacing "}'
    monkeypatch.setattr(cq.ollama, "chat", lambda **kw: {"message": {"content": reply}})
    query = cq.understand("What do viewers think about the pacing?")
    assert query.kind == "opinion"
    assert query.metric is None  # an opinion question never carries a metric
    assert query.sentiment is None and query.versus is None
    assert query.keyword == "pacing"


def test_a_stats_reply_keeps_its_metric_and_tone(monkeypatch):
    from app.agent import comments_query as cq

    reply = '{"kind": "stats", "metric": "ratio", "sentiment": "Positive", "versus": "Negative", "keyword": ""}'
    monkeypatch.setattr(cq.ollama, "chat", lambda **kw: {"message": {"content": reply}})
    query = cq.understand("positive to negative ratio?")
    assert (query.metric, query.sentiment, query.versus, query.keyword) == ("ratio", "Positive", "Negative", None)


def test_a_ratio_with_only_one_side_becomes_a_share(monkeypatch):
    from app.agent import comments_query as cq

    reply = '{"kind": "stats", "metric": "ratio", "sentiment": "Negative", "versus": "Any", "keyword": ""}'
    monkeypatch.setattr(cq.ollama, "chat", lambda **kw: {"message": {"content": reply}})
    query = cq.understand("What fraction of the comments are negative?")
    assert query.metric == "share" and query.sentiment == "Negative"
