"""Understand what kind of question is being asked about the comments.

Either an OPINION question ("what do viewers think of X": answered by searching comments, optionally only the
positive or only the negative ones) or a STATS question ("how many...", "what share..."), answered by counting.
A small model call turns the question into a structured query; ordinary code then runs it.
"""
from typing import Literal

import ollama
from pydantic import BaseModel

from app.core.config import settings
from app.retrieval import comments_stats

Sentiment = Literal["Positive", "Negative", "Neutral"]

SYSTEM_PROMPT = (
    "You turn a question about YouTube comments on the film Interstellar into a structured query.\n"
    "kind: 'stats' ONLY when the question asks for a number, how many, a share or percentage, a ratio, or which "
    "topics are mentioned most. Questions that start with 'what do viewers...', 'how do people feel...' or "
    "'what do fans...' are 'opinion' questions, even when they name a topic or a tone.\n"
    "metric (stats only, otherwise 'none'): count (how many comments), share (what share, proportion, percent or "
    "percentage), ratio (one tone versus another), top_aspects (only when asked WHICH topics are mentioned most).\n"
    "sentiment: Positive, Negative or Neutral when the question is about comments with that tone (praise, love, like, "
    "admire and enjoy mean Positive; complaints, complain, criticism, dislike and hate mean Negative); Neutral only "
    "when the question asks about neutral comments. If the question implies no tone, sentiment is 'Any'. "
    "For a ratio, sentiment is the first side and versus is the second side; otherwise versus is 'Any'.\n"
    "keyword: the single topic word or name the comments must mention, or an empty string if there is none.\n"
    "Examples:\n"
    "'How many comments mention Tom?' -> stats, count, sentiment Any, keyword Tom.\n"
    "'What percent of comments are positive?' -> stats, share, sentiment Positive, keyword empty.\n"
    "'Which topics come up most?' -> stats, top_aspects, sentiment Any.\n"
    "'What do viewers think about the pacing?' -> opinion, metric none, sentiment Any, keyword pacing.\n"
    "'How do people feel about the casting?' -> opinion, metric none, sentiment Any, keyword casting.\n"
    "'What do people criticise about the dialogue?' -> opinion, metric none, sentiment Negative.\n"
    "'What do fans admire about the set design?' -> opinion, metric none, sentiment Positive."
)


class WireQuery(BaseModel):
    """The shape the model is asked to produce. Every field has an explicit value (no nulls), which small models
    handle far more reliably than 'null'. It is converted to a CommentsQuery right after."""

    kind: Literal["opinion", "stats"]
    metric: Literal["none", "count", "share", "ratio", "top_aspects"]
    sentiment: Literal["Any", "Positive", "Negative", "Neutral"]
    versus: Literal["Any", "Positive", "Negative", "Neutral"]
    keyword: str


class CommentsQuery(BaseModel):
    kind: Literal["opinion", "stats"]
    metric: Literal["count", "share", "ratio", "top_aspects"] | None = None
    sentiment: Sentiment | None = None
    versus: Sentiment | None = None
    keyword: str | None = None

    @classmethod
    def from_wire(cls, wire: WireQuery) -> "CommentsQuery":
        metric = None if wire.metric == "none" or wire.kind == "opinion" else wire.metric
        versus = None if wire.versus == "Any" else wire.versus
        if metric == "ratio" and versus is None:
            metric = "share"  # a ratio needs two sides; with one side the question is really about a share
        return cls(
            kind=wire.kind,
            metric=metric,
            sentiment=None if wire.sentiment == "Any" else wire.sentiment,
            versus=versus,
            keyword=wire.keyword.strip() or None,
        )


def understand(question: str, system_prompt: str = SYSTEM_PROMPT) -> CommentsQuery:
    response = ollama.chat(
        model=settings.router_model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],
        think=False,
        options={"temperature": 0, "num_ctx": settings.ollama_num_ctx},
        format=WireQuery.model_json_schema(),
    )
    return CommentsQuery.from_wire(WireQuery.model_validate_json(response["message"]["content"]))


def run_stats(query: CommentsQuery, rows) -> str:
    """Execute a stats query and phrase the result, always stating what was counted."""
    keyword = query.keyword or None
    scope = f" that mention '{keyword}'" if keyword else ""
    tone = f"{query.sentiment.lower()} " if query.sentiment else ""
    total = f"{len(rows):,}"

    if query.metric == "share":
        r = comments_stats.share(rows, query.sentiment, keyword)
        return (f"{r['share']:.1%} of {r['base_name']}{scope} are {tone}comments: "
                f"{r['matching']:,} of {r['base']:,} (the data set has {total} comments in all).")
    if query.metric == "ratio" and query.sentiment and query.versus:
        r = comments_stats.ratio(rows, query.sentiment, query.versus, keyword)
        value = f"{r['ratio']:.2f} to 1" if r["ratio"] is not None else "undefined (no comments on the second side)"
        return (f"The ratio of {query.sentiment.lower()} to {query.versus.lower()} comments{scope} is {value} "
                f"({r['top']:,} against {r['bottom']:,}).")
    if query.metric == "top_aspects":
        top = comments_stats.top_aspects(rows, query.sentiment, keyword)
        listed = ", ".join(f"{word} ({n:,})" for word, n in top)
        return f"The most mentioned topics among {tone}comments{scope} are: {listed}."
    n = comments_stats.count(rows, query.sentiment, keyword)
    return f"{n:,} {tone}comments{scope} (out of {total} comments in all)."
