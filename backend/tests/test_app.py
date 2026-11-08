"""Tests for the chat backend: index switch, citations, shared client, error handling. No Ollama or Weaviate needed."""
import json

import pytest
from fastapi.testclient import TestClient
from weaviate.exceptions import WeaviateConnectionError

from app.agent import graph as g
from app.agent.comments_query import CommentsQuery
from app.core.config import settings
from app.generation import generator
from app.main import app
from app.retrieval import store


def fake_hit(n=1):
    return {"text": f"chunk text {n} " * 30, "distance": 0.25, "chunk_id": f"id{n}", "section": "INT. SHIP", "page": 7}


@pytest.fixture
def client(monkeypatch):
    """The real graph, with the router, search and model replaced by fakes."""
    monkeypatch.setattr(g, "classify", lambda question: "script")
    monkeypatch.setattr(g, "search", lambda collection, question, **kw: [fake_hit(1), fake_hit(2)])
    monkeypatch.setattr(g, "generate", lambda question, chunks, prompt: f"answer from {len(chunks)} chunks")
    return TestClient(app)


# --- index switch -------------------------------------------------------------

def test_v1_uses_the_original_collections(monkeypatch):
    monkeypatch.setattr(settings, "index_version", "v1")
    assert g.collection_name("script") == "Interstellar_Script"
    assert g.collection_name("textbook") == "Interstellar_Textbook"


def test_v2_uses_the_new_collections_but_comments_stay_on_the_original(monkeypatch):
    monkeypatch.setattr(settings, "index_version", "v2")
    assert g.collection_name("script") == "Interstellar_Script_v2"
    assert g.collection_name("textbook") == "Interstellar_Textbook_v2"
    assert g.collection_name("comments") == "Interstellar_Comments"


def test_the_switch_changes_what_gets_searched(client, monkeypatch):
    searched = []
    monkeypatch.setattr(g, "search", lambda collection, question, **kw: searched.append(collection) or [fake_hit()])
    monkeypatch.setattr(settings, "index_version", "v2")
    client.post("/chat", json={"question": "Who is TARS?"})
    assert searched == ["Interstellar_Script_v2"]


# --- citations ----------------------------------------------------------------

def test_chat_returns_citations_for_every_retrieved_chunk(client):
    body = client.post("/chat", json={"question": "Who is TARS?"}).json()
    assert body["answer"] == "answer from 2 chunks"
    assert body["category"] == "script"
    first = body["citations"][0]
    assert first["source"] == "script"
    assert first["chunk_id"] == "id1"
    assert first["section"] == "INT. SHIP" and first["page"] == 7
    assert first["score"] == 0.75  # similarity = 1 - cosine distance
    assert len(first["snippet"]) == 200


def test_multi_route_labels_each_citation_with_its_source(client, monkeypatch):
    monkeypatch.setattr(g, "classify", lambda question: "multi")
    body = client.post("/chat", json={"question": "Is the wormhole realistic?"}).json()
    assert [c["source"] for c in body["citations"]] == ["script"] * 2 + ["textbook"] * 2 + ["comments"] * 2


def test_old_index_hits_without_section_or_page_still_work(client, monkeypatch):
    old = {"text": "short", "distance": 0.5, "chunk_id": "some-uuid", "section": None, "page": None}
    monkeypatch.setattr(g, "search", lambda *a, **kw: [old])
    citation = client.post("/chat", json={"question": "Who is TARS?"}).json()["citations"][0]
    assert citation["section"] is None and citation["page"] is None and citation["snippet"] == "short"


# --- errors -------------------------------------------------------------------

def test_ollama_down_gives_a_clear_503_not_a_crash(client, monkeypatch):
    def down(question):
        raise ConnectionError("refused")

    monkeypatch.setattr(g, "classify", down)
    response = client.post("/chat", json={"question": "hi"})
    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]


def test_weaviate_down_gives_a_clear_503_not_a_crash(client, monkeypatch):
    def down(*a, **kw):
        raise WeaviateConnectionError("refused")

    monkeypatch.setattr(g, "search", down)
    response = client.post("/chat", json={"question": "hi"})
    assert response.status_code == 503
    assert "Weaviate" in response.json()["detail"]


# --- generator and shared client ----------------------------------------------

def test_the_model_is_given_an_explicit_context_size(monkeypatch):
    seen = {}

    def fake_chat(**kwargs):
        seen.update(kwargs)
        return {"message": {"content": "ok"}}

    monkeypatch.setattr(generator.ollama, "chat", fake_chat)
    monkeypatch.setattr(settings, "ollama_num_ctx", 4321)
    generator.generate("q", ["a"])
    assert seen["options"]["num_ctx"] == 4321


class FakeClient:
    def __init__(self):
        self.connected, self.closed = True, False

    def is_connected(self):
        return self.connected

    def close(self):
        self.closed, self.connected = True, False


def test_the_weaviate_connection_is_opened_once_and_reused(monkeypatch):
    made = []
    monkeypatch.setattr(store, "_client", None)
    monkeypatch.setattr(store.weaviate, "connect_to_local", lambda **kw: made.append(FakeClient()) or made[-1])
    assert store.get_client() is store.get_client()
    assert len(made) == 1


def test_a_dropped_connection_is_reopened(monkeypatch):
    made = []
    monkeypatch.setattr(store, "_client", None)
    monkeypatch.setattr(store.weaviate, "connect_to_local", lambda **kw: made.append(FakeClient()) or made[-1])
    first = store.get_client()
    first.connected = False
    assert store.get_client() is made[1] and made[1] is not first


def test_close_client_closes_it_and_the_next_call_reconnects(monkeypatch):
    made = []
    monkeypatch.setattr(store, "_client", None)
    monkeypatch.setattr(store.weaviate, "connect_to_local", lambda **kw: made.append(FakeClient()) or made[-1])
    first = store.get_client()
    store.close_client()
    assert first.closed
    store.get_client()
    assert len(made) == 2


def test_router_and_generator_use_the_same_context_size(monkeypatch):
    """Different num_ctx values make Ollama reload the model between the two calls (measured: 100 s vs 18 s per question)."""
    from app.agent import router

    seen = []

    def fake_chat(**kwargs):
        seen.append(kwargs["options"]["num_ctx"])
        return {"message": {"content": '{"category": "script"}'}}

    monkeypatch.setattr(router.ollama, "chat", fake_chat)
    monkeypatch.setattr(generator.ollama, "chat", fake_chat)
    router.classify("q")
    generator.generate("q", ["a"])
    assert seen[0] == seen[1] == settings.ollama_num_ctx


# --- hybrid search and reranking ---------------------------------------------

def test_retrieval_defaults_are_plain_vector_search_without_rerank(monkeypatch):
    calls = []
    monkeypatch.setattr(g, "search", lambda collection, question, **kw: calls.append(kw) or [fake_hit()])
    g._retrieve("script", "q")
    assert calls[0]["mode"] == "vector" and calls[0]["k"] == 5


def test_hybrid_and_rerank_settings_change_how_script_and_textbook_are_searched(monkeypatch):
    from app.retrieval import rerank as rerank_module

    calls = []
    monkeypatch.setattr(g, "search", lambda collection, question, **kw: calls.append(kw) or [fake_hit(n) for n in range(1, 31)])
    monkeypatch.setattr(rerank_module, "get_model", lambda: type("M", (), {"rerank": lambda self, q, docs: [-float(i) for i in range(len(docs))]})())
    monkeypatch.setattr(settings, "retrieval_mode", "hybrid")
    monkeypatch.setattr(settings, "rerank_enabled", True)
    hits = g._retrieve("textbook", "q")
    assert calls[0]["mode"] == "hybrid" and calls[0]["k"] == 30  # fetch many candidates...
    assert len(hits) == 5 and "rerank_score" in hits[0]  # ...keep the best 5


def test_comments_always_use_plain_vector_search(monkeypatch):
    calls = []
    monkeypatch.setattr(g, "search", lambda collection, question, **kw: calls.append(kw) or [fake_hit()])
    monkeypatch.setattr(settings, "retrieval_mode", "hybrid")
    monkeypatch.setattr(settings, "rerank_enabled", True)
    g._retrieve("comments", "q")
    assert calls[0]["mode"] == "vector" and calls[0]["k"] == 5


def test_citation_score_comes_from_the_ranking_that_was_used():
    base = {"source": "script", "collection": "c", "chunk_id": "x", "section": None, "page": None, "text": "t"}
    assert g._citation({**base, "distance": 0.25, "score": None})["score"] == 0.75
    assert g._citation({**base, "distance": None, "score": 0.9})["score"] == 0.9
    assert g._citation({**base, "distance": 0.25, "score": None, "rerank_score": 3.5})["score"] == 3.5


def test_router_accepts_a_different_prompt(monkeypatch):
    from app.agent import router

    seen = {}
    monkeypatch.setattr(router.ollama, "chat", lambda **kw: seen.update(kw) or {"message": {"content": '{"category": "multi"}'}})
    assert router.classify("q", system_prompt="custom") == "multi"
    assert seen["messages"][0]["content"] == "custom"


def test_rerank_puts_the_highest_scoring_chunk_first_and_keeps_only_top_n(monkeypatch):
    from app.retrieval import rerank as rerank_module

    monkeypatch.setattr(rerank_module, "get_model", lambda: type("M", (), {"rerank": lambda self, q, docs: [0.1, 0.9, 0.5]})())
    hits = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    ranked = rerank_module.rerank("q", hits, top_n=2)
    assert [h["text"] for h in ranked] == ["b", "c"]
    assert ranked[0]["rerank_score"] == 0.9
    assert rerank_module.rerank("q", [], top_n=2) == []


# --- streaming ------------------------------------------------------------------

def parse_sse(text):
    """Turn the raw event stream back into [(event, data), ...]."""
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.split("\n")
        events.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return events


@pytest.fixture
def stream_client(monkeypatch):
    from app.api import chat as chat_module

    monkeypatch.setattr(chat_module, "classify", lambda question: "script")
    monkeypatch.setattr(g, "search", lambda collection, question, **kw: [fake_hit(1), fake_hit(2)])
    monkeypatch.setattr(chat_module, "generate_stream", lambda question, chunks, prompt: iter(["Ninety ", "percent."]))
    return TestClient(app)


def test_the_stream_sends_sources_first_then_the_answer_in_pieces_then_done(stream_client):
    response = stream_client.post("/chat/stream", json={"question": "What is TARS's honesty setting?"})
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["meta", "token", "token", "done"]
    assert events[0][1]["category"] == "script" and len(events[0][1]["citations"]) == 2
    assert "".join(data["text"] for name, data in events if name == "token") == "Ninety percent."
    assert events[-1][1]["seconds"] >= 0


def test_a_computed_counting_answer_is_sent_whole_without_calling_the_model(stream_client, monkeypatch):
    from app.api import chat as chat_module

    monkeypatch.setattr(chat_module, "classify", lambda question: "comments")
    monkeypatch.setattr(g, "understand", lambda q: CommentsQuery(kind="stats", metric="count"))
    monkeypatch.setattr(g, "load_comments", lambda: [{"text": "a", "sentiment": [], "aspects": []}])
    monkeypatch.setattr(chat_module, "generate_stream", lambda *a: pytest.fail("no model call for a counting answer"))
    events = parse_sse(stream_client.post("/chat/stream", json={"question": "How many comments are there?"}).text)
    assert [name for name, _ in events] == ["meta", "token", "done"]
    assert "1 comments" in events[1][1]["text"]


def test_streaming_reports_a_service_that_is_down_before_the_stream_starts(stream_client, monkeypatch):
    from app.api import chat as chat_module

    def down(question):
        raise ConnectionError("refused")

    monkeypatch.setattr(chat_module, "classify", down)
    response = stream_client.post("/chat/stream", json={"question": "hi"})
    assert response.status_code == 503 and "Ollama" in response.json()["detail"]


def test_a_failure_in_the_middle_of_the_stream_arrives_as_an_error_event(stream_client, monkeypatch):
    from app.api import chat as chat_module

    def breaks(question, chunks, prompt):
        yield "Ninety "
        raise ConnectionError("model died")

    monkeypatch.setattr(chat_module, "generate_stream", breaks)
    events = parse_sse(stream_client.post("/chat/stream", json={"question": "hi"}).text)
    assert [name for name, _ in events] == ["meta", "token", "error"]


def test_each_chat_logs_one_json_line(client, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="tesseract"):
        client.post("/chat", json={"question": "Who is TARS?"})
    line = json.loads(next(r.message for r in caplog.records if r.name == "tesseract"))
    assert line["event"] == "chat" and line["category"] == "script" and line["chunk_ids"] == ["id1", "id2"]
    assert line["question"] == "Who is TARS?" and line["streamed"] is False


def test_generate_stream_yields_the_pieces_the_model_sends(monkeypatch):
    sent = {}

    def fake_chat(**kwargs):
        sent.update(kwargs)
        return iter([{"message": {"content": "He"}}, {"message": {"content": "llo"}}])

    monkeypatch.setattr(generator.ollama, "chat", fake_chat)
    assert list(generator.generate_stream("q", ["a"])) == ["He", "llo"]
    assert sent["stream"] is True and sent["options"]["num_ctx"] == settings.ollama_num_ctx


def test_the_weaviate_address_comes_from_settings(monkeypatch):
    seen = {}
    monkeypatch.setattr(store, "_client", None)
    monkeypatch.setattr(store.weaviate, "connect_to_local", lambda **kw: seen.update(kw) or FakeClient())
    monkeypatch.setattr(settings, "weaviate_host", "weaviate")
    store.get_client()
    assert seen == {"host": "weaviate", "port": 8080, "grpc_port": 50051}


def test_cors_origins_come_from_settings():
    from app.main import app as application

    cors = next(m for m in application.user_middleware if m.cls.__name__ == "CORSMiddleware")
    assert "http://localhost:5173" in cors.kwargs["allow_origins"]


# --- prompt hardening ---------------------------------------------------------

def test_hardening_adds_the_injection_rules_only_when_switched_on(monkeypatch):
    plain = generator._messages("q", ["chunk"], "base prompt")[0]["content"]
    assert plain == "base prompt"
    monkeypatch.setattr(settings, "harden_prompt", True)
    hardened = generator._messages("q", ["chunk"], "base prompt")[0]["content"]
    assert hardened.startswith("base prompt") and "untrusted" in hardened and "reveal" in hardened
