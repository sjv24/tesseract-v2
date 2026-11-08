"""Tests for the chat backend: index switch, citations, shared client, error handling. No Ollama or Weaviate needed."""
import pytest
from fastapi.testclient import TestClient
from weaviate.exceptions import WeaviateConnectionError

from app.agent import graph as g
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
    monkeypatch.setattr(store.weaviate, "connect_to_local", lambda: made.append(FakeClient()) or made[-1])
    assert store.get_client() is store.get_client()
    assert len(made) == 1


def test_a_dropped_connection_is_reopened(monkeypatch):
    made = []
    monkeypatch.setattr(store, "_client", None)
    monkeypatch.setattr(store.weaviate, "connect_to_local", lambda: made.append(FakeClient()) or made[-1])
    first = store.get_client()
    first.connected = False
    assert store.get_client() is made[1] and made[1] is not first


def test_close_client_closes_it_and_the_next_call_reconnects(monkeypatch):
    made = []
    monkeypatch.setattr(store, "_client", None)
    monkeypatch.setattr(store.weaviate, "connect_to_local", lambda: made.append(FakeClient()) or made[-1])
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
