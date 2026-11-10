"""Tests for ingestion/embed.py and ingestion/index.py.

Most tests need no services (fake functions replace Ollama). The last group talks to the real Weaviate,
using a throwaway collection called "Test_Ingest_v2" that it deletes afterwards; it is skipped when
Weaviate is not running. These tests never touch the real Interstellar collections.
"""
import re

import pytest
import weaviate

from ingestion import embed, index
from ingestion.chunk import Chunk


# --- embed_texts -----------------------------------------------------------------------------
def fake_embedder(calls):
    """A stand-in for Ollama: records the size of each request and returns tiny 2-number vectors."""

    def embed_fn(batch):
        calls.append(len(batch))
        return [[float(len(text)), 1.0] for text in batch]

    return embed_fn


def test_texts_are_sent_in_batches_and_order_is_kept():
    calls = []
    texts = ["a" * n for n in range(1, 71)]  # 70 texts of different lengths
    vectors = embed.embed_texts(texts, batch_size=32, embed_fn=fake_embedder(calls))
    assert calls == [32, 32, 6]
    assert [v[0] for v in vectors] == [float(n) for n in range(1, 71)]  # vector i belongs to text i


def test_embedding_nothing_returns_nothing():
    assert embed.embed_texts([], embed_fn=fake_embedder([])) == []


def test_a_failed_request_is_retried_then_succeeds():
    attempts = []

    def flaky(batch):
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("Ollama hiccup")
        return [[1.0] for _ in batch]

    pauses = []
    vectors = embed.embed_texts(["x"], embed_fn=flaky, sleep=pauses.append)
    assert vectors == [[1.0]]
    assert len(attempts) == 3
    assert pauses == [2, 4]  # waited 2 s, then 4 s (the fake sleep only records)


def test_a_request_that_keeps_failing_eventually_raises():
    def always_fails(batch):
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        embed.embed_texts(["x"], embed_fn=always_fails, sleep=lambda s: None)


def test_a_wrong_number_of_embeddings_is_an_error_not_a_retry():
    attempts = []

    def short_answer(batch):
        attempts.append(1)
        return [[1.0]]  # asked for 2, got 1

    with pytest.raises(ValueError):
        embed.embed_texts(["a", "b"], embed_fn=short_answer, sleep=lambda s: None)
    assert len(attempts) == 1


# --- pure helpers in index.py ----------------------------------------------------------------
def make_chunk(i, section="Some section"):
    return Chunk(id=f"id{i}", source="script", position=i, text=f"text {i}", section=section, chapter="", page=i)


def test_only_v2_collections_may_be_written():
    index.assert_safe_name("Interstellar_Script_v2")
    for original in ("Interstellar_Script", "Interstellar_Textbook", "Interstellar_Comments"):
        with pytest.raises(ValueError):
            index.assert_safe_name(original)


def test_the_real_collections_configured_for_ingestion_are_all_safe():
    for name in index.COLLECTIONS.values():
        index.assert_safe_name(name)


def test_chunk_uuid_is_stable_valid_and_different_per_chunk():
    first = index.chunk_uuid("abc123")
    assert first == index.chunk_uuid("abc123")
    assert first != index.chunk_uuid("abc124")
    assert re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", first)


def test_chunk_properties_contain_everything_a_citation_needs():
    props = index.chunk_properties(make_chunk(7, section="INT. KITCHEN - DAY"))
    assert props == {
        "chunk_id": "id7",
        "source": "script",
        "position": 7,
        "text": "text 7",
        "section": "INT. KITCHEN - DAY",
        "chapter": "",
        "page": 7,
    }


def test_chunks_and_vectors_must_match_in_number():
    class FakeCollection:
        name = "Whatever_v2"

    with pytest.raises(ValueError):
        index.upsert_chunks(FakeCollection(), [make_chunk(1)], [])


def test_upsert_refuses_an_unsafe_collection_even_if_called_directly():
    class OriginalCollection:
        name = "Interstellar_Script"

    with pytest.raises(ValueError):
        index.upsert_chunks(OriginalCollection(), [make_chunk(1)], [[0.0]])
    with pytest.raises(ValueError):
        index.prune_stale(OriginalCollection(), set())


# --- against the real Weaviate (skipped if it is not running) --------------------------------
def weaviate_is_running() -> bool:
    try:
        client = weaviate.connect_to_local()
    except Exception:
        return False
    client.close()
    return True


needs_weaviate = pytest.mark.skipif(not weaviate_is_running(), reason="Weaviate is not running (docker start weaviate)")


@pytest.fixture
def client():
    connection = weaviate.connect_to_local()
    yield connection
    connection.close()


@needs_weaviate
def test_upsert_is_idempotent_prunes_stale_objects_and_finds_itself(client):
    name = "Test_Ingest_v2"
    collection = index.create_collection(client, name, recreate=True)
    try:
        chunks = [make_chunk(i) for i in (1, 2, 3)]
        vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]  # three different directions

        index.upsert_chunks(collection, chunks, vectors)
        assert index.count_objects(collection) == 3

        index.upsert_chunks(collection, chunks, vectors)  # run it again
        assert index.count_objects(collection) == 3  # replaced, not duplicated

        kept = chunks[:2]  # the chunker "stopped producing" chunk 3
        removed = index.prune_stale(collection, {index.chunk_uuid(c.id) for c in kept})
        assert removed == 1
        assert index.count_objects(collection) == 2

        assert index.self_retrieval_rate(collection, kept, vectors[:2], samples=2) == 1.0
    finally:
        client.collections.delete(name)


@needs_weaviate
def test_a_created_collection_stores_every_field_and_can_be_searched_by_keyword(client):
    name = "Test_Ingest_v2"
    collection = index.create_collection(client, name, recreate=True)
    try:
        chunk = Chunk(id="k1", source="textbook", position=1, text="Gargantua bends light", section="Lensing",
                      chapter="8 Imaging Gargantua", page=88)
        index.upsert_chunks(collection, [chunk], [[1.0, 0.0, 0.0, 0.0]])
        found = collection.query.bm25(query="Gargantua", limit=1).objects[0]
        assert found.properties["chapter"] == "8 Imaging Gargantua"
        assert found.properties["page"] == 88
        assert found.properties["chunk_id"] == "k1"
    finally:
        client.collections.delete(name)


@needs_weaviate
def test_recreate_really_starts_from_scratch(client):
    name = "Test_Ingest_v2"
    collection = index.create_collection(client, name, recreate=True)
    try:
        index.upsert_chunks(collection, [make_chunk(1)], [[1.0, 0.0, 0.0, 0.0]])
        assert index.count_objects(collection) == 1
        fresh = index.create_collection(client, name, recreate=True)
        assert index.count_objects(fresh) == 0
    finally:
        client.collections.delete(name)
