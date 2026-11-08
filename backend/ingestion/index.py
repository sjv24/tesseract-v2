"""Write chunks and their vectors into Weaviate.

Safety rule: this module only ever touches collections whose name ends in "_v2". The original index
(Interstellar_Script, Interstellar_Textbook, Interstellar_Comments) can never be changed from here.
"""
from weaviate.classes.config import Configure, DataType, Property, Tokenization, VectorDistances
from weaviate.util import generate_uuid5

from ingestion.chunk import Chunk

COLLECTIONS = {"script": "Interstellar_Script_v2", "textbook": "Interstellar_Textbook_v2"}
SAFE_SUFFIX = "_v2"
BATCH_SIZE = 100


def assert_safe_name(name: str) -> None:
    if not name.endswith(SAFE_SUFFIX):
        raise ValueError(
            f"refusing to touch collection {name!r}: ingestion may only write to collections ending in {SAFE_SUFFIX!r}"
        )


def chunk_uuid(chunk_id: str) -> str:
    """A UUID computed from the chunk's own ID: the same chunk always gets the same UUID.

    That is what makes re-running the ingestion safe: Weaviate replaces the object instead of adding a copy.
    """
    return str(generate_uuid5(chunk_id))


def chunk_properties(chunk: Chunk) -> dict:
    """The fields stored next to the vector (they are what citations will be built from)."""
    return {
        "chunk_id": chunk.id,
        "source": chunk.source,
        "position": chunk.position,
        "text": chunk.text,
        "section": chunk.section,
        "chapter": chunk.chapter,
        "page": chunk.page,
    }


def create_collection(client, name: str, recreate: bool = False):
    """Create the collection (or reuse it). With recreate=True an existing one is deleted first."""
    assert_safe_name(name)
    if client.collections.exists(name):
        if not recreate:
            return client.collections.use(name)
        client.collections.delete(name)
    return client.collections.create(
        name,
        # we provide the vectors ourselves (computed with Ollama); cosine distance, same as the original index
        vector_config=Configure.Vectors.self_provided(
            vector_index_config=Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE)
        ),
        # BM25 keyword scoring settings (used later for hybrid search)
        inverted_index_config=Configure.inverted_index(bm25_k1=1.2, bm25_b=0.75),
        properties=[
            Property(name="chunk_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD, index_searchable=False),
            Property(name="source", data_type=DataType.TEXT, tokenization=Tokenization.FIELD, index_searchable=False),
            Property(name="position", data_type=DataType.INT),
            Property(name="text", data_type=DataType.TEXT),
            Property(name="section", data_type=DataType.TEXT),
            Property(name="chapter", data_type=DataType.TEXT),
            Property(name="page", data_type=DataType.INT),
        ],
    )


def upsert_chunks(collection, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    """Insert chunks with their vectors. Chunks that already exist are replaced, not duplicated."""
    assert_safe_name(collection.name)
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    with collection.batch.fixed_size(batch_size=BATCH_SIZE) as batch:
        for chunk, vector in zip(chunks, vectors, strict=True):
            batch.add_object(properties=chunk_properties(chunk), uuid=chunk_uuid(chunk.id), vector=vector)
    failed = collection.batch.failed_objects
    if failed:
        raise RuntimeError(f"{len(failed)} objects failed to insert; first error: {failed[0].message}")


def prune_stale(collection, keep_uuids: set[str]) -> int:
    """Delete objects that are no longer produced by the chunker (for example after changing its settings)."""
    assert_safe_name(collection.name)
    stale = [str(o.uuid) for o in collection.iterator(return_properties=["chunk_id"]) if str(o.uuid) not in keep_uuids]
    for uuid in stale:
        collection.data.delete_by_id(uuid)
    return len(stale)


def count_objects(collection) -> int:
    return collection.aggregate.over_all(total_count=True).total_count


def self_retrieval_rate(collection, chunks: list[Chunk], vectors: list[list[float]], samples: int = 5) -> float:
    """Sanity check: searching with a chunk's OWN vector must return that same chunk as the best match."""
    step = max(1, len(chunks) // samples)
    picks = range(0, len(chunks), step)
    hits = 0
    total = 0
    for i in picks:
        top = collection.query.near_vector(near_vector=vectors[i], limit=1).objects
        total += 1
        hits += bool(top) and str(top[0].uuid) == chunk_uuid(chunks[i].id)
    return hits / total if total else 0.0
