import ollama
from openai import OpenAI
from weaviate.classes.query import MetadataQuery

from app.core.config import settings
from app.retrieval.store import get_client

OLLAMA_EMBED_MODEL = "embeddinggemma"
OPENAI_EMBED_MODEL = "text-embedding-3-small"


def embed_ollama(text: str) -> list[float]:
    return ollama.embed(model=OLLAMA_EMBED_MODEL, input=text)["embeddings"][0]


def embed_openai(text: str) -> list[float]:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in .env")
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=text)
    return response.data[0].embedding


EMBEDDERS = {"ollama": embed_ollama, "openai": embed_openai}


def search(
    collection_name: str,
    question: str,
    k: int = 5,
    embedder: str = "ollama",
) -> list[dict]:
    """Top-k chunks. Fields a collection does not have (the old index has no section/page) come back as None, and chunk_id falls back to the object's UUID."""
    vector = EMBEDDERS[embedder](question)
    collection = get_client().collections.use(collection_name)
    result = collection.query.near_vector(
        near_vector=vector,
        limit=k,
        return_metadata=MetadataQuery(distance=True),
    )
    return [
        {
            "text": o.properties["text"],
            "distance": o.metadata.distance,
            "chunk_id": o.properties.get("chunk_id") or str(o.uuid),
            "section": o.properties.get("section"),
            "page": o.properties.get("page"),
        }
        for o in result.objects
    ]
