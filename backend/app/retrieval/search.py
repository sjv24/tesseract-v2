import ollama
import weaviate
from openai import OpenAI
from weaviate.classes.query import MetadataQuery

from app.core.config import settings

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
    vector = EMBEDDERS[embedder](question)
    client = weaviate.connect_to_local()
    try:
        collection = client.collections.use(collection_name)
        result = collection.query.near_vector(
            near_vector=vector,
            limit=k,
            return_metadata=MetadataQuery(distance=True),
        )
        return [
            {"text": o.properties["text"], "distance": o.metadata.distance}
            for o in result.objects
        ]
    finally:
        client.close()
