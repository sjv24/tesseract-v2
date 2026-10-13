import ollama
import weaviate
from weaviate.classes.query import MetadataQuery

EMBED_MODEL = "embeddinggemma"


def embed(text: str) -> list[float]:
    return ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0]


def search(collection_name: str, question: str, k: int = 5) -> list[dict]:
    client = weaviate.connect_to_local()
    try:
        collection = client.collections.use(collection_name)
        result = collection.query.near_vector(
            near_vector=embed(question),
            limit=k,
            return_metadata=MetadataQuery(distance=True),
        )
        return [
            {"text": o.properties["text"], "distance": o.metadata.distance}
            for o in result.objects
        ]
    finally:
        client.close()