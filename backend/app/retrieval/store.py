"""One Weaviate connection shared by every search, instead of opening a new one per question."""
import weaviate
from weaviate import WeaviateClient

_client: WeaviateClient | None = None


def get_client() -> WeaviateClient:
    """Return the shared client, (re)connecting if there is none or the connection dropped."""
    global _client
    if _client is None or not _client.is_connected():
        _client = weaviate.connect_to_local()
    return _client


def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
