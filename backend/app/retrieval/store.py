"""One Weaviate connection shared by every search, instead of opening a new one per question."""
import weaviate
from weaviate import WeaviateClient

from app.core.config import settings

_client: WeaviateClient | None = None


def get_client() -> WeaviateClient:
    """Return the shared client, (re)connecting if there is none or the connection dropped."""
    global _client
    if _client is None or not _client.is_connected():
        _client = weaviate.connect_to_local(
            host=settings.weaviate_host, port=settings.weaviate_port, grpc_port=settings.weaviate_grpc_port
        )
    return _client


def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
