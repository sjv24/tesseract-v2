from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    openai_api_key: str | None = None
    # where Weaviate is; inside Docker Compose the host is the service name "weaviate"
    weaviate_host: str = "localhost"
    weaviate_port: int = 8080
    weaviate_grpc_port: int = 50051
    # which web pages may call the API (comma separated)
    cors_origins: str = "http://localhost:5173,http://localhost:3000"
    # "v1" = the original index, "v2" = the structure-aware chunks (comments have no v2 and always use v1)
    index_version: Literal["v1", "v2"] = "v1"
    # "vector" = meaning-based search; "hybrid" also matches exact keywords (BM25). Applies to script and textbook.
    retrieval_mode: Literal["vector", "hybrid"] = "vector"
    hybrid_alpha: float = 0.75  # 1.0 = all vector, 0.0 = all keyword
    # re-score the top candidates with a cross-encoder and keep the best 5 (script and textbook only)
    rerank_enabled: bool = False
    rerank_candidates: int = 30
    # chunks passed to the model per source (fewer is faster; the reranker makes a small number more reliable)
    top_k: int = 5
    # extra answering rules (copy names, numbers and key phrases exactly as the excerpts word them); off until measured
    strict_prompt: bool = False
    # the language models; the router and the comments-question parser share router_model
    chat_model: str = "qwen3:8b"
    router_model: str = "qwen3:8b"
    # how many tokens the chat model may read; Ollama silently cuts the prompt off beyond this
    ollama_num_ctx: int = 8192

    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")


settings = Settings()
