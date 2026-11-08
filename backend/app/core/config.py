from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    openai_api_key: str | None = None
    # "v1" = the original index, "v2" = the structure-aware chunks (comments have no v2 and always use v1)
    index_version: Literal["v1", "v2"] = "v1"
    # "vector" = meaning-based search; "hybrid" also matches exact keywords (BM25). Applies to script and textbook.
    retrieval_mode: Literal["vector", "hybrid"] = "vector"
    hybrid_alpha: float = 0.75  # 1.0 = all vector, 0.0 = all keyword
    # re-score the top candidates with a cross-encoder and keep the best 5 (script and textbook only)
    rerank_enabled: bool = False
    rerank_candidates: int = 30
    # how many tokens the chat model may read; Ollama silently cuts the prompt off beyond this
    ollama_num_ctx: int = 8192

    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")


settings = Settings()
