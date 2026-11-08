from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    openai_api_key: str | None = None
    # "v1" = the original index, "v2" = the structure-aware chunks (comments have no v2 and always use v1)
    index_version: Literal["v1", "v2"] = "v1"
    # how many tokens the chat model may read; Ollama silently cuts the prompt off beyond this
    ollama_num_ctx: int = 8192

    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")


settings = Settings()
