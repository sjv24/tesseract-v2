from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class Citation(BaseModel):
    """One retrieved chunk the answer was based on."""

    source: Literal["script", "textbook", "comments"]
    collection: str
    chunk_id: str
    section: str | None = None
    page: int | None = None
    score: float  # cosine similarity, 1.0 = identical direction
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    category: Literal["script", "textbook", "comments", "multi"]
    citations: list[Citation] = []
