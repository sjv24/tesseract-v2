from fastapi import APIRouter

from app.generation.generator import generate
from app.retrieval.search import search
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    hits = search("Interstellar_Script", request.question)
    answer = generate(request.question, [hit["text"] for hit in hits])
    return ChatResponse(answer=answer, category="script")
