from fastapi import APIRouter

from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    return ChatResponse(
        answer=f"(stub) You asked: {request.question}",
        category="script",
    )