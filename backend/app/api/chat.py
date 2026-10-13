from fastapi import APIRouter

from app.agent.graph import graph
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = graph.invoke({"question": request.question})
    return ChatResponse(answer=result["answer"], category=result["category"])
