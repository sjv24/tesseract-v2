import logging

import ollama
from fastapi import APIRouter, HTTPException
from weaviate.exceptions import WeaviateBaseError

from app.agent.graph import graph
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()
log = logging.getLogger("tesseract")


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = graph.invoke({"question": request.question})
    except (ConnectionError, ollama.ResponseError) as error:
        log.exception("Ollama failed")
        raise HTTPException(503, "The language model (Ollama) is not reachable. Start Ollama and try again.") from error
    except WeaviateBaseError as error:
        log.exception("Weaviate failed")
        raise HTTPException(503, "The search index (Weaviate) is not reachable. Start Docker and the weaviate container.") from error
    return ChatResponse(answer=result["answer"], category=result["category"], citations=result["citations"])
