import json
import logging
import time

import ollama
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from weaviate.exceptions import WeaviateBaseError

from app.agent import graph as agent
from app.agent.router import classify
from app.generation.generator import generate_stream
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()
log = logging.getLogger("tesseract")

OLLAMA_DOWN = "The language model (Ollama) is not reachable. Start Ollama and try again."
WEAVIATE_DOWN = "The search index (Weaviate) is not reachable. Start Docker and the weaviate container."


def log_request(question: str, category: str, citations: list[dict], seconds: float, streamed: bool) -> None:
    """One JSON line per question: easy to read, grep, or load into a log tool later."""
    log.info(json.dumps({
        "event": "chat",
        "streamed": streamed,
        "question": question,
        "category": category,
        "chunk_ids": [c["chunk_id"] for c in citations],
        "seconds": round(seconds, 2),
    }))


def service_unavailable(error: Exception) -> HTTPException:
    if isinstance(error, WeaviateBaseError):
        log.exception("Weaviate failed")
        return HTTPException(503, WEAVIATE_DOWN)
    log.exception("Ollama failed")
    return HTTPException(503, OLLAMA_DOWN)


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    started = time.perf_counter()
    try:
        result = agent.graph.invoke({"question": request.question})
    except (ConnectionError, ollama.ResponseError, WeaviateBaseError) as error:
        raise service_unavailable(error) from error
    log_request(request.question, result["category"], result["citations"], time.perf_counter() - started, streamed=False)
    return ChatResponse(answer=result["answer"], category=result["category"], citations=result["citations"])


def sse(event: str, data: dict) -> str:
    """One server-sent event: a named message the browser can read as it arrives."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """Like /chat, but sends the sources first and then the answer word by word, so the page can show text at once.

    Events: `meta` (category and citations), many `token` (a piece of the answer), then `done`.
    A failure after the stream has started arrives as an `error` event."""
    started = time.perf_counter()
    try:
        category = classify(request.question)
        plan = agent.prepare(category, request.question)
    except (ConnectionError, ollama.ResponseError, WeaviateBaseError) as error:
        raise service_unavailable(error) from error

    def events():
        yield sse("meta", {"category": category, "citations": plan["citations"]})
        try:
            if "direct_answer" in plan:  # a computed counting answer: nothing to stream from the model
                yield sse("token", {"text": plan["direct_answer"]})
            else:
                for piece in generate_stream(request.question, plan["chunks"], plan["system_prompt"]):
                    yield sse("token", {"text": piece})
        except (ConnectionError, ollama.ResponseError):
            log.exception("Ollama failed during streaming")
            yield sse("error", {"detail": OLLAMA_DOWN})
            return
        seconds = time.perf_counter() - started
        log_request(request.question, category, plan["citations"], seconds, streamed=True)
        yield sse("done", {"seconds": round(seconds, 2)})

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
