from collections.abc import Iterator

import ollama

from app.core.config import settings

DEFAULT_SYSTEM_PROMPT = (
    "You answer questions about the film Interstellar using ONLY the provided "
    "excerpts. If the excerpts do not contain the answer, say you don't know."
)


def build_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(chunks))
    return f"Excerpts:\n{context}\n\nQuestion: {question}"


def _messages(question: str, chunks: list[str], system_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_prompt(question, chunks)},
    ]


def _options() -> dict:
    return {"temperature": 0.2, "num_ctx": settings.ollama_num_ctx}


def generate(
    question: str,
    chunks: list[str],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    response = ollama.chat(
        model=settings.chat_model,
        messages=_messages(question, chunks, system_prompt),
        think=False,
        options=_options(),
    )
    return response["message"]["content"]


def generate_stream(
    question: str,
    chunks: list[str],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Iterator[str]:
    """Same as generate, but yields the answer piece by piece as the model writes it."""
    stream = ollama.chat(
        model=settings.chat_model,
        messages=_messages(question, chunks, system_prompt),
        think=False,
        options=_options(),
        stream=True,
    )
    for part in stream:
        yield part["message"]["content"]
