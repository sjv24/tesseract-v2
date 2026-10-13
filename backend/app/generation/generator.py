import ollama

CHAT_MODEL = "qwen3:8b"

DEFAULT_SYSTEM_PROMPT = (
    "You answer questions about the film Interstellar using ONLY the provided "
    "excerpts. If the excerpts do not contain the answer, say you don't know."
)


def build_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(chunks))
    return f"Excerpts:\n{context}\n\nQuestion: {question}"


def generate(
    question: str,
    chunks: list[str],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_prompt(question, chunks)},
        ],
        think=False,
        options={"temperature": 0.2},
    )
    return response["message"]["content"]
