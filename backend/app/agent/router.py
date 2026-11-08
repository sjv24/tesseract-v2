from typing import Literal

import ollama
from pydantic import BaseModel

from app.core.config import settings

ROUTER_MODEL = "qwen3:8b"

Category = Literal["script", "textbook", "comments", "multi"]

SYSTEM_PROMPT = (
    "You route questions about the film Interstellar to one knowledge source.\n"
    "- script: the screenplay. Plot, characters, dialogue, events and what happens in scenes, "
    "including questions about a character's choices or personality.\n"
    "- textbook: the book 'The Science of Interstellar'. Real physics (black holes, wormholes, gravity, "
    "relativity, time dilation), the scientists and engineers behind the film, and how the film was made "
    "(visual effects, the book author's own account).\n"
    "- comments: YouTube viewer comments. Choose this ONLY when the question asks what viewers, fans, "
    "commenters or audiences think, feel or say, or asks for counts or shares of comments. "
    "A question that does not mention viewers or opinions is never 'comments'.\n"
    "- multi: the question needs more than one source, for example comparing the film with the real "
    "science, or asking how accurate something is and how viewers reacted.\n"
    "Answer with the single best category."
)


class Route(BaseModel):
    category: Category


def classify(question: str, system_prompt: str = SYSTEM_PROMPT) -> Category:
    response = ollama.chat(
        model=ROUTER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        think=False,
        # same num_ctx as the generator: a different value makes Ollama reload the model between the two calls
        options={"temperature": 0, "num_ctx": settings.ollama_num_ctx},
        format=Route.model_json_schema(),
    )
    return Route.model_validate_json(response["message"]["content"]).category
