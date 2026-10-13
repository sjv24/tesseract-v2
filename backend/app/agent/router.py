from typing import Literal

import ollama
from pydantic import BaseModel

ROUTER_MODEL = "qwen3:8b"

Category = Literal["script", "textbook", "comments", "multi"]

SYSTEM_PROMPT = (
    "You route questions about the film Interstellar to one knowledge source. "
    "Answer 'script' for plot, characters, dialogue and events in the film. "
    "Answer 'textbook' for real science: physics, black holes, wormholes, "
    "gravity, relativity, time dilation. "
    "Answer 'comments' for what fans or viewers think, feel or say about the film. "
    "Answer 'multi' when the question needs more than one of these, for example "
    "comparing a scene with the real science, or asking how accurate something is."
)


class Route(BaseModel):
    category: Category


def classify(question: str) -> Category:
    response = ollama.chat(
        model=ROUTER_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        think=False,
        options={"temperature": 0},
        format=Route.model_json_schema(),
    )
    return Route.model_validate_json(response["message"]["content"]).category
