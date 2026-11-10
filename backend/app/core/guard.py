"""A first line of defence against prompt injection: recognise obvious attempts before the model sees them.

This is a deny-list of common phrasings, not a complete defence. A determined attacker can word things differently,
and the model's own prompt cannot be trusted to refuse (measured in EVALS.md). Its value is that it is deterministic,
costs nothing, and stops the most common attempts ("ignore all previous instructions", "repeat your system prompt")
without a model call.
"""
import re

PATTERNS = [
    r"\b(ignore|disregard|forget)\b.{0,30}\b(previous|prior|above|earlier|all)\b.{0,20}\b(instructions?|prompts?|rules|excerpts)\b",
    r"\b(reveal|repeat|print|show|display|tell me|output)\b.{0,40}\b(system prompt|your (instructions|prompt|rules)|hidden instructions)\b",
    r"\b(end|finish|start|begin)\s+your\s+(answer|reply|response)\s+with\b",
    r"\byou are now\b",
    r"\b(pretend|act) (to be|as)\b",
]
_COMPILED = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in PATTERNS]

DECLINE_ANSWER = (
    "I don't know how to help with that: I only answer questions about the film Interstellar from my sources "
    "(the screenplay, 'The Science of Interstellar' and viewer comments), and I don't follow instructions written inside a question."
)


def looks_like_injection(question: str) -> bool:
    return any(pattern.search(question) for pattern in _COMPILED)
