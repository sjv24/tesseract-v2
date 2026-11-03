"""Turn text into vectors (embeddings) with a local Ollama model.

An embedding is a list of numbers that captures the MEANING of a text: texts with similar meaning get
vectors that point in similar directions, which is what makes "search by meaning" possible.
"""
import time
from collections.abc import Callable

import ollama

EMBED_MODEL = "embeddinggemma"  # same model the original index used, so only the chunking differs
BATCH_SIZE = 32  # texts sent to Ollama per request
MAX_ATTEMPTS = 3  # a request is tried up to 3 times before we give up

Vector = list[float]


def embed_batch(texts: list[str]) -> list[Vector]:
    """One request to Ollama for several texts at once (much faster than one request per text)."""
    response = ollama.embed(model=EMBED_MODEL, input=texts)
    return response["embeddings"]


def embed_texts(
    texts: list[str],
    batch_size: int = BATCH_SIZE,
    embed_fn: Callable[[list[str]], list[Vector]] = embed_batch,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Vector]:
    """Embed any number of texts, in batches, returning vectors in the SAME order as the texts.

    `embed_fn` and `sleep` are parameters only so the tests can replace them with fakes.
    """
    vectors: list[Vector] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(embed_with_retries(batch, embed_fn, sleep))
    return vectors


def embed_with_retries(
    batch: list[str],
    embed_fn: Callable[[list[str]], list[Vector]],
    sleep: Callable[[float], None],
) -> list[Vector]:
    """Retry a failed request a couple of times (with a growing pause) before giving up."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = embed_fn(batch)
        except Exception:
            if attempt == MAX_ATTEMPTS:
                raise
            sleep(2**attempt)  # wait 2 s, then 4 s
            continue
        if len(result) != len(batch):  # a wrong answer is a bug, not a glitch: do not retry
            raise ValueError(f"asked for {len(batch)} embeddings but got {len(result)}")
        return result
    raise AssertionError("unreachable")
