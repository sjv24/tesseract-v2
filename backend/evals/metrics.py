"""Pure scoring and progress functions for the eval runner (no Weaviate, no Ollama, so they are easy to test)."""
import math
import re

REFUSAL_PHRASES = (
    "don't know", "do not know", "not contain", "do not provide", "does not provide", "doesn't provide",
    "not provided", "no information", "cannot answer", "can't answer", "not mentioned", "not covered",
    "do not cover", "does not cover", "not specified",
)


def normalize(text: str) -> str:
    """Lower-case, straight quotes, single spaces, so a quote matches however the chunk was cut or line-wrapped."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).lower().strip()


def contains_evidence(chunk_text: str, evidence: list[str]) -> bool:
    chunk = normalize(chunk_text)
    return any(normalize(quote) in chunk for quote in evidence)


def first_hit_rank(chunk_texts: list[str], evidence: list[str]) -> int | None:
    """1-based position of the first retrieved chunk that contains any evidence quote, or None."""
    for rank, text in enumerate(chunk_texts, start=1):
        if contains_evidence(text, evidence):
            return rank
    return None


def rank_and_chars(chunk_texts: list[str], evidence: list[str]) -> dict:
    """The rank of the first chunk holding the evidence, how many characters you read up to and including it,
    and how many characters the whole list holds (so lists of different chunk sizes can be compared fairly)."""
    rank = first_hit_rank(chunk_texts, evidence)
    return {
        "rank": rank,
        "chars_to_hit": sum(len(t) for t in chunk_texts[:rank]) if rank else None,
        "chars": sum(len(t) for t in chunk_texts),
        "sizes": [len(t) for t in chunk_texts],
    }


def chunks_within(sizes: list[int], budget: int) -> int:
    """How many chunks from the start of the list fit inside a character budget."""
    total = count = 0
    for size in sizes:
        total += size
        if total > budget:
            break
        count += 1
    return count


def equal_text_scores(long_lists: list[dict], references: list[dict]) -> dict:
    """For each question, count the long list as a hit only if the evidence shows up within as many characters
    as the reference list contained. This removes the advantage of simply returning bigger chunks."""
    hits = sum(1 for lst, ref in zip(long_lists, references) if lst["chars_to_hit"] is not None and lst["chars_to_hit"] <= ref["chars"])
    ref_hits = sum(1 for ref in references if ref["rank"] is not None)
    return {
        "questions": len(references),
        "hit": hits / len(references),
        "reference_hit": ref_hits / len(references),
        "avg_budget_chars": sum(r["chars"] for r in references) / len(references),
        "avg_chunks": sum(chunks_within(lst["sizes"], ref["chars"]) for lst, ref in zip(long_lists, references)) / len(references),
    }


def retrieval_scores(ranks: list[int | None], k: int = 5) -> dict:
    """hit@k = share of questions with the evidence in the top k; MRR = mean of 1/rank (0 when missed)."""
    if not ranks:
        return {"questions": 0, "hit_at_k": None, "mrr": None}
    return {
        "questions": len(ranks),
        "hit_at_k": sum(1 for r in ranks if r is not None and r <= k) / len(ranks),
        "mrr": sum(1 / r for r in ranks if r is not None) / len(ranks),
    }


def is_refusal(answer: str) -> bool:
    """Heuristic: does the answer say it does not know? (Checked by phrase, so it is a rough measure.)"""
    return any(phrase in normalize(answer) for phrase in REFUSAL_PHRASES)


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile (p in 0..100)."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def progress_line(done: int, total: int, elapsed: float, label: str = "") -> str:
    """e.g. '[ 12/100]  12%  elapsed 04:12  ETA 30:48  q012'. The ETA uses the average time per finished item so far."""
    eta = elapsed / done * (total - done) if done else 0
    width = len(str(total))
    return f"[{done:>{width}}/{total}] {100 * done // total:>3}%  elapsed {format_duration(elapsed)}  ETA {format_duration(eta)}  {label}"
