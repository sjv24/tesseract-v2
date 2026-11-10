"""Can the reranker's best score tell "the excerpts answer this" from "they do not"?

The idea: if even the best-scoring chunk is a poor match, refuse without asking the model. This module measures what
such a gate would do to the refusal rates, choosing the threshold on one half of the questions and checking it on the
other half, so the number reported is not just fitted to the questions it is reported on.

    python -m evals.run_evals --mode gate --input evals/results/<full run>.jsonl
"""
from collections.abc import Callable


def candidate_thresholds(scores: list[float]) -> list[float]:
    ordered = sorted(set(scores))
    return [(a + b) / 2 for a, b in zip(ordered, ordered[1:], strict=False)]


def gate_outcome(rows: list[dict], threshold: float) -> dict:
    """What happens if every question whose best score is below the threshold is refused outright.
    Each row: {"score": float, "answerable": bool, "model_refused": bool}."""
    unanswerable = [r for r in rows if not r["answerable"]]
    answerable = [r for r in rows if r["answerable"]]

    def refused(row: dict) -> bool:
        return row["score"] < threshold or row["model_refused"]

    return {
        "threshold": round(threshold, 3),
        "correct_refusal": sum(refused(r) for r in unanswerable) / len(unanswerable) if unanswerable else None,
        "false_refusal": sum(refused(r) for r in answerable) / len(answerable) if answerable else None,
    }


def pick_threshold(rows: list[dict], max_false_refusal: float = 0.10) -> float | None:
    """The threshold giving the best correct-refusal rate while keeping false refusals at or under the limit.
    Returns None when no gate helps (then the model's own refusals are left alone)."""
    best, best_score = None, gate_outcome(rows, float("-inf"))["correct_refusal"] or 0.0
    for t in candidate_thresholds([r["score"] for r in rows]):
        outcome = gate_outcome(rows, t)
        if outcome["false_refusal"] is None or outcome["false_refusal"] > max_false_refusal:
            continue
        if outcome["correct_refusal"] > best_score:
            best, best_score = t, outcome["correct_refusal"]
    return best


def split_halves(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Alternate rows between two halves (rows are in question-id order, so this mixes topics and sources)."""
    return rows[0::2], rows[1::2]


def cross_check(rows: list[dict], pick: Callable = pick_threshold) -> dict:
    """Pick the threshold on each half and measure it on the other half."""
    first, second = split_halves(rows)
    report = {"no_gate": gate_outcome(rows, float("-inf")), "folds": []}
    for train, test in ((first, second), (second, first)):
        t = pick(train)
        report["folds"].append({
            "threshold": t,
            "test_without_gate": gate_outcome(test, float("-inf")),
            "test_with_gate": gate_outcome(test, t) if t is not None else None,
        })
    return report
