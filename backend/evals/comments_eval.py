"""Scoring for the comments source: did the question become the right structured query, and does the tone filter work?

Run through `python -m evals.run_evals --mode comments`.
"""
import json
from pathlib import Path

from app.agent.comments_query import CommentsQuery, understand
from app.retrieval.search import search

COMMENTS_HOLDOUT = Path(__file__).with_name("comments_holdout.jsonl")


def query_matches(got: CommentsQuery, gold: dict) -> bool:
    """Every field the gold query cares about must match. gold["sentiment"] and gold["keyword"] are lists of
    acceptable values (null allowed), because some questions can reasonably be read more than one way."""
    if got.kind != gold["kind"]:
        return False
    if got.sentiment not in gold["sentiment"]:
        return False
    if gold["kind"] == "opinion":
        return True
    keyword = got.keyword.lower() if got.keyword else None
    return got.metric == gold["metric"] and got.versus == gold.get("versus") and keyword in gold["keyword"]


def tone_match_rate(question: str, sentiment: str, use_filter: bool) -> float:
    """Share of the top-5 comments for the question that carry the wanted tone label."""
    hits = search("Interstellar_Comments", question, embedder="openai", sentiment=sentiment if use_filter else None)
    return sum(sentiment in h["sentiment"] for h in hits) / len(hits) if hits else 0.0


def run_comments_eval(golden: list[dict], progress) -> dict:
    items = [{**q, "set": "golden"} for q in golden if q.get("gold_query")]
    items += [{**json.loads(line), "set": "holdout"} for line in COMMENTS_HOLDOUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = []
    for done, q in enumerate(items, start=1):
        gold = q["gold_query"]
        got = understand(q["question"])
        ok = query_matches(got, gold)
        record = {"id": q["id"], "set": q["set"], "kind": gold["kind"], "ok": ok, "question": q["question"], "got": got.model_dump()}
        wanted = [s for s in gold["sentiment"] if s]
        if gold["kind"] == "opinion" and wanted == gold["sentiment"] and len(wanted) == 1:  # exactly one tone is expected
            record["tone_without_filter"] = tone_match_rate(q["question"], wanted[0], use_filter=False)
            record["tone_with_filter"] = tone_match_rate(q["question"], wanted[0], use_filter=True)
        records.append(record)
        progress(done, len(items), f"{q['id']} {'ok ' if ok else 'WRONG'} {q['question'][:50]}")
    return summarize_comments(records)


def summarize_comments(records: list[dict]) -> dict:
    def rate(rows):
        return sum(r["ok"] for r in rows) / len(rows) if rows else None

    summary = {}
    for name in ("golden", "holdout"):
        rows = [r for r in records if r["set"] == name]
        summary[name] = {
            "questions": len(rows),
            "query_accuracy": rate(rows),
            "stats_accuracy": rate([r for r in rows if r["kind"] == "stats"]),
            "opinion_accuracy": rate([r for r in rows if r["kind"] == "opinion"]),
        }
    toned = [r for r in records if "tone_with_filter" in r]
    if toned:
        summary["tone_filter"] = {
            "questions": len(toned),
            "tone_in_top5_without_filter": sum(r["tone_without_filter"] for r in toned) / len(toned),
            "tone_in_top5_with_filter": sum(r["tone_with_filter"] for r in toned) / len(toned),
        }
    summary["wrong"] = [f"{r['id']}: {r['question']}  ->  {r['got']}" for r in records if not r["ok"]]
    return summary
