"""Run the golden set against the old (v1) and new (v2) index and print the scores.

    cd backend
    python -m evals.run_evals --mode retrieval      # fast (minutes): is the evidence in the top 5 chunks?
    python -m evals.run_evals --mode full           # slow (an hour or so on a CPU): router + answer for every question
    python -m evals.run_evals --mode retrieval --limit 10     # try it on the first 10 questions

Every question prints a progress line with elapsed time and an ETA. Raw results are written one line per
question to evals/results/ (git-ignored) as they finish, so an interrupted run keeps what it has.
"""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import ollama
import weaviate
from weaviate.exceptions import WeaviateBaseError

from app.agent import graph as agent
from app.agent.router import classify
from app.core.config import settings
from app.retrieval.store import close_client
from evals import metrics

DATASET = Path(__file__).with_name("dataset.jsonl")
RESULTS_DIR = Path(__file__).with_name("results")
INDEXES = ("v1", "v2")
AGENTS = {"script": agent.script_agent, "textbook": agent.textbook_agent, "comments": agent.comments_agent, "multi": agent.multi_agent}


def load_dataset(path: Path = DATASET) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def use_index(version: str) -> None:
    """Point the chatbot at one index for the rest of this process (collection_name reads this on every call)."""
    settings.index_version = version


def fetch_all_chunk_texts(collection_name: str) -> list[str]:
    client = weaviate.connect_to_local()
    try:
        return [o.properties["text"] for o in client.collections.use(collection_name).iterator()]
    finally:
        client.close()


def unlocatable_labels(questions: list[dict], version: str, texts_by_source: dict[str, list[str]]) -> set[str]:
    """Question ids whose evidence quote never appears whole inside ONE chunk of this index.
    That happens when the old 245-character chunks cut a quote in half; such a question cannot be scored fairly."""
    missing = set()
    for q in questions:
        texts = texts_by_source[q["gold_route"]]
        if not any(metrics.contains_evidence(t, q["gold_evidence"]) for t in texts):
            missing.add(q["id"])
    return missing


def run_retrieval(questions: list[dict], out_path: Path) -> dict:
    labelled = [q for q in questions if q.get("gold_evidence")]
    texts = {}
    for version in INDEXES:
        use_index(version)
        texts[version] = {s: fetch_all_chunk_texts(agent.collection_name(s)) for s in ("script", "textbook")}
    skipped = {v: unlocatable_labels(labelled, v, texts[v]) for v in INDEXES}
    excluded = skipped["v1"] | skipped["v2"]
    scored = [q for q in labelled if q["id"] not in excluded]
    print(f"{len(labelled)} questions have evidence; {len(excluded)} excluded because a quote is not inside one chunk of "
          f"both indexes ({sorted(excluded) or 'none'}); scoring {len(scored)}.\n")

    records = []
    started = time.time()
    with out_path.open("w", encoding="utf-8") as out:
        for done, q in enumerate(scored, start=1):
            record = {"id": q["id"], "type": q["type"], "route": q["gold_route"]}
            for version in INDEXES:
                use_index(version)
                source = agent.SOURCES[q["gold_route"]]
                collection = agent.collection_name(q["gold_route"])
                hits = agent.search(collection, q["question"], embedder=source["embedder"])
                record[version] = {"rank": metrics.first_hit_rank([h["text"] for h in hits], q["gold_evidence"])}
            records.append(record)
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(metrics.progress_line(done, len(scored), time.time() - started, f"{q['id']}  v1 rank={record['v1']['rank']}  v2 rank={record['v2']['rank']}"), flush=True)
    return summarize_retrieval(records)


def summarize_retrieval(records: list[dict]) -> dict:
    def block(rows):
        return {v: metrics.retrieval_scores([r[v]["rank"] for r in rows]) for v in INDEXES}

    summary = {"overall": block(records)}
    for key in ("route", "type"):
        for value in sorted({r[key] for r in records}):
            summary[f"{key}={value}"] = block([r for r in records if r[key] == value])
    return summary


def run_full(questions: list[dict], out_path: Path) -> dict:
    records = []
    started = time.time()
    with out_path.open("w", encoding="utf-8") as out:
        for done, q in enumerate(questions, start=1):
            record = {"id": q["id"], "type": q["type"], "gold_route": q.get("gold_route"), "answerable": q["answerable"]}
            try:
                t0 = time.time()
                route = classify(q["question"])  # the router does not depend on the index, so ask it once
                record["route"], record["route_seconds"] = route, round(time.time() - t0, 1)
                for version in INDEXES:
                    use_index(version)
                    t0 = time.time()
                    result = AGENTS[route]({"question": q["question"]})
                    record[version] = {
                        "answer": result["answer"],
                        "seconds": round(time.time() - t0, 1),
                        "refused": metrics.is_refusal(result["answer"]),
                        "citations": [c["chunk_id"] for c in result["citations"]],
                    }
            except (ConnectionError, ollama.ResponseError, WeaviateBaseError, RuntimeError) as error:
                record["error"] = f"{type(error).__name__}: {error}"
            records.append(record)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            note = record.get("error", f"route={record.get('route')} v1 {record['v1']['seconds']}s v2 {record['v2']['seconds']}s")
            print(metrics.progress_line(done, len(questions), time.time() - started, f"{q['id']}  {note}"), flush=True)
    return summarize_full(records)


def summarize_full(records: list[dict]) -> dict:
    ok = [r for r in records if "error" not in r]
    routed = [r for r in ok if r["gold_route"]]
    summary = {
        "questions": len(records),
        "errors": len(records) - len(ok),
        "routing_accuracy": sum(r["route"] == r["gold_route"] for r in routed) / len(routed) if routed else None,
        "router_seconds_p50": metrics.percentile([r["route_seconds"] for r in ok], 50),
    }
    for version in INDEXES:
        unanswerable = [r for r in ok if not r["answerable"]]
        answerable = [r for r in ok if r["answerable"]]
        seconds = [r[version]["seconds"] for r in ok]
        summary[version] = {
            "correct_refusal_rate": sum(r[version]["refused"] for r in unanswerable) / len(unanswerable) if unanswerable else None,
            "false_refusal_rate": sum(r[version]["refused"] for r in answerable) / len(answerable) if answerable else None,
            "answer_seconds_p50": metrics.percentile(seconds, 50),
            "answer_seconds_p95": metrics.percentile(seconds, 95),
        }
    return summary


def print_summary(mode: str, summary: dict) -> None:
    print("\n" + "=" * 78)
    if mode == "retrieval":
        print(f"{'slice':<28}{'n':>4}   {'v1 hit@5':>9}{'v2 hit@5':>10}   {'v1 MRR':>7}{'v2 MRR':>8}")
        for name, row in summary.items():
            v1, v2 = row["v1"], row["v2"]
            print(f"{name:<28}{v1['questions']:>4}   {v1['hit_at_k']:>9.0%}{v2['hit_at_k']:>10.0%}   {v1['mrr']:>7.3f}{v2['mrr']:>8.3f}")
    else:
        print(json.dumps(summary, indent=2))
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the chatbot on the golden set, old index vs new.")
    parser.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    parser.add_argument("--limit", type=int, help="only the first N questions (for a quick try)")
    args = parser.parse_args(argv)

    questions = load_dataset()[: args.limit]
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"{stamp}-{args.mode}.jsonl"
    try:
        summary = (run_retrieval if args.mode == "retrieval" else run_full)(questions, out_path)
    except weaviate.exceptions.WeaviateConnectionError:
        print("Weaviate is not reachable on localhost:8080. Start Docker Desktop, then: docker start weaviate")
        return 2
    except ConnectionError:
        print("Ollama is not reachable on localhost:11434. Start the Ollama app and try again.")
        return 2
    finally:
        close_client()
    print_summary(args.mode, summary)
    (RESULTS_DIR / f"{stamp}-{args.mode}-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"raw results: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
