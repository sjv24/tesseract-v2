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
HOLDOUT = Path(__file__).with_name("routing_holdout.jsonl")
RESULTS_DIR = Path(__file__).with_name("results")
INDEXES = ("v1", "v2")
# Whole-chatbot setups compared by --mode full. "baseline" is the original pipeline; "final" has every improvement on.
SMALL_MODEL = "hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M"
PIPELINE_DEFAULTS = {
    "index_version": "v1", "retrieval_mode": "vector", "rerank_enabled": False,
    "top_k": 5, "chat_model": "qwen3:8b", "router_model": "qwen3:8b", "strict_prompt": False,
}
_FINAL = {"index_version": "v2", "retrieval_mode": "hybrid", "rerank_enabled": True}
PIPELINES = {
    name: {**PIPELINE_DEFAULTS, **overrides}
    for name, overrides in {
        "baseline": {},
        "final": _FINAL,
        "final-strict": {**_FINAL, "strict_prompt": True},  # final + the stricter answering rules
        "final-k3": {**_FINAL, "top_k": 3},  # only the 3 best chunks reach the model
        "final-small": {**_FINAL, "chat_model": SMALL_MODEL, "router_model": SMALL_MODEL},  # a 4B model instead of 8B
    }.items()
}


def apply_pipeline(name: str) -> None:
    """Switch the chatbot's settings (index, search style, reranking) for the rest of this process."""
    for field, value in PIPELINES[name].items():
        setattr(settings, field, value)
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


# Each retrieval setup to compare. "40 chunks" lists exist only for the equal-text comparison.
RETRIEVAL_CONFIGS = {
    "v1 vector": {"index": "v1", "k": 5},
    "v2 vector": {"index": "v2", "k": 5},
    "v1 vector (40 chunks)": {"index": "v1", "k": 40},
    "v1 hybrid a=0.5": {"index": "v1", "k": 5, "mode": "hybrid", "alpha": 0.5},
    "v2 hybrid a=0.5": {"index": "v2", "k": 5, "mode": "hybrid", "alpha": 0.5},
    "v2 hybrid a=0.75": {"index": "v2", "k": 5, "mode": "hybrid", "alpha": 0.75},
    "v1 vector + rerank": {"index": "v1", "k": 5, "fetch": 30, "rerank": True},
    "v2 vector + rerank": {"index": "v2", "k": 5, "fetch": 30, "rerank": True},
    "v1 hybrid + rerank": {"index": "v1", "k": 5, "fetch": 30, "mode": "hybrid", "alpha": 0.5, "rerank": True},
    "v2 hybrid + rerank": {"index": "v2", "k": 5, "fetch": 30, "mode": "hybrid", "alpha": 0.75, "rerank": True},
    # the routing choice between script and textbook is ignored: both are searched and the reranker picks the best five
    "v2 hybrid + rerank, both sources": {"index": "v2", "k": 5, "fetch": 20, "mode": "hybrid", "alpha": 0.75, "rerank": True, "both": True},
    "v2 hybrid + rerank, 50 candidates": {"index": "v2", "k": 5, "fetch": 50, "mode": "hybrid", "alpha": 0.75, "rerank": True},
}
EQUAL_TEXT = ("v1 vector (40 chunks)", "v2 vector")  # (list to cut down, reference whose text size sets the budget)


def retrieve(config: dict, question: str, source_name: str) -> list[dict]:
    use_index(config["index"])
    hits = []
    for name in ("script", "textbook") if config.get("both") else (source_name,):
        hits += agent.search(
            agent.collection_name(name), question, k=config.get("fetch", config["k"]),
            embedder=agent.SOURCES[name]["embedder"], mode=config.get("mode", "vector"), alpha=config.get("alpha", 0.5),
        )
    if config.get("rerank"):
        from app.retrieval.rerank import rerank  # imported here so the other modes do not need the reranker installed
        hits = rerank(question, hits, top_n=config["k"])
    return hits


def run_retrieval(questions: list[dict], out_path: Path, config_names: list[str]) -> dict:
    labelled = [q for q in questions if q.get("gold_evidence")]
    texts = {}
    for version in INDEXES:
        use_index(version)
        texts[version] = {s: fetch_all_chunk_texts(agent.collection_name(s)) for s in ("script", "textbook")}
    excluded = unlocatable_labels(labelled, "v1", texts["v1"]) | unlocatable_labels(labelled, "v2", texts["v2"])
    scored = [q for q in labelled if q["id"] not in excluded]
    print(f"{len(labelled)} questions have evidence; {len(excluded)} excluded because a quote is not inside one chunk of "
          f"both indexes ({sorted(excluded) or 'none'}); scoring {len(scored)}.\n")

    records = []
    started = time.time()
    with out_path.open("w", encoding="utf-8") as out:
        for done, q in enumerate(scored, start=1):
            record = {"id": q["id"], "type": q["type"], "route": q["gold_route"], "configs": {}}
            for name in config_names:
                chunk_texts = [h["text"] for h in retrieve(RETRIEVAL_CONFIGS[name], q["question"], q["gold_route"])]
                record["configs"][name] = metrics.rank_and_chars(chunk_texts, q["gold_evidence"])
            records.append(record)
            out.write(json.dumps(record) + "\n")
            out.flush()
            ranks = "  ".join(f"{n}={record['configs'][n]['rank']}" for n in config_names[:3])
            print(metrics.progress_line(done, len(scored), time.time() - started, f"{q['id']}  ranks: {ranks}"), flush=True)
    return summarize_retrieval(records, config_names)


def summarize_retrieval(records: list[dict], config_names: list[str]) -> dict:
    def block(rows):
        return {n: metrics.retrieval_scores([r["configs"][n]["rank"] for r in rows]) for n in config_names}

    summary = {"overall": block(records)}
    for key in ("route", "type"):
        for value in sorted({r[key] for r in records}):
            summary[f"{key}={value}"] = block([r for r in records if r[key] == value])
    long_list, reference = EQUAL_TEXT
    if long_list in config_names and reference in config_names:
        summary["equal_text"] = metrics.equal_text_scores(
            [r["configs"][long_list] for r in records], [r["configs"][reference] for r in records]
        )
    return summary


def run_routing(questions: list[dict], out_path: Path, router_prompt: str | None = None) -> dict:
    """Router only (about 2 s per question). Scores the main golden set and the separate hold-out set."""
    from app.agent import router

    prompt = router_prompt or router.SYSTEM_PROMPT
    items = [{**q, "set": "golden"} for q in questions if q.get("gold_route")]
    items += [{**q, "set": "holdout"} for q in load_dataset(HOLDOUT)]
    records, started = [], time.time()
    with out_path.open("w", encoding="utf-8") as out:
        for done, q in enumerate(items, start=1):
            got = classify(q["question"], prompt)
            records.append({"id": q["id"], "set": q["set"], "gold": q["gold_route"], "got": got, "question": q["question"]})
            out.write(json.dumps(records[-1]) + "\n")
            print(metrics.progress_line(done, len(items), time.time() - started, f"{q['id']}  {q['gold_route']} -> {got}"), flush=True)
    return summarize_routing(records)


def summarize_routing(records: list[dict]) -> dict:
    summary = {}
    for name in ("golden", "holdout"):
        rows = [r for r in records if r["set"] == name]
        summary[name] = {
            "questions": len(rows),
            "accuracy": sum(r["gold"] == r["got"] for r in rows) / len(rows) if rows else None,
            "per_route": {g: sum(r["got"] == g for r in rows if r["gold"] == g) / sum(1 for r in rows if r["gold"] == g)
                          for g in sorted({r["gold"] for r in rows})},
            "wrong": [f"{r['id']}: {r['gold']} -> {r['got']}  {r['question']}" for r in rows if r["gold"] != r["got"]],
        }
    return summary


def run_full(questions: list[dict], out_path: Path, pipelines: list[str] | None = None) -> dict:
    pipelines = pipelines or list(PIPELINES)
    records = []
    started = time.time()
    with out_path.open("w", encoding="utf-8") as out:
        for done, q in enumerate(questions, start=1):
            record = {"id": q["id"], "type": q["type"], "gold_route": q.get("gold_route"), "answerable": q["answerable"]}
            try:
                for name in pipelines:
                    apply_pipeline(name)  # the router model is part of a pipeline, so each one routes for itself
                    t0 = time.time()
                    route = classify(q["question"])
                    route_seconds = round(time.time() - t0, 1)
                    t0 = time.time()
                    result = AGENTS[route]({"question": q["question"]})
                    record[name] = {
                        "route": route,
                        "route_seconds": route_seconds,
                        "answer": result["answer"],
                        "seconds": round(time.time() - t0, 1),
                        "refused": metrics.is_refusal(result["answer"]),
                        "citations": [c["chunk_id"] for c in result["citations"]],
                        "contexts": result.get("contexts", []),  # exactly what the model read, kept for judging
                    }
            except (ConnectionError, ollama.ResponseError, WeaviateBaseError, RuntimeError) as error:
                record["error"] = f"{type(error).__name__}: {error}"
            records.append(record)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            note = record.get("error") or " ".join(f"{n}: {record[n]['route']} {record[n]['seconds']}s" for n in pipelines)
            print(metrics.progress_line(done, len(questions), time.time() - started, f"{q['id']}  {note}"), flush=True)
    return summarize_full(records, pipelines)


def run_gate(results_path: Path, out_path: Path) -> dict:
    """Relevance gate analysis (see evals/relevance_gate.py): needs Weaviate, not the answer model."""
    from evals import relevance_gate

    apply_pipeline("final")
    records = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {q["id"]: q for q in load_dataset()}
    rows, started = [], time.time()
    searchable = [r for r in records if "error" not in r and route_of(r, "final") in ("script", "textbook")]
    for done, record in enumerate(searchable, start=1):
        question = by_id[record["id"]]["question"]
        hits = agent._retrieve(route_of(record, "final"), question)
        rows.append({
            "id": record["id"], "score": max(h["rerank_score"] for h in hits), "answerable": record["answerable"],
            "model_refused": metrics.is_refusal(record["final"]["answer"]),
        })
        print(metrics.progress_line(done, len(searchable), time.time() - started, f"{record['id']} top score {rows[-1]['score']:.2f}"), flush=True)
    out_path.write_text("".join(json.dumps(r) + chr(10) for r in rows), encoding="utf-8")
    return relevance_gate.cross_check(sorted(rows, key=lambda r: r["id"]))


def run_robustness(out_path: Path) -> dict:
    """Hostile and odd inputs through the whole chatbot, with the final pipeline's settings."""
    from evals import robustness

    apply_pipeline("final")
    results, started = [], time.time()
    for done, case in enumerate(robustness.CASES, start=1):
        answer = agent.graph.invoke({"question": case["question"]})["answer"]
        results.append(robustness.score_case(case, answer))
        print(metrics.progress_line(done, len(robustness.CASES), time.time() - started, f"{case['id']} {'ok' if results[-1]['ok'] else 'FAILED'}"), flush=True)
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in results), encoding="utf-8")
    return robustness.summarize_robustness(results)


def route_of(record: dict, pipeline: str) -> str:
    """The route a pipeline chose. (Older result files stored one shared route at the top level.)"""
    return record[pipeline].get("route", record.get("route"))


def summarize_full(records: list[dict], pipelines: list[str]) -> dict:
    ok = [r for r in records if "error" not in r]
    routed = [r for r in ok if r["gold_route"]]
    unanswerable = [r for r in ok if not r["answerable"]]
    answerable = [r for r in ok if r["answerable"]]
    summary = {"questions": len(records), "errors": len(records) - len(ok)}
    for name in pipelines:
        seconds = [r[name]["seconds"] for r in ok]
        summary[name] = {
            "routing_accuracy": sum(route_of(r, name) == r["gold_route"] for r in routed) / len(routed) if routed else None,
            "correct_refusal_rate": sum(metrics.is_refusal(r[name]["answer"]) for r in unanswerable) / len(unanswerable) if unanswerable else None,
            "false_refusal_rate": sum(metrics.is_refusal(r[name]["answer"]) for r in answerable) / len(answerable) if answerable else None,
            "answer_seconds_p50": metrics.percentile(seconds, 50),
            "answer_seconds_p95": metrics.percentile(seconds, 95),
        }
    return summary


def print_summary(mode: str, summary: dict) -> None:
    print("\n" + "=" * 78)
    if mode == "retrieval":
        for name, block in summary.items():
            if name == "equal_text":
                continue
            print(f"\n{name}")
            for config, row in block.items():
                print(f"  {config:<26}{row['questions']:>3} q   hit@5 {row['hit_at_k']:>5.0%}   MRR {row['mrr']:.3f}")
        if "equal_text" in summary:
            eq = summary["equal_text"]
            print(f"\nEqual text: v1 with as many chunks as fit in v2's top-5 text (on average {eq['avg_chunks']:.0f} chunks "
                  f"= {eq['avg_budget_chars']:.0f} characters) finds the evidence {eq['hit']:.0%} of the time "
                  f"(v2 top-5: {eq['reference_hit']:.0%}).")
    else:
        print(json.dumps(summary, indent=2))
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the chatbot on the golden set, old index vs new.")
    parser.add_argument("--mode", choices=["retrieval", "full", "routing", "comments", "judge", "gate", "robustness"], default="retrieval")
    parser.add_argument("--limit", type=int, help="only the first N questions (for a quick try)")
    parser.add_argument("--only-labelled", action="store_true", help="only questions that have evidence quotes (about half the set)")
    parser.add_argument("--with-unanswerable", action="store_true", help="with --only-labelled: also keep the unanswerable questions")
    parser.add_argument("--no-faithfulness", action="store_true", help="judge mode: grade correctness only (about half the time)")
    parser.add_argument("--input", type=Path, help="the results file of a --mode full run (judge mode)")
    parser.add_argument("--pipelines", nargs="+", choices=list(PIPELINES), help="whole-chatbot setups to compare (full mode)")
    parser.add_argument("--configs", nargs="+", choices=list(RETRIEVAL_CONFIGS), help="retrieval setups to compare (retrieval mode)")
    args = parser.parse_args(argv)

    questions = load_dataset()
    if args.only_labelled:  # the questions the judge can grade, plus the unanswerable ones that test refusals
        questions = [q for q in questions if q.get("gold_evidence") or (args.with_unanswerable and not q["answerable"])]
    questions = questions[: args.limit]
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"{stamp}-{args.mode}.jsonl"
    try:
        if args.mode == "retrieval":
            summary = run_retrieval(questions, out_path, args.configs or list(RETRIEVAL_CONFIGS))
        elif args.mode == "comments":
            from evals import comments_eval

            started = time.time()
            summary = comments_eval.run_comments_eval(
                load_dataset(),
                lambda done, total, label: print(metrics.progress_line(done, total, time.time() - started, label), flush=True),
            )
        elif args.mode == "judge":
            from evals import judge

            started = time.time()
            summary = judge.run_judge(
                args.input, load_dataset(), out_path,
                lambda done, total, label: print(metrics.progress_line(done, total, time.time() - started, label), flush=True),
                faithfulness=not args.no_faithfulness,
            )
        elif args.mode == "robustness":
            summary = run_robustness(out_path)
        elif args.mode == "gate":
            summary = run_gate(args.input, out_path)
        elif args.mode == "routing":
            summary = run_routing(questions, out_path)
        else:
            summary = run_full(questions, out_path, args.pipelines)
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
