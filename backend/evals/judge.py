"""Answer-quality judging with a language model (LLM-as-judge), on top of the saved results of --mode full.

Two questions are asked about each answer to a question that has labelled evidence:
  correctness   does the answer say what the evidence passage says?   (needs the gold evidence quote)
  faithfulness  is every claim in the answer supported by the excerpts the chatbot was shown?

Limits worth knowing: the judge is the same small local model that wrote the answers (it can favour its own style),
it sees only short instructions, and its verdicts are not perfect. Treat the numbers as a screening tool, and spot-check
the disagreements by hand (the verdict and reason are saved for every answer).

    python -m evals.run_evals --mode judge --input evals/results/<full run>.jsonl
"""
import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import ollama
from pydantic import BaseModel

from app.core.config import settings

JUDGE_MODEL = "qwen3:8b"  # fixed, so judgements stay comparable when the chatbot's own model changes


class Verdict(BaseModel):
    reason: str
    verdict: Literal["yes", "partly", "no"]


CORRECTNESS_PROMPT = (
    "You grade an answer to a question about the film Interstellar. You are given the QUESTION, a passage of EVIDENCE "
    "that is known to contain the right answer, and the ANSWER to grade.\n"
    "Reply with a one-sentence reason, then a verdict: 'yes' if the answer gives the information the evidence contains, "
    "'partly' if it is related and only partly right or incomplete, 'no' if it is wrong, off topic or says it does not know."
)

FAITHFULNESS_PROMPT = (
    "You check whether an ANSWER is supported by the EXCERPTS it was written from. Ignore whether the answer is "
    "correct in real life; only check support.\n"
    "Reply with a one-sentence reason, then a verdict: 'yes' if every claim in the answer is supported by the excerpts, "
    "'partly' if some claims are supported and some are not, 'no' if the main claims are not in the excerpts."
)


def correctness_message(question: str, answer: str, evidence: list[str]) -> str:
    quotes = "\n".join(f"- {e}" for e in evidence)
    return f"QUESTION: {question}\n\nEVIDENCE (the passage contains this):\n{quotes}\n\nANSWER: {answer}"


def faithfulness_message(answer: str, contexts: list[str]) -> str:
    excerpts = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return f"EXCERPTS:\n{excerpts}\n\nANSWER: {answer}"


def ask_judge(system_prompt: str, message: str) -> Verdict:
    response = ollama.chat(
        model=JUDGE_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": message}],
        think=False,
        options={"temperature": 0, "num_ctx": settings.ollama_num_ctx, "num_predict": 120},
        format=Verdict.model_json_schema(),
    )
    return Verdict.model_validate_json(response["message"]["content"])


def judge_record(record: dict, question: str, evidence: list[str], pipelines: list[str], ask: Callable = ask_judge) -> dict:
    """Judge both pipelines' answers for one question. A refusal is 'no' for correctness without asking the judge."""
    out = {}
    for name in pipelines:
        side = record[name]
        if side["refused"] or not side["answer"].strip():
            out[name] = {"correct": "no", "correct_reason": "the answer refused or was empty", "faithful": None}
            continue
        correct = ask(CORRECTNESS_PROMPT, correctness_message(question, side["answer"], evidence))
        entry = {"correct": correct.verdict, "correct_reason": correct.reason, "faithful": None}
        if side.get("contexts"):
            faithful = ask(FAITHFULNESS_PROMPT, faithfulness_message(side["answer"], side["contexts"]))
            entry.update(faithful=faithful.verdict, faithful_reason=faithful.reason)
        out[name] = entry
    return out


def summarize_judgements(judged: list[dict], pipelines: list[str]) -> dict:
    summary = {"questions": len(judged)}
    for name in pipelines:
        correct = [j[name]["correct"] for j in judged]
        faithful = [j[name]["faithful"] for j in judged if j[name]["faithful"] is not None]
        summary[name] = {
            "correct": correct.count("yes") / len(correct) if correct else None,
            "partly_correct": correct.count("partly") / len(correct) if correct else None,
            "incorrect_or_refused": correct.count("no") / len(correct) if correct else None,
            "faithful": faithful.count("yes") / len(faithful) if faithful else None,
            "faithfulness_judged": len(faithful),
        }
    return summary


def run_judge(results_path: Path, dataset: list[dict], out_path: Path, progress, ask: Callable = ask_judge) -> dict:
    by_id = {q["id"]: q for q in dataset}
    records = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [r for r in records if "error" not in r and by_id.get(r["id"], {}).get("gold_evidence")]
    pipelines = [k for k in records[0] if isinstance(records[0][k], dict) and "answer" in records[0][k]]
    judged = []
    with out_path.open("w", encoding="utf-8") as out:
        for done, record in enumerate(records, start=1):
            q = by_id[record["id"]]
            verdicts = judge_record(record, q["question"], q["gold_evidence"], pipelines, ask)
            judged.append(verdicts)
            out.write(json.dumps({"id": record["id"], "question": q["question"], **verdicts}, ensure_ascii=False) + "\n")
            out.flush()
            progress(done, len(records), f"{record['id']} " + " ".join(f"{n}:{verdicts[n]['correct']}" for n in pipelines))
    return summarize_judgements(judged, pipelines)
