"""Tests for the answer judge. A fake judge replaces the model, so nothing here needs Ollama."""
import json

from evals import judge
from evals.judge import Verdict


def fake_ask(verdicts):
    """Return a judge that answers from a list, remembering what it was asked."""
    asked = []

    def ask(system_prompt, message):
        asked.append((system_prompt, message))
        return Verdict(reason="because", verdict=verdicts[len(asked) - 1])

    ask.asked = asked
    return ask


def side(answer="TARS is 90 percent honest.", refused=False, contexts=("excerpt",)):
    return {"answer": answer, "refused": refused, "contexts": list(contexts)}


def test_a_refusal_is_incorrect_without_asking_the_judge():
    ask = fake_ask([])
    result = judge.judge_record({"a": side("I don't know.", refused=True)}, "q", ["evidence"], ["a"], ask)
    assert result["a"]["correct"] == "no" and result["a"]["faithful"] is None
    assert ask.asked == []


def test_correctness_sees_the_question_the_evidence_and_the_answer():
    ask = fake_ask(["yes", "yes"])
    result = judge.judge_record({"a": side()}, "Who is TARS?", ["a massive, sarcastic robot"], ["a"], ask)
    correctness_prompt, message = ask.asked[0]
    assert correctness_prompt == judge.CORRECTNESS_PROMPT
    assert "Who is TARS?" in message and "a massive, sarcastic robot" in message and "90 percent" in message
    assert result["a"] == {"correct": "yes", "correct_reason": "because", "faithful": "yes", "faithful_reason": "because"}


def test_faithfulness_sees_the_excerpts_and_the_answer_but_not_the_gold_evidence():
    ask = fake_ask(["yes", "partly"])
    judge.judge_record({"a": side(contexts=["first excerpt", "second excerpt"])}, "q", ["SECRET EVIDENCE"], ["a"], ask)
    prompt, message = ask.asked[1]
    assert prompt == judge.FAITHFULNESS_PROMPT
    assert "[1] first excerpt" in message and "[2] second excerpt" in message
    assert "SECRET EVIDENCE" not in message


def test_no_contexts_means_faithfulness_is_not_judged():
    ask = fake_ask(["yes"])
    result = judge.judge_record({"a": side(contexts=[])}, "q", ["e"], ["a"], ask)
    assert result["a"]["faithful"] is None and len(ask.asked) == 1


def test_correctness_only_mode_skips_the_slow_faithfulness_check():
    ask = fake_ask(["yes"])
    result = judge.judge_record({"a": side()}, "q", ["e"], ["a"], ask, faithfulness=False)
    assert result["a"]["correct"] == "yes" and result["a"]["faithful"] is None
    assert len(ask.asked) == 1


def test_summary_counts_each_verdict():
    judged = [
        {"a": {"correct": "yes", "faithful": "yes"}},
        {"a": {"correct": "partly", "faithful": "no"}},
        {"a": {"correct": "no", "faithful": None}},
        {"a": {"correct": "yes", "faithful": "partly"}},
    ]
    s = judge.summarize_judgements(judged, ["a"])["a"]
    assert s["correct"] == 0.5 and s["partly_correct"] == 0.25 and s["incorrect_or_refused"] == 0.25
    assert s["faithful"] == 1 / 3 and s["faithfulness_judged"] == 3


def test_run_judge_only_judges_labelled_questions_and_writes_a_line_each(tmp_path):
    results = tmp_path / "full.jsonl"
    rows = [
        {"id": "q1", "base": side(), "final": side()},
        {"id": "q2", "base": side(), "final": side()},  # no gold evidence: skipped
        {"id": "q3", "error": "ConnectionError: down"},
    ]
    results.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    dataset = [{"id": "q1", "question": "Q1?", "gold_evidence": ["e"]}, {"id": "q2", "question": "Q2?"}, {"id": "q3", "question": "Q3?", "gold_evidence": ["e"]}]
    out = tmp_path / "judged.jsonl"
    seen = []
    summary = judge.run_judge(results, dataset, out, lambda *a: seen.append(a), ask=lambda p, m: Verdict(reason="r", verdict="yes"))
    assert summary["questions"] == 1 and summary["base"]["correct"] == 1.0
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1
    assert seen[0][:2] == (1, 1)
