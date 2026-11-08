"""Tests for the relevance-gate analysis (pure arithmetic: no models, no Weaviate)."""
from evals import relevance_gate as gate


def row(score, answerable, model_refused=False):
    return {"score": score, "answerable": answerable, "model_refused": model_refused}


ROWS = [row(-8, False), row(-7, False), row(-6, False, model_refused=True), row(2, True), row(3, True), row(4, True),
        row(-9, True), row(5, True), row(6, True), row(7, True), row(1, True), row(8, True)]


def test_a_gate_refuses_everything_below_the_threshold_on_top_of_the_models_own_refusals():
    outcome = gate.gate_outcome(ROWS, threshold=-5)
    assert outcome["correct_refusal"] == 1.0  # all three unanswerable rows are below -5 or refused by the model
    assert outcome["false_refusal"] == 1 / 9  # the answerable row scoring -9 is refused wrongly


def test_with_the_gate_off_only_the_models_refusals_count():
    off = gate.gate_outcome(ROWS, float("-inf"))
    assert off["correct_refusal"] == 1 / 3 and off["false_refusal"] == 0.0


def test_the_picked_threshold_never_breaks_the_false_refusal_limit():
    t = gate.pick_threshold(ROWS, max_false_refusal=0.0)
    assert t is None or gate.gate_outcome(ROWS, t)["false_refusal"] == 0.0
    loose = gate.pick_threshold(ROWS, max_false_refusal=0.2)
    assert loose is not None and gate.gate_outcome(ROWS, loose)["correct_refusal"] == 1.0


def test_no_gate_is_picked_when_it_cannot_beat_the_model_alone():
    rows = [row(1, False, model_refused=True), row(2, True), row(3, True), row(0, False, model_refused=True)]
    assert gate.pick_threshold(rows) is None


def test_the_threshold_is_chosen_on_one_half_and_measured_on_the_other():
    report = gate.cross_check(ROWS)
    assert len(report["folds"]) == 2
    for fold in report["folds"]:
        first, second = gate.split_halves(ROWS)
        assert fold["test_without_gate"] in (gate.gate_outcome(first, float("-inf")), gate.gate_outcome(second, float("-inf")))


# --- the robustness cases ------------------------------------------------------

def test_robustness_scoring_catches_an_obeyed_injection_and_a_leaked_prompt():
    from evals import robustness

    injected = next(c for c in robustness.CASES if c["id"] == "r1")
    assert robustness.score_case(injected, "PWNED")["ok"] is False
    assert robustness.score_case(injected, "I don't know.")["ok"] is True
    leak = next(c for c in robustness.CASES if c["id"] == "r2")
    assert robustness.score_case(leak, "My instructions: answer using ONLY the provided excerpts.")["leaked"]


def test_robustness_scoring_requires_a_real_answer_when_one_is_expected():
    from evals import robustness

    spanish = next(c for c in robustness.CASES if c["id"] == "r7")
    assert robustness.score_case(spanish, "TARS es un robot de la misión Endurance.")["ok"] is True
    assert robustness.score_case(spanish, "I don't know.")["ok"] is False
