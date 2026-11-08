"""Tests for the eval scoring and progress functions."""
import json

import pytest

from evals import metrics, run_evals


def test_matching_ignores_case_curly_quotes_and_line_breaks():
    chunk = "TARS\nI’m  joking.\nBing."
    assert metrics.contains_evidence(chunk, ["tars i'm joking."])


def test_first_hit_rank_is_one_based_and_none_when_missing():
    chunks = ["nothing here", "the black hole gargantua", "gargantua again"]
    assert metrics.first_hit_rank(chunks, ["black hole gargantua"]) == 2
    assert metrics.first_hit_rank(chunks, ["wormhole"]) is None


def test_any_one_of_several_quotes_is_enough():
    assert metrics.first_hit_rank(["alpha beta"], ["gamma", "beta"]) == 1


def test_hit_at_k_and_mrr():
    scores = metrics.retrieval_scores([1, 2, None, 7])
    assert scores["questions"] == 4
    assert scores["hit_at_k"] == 0.5  # ranks 1 and 2 are inside the top 5; 7 and None are not
    assert scores["mrr"] == pytest.approx((1 + 1 / 2 + 0 + 1 / 7) / 4)
    assert metrics.retrieval_scores([])["hit_at_k"] is None


@pytest.mark.parametrize("answer,expected", [
    ("The excerpts do not provide information about that.", True),
    ("I don't know.", True),
    ("TARS's honesty setting is ninety percent.", False),
])
def test_refusal_detection(answer, expected):
    assert metrics.is_refusal(answer) is expected


def test_percentile_uses_nearest_rank():
    assert metrics.percentile([10, 20, 30, 40, 50], 50) == 30
    assert metrics.percentile([10, 20, 30, 40, 50], 95) == 50
    assert metrics.percentile([], 50) is None


def test_progress_line_shows_percent_elapsed_and_eta():
    line = metrics.progress_line(25, 100, 600, "q025")
    assert "[ 25/100]" in line and "25%" in line
    assert "elapsed 10:00" in line
    assert "ETA 30:00" in line  # 24 s per question x 75 left = 30 min
    assert line.endswith("q025")


def test_duration_switches_to_hours():
    assert metrics.format_duration(75) == "01:15"
    assert metrics.format_duration(3725) == "1:02:05"


def test_progress_line_before_anything_finished_does_not_divide_by_zero():
    assert "ETA 00:00" in metrics.progress_line(0, 10, 0)


def test_unlocatable_labels_flags_quotes_cut_by_a_chunk_boundary():
    questions = [
        {"id": "a", "gold_route": "script", "gold_evidence": ["ninety percent"]},
        {"id": "b", "gold_route": "script", "gold_evidence": ["split across two"]},
    ]
    texts = {"script": ["TARS: Ninety percent.", "...split ac", "ross two chunks"]}
    assert run_evals.unlocatable_labels(questions, "v1", texts) == {"b"}


def test_summarize_retrieval_groups_by_route_and_type():
    records = [
        {"id": "1", "type": "exact_name", "route": "script", "v1": {"rank": None}, "v2": {"rank": 1}},
        {"id": "2", "type": "concept", "route": "textbook", "v1": {"rank": 2}, "v2": {"rank": 3}},
    ]
    summary = run_evals.summarize_retrieval(records)
    assert summary["overall"]["v1"]["hit_at_k"] == 0.5 and summary["overall"]["v2"]["hit_at_k"] == 1.0
    assert summary["route=script"]["v2"]["questions"] == 1
    assert "type=concept" in summary


def test_summarize_full_counts_refusals_routing_and_latency():
    def rec(i, answerable, gold, route, refused, seconds):
        side = {"answer": "", "seconds": seconds, "refused": refused, "citations": []}
        return {"id": i, "type": "x", "gold_route": gold, "answerable": answerable, "route": route, "route_seconds": 2.0, "v1": side, "v2": side}

    records = [
        rec("1", True, "script", "script", False, 10),
        rec("2", True, "script", "textbook", True, 20),
        rec("3", False, None, "script", True, 30),
        {"id": "4", "type": "x", "gold_route": "script", "answerable": True, "error": "ConnectionError: down"},
    ]
    s = run_evals.summarize_full(records)
    assert s["errors"] == 1
    assert s["routing_accuracy"] == 0.5  # 2 routed questions with a gold route, 1 right
    assert s["v1"]["correct_refusal_rate"] == 1.0
    assert s["v1"]["false_refusal_rate"] == 0.5
    assert s["v1"]["answer_seconds_p50"] == 20


def test_every_labelled_question_has_a_route_and_is_answerable():
    for q in run_evals.load_dataset():
        if q.get("gold_evidence"):
            assert q["answerable"] and q["gold_route"] in ("script", "textbook")
    ids = [q["id"] for q in run_evals.load_dataset()]
    assert len(ids) == len(set(ids))
