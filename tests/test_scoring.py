"""Unit tests for scoring logic using synthetic fixtures only."""

import json
import tempfile
from pathlib import Path

import pytest

from rtlreviewbench.score import score_model_results


@pytest.fixture
def synthetic_gold_manifest():
    """Create a temporary synthetic gold manifest with 5 pairs / 10 cases."""
    # 5 buggy, 5 correct
    gold = [
        {"case_id": "C01", "pair_id": "P01", "upstream_task_id": "Task1", "candidate_role": "buggy"},
        {"case_id": "C02", "pair_id": "P01", "upstream_task_id": "Task1", "candidate_role": "correct"},
        {"case_id": "C03", "pair_id": "P02", "upstream_task_id": "Task2", "candidate_role": "correct"},
        {"case_id": "C04", "pair_id": "P02", "upstream_task_id": "Task2", "candidate_role": "buggy"},
        {"case_id": "C05", "pair_id": "P03", "upstream_task_id": "Task3", "candidate_role": "buggy"},
        {"case_id": "C06", "pair_id": "P03", "upstream_task_id": "Task3", "candidate_role": "correct"},
        {"case_id": "C07", "pair_id": "P04", "upstream_task_id": "Task4", "candidate_role": "correct"},
        {"case_id": "C08", "pair_id": "P04", "upstream_task_id": "Task4", "candidate_role": "buggy"},
        {"case_id": "C09", "pair_id": "P05", "upstream_task_id": "Task5", "candidate_role": "buggy"},
        {"case_id": "C10", "pair_id": "P05", "upstream_task_id": "Task5", "candidate_role": "correct"},
    ]
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as fp:
        json.dump(gold, fp)
        gold_path = fp.name
    return Path(gold_path)


def test_perfect_model_scoring(synthetic_gold_manifest):
    """A perfect model predicting exactly matching gold labels."""
    responses = [
        {"case_id": "C01", "parsed": {"verdict": "buggy", "confidence": 0.95, "reason": "Bug found"}},
        {"case_id": "C02", "parsed": {"verdict": "correct", "confidence": 0.95, "reason": "Valid"}},
        {"case_id": "C03", "parsed": {"verdict": "correct", "confidence": 0.95, "reason": "Valid"}},
        {"case_id": "C04", "parsed": {"verdict": "buggy", "confidence": 0.95, "reason": "Bug found"}},
        {"case_id": "C05", "parsed": {"verdict": "buggy", "confidence": 0.95, "reason": "Bug found"}},
        {"case_id": "C06", "parsed": {"verdict": "correct", "confidence": 0.95, "reason": "Valid"}},
        {"case_id": "C07", "parsed": {"verdict": "correct", "confidence": 0.95, "reason": "Valid"}},
        {"case_id": "C08", "parsed": {"verdict": "buggy", "confidence": 0.95, "reason": "Bug found"}},
        {"case_id": "C09", "parsed": {"verdict": "buggy", "confidence": 0.95, "reason": "Bug found"}},
        {"case_id": "C10", "parsed": {"verdict": "correct", "confidence": 0.95, "reason": "Valid"}},
    ]
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl", encoding="utf-8") as fp:
        for r in responses:
            fp.write(json.dumps(r) + "\n")
        resp_path = fp.name

    summary = score_model_results(resp_path, synthetic_gold_manifest)
    pm = summary["primary_metrics"]
    assert pm["strict_accuracy"] == 1.0
    assert pm["false_acceptance_rate"] == 0.0
    assert pm["false_rejection_rate"] == 0.0
    assert pm["coverage"] == 1.0
    assert pm["balanced_accuracy"] == 1.0
    assert summary["pair_breakdown"]["both_correct"] == 5


def test_all_abstain_model_scoring(synthetic_gold_manifest):
    """An all-abstain model (e.g. mock harness)."""
    responses = [
        {"case_id": f"C{i:02d}", "parsed": {"verdict": "abstain", "confidence": 0.5, "reason": "Unsure"}}
        for i in range(1, 11)
    ]
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl", encoding="utf-8") as fp:
        for r in responses:
            fp.write(json.dumps(r) + "\n")
        resp_path = fp.name

    summary = score_model_results(resp_path, synthetic_gold_manifest)
    pm = summary["primary_metrics"]
    assert pm["strict_accuracy"] == 0.0
    assert pm["false_acceptance_rate"] == 0.0  # abstain is not false acceptance
    assert pm["false_rejection_rate"] == 0.0  # abstain is not false rejection
    assert pm["coverage"] == 0.0
    assert pm["selective_accuracy"] is None
    assert pm["abstention_rate"] == 1.0
    assert summary["pair_breakdown"]["neither_correct"] == 5


def test_all_accepting_model_scoring(synthetic_gold_manifest):
    """A naive model that always predicts 'correct'."""
    responses = [
        {"case_id": f"C{i:02d}", "parsed": {"verdict": "correct", "confidence": 1.0, "reason": "Looks good"}}
        for i in range(1, 11)
    ]
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl", encoding="utf-8") as fp:
        for r in responses:
            fp.write(json.dumps(r) + "\n")
        resp_path = fp.name

    summary = score_model_results(resp_path, synthetic_gold_manifest)
    pm = summary["primary_metrics"]
    # 5 correct cases correctly predicted, 5 buggy cases falsely accepted
    assert pm["strict_accuracy"] == 0.5
    assert pm["false_acceptance_rate"] == 1.0  # all 5 buggy accepted! (5/5)
    assert pm["false_rejection_rate"] == 0.0  # 0 correct rejected
    assert pm["coverage"] == 1.0
    assert pm["recall_buggy"] == 0.0
    assert pm["recall_correct"] == 1.0
    assert pm["balanced_accuracy"] == 0.5
    # In each pair, exactly 1 case was correct and 1 buggy, so each pair gets 1/2 correct
    assert summary["pair_breakdown"]["both_correct"] == 0
    assert summary["pair_breakdown"]["one_correct"] == 5


def test_subgroup_scoring_by_source_and_circuit_type():
    """Verify multi-source and sequential/combinational subgroup metrics computation."""
    gold = [
        {"case_id": "C01", "pair_id": "P01", "dataset_source": "ChipBench", "is_sequential": True, "upstream_task_id": "T1", "candidate_role": "buggy"},
        {"case_id": "C02", "pair_id": "P01", "dataset_source": "ChipBench", "is_sequential": True, "upstream_task_id": "T1", "candidate_role": "correct"},
        {"case_id": "C03", "pair_id": "P02", "dataset_source": "RTLRepairBench", "is_sequential": False, "upstream_task_id": "T2", "candidate_role": "buggy"},
        {"case_id": "C04", "pair_id": "P02", "dataset_source": "RTLRepairBench", "is_sequential": False, "upstream_task_id": "T2", "candidate_role": "correct"},
    ]
    responses = [
        {"case_id": "C01", "parsed": {"verdict": "buggy", "confidence": 0.9, "reason": "ok"}},
        {"case_id": "C02", "parsed": {"verdict": "correct", "confidence": 0.9, "reason": "ok"}},
        {"case_id": "C03", "parsed": {"verdict": "correct", "confidence": 0.8, "reason": "missed"}},  # FN
        {"case_id": "C04", "parsed": {"verdict": "buggy", "confidence": 0.8, "reason": "fp"}},      # FP
    ]
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as fg:
        json.dump(gold, fg)
        gpath = fg.name
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl", encoding="utf-8") as fr:
        for r in responses:
            fr.write(json.dumps(r) + "\n")
        rpath = fr.name

    summary = score_model_results(rpath, gpath)
    by_src = summary["subgroups"]["by_source"]
    assert by_src["ChipBench"]["strict_accuracy_fraction"] == "2/2"
    assert by_src["ChipBench"]["both_correct_fraction"] == "1/1"
    assert by_src["RTLRepairBench"]["strict_accuracy_fraction"] == "0/2"
    assert by_src["RTLRepairBench"]["false_acceptance_fraction"] == "1/1"
    assert by_src["RTLRepairBench"]["false_rejection_fraction"] == "1/1"

    by_ct = summary["subgroups"]["by_circuit_type"]
    assert by_ct["sequential"]["strict_accuracy_fraction"] == "2/2"
    assert by_ct["combinational"]["strict_accuracy_fraction"] == "0/2"

