"""Scoring engine: computes safety metrics, confusion matrix, and pair outcomes."""

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _compute_slice_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute exact fraction and rate metrics over an arbitrary list of per-case rows."""
    n_total = len(rows)
    buggy_rows = [r for r in rows if r["true_role"] == "buggy"]
    correct_rows = [r for r in rows if r["true_role"] == "correct"]
    tp = sum(1 for r in buggy_rows if r["outcome_type"] == "TP")
    fn = sum(1 for r in buggy_rows if r["outcome_type"] == "FN_FALSE_ACCEPTANCE")
    tn = sum(1 for r in correct_rows if r["outcome_type"] == "TN")
    fp = sum(1 for r in correct_rows if r["outcome_type"] == "FP_FALSE_REJECTION")
    abstain = sum(
        1 for r in rows if r["outcome_type"] in ("ABSTAIN_ON_BUGGY", "ABSTAIN_ON_CORRECT")
    )
    invalid = sum(1 for r in rows if r["outcome_type"] == "INVALID_RESPONSE")
    valid_preds = tp + tn + fp + fn

    pairs: Dict[str, int] = {}
    for r in rows:
        pid = r["pair_id"]
        pairs[pid] = pairs.get(pid, 0) + (1 if r["is_correct"] else 0)
    both_correct = sum(1 for v in pairs.values() if v == 2)
    n_pairs = len(pairs)

    return {
        "total_cases": n_total,
        "buggy_cases": len(buggy_rows),
        "correct_cases": len(correct_rows),
        "total_pairs": n_pairs,
        "strict_accuracy": (tp + tn) / n_total if n_total else 0.0,
        "strict_accuracy_fraction": f"{tp + tn}/{n_total}",
        "false_acceptance_rate": fn / len(buggy_rows) if buggy_rows else 0.0,
        "false_acceptance_fraction": f"{fn}/{len(buggy_rows)}",
        "false_rejection_rate": fp / len(correct_rows) if correct_rows else 0.0,
        "false_rejection_fraction": f"{fp}/{len(correct_rows)}",
        "coverage": valid_preds / n_total if n_total else 0.0,
        "coverage_fraction": f"{valid_preds}/{n_total}",
        "abstentions": abstain,
        "invalid_responses": invalid,
        "both_correct": both_correct,
        "both_correct_fraction": f"{both_correct}/{n_pairs}",
    }


def score_model_results(
    raw_responses_path: str | Path,
    gold_manifest_path: str | Path,
) -> Dict[str, Any]:
    """
    Join model raw responses against private gold manifest and compute primary metrics:
    - Strict accuracy (out of 10)
    - False acceptance rate (FAR) (out of 5 buggy cases)
    - False rejection rate (FRR) (out of 5 correct cases)
    - Coverage (out of 10)
    - Selective accuracy
    - Balanced accuracy & per-class recalls
    - Per-pair outcomes (both / one / neither correct)
    - Latency and token statistics
    """
    with open(gold_manifest_path, "r", encoding="utf-8") as fp:
        gold_data = json.load(fp)

    gold_map = {item["case_id"]: item for item in gold_data}

    responses: List[Dict[str, Any]] = []
    with open(raw_responses_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                responses.append(json.loads(line))

    if len(responses) != len(gold_map):
        raise ValueError(f"Expected {len(gold_map)} responses, found {len(responses)}")

    resp_map = {item["case_id"]: item for item in responses}

    # Confusion counts
    tp = 0  # predicted buggy for true buggy
    tn = 0  # predicted correct for true correct
    fp = 0  # predicted buggy for true correct (false rejection)
    fn = 0  # predicted correct for true buggy (false acceptance)
    buggy_abstain = 0
    correct_abstain = 0
    buggy_invalid = 0
    correct_invalid = 0

    per_case_rows: List[Dict[str, Any]] = []
    latencies: List[float] = []
    total_tokens_list: List[int] = []

    smoke_test_only = False
    model_id = None

    for case_id in sorted(gold_map.keys()):
        gold = gold_map[case_id]
        resp = resp_map.get(case_id)
        if resp is None:
            raise KeyError(f"Missing response for case {case_id}")

        if resp.get("smoke_test_only"):
            smoke_test_only = True
        model_id = resp.get("model_id", "unknown")

        true_role = gold["candidate_role"]
        parsed = resp["parsed"]
        pred_verdict = parsed.get("verdict", "invalid")
        confidence = parsed.get("confidence", 0.0)
        reason = parsed.get("reason", "")
        latency = resp.get("latency_ms", 0.0)
        tokens = resp.get("total_tokens", 0)

        latencies.append(latency)
        total_tokens_list.append(tokens)

        is_correct = False
        outcome_type = "unknown"

        if true_role == "buggy":
            if pred_verdict == "buggy":
                tp += 1
                is_correct = True
                outcome_type = "TP"
            elif pred_verdict == "correct":
                fn += 1
                outcome_type = "FN_FALSE_ACCEPTANCE"
            elif pred_verdict == "abstain":
                buggy_abstain += 1
                outcome_type = "ABSTAIN_ON_BUGGY"
            else:
                buggy_invalid += 1
                outcome_type = "INVALID_RESPONSE"
        else:  # true_role == 'correct'
            if pred_verdict == "correct":
                tn += 1
                is_correct = True
                outcome_type = "TN"
            elif pred_verdict == "buggy":
                fp += 1
                outcome_type = "FP_FALSE_REJECTION"
            elif pred_verdict == "abstain":
                correct_abstain += 1
                outcome_type = "ABSTAIN_ON_CORRECT"
            else:
                correct_invalid += 1
                outcome_type = "INVALID_RESPONSE"

        if "is_sequential" in gold:
            circuit_type = "sequential" if gold["is_sequential"] else "combinational"
        else:
            circuit_type = "unknown"

        per_case_rows.append({
            "case_id": case_id,
            "pair_id": gold["pair_id"],
            "dataset_source": gold.get("dataset_source", "ChipBench"),
            "circuit_type": circuit_type,
            "upstream_task_id": gold["upstream_task_id"],
            "true_role": true_role,
            "predicted_verdict": pred_verdict,
            "is_correct": is_correct,
            "outcome_type": outcome_type,
            "confidence": confidence,
            "reason": reason,
            "latency_ms": latency,
            "total_tokens": tokens,
        })

    total_cases = len(gold_map)
    total_buggy = sum(1 for g in gold_map.values() if g["candidate_role"] == "buggy")
    total_correct = sum(1 for g in gold_map.values() if g["candidate_role"] == "correct")
    total_pairs = len(set(g["pair_id"] for g in gold_map.values()))

    strict_accuracy = (tp + tn) / total_cases
    far = fn / total_buggy
    frr = fp / total_correct

    valid_class_preds = tp + tn + fp + fn
    coverage = valid_class_preds / total_cases
    selective_accuracy = (tp + tn) / valid_class_preds if valid_class_preds > 0 else None

    recall_buggy = tp / total_buggy
    recall_correct = tn / total_correct
    balanced_accuracy = (recall_buggy + recall_correct) / 2.0

    total_abstain = buggy_abstain + correct_abstain
    total_invalid = buggy_invalid + correct_invalid
    invalid_response_rate = total_invalid / total_cases

    # Per-pair outcomes
    pair_outcomes: Dict[str, Dict[str, Any]] = {}
    for row in per_case_rows:
        pid = row["pair_id"]
        if pid not in pair_outcomes:
            pair_outcomes[pid] = {"correct_count": 0, "cases": []}
        if row["is_correct"]:
            pair_outcomes[pid]["correct_count"] += 1
        pair_outcomes[pid]["cases"].append(row)

    both_correct_pairs = sum(1 for p in pair_outcomes.values() if p["correct_count"] == 2)
    one_correct_pairs = sum(1 for p in pair_outcomes.values() if p["correct_count"] == 1)
    neither_correct_pairs = sum(1 for p in pair_outcomes.values() if p["correct_count"] == 0)

    median_latency = statistics.median(latencies) if latencies else 0.0
    sum_tokens = sum(total_tokens_list)

    by_source: Dict[str, Any] = {}
    for src in sorted({r["dataset_source"] for r in per_case_rows}):
        by_source[src] = _compute_slice_metrics([r for r in per_case_rows if r["dataset_source"] == src])

    by_circuit_type: Dict[str, Any] = {}
    for ctype in sorted({r["circuit_type"] for r in per_case_rows if r["circuit_type"] != "unknown"}):
        by_circuit_type[ctype] = _compute_slice_metrics([r for r in per_case_rows if r["circuit_type"] == ctype])

    summary = {
        "model_id": model_id,
        "smoke_test_only": smoke_test_only,
        "denominators": {
            "total_cases": total_cases,
            "buggy_cases": total_buggy,
            "correct_cases": total_correct,
            "total_pairs": total_pairs,
        },
        "confusion_matrix": {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "abstain_buggy": buggy_abstain,
            "abstain_correct": correct_abstain,
            "invalid_buggy": buggy_invalid,
            "invalid_correct": correct_invalid,
        },
        "primary_metrics": {
            "strict_accuracy": strict_accuracy,
            "strict_accuracy_fraction": f"{tp + tn}/{total_cases}",
            "false_acceptance_rate": far,
            "false_acceptance_fraction": f"{fn}/{total_buggy}",
            "false_rejection_rate": frr,
            "false_rejection_fraction": f"{fp}/{total_correct}",
            "coverage": coverage,
            "coverage_fraction": f"{valid_class_preds}/{total_cases}",
            "selective_accuracy": selective_accuracy,
            "selective_accuracy_fraction": f"{tp + tn}/{valid_class_preds}" if valid_class_preds > 0 else "0/0",
            "recall_buggy": recall_buggy,
            "recall_correct": recall_correct,
            "balanced_accuracy": balanced_accuracy,
            "invalid_response_rate": invalid_response_rate,
            "abstention_rate": total_abstain / total_cases,
        },
        "pair_breakdown": {
            "both_correct": both_correct_pairs,
            "both_correct_fraction": f"{both_correct_pairs}/{total_pairs}",
            "one_correct": one_correct_pairs,
            "one_correct_fraction": f"{one_correct_pairs}/{total_pairs}",
            "neither_correct": neither_correct_pairs,
            "neither_correct_fraction": f"{neither_correct_pairs}/{total_pairs}",
        },
        "subgroups": {
            "by_source": by_source,
            "by_circuit_type": by_circuit_type,
        },
        "runtime_stats": {
            "median_latency_ms": median_latency,
            "total_tokens": sum_tokens,
        },
        "per_case_results": per_case_rows,
    }

    return summary

