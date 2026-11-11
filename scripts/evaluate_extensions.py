"""Comprehensive scientific evaluation and statistical testing for all RTLReviewBench suites."""

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rtlreviewbench.prompt import extract_json_key_order, parse_model_response
from rtlreviewbench.score import score_model_results
from scripts.run_extensions import parse_pairwise_response

# Six Core pairs audited in reports/v2_oracle_audit.md as simulation/oracle artifacts
# (3 time-0/initial artifacts: P020/C039, P047/C094, P084/C167;
#  3 inertial #delay timing mismatches: P018/C035, P022/C044, P028/C055).
ORACLE_ARTIFACT_PAIRS = frozenset({"P018", "P020", "P022", "P028", "P047", "P084"})

MODEL_FAMILIES: Dict[str, str] = {
    "claude-haiku-5.5": "Anthropic",
    "claude-haiku-5.5-thinking": "Anthropic",
    "claude-4.6-sonnet": "Anthropic",
    "gemini-3.1-pro": "Google",
    "gemini-3.8-flash": "Google",
    "gpt-5.4": "OpenAI",
    "gpt-4.1": "OpenAI",
    "gpt-4.1-mini": "OpenAI",
    "o4-mini": "OpenAI",
    "gpt-oss-120b": "OpenWeight",
    "gemma-4-31b-it": "OpenWeight",
    "llama-3.3-70b-instruct": "OpenWeight",
}


def holm_adjust(pvals: Dict[str, float]) -> Dict[str, float]:
    """Step-down Holm-Bonferroni family-wise error rate adjustment."""
    if not pvals:
        return {}
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: Dict[str, float] = {}
    running_max = 0.0
    for rank, (key, p) in enumerate(items):
        raw_adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, raw_adj)
        adjusted[key] = running_max
    return adjusted


def _binom_pmf(i: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 1.0 if i == 0 else 0.0
    if p >= 1.0:
        return 1.0 if i == n else 0.0
    log_c = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
    return math.exp(log_c + i * math.log(p) + (n - i) * math.log(1.0 - p))


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Dict[str, Any]:
    """Exact two-sided (1 - alpha) Clopper-Pearson binomial confidence interval."""
    if n <= 0:
        return {"k": 0, "n": 0, "rate": 0.0, "low": 0.0, "high": 0.0}
    if k < 0 or k > n:
        raise ValueError(f"Invalid binomial count k={k} for n={n}")

    half_alpha = alpha / 2.0

    if k == 0:
        low = 0.0
    else:
        lo, hi = 0.0, k / n
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            upper_tail = sum(_binom_pmf(i, n, mid) for i in range(k, n + 1))
            if upper_tail < half_alpha:
                lo = mid
            else:
                hi = mid
        low = 0.5 * (lo + hi)

    if k == n:
        high = 1.0
    else:
        lo, hi = k / n, 1.0
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            lower_tail = sum(_binom_pmf(i, n, mid) for i in range(0, k + 1))
            if lower_tail < half_alpha:
                hi = mid
            else:
                lo = mid
        high = 0.5 * (lo + hi)

    return {
        "k": k,
        "n": n,
        "rate": k / n,
        "low": round(low, 6),
        "high": round(high, 6),
    }


def exact_mcnemar(a_ok: List[bool], b_ok: List[bool]) -> Dict[str, Any]:
    """Two-sided exact McNemar test on paired boolean correctness arrays."""
    assert len(a_ok) == len(b_ok)
    n01 = sum(1 for a, b in zip(a_ok, b_ok) if (not a) and b)
    n10 = sum(1 for a, b in zip(a_ok, b_ok) if a and (not b))
    n = n01 + n10
    if n == 0:
        return {"n01_b_better": 0, "n10_a_better": 0, "p_value": 1.0}
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    p = min(1.0, 2.0 * tail)
    return {"n01_b_better": n01, "n10_a_better": n10, "p_value": p}


def reparse_jsonl_if_needed(path: Path, is_pairwise: bool = False) -> List[Dict[str, Any]]:
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    changed = False
    for r in recs:
        raw = r.get("raw_response")
        if raw is not None and str(raw).strip():
            np = parse_pairwise_response(raw) if is_pairwise else parse_model_response(raw)
            if np != r.get("parsed"):
                r["parsed"] = np
                changed = True
    if changed:
        path.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    return recs


def summarize_single_track(raw_dir: Path, gold_path: Path, expected_n: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    gold_data = json.loads(gold_path.read_text(encoding="utf-8"))
    gold_map = {g["case_id"]: g for g in gold_data}

    for sub in sorted(raw_dir.iterdir()):
        if not sub.is_dir() or sub.name == "mock" or sub.name == "verification":
            continue
        rf = sub / "raw_responses.jsonl"
        if not rf.exists():
            continue
        recs = reparse_jsonl_if_needed(rf, is_pairwise=False)
        if len(recs) != expected_n:
            out[sub.name] = {"status": "incomplete", "count": len(recs), "expected": expected_n}
            continue

        s = score_model_results(rf, gold_path)
        strict_json_cnt = sum(1 for r in recs if r["parsed"].get("strict_json", True))
        r_tokens = [r.get("reasoning_tokens", 0) or 0 for r in recs]
        c_tokens = [r.get("completion_tokens", 0) or 0 for r in recs]
        rec_map = {r["case_id"]: r for r in recs}

        # Calibration & Token divergence
        briers = []
        confs_all, confs_ok, confs_err = [], [], []
        ctok_ok, ctok_err, rtok_ok, rtok_err = [], [], [], []
        for row in s["per_case_results"]:
            cid = row["case_id"]
            v = row["predicted_verdict"]
            conf = float(row.get("confidence", 0.5) or 0.5)
            confs_all.append(conf)
            ctok = rec_map[cid].get("completion_tokens", 0) or 0
            rtok = rec_map[cid].get("reasoning_tokens", 0) or 0
            if row["is_correct"]:
                confs_ok.append(conf)
                ctok_ok.append(ctok)
                rtok_ok.append(rtok)
            else:
                confs_err.append(conf)
                ctok_err.append(ctok)
                rtok_err.append(rtok)
            y = 1.0 if row["true_role"] == "buggy" else 0.0
            p_bug = conf if v == "buggy" else ((1.0 - conf) if v == "correct" else 0.5)
            briers.append((p_bug - y) ** 2)

        bins = [
            ("<0.90", lambda c: c < 0.90),
            ("0.90-0.94", lambda c: 0.90 <= c < 0.95),
            ("0.95-0.99", lambda c: 0.95 <= c < 1.0),
            ("1.00", lambda c: c == 1.0),
        ]
        calib_tiers = {}
        for bname, bpred in bins:
            sub_b = [r for r in s["per_case_results"] if bpred(float(r.get("confidence", 0.0) or 0.0))]
            calib_tiers[bname] = {
                "n": len(sub_b),
                "acc": sum(1 for r in sub_b if r["is_correct"]) / len(sub_b) if sub_b else None,
            }

        # Matched-design / operator slices if syn_sem85 or realbug_base/tuned
        by_category: Dict[str, Any] = {}
        cat_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in s["per_case_results"]:
            cat = gold_map[row["case_id"]].get("category", "unknown")
            cat_groups[cat].append(row)
        for cat, grows in sorted(cat_groups.items()):
            tot = len(grows)
            ok = sum(1 for x in grows if x["is_correct"])
            buggy_g = [x for x in grows if x["true_role"] == "buggy"]
            corr_g = [x for x in grows if x["true_role"] == "correct"]
            fn_g = sum(1 for x in buggy_g if x["outcome_type"] == "FN_FALSE_ACCEPTANCE")
            fp_g = sum(1 for x in corr_g if x["outcome_type"] == "FP_FALSE_REJECTION")
            by_category[cat] = {
                "cases": tot,
                "correct_count": ok,
                "buggy_cases": len(buggy_g),
                "correct_cases": len(corr_g),
                "fn": fn_g,
                "fp": fp_g,
                "strict_acc": ok / tot if tot else 0.0,
                "far": fn_g / len(buggy_g) if buggy_g else 0.0,
                "frr": fp_g / len(corr_g) if corr_g else 0.0,
                "acc_ci95": clopper_pearson_ci(ok, tot),
                "far_ci95": clopper_pearson_ci(fn_g, len(buggy_g)),
                "frr_ci95": clopper_pearson_ci(fp_g, len(corr_g)),
            }

        cm = s["confusion_matrix"]
        den = s["denominators"]
        subgroups_ci: Dict[str, Any] = {}
        for sg_group, sg_dict in s["subgroups"].items():
            subgroups_ci[sg_group] = {}
            for sg_name, sg_vals in sg_dict.items():
                k_acc, n_acc = map(int, sg_vals["strict_accuracy_fraction"].split("/"))
                k_far, n_far = map(int, sg_vals["false_acceptance_fraction"].split("/"))
                k_frr, n_frr = map(int, sg_vals["false_rejection_fraction"].split("/"))
                subgroups_ci[sg_group][sg_name] = {
                    "acc_ci95": clopper_pearson_ci(k_acc, n_acc),
                    "far_ci95": clopper_pearson_ci(k_far, n_far),
                    "frr_ci95": clopper_pearson_ci(k_frr, n_frr),
                }

        out[sub.name] = {
            "status": "complete",
            "primary_metrics": s["primary_metrics"],
            "ci95": {
                "strict_acc": clopper_pearson_ci(cm["TP"] + cm["TN"], den["total_cases"]),
                "far": clopper_pearson_ci(cm["FN"], den["buggy_cases"]),
                "frr": clopper_pearson_ci(cm["FP"], den["correct_cases"]),
                "pair_both": clopper_pearson_ci(s["pair_breakdown"]["both_correct"], den["total_pairs"]),
            },
            "confusion_matrix": cm,
            "pair_breakdown": s["pair_breakdown"],
            "subgroups": s["subgroups"],
            "subgroups_ci": subgroups_ci,
            "by_category": by_category,
            "runtime_stats": s["runtime_stats"],
            "strict_json_rate": strict_json_cnt / expected_n,
            "sum_reasoning_tokens": sum(r_tokens),
            "median_reasoning_tokens": statistics.median(r_tokens) if r_tokens else 0,
            "sum_completion_tokens": sum(c_tokens),
            "brier_score": sum(briers) / len(briers) if briers else None,
            "calibration": {
                "mean_conf": sum(confs_all) / len(confs_all) if confs_all else 0.0,
                "mean_conf_ok": sum(confs_ok) / len(confs_ok) if confs_ok else None,
                "mean_conf_err": sum(confs_err) / len(confs_err) if confs_err else None,
                "calib_gap_pp": ((sum(confs_all) / len(confs_all)) - s["primary_metrics"]["strict_accuracy"]) * 100.0 if confs_all else 0.0,
                "tiers": calib_tiers,
            },
            "token_dynamics": {
                "comp_ok_median": statistics.median(ctok_ok) if ctok_ok else None,
                "comp_err_median": statistics.median(ctok_err) if ctok_err else None,
                "reas_ok_median": statistics.median(rtok_ok) if rtok_ok else None,
                "reas_err_median": statistics.median(rtok_err) if rtok_err else None,
            },
            "per_case_results": s["per_case_results"],
        }
    return out


def summarize_pairwise(raw_dir: Path, gold_path: Path, core_scores: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not raw_dir.exists():
        return out
    gold_rows = json.loads(gold_path.read_text(encoding="utf-8"))
    gold_map = {g["query_id"]: g for g in gold_rows}

    for sub in sorted(raw_dir.iterdir()):
        if not sub.is_dir():
            continue
        rf = sub / "raw_responses.jsonl"
        if not rf.exists():
            continue
        recs = reparse_jsonl_if_needed(rf, is_pairwise=True)
        if len(recs) != len(gold_map):
            out[sub.name] = {"status": "incomplete", "count": len(recs), "expected": len(gold_map)}
            continue

        by_dir = {"FWD": [], "REV": []}
        by_pair: Dict[str, Dict[str, Tuple[str, str, float]]] = defaultdict(dict)
        pred_a = pred_b = tie_cnt = neither_cnt = inv_cnt = strict_json_cnt = 0

        for r in recs:
            qid = r["query_id"]
            g = gold_map[qid]
            ch = r["parsed"].get("choice", "invalid")
            conf = float(r["parsed"].get("confidence", 0.0) or 0.0)
            if r["parsed"].get("strict_json", True):
                strict_json_cnt += 1
            if ch == "A":
                pred_a += 1
            elif ch == "B":
                pred_b += 1
            elif ch == "tie":
                tie_cnt += 1
            elif ch == "neither":
                neither_cnt += 1
            else:
                inv_cnt += 1
            is_ok = ch == g["gold_choice"]
            by_dir[g["direction"]].append(is_ok)
            by_pair[g["pair_id"]][g["direction"]] = (ch, g["gold_choice"], conf)

        both_right = one_right = neither_right = order_flip = stuck_a = stuck_b = 0
        swap_ok_vec: List[bool] = []
        pair_ids_sorted = sorted(by_pair.keys())
        for pid in pair_ids_sorted:
            f_ch, f_gold, _ = by_pair[pid]["FWD"]
            r_ch, r_gold, _ = by_pair[pid]["REV"]
            f_ok = f_ch == f_gold
            r_ok = r_ch == r_gold
            swap_ok_vec.append(f_ok and r_ok)
            if f_ok and r_ok:
                both_right += 1
            elif f_ok or r_ok:
                one_right += 1
                if f_ch in ("A", "B") and r_ch in ("A", "B") and f_ch == r_ch:
                    order_flip += 1
                    if f_ch == "A":
                        stuck_a += 1
                    else:
                        stuck_b += 1
            else:
                neither_right += 1

        n_q = len(recs)
        n_p = len(pair_ids_sorted)
        fwd_acc = sum(by_dir["FWD"]) / len(by_dir["FWD"])
        rev_acc = sum(by_dir["REV"]) / len(by_dir["REV"])
        pooled_acc = (sum(by_dir["FWD"]) + sum(by_dir["REV"])) / n_q
        swap_acc = both_right / n_p

        iso_compare = None
        if sub.name in core_scores and core_scores[sub.name].get("status") == "complete":
            iso_pairs: Dict[str, List[bool]] = defaultdict(list)
            for row in core_scores[sub.name]["per_case_results"]:
                iso_pairs[row["pair_id"]].append(row["is_correct"])
            iso_ok_vec = [all(iso_pairs[pid]) for pid in pair_ids_sorted]
            iso_joint = sum(iso_ok_vec) / n_p
            mc = exact_mcnemar(iso_ok_vec, swap_ok_vec)
            iso_compare = {
                "isolated_pair_joint_rate": iso_joint,
                "contrastive_swap_pair_rate": swap_acc,
                "delta_pp": (swap_acc - iso_joint) * 100.0,
                "mcnemar": mc,
            }

        out[sub.name] = {
            "status": "complete",
            "pooled_query_acc": pooled_acc,
            "fwd_acc": fwd_acc,
            "rev_acc": rev_acc,
            "swap_consistent_pair_acc": swap_acc,
            "both_right": both_right,
            "one_right": one_right,
            "neither_right": neither_right,
            "order_flip_count": order_flip,
            "position_stuck_A": stuck_a,
            "position_stuck_B": stuck_b,
            "pred_A_rate": pred_a / n_q,
            "pred_B_rate": pred_b / n_q,
            "position_bias_A_minus_B": (pred_a - pred_b) / n_q,
            "tie_count": tie_cnt,
            "neither_count": neither_cnt,
            "invalid_count": inv_cnt,
            "strict_json_rate": strict_json_cnt / n_q,
            "vs_isolated_pair": iso_compare,
        }
    return out


def summarize_rationale_audit(audit_dir: Path, core_scores: Dict[str, Any]) -> Dict[str, Any]:
    if not audit_dir.exists():
        return {}
    judges: Dict[str, Dict[str, str]] = {}
    items_meta: Dict[str, Dict[str, Any]] = {}
    for sub in sorted(audit_dir.iterdir()):
        jf = sub / "judgments.jsonl"
        if not jf.exists():
            continue
        judges[sub.name] = {}
        for ln in jf.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            judges[sub.name][r["item_id"]] = r["parsed"]["label"]
            items_meta[r["item_id"]] = {
                "reviewer_model": r["reviewer_model"],
                "case_id": r["case_id"],
                "dataset_source": r["dataset_source"],
            }
    if not judges:
        return {}

    judge_names = sorted(judges.keys())
    consensus: Dict[str, str] = {}
    # Majority vote across all available judges for each item_id
    for iid in items_meta:
        votes = [judges[jn][iid] for jn in judge_names if iid in judges[jn] and judges[jn][iid] in ("grounded", "spurious_claim", "vague")]
        if not votes:
            consensus[iid] = "invalid"
        else:
            # Count frequency; if exact 2-judge tie without tiebreaker, map disagreement
            counts = {lbl: votes.count(lbl) for lbl in set(votes)}
            top_lbl, top_cnt = max(counts.items(), key=lambda kv: kv[1])
            if list(counts.values()).count(top_cnt) > 1:
                # 2-way tie with no tiebreaker: prioritize non-grounded if ambiguous or record 'disputed'
                consensus[iid] = "disputed"
            else:
                consensus[iid] = top_lbl

    # Inter-judge agreement on shared items between the two primary (full-coverage) judges
    agreement_stats = {}
    if len(judge_names) >= 2:
        ranked_judges = sorted(judge_names, key=lambda jn: len(judges[jn]), reverse=True)
        j1, j2 = sorted([ranked_judges[0], ranked_judges[1]])
        shared = [iid for iid in judges[j1] if iid in judges[j2] and judges[j1][iid] != "invalid" and judges[j2][iid] != "invalid"]
        if shared:
            agree = sum(1 for i in shared if judges[j1][i] == judges[j2][i]) / len(shared)
            # 3-class Cohen's kappa
            labels = ["grounded", "spurious_claim", "vague"]
            p_e = sum(
                (sum(1 for i in shared if judges[j1][i] == l) / len(shared))
                * (sum(1 for i in shared if judges[j2][i] == l) / len(shared))
                for l in labels
            )
            kappa = (agree - p_e) / (1.0 - p_e) if p_e < 1.0 else 1.0
            disagreed = sum(1 for i in shared if judges[j1][i] != judges[j2][i])
            agreement_stats = {
                "judges": [j1, j2],
                "n_shared": len(shared),
                "n_disagreed": disagreed,
                "raw_agreement": agree,
                "cohens_kappa": kappa,
            }

    per_model: Dict[str, Any] = {}
    by_rev: Dict[str, List[str]] = defaultdict(list)
    for iid, meta in items_meta.items():
        by_rev[meta["reviewer_model"]].append(iid)

    for rev, iids in sorted(by_rev.items()):
        n_tp = len(iids)
        grounded = sum(1 for i in iids if consensus[i] == "grounded")
        spurious = sum(1 for i in iids if consensus[i] == "spurious_claim")
        vague = sum(1 for i in iids if consensus[i] == "vague")
        disputed = sum(1 for i in iids if consensus[i] == "disputed")
        total_buggy = 120
        if rev in core_scores and core_scores[rev].get("status") == "complete":
            total_buggy = core_scores[rev]["denominators"]["buggy_cases"] if "denominators" in core_scores[rev] else 120
        per_model[rev] = {
            "tp_audited": n_tp,
            "grounded": grounded,
            "spurious_claim": spurious,
            "vague": vague,
            "disputed": disputed,
            "grounded_precision_among_tp": grounded / n_tp if n_tp else 0.0,
            "grounded_precision_ci95": clopper_pearson_ci(grounded, n_tp),
            "spurious_rate_among_tp": (spurious + vague) / n_tp if n_tp else 0.0,
            "nominal_tpr": n_tp / total_buggy,
            "effective_grounded_sensitivity": grounded / total_buggy,
            "eff_sens_ci95": clopper_pearson_ci(grounded, total_buggy),
            "inflation_gap_pp": ((n_tp - grounded) / total_buggy) * 100.0,
        }

    return {"agreement": agreement_stats, "per_model": per_model}


def summarize_ensembles(core_scores: Dict[str, Any], gold_path: Path) -> Dict[str, Any]:
    """Compute canonical Core multi-model Boolean gates and agreement regimes."""
    req = ("claude-haiku-5.5", "gemini-3.1-pro", "gemini-3.8-flash", "gpt-oss-120b", "claude-4.6-sonnet")
    if any(m not in core_scores or core_scores[m].get("status") != "complete" for m in req):
        return {"gates": {}, "_top3_diverse_ok": None}

    gold_rows = json.loads(gold_path.read_text(encoding="utf-8"))
    gold_map = {g["case_id"]: g for g in gold_rows}
    case_ids = sorted(gold_map.keys())
    pair_ids = sorted({g["pair_id"] for g in gold_rows})

    pred_by_m: Dict[str, Dict[str, str]] = {}
    for m in req:
        pred_by_m[m] = {r["case_id"]: r["predicted_verdict"] for r in core_scores[m]["per_case_results"]}

    def _eval_rule(name: str, rule_fn) -> Dict[str, Any]:
        tp = tn = fp = fn = abstain = 0
        ok_vec: List[bool] = []
        pair_ok_cnt: Dict[str, int] = defaultdict(int)
        for cid in case_ids:
            g = gold_map[cid]
            role = g["candidate_role"]
            v = rule_fn(cid)
            is_ok = (v == role)
            if v is None:
                abstain += 1
                is_ok = False
            elif role == "buggy":
                if v == "buggy":
                    tp += 1
                else:
                    fn += 1
            else:
                if v == "correct":
                    tn += 1
                else:
                    fp += 1
            ok_vec.append(is_ok)
            if is_ok:
                pair_ok_cnt[g["pair_id"]] += 1

        n_tot = len(case_ids)
        n_cov = tp + tn + fp + fn
        n_bug = tp + fn
        n_cor = tn + fp
        both = sum(1 for p in pair_ids if pair_ok_cnt[p] == 2)
        return {
            "name": name,
            "covered": n_cov,
            "total": n_tot,
            "coverage": n_cov / n_tot if n_tot else 0.0,
            "deferred": abstain,
            "strict_acc": (tp + tn) / n_cov if n_cov else 0.0,
            "strict_acc_ci95": clopper_pearson_ci(tp + tn, n_cov),
            "far": fn / n_bug if n_bug else 0.0,
            "far_ci95": clopper_pearson_ci(fn, n_bug),
            "frr": fp / n_cor if n_cor else 0.0,
            "frr_ci95": clopper_pearson_ci(fp, n_cor),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "pair_both": both,
            "pair_both_rate": both / len(pair_ids) if pair_ids and abstain == 0 else None,
            "_ok_vec": ok_vec,
        }

    top3_div = _eval_rule(
        "Top-3 Diverse Maj (H, P31, Oss)",
        lambda c: "buggy"
        if sum(1 for m in ("claude-haiku-5.5", "gemini-3.1-pro", "gpt-oss-120b") if pred_by_m[m][c] == "buggy") >= 2
        else "correct",
    )
    top3_com = _eval_rule(
        "Top-3 Comm Maj (H, P31, Son)",
        lambda c: "buggy"
        if sum(1 for m in ("claude-haiku-5.5", "gemini-3.1-pro", "claude-4.6-sonnet") if pred_by_m[m][c] == "buggy") >= 2
        else "correct",
    )
    and_hp31 = _eval_rule(
        "AND Gate (H & P31)",
        lambda c: "buggy" if (pred_by_m["claude-haiku-5.5"][c] == "buggy" and pred_by_m["gemini-3.1-pro"][c] == "buggy") else "correct",
    )
    or_hfl = _eval_rule(
        "OR Gate (H | Fl)",
        lambda c: "buggy" if (pred_by_m["claude-haiku-5.5"][c] == "buggy" or pred_by_m["gemini-3.8-flash"][c] == "buggy") else "correct",
    )
    cons_hp31 = _eval_rule(
        "Consensus (H == P31)",
        lambda c: pred_by_m["claude-haiku-5.5"][c] if pred_by_m["claude-haiku-5.5"][c] == pred_by_m["gemini-3.1-pro"][c] else None,
    )
    cons_hfl = _eval_rule(
        "Consensus (H == Fl)",
        lambda c: pred_by_m["claude-haiku-5.5"][c] if pred_by_m["claude-haiku-5.5"][c] == pred_by_m["gemini-3.8-flash"][c] else None,
    )

    top3_div_ok = top3_div["_ok_vec"]
    gates = {}
    for k_name, obj in [
        ("top3_diverse_maj", top3_div),
        ("top3_comm_maj", top3_com),
        ("and_gate_h_p31", and_hp31),
        ("or_gate_h_fl", or_hfl),
        ("consensus_h_p31", cons_hp31),
        ("consensus_h_fl", cons_hfl),
    ]:
        gates[k_name] = {k: v for k, v in obj.items() if k != "_ok_vec"}

    return {"gates": gates, "_top3_diverse_ok": top3_div_ok}


def _majority_verdict(verdicts: List[str]) -> str:
    """Resolve k-sample verdict via strict majority/plurality over buggy vs. correct."""
    n_bug = sum(1 for v in verdicts if v == "buggy")
    n_cor = sum(1 for v in verdicts if v == "correct")
    if n_bug > n_cor and n_bug >= (len(verdicts) / 2.0):
        return "buggy"
    if n_cor > n_bug and n_cor >= (len(verdicts) / 2.0):
        return "correct"
    if n_bug > n_cor:
        return "buggy"
    if n_cor > n_bug:
        return "correct"
    if verdicts and verdicts[0] in ("buggy", "correct"):
        return verdicts[0]
    return "abstain"


def summarize_self_consistency(
    raw_dir: Path,
    gold_path: Path,
    core_scores: Dict[str, Any],
    rf_scores: Dict[str, Any],
    top3_diverse_ok: Optional[List[bool]] = None,
    num_samples: int = 5,
) -> Dict[str, Any]:
    """Evaluate k-sample same-model stochastic self-consistency (temperature > 0)."""
    out: Dict[str, Any] = {}
    if not raw_dir.exists():
        return out
    gold_rows = json.loads(gold_path.read_text(encoding="utf-8"))
    gold_map = {g["case_id"]: g for g in gold_rows}
    case_ids = sorted(gold_map.keys())
    pair_ids = sorted({g["pair_id"] for g in gold_rows})
    expected_n = len(case_ids) * num_samples

    for sub in sorted(raw_dir.iterdir()):
        if not sub.is_dir():
            continue
        rf = sub / "raw_responses.jsonl"
        if not rf.exists():
            continue
        recs = reparse_jsonl_if_needed(rf, is_pairwise=False)
        if len(recs) != expected_n:
            out[sub.name] = {"status": "incomplete", "count": len(recs), "expected": expected_n}
            continue

        by_case: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        by_sample: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        temperatures = set()
        for r in recs:
            cid = r["case_id"]
            s_idx = int(r.get("sample_idx", 0))
            by_case[cid].append(r)
            by_sample[s_idx][cid] = r
            if "temperature" in r:
                temperatures.add(float(r["temperature"]))

        # Per-sample single-draw metrics
        sample_accs: List[float] = []
        sample_fars: List[float] = []
        sample_frrs: List[float] = []
        for s_idx in range(num_samples):
            s_rows = by_sample.get(s_idx, {})
            if len(s_rows) != len(case_ids):
                continue
            tp_s = tn_s = fp_s = fn_s = 0
            for cid in case_ids:
                role = gold_map[cid]["candidate_role"]
                v = s_rows[cid]["parsed"].get("verdict", "invalid")
                if role == "buggy":
                    if v == "buggy":
                        tp_s += 1
                    elif v == "correct":
                        fn_s += 1
                else:
                    if v == "correct":
                        tn_s += 1
                    elif v == "buggy":
                        fp_s += 1
            sample_accs.append((tp_s + tn_s) / len(case_ids))
            sample_fars.append(fn_s / 120.0)
            sample_frrs.append(fp_s / 120.0)

        tp = tn = fp = fn = inv = abst = 0
        maj_ok_vec: List[bool] = []
        maj_bug_ok: List[bool] = []
        maj_cor_ok: List[bool] = []
        pair_ok: Dict[str, int] = defaultdict(int)
        unanimous_correct = 0
        unanimous_any = 0
        split_4_1 = 0
        split_3_2 = 0
        other_splits = 0
        vote_briers: List[float] = []
        strict_json_cnt = 0
        all_comp_tokens: List[int] = []

        for cid in case_ids:
            draws = sorted(by_case[cid], key=lambda x: int(x.get("sample_idx", 0)))
            verdicts = [d["parsed"].get("verdict", "invalid") for d in draws]
            strict_json_cnt += sum(1 for d in draws if d["parsed"].get("strict_json", True))
            all_comp_tokens.extend(int(d.get("completion_tokens", 0) or 0) for d in draws)

            maj_v = _majority_verdict(verdicts)
            role = gold_map[cid]["candidate_role"]
            is_ok = False
            if role == "buggy":
                if maj_v == "buggy":
                    tp += 1
                    is_ok = True
                elif maj_v == "correct":
                    fn += 1
                elif maj_v == "abstain":
                    abst += 1
                else:
                    inv += 1
                maj_bug_ok.append(is_ok)
            else:
                if maj_v == "correct":
                    tn += 1
                    is_ok = True
                elif maj_v == "buggy":
                    fp += 1
                elif maj_v == "abstain":
                    abst += 1
                else:
                    inv += 1
                maj_cor_ok.append(is_ok)

            maj_ok_vec.append(is_ok)
            if is_ok:
                pair_ok[gold_map[cid]["pair_id"]] += 1

            n_role = sum(1 for v in verdicts if v == role)
            if n_role == num_samples:
                unanimous_correct += 1
            vc = Counter(verdicts)
            top_v_cnt = vc.most_common(1)[0][1]
            if top_v_cnt == num_samples:
                unanimous_any += 1
            elif top_v_cnt == num_samples - 1 and num_samples == 5:
                split_4_1 += 1
            elif top_v_cnt == 3 and num_samples == 5:
                split_3_2 += 1
            else:
                other_splits += 1

            p_bug_vote = sum(1 for v in verdicts if v == "buggy") / len(verdicts)
            y = 1.0 if role == "buggy" else 0.0
            vote_briers.append((p_bug_vote - y) ** 2)

        n_tot = len(case_ids)
        n_bug = len(maj_bug_ok)
        n_cor = len(maj_cor_ok)
        both_p = sum(1 for pid in pair_ids if pair_ok[pid] == 2)

        vs_t0 = None
        if sub.name in core_scores and core_scores[sub.name].get("status") == "complete":
            t0_rows = core_scores[sub.name]["per_case_results"]
            t0_ok = [r["is_correct"] for r in t0_rows]
            t0_bug = [r["is_correct"] for r in t0_rows if r["true_role"] == "buggy"]
            t0_cor = [r["is_correct"] for r in t0_rows if r["true_role"] == "correct"]
            pm_t0 = core_scores[sub.name]["primary_metrics"]
            vs_t0 = {
                "t0_acc": pm_t0["strict_accuracy"],
                "maj5_acc": (tp + tn) / n_tot,
                "delta_acc_pp": ((tp + tn) / n_tot - pm_t0["strict_accuracy"]) * 100.0,
                "delta_far_pp": (fn / n_bug - pm_t0["false_acceptance_rate"]) * 100.0,
                "delta_frr_pp": (fp / n_cor - pm_t0["false_rejection_rate"]) * 100.0,
                "flipped_vs_t0_cases": sum(1 for a, b in zip(t0_ok, maj_ok_vec) if a != b),
                "mcnemar_overall": exact_mcnemar(t0_ok, maj_ok_vec),
                "mcnemar_far": exact_mcnemar(t0_bug, maj_bug_ok),
                "mcnemar_frr": exact_mcnemar(t0_cor, maj_cor_ok),
            }

        vs_rf = None
        if sub.name in rf_scores and rf_scores[sub.name].get("status") == "complete":
            rf_rows = rf_scores[sub.name]["per_case_results"]
            rf_ok = [r["is_correct"] for r in rf_rows]
            pm_rf = rf_scores[sub.name]["primary_metrics"]
            vs_rf = {
                "rf_acc": pm_rf["strict_accuracy"],
                "delta_maj5_minus_rf_pp": ((tp + tn) / n_tot - pm_rf["strict_accuracy"]) * 100.0,
                "mcnemar_overall": exact_mcnemar(rf_ok, maj_ok_vec),
            }

        vs_top3 = None
        if top3_diverse_ok is not None and len(top3_diverse_ok) == n_tot:
            vs_top3 = {
                "top3_acc": sum(top3_diverse_ok) / n_tot,
                "delta_maj5_minus_top3_pp": ((tp + tn - sum(top3_diverse_ok)) / n_tot) * 100.0,
                "mcnemar_overall": exact_mcnemar(top3_diverse_ok, maj_ok_vec),
            }

        out[sub.name] = {
            "status": "complete",
            "temperature": sorted(temperatures)[0] if temperatures else 0.7,
            "num_samples": num_samples,
            "total_queries": len(recs),
            "mean_sample_acc": sum(sample_accs) / len(sample_accs) if sample_accs else None,
            "std_sample_acc_pp": (statistics.stdev(sample_accs) * 100.0) if len(sample_accs) > 1 else 0.0,
            "sample_accs": sample_accs,
            "mean_sample_far": sum(sample_fars) / len(sample_fars) if sample_fars else None,
            "mean_sample_frr": sum(sample_frrs) / len(sample_frrs) if sample_frrs else None,
            "majority5": {
                "strict_acc": (tp + tn) / n_tot,
                "strict_acc_fraction": f"{tp + tn}/{n_tot}",
                "far": fn / n_bug,
                "far_fraction": f"{fn}/{n_bug}",
                "frr": fp / n_cor,
                "frr_fraction": f"{fp}/{n_cor}",
                "pair_both": both_p,
                "pair_both_rate": both_p / len(pair_ids),
                "ci95": {
                    "strict_acc": clopper_pearson_ci(tp + tn, n_tot),
                    "far": clopper_pearson_ci(fn, n_bug),
                    "frr": clopper_pearson_ci(fp, n_cor),
                    "pair_both": clopper_pearson_ci(both_p, len(pair_ids)),
                },
            },
            "vote_dynamics": {
                "unanimous_correct": unanimous_correct,
                "unanimous_any": unanimous_any,
                "split_4_1": split_4_1,
                "split_3_2": split_3_2,
                "other_splits": other_splits,
                "any_split_rate": (n_tot - unanimous_any) / n_tot,
                "vote_freq_brier": sum(vote_briers) / len(vote_briers) if vote_briers else None,
            },
            "strict_json_rate": strict_json_cnt / len(recs),
            "total_tokens": sum(int(r.get("total_tokens", 0) or 0) for r in recs),
            "median_completion_tokens": statistics.median(all_comp_tokens) if all_comp_tokens else 0,
            "vs_greedy_t0": vs_t0,
            "vs_reason_first": vs_rf,
            "vs_top3_diverse_maj": vs_top3,
        }
    return out


def _compute_clean114_single_track(
    full_scores: Dict[str, Any],
    raw_dir: Path,
    gold_path: Path,
    exclude_pairs: frozenset[str] = ORACLE_ARTIFACT_PAIRS,
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Rescore a single-track Core suite on Clean-114 (excluding audited oracle-artifact pairs)."""
    gold_data = json.loads(gold_path.read_text(encoding="utf-8"))
    gold_map = {g["case_id"]: g for g in gold_data if g["pair_id"] not in exclude_pairs}
    clean_cids = sorted(gold_map.keys())
    clean_set = set(clean_cids)
    clean_pids = sorted({gold_map[c]["pair_id"] for c in clean_cids})
    n_tot = len(clean_cids)
    n_pairs = len(clean_pids)
    out: Dict[str, Any] = {}

    for m, d in full_scores.items():
        if d.get("status") != "complete":
            continue
        rf = raw_dir / m / "raw_responses.jsonl"
        recs = [json.loads(l) for l in rf.read_text(encoding="utf-8").splitlines() if l.strip()]
        rec_map = {r["case_id"]: r for r in recs if r["case_id"] in clean_set}

        tp = tn = fp = fn = ab = inv = 0
        strict_json_cnt = 0
        briers: List[float] = []
        confs_all, confs_ok, confs_err = [], [], []
        ctok_ok, ctok_err, rtok_ok, rtok_err = [], [], [], []
        r_tokens, c_tokens, latencies = [], [], []
        pair_ok: Dict[str, int] = defaultdict(int)
        by_src: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        by_ctype: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        per_case_clean: List[Dict[str, Any]] = []

        for row in d["per_case_results"]:
            cid = row["case_id"]
            if cid not in clean_set:
                continue
            per_case_clean.append(row)
            g = gold_map[cid]
            rec = rec_map[cid]
            if rec["parsed"].get("strict_json", True):
                strict_json_cnt += 1
            ctok = rec.get("completion_tokens", 0) or 0
            rtok = rec.get("reasoning_tokens", 0) or 0
            lat = rec.get("latency_ms", 0.0) or 0.0
            c_tokens.append(ctok)
            r_tokens.append(rtok)
            latencies.append(lat)

            role = row["true_role"]
            v = row["predicted_verdict"]
            ok = row["is_correct"]
            if ok:
                pair_ok[g["pair_id"]] += 1
                confs_ok.append(float(row.get("confidence", 0.5) or 0.5))
                ctok_ok.append(ctok)
                rtok_ok.append(rtok)
            else:
                confs_err.append(float(row.get("confidence", 0.5) or 0.5))
                ctok_err.append(ctok)
                rtok_err.append(rtok)

            if role == "buggy":
                if v == "buggy":
                    tp += 1
                elif v == "correct":
                    fn += 1
                elif v == "abstain":
                    ab += 1
                else:
                    inv += 1
            else:
                if v == "correct":
                    tn += 1
                elif v == "buggy":
                    fp += 1
                elif v == "abstain":
                    ab += 1
                else:
                    inv += 1

            conf = float(row.get("confidence", 0.5) or 0.5)
            confs_all.append(conf)
            y = 1.0 if role == "buggy" else 0.0
            p_bug = conf if v == "buggy" else ((1.0 - conf) if v == "correct" else 0.5)
            briers.append((p_bug - y) ** 2)

            by_src[g["dataset_source"]].append(row)
            by_ctype["sequential" if g["is_sequential"] else "combinational"].append(row)
            by_cat[g.get("category", "unknown")].append(row)

        def _slice_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
            tot = len(rows)
            okc = sum(1 for x in rows if x["is_correct"])
            bugg = [x for x in rows if x["true_role"] == "buggy"]
            corrg = [x for x in rows if x["true_role"] == "correct"]
            fng = sum(1 for x in bugg if x["outcome_type"] == "FN_FALSE_ACCEPTANCE")
            fpg = sum(1 for x in corrg if x["outcome_type"] == "FP_FALSE_REJECTION")
            return {
                "cases": tot,
                "correct_count": okc,
                "buggy_cases": len(bugg),
                "correct_cases": len(corrg),
                "fn": fng,
                "fp": fpg,
                "strict_acc": okc / tot if tot else 0.0,
                "far": fng / len(bugg) if bugg else 0.0,
                "frr": fpg / len(corrg) if corrg else 0.0,
                "acc_ci95": clopper_pearson_ci(okc, tot),
                "far_ci95": clopper_pearson_ci(fng, len(bugg)),
                "frr_ci95": clopper_pearson_ci(fpg, len(corrg)),
            }

        both = sum(1 for p in clean_pids if pair_ok[p] == 2)
        one = sum(1 for p in clean_pids if pair_ok[p] == 1)
        zero = sum(1 for p in clean_pids if pair_ok[p] == 0)
        covered = tp + tn + fp + fn

        out[m] = {
            "status": "complete",
            "primary_metrics": {
                "strict_accuracy": (tp + tn) / n_tot,
                "selective_accuracy": (tp + tn) / covered if covered else 0.0,
                "false_acceptance_rate": fn / n_pairs,
                "false_rejection_rate": fp / n_pairs,
                "abstention_rate": ab / n_tot,
                "invalid_rate": inv / n_tot,
                "coverage": covered / n_tot,
            },
            "ci95": {
                "strict_acc": clopper_pearson_ci(tp + tn, n_tot),
                "far": clopper_pearson_ci(fn, n_pairs),
                "frr": clopper_pearson_ci(fp, n_pairs),
                "pair_both": clopper_pearson_ci(both, n_pairs),
            },
            "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "ABSTAIN": ab, "INVALID": inv},
            "pair_breakdown": {"both_correct": both, "one_correct": one, "neither_correct": zero, "both_correct_rate": both / n_pairs},
            "subgroups": {
                "by_source": {k: _slice_stats(v) for k, v in sorted(by_src.items())},
                "by_circuit_type": {k: _slice_stats(v) for k, v in sorted(by_ctype.items())},
            },
            "by_category": {k: _slice_stats(v) for k, v in sorted(by_cat.items())},
            "strict_json_rate": strict_json_cnt / n_tot,
            "median_latency_ms": statistics.median(latencies) if latencies else 0.0,
            "sum_reasoning_tokens": sum(r_tokens),
            "sum_completion_tokens": sum(c_tokens),
            "brier_score": sum(briers) / len(briers) if briers else None,
            "calibration": {
                "mean_conf": sum(confs_all) / len(confs_all) if confs_all else 0.0,
                "calib_gap_pp": ((sum(confs_all) / len(confs_all)) - ((tp + tn) / n_tot)) * 100.0 if confs_all else 0.0,
            },
            "token_dynamics": {
                "comp_ok_median": statistics.median(ctok_ok) if ctok_ok else None,
                "comp_err_median": statistics.median(ctok_err) if ctok_err else None,
                "reas_ok_median": statistics.median(rtok_ok) if rtok_ok else None,
                "reas_err_median": statistics.median(rtok_err) if rtok_err else None,
            },
            "per_case_results": per_case_clean,
        }
    return out, clean_cids, clean_pids


def _compute_clean114_ensembles_and_cv(
    core_clean: Dict[str, Any],
    gold_path: Path,
    exclude_pairs: frozenset[str] = ORACLE_ARTIFACT_PAIRS,
) -> Dict[str, Any]:
    """Compute Clean-114 multi-model gates, 5-fold pair-level CV, and out-of-source checks."""
    gold_rows = [g for g in json.loads(gold_path.read_text(encoding="utf-8")) if g["pair_id"] not in exclude_pairs]
    gold_map = {g["case_id"]: g for g in gold_rows}
    case_ids = sorted(gold_map.keys())
    pair_ids = sorted({g["pair_id"] for g in gold_rows})

    req = ("claude-haiku-5.5", "gemini-3.1-pro", "gemini-3.8-flash", "gpt-oss-120b", "claude-4.6-sonnet")
    if any(m not in core_clean for m in req):
        return {"gates": {}, "_top3_diverse_ok": None}

    pred_by_m: Dict[str, Dict[str, str]] = {
        m: {r["case_id"]: r["predicted_verdict"] for r in d["per_case_results"]}
        for m, d in core_clean.items()
    }

    def _eval_rule(name: str, rule_fn, subset_cids: Optional[List[str]] = None) -> Dict[str, Any]:
        cids = subset_cids if subset_cids is not None else case_ids
        pids = sorted({gold_map[c]["pair_id"] for c in cids})
        tp = tn = fp = fn = abstain = 0
        ok_vec: List[bool] = []
        pair_ok_cnt: Dict[str, int] = defaultdict(int)
        for cid in cids:
            role = gold_map[cid]["candidate_role"]
            v = rule_fn(cid)
            is_ok = (v == role)
            if v is None:
                abstain += 1
                is_ok = False
            elif role == "buggy":
                if v == "buggy":
                    tp += 1
                else:
                    fn += 1
            else:
                if v == "correct":
                    tn += 1
                else:
                    fp += 1
            ok_vec.append(is_ok)
            if is_ok:
                pair_ok_cnt[gold_map[cid]["pair_id"]] += 1

        n_tot = len(cids)
        n_cov = tp + tn + fp + fn
        n_bug = tp + fn
        n_cor = tn + fp
        both = sum(1 for p in pids if pair_ok_cnt[p] == 2)
        return {
            "name": name,
            "covered": n_cov,
            "total": n_tot,
            "coverage": n_cov / n_tot if n_tot else 0.0,
            "deferred": abstain,
            "strict_acc": (tp + tn) / n_cov if n_cov else 0.0,
            "strict_acc_ci95": clopper_pearson_ci(tp + tn, n_cov),
            "far": fn / n_bug if n_bug else 0.0,
            "far_ci95": clopper_pearson_ci(fn, n_bug),
            "frr": fp / n_cor if n_cor else 0.0,
            "frr_ci95": clopper_pearson_ci(fp, n_cor),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "pair_both": both,
            "pair_both_rate": both / len(pids) if pids and abstain == 0 else None,
            "_ok_vec": ok_vec,
        }

    top3_div = _eval_rule(
        "Top-3 Diverse Maj (H, P31, Oss)",
        lambda c: "buggy"
        if sum(1 for m in ("claude-haiku-5.5", "gemini-3.1-pro", "gpt-oss-120b") if pred_by_m[m][c] == "buggy") >= 2
        else "correct",
    )
    top3_com = _eval_rule(
        "Top-3 Comm Maj (H, P31, Son)",
        lambda c: "buggy"
        if sum(1 for m in ("claude-haiku-5.5", "gemini-3.1-pro", "claude-4.6-sonnet") if pred_by_m[m][c] == "buggy") >= 2
        else "correct",
    )
    and_hp31 = _eval_rule(
        "AND Gate (H & P31)",
        lambda c: "buggy" if (pred_by_m["claude-haiku-5.5"][c] == "buggy" and pred_by_m["gemini-3.1-pro"][c] == "buggy") else "correct",
    )
    or_hfl = _eval_rule(
        "OR Gate (H | Fl)",
        lambda c: "buggy" if (pred_by_m["claude-haiku-5.5"][c] == "buggy" or pred_by_m["gemini-3.8-flash"][c] == "buggy") else "correct",
    )
    cons_hp31 = _eval_rule(
        "Consensus (H == P31)",
        lambda c: pred_by_m["claude-haiku-5.5"][c] if pred_by_m["claude-haiku-5.5"][c] == pred_by_m["gemini-3.1-pro"][c] else None,
    )
    cons_hfl = _eval_rule(
        "Consensus (H == Fl)",
        lambda c: pred_by_m["claude-haiku-5.5"][c] if pred_by_m["claude-haiku-5.5"][c] == pred_by_m["gemini-3.8-flash"][c] else None,
    )

    # 5-fold pair-level cross-validation selecting top-1 model from each of the top-3 families on training folds
    import itertools
    by_fam: Dict[str, List[str]] = defaultdict(list)
    for m, fam in MODEL_FAMILIES.items():
        if m in pred_by_m:
            by_fam[fam].append(m)

    folds = [pair_ids[i::5] for i in range(5)]
    cv_top_fam_ok = cv_top_fam_fn = cv_top_fam_fp = 0
    cv_all_triples_ok = cv_all_triples_fn = cv_all_triples_fp = 0
    triples_all = []
    for f_comb in itertools.combinations(sorted(by_fam.keys()), 3):
        for m1 in by_fam[f_comb[0]]:
            for m2 in by_fam[f_comb[1]]:
                for m3 in by_fam[f_comb[2]]:
                    triples_all.append((m1, m2, m3))

    for fold_pids in folds:
        test_pset = set(fold_pids)
        tr_cids = [c for c in case_ids if gold_map[c]["pair_id"] not in test_pset]
        te_cids = [c for c in case_ids if gold_map[c]["pair_id"] in test_pset]

        # Strategy A: top-1 single model from each of the top-3 distinct model families on train fold
        fam_best = {
            fam: max(ms, key=lambda m: sum(1 for c in tr_cids if pred_by_m[m][c] == gold_map[c]["candidate_role"]))
            for fam, ms in by_fam.items()
        }
        top3_f = sorted(
            fam_best.values(),
            key=lambda m: -sum(1 for c in tr_cids if pred_by_m[m][c] == gold_map[c]["candidate_role"]),
        )[:3]
        for c in te_cids:
            pred = "buggy" if sum(1 for m in top3_f if pred_by_m[m][c] == "buggy") >= 2 else "correct"
            role = gold_map[c]["candidate_role"]
            if pred == role:
                cv_top_fam_ok += 1
            elif role == "buggy":
                cv_top_fam_fn += 1
            else:
                cv_top_fam_fp += 1

        # Strategy B: unconstrained joint optimization over all cross-family triples on train fold
        best_t = max(
            triples_all,
            key=lambda t: sum(
                1 for c in tr_cids
                if ("buggy" if sum(1 for m in t if pred_by_m[m][c] == "buggy") >= 2 else "correct") == gold_map[c]["candidate_role"]
            ),
        )
        for c in te_cids:
            pred = "buggy" if sum(1 for m in best_t if pred_by_m[m][c] == "buggy") >= 2 else "correct"
            role = gold_map[c]["candidate_role"]
            if pred == role:
                cv_all_triples_ok += 1
            elif role == "buggy":
                cv_all_triples_fn += 1
            else:
                cv_all_triples_fp += 1

    rb_clean_cids = [c for c in case_ids if gold_map[c]["dataset_source"] == "RTLRepairBench"]
    rb_clean_div = _eval_rule(
        "Top-3 Diverse on RB Clean (97p)",
        lambda c: "buggy"
        if sum(1 for m in ("claude-haiku-5.5", "gemini-3.1-pro", "gpt-oss-120b") if pred_by_m[m][c] == "buggy") >= 2
        else "correct",
        subset_cids=rb_clean_cids,
    )

    gates = {
        k: {kk: vv for kk, vv in obj.items() if kk != "_ok_vec"}
        for k, obj in [
            ("top3_diverse_maj", top3_div),
            ("top3_comm_maj", top3_com),
            ("and_gate_h_p31", and_hp31),
            ("or_gate_h_fl", or_hfl),
            ("consensus_h_p31", cons_hp31),
            ("consensus_h_fl", cons_hfl),
        ]
    }
    gates["cv5_top_family_maj"] = {
        "strict_acc": cv_top_fam_ok / len(case_ids),
        "strict_acc_ci95": clopper_pearson_ci(cv_top_fam_ok, len(case_ids)),
        "far": cv_top_fam_fn / len(pair_ids),
        "frr": cv_top_fam_fp / len(pair_ids),
        "correct": cv_top_fam_ok,
        "total": len(case_ids),
    }
    gates["cv5_joint_triple_maj"] = {
        "strict_acc": cv_all_triples_ok / len(case_ids),
        "strict_acc_ci95": clopper_pearson_ci(cv_all_triples_ok, len(case_ids)),
        "far": cv_all_triples_fn / len(pair_ids),
        "frr": cv_all_triples_fp / len(pair_ids),
        "correct": cv_all_triples_ok,
        "total": len(case_ids),
    }
    gates["rb_clean_top3_diverse"] = {kk: vv for kk, vv in rb_clean_div.items() if kk != "_ok_vec"}
    return {"gates": gates, "_top3_diverse_ok": top3_div["_ok_vec"]}


def _compute_clean114_pairwise(
    raw_dir: Path,
    gold_path: Path,
    core_clean: Dict[str, Any],
    exclude_pairs: frozenset[str] = ORACLE_ARTIFACT_PAIRS,
) -> Dict[str, Any]:
    """Summarize bidirectional pairwise results on Clean-114 pairs."""
    out: Dict[str, Any] = {}
    if not raw_dir.exists():
        return out
    gold_rows = [g for g in json.loads(gold_path.read_text(encoding="utf-8")) if g["pair_id"] not in exclude_pairs]
    gold_map = {g["query_id"]: g for g in gold_rows}
    clean_pids = sorted({g["pair_id"] for g in gold_rows})

    for sub in sorted(raw_dir.iterdir()):
        if not sub.is_dir():
            continue
        rf = sub / "raw_responses.jsonl"
        if not rf.exists():
            continue
        recs = [json.loads(l) for l in rf.read_text(encoding="utf-8").splitlines() if l.strip()]
        by_dir = {"FWD": [], "REV": []}
        by_pair: Dict[str, Dict[str, Tuple[str, str]]] = defaultdict(dict)
        pred_a = pred_b = tie_cnt = neither_cnt = inv_cnt = strict_json_cnt = 0

        for r in recs:
            qid = r["query_id"]
            if qid not in gold_map:
                continue
            g = gold_map[qid]
            ch = r["parsed"].get("choice", "invalid")
            if r["parsed"].get("strict_json", True):
                strict_json_cnt += 1
            if ch == "A":
                pred_a += 1
            elif ch == "B":
                pred_b += 1
            elif ch == "tie":
                tie_cnt += 1
            elif ch == "neither":
                neither_cnt += 1
            else:
                inv_cnt += 1
            by_dir[g["direction"]].append(ch == g["gold_choice"])
            by_pair[g["pair_id"]][g["direction"]] = (ch, g["gold_choice"])

        both_right = one_right = neither_right = order_flip = stuck_a = stuck_b = 0
        swap_ok_vec: List[bool] = []
        for pid in clean_pids:
            f_ch, f_gold = by_pair[pid]["FWD"]
            r_ch, r_gold = by_pair[pid]["REV"]
            f_ok = (f_ch == f_gold)
            r_ok = (r_ch == r_gold)
            swap_ok_vec.append(f_ok and r_ok)
            if f_ok and r_ok:
                both_right += 1
            elif f_ok or r_ok:
                one_right += 1
                if f_ch in ("A", "B") and r_ch in ("A", "B") and f_ch == r_ch:
                    order_flip += 1
                    if f_ch == "A":
                        stuck_a += 1
                    else:
                        stuck_b += 1
            else:
                neither_right += 1

        n_p = len(clean_pids)
        n_q = 2 * n_p
        iso_compare = None
        if sub.name in core_clean:
            iso_pairs: Dict[str, List[bool]] = defaultdict(list)
            for row in core_clean[sub.name]["per_case_results"]:
                iso_pairs[row["pair_id"]].append(row["is_correct"])
            iso_ok_vec = [all(iso_pairs[pid]) for pid in clean_pids]
            iso_joint = sum(iso_ok_vec) / n_p
            mc = exact_mcnemar(iso_ok_vec, swap_ok_vec)
            iso_compare = {
                "isolated_pair_joint_rate": iso_joint,
                "contrastive_swap_pair_rate": both_right / n_p,
                "delta_pp": (both_right / n_p - iso_joint) * 100.0,
                "mcnemar": mc,
            }

        out[sub.name] = {
            "status": "complete",
            "pooled_query_acc": (sum(by_dir["FWD"]) + sum(by_dir["REV"])) / n_q,
            "pooled_query_acc_ci95": clopper_pearson_ci(sum(by_dir["FWD"]) + sum(by_dir["REV"]), n_q),
            "fwd_acc": sum(by_dir["FWD"]) / len(by_dir["FWD"]),
            "rev_acc": sum(by_dir["REV"]) / len(by_dir["REV"]),
            "swap_consistent_pair_acc": both_right / n_p,
            "swap_consistent_ci95": clopper_pearson_ci(both_right, n_p),
            "both_right": both_right,
            "one_right": one_right,
            "neither_right": neither_right,
            "order_flip_count": order_flip,
            "position_stuck_A": stuck_a,
            "position_stuck_B": stuck_b,
            "position_bias_A_minus_B": (pred_a - pred_b) / n_q,
            "tie_count": tie_cnt,
            "neither_count": neither_cnt,
            "invalid_count": inv_cnt,
            "strict_json_rate": strict_json_cnt / n_q,
            "vs_isolated_pair": iso_compare,
        }
    return out


def _compute_clean114_rationale_audit(
    audit_dir: Path,
    gold_path: Path,
    exclude_pairs: frozenset[str] = ORACLE_ARTIFACT_PAIRS,
) -> Dict[str, Any]:
    """Summarize blinded 3-judge rationale audit restricted to Clean-114 buggy cases."""
    if not audit_dir.exists():
        return {}
    gold_map = {g["case_id"]: g for g in json.loads(gold_path.read_text(encoding="utf-8"))}
    clean_bug_cids = {c for c, g in gold_map.items() if g["pair_id"] not in exclude_pairs and g["candidate_role"] == "buggy"}
    n_bug_clean = len(clean_bug_cids)

    judges: Dict[str, Dict[str, str]] = {}
    items_meta: Dict[str, Dict[str, Any]] = {}
    for sub in sorted(audit_dir.iterdir()):
        jf = sub / "judgments.jsonl"
        if not jf.exists():
            continue
        judges[sub.name] = {}
        for ln in jf.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            judges[sub.name][r["item_id"]] = r["parsed"]["label"]
            items_meta[r["item_id"]] = {
                "reviewer_model": r["reviewer_model"],
                "case_id": r["case_id"],
                "dataset_source": r["dataset_source"],
            }

    jnames = sorted(judges.keys())
    consensus: Dict[str, str] = {}
    for iid in items_meta:
        votes = [judges[jn][iid] for jn in jnames if iid in judges[jn] and judges[jn][iid] in ("grounded", "spurious_claim", "vague")]
        if not votes:
            consensus[iid] = "invalid"
        else:
            cnt = Counter(votes)
            top_lbl, top_cnt = cnt.most_common(1)[0]
            consensus[iid] = "disputed" if list(cnt.values()).count(top_cnt) > 1 else top_lbl

    by_rev: Dict[str, List[str]] = defaultdict(list)
    art_items: List[Dict[str, str]] = []
    for iid, mt in items_meta.items():
        if mt["case_id"] in clean_bug_cids:
            by_rev[mt["reviewer_model"]].append(iid)
        else:
            art_items.append({"item_id": iid, "reviewer": mt["reviewer_model"], "case_id": mt["case_id"], "label": consensus[iid]})

    per_model: Dict[str, Any] = {}
    for rev, iids in sorted(by_rev.items()):
        n_tp = len(iids)
        gr = sum(1 for i in iids if consensus[i] == "grounded")
        sp = sum(1 for i in iids if consensus[i] == "spurious_claim")
        vg = sum(1 for i in iids if consensus[i] == "vague")
        dp = sum(1 for i in iids if consensus[i] == "disputed")
        per_model[rev] = {
            "tp_audited": n_tp,
            "grounded": gr,
            "spurious_claim": sp,
            "vague": vg,
            "disputed": dp,
            "grounded_precision_among_tp": gr / n_tp if n_tp else 0.0,
            "grounded_precision_ci95": clopper_pearson_ci(gr, n_tp),
            "nominal_tpr": n_tp / n_bug_clean,
            "effective_grounded_sensitivity": gr / n_bug_clean,
            "eff_sens_ci95": clopper_pearson_ci(gr, n_bug_clean),
            "inflation_gap_pp": ((n_tp - gr) / n_bug_clean) * 100.0,
        }

    # Agreement on Clean-114 subset
    ranked = sorted(jnames, key=lambda jn: len(judges[jn]), reverse=True)
    j1, j2 = sorted([ranked[0], ranked[1]])
    shared_clean = [iid for iid, mt in items_meta.items() if mt["case_id"] in clean_bug_cids and iid in judges[j1] and iid in judges[j2] and judges[j1][iid] != "invalid" and judges[j2][iid] != "invalid"]
    labels = ["grounded", "spurious_claim", "vague"]
    agree_c = sum(1 for i in shared_clean if judges[j1][i] == judges[j2][i]) / len(shared_clean)
    pe_c = sum(
        (sum(1 for i in shared_clean if judges[j1][i] == l) / len(shared_clean))
        * (sum(1 for i in shared_clean if judges[j2][i] == l) / len(shared_clean))
        for l in labels
    )
    kappa_c = (agree_c - pe_c) / (1.0 - pe_c) if pe_c < 1.0 else 1.0

    return {
        "n_buggy_clean": n_bug_clean,
        "total_tp_clean": sum(v["tp_audited"] for v in per_model.values()),
        "agreement": {
            "judges": [j1, j2],
            "n_shared": len(shared_clean),
            "n_disagreed": sum(1 for i in shared_clean if judges[j1][i] != judges[j2][i]),
            "raw_agreement": agree_c,
            "cohens_kappa": kappa_c,
        },
        "per_model": per_model,
        "excluded_artifact_tps": art_items,
    }


def _compute_clean114_self_consistency(
    raw_dir: Path,
    gold_path: Path,
    core_clean: Dict[str, Any],
    rf_clean: Dict[str, Any],
    top3_diverse_ok: Optional[List[bool]] = None,
    num_samples: int = 5,
    exclude_pairs: frozenset[str] = ORACLE_ARTIFACT_PAIRS,
) -> Dict[str, Any]:
    """Summarize SC@5 on Clean-114 (228 cases)."""
    out: Dict[str, Any] = {}
    if not raw_dir.exists():
        return out
    gold_rows = [g for g in json.loads(gold_path.read_text(encoding="utf-8")) if g["pair_id"] not in exclude_pairs]
    gold_map = {g["case_id"]: g for g in gold_rows}
    case_ids = sorted(gold_map.keys())
    clean_set = set(case_ids)
    pair_ids = sorted({g["pair_id"] for g in gold_rows})
    n_tot = len(case_ids)
    n_pairs = len(pair_ids)

    for sub in sorted(raw_dir.iterdir()):
        if not sub.is_dir():
            continue
        rf = sub / "raw_responses.jsonl"
        if not rf.exists():
            continue
        recs = [json.loads(l) for l in rf.read_text(encoding="utf-8").splitlines() if l.strip()]
        by_case: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        by_sample: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        for r in recs:
            cid = r["case_id"]
            if cid not in clean_set:
                continue
            by_case[cid].append(r)
            by_sample[int(r.get("sample_idx", 0))][cid] = r

        sample_accs: List[float] = []
        for s_idx in range(num_samples):
            s_rows = by_sample.get(s_idx, {})
            if len(s_rows) != n_tot:
                continue
            ok_s = sum(1 for cid in case_ids if s_rows[cid]["parsed"].get("verdict") == gold_map[cid]["candidate_role"])
            sample_accs.append(ok_s / n_tot)

        tp = tn = fp = fn = inv = abst = 0
        maj_ok_vec: List[bool] = []
        pair_ok: Dict[str, int] = defaultdict(int)
        unanimous_correct = unanimous_any = split_4_1 = split_3_2 = other_splits = 0
        vote_briers: List[float] = []
        for cid in case_ids:
            draws = sorted(by_case[cid], key=lambda x: int(x.get("sample_idx", 0)))
            verdicts = [d["parsed"].get("verdict", "invalid") for d in draws]
            mv = _majority_verdict(verdicts)
            role = gold_map[cid]["candidate_role"]
            is_ok = (mv == role)
            if role == "buggy":
                if mv == "buggy":
                    tp += 1
                elif mv == "correct":
                    fn += 1
                elif mv == "abstain":
                    abst += 1
                else:
                    inv += 1
            else:
                if mv == "correct":
                    tn += 1
                elif mv == "buggy":
                    fp += 1
                elif mv == "abstain":
                    abst += 1
                else:
                    inv += 1
            maj_ok_vec.append(is_ok)
            if is_ok:
                pair_ok[gold_map[cid]["pair_id"]] += 1
            if sum(1 for v in verdicts if v == role) == num_samples:
                unanimous_correct += 1
            top_c = Counter(verdicts).most_common(1)[0][1]
            if top_c == num_samples:
                unanimous_any += 1
            elif top_c == num_samples - 1:
                split_4_1 += 1
            elif top_c == 3:
                split_3_2 += 1
            else:
                other_splits += 1
            p_bug_vote = sum(1 for v in verdicts if v == "buggy") / len(verdicts)
            vote_briers.append((p_bug_vote - (1.0 if role == "buggy" else 0.0)) ** 2)

        both_p = sum(1 for pid in pair_ids if pair_ok[pid] == 2)
        vs_t0 = vs_rf = vs_top3 = None
        if sub.name in core_clean:
            t0_ok = [r["is_correct"] for r in core_clean[sub.name]["per_case_results"]]
            vs_t0 = {"mcnemar_overall": exact_mcnemar(t0_ok, maj_ok_vec), "delta_acc_pp": ((tp + tn) / n_tot - core_clean[sub.name]["primary_metrics"]["strict_accuracy"]) * 100.0}
        if sub.name in rf_clean:
            rf_ok = [r["is_correct"] for r in rf_clean[sub.name]["per_case_results"]]
            vs_rf = {"mcnemar_overall": exact_mcnemar(rf_ok, maj_ok_vec), "delta_maj5_minus_rf_pp": ((tp + tn) / n_tot - rf_clean[sub.name]["primary_metrics"]["strict_accuracy"]) * 100.0}
        if top3_diverse_ok is not None and len(top3_diverse_ok) == n_tot:
            vs_top3 = {"mcnemar_overall": exact_mcnemar(top3_diverse_ok, maj_ok_vec), "delta_maj5_minus_top3_pp": ((tp + tn - sum(top3_diverse_ok)) / n_tot) * 100.0}

        out[sub.name] = {
            "status": "complete",
            "mean_sample_acc": sum(sample_accs) / len(sample_accs) if sample_accs else None,
            "std_sample_acc_pp": (statistics.stdev(sample_accs) * 100.0) if len(sample_accs) > 1 else 0.0,
            "sample_accs": sample_accs,
            "majority5": {
                "strict_acc": (tp + tn) / n_tot,
                "strict_acc_fraction": f"{tp + tn}/{n_tot}",
                "far": fn / n_pairs,
                "far_fraction": f"{fn}/{n_pairs}",
                "frr": fp / n_pairs,
                "frr_fraction": f"{fp}/{n_pairs}",
                "pair_both": both_p,
                "pair_both_rate": both_p / n_pairs,
                "ci95": {
                    "strict_acc": clopper_pearson_ci(tp + tn, n_tot),
                    "far": clopper_pearson_ci(fn, n_pairs),
                    "frr": clopper_pearson_ci(fp, n_pairs),
                    "pair_both": clopper_pearson_ci(both_p, n_pairs),
                },
            },
            "vote_dynamics": {
                "unanimous_correct": unanimous_correct,
                "unanimous_wrong": unanimous_any - unanimous_correct,
                "unanimous_any": unanimous_any,
                "split_4_1": split_4_1,
                "split_3_2": split_3_2,
                "other_splits": other_splits,
                "any_split": n_tot - unanimous_any,
                "vote_freq_brier": sum(vote_briers) / len(vote_briers) if vote_briers else None,
            },
            "vs_greedy_t0": vs_t0,
            "vs_reason_first": vs_rf,
            "vs_top3_diverse_maj": vs_top3,
        }
    return out


RF_ABLATION_MODELS: Tuple[str, ...] = ("gpt-4.1", "gpt-4.1-mini", "gemma-4-31b-it")
RF_ABLATION_EXPECTED_ORDERS: Dict[str, List[str]] = {
    "vf": ["verdict", "confidence", "reason"],
    "vf_trace": ["verdict", "confidence", "reason"],
    "rf_order_only": ["reason", "verdict", "confidence"],
    "rf": ["reason", "verdict", "confidence"],
}


def _score_rf_ablation_cell(
    raw_file: Path,
    gold_subset: List[Dict[str, Any]],
    expected_order: List[str],
    full_expected_n: int = 240,
) -> Dict[str, Any]:
    """Read-only scorer for one 2x2 cell file on a given gold subset (Clean-114 or Full-120)."""
    if not raw_file.exists():
        return {"status": "missing", "count": 0, "expected": full_expected_n}
    recs = [json.loads(l) for l in raw_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    rec_map = {r["case_id"]: r for r in recs if "case_id" in r}
    if len(rec_map) != full_expected_n:
        return {"status": "incomplete", "count": len(rec_map), "expected": full_expected_n}

    gold_map = {g["case_id"]: g for g in gold_subset}
    cids = sorted(gold_map.keys())
    pids = sorted({gold_map[c]["pair_id"] for c in cids})

    tp = tn = fp = fn = abst = inv = 0
    strict_json_cnt = 0
    key_order_violations = 0
    prefix_order_violations = 0
    briers: List[float] = []
    comp_tokens: List[int] = []
    prompt_sum = comp_sum = reas_sum = total_sum = 0
    pair_ok: Dict[str, int] = defaultdict(int)
    ok_by_cid: Dict[str, bool] = {}

    for cid in cids:
        g = gold_map[cid]
        r = rec_map[cid]
        raw_txt = r.get("raw_response")
        parsed = parse_model_response(raw_txt) if raw_txt is not None else r.get("parsed", {})
        if parsed.get("strict_json", False):
            strict_json_cnt += 1
        emitted = extract_json_key_order(raw_txt or "", required_key="verdict")
        if emitted != expected_order:
            key_order_violations += 1
        if not emitted or emitted[0] != expected_order[0]:
            prefix_order_violations += 1

        ptok = int(r.get("prompt_tokens", 0) or 0)
        ctok = int(r.get("completion_tokens", 0) or 0)
        rtok = int(r.get("reasoning_tokens", 0) or 0)
        ttok = int(r.get("total_tokens", 0) or 0)
        prompt_sum += ptok
        comp_sum += ctok
        reas_sum += rtok
        total_sum += ttok
        comp_tokens.append(ctok)

        v = parsed.get("verdict", "invalid")
        role = g["candidate_role"]
        is_ok = False
        if role == "buggy":
            if v == "buggy":
                tp += 1
                is_ok = True
            elif v == "correct":
                fn += 1
            elif v == "abstain":
                abst += 1
            else:
                inv += 1
        else:
            if v == "correct":
                tn += 1
                is_ok = True
            elif v == "buggy":
                fp += 1
            elif v == "abstain":
                abst += 1
            else:
                inv += 1

        ok_by_cid[cid] = is_ok
        if is_ok:
            pair_ok[g["pair_id"]] += 1

        conf = float(parsed.get("confidence", 0.5) or 0.5)
        y = 1.0 if role == "buggy" else 0.0
        p_bug = conf if v == "buggy" else ((1.0 - conf) if v == "correct" else 0.5)
        briers.append((p_bug - y) ** 2)

    n_tot = len(cids)
    n_pairs = len(pids)
    both = sum(1 for p in pids if pair_ok[p] == 2)
    return {
        "status": "complete",
        "n_cases": n_tot,
        "n_pairs": n_pairs,
        "strict_acc": (tp + tn) / n_tot,
        "acc_ci95": clopper_pearson_ci(tp + tn, n_tot),
        "far": fn / n_pairs,
        "far_ci95": clopper_pearson_ci(fn, n_pairs),
        "frr": fp / n_pairs,
        "frr_ci95": clopper_pearson_ci(fp, n_pairs),
        "pair_both": both,
        "pair_both_rate": both / n_pairs,
        "pair_both_ci95": clopper_pearson_ci(both, n_pairs),
        "brier_score": sum(briers) / len(briers),
        "median_completion_tokens": statistics.median(comp_tokens),
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "ABSTAIN": abst, "INVALID": inv},
        "strict_json_rate": strict_json_cnt / n_tot,
        "key_order_violations": key_order_violations,
        "prefix_order_violations": prefix_order_violations,
        "tokens": {
            "prompt_tokens": prompt_sum,
            "completion_tokens": comp_sum,
            "reasoning_tokens": reas_sum,
            "total_tokens": total_sum,
        },
        "_ok_by_cid": ok_by_cid,
    }


def summarize_rf_ablation(
    core_dir: Path,
    rf_dir: Path,
    ablation_dir: Path,
    gold_path: Path,
    exclude_pairs: frozenset[str] = frozenset(),
    models: Tuple[str, ...] = RF_ABLATION_MODELS,
    full_expected_n: Optional[int] = None,
) -> Dict[str, Any]:
    """Score the 2x2 Reason-First ablation (VF, A:vf_trace, B:rf_order_only, RF) with 12-test Holm adjustment."""
    all_gold = json.loads(gold_path.read_text(encoding="utf-8"))
    exp_full_n = full_expected_n if full_expected_n is not None else len(all_gold)
    gold_subset = [g for g in all_gold if g["pair_id"] not in exclude_pairs]
    cids = sorted(g["case_id"] for g in gold_subset)
    gold_map = {g["case_id"]: g for g in gold_subset}

    per_model: Dict[str, Any] = {}
    pvals_12: Dict[str, float] = {}

    for m in models:
        cell_files = {
            "vf": core_dir / m / "raw_responses.jsonl",
            "vf_trace": ablation_dir / "vf_trace" / m / "raw_responses.jsonl",
            "rf_order_only": ablation_dir / "rf_order_only" / m / "raw_responses.jsonl",
            "rf": rf_dir / m / "raw_responses.jsonl",
        }
        cells = {
            cname: _score_rf_ablation_cell(cpath, gold_subset, RF_ABLATION_EXPECTED_ORDERS[cname], full_expected_n=exp_full_n)
            for cname, cpath in cell_files.items()
        }
        if any(c.get("status") != "complete" for c in cells.values()):
            per_model[m] = {
                "status": "incomplete",
                "cells": {k: {kk: vv for kk, vv in v.items() if kk != "_ok_by_cid"} for k, v in cells.items()},
            }
            continue

        ok_vecs = {
            cname: [cells[cname]["_ok_by_cid"][cid] for cid in cids]
            for cname in ("vf", "vf_trace", "rf_order_only", "rf")
        }
        bug_cids = [c for c in cids if gold_map[c]["candidate_role"] == "buggy"]
        cor_cids = [c for c in cids if gold_map[c]["candidate_role"] == "correct"]
        cmps_spec = [
            ("A_vs_VF", "vf", "vf_trace"),
            ("B_vs_VF", "vf", "rf_order_only"),
            ("RF_vs_A", "vf_trace", "rf"),
            ("RF_vs_B", "rf_order_only", "rf"),
        ]
        comparisons: Dict[str, Any] = {}
        for cmp_name, base_cell, test_cell in cmps_spec:
            mc = exact_mcnemar(ok_vecs[base_cell], ok_vecs[test_cell])
            mc_far = exact_mcnemar(
                [cells[base_cell]["_ok_by_cid"][c] for c in bug_cids],
                [cells[test_cell]["_ok_by_cid"][c] for c in bug_cids],
            )
            mc_frr = exact_mcnemar(
                [cells[base_cell]["_ok_by_cid"][c] for c in cor_cids],
                [cells[test_cell]["_ok_by_cid"][c] for c in cor_cids],
            )
            comparisons[cmp_name] = {
                "from_cell": base_cell,
                "to_cell": test_cell,
                "delta_acc_pp": (cells[test_cell]["strict_acc"] - cells[base_cell]["strict_acc"]) * 100.0,
                "delta_frr_pp": (cells[test_cell]["frr"] - cells[base_cell]["frr"]) * 100.0,
                "delta_far_pp": (cells[test_cell]["far"] - cells[base_cell]["far"]) * 100.0,
                "n01_b_better": mc["n01_b_better"],
                "n10_a_better": mc["n10_a_better"],
                "p_value": mc["p_value"],
                "mcnemar_far": mc_far,
                "mcnemar_frr": mc_frr,
            }
            pvals_12[f"{m}__{cmp_name}"] = mc["p_value"]

        acc_vf = cells["vf"]["strict_acc"]
        acc_a = cells["vf_trace"]["strict_acc"]
        acc_b = cells["rf_order_only"]["strict_acc"]
        acc_rf = cells["rf"]["strict_acc"]
        total_gain_pp = (acc_rf - acc_vf) * 100.0
        order_effect_pp = (acc_b - acc_vf) * 100.0
        instruction_effect_pp = (acc_a - acc_vf) * 100.0
        interaction_pp = total_gain_pp - order_effect_pp - instruction_effect_pp

        per_model[m] = {
            "status": "complete",
            "cells": {k: {kk: vv for kk, vv in v.items() if kk != "_ok_by_cid"} for k, v in cells.items()},
            "decomposition_pp": {
                "total_rf_gain_pp": total_gain_pp,
                "order_effect_pp": order_effect_pp,
                "instruction_effect_pp": instruction_effect_pp,
                "interaction_pp": interaction_pp,
            },
            "comparisons": comparisons,
        }

    holm_12 = holm_adjust(pvals_12)
    for m, mdata in per_model.items():
        if mdata.get("status") != "complete":
            continue
        for cmp_name, cobj in mdata["comparisons"].items():
            cobj["p_value_holm"] = holm_12[f"{m}__{cmp_name}"]

    complete_ms = [mdata for mdata in per_model.values() if mdata.get("status") == "complete"]
    mean_decomp = {}
    if complete_ms:
        for k in ("total_rf_gain_pp", "order_effect_pp", "instruction_effect_pp", "interaction_pp"):
            mean_decomp[k] = sum(md["decomposition_pp"][k] for md in complete_ms) / len(complete_ms)

    return {
        "n_cases": len(cids),
        "n_pairs": len(gold_subset) // 2,
        "models": per_model,
        "mean_decomposition_pp": mean_decomp,
        "holm_family_12": holm_12,
    }


def write_rf_ablation_report(rf_clean: Dict[str, Any], rf_full: Dict[str, Any], out_path: Path) -> None:
    """Write markdown summary of the 2x2 Reason-First ablation to reports/rf_ablation.md."""
    cell_labels = [
        ("vf", "VF (Verdict-1st)"),
        ("vf_trace", "A: vf_trace"),
        ("rf_order_only", "B: rf_order_only"),
        ("rf", "RF (Reason-1st)"),
    ]
    lines = [
        "# 2x2 Reason-First Ablation Report (`vf_trace` & `rf_order_only`)",
        "",
        "Separates JSON key order (`reason -> verdict -> confidence`) from step-by-step tracing instruction across `gpt-4.1`, `gpt-4.1-mini`, and `gemma-4-31b-it` (greedy `T=0.0`, `n=240` cases/cell, `Clean-114` primary with `Full-120` sensitivity).",
        "",
    ]

    for split_title, data in (("Clean-114 Primary (228 cases)", rf_clean), ("Full-120 Sensitivity (240 cases)", rf_full)):
        lines.append(f"## {split_title}")
        lines.append("")
        lines.append("| Model | Cell | Acc | 95% CI | FAR | FRR | Pair-Both | Brier | Med. Compl. Tok. | Invalid / Abstain | Key-Order Viol. |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for m in RF_ABLATION_MODELS:
            md = data["models"].get(m, {})
            if md.get("status") != "complete":
                lines.append(f"| `{m}` | INCOMPLETE | — | — | — | — | — | — | — | — | — |")
                continue
            for ckey, clabel in cell_labels:
                c = md["cells"][ckey]
                ci = c["acc_ci95"]
                cm = c["confusion_matrix"]
                lines.append(
                    f"| `{m}` | {clabel} | {c['strict_acc']*100:.1f}% | [{ci['low']*100:.1f}, {ci['high']*100:.1f}] | "
                    f"{c['far']*100:.1f}% | {c['frr']*100:.1f}% | {c['pair_both_rate']*100:.1f}% | "
                    f"{c['brier_score']:.4f} | {c['median_completion_tokens']:.1f} | "
                    f"{cm['INVALID']} / {cm['ABSTAIN']} | {c['key_order_violations']} |"
                )
        lines.append("")
        lines.append(f"### RF Accuracy Decomposition ({split_title})")
        lines.append("")
        lines.append("| Model | VF -> RF Total (pp) | Order Effect `B - VF` (pp) | Instruction Effect `A - VF` (pp) | Interaction (pp) |")
        lines.append("|---|---:|---:|---:|---:|")
        for m in RF_ABLATION_MODELS:
            md = data["models"].get(m, {})
            if md.get("status") != "complete":
                continue
            dp = md["decomposition_pp"]
            lines.append(
                f"| `{m}` | {dp['total_rf_gain_pp']:+.1f} | {dp['order_effect_pp']:+.1f} | "
                f"{dp['instruction_effect_pp']:+.1f} | {dp['interaction_pp']:+.1f} |"
            )
        if data.get("mean_decomposition_pp"):
            mp = data["mean_decomposition_pp"]
            lines.append(
                f"| **Mean (3 Models)** | **{mp['total_rf_gain_pp']:+.1f}** | **{mp['order_effect_pp']:+.1f}** | "
                f"**{mp['instruction_effect_pp']:+.1f}** | **{mp['interaction_pp']:+.1f}** |"
            )
        lines.append("")
        lines.append(f"### Exact McNemar Tests — 12-Test Holm Family ({split_title})")
        lines.append("")
        lines.append("| Model | Comparison | Delta Acc (pp) | `n01` (new+) / `n10` (base+) | Raw `p` | Holm `p` (12-test) |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for m in RF_ABLATION_MODELS:
            md = data["models"].get(m, {})
            if md.get("status") != "complete":
                continue
            for cmp_name in ("A_vs_VF", "B_vs_VF", "RF_vs_A", "RF_vs_B"):
                c = md["comparisons"][cmp_name]
                lines.append(
                    f"| `{m}` | `{cmp_name}` | {c['delta_acc_pp']:+.1f} | {c['n01_b_better']} / {c['n10_a_better']} | "
                    f"{c['p_value']:.6g} | {c['p_value_holm']:.6g} |"
                )
        lines.append("")

    # Token & query ledger on the two new cells (Full-120)
    lines.append("## New Queries Executed (2x2 New Cells Only)")
    lines.append("")
    lines.append("| Cell | Model | Queries | Prompt Tok. | Compl. Tok. | Reas. Tok. | Total Tok. | Strict JSON | Key-Order Viol. | Invalid | Abstain |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    q_tot = p_tot = c_tot = r_tot = t_tot = 0
    for ckey in ("vf_trace", "rf_order_only"):
        for m in RF_ABLATION_MODELS:
            md = rf_full["models"].get(m, {})
            if md.get("status") != "complete":
                continue
            c = md["cells"][ckey]
            tk = c["tokens"]
            cm = c["confusion_matrix"]
            q_tot += c["n_cases"]
            p_tot += tk["prompt_tokens"]
            c_tot += tk["completion_tokens"]
            r_tot += tk["reasoning_tokens"]
            t_tot += tk["total_tokens"]
            lines.append(
                f"| `{ckey}` | `{m}` | {c['n_cases']} | {tk['prompt_tokens']:,} | {tk['completion_tokens']:,} | "
                f"{tk['reasoning_tokens']:,} | {tk['total_tokens']:,} | {c['strict_json_rate']*100:.1f}% | "
                f"{c['key_order_violations']} | {cm['INVALID']} | {cm['ABSTAIN']} |"
            )
    lines.append(f"| **Total New Cells** | **3 Models x 2 Cells** | **{q_tot:,}** | **{p_tot:,}** | **{c_tot:,}** | **{r_tot:,}** | **{t_tot:,}** | — | — | — | — |")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    core = summarize_single_track(ROOT / "results_v2", ROOT / "manifests" / "v2_gold.json", 240)
    rf = summarize_single_track(ROOT / "results_ext" / "reason_first", ROOT / "manifests" / "v2_gold.json", 240) if (ROOT / "results_ext" / "reason_first").exists() else {}
    syn = summarize_single_track(ROOT / "results_ext" / "syn_sem85", ROOT / "manifests" / "syn_gold.json", 100) if (ROOT / "results_ext" / "syn_sem85").exists() else {}
    pw = summarize_pairwise(ROOT / "results_ext" / "pairwise", ROOT / "manifests" / "pairwise_gold.json", core)
    ra = summarize_rationale_audit(ROOT / "results_ext" / "rationale_audit", core)
    ens_bundle = summarize_ensembles(core, ROOT / "manifests" / "v2_gold.json")
    sc = summarize_self_consistency(
        ROOT / "results_ext" / "self_consistency",
        ROOT / "manifests" / "v2_gold.json",
        core,
        rf,
        top3_diverse_ok=ens_bundle.get("_top3_diverse_ok"),
        num_samples=5,
    )

    # Paired comparisons within Core
    core_comparisons = {}
    pairs_to_test = [
        ("claude-haiku-5.5", "claude-haiku-5.5-thinking"),
        ("claude-haiku-5.5", "claude-4.6-sonnet"),
        ("claude-haiku-5.5", "gemini-3.1-pro"),
        ("claude-4.6-sonnet", "gemini-3.1-pro"),
        ("gpt-4.1-mini", "gpt-4.1"),
        ("gpt-4.1-mini", "gpt-5.4"),
        ("gpt-4.1", "gpt-5.4"),
        ("gpt-5.4", "o4-mini"),
        ("gpt-5.4", "gpt-oss-120b"),
        ("gemini-3.8-flash", "gemini-3.1-pro"),
        ("gpt-4.1-mini", "o4-mini"),
        ("llama-3.3-70b-instruct", "gemma-4-31b-it"),
        ("gemma-4-31b-it", "gpt-oss-120b"),
    ]
    for a, b in pairs_to_test:
        if a in core and b in core and core[a].get("status") == "complete" and core[b].get("status") == "complete":
            ra_list = core[a]["per_case_results"]
            rb_list = core[b]["per_case_results"]
            a_ok = [r["is_correct"] for r in ra_list]
            b_ok = [r["is_correct"] for r in rb_list]
            a_bug = [r["is_correct"] for r in ra_list if r["true_role"] == "buggy"]
            b_bug = [r["is_correct"] for r in rb_list if r["true_role"] == "buggy"]
            a_cor = [r["is_correct"] for r in ra_list if r["true_role"] == "correct"]
            b_cor = [r["is_correct"] for r in rb_list if r["true_role"] == "correct"]
            core_comparisons[f"{a}__vs__{b}"] = {
                "overall": exact_mcnemar(a_ok, b_ok),
                "buggy_far": exact_mcnemar(a_bug, b_bug),
                "correct_frr": exact_mcnemar(a_cor, b_cor),
            }

    # Family-wise Holm-Bonferroni adjustments on Full-120 comparisons
    cc_holm_all = holm_adjust({k: v["overall"]["p_value"] for k, v in core_comparisons.items()})
    cc_holm_far = holm_adjust({k: v["buggy_far"]["p_value"] for k, v in core_comparisons.items()})
    cc_holm_frr = holm_adjust({k: v["correct_frr"]["p_value"] for k, v in core_comparisons.items()})
    for k, v in core_comparisons.items():
        v["overall"]["p_value_holm"] = cc_holm_all[k]
        v["buggy_far"]["p_value_holm"] = cc_holm_far[k]
        v["correct_frr"]["p_value_holm"] = cc_holm_frr[k]

    pw_holm = holm_adjust({m: p["vs_isolated_pair"]["mcnemar"]["p_value"] for m, p in pw.items() if p.get("status") == "complete" and p.get("vs_isolated_pair")})
    for m, p_val_h in pw_holm.items():
        pw[m]["vs_isolated_pair"]["mcnemar"]["p_value_holm"] = p_val_h

    # Reason-first paired deltas
    rf_deltas = {}
    for m, m_rf in rf.items():
        if m in core and core[m].get("status") == "complete" and m_rf.get("status") == "complete":
            ra_list = core[m]["per_case_results"]
            rb_list = m_rf["per_case_results"]
            a_ok = [r["is_correct"] for r in ra_list]
            b_ok = [r["is_correct"] for r in rb_list]
            a_bug = [r["is_correct"] for r in ra_list if r["true_role"] == "buggy"]
            b_bug = [r["is_correct"] for r in rb_list if r["true_role"] == "buggy"]
            a_cor = [r["is_correct"] for r in ra_list if r["true_role"] == "correct"]
            b_cor = [r["is_correct"] for r in rb_list if r["true_role"] == "correct"]
            pm_a = core[m]["primary_metrics"]
            pm_b = m_rf["primary_metrics"]
            rf_deltas[m] = {
                "acc_vf": pm_a["strict_accuracy"],
                "acc_rf": pm_b["strict_accuracy"],
                "delta_acc_pp": (pm_b["strict_accuracy"] - pm_a["strict_accuracy"]) * 100.0,
                "far_vf": pm_a["false_acceptance_rate"],
                "far_rf": pm_b["false_acceptance_rate"],
                "frr_vf": pm_a["false_rejection_rate"],
                "frr_rf": pm_b["false_rejection_rate"],
                "delta_frr_pp": (pm_b["false_rejection_rate"] - pm_a["false_rejection_rate"]) * 100.0,
                "pair_both_vf": core[m]["pair_breakdown"]["both_correct"] / 120.0,
                "pair_both_rf": m_rf["pair_breakdown"]["both_correct"] / 120.0,
                "strict_json_vf": core[m]["strict_json_rate"],
                "strict_json_rf": m_rf["strict_json_rate"],
                "brier_vf": core[m]["brier_score"],
                "brier_rf": m_rf["brier_score"],
                "calgap_vf_pp": core[m]["calibration"]["calib_gap_pp"],
                "calgap_rf_pp": m_rf["calibration"]["calib_gap_pp"],
                "comp_ok_med_vf": core[m]["token_dynamics"]["comp_ok_median"],
                "comp_ok_med_rf": m_rf["token_dynamics"]["comp_ok_median"],
                "comp_err_med_vf": core[m]["token_dynamics"]["comp_err_median"],
                "comp_err_med_rf": m_rf["token_dynamics"]["comp_err_median"],
                "mcnemar": exact_mcnemar(a_ok, b_ok),
                "mcnemar_far": exact_mcnemar(a_bug, b_bug),
                "mcnemar_frr": exact_mcnemar(a_cor, b_cor),
            }

    rf_holm_all = holm_adjust({m: d["mcnemar"]["p_value"] for m, d in rf_deltas.items()})
    rf_holm_far = holm_adjust({m: d["mcnemar_far"]["p_value"] for m, d in rf_deltas.items()})
    rf_holm_frr = holm_adjust({m: d["mcnemar_frr"]["p_value"] for m, d in rf_deltas.items()})
    for m, d in rf_deltas.items():
        d["mcnemar"]["p_value_holm"] = rf_holm_all[m]
        d["mcnemar_far"]["p_value_holm"] = rf_holm_far[m]
        d["mcnemar_frr"]["p_value_holm"] = rf_holm_frr[m]

    # Matched-design contrast between SynMut-50 and RTLRepairBench-Real (Full-44 + Clean-42)
    syn_gold = json.loads((ROOT / "manifests" / "syn_gold.json").read_text(encoding="utf-8"))
    core_gold = json.loads((ROOT / "manifests" / "v2_gold.json").read_text(encoding="utf-8"))
    syn_design_by_case = {g["case_id"]: g["base_design_id"] for g in syn_gold}
    core_design_by_case = {g["case_id"]: g["base_design_id"] for g in core_gold}
    syn_designs = set(syn_design_by_case.values())
    real_designs = {g["base_design_id"] for g in core_gold if g["dataset_source"] == "RTLRepairBench"}
    artifact_designs = {g["base_design_id"] for g in core_gold if g["pair_id"] in ORACLE_ARTIFACT_PAIRS}
    overlap_designs = sorted(syn_designs & real_designs)
    clean_overlap_designs = sorted(set(overlap_designs) - artifact_designs)

    def _matched_table(designs_list: List[str]) -> Dict[str, Any]:
        out_m: Dict[str, Any] = {}
        d_set = set(designs_list)
        for m, ms in syn.items():
            if ms.get("status") != "complete" or m not in core or core[m].get("status") != "complete":
                continue
            syn_by_key = {(syn_design_by_case[r["case_id"]], r["true_role"]): r for r in ms["per_case_results"] if syn_design_by_case[r["case_id"]] in d_set}
            real_by_key = {(core_design_by_case[r["case_id"]], r["true_role"]): r for r in core[m]["per_case_results"] if r["dataset_source"] == "RTLRepairBench" and core_design_by_case[r["case_id"]] in d_set}
            keys_sorted = sorted(syn_by_key.keys())
            syn_ok_vec = [syn_by_key[k]["is_correct"] for k in keys_sorted]
            real_ok_vec = [real_by_key[k]["is_correct"] for k in keys_sorted]
            bug_keys = [k for k in keys_sorted if k[1] == "buggy"]
            cor_keys = [k for k in keys_sorted if k[1] == "correct"]
            syn_bug_fn = sum(1 for k in bug_keys if not syn_by_key[k]["is_correct"])
            real_bug_fn = sum(1 for k in bug_keys if not real_by_key[k]["is_correct"])
            syn_cor_fp = sum(1 for k in cor_keys if not syn_by_key[k]["is_correct"])
            real_cor_fp = sum(1 for k in cor_keys if not real_by_key[k]["is_correct"])
            out_m[m] = {
                "overlap_designs": len(designs_list),
                "syn_all_acc": ms["primary_metrics"]["strict_accuracy"],
                "real_all_acc": core[m]["subgroups"]["by_source"]["RTLRepairBench"]["strict_accuracy"],
                "syn_matched_cases": len(keys_sorted),
                "syn_matched_acc": sum(syn_ok_vec) / len(syn_ok_vec),
                "real_matched_acc": sum(real_ok_vec) / len(real_ok_vec),
                "syn_matched_acc_ci95": clopper_pearson_ci(sum(syn_ok_vec), len(syn_ok_vec)),
                "real_matched_acc_ci95": clopper_pearson_ci(sum(real_ok_vec), len(real_ok_vec)),
                "syn_matched_far": syn_bug_fn / len(bug_keys),
                "real_matched_far": real_bug_fn / len(bug_keys),
                "syn_matched_far_ci95": clopper_pearson_ci(syn_bug_fn, len(bug_keys)),
                "real_matched_far_ci95": clopper_pearson_ci(real_bug_fn, len(bug_keys)),
                "syn_matched_frr": syn_cor_fp / len(cor_keys),
                "real_matched_frr": real_cor_fp / len(cor_keys),
                "matched_mcnemar": exact_mcnemar(real_ok_vec, syn_ok_vec),
                "matched_far_mcnemar": exact_mcnemar([real_by_key[k]["is_correct"] for k in bug_keys], [syn_by_key[k]["is_correct"] for k in bug_keys]),
                "syn_far": ms["primary_metrics"]["false_acceptance_rate"],
                "syn_frr": ms["primary_metrics"]["false_rejection_rate"],
            }
        hm_acc = holm_adjust({m: v["matched_mcnemar"]["p_value"] for m, v in out_m.items()})
        hm_far = holm_adjust({m: v["matched_far_mcnemar"]["p_value"] for m, v in out_m.items()})
        for m, v in out_m.items():
            v["matched_mcnemar"]["p_value_holm"] = hm_acc[m]
            v["matched_far_mcnemar"]["p_value_holm"] = hm_far[m]
        return out_m

    matched_contrast = _matched_table(overlap_designs)
    matched_42_contrast = _matched_table(clean_overlap_designs)

    # Pooled SynMut-50 per-operator stats across all complete models
    complete_syn_models = [ms for ms in syn.values() if ms.get("status") == "complete"]
    syn_pooled: Dict[str, Any] = {}
    if complete_syn_models:
        op_fn: Dict[str, int] = defaultdict(int)
        op_n: Dict[str, int] = defaultdict(int)
        tot_fn = tot_bug = 0
        for ms in complete_syn_models:
            for op, vals in ms["by_category"].items():
                op_fn[op] += vals["fn"]
                op_n[op] += vals["buggy_cases"]
                tot_fn += vals["fn"]
                tot_bug += vals["buggy_cases"]
        syn_pooled = {
            "n_models": len(complete_syn_models),
            "total_far_ci95": clopper_pearson_ci(tot_fn, tot_bug),
            "by_operator": {
                op: clopper_pearson_ci(op_fn[op], op_n[op])
                for op in sorted(op_n.keys(), key=lambda k: -(op_fn[k] / op_n[k] if op_n[k] else 0.0))
            },
        }

    # --- Clean-114 Primary Scientific Split (Excluding 6 Oracle-Artifact Pairs) ---
    core_clean, clean_cids, clean_pids = _compute_clean114_single_track(core, ROOT / "results_v2", ROOT / "manifests" / "v2_gold.json")
    rf_clean, _, _ = _compute_clean114_single_track(rf, ROOT / "results_ext" / "reason_first", ROOT / "manifests" / "v2_gold.json") if rf else ({}, [], [])
    ens_clean_bundle = _compute_clean114_ensembles_and_cv(core_clean, ROOT / "manifests" / "v2_gold.json")
    pw_clean = _compute_clean114_pairwise(ROOT / "results_ext" / "pairwise", ROOT / "manifests" / "pairwise_gold.json", core_clean)
    ra_clean = _compute_clean114_rationale_audit(ROOT / "results_ext" / "rationale_audit", ROOT / "manifests" / "v2_gold.json")
    sc_clean = _compute_clean114_self_consistency(
        ROOT / "results_ext" / "self_consistency",
        ROOT / "manifests" / "v2_gold.json",
        core_clean,
        rf_clean,
        top3_diverse_ok=ens_clean_bundle.get("_top3_diverse_ok"),
        num_samples=5,
    )

    core_clean_comparisons = {}
    for a, b in pairs_to_test:
        if a in core_clean and b in core_clean:
            ra_list = core_clean[a]["per_case_results"]
            rb_list = core_clean[b]["per_case_results"]
            core_clean_comparisons[f"{a}__vs__{b}"] = {
                "overall": exact_mcnemar([r["is_correct"] for r in ra_list], [r["is_correct"] for r in rb_list]),
                "buggy_far": exact_mcnemar([r["is_correct"] for r in ra_list if r["true_role"] == "buggy"], [r["is_correct"] for r in rb_list if r["true_role"] == "buggy"]),
                "correct_frr": exact_mcnemar([r["is_correct"] for r in ra_list if r["true_role"] == "correct"], [r["is_correct"] for r in rb_list if r["true_role"] == "correct"]),
            }
    ccc_ha = holm_adjust({k: v["overall"]["p_value"] for k, v in core_clean_comparisons.items()})
    ccc_hf = holm_adjust({k: v["buggy_far"]["p_value"] for k, v in core_clean_comparisons.items()})
    ccc_hr = holm_adjust({k: v["correct_frr"]["p_value"] for k, v in core_clean_comparisons.items()})
    for k, v in core_clean_comparisons.items():
        v["overall"]["p_value_holm"] = ccc_ha[k]
        v["buggy_far"]["p_value_holm"] = ccc_hf[k]
        v["correct_frr"]["p_value_holm"] = ccc_hr[k]

    pw_clean_holm = holm_adjust({m: p["vs_isolated_pair"]["mcnemar"]["p_value"] for m, p in pw_clean.items() if p.get("vs_isolated_pair")})
    for m, ph in pw_clean_holm.items():
        pw_clean[m]["vs_isolated_pair"]["mcnemar"]["p_value_holm"] = ph

    rf_clean_deltas = {}
    n_clean_p = len(clean_pids)
    for m, m_rf in rf_clean.items():
        if m in core_clean:
            ra_list = core_clean[m]["per_case_results"]
            rb_list = m_rf["per_case_results"]
            pm_a = core_clean[m]["primary_metrics"]
            pm_b = m_rf["primary_metrics"]
            rf_clean_deltas[m] = {
                "acc_vf": pm_a["strict_accuracy"],
                "acc_rf": pm_b["strict_accuracy"],
                "delta_acc_pp": (pm_b["strict_accuracy"] - pm_a["strict_accuracy"]) * 100.0,
                "far_vf": pm_a["false_acceptance_rate"],
                "far_rf": pm_b["false_acceptance_rate"],
                "frr_vf": pm_a["false_rejection_rate"],
                "frr_rf": pm_b["false_rejection_rate"],
                "delta_frr_pp": (pm_b["false_rejection_rate"] - pm_a["false_rejection_rate"]) * 100.0,
                "pair_both_vf": core_clean[m]["pair_breakdown"]["both_correct"] / n_clean_p,
                "pair_both_rf": m_rf["pair_breakdown"]["both_correct"] / n_clean_p,
                "brier_vf": core_clean[m]["brier_score"],
                "brier_rf": m_rf["brier_score"],
                "comp_ok_med_vf": core_clean[m]["token_dynamics"]["comp_ok_median"],
                "comp_ok_med_rf": m_rf["token_dynamics"]["comp_ok_median"],
                "mcnemar": exact_mcnemar([r["is_correct"] for r in ra_list], [r["is_correct"] for r in rb_list]),
                "mcnemar_far": exact_mcnemar([r["is_correct"] for r in ra_list if r["true_role"] == "buggy"], [r["is_correct"] for r in rb_list if r["true_role"] == "buggy"]),
                "mcnemar_frr": exact_mcnemar([r["is_correct"] for r in ra_list if r["true_role"] == "correct"], [r["is_correct"] for r in rb_list if r["true_role"] == "correct"]),
            }
    rfc_ha = holm_adjust({m: d["mcnemar"]["p_value"] for m, d in rf_clean_deltas.items()})
    rfc_hf = holm_adjust({m: d["mcnemar_far"]["p_value"] for m, d in rf_clean_deltas.items()})
    rfc_hr = holm_adjust({m: d["mcnemar_frr"]["p_value"] for m, d in rf_clean_deltas.items()})
    for m, d in rf_clean_deltas.items():
        d["mcnemar"]["p_value_holm"] = rfc_ha[m]
        d["mcnemar_far"]["p_value_holm"] = rfc_hf[m]
        d["mcnemar_frr"]["p_value_holm"] = rfc_hr[m]

    # --- Task B: 2x2 Reason-First Ablation (vf_trace & rf_order_only) ---
    rf_ablation_dir = ROOT / "results_ext" / "rf_ablation"
    rf_ablation_full = summarize_rf_ablation(
        ROOT / "results_v2",
        ROOT / "results_ext" / "reason_first",
        rf_ablation_dir,
        ROOT / "manifests" / "v2_gold.json",
    )
    rf_ablation_clean = summarize_rf_ablation(
        ROOT / "results_v2",
        ROOT / "results_ext" / "reason_first",
        rf_ablation_dir,
        ROOT / "manifests" / "v2_gold.json",
        exclude_pairs=ORACLE_ARTIFACT_PAIRS,
    )
    write_rf_ablation_report(rf_ablation_clean, rf_ablation_full, ROOT / "reports" / "rf_ablation.md")

    # Strip per_case_results for concise summary file, print tables nicely
    summary_clean = {
        "excluded_oracle_artifact_pairs": sorted(ORACLE_ARTIFACT_PAIRS),
        "core_v2": {m: {k: v for k, v in d.items() if k != "per_case_results"} for m, d in core.items()},
        "core_comparisons": core_comparisons,
        "ensembles": ens_bundle["gates"],
        "reason_first": {m: {k: v for k, v in d.items() if k != "per_case_results"} for m, d in rf.items()},
        "reason_first_deltas": rf_deltas,
        "syn_sem85": {m: {k: v for k, v in d.items() if k != "per_case_results"} for m, d in syn.items()},
        "syn_pooled_operators": syn_pooled,
        "matched_syn_vs_real": matched_contrast,
        "matched_42_syn_vs_real": matched_42_contrast,
        "pairwise": pw,
        "rationale_audit": ra,
        "self_consistency": sc,
        "core_clean114": {m: {k: v for k, v in d.items() if k != "per_case_results"} for m, d in core_clean.items()},
        "core_clean114_comparisons": core_clean_comparisons,
        "ensembles_clean114": ens_clean_bundle["gates"],
        "reason_first_clean114": {m: {k: v for k, v in d.items() if k != "per_case_results"} for m, d in rf_clean.items()},
        "reason_first_deltas_clean114": rf_clean_deltas,
        "pairwise_clean114": pw_clean,
        "rationale_audit_clean114": ra_clean,
        "self_consistency_clean114": sc_clean,
        "rf_ablation": rf_ablation_full,
        "rf_ablation_clean114": rf_ablation_clean,
    }
    out_dir = ROOT / "results_ext"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "all_metrics.json").write_text(json.dumps(summary_clean, indent=2), encoding="utf-8")

    print("\n=== CORE_V2 SUMMARY (N=240) ===")
    for m, d in sorted(core.items(), key=lambda kv: kv[1].get("primary_metrics", {}).get("strict_accuracy", -1), reverse=True):
        if d.get("status") != "complete":
            print(f"  {m:28s} INCOMPLETE ({d['count']}/{d['expected']})")
            continue
        pm = d["primary_metrics"]
        pb = d["pair_breakdown"]
        ci = d["ci95"]["strict_acc"]
        print(
            f"  {m:28s} Acc={pm['strict_accuracy']*100:5.1f}% [{ci['low']*100:.1f},{ci['high']*100:.1f}]  FAR={pm['false_acceptance_rate']*100:5.1f}%  "
            f"FRR={pm['false_rejection_rate']*100:5.1f}%  Both={pb['both_correct']:3d}/120  StrictJSON={d['strict_json_rate']*100:5.1f}%  RTok={d['sum_reasoning_tokens']}"
        )

    if rf_deltas:
        print("\n=== REASON-FIRST DELTAS ===")
        for m, dd in rf_deltas.items():
            print(
                f"  {m:28s} Acc: {dd['acc_vf']*100:5.1f}% -> {dd['acc_rf']*100:5.1f}% ({dd['delta_acc_pp']:+5.1f} pp, p={dd['mcnemar']['p_value']:.4f}) | "
                f"FRR: {dd['frr_vf']*100:5.1f}% -> {dd['frr_rf']*100:5.1f}% ({dd['delta_frr_pp']:+5.1f} pp)"
            )

    if matched_contrast:
        print("\n=== SYN_SEM85 vs REAL MATCHED ===")
        for m, mc in matched_contrast.items():
            print(
                f"  {m:28s} SynAll={mc['syn_all_acc']*100:5.1f}% vs RealAll={mc['real_all_acc']*100:5.1f}% | "
                f"Matched(44): Syn={mc['syn_matched_acc']*100:5.1f}% vs Real={mc['real_matched_acc']*100:5.1f}%"
            )

    if pw:
        print("\n=== PAIRWISE BIDIRECTIONAL (N=240 queries / 120 pairs) ===")
        for m, p in pw.items():
            if p.get("status") != "complete":
                print(f"  {m:28s} INCOMPLETE ({p['count']}/{p['expected']})")
                continue
            vs_iso = f" | Swap={p['swap_consistent_pair_acc']*100:5.1f}% vs IsoBoth={p['vs_isolated_pair']['isolated_pair_joint_rate']*100:5.1f}% (p={p['vs_isolated_pair']['mcnemar']['p_value']:.4f})" if p["vs_isolated_pair"] else ""
            print(
                f"  {m:28s} Pool={p['pooled_query_acc']*100:5.1f}% (FWD={p['fwd_acc']*100:5.1f}%, REV={p['rev_acc']*100:5.1f}%) "
                f"Flip={p['order_flip_count']:2d} Bias(A-B)={p['position_bias_A_minus_B']*100:+5.1f}pp{vs_iso}"
            )

    if ra and ra.get("per_model"):
        print("\n=== RATIONALE FIDELITY AUDIT ===")
        if ra.get("agreement"):
            ag = ra["agreement"]
            print(f"  Inter-judge ({ag['judges'][0]} vs {ag['judges'][1]}, n={ag['n_shared']}): Agree={ag['raw_agreement']*100:.1f}%, kappa={ag['cohens_kappa']:.3f}")
        for m, rm in ra["per_model"].items():
            print(
                f"  {m:28s} TP={rm['tp_audited']:3d} Grounded={rm['grounded']:3d} ({rm['grounded_precision_among_tp']*100:5.1f}%) "
                f"Spurious={rm['spurious_claim']:2d} Vague={rm['vague']:2d} | NomTPR={rm['nominal_tpr']*100:5.1f}% -> EffSens={rm['effective_grounded_sensitivity']*100:5.1f}%"
            )

    if sc:
        print("\n=== SELF-CONSISTENCY (MAJORITY@5 @ T=0.7) ===")
        for m, sd in sc.items():
            if sd.get("status") != "complete":
                print(f"  {m:28s} INCOMPLETE ({sd['count']}/{sd['expected']})")
                continue
            mj = sd["majority5"]
            vd = sd["vote_dynamics"]
            vt0 = sd["vs_greedy_t0"]
            t0_str = f"vsT0={vt0['delta_acc_pp']:+5.1f}pp (p={vt0['mcnemar_overall']['p_value']:.4f})" if vt0 else ""
            print(
                f"  {m:28s} Mean1={sd['mean_sample_acc']*100:5.1f}%+-{sd['std_sample_acc_pp']:.2f}  "
                f"Maj@5={mj['strict_acc']*100:5.1f}% ({mj['strict_acc_fraction']}) FAR={mj['far']*100:4.1f}% FRR={mj['frr']*100:4.1f}% "
                f"Splits(4:1/3:2)={vd['split_4_1']}/{vd['split_3_2']}  {t0_str}"
            )

    if rf_ablation_clean.get("models"):
        print("\n=== 2x2 REASON-FIRST ABLATION (Clean-114 Primary) ===")
        for m, md in rf_ablation_clean["models"].items():
            if md.get("status") != "complete":
                print(f"  {m:28s} INCOMPLETE")
                continue
            cs = md["cells"]
            dp = md["decomposition_pp"]
            print(
                f"  {m:28s} VF={cs['vf']['strict_acc']*100:5.1f}%  A(vf_trace)={cs['vf_trace']['strict_acc']*100:5.1f}%  "
                f"B(rf_order_only)={cs['rf_order_only']['strict_acc']*100:5.1f}%  RF={cs['rf']['strict_acc']*100:5.1f}%  | "
                f"Order={dp['order_effect_pp']:+5.1f}pp  Instr={dp['instruction_effect_pp']:+5.1f}pp  Inter={dp['interaction_pp']:+5.1f}pp"
            )


if __name__ == "__main__":
    main()
