"""Tests for extended manifests (SynMut-50, Pairwise-240) and robust JSON extraction."""

import json
import re
from pathlib import Path

from rtlreviewbench.prompt import (
    REASON_FIRST_SYSTEM_PROMPT,
    REASON_FIRST_USER_TEMPLATE,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    extract_json_key_order,
    parse_model_response,
)
from scripts.evaluate_extensions import (
    ORACLE_ARTIFACT_PAIRS,
    clopper_pearson_ci,
    exact_mcnemar,
    holm_adjust,
    summarize_rf_ablation,
    summarize_self_consistency,
)
from scripts.run_extensions import (
    RF_ORDER_ONLY_SYSTEM_PROMPT,
    RF_ORDER_ONLY_USER_TEMPLATE,
    VF_TRACE_SYSTEM_PROMPT,
    VF_TRACE_USER_TEMPLATE,
    build_extension_review_prompt,
    parse_pairwise_response,
)

FORBIDDEN_PUBLIC_KEYS = {
    "candidate_role",
    "gold_label",
    "gold_choice",
    "label",
    "is_buggy",
    "ground_truth",
    "verification",
    "faulty_test_result",
    "fixed_test_result",
    "mismatches",
    "upstream_task_id",
}

COMMENT_LEAK_PAT = re.compile(
    r"\b(bugs?|buggy|broken|golden|wrong|faulty?|mistake|accidentally|uhh|repair|injected)\b",
    re.IGNORECASE,
)


def test_syn_manifest_cleanliness_and_balance():
    pub_path = Path("manifests/syn_public.jsonl")
    gold_path = Path("manifests/syn_gold.json")
    assert pub_path.exists() and gold_path.exists()

    pub_rows = [json.loads(l) for l in pub_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    gold_rows = json.loads(gold_path.read_text(encoding="utf-8"))

    assert len(pub_rows) == 100
    assert len(gold_rows) == 100
    assert sum(1 for g in gold_rows if g["candidate_role"] == "buggy") == 50
    assert sum(1 for g in gold_rows if g["candidate_role"] == "correct") == 50
    assert len({g["base_design_id"] for g in gold_rows}) == 50

    for row in pub_rows:
        for fk in FORBIDDEN_PUBLIC_KEYS:
            assert fk not in row, f"Forbidden key {fk} in syn_public {row.get('case_id')}"
        assert row["filename"] == "design.sv"
        rtl = row["candidate_rtl"]
        assert "RefModule" not in rtl
        assert not re.search(r"\bProb\d+_\w+", rtl)
        assert re.search(r"^\s*module\s+TopModule\b", rtl, re.MULTILINE)
        comments = "\n".join(re.findall(r"//[^\n]*", rtl) + re.findall(r"/\*.*?\*/", rtl, re.DOTALL))
        assert not COMMENT_LEAK_PAT.search(comments), f"Comment leak in {row['case_id']}"


def test_pairwise_manifest_cleanliness_and_symmetry():
    pub_path = Path("manifests/pairwise_public.jsonl")
    gold_path = Path("manifests/pairwise_gold.json")
    assert pub_path.exists() and gold_path.exists()

    pub_rows = [json.loads(l) for l in pub_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    gold_rows = json.loads(gold_path.read_text(encoding="utf-8"))

    assert len(pub_rows) == 240
    assert len(gold_rows) == 240
    assert sum(1 for g in gold_rows if g["gold_choice"] == "A") == 120
    assert sum(1 for g in gold_rows if g["gold_choice"] == "B") == 120

    by_pair = {}
    for row in pub_rows:
        for fk in FORBIDDEN_PUBLIC_KEYS:
            assert fk not in row, f"Forbidden key {fk} in pairwise_public {row.get('query_id')}"
        for rtl_key in ("candidate_a_rtl", "candidate_b_rtl"):
            rtl = row[rtl_key]
            assert "RefModule" not in rtl
            assert not re.search(r"\bProb\d+_\w+", rtl)
            comments = "\n".join(re.findall(r"//[^\n]*", rtl) + re.findall(r"/\*.*?\*/", rtl, re.DOTALL))
            assert not COMMENT_LEAK_PAT.search(comments), f"Comment leak in {row['query_id']}"
        by_pair.setdefault(row["pair_id"], {})[row["direction"]] = row

    assert len(by_pair) == 120
    for pid, dirs in by_pair.items():
        assert set(dirs.keys()) == {"FWD", "REV"}, f"Missing direction for {pid}"
        assert dirs["FWD"]["candidate_a_rtl"] == dirs["REV"]["candidate_b_rtl"]
        assert dirs["FWD"]["candidate_b_rtl"] == dirs["REV"]["candidate_a_rtl"]


def test_balanced_json_extraction_with_verilog_braces():
    strict_text = '{"verdict":"correct","confidence":0.9,"reason":"Clean shift register."}'
    p1 = parse_model_response(strict_text)
    assert p1["verdict"] == "correct"
    assert p1["valid_json"] is True
    assert p1["strict_json"] is True

    prose_with_braces = (
        "Let us trace `a_tem <= {a, a_tem[6:0]}` vs `{a_tem[6:0], a}`.\n"
        "Clearly MSB shift is reversed.\n\n"
        '{"verdict":"buggy","confidence":0.95,"reason":"Reversed shift direction in `{a, a_tem[6:0]}`."}'
    )
    p2 = parse_model_response(prose_with_braces)
    assert p2["verdict"] == "buggy"
    assert p2["valid_json"] is True
    assert p2["strict_json"] is False

    pw_prose = (
        "Candidate A uses `{q[5], q[4:0]}` while Candidate B uses `{q[4:0], q[5]}`.\n"
        '{"choice":"B","confidence":0.98,"reason":"Candidate B rotates left properly."}'
    )
    p3 = parse_pairwise_response(pw_prose)
    assert p3["choice"] == "B"
    assert p3["valid_json"] is True
    assert p3["strict_json"] is False

    bad = parse_model_response('{"verdict":"buggy","confidence":1.5,"reason":"bad"}')
    assert bad["verdict"] == "invalid"


def test_clopper_pearson_and_exact_mcnemar():
    c0 = clopper_pearson_ci(0, 10)
    assert c0["low"] == 0.0
    assert abs(c0["high"] - (1.0 - 0.025 ** 0.1)) < 1e-5

    c10 = clopper_pearson_ci(10, 10)
    assert c10["high"] == 1.0
    assert abs(c10["low"] - (0.025 ** 0.1)) < 1e-5

    c3 = clopper_pearson_ci(3, 10)
    c7 = clopper_pearson_ci(7, 10)
    assert abs(c3["low"] - (1.0 - c7["high"])) < 1e-5
    assert abs(c3["high"] - (1.0 - c7["low"])) < 1e-5

    mc_same = exact_mcnemar([True, False, True], [True, False, True])
    assert mc_same["p_value"] == 1.0
    assert mc_same["n01_b_better"] == 0

    mc_9_0 = exact_mcnemar([False] * 9 + [True], [True] * 10)
    assert abs(mc_9_0["p_value"] - (2.0 / (2 ** 9))) < 1e-12


def test_summarize_self_consistency_fixture(tmp_path: Path):
    gold = [
        {"case_id": "C001", "pair_id": "P001", "candidate_role": "buggy"},
        {"case_id": "C002", "pair_id": "P001", "candidate_role": "correct"},
        {"case_id": "C003", "pair_id": "P002", "candidate_role": "buggy"},
        {"case_id": "C004", "pair_id": "P002", "candidate_role": "correct"},
    ]
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps(gold), encoding="utf-8")

    # Complete model: C001 buggy (4 buggy, 1 correct -> buggy OK),
    # C002 correct (5 correct -> correct OK),
    # C003 buggy (3 correct, 2 buggy -> FN under maj@5!),
    # C004 correct (3 buggy, 2 correct -> FP under maj@5!)
    vote_plan = {
        "C001": ["buggy", "buggy", "buggy", "buggy", "correct"],
        "C002": ["correct", "correct", "correct", "correct", "correct"],
        "C003": ["correct", "buggy", "correct", "buggy", "correct"],
        "C004": ["buggy", "correct", "buggy", "correct", "buggy"],
    }
    raw_dir = tmp_path / "sc"
    m_dir = raw_dir / "test-model"
    m_dir.mkdir(parents=True)
    recs = []
    for s_idx in range(5):
        for g in gold:
            cid = g["case_id"]
            v = vote_plan[cid][s_idx]
            recs.append({
                "case_id": cid,
                "pair_id": g["pair_id"],
                "sample_idx": s_idx,
                "sample_id": f"{cid}__s{s_idx}",
                "temperature": 0.7,
                "raw_response": json.dumps({"verdict": v, "confidence": 0.9, "reason": "ok"}),
                "parsed": {"verdict": v, "confidence": 0.9, "reason": "ok", "valid_json": True, "strict_json": True},
                "completion_tokens": 100,
                "total_tokens": 200,
            })
    (m_dir / "raw_responses.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")

    inc_dir = raw_dir / "incomplete-model"
    inc_dir.mkdir()
    (inc_dir / "raw_responses.jsonl").write_text(json.dumps(recs[0]) + "\n", encoding="utf-8")

    out = summarize_self_consistency(raw_dir, gold_path, {}, {}, None, num_samples=5)
    assert out["incomplete-model"]["status"] == "incomplete"
    tm = out["test-model"]
    assert tm["status"] == "complete"
    assert tm["majority5"]["strict_acc"] == 0.5
    assert tm["majority5"]["far"] == 0.5
    assert tm["majority5"]["frr"] == 0.5
    assert tm["vote_dynamics"]["unanimous_correct"] == 1
    assert tm["vote_dynamics"]["split_4_1"] == 1
    assert tm["vote_dynamics"]["split_3_2"] == 2
    assert len(tm["sample_accs"]) == 5


def test_holm_adjust_and_oracle_artifact_exclusion():
    adj = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.50})
    assert abs(adj["a"] - 0.04) < 1e-12
    assert abs(adj["c"] - 0.09) < 1e-12
    assert abs(adj["b"] - 0.09) < 1e-12
    assert abs(adj["d"] - 0.50) < 1e-12
    assert all(0.0 <= v <= 1.0 for v in adj.values())

    assert len(ORACLE_ARTIFACT_PAIRS) == 6
    assert ORACLE_ARTIFACT_PAIRS == {"P018", "P020", "P022", "P028", "P047", "P084"}

    gold_rows = json.loads(Path("manifests/v2_gold.json").read_text(encoding="utf-8"))
    clean_pairs = {g["pair_id"] for g in gold_rows if g["pair_id"] not in ORACLE_ARTIFACT_PAIRS}
    assert len(clean_pairs) == 114


def test_all_metrics_clean114_consistency():
    metrics_path = Path("results_ext/all_metrics.json")
    assert metrics_path.exists()
    data = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert set(data["excluded_oracle_artifact_pairs"]) == ORACLE_ARTIFACT_PAIRS
    assert len(data["core_clean114"]) == 12
    for m, row in data["core_clean114"].items():
        assert row["status"] == "complete", m
        assert row["ci95"]["strict_acc"]["n"] == 228
        assert row["ci95"]["far"]["n"] == 114
        assert row["ci95"]["frr"]["n"] == 114

    for cmp_row in data["core_clean114_comparisons"].values():
        assert "p_value_holm" in cmp_row["overall"]
        assert "p_value_holm" in cmp_row["correct_frr"]

    m42 = data["matched_42_syn_vs_real"]
    assert all(v["overlap_designs"] == 42 for v in m42.values())
    assert all("p_value_holm" in v["matched_mcnemar"] for v in m42.values())
    assert all("p_value_holm" in v["mcnemar"] for m, v in data["reason_first_deltas_clean114"].items())
    assert all("p_value_holm" in v["vs_isolated_pair"]["mcnemar"] for v in data["pairwise_clean114"].values())
    assert data["rationale_audit_clean114"]["n_buggy_clean"] == 114
    assert data["rationale_audit_clean114"]["total_tp_clean"] == 1309

    for key, exp_n in (("rf_ablation_clean114", 228), ("rf_ablation", 240)):
        ab = data[key]
        assert ab["n_cases"] == exp_n
        assert len(ab["models"]) == 3
        assert len(ab["holm_family_12"]) == 12
        for m, m_row in ab["models"].items():
            assert set(m_row["cells"].keys()) == {"vf", "vf_trace", "rf_order_only", "rf"}
            for c_row in m_row["cells"].values():
                assert c_row["status"] == "complete", (key, m)
                assert c_row["n_cases"] == exp_n
                assert c_row["acc_ci95"]["n"] == exp_n
            assert all("p_value_holm" in cmp for cmp in m_row["comparisons"].values())


def test_extract_json_key_order_and_rf_ablation_prompts():
    assert extract_json_key_order('{"verdict":"correct","confidence":0.9,"reason":"ok"}') == [
        "verdict",
        "confidence",
        "reason",
    ]
    assert extract_json_key_order(
        'Preface `{a, b}` text\n```json\n{"reason":"step 1","verdict":"buggy","confidence":0.8}\n```'
    ) == ["reason", "verdict", "confidence"]
    assert extract_json_key_order("not json") is None

    # Exact baseline prompts remain frozen
    assert SYSTEM_PROMPT.splitlines()[3].startswith('{"verdict":')
    assert REASON_FIRST_SYSTEM_PROMPT.splitlines()[3].startswith('{"reason":')

    # Cell A (vf_trace): Verdict-First schema order + tracing instruction only
    pr_a = build_extension_review_prompt("spec here", "module TopModule(); endmodule", "vf_trace")
    assert pr_a["system"] == VF_TRACE_SYSTEM_PROMPT
    assert 'In "reason", trace the RTL behavior step by step against the specification.' in pr_a["system"]
    assert pr_a["user"].splitlines()[:-1] == USER_TEMPLATE.format(
        specification="spec here", candidate_rtl="module TopModule(); endmodule"
    ).splitlines()[:-1]
    assert pr_a["user"].splitlines()[-1] == "Output JSON only, with verdict, confidence, and a step-by-step reason."

    # Cell B (rf_order_only): Reason-First key order without tracing instruction
    pr_b = build_extension_review_prompt("spec here", "module TopModule(); endmodule", "rf_order_only")
    assert pr_b["system"] == RF_ORDER_ONLY_SYSTEM_PROMPT
    assert "step by step" not in pr_b["system"].lower()
    assert pr_b["user"].splitlines()[:-1] == USER_TEMPLATE.format(
        specification="spec here", candidate_rtl="module TopModule(); endmodule"
    ).splitlines()[:-1]
    assert pr_b["user"].splitlines()[-1] == "Output JSON only, with reason, verdict, and confidence, in that key order."
    assert REASON_FIRST_USER_TEMPLATE.splitlines()[:-1] == USER_TEMPLATE.splitlines()[:-1]


def test_summarize_rf_ablation_fixture(tmp_path: Path):
    # 4 cases (2 clean pairs + 0 artifacts, plus optional excluded pair)
    gold = [
        {"case_id": "C001", "pair_id": "P001", "candidate_role": "buggy"},
        {"case_id": "C002", "pair_id": "P001", "candidate_role": "correct"},
        {"case_id": "C003", "pair_id": "P002", "candidate_role": "buggy"},
        {"case_id": "C004", "pair_id": "P002", "candidate_role": "correct"},
        {"case_id": "C005", "pair_id": "P018", "candidate_role": "buggy"},
        {"case_id": "C006", "pair_id": "P018", "candidate_role": "correct"},
    ]
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps(gold), encoding="utf-8")

    # On clean 4 cases: VF=50%, A=75%, B=75%, RF=100% -> order=+25, instr=+25, interaction=0
    plans = {
        "vf": {"C001": "buggy", "C002": "buggy", "C003": "correct", "C004": "correct", "C005": "correct", "C006": "correct"},
        "vf_trace": {"C001": "buggy", "C002": "correct", "C003": "correct", "C004": "correct", "C005": "correct", "C006": "correct"},
        "rf_order_only": {"C001": "buggy", "C002": "buggy", "C003": "buggy", "C004": "correct", "C005": "correct", "C006": "correct"},
        "rf": {"C001": "buggy", "C002": "correct", "C003": "buggy", "C004": "correct", "C005": "buggy", "C006": "correct"},
    }
    core_dir = tmp_path / "core"
    rf_dir = tmp_path / "rf"
    abl_dir = tmp_path / "abl"
    models = ("gpt-4.1", "gpt-4.1-mini", "gemma-4-31b-it")

    def _write_cell(folder: Path, cell_key: str, order_keys: list[str]) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        recs = []
        for g in gold:
            v = plans[cell_key][g["case_id"]]
            payload = {k: (v if k == "verdict" else (0.9 if k == "confidence" else "ok")) for k in order_keys}
            recs.append({
                "case_id": g["case_id"],
                "pair_id": g["pair_id"],
                "raw_response": json.dumps(payload),
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "reasoning_tokens": 0,
                "total_tokens": 150,
            })
        (folder / "raw_responses.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")

    for m in models:
        _write_cell(core_dir / m, "vf", ["verdict", "confidence", "reason"])
        _write_cell(abl_dir / "vf_trace" / m, "vf_trace", ["verdict", "confidence", "reason"])
        _write_cell(abl_dir / "rf_order_only" / m, "rf_order_only", ["reason", "verdict", "confidence"])
        _write_cell(rf_dir / m, "rf", ["reason", "verdict", "confidence"])

    res_clean = summarize_rf_ablation(
        core_dir, rf_dir, abl_dir, gold_path, exclude_pairs=ORACLE_ARTIFACT_PAIRS, models=models, full_expected_n=6
    )
    assert res_clean["n_cases"] == 4
    assert len(res_clean["holm_family_12"]) == 12
    m0 = res_clean["models"]["gpt-4.1"]
    assert m0["cells"]["vf"]["strict_acc"] == 0.5
    assert m0["cells"]["vf_trace"]["strict_acc"] == 0.75
    assert m0["cells"]["rf_order_only"]["strict_acc"] == 0.75
    assert m0["cells"]["rf"]["strict_acc"] == 1.0
    assert abs(m0["decomposition_pp"]["order_effect_pp"] - 25.0) < 1e-9
    assert abs(m0["decomposition_pp"]["instruction_effect_pp"] - 25.0) < 1e-9
    assert abs(m0["decomposition_pp"]["interaction_pp"]) < 1e-9
    assert m0["cells"]["vf_trace"]["key_order_violations"] == 0
    assert all("p_value_holm" in c for c in m0["comparisons"].values())


