"""Tests verifying that public prompts and manifests do not leak ground-truth labels."""

import json
import re
from pathlib import Path

import pytest
import yaml

CONFIG_PATHS = [
    Path("configs/pilot.yaml"),
    Path("configs/clarified_50.yaml"),
    Path("configs/v2_multisource.yaml"),
]


@pytest.fixture(params=CONFIG_PATHS, ids=["pilot", "clarified_50", "v2_multisource"])
def public_cases(request):
    cfg_path = request.param
    if not cfg_path.exists():
        pytest.skip(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)
    pub_path = Path(config["paths"]["pilot_public"])
    if not pub_path.exists():
        pytest.skip(f"Public manifest not found at {pub_path}.")
    cases = []
    with open(pub_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def test_public_manifest_keys(public_cases):
    """Ensure forbidden truth fields are absent from public cases."""
    forbidden_keys = {
        "candidate_role",
        "gold_label",
        "label",
        "is_buggy",
        "ground_truth",
        "verification",
        "faulty_test_result",
        "fixed_test_result",
        "mismatches",
    }
    for case in public_cases:
        for k in forbidden_keys:
            assert k not in case, f"Forbidden key '{k}' found in public case {case.get('case_id')}"


def test_presentation_filename_generic(public_cases):
    """Ensure candidate presentation filename is generic, e.g. design.sv."""
    for case in public_cases:
        filename = case.get("filename", "")
        assert filename == "design.sv"
        assert "ref" not in filename.lower()
        assert "bug" not in filename.lower()


def test_specification_text_cleanliness(public_cases):
    """Ensure specification text does not reveal that a bug exists or instruct repair."""
    forbidden_phrases = [
        r"code below.*has bug",
        r"\bthis\s+module\s+has\s+(?:a\s+)?bug\b",
        r"\bfixes\s+the\s+bug\b",
        r"please fix",
        r"the bug is",
        r"\bfaulty\b",
        r"\bbuggy\b",
        r"\bRefModule\b",
        r"^\s*module\s+TopModule\b",
    ]
    for case in public_cases:
        spec = case.get("specification", "")
        for pat in forbidden_phrases:
            assert not re.search(pat, spec, re.IGNORECASE | re.MULTILINE), (
                f"Specification in {case['case_id']} contains forbidden text matching '{pat}'"
            )


def test_candidate_rtl_cleanliness(public_cases):
    """Ensure candidate RTL does not contain tell-tale labels, Prob IDs, or diagnostic comments."""
    comment_leak_pat = re.compile(
        r"\b(bugs?|buggy|broken|golden|wrong|faulty?|mistake|accidentally|uhh|repair|injected)\b",
        re.IGNORECASE,
    )
    for case in public_cases:
        rtl = case.get("candidate_rtl", "")
        assert "RefModule" not in rtl, f"Candidate RTL in {case['case_id']} contains 'RefModule'"
        assert not re.search(r"\bProb\d+_\w+", rtl), (
            f"Candidate RTL in {case['case_id']} leaks upstream Prob ID"
        )
        assert re.search(r"^\s*module\s+TopModule\b", rtl, re.MULTILINE), (
            f"Candidate RTL in {case['case_id']} should declare TopModule"
        )
        comments = "\n".join(
            re.findall(r"//[^\n]*", rtl) + re.findall(r"/\*.*?\*/", rtl, re.DOTALL)
        )
        assert not comment_leak_pat.search(comments), (
            f"Candidate RTL comments in {case['case_id']} contain forbidden diagnostic hint"
        )

