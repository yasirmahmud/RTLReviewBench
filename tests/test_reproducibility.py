"""Tests verifying reproducibility and SHA-256 hash integrity of the frozen pilot."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

CONFIG_PATHS = [
    Path("configs/pilot.yaml"),
    Path("configs/clarified_50.yaml"),
    Path("configs/v2_multisource.yaml"),
]


@pytest.fixture(params=CONFIG_PATHS, ids=["pilot", "clarified_50", "v2_multisource"])
def gold_and_public(request):
    cfg_path = request.param
    if not cfg_path.exists():
        pytest.skip(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)
    gold_path = Path(config["paths"]["pilot_gold"])
    pub_path = Path(config["paths"]["pilot_public"])
    if not gold_path.exists() or not pub_path.exists():
        pytest.skip(f"Manifests not found for {cfg_path}.")

    with open(gold_path, "r", encoding="utf-8") as fp:
        gold = json.load(fp)
    pub = []
    with open(pub_path, "r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                pub.append(json.loads(line))

    return gold, pub


def test_sha256_hash_integrity(gold_and_public):
    """Verify that candidate and spec SHA-256 hashes recorded in gold match public text."""
    gold_list, pub_list = gold_and_public
    pub_map = {c["case_id"]: c for c in pub_list}

    for g in gold_list:
        cid = g["case_id"]
        assert cid in pub_map
        p = pub_map[cid]

        # Spec hash
        computed_spec_sha = hashlib.sha256(p["specification"].encode("utf-8")).hexdigest()
        assert computed_spec_sha == g["spec_sha256"], f"Spec SHA mismatch in {cid}"

        # Candidate RTL hash
        computed_rtl_sha = hashlib.sha256(p["candidate_rtl"].encode("utf-8")).hexdigest()
        assert computed_rtl_sha == g["candidate_sha256"], f"Candidate RTL SHA mismatch in {cid}"


def test_deterministic_case_ids(gold_and_public):
    """Verify case IDs are sequentially ordered C001.. or C01.."""
    gold_list, pub_list = gold_and_public
    total_cases = len(gold_list)
    case_width = 3 if total_cases >= 100 else 2
    expected_cids = [f"C{i:0{case_width}d}" for i in range(1, total_cases + 1)]
    assert [g["case_id"] for g in gold_list] == expected_cids
    assert [p["case_id"] for p in pub_list] == expected_cids

