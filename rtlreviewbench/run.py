"""Execution pipeline: evaluates public cases with a selected model provider."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from rtlreviewbench.prompt import build_review_prompt
from rtlreviewbench.providers import get_provider
from rtlreviewbench.score import score_model_results


SPLIT_MANIFESTS: Dict[str, Dict[str, Any]] = {
    "core_clean": {
        "public": Path("manifests/v2_clean114_public.jsonl"),
        "gold": Path("manifests/v2_clean114_gold.json"),
        "expected_cases": 228,
    },
    "core_full": {
        "public": Path("manifests/v2_public.jsonl"),
        "gold": Path("manifests/v2_gold.json"),
        "expected_cases": 240,
    },
    "synmut50": {
        "public": Path("manifests/syn_public.jsonl"),
        "gold": Path("manifests/syn_gold.json"),
        "expected_cases": 100,
    },
    "hard": {
        "public": Path("manifests/hard_public.jsonl"),
        "gold": Path("manifests/hard_gold.json"),
        "expected_cases": 44,
    },
}


def evaluate_public_cases(
    model_name: str,
    config: Dict[str, Any],
    public_manifest_path: Optional[str | Path] = None,
    results_dir: Optional[str | Path] = None,
    max_workers: int = 2,
    resume: bool = True,
    prompt_mode: str = "verdict_first",
) -> List[Dict[str, Any]]:
    """
    Evaluate all public cases using the specified model provider.
    Supports incremental checkpointing & safe concurrency while maintaining strict info control.
    STRICT INFORMATION CONTROL:
    This function reads ONLY the public JSONL manifest and has NO access to gold labels.
    """
    if public_manifest_path is None:
        public_manifest_path = config["paths"]["pilot_public"]
    if results_dir is None:
        results_dir = config["paths"]["results_dir"]

    pub_path = Path(public_manifest_path)
    if not pub_path.exists():
        raise FileNotFoundError(
            f"Public manifest not found at {pub_path}. Run build_pilot.py first."
        )

    public_cases: List[Dict[str, Any]] = []
    with open(pub_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                public_cases.append(json.loads(line))

    expected_count = int(config.get("total_cases", len(public_cases)))
    if len(public_cases) != expected_count:
        raise ValueError(
            f"Public manifest must contain exactly {expected_count} cases, found {len(public_cases)}"
        )

    public_ids = {c["case_id"] for c in public_cases}
    provider = get_provider(model_name, config)
    out_dir = Path(results_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "raw_responses.jsonl"

    completed_map: Dict[str, Dict[str, Any]] = {}
    if resume and out_file.exists():
        with open(out_file, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    cid = rec.get("case_id")
                    rec_mode = rec.get("prompt_mode", "verdict_first")
                    if cid in public_ids and rec_mode == prompt_mode:
                        completed_map[cid] = rec
                except json.JSONDecodeError:
                    continue

    remaining_cases = [c for c in public_cases if c["case_id"] not in completed_map]
    if completed_map:
        print(
            f"    Resuming '{model_name}': {len(completed_map)}/{len(public_cases)} already completed, "
            f"{len(remaining_cases)} remaining.",
            flush=True,
        )

    lock = threading.Lock()

    def eval_one_case(case: Dict[str, Any]) -> Dict[str, Any]:
        prompt = build_review_prompt(
            specification=case["specification"],
            candidate_rtl=case["candidate_rtl"],
            prompt_mode=prompt_mode,
        )
        call_res = provider.call(prompt["system"], prompt["user"])
        return {
            "case_id": case["case_id"],
            "pair_id": case["pair_id"],
            "dataset_source": case.get("dataset_source", "ChipBench"),
            "model_id": model_name,
            "prompt_mode": prompt_mode,
            "parsed": call_res["parsed"],
            "raw_response": call_res["raw_response"],
            "latency_ms": call_res.get("latency_ms", 0.0),
            "prompt_tokens": call_res.get("prompt_tokens", 0),
            "completion_tokens": call_res.get("completion_tokens", 0),
            "total_tokens": call_res.get("total_tokens", 0),
            "smoke_test_only": call_res.get("smoke_test_only", False),
        }

    # Overwrite file with clean baseline of already valid completed items first
    with open(out_file, "w", encoding="utf-8") as fp:
        for cid in [c["case_id"] for c in public_cases]:
            if cid in completed_map:
                fp.write(json.dumps(completed_map[cid]) + "\n")

    if remaining_cases:
        workers = 1 if model_name == "mock" else max_workers
        done_count = len(completed_map)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(eval_one_case, case): case for case in remaining_cases}
            for fut in as_completed(futures):
                rec = fut.result()
                with lock:
                    completed_map[rec["case_id"]] = rec
                    done_count += 1
                    with open(out_file, "a", encoding="utf-8") as fp:
                        fp.write(json.dumps(rec) + "\n")
                    if done_count % 10 == 0 or done_count == len(public_cases):
                        print(f"    [{model_name}] Progress: {done_count}/{len(public_cases)} cases complete", flush=True)

    # Sort final results cleanly to match public_cases order
    ordered_results = [completed_map[c["case_id"]] for c in public_cases]
    with open(out_file, "w", encoding="utf-8") as fp:
        for r in ordered_results:
            fp.write(json.dumps(r) + "\n")

    return ordered_results


def _resolve_repo_path(repo_root: Path, rel_or_abs: Path) -> Path:
    return rel_or_abs if rel_or_abs.is_absolute() else (repo_root / rel_or_abs)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import yaml

    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(
        prog="python -m rtlreviewbench.run",
        description="Evaluate an LLM on a frozen RTLReviewBench split and compute safety metrics.",
    )
    ap.add_argument("--model", required=True, help="Model ID (e.g. mock, gpt-4.1-mini, claude-haiku-5.5).")
    ap.add_argument(
        "--split",
        default="core_clean",
        choices=sorted(SPLIT_MANIFESTS.keys()),
        help="Benchmark split to evaluate (default: core_clean).",
    )
    ap.add_argument(
        "--prompt-mode",
        default="verdict_first",
        choices=["verdict_first", "reason_first"],
        help="Frozen prompt schema/condition (default: verdict_first).",
    )
    ap.add_argument("--config", default="configs/v2_multisource.yaml", help="Config YAML path.")
    ap.add_argument("--results-dir", default=None, help="Output directory override (default: results_cli/<split>/<prompt_mode>).")
    ap.add_argument("--workers", type=int, default=3, help="Concurrent API workers (default: 3).")
    ap.add_argument("--no-resume", action="store_true", help="Ignore existing raw_responses.jsonl checkpoint.")
    ap.add_argument("--no-score", action="store_true", help="Skip scoring against gold manifest after inference.")
    args = ap.parse_args(argv)

    cfg_path = _resolve_repo_path(repo_root, Path(args.config))
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {"models": {}}

    split_info = SPLIT_MANIFESTS[args.split]
    pub_path = _resolve_repo_path(repo_root, split_info["public"])
    gold_path = _resolve_repo_path(repo_root, split_info["gold"])

    run_cfg = dict(config)
    run_cfg["total_cases"] = split_info["expected_cases"]
    out_dir = (
        _resolve_repo_path(repo_root, Path(args.results_dir))
        if args.results_dir
        else repo_root / "results_cli" / args.split / args.prompt_mode
    )

    print(
        f"==> Evaluating model='{args.model}' split='{args.split}' "
        f"({split_info['expected_cases']} cases) prompt_mode='{args.prompt_mode}'...",
        flush=True,
    )
    evaluate_public_cases(
        model_name=args.model,
        config=run_cfg,
        public_manifest_path=pub_path,
        results_dir=out_dir,
        max_workers=args.workers,
        resume=not args.no_resume,
        prompt_mode=args.prompt_mode,
    )
    raw_file = out_dir / args.model / "raw_responses.jsonl"
    print(f"    Saved raw responses -> {raw_file}", flush=True)

    if not args.no_score and gold_path.exists():
        summary = score_model_results(raw_file, gold_path)
        pm = summary["primary_metrics"]
        pb = summary["pair_breakdown"]
        cm = summary["confusion_matrix"]
        score_file = out_dir / args.model / "summary.json"
        score_file.write_text(json.dumps({k: v for k, v in summary.items() if k != "per_case_results"}, indent=2) + "\n", encoding="utf-8")
        print(
            f"[SCORE | {args.split} | {args.model}] "
            f"Acc={pm['strict_accuracy']*100:.1f}% ({pm['strict_accuracy_fraction']}) | "
            f"FAR={pm['false_acceptance_rate']*100:.1f}% ({pm['false_acceptance_fraction']}) | "
            f"FRR={pm['false_rejection_rate']*100:.1f}% ({pm['false_rejection_fraction']}) | "
            f"PairBoth={pb['both_correct_fraction']} | Cov={pm['coverage']*100:.1f}% "
            f"(TP={cm['TP']} TN={cm['TN']} FP={cm['FP']} FN={cm['FN']})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
