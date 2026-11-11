"""Resumable, checkpointed experiment runner for RTLReviewBench extensions:
- core_v2      : Multi-tier model evaluation on manifests/v2_public.jsonl -> results_v2/<model>/raw_responses.jsonl
- reason_first : Causal Schema-Order Intervention (reason-before-verdict JSON) -> results_ext/reason_first/<model>/raw_responses.jsonl
- syn_sem85    : Synthetic Semantic Mutation contrast track -> results_ext/syn_sem85/<model>/raw_responses.jsonl
- pairwise     : Bidirectional contrastive pairwise discrimination -> results_ext/pairwise/<model>/raw_responses.jsonl
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rtlreviewbench.prompt import (
    REASON_FIRST_SYSTEM_PROMPT,
    REASON_FIRST_USER_TEMPLATE,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    _extract_json_object,
    build_review_prompt,
    extract_json_key_order,
    parse_model_response,
)

# --- Task B: 2x2 Reason-First Ablation Prompts (do not alter rtlreviewbench/prompt.py baselines) ---

VF_TRACE_SYSTEM_PROMPT = """You are reviewing RTL for functional conformance to the given specification.
Judge only the supplied RTL implementation. Do not write or repair RTL.
If evidence is insufficient, you may abstain. Return only valid JSON matching
{"verdict":"correct|buggy|abstain","confidence":0.0,"reason":"step-by-step functional check of signals, operators, and cycle timing against the spec"}.
In "reason", trace the RTL behavior step by step against the specification.
Use one of the three literal verdict strings. Confidence must be between 0 and 1.
Do not assume that syntactically valid code is functionally correct."""

VF_TRACE_USER_TEMPLATE = """SPECIFICATION (verbatim from the existing dataset):
{specification}

RTL IMPLEMENTATION (candidate under review):
```verilog
{candidate_rtl}
```

Is this RTL functionally consistent with the specification?
Output JSON only, with verdict, confidence, and a step-by-step reason."""

RF_ORDER_ONLY_SYSTEM_PROMPT = """You are reviewing RTL for functional conformance to the given specification.
Judge only the supplied RTL implementation. Do not write or repair RTL.
If evidence is insufficient, you may abstain. Return only valid JSON matching
{"reason":"short explanation","verdict":"correct|buggy|abstain","confidence":0.0}.
Use one of the three literal verdict strings. Confidence must be between 0 and 1.
Do not assume that syntactically valid code is functionally correct."""

RF_ORDER_ONLY_USER_TEMPLATE = """SPECIFICATION (verbatim from the existing dataset):
{specification}

RTL IMPLEMENTATION (candidate under review):
```verilog
{candidate_rtl}
```

Is this RTL functionally consistent with the specification?
Output JSON only, with reason, verdict, and confidence, in that key order."""

REQUESTED_KEY_ORDER: Dict[str, List[str]] = {
    "core_v2": ["verdict", "confidence", "reason"],
    "vf_trace": ["verdict", "confidence", "reason"],
    "rf_order_only": ["reason", "verdict", "confidence"],
    "reason_first": ["reason", "verdict", "confidence"],
}


def build_extension_review_prompt(specification: str, candidate_rtl: str, task_name: str) -> Dict[str, str]:
    spec = specification.strip()
    rtl = candidate_rtl.strip()
    if task_name in ("core_v2", "syn_sem85", "self_consistency"):
        return build_review_prompt(spec, rtl, prompt_mode="verdict_first")
    if task_name == "reason_first":
        return build_review_prompt(spec, rtl, prompt_mode="reason_first")
    if task_name == "vf_trace":
        return {
            "system": VF_TRACE_SYSTEM_PROMPT,
            "user": VF_TRACE_USER_TEMPLATE.format(specification=spec, candidate_rtl=rtl),
        }
    if task_name == "rf_order_only":
        return {
            "system": RF_ORDER_ONLY_SYSTEM_PROMPT,
            "user": RF_ORDER_ONLY_USER_TEMPLATE.format(specification=spec, candidate_rtl=rtl),
        }
    raise ValueError(f"Unknown review task prompt: {task_name}")


PAIRWISE_SYSTEM_PROMPT = """You are reviewing two candidate RTL implementations (Candidate A and Candidate B) for functional conformance to the given specification.
Judge only the supplied RTL implementations. Do not write or repair RTL.
Exactly one implementation may conform cleanly, or both may fail/tie if evidence is ambiguous.
Return only valid JSON matching {"choice":"A|B|tie|neither","confidence":0.0,"reason":"short explanation"}.
Use one of the four literal choice strings ("A", "B", "tie", "neither"). Confidence must be between 0 and 1."""

PAIRWISE_USER_TEMPLATE = """SPECIFICATION (verbatim from the existing dataset):
{specification}

CANDIDATE A:
```verilog
{candidate_a_rtl}
```

CANDIDATE B:
```verilog
{candidate_b_rtl}
```

Which candidate RTL is functionally consistent with the specification?
Output JSON only, with choice ("A", "B", "tie", or "neither"), confidence, and a short reason."""


def parse_pairwise_response(raw_text: str) -> Dict[str, Any]:
    if not raw_text or not raw_text.strip():
        return {"choice": "invalid", "confidence": 0.0, "reason": "EMPTY_RESPONSE", "valid_json": False, "strict_json": False}
    cleaned = raw_text.strip()
    was_fenced = False
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        was_fenced = True
    data, direct_parse, err_reason = _extract_json_object(cleaned, required_key="choice")
    strict_json = direct_parse and not was_fenced
    if data is None:
        return {"choice": "invalid", "confidence": 0.0, "reason": err_reason or "JSON_DECODE_ERROR", "valid_json": False, "strict_json": False}
    raw_choice = str(data.get("choice", "")).strip()
    norm = raw_choice.upper() if raw_choice.lower() in {"a", "b"} else raw_choice.lower()
    if norm not in {"A", "B", "tie", "neither"}:
        return {"choice": "invalid", "confidence": 0.0, "reason": f"INVALID_CHOICE: '{raw_choice}'", "valid_json": True, "strict_json": strict_json}
    try:
        conf = float(data.get("confidence", 0.0))
        if not (0.0 <= conf <= 1.0):
            return {"choice": "invalid", "confidence": conf, "reason": f"CONF_OUT_OF_BOUNDS: {conf}", "valid_json": True, "strict_json": strict_json}
    except (ValueError, TypeError):
        return {"choice": "invalid", "confidence": 0.0, "reason": "CONF_NOT_FLOAT", "valid_json": True, "strict_json": strict_json}
    reason = data.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)
    return {"choice": norm, "confidence": conf, "reason": reason.strip(), "valid_json": True, "strict_json": strict_json}


class GlobalRateLimiter:
    """Thread-safe adaptive RPM pacer preventing burst 429 storms across parallel workers."""

    def __init__(self, rpm: float = 60.0) -> None:
        self._lock = threading.Lock()
        self._min_interval = 60.0 / max(rpm, 1.0)
        self._next_allowed = 0.0
        self._cooldown_until = 0.0
        self._last_backoff = 0.0

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                target = max(self._next_allowed, self._cooldown_until, now)
                delay = target - now
                if delay <= 0.0:
                    self._next_allowed = now + self._min_interval
                    return
                if target == self._next_allowed and now >= self._cooldown_until:
                    self._next_allowed = target + self._min_interval
                    reserve_wait = delay
                else:
                    reserve_wait = None
            if reserve_wait is not None:
                time.sleep(reserve_wait)
                return
            time.sleep(min(delay, 2.0))

    def backoff(self, wait_sec: float) -> None:
        with self._lock:
            now = time.monotonic()
            if now - self._last_backoff > 5.0:
                self._min_interval = min(self._min_interval * 1.15, 2.5)
                self._last_backoff = now
            self._cooldown_until = max(self._cooldown_until, now + wait_sec)


def call_chat_completion(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    base_url: str,
    limiter: GlobalRateLimiter | None = None,
    max_tokens: int = 16384,
    timeout: int = 180,
    max_attempts: int = 12,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    last_err = None
    start = time.perf_counter()
    for attempt in range(1, max_attempts + 1):
        if limiter is not None:
            limiter.acquire()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout, allow_redirects=False)
        except requests.RequestException as exc:
            last_err = exc
            time.sleep(min(20.0, 1.5 * attempt))
            continue

        if resp.status_code == 200:
            latency_ms = (time.perf_counter() - start) * 1000.0
            data = resp.json()
            msg = data["choices"][0]["message"]
            raw_text = msg.get("content") or ""
            rc = msg.get("reasoning_content")
            usage = data.get("usage", {})
            details = usage.get("completion_tokens_details") or {}
            return {
                "raw_response": raw_text,
                "reasoning_content": rc,
                "served_model": data.get("model"),
                "system_fingerprint": data.get("system_fingerprint"),
                "latency_ms": latency_ms,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "reasoning_tokens": details.get("reasoning_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        if resp.status_code == 429 and attempt < max_attempts:
            retry_after = resp.headers.get("Retry-After")
            wait_sec = float(retry_after) if retry_after and retry_after.isdigit() else min(60.0, 15.0 * attempt)
            print(f"    [{model_id} 429] Sleep {wait_sec:.0f}s (Retry-After={retry_after}, attempt {attempt}/{max_attempts}): {resp.text[:200]}", flush=True)
            if limiter is not None:
                limiter.backoff(wait_sec)
            last_err = RuntimeError(f"HTTP 429: {resp.text[:200]}")
            time.sleep(wait_sec)
            continue

        if resp.status_code in (500, 502, 503, 504, 408) and attempt < max_attempts:
            last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:120]}")
            time.sleep(min(30.0, 2.5 * attempt))
            continue

        if resp.status_code == 400 and "potentially violating" in resp.text.lower() and attempt < min(4, max_attempts):
            print(f"    [{model_id} 400-filter] Transient retry {attempt}/4: {resp.text[:140]}", flush=True)
            last_err = RuntimeError(f"HTTP 400 filter: {resp.text[:160]}")
            time.sleep(3.0 * attempt)
            continue

        if resp.status_code == 400 and "potentially violating" in resp.text.lower():
            print(f"    [{model_id} 400-filter] Persistent filter block; recording blocked/invalid response.", flush=True)
            return {
                "raw_response": "",
                "reasoning_content": None,
                "latency_ms": (time.perf_counter() - start) * 1000.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 0,
                "api_error": f"HTTP_400_CONTENT_FILTER: {resp.text[:220]}",
            }

        raise RuntimeError(f"HTTP {resp.status_code} on {model_id}: {resp.text[:250]}")

    raise RuntimeError(f"Failed after {max_attempts} attempts on {model_id}: {last_err}")


def run_task(
    task_name: str,
    model_id: str,
    workers: int = 3,
    limiter: GlobalRateLimiter | None = None,
    temperature: float | None = None,
    num_samples: int = 5,
    limit: int | None = None,
    out_dir: Path | str | None = None,
) -> None:
    load_dotenv(ROOT / ".env", override=False)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    base_url = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")

    if task_name == "core_v2":
        manifest_path = ROOT / "manifests" / "v2_public.jsonl"
        default_out = ROOT / "results_v2" / model_id / "raw_responses.jsonl"
        id_field = "case_id"
    elif task_name == "reason_first":
        manifest_path = ROOT / "manifests" / "v2_public.jsonl"
        default_out = ROOT / "results_ext" / "reason_first" / model_id / "raw_responses.jsonl"
        id_field = "case_id"
    elif task_name in ("vf_trace", "rf_order_only"):
        manifest_path = ROOT / "manifests" / "v2_public.jsonl"
        default_out = ROOT / "results_ext" / "rf_ablation" / task_name / model_id / "raw_responses.jsonl"
        id_field = "case_id"
    elif task_name == "syn_sem85":
        manifest_path = ROOT / "manifests" / "syn_public.jsonl"
        default_out = ROOT / "results_ext" / "syn_sem85" / model_id / "raw_responses.jsonl"
        id_field = "case_id"
    elif task_name == "pairwise":
        manifest_path = ROOT / "manifests" / "pairwise_public.jsonl"
        default_out = ROOT / "results_ext" / "pairwise" / model_id / "raw_responses.jsonl"
        id_field = "query_id"
    elif task_name == "self_consistency":
        manifest_path = ROOT / "manifests" / "v2_public.jsonl"
        default_out = ROOT / "results_ext" / "self_consistency" / model_id / "raw_responses.jsonl"
        id_field = "sample_id"
    else:
        raise ValueError(f"Unknown task: {task_name}")

    out_file = (Path(out_dir) / model_id / "raw_responses.jsonl") if out_dir is not None else default_out
    eff_temp = (0.7 if task_name == "self_consistency" else 0.0) if temperature is None else float(temperature)

    base_rows = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit is not None:
        base_rows = base_rows[:limit]
    if task_name == "self_consistency":
        rows = []
        for s_idx in range(num_samples):
            for r in base_rows:
                item = dict(r)
                item["sample_idx"] = s_idx
                item["sample_id"] = f"{r['case_id']}__s{s_idx}"
                rows.append(item)
    else:
        rows = base_rows
    valid_ids = {r[id_field] for r in rows}
    out_file.parent.mkdir(parents=True, exist_ok=True)

    completed: Dict[str, Dict[str, Any]] = {}
    reparsed_any = False
    if out_file.exists():
        for line in out_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rid = rec.get(id_field)
                if rid not in valid_ids:
                    continue
                raw_resp = rec.get("raw_response")
                if raw_resp is not None and str(raw_resp).strip():
                    new_parsed = parse_pairwise_response(raw_resp) if task_name == "pairwise" else parse_model_response(raw_resp)
                    if new_parsed != rec.get("parsed"):
                        rec["parsed"] = new_parsed
                        reparsed_any = True
                completed[rid] = rec
            except json.JSONDecodeError:
                continue

    remaining = [r for r in rows if r[id_field] not in completed]
    print(f"[{task_name} | {model_id}] Loaded {len(completed)}/{len(rows)} completed; running {len(remaining)} (workers={workers}, reparsed={reparsed_any})", flush=True)
    if not remaining:
        if reparsed_any:
            ordered = [completed[r[id_field]] for r in rows]
            with open(out_file, "w", encoding="utf-8") as fp:
                for rec in ordered:
                    fp.write(json.dumps(rec) + "\n")
        return

    # Rewrite clean checkpoint baseline
    with open(out_file, "w", encoding="utf-8") as fp:
        for r in rows:
            if r[id_field] in completed:
                fp.write(json.dumps(completed[r[id_field]]) + "\n")

    lock = threading.Lock()
    done_cnt = len(completed)
    req_order = REQUESTED_KEY_ORDER.get(task_name)

    def _run_one(item: Dict[str, Any]) -> Dict[str, Any]:
        if task_name == "pairwise":
            sys_p = PAIRWISE_SYSTEM_PROMPT
            usr_p = PAIRWISE_USER_TEMPLATE.format(
                specification=item["specification"].strip(),
                candidate_a_rtl=item["candidate_a_rtl"].strip(),
                candidate_b_rtl=item["candidate_b_rtl"].strip(),
            )
        else:
            pr = build_extension_review_prompt(item["specification"], item["candidate_rtl"], task_name)
            sys_p, usr_p = pr["system"], pr["user"]

        res = call_chat_completion(model_id, sys_p, usr_p, api_key, base_url, limiter=limiter, temperature=eff_temp)
        if task_name == "pairwise":
            parsed = parse_pairwise_response(res["raw_response"])
            rec = {
                "query_id": item["query_id"],
                "pair_id": item["pair_id"],
                "direction": item["direction"],
                "dataset_source": item.get("dataset_source", ""),
                "model_id": model_id,
                "served_model": res.get("served_model"),
                "parsed": parsed,
                "raw_response": res["raw_response"],
                "reasoning_content": res["reasoning_content"],
                "latency_ms": res["latency_ms"],
                "prompt_tokens": res["prompt_tokens"],
                "completion_tokens": res["completion_tokens"],
                "reasoning_tokens": res["reasoning_tokens"],
                "total_tokens": res["total_tokens"],
                "smoke_test_only": limit is not None,
            }
        else:
            parsed = parse_model_response(res["raw_response"])
            emitted_order = extract_json_key_order(res["raw_response"], required_key="verdict")
            rec = {
                "case_id": item["case_id"],
                "pair_id": item["pair_id"],
                "dataset_source": item.get("dataset_source", ""),
                "model_id": model_id,
                "prompt_mode": task_name,
                "served_model": res.get("served_model"),
                "temperature": eff_temp,
                "parsed": parsed,
                "strict_json": parsed.get("strict_json", False),
                "requested_key_order": req_order,
                "emitted_key_order": emitted_order,
                "key_order_match": (emitted_order == req_order) if (emitted_order is not None and req_order is not None) else False,
                "raw_response": res["raw_response"],
                "reasoning_content": res["reasoning_content"],
                "latency_ms": res["latency_ms"],
                "prompt_tokens": res["prompt_tokens"],
                "completion_tokens": res["completion_tokens"],
                "reasoning_tokens": res["reasoning_tokens"],
                "total_tokens": res["total_tokens"],
                "smoke_test_only": limit is not None,
            }
            if task_name == "self_consistency":
                rec["sample_idx"] = item["sample_idx"]
                rec["sample_id"] = item["sample_id"]
        return rec

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_run_one, item): item for item in remaining}
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                completed[rec[id_field]] = rec
                done_cnt += 1
                with open(out_file, "a", encoding="utf-8") as fp:
                    fp.write(json.dumps(rec) + "\n")
                if done_cnt % 40 == 0 or done_cnt == len(rows):
                    print(f"    [{task_name} | {model_id}] {done_cnt}/{len(rows)} complete", flush=True)

    ordered = [completed[r[id_field]] for r in rows]
    with open(out_file, "w", encoding="utf-8") as fp:
        for rec in ordered:
            fp.write(json.dumps(rec) + "\n")
    print(f"[{task_name} | {model_id}] DONE ({len(ordered)} records) -> {out_file}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task",
        required=True,
        choices=[
            "core_v2",
            "reason_first",
            "vf_trace",
            "rf_order_only",
            "syn_sem85",
            "pairwise",
            "self_consistency",
        ],
    )
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--parallel-models", type=int, default=1)
    ap.add_argument("--rpm", type=float, default=60.0)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    shared_limiter = GlobalRateLimiter(rpm=args.rpm)

    if args.parallel_models > 1 and len(args.models) > 1:
        with ThreadPoolExecutor(max_workers=args.parallel_models) as mpool:
            list(
                mpool.map(
                    lambda m: run_task(
                        args.task,
                        m,
                        workers=args.workers,
                        limiter=shared_limiter,
                        temperature=args.temperature,
                        num_samples=args.num_samples,
                        limit=args.limit,
                        out_dir=args.out_dir,
                    ),
                    args.models,
                )
            )
    else:
        for m in args.models:
            run_task(
                args.task,
                m,
                workers=args.workers,
                limiter=shared_limiter,
                temperature=args.temperature,
                num_samples=args.num_samples,
                limit=args.limit,
                out_dir=args.out_dir,
            )
