"""Prompt formatting, schema enforcement, and model response parsing."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

SYSTEM_PROMPT = """You are reviewing RTL for functional conformance to the given specification.
Judge only the supplied RTL implementation. Do not write or repair RTL.
If evidence is insufficient, you may abstain. Return only valid JSON matching
{"verdict":"correct|buggy|abstain","confidence":0.0,"reason":"short explanation"}.
Use one of the three literal verdict strings. Confidence must be between 0 and 1.
Do not assume that syntactically valid code is functionally correct."""

USER_TEMPLATE = """SPECIFICATION (verbatim from the existing dataset):
{specification}

RTL IMPLEMENTATION (candidate under review):
```verilog
{candidate_rtl}
```

Is this RTL functionally consistent with the specification?
Output JSON only, with verdict, confidence, and a short reason."""

REASON_FIRST_SYSTEM_PROMPT = """You are reviewing RTL for functional conformance to the given specification.
Judge only the supplied RTL implementation. Do not write or repair RTL.
If evidence is insufficient, you may abstain. Return only valid JSON matching
{"reason":"step-by-step functional check of signals, operators, and cycle timing against the spec","verdict":"correct|buggy|abstain","confidence":0.0}.
First trace the RTL behavior inside "reason", then commit to one of the three literal verdict strings.
Confidence must be between 0 and 1. Do not assume that syntactically valid code is functionally correct."""

REASON_FIRST_USER_TEMPLATE = """SPECIFICATION (verbatim from the existing dataset):
{specification}

RTL IMPLEMENTATION (candidate under review):
```verilog
{candidate_rtl}
```

Is this RTL functionally consistent with the specification?
Output JSON only, reasoning first in "reason", then giving "verdict" and "confidence"."""

VALID_VERDICTS = {"correct", "buggy", "abstain"}


def build_review_prompt(
    specification: str,
    candidate_rtl: str,
    prompt_mode: str = "verdict_first",
) -> Dict[str, str]:
    """Construct frozen system and user messages for a candidate evaluation."""
    spec = specification.strip()
    rtl = candidate_rtl.strip()
    if prompt_mode == "verdict_first":
        return {
            "system": SYSTEM_PROMPT,
            "user": USER_TEMPLATE.format(specification=spec, candidate_rtl=rtl),
        }
    if prompt_mode == "reason_first":
        return {
            "system": REASON_FIRST_SYSTEM_PROMPT,
            "user": REASON_FIRST_USER_TEMPLATE.format(specification=spec, candidate_rtl=rtl),
        }
    raise ValueError(f"Unknown prompt_mode: {prompt_mode}")


def _extract_json_object(cleaned: str, required_key: str = "verdict") -> Tuple[Optional[Dict[str, Any]], bool, Optional[str]]:
    """
    Parse JSON from model output.
    Returns (parsed_dict_or_None, strict_json_only, error_reason_or_None).
    Uses JSONDecoder.raw_decode scanning when prose (which may contain Verilog `{...}`
    concatenations) precedes or follows the JSON payload.
    """
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data, True, None
        return None, False, "TOP_LEVEL_NOT_OBJECT"
    except json.JSONDecodeError as exc:
        first_err = str(exc)

    decoder = json.JSONDecoder()
    candidates: list[Dict[str, Any]] = []
    for idx, ch in enumerate(cleaned):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned, idx)
            if isinstance(obj, dict):
                candidates.append(obj)
        except json.JSONDecodeError:
            continue

    if not candidates:
        return None, False, f"JSON_DECODE_ERROR: {first_err}"

    # Prefer the final JSON object carrying the target schema key (e.g., if CoT precedes JSON)
    matching = [c for c in candidates if required_key in c]
    chosen = matching[-1] if matching else candidates[-1]
    return chosen, False, None


def extract_json_key_order(raw_text: str, required_key: str = "verdict") -> Optional[List[str]]:
    """Return top-level JSON key sequence in exact emitted order, or None if invalid JSON."""
    if not raw_text or not raw_text.strip():
        return None
    cleaned = raw_text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    data, _, _ = _extract_json_object(cleaned, required_key=required_key)
    if data is None:
        return None
    return list(data.keys())


def parse_model_response(raw_text: str) -> Dict[str, Any]:
    """
    Parse and strictly validate model output against the JSON contract.
    Returns parsed dictionary with valid keys, or marks verdict as 'invalid'.
    """
    if not raw_text or not raw_text.strip():
        return {
            "verdict": "invalid",
            "confidence": 0.0,
            "reason": "EMPTY_RESPONSE",
            "valid_json": False,
            "strict_json": False,
            "raw": raw_text,
        }

    raw_stripped = raw_text.strip()
    cleaned = raw_stripped
    was_fenced = False
    # Strip outer markdown code blocks if the model wrapped its JSON
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()
        was_fenced = True

    data, direct_parse, err_reason = _extract_json_object(cleaned, required_key="verdict")
    strict_json = direct_parse and not was_fenced
    if data is None:
        return {
            "verdict": "invalid",
            "confidence": 0.0,
            "reason": err_reason or "JSON_DECODE_ERROR",
            "valid_json": False,
            "strict_json": False,
            "raw": raw_text,
        }

    raw_verdict = str(data.get("verdict", "")).strip().lower()
    if raw_verdict not in VALID_VERDICTS:
        return {
            "verdict": "invalid",
            "confidence": 0.0,
            "reason": f"INVALID_VERDICT_VALUE: '{raw_verdict}'",
            "valid_json": True,
            "strict_json": strict_json,
            "raw": raw_text,
        }

    try:
        raw_conf = float(data.get("confidence", 0.0))
        if not (0.0 <= raw_conf <= 1.0):
            return {
                "verdict": "invalid",
                "confidence": raw_conf,
                "reason": f"CONFIDENCE_OUT_OF_BOUNDS: {raw_conf}",
                "valid_json": True,
                "strict_json": strict_json,
                "raw": raw_text,
            }
    except (ValueError, TypeError):
        return {
            "verdict": "invalid",
            "confidence": 0.0,
            "reason": "CONFIDENCE_NOT_FLOAT",
            "valid_json": True,
            "strict_json": strict_json,
            "raw": raw_text,
        }

    reason_str = str(data.get("reason", "")).strip()

    return {
        "verdict": raw_verdict,
        "confidence": raw_conf,
        "reason": reason_str,
        "valid_json": True,
        "strict_json": strict_json,
        "raw": raw_text,
    }

