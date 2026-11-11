"""Model provider interfaces: Mock harness plumbing and an OpenAI-compatible API adapter."""

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

from rtlreviewbench.prompt import parse_model_response


class ModelProvider(ABC):
    """Abstract base class for review model providers."""

    @abstractmethod
    def call(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Execute one review query."""
        raise NotImplementedError


class MockProvider(ModelProvider):
    """
    Harness plumbing provider.
    DOES NOT read gold labels to simulate success; output must never appear as a real model result.
    """

    def __init__(self, behavior: str = "abstain", smoke_test_only: bool = True):
        self.behavior = behavior
        self.smoke_test_only = smoke_test_only

    def call(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        start = time.perf_counter()
        if self.behavior == "malformed":
            raw_text = "This is not valid JSON output from the model."
        elif self.behavior == "invalid_verdict":
            raw_text = json.dumps({
                "verdict": "maybe_working",
                "confidence": 0.5,
                "reason": "Non-standard verdict"
            })
        else:
            raw_text = json.dumps({
                "verdict": "abstain",
                "confidence": 0.5,
                "reason": "Mock provider constant abstention for plumbing validation"
            })

        latency_ms = (time.perf_counter() - start) * 1000.0
        parsed = parse_model_response(raw_text)
        return {
            "parsed": parsed,
            "raw_response": raw_text,
            "latency_ms": latency_ms,
            "prompt_tokens": len(system_prompt.split()) + len(user_prompt.split()),
            "completion_tokens": len(raw_text.split()),
            "total_tokens": len(system_prompt.split()) + len(user_prompt.split()) + len(raw_text.split()),
            "smoke_test_only": True,
            "provider": "mock",
            "model_id": f"mock-{self.behavior}",
        }


class OpenAICompatibleProvider(ModelProvider):
    """
    Adapter for any OpenAI-compatible chat-completions endpoint.
    Used with models such as gemini-3.8-flash, gpt-4.1-mini, claude-haiku-5.5.
    """

    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        max_attempts: int = 8,
        timeout: int = 60,
    ):
        load_dotenv(override=False)
        self.model_id = model_id
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. Real model execution blocked."
            )
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.timeout = timeout

    def call(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = int(self.max_tokens)

        last_error = None
        start = time.perf_counter()

        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    allow_redirects=False,
                )
            except requests.RequestException as err:
                last_error = err
                time.sleep(1.0 * attempt)
                continue

            if resp.status_code == 200:
                latency_ms = (time.perf_counter() - start) * 1000.0
                data = resp.json()
                raw_text = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                parsed = parse_model_response(raw_text)
                return {
                    "parsed": parsed,
                    "raw_response": raw_text,
                    "latency_ms": latency_ms,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "smoke_test_only": False,
                    "provider": "openai_compatible",
                    "model_id": self.model_id,
                }

            if resp.status_code == 429 and attempt < self.max_attempts:
                retry_after = resp.headers.get("Retry-After")
                wait_sec = float(retry_after) if retry_after and retry_after.isdigit() else min(65.0, 15.0 * attempt)
                print(f"    [429 Rate Limit] Sleeping {wait_sec:.0f}s before retry {attempt}/{self.max_attempts}...", flush=True)
                last_error = RuntimeError(f"HTTP 429: {resp.text[:150]}")
                time.sleep(wait_sec)
                continue

            if resp.status_code in [500, 502, 503, 504] and attempt < self.max_attempts:
                last_error = RuntimeError(f"HTTP {resp.status_code}")
                time.sleep(2.0 * attempt)
                continue

            raise RuntimeError(
                f"API request failed with HTTP {resp.status_code}: {resp.text[:200]}"
            )

        raise RuntimeError(
            f"API call failed after {self.max_attempts} attempts: {last_error}"
        )


def get_provider(model_name: str, config: Dict[str, Any]) -> ModelProvider:
    """Factory to instantiate provider from config."""
    models_cfg = config.get("models", {})
    if model_name not in models_cfg:
        # Default fallback to the OpenAI-compatible provider with the specified model_name
        return OpenAICompatibleProvider(model_id=model_name)

    spec = models_cfg[model_name]
    prov_type = spec.get("provider", "openai_compatible")

    if prov_type == "mock":
        return MockProvider(
            behavior=spec.get("behavior", "abstain"),
            smoke_test_only=spec.get("smoke_test_only", True),
        )
    elif prov_type == "openai_compatible":
        return OpenAICompatibleProvider(
            model_id=spec.get("model_id", model_name),
            temperature=float(spec.get("temperature", 0.0)),
            max_tokens=spec.get("max_tokens"),
        )
    else:
        raise ValueError(f"Unknown provider type: {prov_type}")

