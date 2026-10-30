"""Provider-agnostic LLM client over OpenAI-compatible chat completion APIs.

Includes a disk cache (re-runs are free and deterministic) and client-side
rate limiting with retries, which matters when working with free tiers.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import yaml


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, exactly as the model produced it
    # Provider-specific fields that must be sent back unchanged in the history
    # (e.g. Gemini's thought_signature, without which follow-up turns fail with 400).
    extra_content: dict | None = None


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False


class LLMClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


class MissingAPIKeyError(RuntimeError):
    pass


@dataclass
class ModelConfig:
    name: str
    provider: str
    model: str
    base_url: str
    api_key_env: str | None
    rpm: int | None = None
    temperature: float = 0.0
    max_tokens: int = 2048


def load_model_configs(path: str | Path) -> dict[str, ModelConfig]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    providers = raw["providers"]
    configs = {}
    for name, spec in raw["models"].items():
        provider = providers[spec["provider"]]
        configs[name] = ModelConfig(
            name=name,
            provider=spec["provider"],
            model=spec["model"],
            base_url=provider["base_url"],
            api_key_env=provider.get("api_key_env"),
            rpm=spec.get("rpm"),
            temperature=spec.get("temperature", 0.0),
            max_tokens=spec.get("max_tokens", 2048),
        )
    return configs


class OpenAICompatClient:
    """Chat client for any OpenAI-compatible endpoint (Gemini, Groq, OpenRouter, Ollama...)."""

    def __init__(
        self,
        cfg: ModelConfig,
        cache_dir: Path | None = None,
        cache_salt: str = "",
        max_retries: int = 6,
    ):
        from openai import OpenAI

        if cfg.api_key_env:
            api_key = os.environ.get(cfg.api_key_env)
            if not api_key:
                raise MissingAPIKeyError(
                    f"Model '{cfg.name}' needs the {cfg.api_key_env} environment variable "
                    "(add it to your .env file)."
                )
        else:
            api_key = "not-needed"
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.cache_salt = cache_salt
        self.max_retries = max_retries
        self._client = OpenAI(base_url=cfg.base_url, api_key=api_key, max_retries=0, timeout=120)
        self._last_request = 0.0

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "tools": tools,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        cache_path = self._cache_path(payload)
        if cache_path is not None and cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            data["tool_calls"] = [ToolCall(**tc) for tc in data["tool_calls"]]
            return LLMResponse(**{**data, "cached": True})

        response = self._request_with_retries(payload)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(asdict(response)), encoding="utf-8")
        return response

    def _cache_path(self, payload: dict) -> Path | None:
        if self.cache_dir is None:
            return None
        key = json.dumps({"salt": self.cache_salt, **payload}, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / self.cfg.name / f"{digest}.json"

    def _throttle(self) -> None:
        if not self.cfg.rpm:
            return
        min_interval = 60.0 / self.cfg.rpm
        wait = self._last_request + min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _request_with_retries(self, payload: dict) -> LLMResponse:
        import openai

        retryable = (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
        )
        for attempt in range(self.max_retries + 1):
            self._throttle()
            start = time.perf_counter()
            try:
                completion = self._client.chat.completions.create(**payload)
            except retryable:
                if attempt == self.max_retries:
                    raise
                time.sleep(min(60.0, 2.0 * 2**attempt))
                continue
            latency = time.perf_counter() - start
            return _parse_completion(completion, latency)
        raise AssertionError("unreachable")


def _parse_completion(completion, latency: float) -> LLMResponse:
    message = completion.choices[0].message
    tool_calls = [
        ToolCall(
            id=tc.id or f"call_{i}",
            name=tc.function.name,
            arguments=tc.function.arguments or "{}",
            extra_content=getattr(tc, "extra_content", None),
        )
        for i, tc in enumerate(message.tool_calls or [])
    ]
    usage = completion.usage
    return LLMResponse(
        content=message.content,
        tool_calls=tool_calls,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        latency_s=latency,
    )
