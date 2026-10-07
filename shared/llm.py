# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Multi-provider LLM client for AEL workflows.

Uses raw httpx calls (no openai/anthropic/google SDK dependencies).
Supports 8 provider backends via unified interface:
  - OpenAI (GPT-4.1, GPT-4o-mini, GPT-5.4, o4-mini)
  - Anthropic (Claude Haiku 4.5, Sonnet 4.6, Opus 4.6)
  - Google (Gemini Flash-Lite, 2.5 Flash, 2.5 Pro)
  - DeepSeek (V3, R1) — OpenAI-compatible
  - Mistral (Nemo, Codestral, Large 3) — OpenAI-compatible
  - xAI (Grok) — OpenAI-compatible
  - OpenRouter (Llama 4, Qwen3, any model) — OpenAI-compatible
  - Ollama (local inference) — OpenAI-compatible

Usage:
    from shared.llm import LLMClient

    # OpenAI (default)
    client = LLMClient(collector=collector, agent_name="TrendSurfer")  # AEL_MODEL, else default_model

    # Anthropic
    client = LLMClient(model="claude-haiku-4-5-20251001")

    # Google
    client = LLMClient(model="gemini-2.5-flash")

    # DeepSeek
    client = LLMClient(model="deepseek-chat")

    # Mistral
    client = LLMClient(model="open-mistral-nemo")

    # Task-type routing (selects best available model automatically)
    client = LLMClient(task_type="extraction")  # → cheapest available model

    # Same interface for all providers
    text = client.invoke([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ])
"""

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Type

import httpx
from pydantic import BaseModel

API_URL = "https://api.openai.com/v1/chat/completions"
EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Per-call HTTP timeout (seconds). Configurable for slow local backends (e.g. a
# large model served by vLLM in eager mode). Default 600s; override AEL_LLM_TIMEOUT.
_LLM_TIMEOUT = float(os.environ.get("AEL_LLM_TIMEOUT", "600"))


def _vllm_endpoint(model_name: str) -> Optional[str]:
    """Per-model vLLM endpoint from AEL_VLLM_ENDPOINTS (for a diverse committee).

    A single ``VLLM_BASE_URL`` routes every ``vllm/`` model to one server — fine for a
    single served model, but the LLM-Economist committee needs to address several distinct
    models (e.g. gpt-oss/Phi-4-mini/Olmo) at once. ``AEL_VLLM_ENDPOINTS`` maps each served-model-name
    to its own chat-completions URL:

        AEL_VLLM_ENDPOINTS="gpt-oss-20b=http://localhost:11436/v1/chat/completions,\
phi-4-mini=http://localhost:11435/v1/chat/completions"

    Returns the URL for ``model_name`` (the served name, prefix already stripped), or None
    to fall back to ``VLLM_BASE_URL`` / the default. Never raises.
    """
    raw = os.environ.get("AEL_VLLM_ENDPOINTS", "").strip()
    if not raw:
        return None
    for pair in raw.split(","):
        name, sep, url = pair.partition("=")
        if sep and name.strip() == model_name:
            return url.strip() or None
    return None


def _thinking_kwargs(provider: str, model: str = "") -> Optional[Dict[str, Any]]:
    """Return chat_template_kwargs to disable a reasoning model's <think> phase.

    Gated by AEL_DISABLE_THINKING (set in the empirical-run sbatch) and only for
    local OpenAI-compatible backends (ollama **and vLLM**) serving a **Qwen3.x** model
    whose chat template honours ``enable_thinking``. Restricting to Qwen avoids the HTTP
    400 that Mistral/Gemma tokenizers raise on this kwarg. Keeps generations short and
    avoids 10x-slower, timeout-prone reasoning traces. No-op otherwise.
    """
    if provider not in ("ollama", "vllm"):
        return None
    if os.environ.get("AEL_DISABLE_THINKING", "").lower() not in ("1", "true", "yes"):
        return None
    if "qwen" not in (model or "").lower():
        return None
    return {"enable_thinking": False}

# Import provider routing from single source of truth (V0.6)
from shared.provider_map import (  # noqa: E402
    PROVIDER_ROUTES,
    OPENAI_COMPATIBLE_PROVIDERS,
    PROVIDER_API_KEY_ENV,
    detect_provider,
)

# Model tiers for task-based selection (legacy, kept for backward compat)
MODEL_TIERS: Dict[str, List[str]] = {
    "routine": ["gpt-4o-mini", "claude-haiku-4-5-20251001", "gemini-2.5-flash"],
    "complex": ["gpt-4o", "claude-sonnet-4-5-20250514", "gemini-2.5-pro"],
}


class LLMClient:
    """
    Multi-provider LLM client using raw httpx.

    Supported providers:
    - vLLM (local open-weight serving) — default (vllm/qwen3.6-27b-fp8, see shared/model_config.py)
    - OpenAI (GPT-4.1, GPT-4o-mini, GPT-5.4, o4-mini)
    - Anthropic (Claude Haiku 4.5, Sonnet 4.6, Opus 4.6)
    - Google (Gemini Flash-Lite, 2.5 Flash, 2.5 Pro)
    - DeepSeek (V3, R1) — OpenAI-compatible
    - Mistral (Nemo, Codestral, Large 3) — OpenAI-compatible
    - xAI (Grok) — OpenAI-compatible
    - OpenRouter (Llama 4, Qwen3) — OpenAI-compatible
    - Ollama (local inference) — OpenAI-compatible

    Features:
    - Direct HTTP calls (no SDK dependencies)
    - Automatic provider routing from model name
    - Task-type-based model selection via ModelRouter
    - Optional Pydantic model parsing of JSON responses
    - Automatic observability metrics via MetricsCollector
    - Prompt caching for repeated system prompts (hash-based dedup)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.3,
        api_key: Optional[str] = None,
        collector: Optional[Any] = None,
        agent_name: str = "",
        enable_prompt_cache: bool = False,
        semantic_cache: Optional[Any] = None,
        task_type: Optional[str] = None,
        router: Optional[Any] = None,
        allow_env_override: bool = True,
    ):
        # If task_type is provided, use ModelRouter to select model
        if task_type is not None:
            if router is None:
                from shared.llm_router import ModelRouter
                router = ModelRouter()
            model = router.select_model(task_type)

        # AEL_MODEL env var overrides model globally (zero-config model switching)
        # e.g. export AEL_MODEL=ollama/llama-4-scout  to use Ollama for all agents.
        # allow_env_override=False pins the explicit model — required by the LLM-Economist
        # committee, whose members are DISTINCT models (Qwen/Mistral/Gemma) and must not be
        # collapsed onto the single AEL_MODEL the agents use.
        env_model = os.environ.get("AEL_MODEL")
        if env_model and task_type is None and allow_env_override:
            model = env_model
        # No model named by the caller or AEL_MODEL: ael_config.yaml default_model (local vLLM)
        if not model:
            from shared.model_config import default_model
            model = default_model()

        self.model = model
        self.temperature = temperature
        self.provider = detect_provider(model)
        self.collector = collector
        self.agent_name = agent_name
        self.enable_prompt_cache = enable_prompt_cache
        self.semantic_cache = semantic_cache
        self._router = router
        # Simple hash-based prompt cache: hash(messages) -> response text
        self._prompt_cache: Dict[str, str] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Resolve API key based on provider. Agents pass the OpenAI key explicitly; when
        # AEL_MODEL routes the call to another provider (vLLM, Ollama, a commercial API), that
        # key is not sent there and the provider's own key variable is used instead.
        env_key = PROVIDER_API_KEY_ENV.get(self.provider, "OPENAI_API_KEY")
        _openai_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key and not (self.provider != "openai" and api_key == _openai_key):
            self.api_key = api_key
        else:
            self.api_key = os.environ.get(env_key, "")

        # Store env key name for validation at call time
        self._env_key = PROVIDER_API_KEY_ENV.get(self.provider, "OPENAI_API_KEY")

    def invoke(
        self,
        messages: List[Dict[str, str]],
        parse_as: Optional[Type[BaseModel]] = None,
        response_format: Optional[Dict[str, str]] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """
        Send messages to the configured LLM provider and return the response.

        Automatically routes to OpenAI, Anthropic, or Google based on model name.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            parse_as: Optional Pydantic model class. If provided, the response
                      content is parsed as JSON and validated into this model.
            response_format: Optional response format dict, e.g.
                      {"type": "json_object"} for JSON mode.

        Returns:
            If parse_as is None: the response text (str).
            If parse_as is set: an instance of the Pydantic model.
        """
        # Validate API key before making a call (skip for local servers: ollama / vllm)
        if not self.api_key and self.provider not in ("ollama", "vllm"):
            raise ValueError(
                f"No API key found for provider '{self.provider}'.\n"
                f"Set the {self._env_key} environment variable, or pass api_key= to LLMClient.\n"
                f"  export {self._env_key}=your-key-here\n"
                f"Or use Ollama for free local inference: model='ollama/llama-4-scout'"
            )

        # Quick Ollama reachability check (avoids 180s timeout surprise)
        if self.provider == "ollama" and not getattr(self, "_ollama_checked", False):
            self._ollama_checked = True
            _host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            try:
                httpx.get(f"{_host}/api/tags", timeout=3.0)
            except Exception:
                raise ConnectionError(
                    f"Ollama is not reachable at {_host}.\n"
                    f"Start Ollama first:  ollama serve\n"
                    f"Then pull a model:   ollama pull {self.model.replace('ollama/', '')}\n"
                    f"Install from: https://ollama.com"
                )

        # Prompt cache check (deterministic calls only: temperature=0)
        cache_key = None
        if self.enable_prompt_cache and self.temperature == 0 and parse_as is None:
            cache_key = self._cache_key(messages)
            if cache_key in self._prompt_cache:
                self._cache_hits += 1
                return self._prompt_cache[cache_key]

        # Semantic cache check (persistent, cross-session, temperature=0 only)
        if (self.semantic_cache is not None
                and self.temperature == 0
                and parse_as is None):
            try:
                sc_text = " ".join(m.get("content", "") for m in messages)
                _messages = messages
                _response_format = response_format
                _self = self

                def _llm_on_miss(_prompt):
                    """Actual LLM call on semantic cache miss."""
                    _start = time.time()
                    if _self.provider == "anthropic":
                        _c, _u = _self._invoke_anthropic(_messages)
                    elif _self.provider == "google":
                        _c, _u = _self._invoke_google(_messages)
                    elif _self.provider in OPENAI_COMPATIBLE_PROVIDERS:
                        _c, _u = _self._invoke_openai_compatible(
                            _messages, _response_format, max_tokens=max_tokens
                        )
                    elif _self.provider == "openai":
                        _c, _u = _self._invoke_openai(_messages, _response_format)
                    else:
                        raise ValueError(
                            f"Model '{_self.model}' maps to provider '{_self.provider}', which "
                            "has no endpoint configured in shared/provider_map.py"
                        )
                    _latency = time.time() - _start
                    if _self.collector is not None and _u is not None:
                        from shared.observability import LLMCallRecord, estimate_cost
                        _pt = _u.get("prompt_tokens", 0)
                        _ct = _u.get("completion_tokens", 0)
                        _cost = estimate_cost(_self.model, _pt, _ct)
                        _self.collector.record_llm_call(LLMCallRecord(
                            agent=_self.agent_name, model=_self.model,
                            prompt_tokens=_pt, completion_tokens=_ct,
                            total_tokens=_u.get("total_tokens", 0),
                            cost_usd=_cost, latency_seconds=_latency,
                            temperature=_self.temperature,
                        ))
                    return _c

                sc_result = _self.semantic_cache.get_or_compute(
                    prompt=sc_text,
                    llm_fn=_llm_on_miss,
                    embed_fn=_self.embed,
                    model=_self.model,
                )
                if cache_key is not None:
                    self._prompt_cache[cache_key] = sc_result.response
                return sc_result.response
            except Exception:
                pass  # Fall through to normal path on any error

        start = time.time()

        # Progress feedback for long LLM calls
        _agent_label = f"[{self.agent_name}] " if self.agent_name else ""
        _quiet = os.environ.get("AGENT_QUIET_MODE", "").lower() in ("true", "1")
        if not _quiet:
            print(f"  {_agent_label}Calling {self.model}...", end="", flush=True)

        # When a Pydantic parse target is declared, force OpenAI / compatible
        # providers into JSON mode so the model cannot return prose-wrapped
        # output or a truncated mid-string response that passes nothing
        # through safe_json_loads. Anthropic / Google keep their own behaviour
        # (prompt-level instructions in the stage templates cover those).
        if (
            parse_as is not None
            and response_format is None
            and self.provider in ({"openai"} | set(OPENAI_COMPATIBLE_PROVIDERS))
        ):
            response_format = {"type": "json_object"}

        try:
            # Route to provider
            if self.provider == "anthropic":
                content, usage = self._invoke_anthropic(messages)
            elif self.provider == "google":
                content, usage = self._invoke_google(messages)
            elif self.provider in OPENAI_COMPATIBLE_PROVIDERS:
                content, usage = self._invoke_openai_compatible(
                    messages, response_format, max_tokens=max_tokens
                )
            elif self.provider == "openai":
                content, usage = self._invoke_openai(messages, response_format)
            else:
                raise ValueError(
                    f"Model '{self.model}' maps to provider '{self.provider}', which has no "
                    "endpoint configured in shared/provider_map.py"
                )

            latency = time.time() - start
            if not _quiet:
                print(f" done ({latency:.1f}s)", flush=True)

            # Record metrics
            if self.collector is not None and usage is not None:
                from shared.observability import LLMCallRecord, estimate_cost

                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)

                cost = estimate_cost(self.model, prompt_tokens, completion_tokens)
                record = LLMCallRecord(
                    agent=self.agent_name,
                    model=self.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost_usd=cost,
                    latency_seconds=latency,
                    temperature=self.temperature,
                )
                self.collector.record_llm_call(record)

            if parse_as is not None:
                import json as _json
                from shared.json_repair import safe_json_loads

                parsed = safe_json_loads(content, fallback=None)
                if parsed is None:
                    # Give the caller enough context to diagnose the failure
                    # rather than just the first 200 chars. Include length +
                    # tail (useful when the response was truncated mid-JSON)
                    # and the direct json.loads error.
                    try:
                        _json.loads(content)
                        _why = "unknown (repair chain returned None)"
                    except _json.JSONDecodeError as _je:
                        _why = f"JSONDecodeError: {_je.msg} at line {_je.lineno} col {_je.colno} (pos {_je.pos})"
                    raise ValueError(
                        f"Failed to parse LLM response as JSON for {parse_as.__name__} "
                        f"(len={len(content)}, reason={_why}).\n"
                        f"HEAD: {content[:200]!r}\n"
                        f"TAIL: {content[-200:]!r}"
                    )
                return parse_as(**parsed)

            # Store in prompt cache on success
            if cache_key is not None:
                self._cache_misses += 1
                self._prompt_cache[cache_key] = content

            return content

        except Exception as e:
            latency = time.time() - start
            if not _quiet:
                print(f" FAILED ({latency:.1f}s)", flush=True)
            # credentials never leave through an error message or the metrics
            try:
                from shared.tools.redact import redact_secrets
                _err = redact_secrets(str(e))
                if e.args:
                    e.args = (_err,) + tuple(e.args[1:])
            except Exception:
                _err = "error (message withheld)"
            if self.collector is not None:
                from shared.observability import LLMCallRecord

                record = LLMCallRecord(
                    agent=self.agent_name,
                    model=self.model,
                    latency_seconds=latency,
                    error=_err,
                    temperature=self.temperature,
                )
                self.collector.record_llm_call(record)
            raise

    def _invoke_openai(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
    ) -> tuple:
        """Call OpenAI chat completions API. Returns (content, usage)."""
        body: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
        }
        if response_format is not None:
            body["response_format"] = response_format

        resp = httpx.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=_LLM_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage")
        return content, usage

    def _invoke_openai_compatible(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple:
        """
        Call an OpenAI-compatible API (DeepSeek, Mistral, xAI, OpenRouter, Ollama).

        Uses the same request/response format as OpenAI but with a different
        base URL and API key.
        """
        config = OPENAI_COMPATIBLE_PROVIDERS.get(self.provider, {})
        url = config.get("base_url", API_URL)

        # For local backends (vllm/ Layer 1, ollama/ Layer 2) the SERVED model name has no
        # provider prefix — strip it (vllm/qwen3.6-27b-fp8 -> qwen3.6-27b-fp8), else the
        # server 404s ("model does not exist").
        model_name = self.model
        for _pfx in ("vllm/", "ollama/"):
            if self.provider == _pfx[:-1] and model_name.startswith(_pfx):
                model_name = model_name[len(_pfx):]
                break

        # Layer 1 HPC vLLM endpoint is deployment-specific. Prefer a per-model endpoint
        # (AEL_VLLM_ENDPOINTS — lets the committee address several servers at once), else a
        # single VLLM_BASE_URL, else the default.
        if self.provider == "vllm":
            url = _vllm_endpoint(model_name) or os.environ.get("VLLM_BASE_URL", url)
        # For OpenRouter, keep the full model name (meta-llama/llama-4-scout)

        # Explicit generous cap so verbose structured outputs (e.g. a full QuestionList) are not
        # truncated mid-JSON by the server default. BUT it must never push prompt+max_tokens past
        # the server's max_model_len, or vLLM returns 400 Bad Request (this silently dropped every
        # LARGE-prompt judge call — e.g. DataTeam outputs — from the tier-2 results). So cap it to
        # the remaining context: max_tokens = clamp(AEL_MAX_TOKENS, 256, ctx - prompt_est - margin).
        _prompt_est = sum(len(str(m.get("content", ""))) for m in messages) // 4
        _ctx = int(os.environ.get("VLLM_MAX_MODEL_LEN", "16384"))
        # A per-call max_tokens (for verbose calls with a known bound, e.g. equation formulation)
        # overrides AEL_MAX_TOKENS but is still clamped to the remaining context window.
        _cap = max_tokens if max_tokens is not None else int(os.environ.get("AEL_MAX_TOKENS", "8192"))
        _max_tok = max(256, min(_cap, _ctx - _prompt_est - 512))
        body: Dict[str, Any] = {
            "model": model_name,
            "temperature": self.temperature,
            "messages": messages,
            "max_tokens": _max_tok,
        }
        if response_format is not None:
            body["response_format"] = response_format
        # Disable reasoning <think> phase for local thinking models when requested
        # (keeps generations short; avoids timeout-prone reasoning traces).
        _think = _thinking_kwargs(self.provider, self.model)
        if _think is not None:
            body["chat_template_kwargs"] = _think

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }
        # Ollama doesn't require auth; others use Bearer token
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # OpenRouter requires extra headers
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/AgenticEconLab/agentic-econ-lab"
            headers["X-Title"] = "AEL-Pipeline"

        resp = httpx.post(url, headers=headers, json=body, timeout=_LLM_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage")
        return content, usage

    def _invoke_anthropic(
        self,
        messages: List[Dict[str, str]],
    ) -> tuple:
        """Call Anthropic messages API. Returns (content, usage)."""
        # Extract system message if present
        system_text = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                user_messages.append(msg)

        body: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": user_messages or [{"role": "user", "content": ""}],
        }
        if system_text:
            body["system"] = system_text
        if self.temperature is not None:
            body["temperature"] = self.temperature

        resp = httpx.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=_LLM_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        # Anthropic returns content as list of blocks
        content_blocks = data.get("content", [])
        content = ""
        for block in content_blocks:
            if block.get("type") == "text":
                content += block.get("text", "")

        # Map Anthropic usage to OpenAI-style
        anthropic_usage = data.get("usage", {})
        usage = {
            "prompt_tokens": anthropic_usage.get("input_tokens", 0),
            "completion_tokens": anthropic_usage.get("output_tokens", 0),
            "total_tokens": (
                anthropic_usage.get("input_tokens", 0)
                + anthropic_usage.get("output_tokens", 0)
            ),
        }
        return content, usage

    def _invoke_google(
        self,
        messages: List[Dict[str, str]],
    ) -> tuple:
        """Call Google Gemini API. Returns (content, usage)."""
        # Convert OpenAI-style messages to Gemini format
        system_text = ""
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}],
                })

        body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
            },
        }
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}

        # the key travels in a header, not the URL: HTTP errors quote the URL
        url = f"{GOOGLE_URL}/{self.model}:generateContent"

        resp = httpx.post(
            url,
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            json=body,
            timeout=_LLM_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        # Extract content from Gemini response
        candidates = data.get("candidates", [])
        content = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                content += part.get("text", "")

        # Map Gemini usage
        gemini_usage = data.get("usageMetadata", {})
        usage = {
            "prompt_tokens": gemini_usage.get("promptTokenCount", 0),
            "completion_tokens": gemini_usage.get("candidatesTokenCount", 0),
            "total_tokens": gemini_usage.get("totalTokenCount", 0),
        }
        return content, usage

    def format_and_invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        variables: Optional[Dict[str, str]] = None,
        parse_as: Optional[Type[BaseModel]] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Convenience method: format prompts with variables and invoke.

        Args:
            system_prompt: System message template (use {var} for substitution).
            user_prompt: User message template (use {var} for substitution).
            variables: Dict of template variables to substitute.
            parse_as: Optional Pydantic model to parse JSON response into.
            response_format: Optional response format dict for JSON mode.

        Returns:
            Response text or Pydantic model instance.
        """
        if variables:
            system_prompt = system_prompt.format(**variables)
            user_prompt = user_prompt.format(**variables)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.invoke(messages, parse_as=parse_as, response_format=response_format)

    def _cache_key(self, messages: List[Dict[str, str]]) -> str:
        """Compute a deterministic hash for a message list."""
        raw = json.dumps(messages, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return prompt cache statistics."""
        total = self._cache_hits + self._cache_misses
        return {
            "enabled": self.enable_prompt_cache,
            "entries": len(self._prompt_cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": round(self._cache_hits / total, 4) if total > 0 else 0.0,
        }

    def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Compute embeddings for a list of texts via the OpenAI embeddings API.

        Args:
            texts: List of text strings to embed.
            model: Embedding model name (default: text-embedding-3-small).

        Returns:
            List of embedding vectors (list of floats), one per input text.
        """
        if not texts:
            return []

        embed_model = model or "text-embedding-3-small"
        # Embeddings always go to OpenAI, so they use the OpenAI key, never another provider's.
        openai_key = self.api_key if self.provider == "openai" else os.environ.get("OPENAI_API_KEY", "")
        start = time.time()

        try:
            if not openai_key:
                raise ValueError("embeddings use the OpenAI endpoint; OPENAI_API_KEY is not set")
            resp = httpx.post(
                EMBEDDINGS_URL,
                headers={
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json",
                },
                json={"input": texts, "model": embed_model},
                timeout=_LLM_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            latency = time.time() - start
            usage = data.get("usage", {})
            embeddings = [item["embedding"] for item in data["data"]]
            dimension = len(embeddings[0]) if embeddings else 0

            if self.collector is not None:
                from shared.observability import EmbeddingCallRecord, estimate_cost

                input_tokens = usage.get("prompt_tokens", usage.get("total_tokens", 0))
                cost = estimate_cost(embed_model, input_tokens, 0)
                record = EmbeddingCallRecord(
                    agent=self.agent_name,
                    model=embed_model,
                    input_tokens=input_tokens,
                    num_texts=len(texts),
                    dimension=dimension,
                    cost_usd=cost,
                    latency_seconds=latency,
                )
                self.collector.record_embedding_call(record)

            return embeddings

        except Exception as e:
            latency = time.time() - start
            if self.collector is not None:
                from shared.observability import EmbeddingCallRecord

                record = EmbeddingCallRecord(
                    agent=self.agent_name,
                    model=embed_model,
                    num_texts=len(texts),
                    latency_seconds=latency,
                    error=str(e),
                )
                self.collector.record_embedding_call(record)
            raise
