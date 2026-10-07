# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Observability and performance monitoring layer for AEL workflows.

Captures LLM API usage (tokens, cost, latency), external tool/API invocations
(success rates, latency), and per-agent attribution across all AEL workflows.

Usage:
    from shared.observability import MetricsCollector, tracked_get
    from shared.llm import LLMClient

    collector = MetricsCollector()

    # LLM calls are tracked automatically via LLMClient
    client = LLMClient(collector=collector, agent_name="TrendSurfer")

    # Wrap external API calls
    response = tracked_get("https://api.semanticscholar.org/...", collector=collector, agent="TopicCrawler")
"""

import os
import json
import time
import random
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Note: the old framework callback handler (AELObservabilityHandler) has been removed.
# Observability is now handled directly by shared.llm.LLMClient which records
# metrics to MetricsCollector on every call.  The tracked_*() wrappers below
# continue to work unchanged for external API calls.


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

# Default pricing: (input_cost_per_1k_tokens, output_cost_per_1k_tokens)
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # OpenAI models
    "gpt-4.1-nano": (0.00002, 0.00015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o-mini-2024-07-18": (0.00015, 0.0006),
    "gpt-5-mini": (0.00025, 0.002),
    "gpt-4.1-mini": (0.0004, 0.0016),
    "o4-mini": (0.0011, 0.0044),
    "gpt-4.1": (0.002, 0.008),
    "gpt-5.4": (0.0025, 0.02),
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-2024-08-06": (0.0025, 0.01),
    "gpt-4o-2024-11-20": (0.0025, 0.01),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4": (0.03, 0.06),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    # Anthropic models
    "claude-haiku-4-5-20251001": (0.001, 0.005),
    "claude-3-5-haiku-20241022": (0.0008, 0.004),
    "claude-sonnet-4-6": (0.003, 0.015),
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
    "claude-opus-4-6": (0.005, 0.025),
    # Google models
    "gemini-flash-lite": (0.0001, 0.0004),
    "gemini-2.5-flash": (0.00015, 0.0006),
    "gemini-2.5-pro": (0.00125, 0.01),
    # DeepSeek models (OpenAI-compatible)
    "deepseek-chat": (0.00014, 0.00028),
    "deepseek-reasoner": (0.0005, 0.00218),
    # Mistral models (OpenAI-compatible)
    "open-mistral-nemo": (0.00002, 0.00004),
    "codestral-latest": (0.0003, 0.0009),
    "mistral-large-latest": (0.0005, 0.0015),
    # xAI models (OpenAI-compatible)
    "grok-3-mini-fast": (0.0002, 0.0005),
    # OpenRouter models (Llama, Qwen)
    "meta-llama/llama-4-scout": (0.00008, 0.0003),
    "meta-llama/llama-4-maverick": (0.00015, 0.0006),
    "qwen/qwen3-32b": (0.00015, 0.00075),
    # Ollama models (local — zero cost)
    "ollama/llama-4-scout": (0.0, 0.0),
    "ollama/qwen3-32b": (0.0, 0.0),
    # OpenAI embedding models (output cost is 0 for embeddings)
    "text-embedding-3-small": (0.00002, 0.0),
    "text-embedding-3-large": (0.00013, 0.0),
}

# Allow override via environment variable (JSON file path)
_pricing_override = os.environ.get("AEL_PRICING_CONFIG")
if _pricing_override and os.path.isfile(_pricing_override):
    try:
        with open(_pricing_override, "r") as _f:
            _custom = json.load(_f)
            for _model, _prices in _custom.items():
                MODEL_PRICING[_model] = (float(_prices[0]), float(_prices[1]))
    except Exception as _e:
        import logging
        logging.getLogger("ael.observability").debug("Pricing config override failed: %s", _e)



def _redact_exc(e: BaseException) -> str:
    """Redacted text of an HTTP exception, also written back into the exception.

    requests' errors embed the full URL with its query string, so a re-raised error printed
    the FRED API key in the job logs. The type is kept so
    callers' except clauses still match."""
    try:
        from shared.tools.redact import redact_secrets
        msg = redact_secrets(str(e))
        if e.args:
            e.args = (msg,) + tuple(e.args[1:])
        return msg
    except Exception:
        return "error (message withheld)"

def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost in USD for a single LLM call."""
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        # Try prefix matching (e.g. "gpt-4o-mini-..." -> "gpt-4o-mini")
        for key in MODEL_PRICING:
            if model.startswith(key):
                pricing = MODEL_PRICING[key]
                break
    if pricing is None:
        return 0.0
    input_cost, output_cost = pricing
    return (prompt_tokens / 1000.0) * input_cost + (completion_tokens / 1000.0) * output_cost


# ---------------------------------------------------------------------------
# Data classes (lightweight, __slots__ based)
# ---------------------------------------------------------------------------

class LLMCallRecord:
    """Record of a single LLM API call."""
    __slots__ = (
        "agent", "stage", "model", "prompt_tokens", "completion_tokens",
        "total_tokens", "cost_usd", "latency_seconds", "timestamp", "error",
        "temperature",
    )

    def __init__(
        self,
        agent: str = "",
        stage: str = "",
        model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_seconds: float = 0.0,
        timestamp: Optional[str] = None,
        error: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        self.agent = agent
        self.stage = stage
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.cost_usd = cost_usd
        self.latency_seconds = latency_seconds
        self.timestamp = timestamp or datetime.now().isoformat()
        self.error = error
        self.temperature = temperature

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "agent": self.agent,
            "stage": self.stage,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 8),
            "latency_seconds": round(self.latency_seconds, 4),
            "timestamp": self.timestamp,
        }
        if self.error:
            d["error"] = self.error
        if self.temperature is not None:
            d["temperature"] = self.temperature
        return d


class EmbeddingCallRecord:
    """Record of a single embedding API call."""
    __slots__ = (
        "agent", "stage", "model", "input_tokens", "num_texts",
        "dimension", "cost_usd", "latency_seconds", "timestamp", "error",
    )

    def __init__(
        self,
        agent: str = "",
        stage: str = "",
        model: str = "",
        input_tokens: int = 0,
        num_texts: int = 0,
        dimension: int = 0,
        cost_usd: float = 0.0,
        latency_seconds: float = 0.0,
        timestamp: Optional[str] = None,
        error: Optional[str] = None,
    ):
        self.agent = agent
        self.stage = stage
        self.model = model
        self.input_tokens = input_tokens
        self.num_texts = num_texts
        self.dimension = dimension
        self.cost_usd = cost_usd
        self.latency_seconds = latency_seconds
        self.timestamp = timestamp or datetime.now().isoformat()
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "agent": self.agent,
            "stage": self.stage,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "num_texts": self.num_texts,
            "dimension": self.dimension,
            "cost_usd": round(self.cost_usd, 8),
            "latency_seconds": round(self.latency_seconds, 4),
            "timestamp": self.timestamp,
        }
        if self.error:
            d["error"] = self.error
        return d


class ToolCallRecord:
    """Record of a single external tool/API call."""
    __slots__ = (
        "agent", "stage", "tool_name", "url", "http_method", "status_code",
        "latency_seconds", "success", "timestamp", "error",
    )

    def __init__(
        self,
        agent: str = "",
        stage: str = "",
        tool_name: str = "",
        url: str = "",
        http_method: str = "GET",
        status_code: Optional[int] = None,
        latency_seconds: float = 0.0,
        success: bool = True,
        timestamp: Optional[str] = None,
        error: Optional[str] = None,
    ):
        self.agent = agent
        self.stage = stage
        self.tool_name = tool_name
        self.url = url
        self.http_method = http_method
        self.status_code = status_code
        self.latency_seconds = latency_seconds
        self.success = success
        self.timestamp = timestamp or datetime.now().isoformat()
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "agent": self.agent,
            "stage": self.stage,
            "tool_name": self.tool_name,
            "url": self.url,
            "http_method": self.http_method,
            "status_code": self.status_code,
            "latency_seconds": round(self.latency_seconds, 4),
            "success": self.success,
            "timestamp": self.timestamp,
        }
        if self.error:
            d["error"] = self.error
        return d


# ---------------------------------------------------------------------------
# URL-to-tool-name mapping
# ---------------------------------------------------------------------------

_URL_TOOL_MAP = {
    "api.semanticscholar.org": "SemanticScholar",
    "api.openalex.org": "OpenAlex",
    "export.arxiv.org": "arXiv",
    "arxiv.org": "arXiv",
    "api.stlouisfed.org": "FRED",
    "api.worldbank.org": "WorldBank",
    "api.firecrawl.dev": "Firecrawl",
    "api.serper.dev": "Serper",
    "api.search.brave.com": "Brave",
    "api.tavily.com": "Tavily",
    "query1.finance.yahoo.com": "YahooFinance",
    "query2.finance.yahoo.com": "YahooFinance",
}


def _resolve_tool_name(url: str) -> str:
    """Resolve a URL to a human-readable tool name."""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
        for pattern, name in _URL_TOOL_MAP.items():
            if pattern in host:
                return name
        return host or "Unknown"
    except Exception as e:
        import logging
        logging.getLogger("ael.observability").debug("URL tool name resolution failed: %s", e)
        return "Unknown"


# ---------------------------------------------------------------------------
# MetricsCollector (thread-safe central store)
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Thread-safe central store for all observability metric records.

    One instance is created per workflow run and shared across all agents/stages.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._llm_calls: List[LLMCallRecord] = []
        self._tool_calls: List[ToolCallRecord] = []
        self._embedding_calls: List[EmbeddingCallRecord] = []
        self._current_stage: str = ""

    def set_context(self, stage: str = ""):
        """Update the current stage context."""
        with self._lock:
            self._current_stage = stage

    @property
    def current_stage(self) -> str:
        with self._lock:
            return self._current_stage

    def record_llm_call(self, record: LLMCallRecord):
        """Record an LLM API call."""
        if not record.stage:
            record.stage = self.current_stage
        with self._lock:
            self._llm_calls.append(record)

    def record_tool_call(self, record: ToolCallRecord):
        """Record an external tool/API call."""
        if not record.stage:
            record.stage = self.current_stage
        with self._lock:
            self._tool_calls.append(record)

    def record_embedding_call(self, record: EmbeddingCallRecord):
        """Record an embedding API call."""
        with self._lock:
            if not record.stage:
                record.stage = self._current_stage
            self._embedding_calls.append(record)

    def merge_from(self, other: "MetricsCollector"):
        """Merge records from another collector into this one.

        Used by the pipeline orchestrator to accumulate per-team metrics
        into the global pipeline collector after each team completes.
        """
        with self._lock:
            self._llm_calls.extend(other.llm_calls)
            self._tool_calls.extend(other.tool_calls)
            self._embedding_calls.extend(other.embedding_calls)

    def get_detailed_records(self) -> Dict[str, Any]:
        """Return all raw records as serializable dicts."""
        with self._lock:
            result: Dict[str, Any] = {
                "llm_calls": [r.to_dict() for r in self._llm_calls],
                "tool_calls": [r.to_dict() for r in self._tool_calls],
            }
            if self._embedding_calls:
                result["embedding_calls"] = [r.to_dict() for r in self._embedding_calls]
            return result

    @property
    def llm_calls(self) -> List[LLMCallRecord]:
        """Read-only access to LLM call records."""
        with self._lock:
            return list(self._llm_calls)

    @property
    def tool_calls(self) -> List[ToolCallRecord]:
        """Read-only access to tool call records."""
        with self._lock:
            return list(self._tool_calls)

    @property
    def embedding_calls(self) -> List[EmbeddingCallRecord]:
        """Read-only access to embedding call records."""
        with self._lock:
            return list(self._embedding_calls)

    def to_otel_spans(
        self, trace_id: Optional[str] = None, team: str = "", mode: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Convert all records to OTel-compatible span dicts.

        Uses gen_ai semantic conventions for attribute naming.
        Returns lightweight dicts (no OTel SDK required).

        Args:
            trace_id: Shared trace ID for all spans.
            team: Team name attribute.
            mode: Mode name attribute.

        Returns:
            List of span dicts ready for JSON export or OTel conversion.
        """
        from shared.telemetry.gen_ai_conventions import (
            GenAIAttributes,
            llm_call_attributes,
            tool_call_attributes,
            embedding_call_attributes,
        )
        import uuid

        tid = trace_id or uuid.uuid4().hex
        spans = []

        with self._lock:
            for rec in self._llm_calls:
                attrs = llm_call_attributes(rec)
                if team:
                    attrs[GenAIAttributes.AEL_TEAM] = team
                if mode:
                    attrs[GenAIAttributes.AEL_MODE] = mode
                spans.append({
                    "trace_id": tid,
                    "span_id": uuid.uuid4().hex[:16],
                    "name": f"gen_ai.chat {rec.model}",
                    "kind": "CLIENT",
                    "start_time": rec.timestamp,
                    "duration_ms": round(rec.latency_seconds * 1000, 2),
                    "attributes": attrs,
                    "status": "ERROR" if rec.error else "OK",
                })

            for rec in self._tool_calls:
                attrs = tool_call_attributes(rec)
                if team:
                    attrs[GenAIAttributes.AEL_TEAM] = team
                if mode:
                    attrs[GenAIAttributes.AEL_MODE] = mode
                spans.append({
                    "trace_id": tid,
                    "span_id": uuid.uuid4().hex[:16],
                    "name": f"tool.{rec.tool_name}",
                    "kind": "CLIENT",
                    "start_time": rec.timestamp,
                    "duration_ms": round(rec.latency_seconds * 1000, 2),
                    "attributes": attrs,
                    "status": "ERROR" if rec.error else "OK",
                })

            for rec in self._embedding_calls:
                attrs = embedding_call_attributes(rec)
                if team:
                    attrs[GenAIAttributes.AEL_TEAM] = team
                if mode:
                    attrs[GenAIAttributes.AEL_MODE] = mode
                spans.append({
                    "trace_id": tid,
                    "span_id": uuid.uuid4().hex[:16],
                    "name": f"gen_ai.embeddings {rec.model}",
                    "kind": "CLIENT",
                    "start_time": rec.timestamp,
                    "duration_ms": round(rec.latency_seconds * 1000, 2),
                    "attributes": attrs,
                    "status": "ERROR" if rec.error else "OK",
                })

        return spans

    def get_summary(self) -> Dict[str, Any]:
        """
        Return an aggregated summary suitable for inclusion in execution_log.json.

        Structure:
        {
            "schema_version": "1.0.0",
            "llm": { total_calls, total_tokens, total_cost_usd, ... },
            "tools": { total_calls, success_rate, ... },
            "by_agent": { agent_name: { llm: {...}, tools: {...} }, ... },
            "by_stage": { stage_name: { llm: {...}, tools: {...} }, ... },
        }
        """
        with self._lock:
            llm_calls = list(self._llm_calls)
            tool_calls = list(self._tool_calls)
            embedding_calls = list(self._embedding_calls)

        summary: Dict[str, Any] = {
            "schema_version": "1.0.0",
            "llm": self._aggregate_llm(llm_calls),
            "tools": self._aggregate_tools(tool_calls),
            "by_agent": self._group_by("agent", llm_calls, tool_calls),
            "by_stage": self._group_by("stage", llm_calls, tool_calls),
        }
        if embedding_calls:
            summary["embeddings"] = self._aggregate_embeddings(embedding_calls)
        return summary

    # -- Private aggregation helpers --

    @staticmethod
    def _aggregate_llm(records: List[LLMCallRecord]) -> Dict[str, Any]:
        if not records:
            return {
                "total_calls": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "avg_latency_seconds": 0.0,
                "error_count": 0,
                "temperatures": [],
            }
        total_prompt = sum(r.prompt_tokens for r in records)
        total_completion = sum(r.completion_tokens for r in records)
        total_tokens = sum(r.total_tokens for r in records)
        total_cost = sum(r.cost_usd for r in records)
        avg_latency = sum(r.latency_seconds for r in records) / len(records)
        error_count = sum(1 for r in records if r.error)
        # distinct sampling temperatures observed (lets the release gate verify the
        # config's temperature claim against what actually ran — see evaluation/audits/hallucination_audit.py)
        temperatures = sorted({r.temperature for r in records if r.temperature is not None})
        return {
            "total_calls": len(records),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "avg_latency_seconds": round(avg_latency, 4),
            "error_count": error_count,
            "temperatures": temperatures,
        }

    @staticmethod
    def _aggregate_tools(records: List[ToolCallRecord]) -> Dict[str, Any]:
        if not records:
            return {
                "total_calls": 0,
                "success_count": 0,
                "error_count": 0,
                "success_rate": 0.0,
                "avg_latency_seconds": 0.0,
            }
        success_count = sum(1 for r in records if r.success)
        error_count = len(records) - success_count
        avg_latency = sum(r.latency_seconds for r in records) / len(records)
        return {
            "total_calls": len(records),
            "success_count": success_count,
            "error_count": error_count,
            "success_rate": round(success_count / len(records), 4),
            "avg_latency_seconds": round(avg_latency, 4),
        }

    @staticmethod
    def _aggregate_embeddings(records: List[EmbeddingCallRecord]) -> Dict[str, Any]:
        if not records:
            return {
                "total_calls": 0,
                "total_input_tokens": 0,
                "total_texts_embedded": 0,
                "total_cost_usd": 0.0,
                "avg_latency_seconds": 0.0,
                "error_count": 0,
            }
        total_tokens = sum(r.input_tokens for r in records)
        total_texts = sum(r.num_texts for r in records)
        total_cost = sum(r.cost_usd for r in records)
        avg_latency = sum(r.latency_seconds for r in records) / len(records)
        error_count = sum(1 for r in records if r.error)
        return {
            "total_calls": len(records),
            "total_input_tokens": total_tokens,
            "total_texts_embedded": total_texts,
            "total_cost_usd": round(total_cost, 6),
            "avg_latency_seconds": round(avg_latency, 4),
            "error_count": error_count,
        }

    @staticmethod
    def _group_by(
        key: str,
        llm_calls: List[LLMCallRecord],
        tool_calls: List[ToolCallRecord],
    ) -> Dict[str, Any]:
        groups: Dict[str, Any] = {}

        # Group LLM calls
        llm_groups: Dict[str, List[LLMCallRecord]] = {}
        for r in llm_calls:
            group_key = getattr(r, key, "") or "unknown"
            llm_groups.setdefault(group_key, []).append(r)

        # Group tool calls
        tool_groups: Dict[str, List[ToolCallRecord]] = {}
        for r in tool_calls:
            group_key = getattr(r, key, "") or "unknown"
            tool_groups.setdefault(group_key, []).append(r)

        all_keys = set(llm_groups.keys()) | set(tool_groups.keys())
        for gk in sorted(all_keys):
            groups[gk] = {
                "llm": MetricsCollector._aggregate_llm(llm_groups.get(gk, [])),
                "tools": MetricsCollector._aggregate_tools(tool_groups.get(gk, [])),
            }

        return groups


# ---------------------------------------------------------------------------
# Tracked wrapper functions for external API calls
# ---------------------------------------------------------------------------

def _retry_wait(attempt: int, retry_backoff: float, exc: Exception) -> float:
    """Backoff before a retry: exponential, honoring a Retry-After header when the
    server sends one (429/503), plus jitter, capped at 60s."""
    wait = retry_backoff * (2 ** attempt)
    hdrs = getattr(getattr(exc, "response", None), "headers", None) or {}
    ra = hdrs.get("Retry-After") or hdrs.get("retry-after")
    if ra:
        try:
            wait = max(wait, float(ra))
        except (ValueError, TypeError):
            pass
    return min(wait + random.uniform(0.0, 0.5 * wait), 60.0)


def _http_timeout() -> float:
    """Default per-request HTTP timeout (seconds) for the tracked_* wrappers. Bounds every
    external call so a stuck endpoint can't hang the pipeline indefinitely (the same failure
    class as a hung web-crawl request). Override with ``AEL_HTTP_TIMEOUT``; applied ONLY when the
    caller didn't pass its own ``timeout=``."""
    try:
        return float(os.environ.get("AEL_HTTP_TIMEOUT", "30"))
    except (TypeError, ValueError):
        return 30.0


_SOCKET_TO_LOCK = threading.Lock()


class _bounded_socket:
    """Bound library calls that fetch via urllib/raw sockets (arxiv, yfinance, eikon) with the
    PROCESS-GLOBAL socket default timeout. The global is set/restored under a lock — without
    it, two concurrent wrapper calls (e.g. under shared.parallel.parallel_map) race: one
    thread's restore can strip the bound from the other's in-flight call, or persist a stale
    value. Serializing these (rare) fetches is an acceptable cost for a correct bound."""

    def __enter__(self):
        import socket
        _SOCKET_TO_LOCK.acquire()
        self._socket = socket
        self._old = socket.getdefaulttimeout()
        socket.setdefaulttimeout(_http_timeout())
        return self

    def __exit__(self, *exc):
        try:
            self._socket.setdefaulttimeout(self._old)
        finally:
            _SOCKET_TO_LOCK.release()
        return False


# Rate-limited APIs need PACING, not just retry — Semantic
# Scholar's unauthenticated tier is ~1 req/s and the sourcing agents burst it through
# web_fetch (which passes retries=0), so every burst 429'd straight to the Crossref
# fallback. tracked_get paces requests per domain (thread-safe slot reservation) and
# guarantees at least one Retry-After-honoring retry for paced domains.
_DOMAIN_MIN_INTERVAL: Dict[str, float] = {
    "api.semanticscholar.org": float(os.environ.get("S2_MIN_INTERVAL", "1.1")),
}
_DOMAIN_LAST_SLOT: Dict[str, float] = {}
_DOMAIN_PACE_LOCK = threading.Lock()


def _pace_domain(url: str) -> bool:
    """Sleep so calls to a rate-limited domain are at least its min-interval apart.
    Returns True when the domain is paced (caller then guarantees >=1 retry)."""
    try:
        from urllib.parse import urlparse
        host = (urlparse(url).netloc or "").lower()
    except Exception:                                       # pragma: no cover - defensive
        return False
    interval = _DOMAIN_MIN_INTERVAL.get(host)
    if not interval or interval <= 0:
        return False
    now = time.monotonic()
    with _DOMAIN_PACE_LOCK:
        slot = max(now, _DOMAIN_LAST_SLOT.get(host, 0.0) + interval)
        _DOMAIN_LAST_SLOT[host] = slot
    if slot > now:
        time.sleep(slot - now)
    return True


def tracked_get(
    url: str,
    collector: Optional[MetricsCollector] = None,
    agent: str = "",
    retries: int = 0,
    retry_on: tuple = (429, 500, 502, 503, 504),
    retry_backoff: float = 2.0,
    **kwargs: Any,
) -> "requests.Response":
    """
    Drop-in replacement for requests.get() with observability tracking and retry.

    Args:
        url: The URL to GET.
        collector: MetricsCollector instance (if None, behaves like plain requests.get).
        agent: Agent name for attribution.
        retries: Number of retry attempts on transient errors (default: 0).
        retry_on: HTTP status codes that trigger a retry.
        retry_backoff: Base backoff in seconds (doubled each attempt).
        **kwargs: Passed through to requests.get().

    Returns:
        requests.Response object.
    """
    import requests

    kwargs.setdefault("timeout", _http_timeout())  # bound the call — never hang unbounded
    if _pace_domain(url):
        retries = max(retries, 1)      # paced domains always get one Retry-After retry
    tool_name = _resolve_tool_name(url)
    last_exception = None

    for attempt in range(retries + 1):
        start = time.time()
        error_msg = None
        status_code = None
        success = True

        try:
            response = requests.get(url, **kwargs)
            status_code = response.status_code
            success = response.ok
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            error_msg = _redact_exc(e)
            success = False
            last_exception = e

            # Check if we should retry
            resp_status = getattr(getattr(e, "response", None), "status_code", None)
            if attempt < retries and resp_status in retry_on:
                time.sleep(_retry_wait(attempt, retry_backoff, e))
                # Intermediate retry attempts are NOT recorded as tool failures — only the
                # FINAL outcome counts toward success_rate (a transient 429 that succeeds on
                # retry is not a failure). Backoff honors Retry-After + jitter.
                continue
            # Not retryable or exhausted retries — record (in finally) and raise
            raise
        finally:
            # Record on success or final failure (non-retry path)
            if success or attempt == retries or (
                last_exception is not None
                and getattr(getattr(last_exception, "response", None), "status_code", None) not in retry_on
            ):
                latency = time.time() - start
                if collector is not None:
                    record = ToolCallRecord(
                        agent=agent,
                        tool_name=tool_name,
                        url=url,
                        http_method="GET",
                        status_code=status_code,
                        latency_seconds=latency,
                        success=success,
                        error=error_msg,
                    )
                    collector.record_tool_call(record)

    # Should not reach here, but just in case
    raise last_exception  # type: ignore[misc]


def tracked_post(
    url: str,
    collector: Optional[MetricsCollector] = None,
    agent: str = "",
    retries: int = 0,
    retry_on: tuple = (429, 500, 502, 503, 504),
    retry_backoff: float = 2.0,
    **kwargs: Any,
) -> "requests.Response":
    """Drop-in replacement for requests.post() with observability tracking and retry."""
    import requests

    kwargs.setdefault("timeout", _http_timeout())  # bound the call — never hang unbounded
    if _pace_domain(url):
        retries = max(retries, 1)      # paced domains always get one Retry-After retry
    tool_name = _resolve_tool_name(url)
    last_exception = None

    for attempt in range(retries + 1):
        start = time.time()
        error_msg = None
        status_code = None
        success = True

        try:
            response = requests.post(url, **kwargs)
            status_code = response.status_code
            success = response.ok
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            error_msg = _redact_exc(e)
            success = False
            last_exception = e

            resp_status = getattr(getattr(e, "response", None), "status_code", None)
            if attempt < retries and resp_status in retry_on:
                time.sleep(_retry_wait(attempt, retry_backoff, e))
                # Intermediate retries are not recorded as failures (final outcome only).
                continue
            raise
        finally:
            if success or attempt == retries or (
                last_exception is not None
                and getattr(getattr(last_exception, "response", None), "status_code", None) not in retry_on
            ):
                latency = time.time() - start
                if collector is not None:
                    record = ToolCallRecord(
                        agent=agent,
                        tool_name=tool_name,
                        url=url,
                        http_method="POST",
                        status_code=status_code,
                        latency_seconds=latency,
                        success=success,
                        error=error_msg,
                    )
                    collector.record_tool_call(record)

    raise last_exception  # type: ignore[misc]


def tracked_arxiv_search(
    query: str,
    max_results: int = 10,
    sort_by: Any = None,
    collector: Optional[MetricsCollector] = None,
    agent: str = "",
) -> list:
    """
    Wrapper around arxiv.Search().results() with observability tracking.

    Returns a list of arxiv Result objects.
    """
    import arxiv

    if sort_by is None:
        sort_by = arxiv.SortCriterion.Relevance
    elif isinstance(sort_by, str):
        # Callers (including ToolRegistry input schemas) pass "relevance" /
        # "lastUpdatedDate" / "submittedDate" as strings; the arxiv package
        # requires a SortCriterion enum in current versions.
        _map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastupdateddate": arxiv.SortCriterion.LastUpdatedDate,
            "submitteddate": arxiv.SortCriterion.SubmittedDate,
        }
        sort_by = _map.get(sort_by.lower().replace("_", ""),
                           arxiv.SortCriterion.Relevance)

    start = time.time()
    error_msg = None
    success = True
    results = []

    try:
        search = arxiv.Search(query=query, max_results=max_results, sort_by=sort_by)
        # arxiv >= 2.0 removed Search.results(); the supported API is Client().results(search).
        client = arxiv.Client()
        # The arxiv lib fetches via urllib (urlopen), which has NO timeout by default and
        # would hang the run if the endpoint stalls. Bound via the lock-guarded global
        # socket timeout (see _bounded_socket), matching AEL_HTTP_TIMEOUT.
        with _bounded_socket():
            results = list(client.results(search))
        return results
    except Exception as e:
        error_msg = _redact_exc(e)
        success = False
        raise
    finally:
        latency = time.time() - start
        if collector is not None:
            record = ToolCallRecord(
                agent=agent,
                tool_name="arXiv",
                url=f"https://export.arxiv.org/api/query?search_query={query[:80]}",
                http_method="GET",
                status_code=200 if success else None,
                latency_seconds=latency,
                success=success,
                error=error_msg,
            )
            collector.record_tool_call(record)


def tracked_fred_get_series(
    series_id: str,
    collector: Optional[MetricsCollector] = None,
    agent: str = "",
    **kwargs: Any,
) -> Any:
    """Wrapper for FRED API series data retrieval with observability tracking."""
    import requests

    api_key = kwargs.pop("api_key", os.environ.get("FRED_API_KEY", ""))
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    params.update(kwargs)

    start = time.time()
    error_msg = None
    status_code = None
    success = True

    try:
        response = requests.get(url, params=params, timeout=_http_timeout())
        status_code = response.status_code
        success = response.ok
        response.raise_for_status()
        return response.json()
    except Exception as e:
        error_msg = _redact_exc(e)
        success = False
        raise
    finally:
        latency = time.time() - start
        if collector is not None:
            record = ToolCallRecord(
                agent=agent,
                tool_name="FRED",
                url=url,
                http_method="GET",
                status_code=status_code,
                latency_seconds=latency,
                success=success,
                error=error_msg,
            )
            collector.record_tool_call(record)


def tracked_yfinance_history(
    ticker: str,
    period: str = "1y",
    collector: Optional[MetricsCollector] = None,
    agent: str = "",
    **kwargs: Any,
) -> Any:
    """Wrapper for yfinance ticker.history() with observability tracking."""
    start = time.time()
    error_msg = None
    success = True

    try:
        import yfinance as yf
        # yfinance fetches via its own requests session with NO default timeout — bound via
        # the lock-guarded global socket timeout (see _bounded_socket).
        with _bounded_socket():
            t = yf.Ticker(ticker)
            data = t.history(period=period, **kwargs)
        return data
    except Exception as e:
        error_msg = _redact_exc(e)
        success = False
        raise
    finally:
        latency = time.time() - start
        if collector is not None:
            record = ToolCallRecord(
                agent=agent,
                tool_name="YahooFinance",
                url=f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}",
                http_method="GET",
                status_code=200 if success else None,
                latency_seconds=latency,
                success=success,
                error=error_msg,
            )
            collector.record_tool_call(record)


def tracked_refinitiv_get_data(
    instruments: list,
    fields: list,
    parameters: Optional[Dict] = None,
    collector: Optional[MetricsCollector] = None,
    agent: str = "",
) -> Any:
    """Wrapper for Refinitiv Eikon ek.get_data() with observability tracking."""
    start = time.time()
    error_msg = None
    success = True
    status_code = None

    try:
        import eikon as ek
        # Eikon talks to the local terminal proxy; still bound the call so a wedged
        # terminal can't hang the run indefinitely (lock-guarded — see _bounded_socket).
        with _bounded_socket():
            df, err = ek.get_data(instruments=instruments, fields=fields, parameters=parameters or {})
        if err:
            error_msg = str(err)
            success = False
        else:
            status_code = 200
        return df, err
    except Exception as e:
        error_msg = _redact_exc(e)
        success = False
        raise
    finally:
        latency = time.time() - start
        if collector is not None:
            record = ToolCallRecord(
                agent=agent,
                tool_name="RefinitivEikon",
                url=f"eikon://get_data?instruments={','.join(instruments[:3])}",
                http_method="GET",
                status_code=status_code,
                latency_seconds=latency,
                success=success,
                error=error_msg,
            )
            collector.record_tool_call(record)


def tracked_file_parse(
    file_path: str,
    collector: Optional[MetricsCollector] = None,
    agent: str = "",
) -> Any:
    """Wrapper for pandas file parsing (CSV, Excel, etc.) with observability tracking."""
    import pandas as pd

    ext = os.path.splitext(file_path)[1].lower()
    parsers = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
        ".dta": pd.read_stata,
        ".sav": pd.read_spss,
        ".sas7bdat": pd.read_sas,
        ".json": pd.read_json,
        ".parquet": pd.read_parquet,
    }
    parser = parsers.get(ext, pd.read_csv)
    format_name = {
        ".csv": "CSV", ".xlsx": "Excel", ".xls": "Excel", ".dta": "Stata",
        ".sav": "SPSS", ".sas7bdat": "SAS", ".json": "JSON", ".parquet": "Parquet",
    }.get(ext, "CSV")

    start = time.time()
    error_msg = None
    success = True

    try:
        df = parser(file_path)
        return df
    except Exception as e:
        error_msg = _redact_exc(e)
        success = False
        raise
    finally:
        latency = time.time() - start
        if collector is not None:
            record = ToolCallRecord(
                agent=agent,
                tool_name=f"FileParse:{format_name}",
                url=f"file://{os.path.basename(file_path)}",
                http_method="READ",
                status_code=200 if success else None,
                latency_seconds=latency,
                success=success,
                error=error_msg,
            )
            collector.record_tool_call(record)
