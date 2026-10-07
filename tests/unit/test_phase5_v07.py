# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""AEL V0.7 Phase 5 — Platform modernization (compaction, constrained decoding, cache, MCP v07)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from shared.llm_helpers import (
    CompactionResult,
    ConstrainedLLM,
    ConstrainedValidationError,
    compact_conversation,
)
from shared.llm_helpers.compaction import estimate_tokens
from shared.cache import CacheEntry, SemanticCacheGateway
from shared.protocols.mcp_v07_extensions import (
    MCP_PROTOCOL_VERSION,
    MCPTask,
    MCPTaskRegistry,
    OAuth21Authorizer,
    OAuth21ResourceMetadata,
    make_well_known_handler,
    negotiate_protocol_version,
)


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------

class TestCompaction:
    def test_estimate_tokens_monotone(self):
        assert estimate_tokens("short") < estimate_tokens("a much longer text block")

    def test_under_budget_no_op(self):
        msgs = [{"role": "user", "content": "hi"}]
        res = compact_conversation(msgs, budget_tokens=1000)
        assert res.strategy == "no_op"
        assert res.tokens_before == res.tokens_after

    def test_over_budget_summary_middle_preserves_head_and_tail(self):
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(20):
            msgs.append({"role": "user", "content": "x" * 200})
        res = compact_conversation(msgs, budget_tokens=100)
        # Head (system) must remain
        assert res.messages[0]["role"] == "system"
        # Must include a compaction-summary message inserted at index 1.
        # the summary uses role=user so providers that
        # reject multiple system messages accept the compacted sequence.
        assert "compaction summary" in res.messages[1]["content"].lower()
        assert res.messages[1]["role"] == "user"
        # Tail has at least 2 original messages
        assert len(res.messages) >= 3

    def test_preserve_head_tail_strategy(self):
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(10):
            msgs.append({"role": "user", "content": "y" * 100})
        res = compact_conversation(msgs, budget_tokens=100, strategy="preserve_head_tail")
        assert res.strategy == "preserve_head_tail"
        assert res.messages[0]["role"] == "system"
        assert res.tokens_after <= 100 + 10  # +intercept slack

    def test_reduction_ratio_positive_on_compaction(self):
        msgs = [{"role": "system", "content": "sys"}] + [
            {"role": "user", "content": "x" * 500} for _ in range(30)
        ]
        res = compact_conversation(msgs, budget_tokens=200)
        assert res.reduction_ratio > 0.30  # target: ≥30% savings

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            compact_conversation(
                [{"role": "user", "content": "x" * 1000}],
                budget_tokens=10,
                strategy="bogus",
            )

    def test_provider_native_fallback_graceful(self):
        class FakeClient:
            def compact_messages(self, messages, provider, budget_tokens):
                raise RuntimeError("fail")
        msgs = [{"role": "system", "content": "s"}] + [
            {"role": "user", "content": "x" * 100} for _ in range(10)
        ]
        res = compact_conversation(msgs, budget_tokens=50, llm_client=FakeClient())
        # Should fall back to summary_middle without raising
        assert res.strategy == "summary_middle"


# ---------------------------------------------------------------------------
# Constrained decoding
# ---------------------------------------------------------------------------

class _Schema(BaseModel):
    name: str
    count: int = Field(ge=0)


class _StubLLM:
    def __init__(self, responses):
        self._responses = list(responses)
    def invoke(self, prompt, schema=None, json_mode=None):
        return self._responses.pop(0)


class TestConstrainedLLM:
    def test_single_call_success(self):
        llm = _StubLLM(['{"name": "x", "count": 3}'])
        out = ConstrainedLLM(llm).invoke("produce json", _Schema)
        assert out.name == "x"
        assert out.count == 3

    def test_json_repair_path(self):
        # Malformed output (markdown fences) -> custom repair_fn fixes it
        llm = _StubLLM(['```json\n{"name": "x", "count": 3}\n```'])
        def strip_fences(text: str) -> str:
            return text.strip().removeprefix("```json").removesuffix("```").strip()
        out = ConstrainedLLM(llm, repair_fn=strip_fences).invoke("p", _Schema)
        assert out.name == "x"

    def test_retries_on_validation_error(self):
        llm = _StubLLM([
            '{"name": "x", "count": -1}',   # invalid (ge=0)
            '{"name": "x", "count": 5}',
        ])
        out = ConstrainedLLM(llm).invoke("p", _Schema, max_retries=1)
        assert out.count == 5

    def test_raises_after_exhausting_retries(self):
        llm = _StubLLM([
            '{"name": "x", "count": -1}',
            '{"name": "x", "count": -1}',
        ])
        with pytest.raises(ConstrainedValidationError):
            ConstrainedLLM(llm).invoke("p", _Schema, max_retries=1)

    def test_client_without_kwargs_still_works(self):
        class PlainLLM:
            def invoke(self, prompt):
                return '{"name": "y", "count": 7}'
        out = ConstrainedLLM(PlainLLM()).invoke("p", _Schema)
        assert out.name == "y"


# ---------------------------------------------------------------------------
# Semantic cache gateway
# ---------------------------------------------------------------------------

class TestCacheGateway:
    def test_disabled_by_default(self):
        g = SemanticCacheGateway()
        assert g.enabled is False

    def test_passthrough_when_disabled(self):
        g = SemanticCacheGateway(enabled=False)
        calls = []
        def fetch(p):
            calls.append(p)
            return "x"
        g.lookup_or_call("prompt", model="m", fetch=fetch)
        g.lookup_or_call("prompt", model="m", fetch=fetch)
        assert len(calls) == 2  # no caching

    def test_hit_when_enabled(self):
        g = SemanticCacheGateway(enabled=True, similarity_threshold=0.6)
        calls = []
        def fetch(p):
            calls.append(p)
            return "answer-" + p
        g.lookup_or_call("what is economics", model="m", fetch=fetch)
        g.lookup_or_call("what is economics really", model="m", fetch=fetch)
        # Second call should hit the cache because similarity > 0.6
        assert len(calls) == 1
        stats = g.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_key_separation_by_model(self):
        g = SemanticCacheGateway(enabled=True, similarity_threshold=0.5)
        calls = []
        def fetch(p):
            calls.append(p)
            return p
        g.lookup_or_call("p", model="A", fetch=fetch)
        g.lookup_or_call("p", model="B", fetch=fetch)
        assert len(calls) == 2  # different models = different cache keys

    def test_key_separation_by_seed(self):
        g = SemanticCacheGateway(enabled=True, similarity_threshold=0.5)
        def fetch(p): return p
        g.lookup_or_call("p", model="m", seed=1, fetch=fetch)
        assert g.get("p", model="m", seed=2) is None

    def test_eviction(self):
        g = SemanticCacheGateway(enabled=True, max_entries=2, similarity_threshold=0.99)
        g.put("aaa", "1", model="m")
        g.put("bbb", "2", model="m")
        # Triggering a put that overflows calls _evict_if_needed
        g.put("ccc", "3", model="m")
        # Note: put() does not evict directly; lookup_or_call does. Simulate:
        g.lookup_or_call("ddd", model="m", fetch=lambda p: p)
        assert sum(len(v) for v in g._store.values()) <= 3  # within bound

    def test_clear_resets_stats(self):
        g = SemanticCacheGateway(enabled=True)
        g.lookup_or_call("x", model="m", fetch=lambda p: p)
        g.clear()
        assert g.stats["hits"] == 0 and g.stats["misses"] == 0


# ---------------------------------------------------------------------------
# MCP V0.7 extensions
# ---------------------------------------------------------------------------

class TestMCPv07:
    def test_protocol_version_constant(self):
        assert isinstance(MCP_PROTOCOL_VERSION, str)
        assert "2026" in MCP_PROTOCOL_VERSION

    def test_negotiate_fallback(self):
        assert negotiate_protocol_version(None) == MCP_PROTOCOL_VERSION
        assert negotiate_protocol_version("bogus") == MCP_PROTOCOL_VERSION

    def test_negotiate_exact_match(self):
        assert negotiate_protocol_version(MCP_PROTOCOL_VERSION) == MCP_PROTOCOL_VERSION

    def test_well_known_metadata_roundtrip(self):
        md = OAuth21ResourceMetadata(
            resource="https://ael.example.com/mcp",
            authorization_servers=["https://auth.example.com"],
            scopes_supported=["mcp:read", "mcp:invoke"],
        )
        d = md.to_dict()
        assert d["resource"] == "https://ael.example.com/mcp"
        assert "mcp:invoke" in d["scopes_supported"]

    def test_oauth_authorize_rejects_missing_bearer(self):
        auth = OAuth21Authorizer(
            metadata=OAuth21ResourceMetadata("r", ["s"]),
            allowed_tokens=["tok-1"],
        )
        with pytest.raises(PermissionError):
            auth.authorize({})

    def test_oauth_authorize_rejects_bad_token(self):
        auth = OAuth21Authorizer(
            metadata=OAuth21ResourceMetadata("r", ["s"]),
            allowed_tokens=["tok-1"],
        )
        with pytest.raises(PermissionError):
            auth.authorize({"Authorization": "Bearer wrong"})

    def test_oauth_authorize_accepts_valid(self):
        auth = OAuth21Authorizer(
            metadata=OAuth21ResourceMetadata("r", ["s"]),
            allowed_tokens=["tok-1"],
        )
        auth.authorize({"Authorization": "Bearer tok-1"})  # no raise

    def test_well_known_handler(self):
        auth = OAuth21Authorizer(
            metadata=OAuth21ResourceMetadata(
                "https://x/mcp", ["https://auth/x"], scopes_supported=["read"]
            ),
        )
        payload = make_well_known_handler(auth)()
        assert payload["resource"] == "https://x/mcp"
        assert payload["scopes_supported"] == ["read"]

    def test_task_registry_lifecycle(self):
        reg = MCPTaskRegistry()
        task = reg.create("arxiv_search", {"q": "x"})
        assert task.status == "pending"
        reg.start(task.id)
        assert reg.get(task.id).status == "running"
        reg.complete(task.id, {"n": 5})
        completed = reg.get(task.id)
        assert completed.status == "completed"
        assert completed.result == {"n": 5}

    def test_task_retry_then_fail(self):
        reg = MCPTaskRegistry()
        t = reg.create("x", {}, max_retries=2)
        reg.fail(t.id, "first", retryable=True)
        assert reg.get(t.id).status == "pending"
        assert reg.get(t.id).retries == 1
        reg.fail(t.id, "second", retryable=True)
        assert reg.get(t.id).retries == 2
        reg.fail(t.id, "third", retryable=True)
        # After exhausting retries the task is marked failed
        assert reg.get(t.id).status == "failed"

    def test_task_list_filter(self):
        reg = MCPTaskRegistry()
        t1 = reg.create("a", {})
        t2 = reg.create("b", {})
        reg.complete(t2.id, None)
        pending = reg.list(status="pending")
        assert len(pending) == 1 and pending[0].id == t1.id

    def test_task_expiry(self):
        reg = MCPTaskRegistry(ttl_seconds=0)
        t = reg.create("x", {})
        # Force expiry check
        import time as _time
        _time.sleep(0.01)
        got = reg.get(t.id)
        assert got.status == "expired"

    def test_prune_expired(self):
        reg = MCPTaskRegistry(ttl_seconds=0)
        _ = reg.create("x", {})
        _ = reg.create("y", {})
        import time as _time
        _time.sleep(0.01)
        # First .get marks them expired
        for t in reg.list():
            reg.get(t.id)
        n = reg.prune_expired()
        assert n == 2

    def test_unknown_task_raises(self):
        reg = MCPTaskRegistry()
        with pytest.raises(KeyError):
            reg.start("nope")
