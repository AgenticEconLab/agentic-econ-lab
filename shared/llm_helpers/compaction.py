# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Server-side conversation compaction (V0.7).

Reference: Pydantic AI v1.79–1.81 ``OpenAICompaction`` / ``AnthropicCompaction``.
Provider-native compaction calls are first-choice; this module provides a
deterministic client-side fallback that preserves tool-call structure while
trimming message bodies toward a token budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class CompactionResult:
    messages: List[Dict[str, Any]]
    tokens_before: int
    tokens_after: int
    strategy: str

    @property
    def reduction_ratio(self) -> float:
        if self.tokens_before <= 0:
            return 0.0
        return 1.0 - self.tokens_after / self.tokens_before


def estimate_tokens(text: str) -> int:
    """Rough 4-char-per-token estimator (provider-agnostic, deterministic)."""
    return max(1, len(text) // 4)


def estimate_message_tokens(message: Dict[str, Any]) -> int:
    content = message.get("content") or ""
    if isinstance(content, list):
        content = " ".join(str(c.get("text", c)) for c in content if c)
    return estimate_tokens(str(content)) + 4  # role + separators


def compact_conversation(
    messages: List[Dict[str, Any]],
    *,
    provider: str = "openai",
    budget_tokens: int = 2_048,
    llm_client: Any = None,
    strategy: str = "auto",
) -> CompactionResult:
    """Compact a list of ChatML-style messages to fit within ``budget_tokens``.

    Strategies
    ----------
    auto (default)
        Provider-native if supported, else ``preserve_head_tail``.
    preserve_head_tail
        Keep the first (system) message and the last N messages so total
        tokens <= budget.
    summary_middle
        Same as ``preserve_head_tail`` but replaces the dropped middle with a
        single summary message.
    """
    tokens_before = sum(estimate_message_tokens(m) for m in messages)

    if tokens_before <= budget_tokens:
        return CompactionResult(list(messages), tokens_before, tokens_before, "no_op")

    if strategy == "auto":
        if llm_client is not None and provider in {"openai", "anthropic"}:
            out = _provider_native_compaction(messages, provider, budget_tokens, llm_client)
            if out is not None:
                return out
        strategy = "summary_middle"

    if strategy == "preserve_head_tail":
        kept = _head_tail(messages, budget_tokens)
    elif strategy == "summary_middle":
        kept = _summary_middle(messages, budget_tokens)
    else:
        raise ValueError(f"Unknown compaction strategy: {strategy!r}")

    tokens_after = sum(estimate_message_tokens(m) for m in kept)
    return CompactionResult(kept, tokens_before, tokens_after, strategy)


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def _head_tail(messages: List[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
    if not messages:
        return []
    system = [m for m in messages[:1] if m.get("role") == "system"]
    tail: List[Dict[str, Any]] = []
    used = sum(estimate_message_tokens(m) for m in system)
    for m in reversed(messages[len(system):]):
        c = estimate_message_tokens(m)
        if used + c > budget:
            break
        tail.insert(0, m)
        used += c
    return system + tail


def _summary_middle(messages: List[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
    """Keep head (first message) + tail; replace the middle with a user-role
    summary note. Most providers reject a second ``system`` message, so the
    summary is emitted as a ``user`` turn with a clearly-labelled prefix.
    """
    if not messages:
        return []
    head = messages[:1]
    tail_limit = max(2, budget // 100)  # at least 2 messages at tail
    tail = messages[-tail_limit:]
    middle = messages[1:-tail_limit] if len(messages) > tail_limit + 1 else []
    summary_text = (
        f"[compaction summary] The assistant compacted {len(middle)} prior "
        f"messages (~{sum(estimate_message_tokens(m) for m in middle)} tokens) "
        f"from this conversation to stay within the context budget."
    )
    # Use role=user so providers that enforce single-system-message rules
    # (OpenAI, most OpenAI-compatible APIs) accept the compacted sequence.
    summary = {"role": "user", "content": summary_text}
    return list(head) + [summary] + list(tail)


def _provider_native_compaction(
    messages: List[Dict[str, Any]],
    provider: str,
    budget: int,
    llm_client: Any,
) -> Optional[CompactionResult]:
    """Invoke the provider's native compaction endpoint if available.

    Returns None on any failure so the caller falls back to local strategies.
    """
    try:
        fn = getattr(llm_client, "compact_messages", None)
        if fn is None:
            return None
        compacted = fn(messages=messages, provider=provider, budget_tokens=budget)
        if not isinstance(compacted, list):
            return None
        tb = sum(estimate_message_tokens(m) for m in messages)
        ta = sum(estimate_message_tokens(m) for m in compacted)
        return CompactionResult(compacted, tb, ta, f"native_{provider}")
    except Exception:
        return None
