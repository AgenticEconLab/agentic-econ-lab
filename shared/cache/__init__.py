# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Caching layer (V0.7 — Trial).

Semantic caching gateway in front of LLMClient. Feature-flagged (disabled
for K=5 multi-run evaluation; enabled for single-run developer workflows).
"""

from shared.cache.semantic_gateway import (
    CacheEntry,
    SemanticCacheGateway,
)

__all__ = [
    "CacheEntry",
    "SemanticCacheGateway",
]
