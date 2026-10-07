# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Run independent I/O-bound tasks (LLM calls) concurrently.

vLLM batches concurrent requests on the GPU for ~the same wall-time as a single one
(measured on A100: 8 concurrent = 7.5x aggregate throughput at ~the same latency). The agent
pipeline issues many independent LLM calls inside per-item loops *sequentially*, leaving the
GPU ~90% idle. ``parallel_map`` runs them through a thread pool — the calls block on the vLLM
HTTP endpoint, so the GIL is released during the wait and threads give true concurrency — to
exploit that batching with no change to the serving stack.

Concurrency is capped by AEL_LLM_CONCURRENCY (default 8). Order is preserved. A per-item
exception is returned in place of its result (not raised) so one failure can't drop the batch.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Optional

DEFAULT_WORKERS = max(1, int(os.getenv("AEL_LLM_CONCURRENCY", "8")))


def _safe(fn: Callable[[Any], Any], item: Any) -> Any:
    try:
        return fn(item)
    except Exception as e:  # keep the batch alive; caller can filter exceptions
        return e


def parallel_map(fn: Callable[[Any], Any], items, max_workers: Optional[int] = None) -> List[Any]:
    """Apply ``fn`` to each item concurrently; return results in input order.

    Falls back to a plain sequential map for <=1 item or workers<=1 (no thread overhead).
    """
    items = list(items)
    workers = DEFAULT_WORKERS if max_workers is None else max(1, max_workers)
    if len(items) <= 1 or workers <= 1:
        return [_safe(fn, it) for it in items]
    workers = min(workers, len(items))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ael-llm") as ex:
        return list(ex.map(lambda it: _safe(fn, it), items))
