# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Tests for shared.parallel.parallel_map (concurrent LLM-call helper)."""
import time
from shared.parallel import parallel_map


def test_preserves_order_and_runs_concurrently():
    def slow(x):
        time.sleep(0.15)
        return x * 2
    t0 = time.time()
    out = parallel_map(slow, list(range(8)), max_workers=8)
    assert out == [0, 2, 4, 6, 8, 10, 12, 14]          # order preserved
    assert time.time() - t0 < 0.6                        # 8x0.15s=1.2s sequential -> concurrent


def test_exception_returned_in_place_not_raised():
    def maybe(x):
        if x == 2:
            raise ValueError("boom")
        return x
    out = parallel_map(maybe, [0, 1, 2, 3], max_workers=4)
    assert out[0] == 0 and out[1] == 1 and out[3] == 3
    assert isinstance(out[2], ValueError)


def test_sequential_fallback_for_single_item():
    assert parallel_map(lambda x: x + 1, [41]) == [42]
