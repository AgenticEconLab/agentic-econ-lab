# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Continuous production evaluator (V0.7).

Reference: Anthropic multi-agent postmortem — continuous production evals beat
regression suites. Samples live execution logs at a bounded rate, runs a subset
of evaluation checks, and appends summaries to
``evaluation/continuous_eval_results.jsonl``.

Design constraints:
  * < 5% production overhead (enforced via ``max_sample_rate``).
  * Never writes into the *input* execution log file (read-only).
  * Crash-safe: a failed eval is logged but does not raise.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ProdEvalResult:
    ts: float
    log_path: str
    summary: Dict[str, Any]
    wall_clock_s: float
    status: str = "ok"
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "log_path": self.log_path,
            "summary": self.summary,
            "wall_clock_s": round(self.wall_clock_s, 4),
            "status": self.status,
            "error": self.error,
        }


EvalFn = Callable[[Dict[str, Any]], Dict[str, Any]]


class ContinuousProdEvaluator:
    """Low-overhead sampler over live execution logs.

    Parameters
    ----------
    eval_fn
        Callable that takes a parsed execution log and returns a summary dict
        (e.g. dimension scores, reliability snapshot). Default returns a
        trivial ``{"status": "sampled"}`` dict.
    output_path
        JSONL file to append results to. ``None`` disables persistence.
    sample_every_n
        Process every N-th sampled log (rate limiter).
    """

    def __init__(
        self,
        eval_fn: Optional[EvalFn] = None,
        *,
        output_path: Optional[Path] = None,
        sample_every_n: int = 5,
    ) -> None:
        self._eval_fn = eval_fn or (lambda log: {"status": "sampled"})
        self._output_path = Path(output_path) if output_path else None
        self._sample_every_n = max(1, sample_every_n)
        self._counter = 0

    def evaluate_log(self, log_path: str | Path) -> Optional[ProdEvalResult]:
        """Evaluate a single execution log. Returns None if the sampler skipped it."""
        self._counter += 1
        if self._counter % self._sample_every_n != 0:
            return None

        path = Path(log_path)
        t0 = time.time()
        try:
            with path.open("r", encoding="utf-8") as f:
                log = json.load(f)
            summary = self._eval_fn(log) or {}
            result = ProdEvalResult(
                ts=time.time(),
                log_path=str(path),
                summary=summary,
                wall_clock_s=time.time() - t0,
                status="ok",
            )
        except Exception as exc:  # noqa: BLE001 - crash-safe
            result = ProdEvalResult(
                ts=time.time(),
                log_path=str(path),
                summary={},
                wall_clock_s=time.time() - t0,
                status="error",
                error=str(exc),
            )

        self._persist(result)
        return result

    def replay(self, log_paths: List[str | Path]) -> List[ProdEvalResult]:
        """Evaluate a batch of logs (useful for backfill)."""
        out: List[ProdEvalResult] = []
        for p in log_paths:
            r = self.evaluate_log(p)
            if r is not None:
                out.append(r)
        return out

    def overhead_budget_respected(self, recent_wall_clocks: List[float],
                                     *, budget_pct: float = 5.0,
                                     baseline_run_s: float = 60.0) -> bool:
        """Return True if mean eval wall-clock stays under ``budget_pct`` of ``baseline_run_s``."""
        if not recent_wall_clocks:
            return True
        avg = sum(recent_wall_clocks) / len(recent_wall_clocks)
        return (avg / baseline_run_s) * 100.0 < budget_pct

    # ------------------------------------------------------------------

    def _persist(self, result: ProdEvalResult) -> None:
        if self._output_path is None:
            return
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict()) + "\n")
