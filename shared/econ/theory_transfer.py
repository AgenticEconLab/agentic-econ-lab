# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Theory-guided transfer learning trainer (V0.7).

Implements the Chib-Tan 90/10 recipe (arXiv 2512.21031):
  1. Pre-train a small forecasting transformer on *synthetic* DSGE trajectories.
  2. Mix in 10% empirical data for the final pre-training pass.
  3. Fine-tune on empirical data only.

This module ships a numpy-free reference implementation: the "transformer" is
a per-variable ridge-regression predictor with percentile tokenization. It
captures the *recipe* (synthetic pretrain → empirical fine-tune) and the 90/10
mix ratio without committing to a heavyweight ML dependency. Teams that want a
full neural implementation can subclass :class:`TheoryTransferTrainer` and
override ``_fit_block`` / ``_predict`` while keeping the public API stable.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple


Tokenization = Literal["percentile", "plain"]
Architecture = Literal["per_variable", "shared"]


@dataclass
class TheoryTransferConfig:
    pretrain_batches: int = 100
    pretrain_size_synthetic: int = 9_000
    pretrain_size_empirical: int = 1_000
    finetune_steps: int = 500
    finetune_lr: float = 1e-2
    ridge_lambda: float = 0.1
    tokenization: Tokenization = "percentile"
    architecture: Architecture = "per_variable"
    n_quantile_bins: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        if self.pretrain_size_synthetic + self.pretrain_size_empirical == 0:
            raise ValueError("pretrain sizes cannot both be zero")
        if self.finetune_steps < 0:
            raise ValueError("finetune_steps must be non-negative")


@dataclass
class TheoryTransferModel:
    """Fitted model produced by TheoryTransferTrainer.

    The model is a dictionary of per-variable ridge regressors, each mapping a
    window of the variable's history (optionally percentile-tokenised) to the
    next-period value.
    """
    variables: List[str]
    window: int
    coeffs: Dict[str, List[float]] = field(default_factory=dict)
    tokenization: Tokenization = "percentile"
    quantile_edges: Dict[str, List[float]] = field(default_factory=dict)


class TheoryTransferTrainer:
    """Chib-Tan 90/10 trainer."""

    def __init__(
        self,
        config: Optional[TheoryTransferConfig] = None,
        collector: Any = None,
        *,
        window: int = 4,
    ) -> None:
        self.config = config or TheoryTransferConfig()
        self.collector = collector
        self.window = window
        self._rng = random.Random(self.config.seed)

    # ------------------------------------------------------------------
    # Public training API
    # ------------------------------------------------------------------

    def pretrain(
        self,
        synthetic: Dict[str, List[float]],
        empirical: Dict[str, List[float]],
    ) -> TheoryTransferModel:
        """Build a mixed-batch pretrain set (90% synthetic + 10% empirical) and fit.

        The ratio is derived from ``config.pretrain_size_synthetic`` /
        ``config.pretrain_size_empirical`` so callers can adjust it.
        """
        variables = self._select_variables(synthetic)
        model = TheoryTransferModel(
            variables=variables,
            window=self.window,
            tokenization=self.config.tokenization,
        )

        # Percentile edges built from synthetic distribution (as in Chib-Tan).
        if self.config.tokenization == "percentile":
            for v in variables:
                model.quantile_edges[v] = _quantile_edges(
                    synthetic.get(v, []), self.config.n_quantile_bins,
                )

        # Build mixed training windows.
        mixed_pairs: List[Tuple[str, List[float], float]] = []
        n_syn = self.config.pretrain_size_synthetic
        n_emp = self.config.pretrain_size_empirical
        for v in variables:
            syn_pairs = _make_windows(synthetic.get(v, []), self.window)
            emp_pairs = _make_windows(empirical.get(v, []), self.window)
            syn_sample = _take_random(self._rng, syn_pairs, n_syn)
            emp_sample = _take_random(self._rng, emp_pairs, n_emp)
            for (x, y) in syn_sample + emp_sample:
                xt = self._tokenize(v, x, model)
                mixed_pairs.append((v, xt, y))

        # Fit per-variable ridge regressors.
        for v in variables:
            subset = [(x, y) for (var, x, y) in mixed_pairs if var == v]
            if not subset:
                continue
            model.coeffs[v] = _ridge_fit(
                [x for (x, _) in subset],
                [y for (_, y) in subset],
                self.config.ridge_lambda,
            )
        return model

    def finetune(
        self,
        model: TheoryTransferModel,
        empirical: Dict[str, List[float]],
    ) -> TheoryTransferModel:
        """Fine-tune on empirical data only (gradient steps with a small lr)."""
        if self.config.finetune_steps == 0:
            return model
        variables = [v for v in model.variables if v in empirical]
        for v in variables:
            pairs = _make_windows(empirical.get(v, []), self.window)
            if not pairs:
                continue
            beta = list(model.coeffs.get(v, [0.0] * (self.window + 1)))
            lr = self.config.finetune_lr
            for _ in range(self.config.finetune_steps):
                (x, y) = pairs[self._rng.randrange(len(pairs))]
                xt = self._tokenize(v, x, model)
                y_hat = _predict_one(xt, beta)
                err = y_hat - y
                # Gradient: d(loss)/d(beta_i) = err * xt_i (with intercept in beta[0])
                beta[0] -= lr * err
                for i, xi in enumerate(xt):
                    beta[i + 1] -= lr * err * xi
            model.coeffs[v] = beta
        return model

    def forecast(
        self,
        model: TheoryTransferModel,
        empirical: Dict[str, List[float]],
        horizon: int = 4,
    ) -> Dict[str, List[float]]:
        """Recursive one-step-ahead forecast for each variable, horizon steps."""
        out: Dict[str, List[float]] = {v: [] for v in model.variables}
        for v in model.variables:
            series = list(empirical.get(v, []))
            if len(series) < model.window:
                continue
            for _ in range(horizon):
                x = series[-model.window:]
                xt = self._tokenize(v, x, model)
                beta = model.coeffs.get(v, [0.0] * (len(xt) + 1))
                y_hat = _predict_one(xt, beta)
                out[v].append(y_hat)
                series.append(y_hat)
        return out

    def score(
        self,
        model: TheoryTransferModel,
        empirical: Dict[str, List[float]],
    ) -> Dict[str, float]:
        """Return one-step-ahead RMSE per variable."""
        result: Dict[str, float] = {}
        for v in model.variables:
            pairs = _make_windows(empirical.get(v, []), self.window)
            if not pairs:
                result[v] = float("inf")
                continue
            beta = model.coeffs.get(v, [0.0] * (self.window + 1))
            errs = []
            for (x, y) in pairs:
                xt = self._tokenize(v, x, model)
                y_hat = _predict_one(xt, beta)
                errs.append((y_hat - y) ** 2)
            result[v] = math.sqrt(sum(errs) / len(errs))
        return result

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _select_variables(self, synthetic: Dict[str, List[float]]) -> List[str]:
        skip = {"traj_id", "t"}
        return [k for k in synthetic if k not in skip]

    def _tokenize(self, variable: str, xs: Sequence[float], model: TheoryTransferModel) -> List[float]:
        if model.tokenization != "percentile":
            return list(xs)
        edges = model.quantile_edges.get(variable)
        if not edges:
            return list(xs)
        return [_bin(x, edges) for x in xs]


# ---------------------------------------------------------------------------
# helper math (numpy-free)
# ---------------------------------------------------------------------------

def _make_windows(series: Sequence[float], window: int) -> List[Tuple[List[float], float]]:
    n = len(series)
    if n <= window:
        return []
    return [(list(series[i:i + window]), float(series[i + window])) for i in range(n - window)]


def _take_random(rng: random.Random, pool: List[Tuple[List[float], float]], k: int) -> List[Tuple[List[float], float]]:
    if k <= 0 or not pool:
        return []
    if k >= len(pool):
        return list(pool)
    return rng.sample(pool, k)


def _quantile_edges(series: Sequence[float], n_bins: int) -> List[float]:
    if not series or n_bins <= 1:
        return []
    s = sorted(series)
    n = len(s)
    return [s[int(i / n_bins * n)] for i in range(1, n_bins)]


def _bin(x: float, edges: List[float]) -> float:
    """Return the bin index (0..len(edges)) for x against sorted edges."""
    for i, e in enumerate(edges):
        if x < e:
            return float(i)
    return float(len(edges))


def _ridge_fit(X: List[List[float]], y: List[float], lam: float) -> List[float]:
    """Closed-form ridge with intercept: beta = (X'X + lam*I)^-1 X'y.

    X rows are tokenised features; intercept is prepended. Returns
    [intercept, *slopes].
    """
    if not X:
        return []
    n, k = len(X), len(X[0])
    # Augment with intercept column
    A = [[1.0] + row for row in X]
    d = k + 1
    ata = [[0.0] * d for _ in range(d)]
    aty = [0.0] * d
    for i in range(n):
        row = A[i]
        yi = y[i]
        for a in range(d):
            aty[a] += row[a] * yi
            for b in range(d):
                ata[a][b] += row[a] * row[b]
    # Add ridge (skip intercept to avoid shrinking the mean)
    for a in range(1, d):
        ata[a][a] += lam
    # Solve via Gauss-Jordan
    M = [ata[r] + [aty[r]] for r in range(d)]
    for col in range(d):
        pivot = col
        for r in range(col + 1, d):
            if abs(M[r][col]) > abs(M[pivot][col]):
                pivot = r
        M[col], M[pivot] = M[pivot], M[col]
        if abs(M[col][col]) < 1e-12:
            return [0.0] * d
        pv = M[col][col]
        M[col] = [v / pv for v in M[col]]
        for r in range(d):
            if r == col:
                continue
            f = M[r][col]
            if f == 0.0:
                continue
            M[r] = [M[r][i] - f * M[col][i] for i in range(d + 1)]
    return [M[r][d] for r in range(d)]


def _predict_one(xt: Sequence[float], beta: Sequence[float]) -> float:
    """beta[0] + beta[1..] dot xt."""
    if not beta:
        return 0.0
    y = beta[0]
    for i, xi in enumerate(xt):
        if i + 1 < len(beta):
            y += beta[i + 1] * xi
    return y


# ---------------------------------------------------------------------------
# DRL-for-DSGE bounded rationality stub (BoE SWP 1142) — V0.7 Trial
# ---------------------------------------------------------------------------

def drl_dsge_not_implemented() -> None:
    """Stub pointer for BoE DRL-for-DSGE bounded-rationality item.

    Not implemented in this release. Raising here documents the
    expectation that callers gate on ``drl_dsge_enabled`` feature flag.
    """
    raise NotImplementedError(
        "DRL-for-DSGE is not implemented in this release"
    )
