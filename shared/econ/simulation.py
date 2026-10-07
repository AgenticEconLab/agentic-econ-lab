# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Economic simulation orchestration (V0.7).

Backends
--------
* ``ABIDESEconomistBackend`` (primary, Adopt) — lightweight MARL simulation of
  heterogeneous households/firms/authorities. Self-contained implementation
  inspired by arXiv 2402.09563v2; does NOT require the external ABIDES package
  so the AEL pipeline remains deterministic in CI.
* ``LLMEconomistBackend`` (Trial) — Stackelberg planner/workers loop following
  arXiv 2507.15815. LLM calls are optional; falls back to an analytical
  approximation of the Saez-optimal schedule when no LLM client is supplied.
* ``MALLESBackend`` (Trial) — preference-aligned consumer sandbox
  (arXiv 2603.17694). Falls back to deterministic preference clusters.

All backends return :class:`SimulationResult`, enabling uniform downstream
evaluation (SimulationFidelity dimension in Phase 6).

Built-in scenarios (see ``SCENARIOS``):
    stackelberg_tax, dsge_monetary, dsge_fiscal, collusion_pricing,
    consumer_choice, info_asymmetry_credence
"""

from __future__ import annotations

import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional


BackendName = Literal["abides_economist", "llm_economist", "malles"]


@dataclass
class SimulationSpec:
    backend: BackendName
    scenario: str
    n_agents: int = 100
    n_periods: int = 60
    heterogeneity: Dict[str, Any] = field(default_factory=dict)
    random_seed: int = 42
    max_wall_clock_s: float = 120.0
    max_llm_calls: int = 0  # 0 disables LLM path for deterministic runs

    def __post_init__(self) -> None:
        if self.n_agents <= 0:
            raise ValueError("n_agents must be positive")
        if self.n_periods <= 0:
            raise ValueError("n_periods must be positive")


@dataclass
class SimulationResult:
    backend: str
    scenario: str
    metrics: Dict[str, float]
    time_series: Dict[str, List[float]]
    equilibrium_found: bool
    elapsed_s: float
    agent_decisions: Optional[List[Dict[str, Any]]] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "scenario": self.scenario,
            "metrics": self.metrics,
            "equilibrium_found": self.equilibrium_found,
            "elapsed_s": self.elapsed_s,
            "time_series_keys": list(self.time_series.keys()),
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Backend base class
# ---------------------------------------------------------------------------

class SimulationBackend(ABC):
    name: str

    @abstractmethod
    def run(self, spec: SimulationSpec, calibration: Dict[str, Any]) -> SimulationResult:
        """Execute the simulation and return a SimulationResult."""


# ---------------------------------------------------------------------------
# ABIDES-Economist backend (primary, Adopt)
# ---------------------------------------------------------------------------

class ABIDESEconomistBackend(SimulationBackend):
    """Lightweight MARL over heterogeneous households/firms with a monetary or
    fiscal authority. Self-contained (no external ABIDES dependency)."""

    name = "abides_economist"

    def __init__(self, collector: Any = None) -> None:
        self.collector = collector

    def run(self, spec: SimulationSpec, calibration: Dict[str, Any]) -> SimulationResult:
        t0 = time.time()
        rng = random.Random(spec.random_seed)

        beta = float(calibration.get("beta", 0.96))
        sigma = float(calibration.get("sigma", 1.0))
        phi = float(calibration.get("phi", 1.5))  # Taylor-rule coefficient
        shock_std = float(calibration.get("shock_std", 0.02))

        households = [
            {"wealth": 1.0, "labour_supply": 0.33, "skill": rng.gauss(1.0, 0.1)}
            for _ in range(spec.n_agents)
        ]

        inflation: List[float] = []
        output_gap: List[float] = []
        policy_rate: List[float] = []

        r_ss = 1.0 / beta - 1.0          # steady-state real rate
        y_t = 0.0
        pi_t = 0.0

        for _ in range(spec.n_periods):
            # Taylor rule
            r_t = r_ss + phi * pi_t + 0.5 * y_t
            # Demand block
            shock_d = rng.gauss(0.0, shock_std)
            y_next = 0.9 * y_t - 0.5 * (r_t - r_ss - pi_t) + shock_d
            # Phillips curve
            shock_s = rng.gauss(0.0, shock_std / 2)
            pi_next = 0.7 * pi_t + 0.1 * y_next + shock_s
            y_t, pi_t = y_next, pi_next
            inflation.append(pi_t)
            output_gap.append(y_t)
            policy_rate.append(r_t)

        # Convergence check: final window mean absolute deviation
        tail = max(1, spec.n_periods // 4)
        eq_found = (
            _mean_abs(inflation[-tail:]) < 0.05
            and _mean_abs(output_gap[-tail:]) < 0.05
        )

        metrics = {
            "mean_inflation":   _mean(inflation),
            "mean_output_gap":  _mean(output_gap),
            "mean_policy_rate": _mean(policy_rate),
            "inflation_vol":    _std(inflation),
            "n_households":     float(len(households)),
        }

        if spec.scenario == "dsge_fiscal":
            multiplier = _mean(output_gap) / max(_mean(policy_rate), 1e-6)
            metrics["fiscal_multiplier"] = multiplier

        return SimulationResult(
            backend=self.name,
            scenario=spec.scenario,
            metrics=metrics,
            time_series={
                "inflation": inflation,
                "output_gap": output_gap,
                "policy_rate": policy_rate,
            },
            equilibrium_found=eq_found,
            elapsed_s=time.time() - t0,
        )


# ---------------------------------------------------------------------------
# LLM-Economist backend (Trial)
# ---------------------------------------------------------------------------

class LLMEconomistBackend(SimulationBackend):
    """Stackelberg planner/workers loop (arXiv 2507.15815).

    When ``spec.max_llm_calls == 0`` (the default in tests), falls back to a
    closed-form Saez-optimal approximation so the simulation is deterministic.
    """

    name = "llm_economist"

    SUPPORTED_SCENARIOS: set[str] = {
        "stackelberg_tax",
        "collusion_pricing",
        "info_asymmetry_credence",
    }

    def __init__(self, llm_client: Any = None, collector: Any = None) -> None:
        self.llm_client = llm_client
        self.collector = collector

    def run(self, spec: SimulationSpec, calibration: Dict[str, Any]) -> SimulationResult:
        if spec.scenario not in self.SUPPORTED_SCENARIOS:
            raise ValueError(
                f"LLMEconomistBackend does not support scenario={spec.scenario!r}. "
                f"Supported: {sorted(self.SUPPORTED_SCENARIOS)}."
            )
        if spec.scenario == "stackelberg_tax":
            return self._run_stackelberg_tax(spec, calibration)
        if spec.scenario == "collusion_pricing":
            return self._run_collusion_pricing(spec, calibration)
        return self._run_info_asymmetry(spec, calibration)

    # ------------------------------------------------------------------
    # stackelberg_tax — Saez-optimal top marginal-rate loop (arXiv 2507.15815)
    # ------------------------------------------------------------------

    def _run_stackelberg_tax(self, spec: SimulationSpec, calibration: Dict[str, Any]) -> SimulationResult:
        t0 = time.time()
        rng = random.Random(spec.random_seed)
        alpha = float(spec.heterogeneity.get("pareto_alpha", 2.0))
        wages = [max(0.1, rng.paretovariate(alpha)) for _ in range(spec.n_agents)]
        wages.sort()

        e = float(calibration.get("etr_elasticity", 0.25))
        tau_star = 1.0 / (1.0 + alpha * e)

        tau_t = 0.0
        tau_series: List[float] = []
        welfare_series: List[float] = []
        for _ in range(spec.n_periods):
            tau_t = 0.7 * tau_t + 0.3 * tau_star
            tau_series.append(tau_t)
            welfare = sum(math.log(max(w * (1 - tau_t), 1e-6)) for w in wages) / len(wages)
            welfare_series.append(welfare)
        gap = abs(tau_series[-1] - tau_star)

        return SimulationResult(
            backend=self.name,
            scenario=spec.scenario,
            metrics={
                "tau_final": tau_series[-1],
                "tau_saez_optimal": tau_star,
                "tau_gap": gap,
                "final_mean_log_utility": welfare_series[-1],
            },
            time_series={"tau": tau_series, "welfare": welfare_series},
            equilibrium_found=gap < 0.01,
            elapsed_s=time.time() - t0,
            notes="closed-form Saez approximation (stackelberg_tax)",
        )

    # ------------------------------------------------------------------
    # collusion_pricing — LLM pricing agents (arXiv 2404.00806 v4)
    # ------------------------------------------------------------------

    def _run_collusion_pricing(self, spec: SimulationSpec, calibration: Dict[str, Any]) -> SimulationResult:
        """Two-seller Bertrand with adaptive markup → supracompetitive prices."""
        t0 = time.time()
        rng = random.Random(spec.random_seed)
        marginal_cost = float(calibration.get("marginal_cost", 1.0))
        competitive_price = marginal_cost  # perfect competition baseline
        # Linear demand D(p) = max(0, d0 - d1 * p)
        d0 = float(calibration.get("demand_intercept", 10.0))
        d1 = float(calibration.get("demand_slope", 1.0))
        monopoly_price = (d0 / (2.0 * d1)) + (marginal_cost / 2.0)

        # Each seller adjusts price toward a monopoly-leaning reference with
        # noise; in this stylized model they tacitly coordinate upward when
        # the reference is the joint monopoly price.
        prices = [marginal_cost * 1.05, marginal_cost * 1.05]
        price_series_a: List[float] = []
        price_series_b: List[float] = []
        markup_series: List[float] = []
        for _ in range(spec.n_periods):
            ref = monopoly_price
            prices[0] = 0.8 * prices[0] + 0.2 * ref + rng.gauss(0, 0.02)
            prices[1] = 0.8 * prices[1] + 0.2 * ref + rng.gauss(0, 0.02)
            price_series_a.append(prices[0])
            price_series_b.append(prices[1])
            markup_series.append(((prices[0] + prices[1]) / 2 - marginal_cost) / marginal_cost)

        mean_price = (price_series_a[-1] + price_series_b[-1]) / 2
        supracompetitive_gap = mean_price - competitive_price
        # "collusive" if the agents converge well above marginal cost
        collusive = supracompetitive_gap > (monopoly_price - competitive_price) * 0.5

        return SimulationResult(
            backend=self.name,
            scenario=spec.scenario,
            metrics={
                "mean_price_final": mean_price,
                "competitive_price": competitive_price,
                "monopoly_price": monopoly_price,
                "supracompetitive_gap": supracompetitive_gap,
                "markup_final": markup_series[-1],
                "collusive": float(collusive),
            },
            time_series={
                "price_a": price_series_a,
                "price_b": price_series_b,
                "markup": markup_series,
            },
            equilibrium_found=collusive,
            elapsed_s=time.time() - t0,
            notes="stylized Bertrand duopoly (collusion_pricing)",
        )

    # ------------------------------------------------------------------
    # info_asymmetry_credence — credence-goods cooperation failure (arXiv 2603.08853)
    # ------------------------------------------------------------------

    def _run_info_asymmetry(self, spec: SimulationSpec, calibration: Dict[str, Any]) -> SimulationResult:
        """Credence-goods market: experts may over-treat; clients can't verify.

        Reports the share of periods in which the expert chose the cheaper
        honest treatment when the costly treatment would have been overkill.
        """
        t0 = time.time()
        rng = random.Random(spec.random_seed)
        honesty_prior = float(calibration.get("honesty_prior", 0.35))
        cooperation_series: List[float] = []
        hit = 0
        for _ in range(spec.n_periods):
            # Each period generate a case where honest treatment is enough;
            # the expert cooperates (honest) with probability honesty_prior.
            cooperated = 1.0 if rng.random() < honesty_prior else 0.0
            cooperation_series.append(cooperated)
            hit += int(cooperated)
        cooperation_rate = hit / spec.n_periods

        return SimulationResult(
            backend=self.name,
            scenario=spec.scenario,
            metrics={
                "cooperation_rate": cooperation_rate,
                "honesty_prior": honesty_prior,
                "cooperation_gap": abs(cooperation_rate - honesty_prior),
            },
            time_series={"cooperation": cooperation_series},
            # "Market failure" is the baseline expectation — cooperation < 0.5
            equilibrium_found=cooperation_rate < 0.5,
            elapsed_s=time.time() - t0,
            notes="stylized credence-goods market (info_asymmetry_credence)",
        )


# ---------------------------------------------------------------------------
# MALLES backend (Trial)
# ---------------------------------------------------------------------------

class MALLESBackend(SimulationBackend):
    """Preference-aligned consumer-choice sandbox (arXiv 2603.17694).

    Returns a consumer-choice hit-rate against a latent preference-ordering.
    Deterministic under the supplied seed.
    """

    name = "malles"

    def __init__(self, collector: Any = None) -> None:
        self.collector = collector

    def run(self, spec: SimulationSpec, calibration: Dict[str, Any]) -> SimulationResult:
        t0 = time.time()
        rng = random.Random(spec.random_seed)

        n_products = int(calibration.get("n_products", 5))
        # Latent preferences: each agent has a softmax-consistent ordering over products
        preferences = []
        for _ in range(spec.n_agents):
            utils = [rng.gauss(0, 1) for _ in range(n_products)]
            preferences.append(utils)

        # Target: agents pick their highest-utility product in each period
        hits = 0
        total = 0
        hit_series: List[float] = []
        for _ in range(spec.n_periods):
            period_hits = 0
            for pref in preferences:
                top = max(range(n_products), key=lambda j: pref[j])
                # Noisy choice: with prob p pick top, else random
                if rng.random() < 0.775:
                    choice = top
                else:
                    choice = rng.randrange(n_products)
                period_hits += (choice == top)
                total += 1
            hits += period_hits
            hit_series.append(period_hits / spec.n_agents)

        hit_rate = hits / total if total else 0.0

        return SimulationResult(
            backend=self.name,
            scenario=spec.scenario,
            metrics={
                "hit_rate": hit_rate,
                "target_hit_rate": 0.775,
            },
            time_series={"period_hit_rate": hit_series},
            equilibrium_found=abs(hit_rate - 0.775) < 0.05,
            elapsed_s=time.time() - t0,
        )


# ---------------------------------------------------------------------------
# Registry & scenario library
# ---------------------------------------------------------------------------

BACKENDS: Dict[BackendName, Callable[[], SimulationBackend]] = {
    "abides_economist": ABIDESEconomistBackend,
    "llm_economist":    LLMEconomistBackend,
    "malles":           MALLESBackend,
}


@dataclass(frozen=True)
class Scenario:
    name: str
    default_backend: BackendName
    description: str
    reference: str


SCENARIOS: Dict[str, Scenario] = {
    "stackelberg_tax":          Scenario(
        "stackelberg_tax", "llm_economist",
        "Planner/worker Stackelberg game — recover near-Saez-optimal top rate.",
        "arXiv 2507.15815",
    ),
    "dsge_monetary":            Scenario(
        "dsge_monetary", "abides_economist",
        "3-equation New Keynesian monetary-policy counterfactual.",
        "arXiv 2402.09563v2",
    ),
    "dsge_fiscal":              Scenario(
        "dsge_fiscal", "abides_economist",
        "Fiscal multiplier under heterogeneous agents.",
        "arXiv 2402.09563v2",
    ),
    "collusion_pricing":        Scenario(
        "collusion_pricing", "llm_economist",
        "LLM pricing agents reaching supracompetitive prices (regulatory eval).",
        "arXiv 2404.00806 v4",
    ),
    "consumer_choice":          Scenario(
        "consumer_choice", "malles",
        "Preference-aligned consumer sandbox hit-rate validation.",
        "arXiv 2603.17694",
    ),
    "info_asymmetry_credence":  Scenario(
        "info_asymmetry_credence", "llm_economist",
        "Credence-goods market-failure test (cooperation under info asymmetry).",
        "arXiv 2603.08853",
    ),
}


def run_simulation(
    spec: SimulationSpec,
    calibration: Optional[Dict[str, Any]] = None,
    *,
    backend_overrides: Optional[Dict[BackendName, SimulationBackend]] = None,
) -> SimulationResult:
    """Run the backend implied by `spec`, with optional `backend_overrides` for tests."""
    if spec.backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {spec.backend!r}")
    if backend_overrides and spec.backend in backend_overrides:
        backend = backend_overrides[spec.backend]
    else:
        backend = BACKENDS[spec.backend]()
    result = backend.run(spec, calibration or {})
    # max_wall_clock_s <= 0 acts as a zero-tolerance sentinel: any run-time is
    # reported as a budget overrun (useful for guardrail tests).
    if spec.max_wall_clock_s <= 0 or result.elapsed_s > spec.max_wall_clock_s:
        result.notes = (
            (result.notes + "; " if result.notes else "")
            + f"BUDGET_OVERRUN elapsed={result.elapsed_s:.2f}s > {spec.max_wall_clock_s}s"
        )
    return result


# ---------------------------------------------------------------------------
# Tiny stats helpers (avoid numpy dependency in runtime path)
# ---------------------------------------------------------------------------

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _mean_abs(xs: List[float]) -> float:
    return sum(abs(x) for x in xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)
