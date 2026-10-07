# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Simulation-based (SMM) calibration — the fallback for models the closed-form harness refuses.

The deterministic harness (`harness.run`) parses equations into sympy and derives *closed-form*
structural moments. Agent-based / deep-RL / stochastic-dynamic models have NO closed-form moments —
their behaviour is only defined by *running* the model. Such models are calibrated by the
**Simulated Method of Moments** (SMM): simulate at a parameter vector, compute moments from the
simulated paths, and search parameters to match cited external targets. This module provides:

  * ``classify_executability`` — is the model numerically runnable at all, or does it contain
    *learned / opaque* components (a neural encoder, an MLP policy, an LLM call, an argmax over a
    trained policy, an undefined market-clearing operator)?  Those have no numerical realization
    without first TRAINING them, so no simulator — and hence no SMM — can be built automatically.
  * ``smm_estimate`` — a real SMM optimizer: minimize the weighted distance between *simulated*
    moments and external targets over the free parameters (reuses the harness's bounded LSQ).
  * ``try_simulation_calibrate`` — routing: gate → build a simulator → SMM, returning a
    ``FitOutcome`` with an HONEST verdict (``simulation_calibrated`` on success; otherwise
    ``uncalibratable`` with a PRECISE reason). Never a fabricated fit.

Honesty boundary: automatic construction of a runnable simulator from arbitrary symbolic equations
is NOT attempted here (it would be fabrication-prone) — ``build_moment_simulator`` returns None, and
a concrete simulator is expected to be supplied (e.g. by wiring the V0.7 SimulationStage backends).
The SMM engine itself is real and exercised via an injected simulator.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .types import FitOutcome, MomentResult
from . import estimate

# Operators/keywords that make a model NON-executable without training: a learned function has no
# numerical value until its weights exist. Matched case-insensitively against each equation's text.
# Ambiguous tokens are ANCHORED to their ML context — bare "attention"/"gradient" are standard
# economics vocabulary (rational inattention, wage gradient) and must NOT trigger the gate.
_OPAQUE_PATTERNS = [
    r"encoder", r"decoder", r"\bmlp\b", r"softmax", r"transformer",
    r"attention\s*\(", r"\battention (mechanism|layer|head|weights|network)", r"self-attention",
    r"\bneural\b", r"f_?\{?\\?text\{?\s*llm", r"\bllm\b", r"arg\s*max",
    r"\\?pi_?\\?theta", r"policy\s*network", r"q[-_ ]?network", r"value\s*network",
    r"(pre-?)?trained (model|network|policy|agent)", r"learned (policy|representation|parameters|function)",
    r"clearingprice", r"\bembedding",
    r"gradient (descent|ascent|update|step|clipping)", r"policy gradient", r"stochastic gradient",
    r"backprop", r"\breplay\s*buffer", r"\bactor[-\s]?critic", r"\bppo\b", r"\bdqn\b",
]
_OPAQUE_RE = re.compile("|".join(_OPAQUE_PATTERNS), re.IGNORECASE)


def _eq_text(e: Any) -> str:
    """Best-effort plaintext of one equation (dict or object), across the fields models use."""
    if not isinstance(e, dict):
        e = getattr(e, "__dict__", {}) or {}
    parts = []
    for k in ("equation_latex", "latex", "equation_plain", "plain", "equation",
              "lhs", "rhs", "left_hand_side", "right_hand_side", "equation_name", "name",
              "description"):
        v = e.get(k)
        if isinstance(v, str):
            parts.append(v)
    return " ".join(parts)


def classify_executability(equations: Sequence[Any]) -> Tuple[bool, str]:
    """Return ``(executable, reason)``.

    A model is NOT executable if ANY equation references a learned/opaque operator — a neural net,
    an LLM call, a trained/learned policy, an argmax over such a policy, or an undefined algorithmic
    operator (e.g. a bare ``ClearingPrice``). Those need training before they can be simulated, so
    an automatic SMM cannot proceed. ``reason`` names the offending tokens for an honest verdict."""
    hits = set()
    for e in equations or []:
        for m in _OPAQUE_RE.finditer(_eq_text(e)):
            hits.add(m.group(0).lower().strip("\\ "))
    if hits:
        return False, (
            "requires_training: model has learned/opaque components ("
            + ", ".join(sorted(hits))
            + ") with no numerical realization — needs training + simulation-based estimation, "
            "not closed-form moment matching or automatic SMM"
        )
    return True, ""


def _avg_moments(
    simulate_moments: Callable[[Dict[str, float], int], Dict[str, float]],
    params: Dict[str, float], moment_keys: List[str], n_reps: int, base_seed: int,
) -> Dict[str, float]:
    """Average simulated moments over ``n_reps`` fixed seeds — tames Monte-Carlo noise so the SMM
    objective is smooth enough for least-squares (common-random-numbers across the parameter search)."""
    acc = {k: 0.0 for k in moment_keys}
    cnt = {k: 0 for k in moment_keys}
    for r in range(n_reps):
        sm = simulate_moments(params, base_seed + r) or {}
        for k in moment_keys:
            v = sm.get(k)
            if isinstance(v, (int, float)) and math.isfinite(v):
                acc[k] += float(v)
                cnt[k] += 1
    return {k: (acc[k] / cnt[k] if cnt[k] else float("nan")) for k in moment_keys}


def _moment_match(avg: Dict[str, float], matched: List[str], target_book: Dict[str, Any]) -> Optional[float]:
    """Descriptive fit: 1 - mean relative error of simulated moments vs their targets (in [0, 1])."""
    rels = []
    for k in matched:
        mv = avg.get(k)
        if mv is None or not math.isfinite(mv):
            continue
        tv = float(target_book[k].value)
        denom = abs(tv) if abs(tv) > 1e-9 else 1.0
        rels.append(abs(mv - tv) / denom)
    if not rels:
        return None
    return round(max(0.0, 1.0 - sum(rels) / len(rels)), 4)


def smm_estimate(
    simulate_moments: Callable[[Dict[str, float], int], Dict[str, float]],
    moment_keys: List[str],
    target_book: Dict[str, Any],
    free_params: List[str],
    x0: Sequence[float],
    bounds: Dict[str, Tuple[float, float]],
    *,
    n_reps: int = 6,
    base_seed: int = 1000,
) -> Tuple[Optional[Dict[str, float]], Optional[float]]:
    """Simulated Method of Moments.

    Minimize the weighted distance ``(simulated_moment - target)/se`` between the model's SIMULATED
    moments and the cited external targets, over ``free_params`` (within ``bounds``).
    ``simulate_moments(params, seed) -> {moment_key: value}`` runs the model. Returns
    ``(estimated {param: value}, moment_match_score)`` or ``(None, None)`` if it can't proceed /
    the optimizer fails. Never raises through to the caller as a fabricated fit."""
    matched = [k for k in moment_keys if k in target_book]
    if not matched or not free_params:
        return None, None
    lo = [bounds[p][0] for p in free_params]
    hi = [bounds[p][1] for p in free_params]

    def resid(x):
        params = {p: float(x[i]) for i, p in enumerate(free_params)}
        avg = _avg_moments(simulate_moments, params, matched, n_reps, base_seed)
        out = []
        for k in matched:
            tr = target_book[k]
            mv = avg[k]
            se = tr.std_error if (getattr(tr, "std_error", 0) and tr.std_error > 0) else max(abs(tr.value) * 0.1, 1e-3)
            out.append((mv - tr.value) / se if math.isfinite(mv) else 1e3)
        return out

    opt, ok = estimate.estimate(free_params, list(x0), lo, hi, resid)
    if not ok:
        return None, None
    # Tighter final evaluation (more reps) for the reported fit.
    avg = _avg_moments(simulate_moments, opt, matched, max(n_reps * 2, 12), base_seed)
    return opt, _moment_match(avg, matched, target_book)


def build_moment_simulator(formal_model: Dict[str, Any], calibrated_parameters: List[Any]):
    """Build a runnable moment-simulator for a model the closed-form path refused.

    Returns ``(simulate_moments, moment_keys, free_params, x0, bounds)`` or ``None``.

    Delegates to the runnable-spine registry (``sim_backends`` — Tier-2 worked instances of the
    simulation spine): a matching instance
    returns a real simulator; otherwise None -> honest ``uncalibratable``. We do NOT try to
    compile an arbitrary symbolic model into a simulator (fabrication-prone); coverage grows
    mechanism-first (Tier 0/1), with new instances only under the Tier-2 admission contract."""
    try:
        from .sim_backends import build_from_registry
        return build_from_registry(formal_model, calibrated_parameters)
    except Exception:
        return None


def _canon(sym: str) -> str:
    from .archetypes import canonical_symbol
    return canonical_symbol(sym)


def _declared_ranges(formal_model: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    from .archetypes import _parse_range
    out = {}
    for p in formal_model.get("parameters") or []:
        g = (lambda k: p.get(k)) if isinstance(p, dict) else (lambda k: getattr(p, k, None))
        sym, rng = g("parameter_symbol") or g("symbol"), _parse_range(g("typical_range"))
        if sym and rng and rng[0] < rng[1]:
            out[_canon(sym)] = rng
    return out


def _uncal(reason: str, coverage: Dict[str, int]) -> FitOutcome:
    return FitOutcome(calibration_status="uncalibratable", fit_score=None,
                      uncalibratable_reason=reason, coverage=coverage)


def try_simulation_calibrate(
    formal_model: Dict[str, Any],
    calibrated_parameters: List[Any],
    *,
    simulator_builder: Optional[Callable] = None,
    target_book: Optional[Dict[str, Any]] = None,
) -> FitOutcome:
    """Route a closed-form-refused model to simulation-based calibration.

    Gate (executability) → build a simulator → SMM. Always returns a ``FitOutcome``:
    ``simulation_calibrated`` (simulator of the model's own equations) or ``archetype_calibrated``
    (registered canonical stand-in; only the shared parameters are estimated) on success, else ``uncalibratable`` with a PRECISE, honest reason
    (learned components need training / no runnable simulator / no external target / SMM did not
    converge). ``simulator_builder`` and ``target_book`` are injectable for testing."""
    equations = formal_model.get("equations") or []
    n_eqs = len(equations)

    # Try the runnable-archetype registry FIRST: an archetype match (already conservative —
    # wording + the identifying parameter) is our chosen estimator for that model class, and
    # must not be vetoed by opaque tokens in unrelated equations (the gate false-positived on
    # standard economics vocabulary like rational in"attention" / wage "gradient"). The
    # executability gate now only generates the PRECISE reason when no simulator was built.
    # A registry match simulates a canonical STAND-IN (archetype), not the model's own equations;
    # an injected builder is taken to simulate the model itself.
    sim_name, is_archetype = "", False
    if simulator_builder is not None:
        spec = simulator_builder(formal_model, calibrated_parameters)
        sim_name = getattr(simulator_builder, "__name__", "injected_simulator")
    else:
        try:
            from .sim_backends import match_registry
            hit = match_registry(formal_model, calibrated_parameters)
        except Exception:
            hit = None
        spec = hit[1] if hit else None
        if hit:
            sim_name, is_archetype = hit[0], True
    if spec is None:
        executable, reason = classify_executability(equations)
        if not executable:
            return _uncal(reason, {"total_eqs": n_eqs, "simulation_attempted": 1, "executable": 0})
        return _uncal(
            "requires_simulation_based_estimation: model is stochastic/dynamic (no closed-form "
            "structural moments) — needs a runnable simulator (e.g. via SimulationStage) for SMM; "
            "automatic construction from symbolic equations is not supported",
            {"total_eqs": n_eqs, "simulation_attempted": 1, "simulator_built": 0})

    simulate_moments, moment_keys, free, x0, bounds = spec

    # The stand-in's admissible range must agree with what the model declares for the parameter
    # it is mapped to: intersect the two and decline when they do not overlap. The stand-in's
    # optimum does not depend on the model, so without this every model received the same
    # value (phi_pi = 2.361, savings = 0.207), mostly outside its own declared range.
    declared = _declared_ranges(formal_model)
    bounds = dict(bounds)
    if is_archetype:
        for p in free:
            rng = declared.get(_canon(p))
            if rng is None:
                continue
            lo, hi = max(bounds[p][0], rng[0]), min(bounds[p][1], rng[1])
            if lo >= hi:
                return _uncal(
                    f"stand_in_range_mismatch: the stand-in '{sim_name}' admits {p} in "
                    f"[{bounds[p][0]}, {bounds[p][1]}], the model declares [{rng[0]}, {rng[1]}]",
                    {"total_eqs": n_eqs, "method": "SMM", "simulator_built": 1})
            bounds[p] = (lo, hi)
        x0 = [min(max(x, bounds[p][0]), bounds[p][1]) for x, p in zip(x0, free)]

    if target_book is None:
        from . import targets as targets_mod
        target_book = targets_mod.load_targets()

    matched = [k for k in moment_keys if k in target_book]
    if not matched:
        return _uncal(
            "no_external_target_for_simulated_moments: the model's simulated moments have no "
            "matching cited target in the target book",
            {"total_eqs": n_eqs, "simulated_moments": len(moment_keys), "moments_with_target": 0})

    opt, match = smm_estimate(simulate_moments, matched, target_book, free, x0, bounds)
    if opt is None:
        return _uncal("smm_estimation_failed: optimizer did not converge",
                      {"total_eqs": n_eqs, "method": "SMM", "moments_with_target": len(matched)})
    from .harness import at_bound
    stuck = at_bound(opt, bounds)
    if stuck:
        return _uncal(
            f"target_outside_declared_range: SMM on '{sim_name}' stopped at a bound of "
            + ", ".join(f"{p} [{bounds[p][0]}, {bounds[p][1]}]" for p in stuck)
            + " without reproducing the target",
            {"total_eqs": n_eqs, "method": "SMM", "moments_with_target": len(matched),
             "bound_active_params": len(stuck)})

    # Per-moment results at the estimated parameters (same seeds as the reported match score), so
    # downstream counts of targets and matched moments agree with the fit score.
    avg = _avg_moments(simulate_moments, opt, matched, 12, 1000)
    results = []
    for k in matched:
        tr = target_book[k]
        mv = avg.get(k)
        ok = mv is not None and math.isfinite(mv)
        denom = abs(tr.value) if abs(tr.value) > 1e-9 else 1.0
        results.append(MomentResult(
            moment_key=k, computed=(float(mv) if ok else None), target=float(tr.value),
            std_error=getattr(tr, "std_error", None),
            rel_error=(abs(mv - tr.value) / denom if ok else None),
            free_params=list(free), role=("over_id" if len(matched) > len(free) else "pin"),
            source_citation=getattr(tr, "source_citation", "")))

    return FitOutcome(
        calibration_status=("archetype_calibrated" if is_archetype else "simulation_calibrated"),
        fit_score=match, moment_match_score=match,
        moments=results,
        degrees_of_freedom=len(matched) - len(free),
        estimated=True, estimated_params=opt,
        coverage={"total_eqs": n_eqs, "method": "SMM", "moments_matched": len(matched),
                  "free_params": len(free)},
        simulator=sim_name, simulator_is_archetype=is_archetype,
    )
