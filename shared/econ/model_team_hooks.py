# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
ModelTeam post-stage hooks (V0.7).

Thin integration helpers that let each ModelTeam MasterOrchestrator call
into `shared/econ/` and `shared/verification/` with 1–3 lines, rather than
duplicating integration logic across 4 mode directories.

All hooks are gated on V0.7 feature flags and degrade gracefully when the
flag is off or required inputs are absent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from shared.feature_flags import is_enabled


# ---------------------------------------------------------------------------
# DASES Falsifier paired with Theorist
# ---------------------------------------------------------------------------

class _StrPromptLLM:
    """Adapter so AdversarialFalsifier's ``llm_client.invoke(prompt_str)`` reaches
    the shared LLMClient, whose ``invoke`` expects a messages list. Without this the
    round-2 LLM counterexample pass raises internally (caught -> None) and the
    Falsifier degenerates into a rule-only probe that can never validate an
    admissible counterexample."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def invoke(self, prompt: Any) -> Any:
        messages = ([{"role": "user", "content": prompt}]
                    if isinstance(prompt, str) else prompt)
        return self._client.invoke(messages)


def _default_falsifier_llm(*, collector: Any = None) -> Optional["_StrPromptLLM"]:
    """Build the configured LLMClient for the Falsifier's adversarial round.

    Returns ``None`` (-> report falsifier N/A) when no LLM is actually usable —
    i.e. a cloud provider with no API key and not a local (ollama/vllm) server.
    This mirrors ``LLMClient.invoke``'s own guard so we never report a meaningful
    verdict without actually running a counterexample probe."""
    try:
        from shared.llm import LLMClient
        client = LLMClient(temperature=0.3, collector=collector, agent_name="Falsifier")
    except Exception:
        return None
    if not getattr(client, "api_key", "") and getattr(client, "provider", "") not in ("ollama", "vllm"):
        return None
    return _StrPromptLLM(client)


def run_falsifier_pass(
    theory_output: Dict[str, Any],
    *,
    collector: Any = None,
    max_claims: int = 6,
    llm_client: Any = None,
) -> List[Dict[str, Any]]:
    """Run the DASES AdversarialFalsifier over claims in ``theory_output``.

    Extracts claim strings from assumptions, component descriptions, and
    propositions and produces one :class:`FalsifierReport` per claim. Gated
    on ``falsifier_enabled``. Returns empty list when disabled or when no
    claims are found.

    An LLM client is REQUIRED for the round-2 adversarial counterexample pass —
    without it the Falsifier is a rule-only probe (no admissibility-validated
    counterexamples). A configured client is built automatically when none is
    passed; if no LLM is usable each claim is reported with verdict ``"n/a"``
    rather than a misleading rule-only verdict. With an LLM, verdicts are
    meaningful and mixed (``falsified`` only for validated in-scope
    counterexamples; otherwise ``corroborated`` / ``needs_revision``).
    """
    if not is_enabled("falsifier_enabled"):
        return []
    try:
        from shared.verification import AdversarialFalsifier
    except Exception:
        return []

    claims = _collect_theory_claims(theory_output, max_claims=max_claims)
    if not claims:
        return []

    if llm_client is None:
        llm_client = _default_falsifier_llm(collector=collector)
    has_llm = llm_client is not None

    falsifier = AdversarialFalsifier(max_rounds=2, llm_client=llm_client, collector=collector)
    reports: List[Dict[str, Any]] = []
    for claim in claims:
        try:
            report = falsifier.run(claim)
            # No usable LLM -> the adversarial round never ran; reporting a real
            # verdict ('corroborated'/'falsified') would be unfounded, so surface
            # 'n/a' instead. With an LLM the verdict is meaningful: a counterexample
            # only falsifies when it is ADMISSIBLE (validated in-scope contradiction),
            # otherwise the claim is 'corroborated'/'needs_revision'.
            verdict = report.verdict if has_llm else "n/a"
            reports.append({
                "claim": report.claim,
                "verdict": verdict,
                "llm_assisted": has_llm,
                "rounds": report.rounds,
                "admissible_counterexamples": list(report.admissible_counterexamples),
                "rejected_counterexamples": list(report.rejected_counterexamples),
                "mechanistic_critique": report.mechanistic_critique,
                "suggested_revisions": list(report.suggested_revisions),
            })
        except Exception as exc:  # noqa: BLE001 - best-effort hook
            reports.append({"claim": claim, "error": str(exc), "verdict": "error"})
    return reports


def _collect_theory_claims(
    theory_output: Dict[str, Any],
    *,
    max_claims: int,
) -> List[str]:
    """Best-effort extraction of claim strings from TheoryStage output JSON."""
    claims: List[str] = []
    frameworks = theory_output.get("frameworks") or theory_output.get("theoretical_frameworks") or []
    # Claim-bearing dict fields, in priority order. Note the ModelTeam TheoryStage
    # schema uses ``assumption_statement`` (not ``statement``) for assumptions and
    # plain strings for testable_predictions — both must be picked up here or the
    # Falsifier silently finds zero claims and reports "0 evaluated".
    _STATEMENT_KEYS = ("statement", "assumption_statement", "description", "text")
    if isinstance(frameworks, list):
        for fw in frameworks:
            if not isinstance(fw, dict):
                continue
            # Assumptions / propositions / hypotheses (objects or strings)
            for key in ("assumptions", "propositions", "hypotheses"):
                values = fw.get(key, [])
                if isinstance(values, list):
                    for v in values:
                        if isinstance(v, str) and len(v) > 15:
                            claims.append(v)
                        elif isinstance(v, dict):
                            for sk in _STATEMENT_KEYS:
                                sv = v.get(sk)
                                if isinstance(sv, str) and len(sv) > 15:
                                    claims.append(sv)
                                    break
            # Testable predictions are plain strings in this schema.
            preds = fw.get("testable_predictions", [])
            if isinstance(preds, list):
                for p in preds:
                    if isinstance(p, str) and len(p) > 15:
                        claims.append(p)
            # Component descriptions (schema key is ``conceptual_components``;
            # accept ``components`` too for forward/back-compat).
            components = fw.get("conceptual_components") or fw.get("components") or []
            if isinstance(components, list):
                for c in components:
                    if isinstance(c, dict) and isinstance(c.get("description"), str) and len(c.get("description")) > 15:
                        claims.append(c["description"])
    # Dedupe while preserving order
    seen: set[str] = set()
    deduped: List[str] = []
    for c in claims:
        s = c.strip()
        if s and s not in seen:
            seen.add(s)
            deduped.append(s)
        if len(deduped) >= max_claims:
            break
    return deduped


# ---------------------------------------------------------------------------
# CausalFM estimator in CalibrationStage
# ---------------------------------------------------------------------------

def run_causal_fm_pass(
    calibration_output: Dict[str, Any],
    *,
    data_rows: Optional[Iterable[Dict[str, Any]]] = None,
    collector: Any = None,
) -> Optional[Dict[str, Any]]:
    """Best-effort CausalFMClient call on calibration output.

    When ``causal_fm_enabled`` is true, extract a causal spec from the
    calibration output and run CausalFMClient on the observations passed in
    ``data_rows``. Returns the CausalEstimate as a dict, None if the hook is
    disabled or no causal spec can be formed, or a refusal record
    (``status="refused"``, ``ate=None``) when no observations are supplied:
    no estimate is produced from generated data.
    """
    if not is_enabled("causal_fm_enabled"):
        return None
    try:
        from shared.econ import CausalAdjustment, CausalFMClient
    except Exception:
        return None

    spec = _extract_causal_adjustment(calibration_output)
    if spec is None:
        return None

    rows = list(data_rows) if data_rows is not None else []
    if not rows:
        return {
            "status": "refused",
            "ate": None,
            "method": spec.method,
            "treatment": spec.treatment,
            "outcome": spec.outcome,
            "reason": "no observed data rows were supplied; the causal estimate "
                      "requires real observations",
        }

    try:
        client = CausalFMClient(collector=collector, n_bootstrap=120)
        estimate = client.estimate(rows, spec)
        return estimate.to_dict()
    except Exception as exc:  # noqa: BLE001 - best-effort hook
        return {"error": str(exc), "method": spec.method}


def _extract_causal_adjustment(
    calibration_output: Dict[str, Any],
) -> Optional[Any]:
    """Pull a causal adjustment spec from calibration output or synthesise one.

    Looks for an explicit ``causal_adjustment`` field first (set by V0.7
    ModelDesignStage); if absent, tries to infer from the first calibrated
    model's parameters and empirical targets.
    """
    try:
        from shared.econ import CausalAdjustment
    except Exception:
        return None

    # Explicit override wins
    explicit = calibration_output.get("causal_adjustment")
    if isinstance(explicit, dict) and explicit.get("treatment") and explicit.get("outcome"):
        try:
            return CausalAdjustment(**explicit)
        except Exception:
            return None

    models = calibration_output.get("calibrated_models") or []
    if not models:
        return None
    model = models[0]
    if not isinstance(model, dict):
        return None
    params = model.get("calibrated_parameters") or []
    targets = model.get("empirical_targets") or []
    if not params or not targets:
        return None
    treatment = _safe_name(params[0])
    outcome = _safe_name(targets[0])
    if not treatment or not outcome or treatment == outcome:
        return None
    covariates = [_safe_name(p) for p in params[1:3] if _safe_name(p) and _safe_name(p) != outcome]
    try:
        return CausalAdjustment(
            method="back_door",
            treatment=treatment,
            outcome=outcome,
            adjustment_set=[c for c in covariates if c],
        )
    except Exception:
        return None


def _safe_name(entry: Any) -> Optional[str]:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        for key in ("name", "parameter_name", "symbol", "indicator"):
            if isinstance(entry.get(key), str):
                return entry[key]
    return None


# ---------------------------------------------------------------------------
# Optional SimulationStage invocation from the MasterOrchestrator
# ---------------------------------------------------------------------------

def maybe_run_simulation_stage(
    calibration_output_path: str | Path,
    output_dir: str | Path,
    *,
    mode: str,
    collector: Any = None,
) -> Optional[Dict[str, Any]]:
    """Conditionally invoke the optional 4th ModelTeam stage.

    Gated on ``simulation_enabled`` feature flag. Returns the payload written
    to ``simulation_output.json`` or None when disabled.
    """
    if not is_enabled("simulation_enabled"):
        return None
    try:
        from shared.econ.stage import run_simulation_stage
    except Exception:
        return None

    out_path = Path(output_dir) / "simulation_output.json"
    try:
        payload = run_simulation_stage(
            calibration_output_path=calibration_output_path,
            output_path=out_path,
            simulation_enabled=True,
        )
    except Exception as exc:  # noqa: BLE001 - best-effort hook
        payload = {"simulation_error": str(exc), "mode": mode}
        out_path.write_text(json.dumps(payload, indent=2))
    return payload


# ---------------------------------------------------------------------------
# Convenience writer — persist falsifier reports alongside theory output
# ---------------------------------------------------------------------------

def attach_falsifier_reports(
    theory_file: str | Path,
    reports: List[Dict[str, Any]],
) -> None:
    """Merge falsifier reports into the theory output file in-place."""
    if not reports:
        return
    path = Path(theory_file)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    data["falsifier_reports"] = reports
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def attach_causal_estimate(
    calibration_file: str | Path,
    estimate: Optional[Dict[str, Any]],
) -> None:
    """Merge a causal estimate into the calibration output file in-place."""
    if not estimate:
        return
    path = Path(calibration_file)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    data["causal_estimate"] = estimate
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass
