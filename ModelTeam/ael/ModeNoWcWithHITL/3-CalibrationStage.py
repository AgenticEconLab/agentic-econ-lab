# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Calibration Stage - Automated Mode (No Firecrawl, No HITL)
This script calibrates model parameters based on empirical targets and benchmarks.

Pipeline:
1. Calibrator: Extract empirical targets, perform empirical alignment, calibrate parameters

Input: Formal mathematical models from Stage 2 (model_design_output.json)
Output: Calibrated model with parameter values and fit metrics
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
# Add parent directories to path for shared imports
import sys
from pathlib import Path as _Path
_agents_dir = _Path(__file__).resolve().parent.parent.parent.parent
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

from shared.llm import LLMClient
from shared.observability import MetricsCollector, tracked_fred_get_series
from shared.json_repair import repair_json
from shared.parallel import parallel_map
from shared.tools.sandbox_tool import CodeSandbox, ExecutionResult
from shared.auto_input import auto_input, get_default
from ModelTeam.ael.schemas.stage_outputs import (
    FormalMathematicalModel, EmpiricalTarget, CalibrationStrategy,
    CalibratedParameter, ModelMoment, FitMetrics, SandboxValidation,
    CalibratedModel, CalibrationOutput,
)

# Load environment variables
load_dotenv()


def _parse_sandbox_json(stdout: str) -> Optional[dict]:
    """Extract the {"validation": "pass"/"fail", "moments": [...]} JSON the validation
    script prints. Returns the parsed dict, or None if no parseable object is found.
    This is the ground-truth verdict — `validated`/`fit_metrics` must follow it, not the
    LLM's narration."""
    if not stdout:
        return None
    s = stdout.strip()
    try:
        d = json.loads(s)
        if isinstance(d, dict):
            return d
    except Exception:
        pass
    import re
    m = re.search(r"\{.*\}", s, re.DOTALL)  # fall back to the outermost {...} block
    if m:
        try:
            d = json.loads(m.group(0))
            return d if isinstance(d, dict) else None
        except Exception:
            return None
    return None


def _is_tautological_moment(m: dict) -> bool:
    """A 'moment' whose computed value exactly equals its own target (0 error) is
    NOT an empirical moment — the validation code read the target value straight
    into the formula, so the parameter is being validated against itself. Such
    moments inflate fit to a meaningless 1.0 and must not count as a real pass.
    Returns True for computed == target (bit-exact) or a reported error_pct of 0."""
    if not isinstance(m, dict):
        return False
    try:
        comp = float(m.get("computed"))
        targ = float(m.get("target"))
    except (TypeError, ValueError):
        return False
    if comp == targ:
        return True
    ep = m.get("error_pct")
    try:
        return ep is not None and float(ep) == 0.0
    except (TypeError, ValueError):
        return False


def _coerce_scalar_value(raw) -> Optional[float]:
    """Coerce an LLM ``calibrated_value`` to a single float, or return None if it is genuinely
    non-scalar (a vector / multiple numbers) or unparseable.

    None means DROP the parameter — never fabricate a 0.0 that reads as calibrated but isn't (a
    fake 0.0 pollutes the fit and inflates the parameter count with a meaningless value). A single
    number, incl. one embedded in a string (``'0.96'``, ``'0.96 (annual)'``, ``'[0.96]'``), is
    repaired; a vector (``[0.1, 0.2]``, ``'Normalized Vector [0.1, 0.2, ...]'``) or a range
    (``'0.95 to 0.99'``) yields None."""
    if isinstance(raw, bool):                       # bool is an int subclass — reject explicitly
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, (list, tuple)):
        nums = [x for x in raw if isinstance(x, (int, float)) and not isinstance(x, bool)]
        return float(nums[0]) if len(nums) == 1 else None   # single-element list ok; vector -> None
    if isinstance(raw, str):
        import re
        nums = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", raw)
        if len(nums) == 1:
            try:
                return float(nums[0])
            except ValueError:
                return None
        return None                                 # zero numbers, or a vector/range in a string
    return None


# Prepended to LLM-generated calibration-validation code before sandbox execution.
# numpy has no erf/erfc; LLMs reliably use np.erf for normal-CDF / probit terms, which
# crashes the script. This defines them on the cached numpy module so the model's own
# `import numpy as np` then sees np.erf / np.erfc. (Pure stdlib math — no new deps.)
_SANDBOX_MATH_SHIM = (
    "import numpy as _np_shim, math as _math_shim\n"
    "_np_shim.erf = _np_shim.vectorize(_math_shim.erf)\n"
    "_np_shim.erfc = _np_shim.vectorize(_math_shim.erfc)\n"
)


# ========== AGENT ==========

class Calibrator:
    """Agent for calibrating model parameters using empirical targets."""
    
    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "Calibrator"
        self.api_key = openai_api_key
        self.collector = collector
        self.llm = LLMClient(
            temperature=0.3,
            api_key=self.api_key,
            collector=collector,
            agent_name=self.agent_name,
        )
    
    def calibrate_model(
        self,
        formal_model: FormalMathematicalModel,
        targets_only: bool = False,
    ) -> CalibratedModel:
        """Calibrate a formal mathematical model.

        ``targets_only=True`` runs ONLY Step 1 (extract empirical targets) and returns a
        minimal CalibratedModel — used by the Model↔Data feasibility loop, whose DRS needs
        only variables + targets, not the (slow) parameter estimation. See
        README.md, "Running". Full calibration runs once at
        convergence.
        """

        print(f"\n[{self.agent_name}] Calibrating model:")
        print(f"  {formal_model.model_title}")

        # Step 1: Extract empirical targets
        print(f"  Step 1: Extracting empirical targets and benchmarks...")
        empirical_targets = self._extract_empirical_targets(formal_model)

        if targets_only:
            print(f"  [targets-only] DRS feasibility cycle — skipping estimation (Steps 2-6).")
            return self._targets_only_model(formal_model, empirical_targets)

        # Step 2: Design calibration strategy
        print(f"  Step 2: Designing calibration strategy...")
        calibration_strategy = self._design_calibration_strategy(formal_model, empirical_targets)

        # Steps 3-5: Calibrate -> sandbox-validate -> fit, inside a bounded
        # REFINEMENT loop. Iteration 0 is the original single-shot calibration.
        # Each subsequent iteration feeds the sandbox's per-moment errors back to
        # the LLM ("moment X computed C vs target T -> adjust params within their
        # literature ranges to close the gap") and re-validates. We keep the
        # best-fit iteration. Any error in the loop falls back to the best result
        # captured so far, so the single-shot path always remains a valid fallback.
        print(f"  Step 3: Calibrating parameters...")
        calibrated_parameters = self._calibrate_parameters(
            formal_model,
            empirical_targets,
            calibration_strategy
        )

        (calibrated_parameters, sandbox_validation,
         model_moments, fit_metrics) = self._refine_calibration_loop(
            formal_model,
            empirical_targets,
            calibration_strategy,
            calibrated_parameters,
            max_iterations=3,
        )

        # Gate the REPORTED fit_score: a fit number is only meaningful if there ARE
        # calibrated parameters AND the sandbox actually validated them (verdict
        # 'pass', not a tautological/failed run). Otherwise the moment-based score
        # is not an empirically validated fit -> force it to 0.0. (The raw moment
        # score still drove the refinement selection above; only the surfaced
        # number is gated.) See drawback (3): average_fit_score must not be inflated.
        # A 'calibrated' (over-id validated) OR 'point_calibrated' (estimated params reproduce
        # targets) harness verdict is a LEGITIMATE calibration — keep its fit. Only gate genuinely
        # invalid results (uncalibratable / harness error / weak partial), and PREPEND the gate note
        # rather than destroying the harness's honest assessment (estimated params, moment_match).
        _verdict = getattr(sandbox_validation, "verdict", "") or ""
        # 'simulation_calibrated' = a genuine SMM fit (agent-based/stochastic model calibrated by
        # simulation, not closed form) — as legitimate as an over-id or point calibration.
        # 'archetype_calibrated' = SMM on a registered canonical stand-in: legitimate for the
        # parameters it estimated (reported as such), not a calibration of the whole model.
        _legit_calib = _verdict in ("calibrated", "point_calibrated", "simulation_calibrated",
                                    "archetype_calibrated")
        if (not calibrated_parameters or not _legit_calib) and fit_metrics.fit_score:
            _reason = ("no calibrated parameters" if not calibrated_parameters
                       else f"harness verdict='{_verdict or 'none'}' (not a validated/point calibration)")
            fit_metrics = fit_metrics.model_copy(update={
                "fit_score": 0.0,
                "fit_assessment": (f"Gated to 0.0 ({_reason}); {fit_metrics.fit_assessment}")[:600],
            })
            print(f"    Fit gated to 0.0: {_reason}")

        # Step 6: Synthesize calibrated model
        print(f"  Step 6: Synthesizing calibrated model...")
        calibrated_model = self._synthesize_calibrated_model(
            formal_model,
            empirical_targets,
            calibration_strategy,
            calibrated_parameters,
            model_moments,
            fit_metrics,
            sandbox_validation
        )

        # Honesty: 'uncalibratable' means NEITHER closed-form moment matching NOR simulation-based
        # estimation applies to this model (e.g. it needs a trained agent) — NOT that it fitted badly.
        # fit_score must not read as a real 0.0 fit: surface it as N/A with the harness's precise reason.
        if _verdict == "uncalibratable":
            _why = (getattr(sandbox_validation, "error", "") or fit_metrics.fit_assessment or "")[:280]
            calibrated_model = calibrated_model.model_copy(update={
                "calibration_summary": (
                    "N/A — NOT calibratable by this harness (closed-form moment matching + "
                    f"simulation-based estimation both inapplicable): {_why}. The reported fit_score "
                    "is a schema placeholder, NOT a fitted value."),
                "metadata": {**(calibrated_model.metadata or {}),
                             "calibration_status": "uncalibratable",
                             "fit_score_applicable": False,
                             "uncalibratable_reason": _why},
            })
            print(f"[{self.agent_name}] Calibration: N/A (uncalibratable) — {_why[:120]}")
        else:
            print(f"[{self.agent_name}] Calibration complete: Fit score = {fit_metrics.fit_score:.3f}")
        return calibrated_model

    def _targets_only_model(
        self,
        formal_model: FormalMathematicalModel,
        empirical_targets: List[EmpiricalTarget],
    ) -> CalibratedModel:
        """Minimal CalibratedModel carrying only the empirical targets (feasibility cycle).

        Parameter estimation (Steps 2-6) is skipped; the DRS reads only variables + targets.
        Fit is 0.0 and honestly labelled as not-calibrated so nothing downstream mistakes
        this for a real calibration.
        """
        return CalibratedModel(
            model_title=formal_model.model_title,
            based_on_model=formal_model.model_title,
            calibration_summary="Targets-only (feasibility cycle); full calibration deferred to convergence.",
            empirical_targets=empirical_targets,
            calibration_strategy=CalibrationStrategy(
                strategy_type="none",
                description="Targets-only feasibility cycle — no parameter estimation.",
                parameters_to_calibrate=[],
                targets_to_match=[t.target_id for t in empirical_targets],
                calibration_order=[],
                identification_notes="Skipped — DRS-only feasibility cycle.",
            ),
            calibrated_parameters=[],
            model_moments=[],
            fit_metrics=FitMetrics(
                total_targets=len(empirical_targets),
                targets_matched=0,
                mean_absolute_error=0.0,
                mean_relative_error=0.0,
                rmse=0.0,
                fit_score=0.0,
                fit_assessment="Not calibrated — targets-only feasibility cycle.",
            ),
            calibration_notes="Empirical targets extracted for the DRS only; parameter estimation skipped this cycle.",
            robustness_checks=[],
            next_steps=[],
            metadata={"targets_only": True},
        )

    def _extract_empirical_targets(
        self,
        formal_model: FormalMathematicalModel
    ) -> List[EmpiricalTarget]:
        """Extract empirical targets and benchmarks."""

        # Prepare context
        params_text = "\n".join([
            f"- {getattr(p, 'parameter_symbol', '')}: {getattr(p, 'parameter_name', '')} (Range: {getattr(p, 'typical_range', '')})"
            for p in formal_model.parameters[:10]
        ])

        vars_text = "\n".join([
            f"- {getattr(v, 'variable_symbol', '')}: {getattr(v, 'variable_name', '')}"
            for v in formal_model.variables[:15]
        ])

        # Fetch real FRED data to ground calibration targets
        fred_context = ""
        fred_series = {"GDPC1": "Real GDP", "UNRATE": "Unemployment Rate", "CPIAUCSL": "CPI"}
        fred_parts = []
        for series_id, label in fred_series.items():
            try:
                response = tracked_fred_get_series(
                    series_id,
                    collector=self.collector,
                    agent=self.agent_name,
                    observation_start="2020-01-01",
                )
                observations = response.get("observations", [])
                valid = [o for o in observations if o.get("value") not in (None, "", ".")]
                if valid:
                    recent = valid[-4:]  # last 4 observations
                    values_str = ", ".join([f"{o['date']}: {o['value']}" for o in recent])
                    fred_parts.append(f"- {label} ({series_id}): {values_str}")
            except Exception as e:
                print(f"    FRED {series_id} fetch failed (non-critical): {e}")
        if fred_parts:
            fred_context = "\n".join(fred_parts)

        try:
            result = self.llm.invoke([
                {"role": "system", "content": "You are an expert at identifying empirical targets for economic model calibration."},
                {"role": "user", "content": """Identify 8-12 empirical targets/moments for calibrating this model.

Model: {model_title}
Approach: {approach}

Parameters:
{parameters}

Variables:
{variables}

Recent FRED Data (use these real values to inform empirical targets):
{fred_context}

For each target, provide:
- target_id: Short ID (e.g., "T1", "T2")
- target_name: Name (e.g., "Capital-output ratio", "Labor share")
- target_type: moment, ratio, elasticity, correlation, or volatility
- description: What this measures
- empirical_value: Typical value from literature (use realistic numbers)
- empirical_source: Source (e.g., "US data 1960-2020", "OECD average")
- standard_error: Standard error if known (or null)
- related_variables: Model variable symbols
- related_parameters: Parameter symbols that affect this
- importance: High, Medium, or Low

Return as JSON with "targets" array.
Example: {{
  "targets": [
    {{
      "target_id": "T1",
      "target_name": "Capital-output ratio",
      "target_type": "ratio",
      "description": "Steady state capital to output ratio",
      "empirical_value": 2.5,
      "empirical_source": "US data 1960-2020, average K/Y",
      "standard_error": 0.1,
      "related_variables": ["K", "Y"],
      "related_parameters": ["alpha", "delta", "beta"],
      "importance": "High"
    }}
  ]
}}

Respond with ONLY the JSON object, no other text.
""".format(
                    model_title=formal_model.model_title,
                    approach=getattr(formal_model.solution_method, 'solution_approach', 'Unknown'),
                    parameters=params_text,
                    variables=vars_text,
                    fred_context=fred_context or "Not available"
                )}
            ])

            data = json.loads(repair_json(result))
            targets = []
            for t in data.get("targets", []):
                # Coerce numeric fields (LLM sometimes returns strings)
                for fld in ("empirical_value", "standard_error"):
                    raw = t.get(fld)
                    if raw is not None:
                        try:
                            t[fld] = float(raw)
                        except (ValueError, TypeError):
                            t[fld] = 0.0 if fld == "empirical_value" else None
                targets.append(EmpiricalTarget(**t))

            return targets
        
        except Exception as e:
            print(f"    Error extracting targets: {e}")
            return []
    
    def _design_calibration_strategy(
        self,
        formal_model: FormalMathematicalModel,
        empirical_targets: List[EmpiricalTarget]
    ) -> CalibrationStrategy:
        """Design calibration strategy."""
        
        params_text = "\n".join([
            f"- {getattr(p, 'parameter_symbol', '')}: {getattr(p, 'parameter_name', '')}"
            for p in formal_model.parameters[:10]
        ])
        
        targets_text = "\n".join([
            f"- {t.target_id}: {t.target_name} (Importance: {t.importance})"
            for t in empirical_targets
        ])
        
        try:
            result = self.llm.invoke([
                {"role": "system", "content": "You are an expert at designing calibration strategies for economic models."},
                {"role": "user", "content": """Design a calibration strategy for this model.

Model: {model_title}

Parameters:
{parameters}

Targets:
{targets}

Provide:
- strategy_type: direct, indirect, SMM, GMM, or maximum_likelihood
- description: Description of the strategy (2-3 sentences)
- parameters_to_calibrate: List of parameter symbols to calibrate
- targets_to_match: List of target IDs to match
- calibration_order: Ordered list of parameters (which to calibrate first)
- identification_notes: Notes on how parameters are identified

Return as JSON.
Example: {{
  "strategy_type": "direct",
  "description": "Use direct calibration for easily identifiable parameters, then indirect calibration for remaining parameters by matching model moments to empirical targets.",
  "parameters_to_calibrate": ["beta", "sigma", "alpha", "delta"],
  "targets_to_match": ["T1", "T2", "T3", "T4"],
  "calibration_order": ["beta", "delta", "alpha", "sigma"],
  "identification_notes": "Beta identified from interest rate, delta from investment rate, alpha from labor share, sigma from consumption volatility."
}}

Respond with ONLY the JSON object, no other text.
""".format(
                    model_title=formal_model.model_title,
                    parameters=params_text,
                    targets=targets_text
                )}
            ])

            data = json.loads(repair_json(result))
            strategy = CalibrationStrategy(**data)
            
            return strategy
        
        except Exception as e:
            print(f"    Error designing strategy: {e}")
            return CalibrationStrategy(
                strategy_type="direct",
                description="Standard calibration approach",
                parameters_to_calibrate=[],
                targets_to_match=[],
                calibration_order=[],
                identification_notes=""
            )
    
    def _calibrate_parameters(
        self,
        formal_model: FormalMathematicalModel,
        empirical_targets: List[EmpiricalTarget],
        calibration_strategy: CalibrationStrategy
    ) -> List[CalibratedParameter]:
        """Calibrate model parameters."""
        
        params_text = "\n".join([
            f"- {getattr(p, 'parameter_symbol', '')}: {getattr(p, 'parameter_name', '')} (Range: {getattr(p, 'typical_range', '')})"
            for p in formal_model.parameters
        ])
        
        targets_text = "\n".join([
            f"- {t.target_id}: {t.target_name} = {t.empirical_value} (from {t.empirical_source})"
            for t in empirical_targets
        ])
        
        strategy_text = f"""
Strategy: {calibration_strategy.strategy_type}
Order: {' -> '.join(calibration_strategy.calibration_order)}
{calibration_strategy.identification_notes}
        """.strip()
        
        try:
            result = self.llm.invoke([
                {"role": "system", "content": "You are an expert at calibrating parameters for economic models."},
                {"role": "user", "content": """Calibrate the parameters for this model based on empirical targets.

Model: {model_title}

Parameters:
{parameters}

Empirical Targets:
{targets}

Calibration Strategy:
{strategy}

For each parameter to calibrate, provide:
- parameter_symbol: Symbol
- parameter_name: Name
- calibrated_value: Calibrated value (realistic number)
- calibration_method: How it was calibrated
- target_matched: Target ID matched (if applicable)
- literature_range: Typical range
- justification: Justification for this value
- sensitivity: High, Medium, or Low

Return as JSON with "calibrated_parameters" array.
Example: {{
  "calibrated_parameters": [
    {{
      "parameter_symbol": "beta",
      "parameter_name": "Discount factor",
      "calibrated_value": 0.96,
      "calibration_method": "Directly set to match annual interest rate of 4%",
      "target_matched": "T1",
      "literature_range": "[0.95, 0.99]",
      "justification": "Standard value in macro literature, implies 4% annual discount rate",
      "sensitivity": "Medium"
    }}
  ]
}}

Respond with ONLY the JSON object, no other text.
""".format(
                    model_title=formal_model.model_title,
                    parameters=params_text,
                    targets=targets_text,
                    strategy=strategy_text
                )}
            ])

            data = json.loads(repair_json(result))
            calibrated_params = []
            for p in data.get("calibrated_parameters", []):
                # Coerce calibrated_value to a single float (LLM sometimes returns strings or,
                # for RL/agent models, whole vectors). A genuinely non-scalar value is DROPPED,
                # not defaulted to 0.0 — a fake 0.0 reads as calibrated but isn't, polluting the
                # fit and the param count.
                if isinstance(p, dict):
                    scalar = _coerce_scalar_value(p.get("calibrated_value", None))
                    if scalar is None:
                        print(f"    Dropping parameter {p.get('parameter_symbol', '?')}: non-scalar "
                              f"calibrated_value {p.get('calibrated_value')!r} (cannot represent as a scalar)")
                        continue
                    p["calibrated_value"] = scalar
                # Per-item try/except: a single bad param (e.g. a str field arriving
                # as a list) must be SKIPPED, never abort the whole list. CalibratedParameter
                # inherits LLMCoercedModel so list/dict-for-str shapes auto-coerce; anything
                # still invalid is dropped while the rest survive — this is what kept the
                # refinement loop stuck at iter 0 (bare except -> return []).
                try:
                    calibrated_params.append(CalibratedParameter(**p))
                except Exception as item_err:
                    sym = p.get('parameter_symbol', '?') if isinstance(p, dict) else repr(p)
                    print(f"    Skipping invalid calibrated parameter {sym}: {item_err}")
                    continue

            return calibrated_params

        except Exception as e:
            print(f"    Error calibrating parameters: {e}")
            return []

    def _evaluate_calibration(
        self,
        formal_model: FormalMathematicalModel,
        calibrated_parameters: List[CalibratedParameter],
        empirical_targets: List[EmpiricalTarget],
    ) -> Tuple[SandboxValidation, List[ModelMoment], FitMetrics]:
        """Run sandbox validation -> model moments -> fit metrics for one
        candidate parameter set. Shared by the single-shot path and every
        refinement iteration so they are scored identically."""
        # V0.7 DETERMINISTIC calibration harness (replaces LLM-generated sandbox self-validation,
        # which produced tautological/invented-target/non-reproducible fits). The LLM only PROPOSED
        # these parameters; the harness independently SCORES them against CITED external targets using
        # sympy-derived model moments. No LLM call, no fabricated/tautological fit.
        try:
            from ModelTeam.ael.calib_harness import run as _calib_run
        except Exception as _exc:  # sympy not installed -> honest N/A, NEVER the old fabrication
            _sv = SandboxValidation(
                validated=False, verdict="harness_unavailable",
                validation_method="deterministic_moment_harness",
                error=(f"calib_harness unavailable ({type(_exc).__name__}: {str(_exc)[:80]}); "
                       "pip install sympy. Calibration NOT validated (honest N/A, not fabricated)."))
            _fm = FitMetrics(total_targets=0, targets_matched=0, mean_absolute_error=0.0,
                             mean_relative_error=0.0, rmse=0.0, fit_score=0.0,
                             fit_assessment="deterministic harness unavailable (sympy missing) — unvalidated")
            return _sv, [], _fm
        try:
            _fm_dict = formal_model.model_dump() if hasattr(formal_model, "model_dump") else dict(formal_model)
            _params = [p.model_dump() if hasattr(p, "model_dump") else dict(p) for p in calibrated_parameters]
            _outcome = _calib_run(_fm_dict, _params)
            return self._harness_outcome_to_schema(_outcome)
        except Exception as _exc:
            _sv = SandboxValidation(
                validated=False, verdict="harness_error",
                validation_method="deterministic_moment_harness",
                error=f"{type(_exc).__name__}: {str(_exc)[:140]}")
            _fm = FitMetrics(total_targets=0, targets_matched=0, mean_absolute_error=0.0,
                             mean_relative_error=0.0, rmse=0.0, fit_score=0.0, fit_assessment="harness error")
            return _sv, [], _fm

    def _harness_outcome_to_schema(self, outcome):
        """Map a calib_harness FitOutcome to (SandboxValidation, List[ModelMoment], FitMetrics).

        Honesty contract: ``validated`` is True only when the harness produced a FORMAL
        over-identification fit (calibration_status=='calibrated'); a just-identified or
        uncalibratable model is validated=False and its FitMetrics.fit_score carries the DESCRIPTIVE
        moment_match_score (how well the proposed params reproduce external targets), with the formal
        verdict + reason spelled out in fit_assessment. Never fabricates a passing fit.
        """
        moments = []
        for r in (outcome.moments or []):
            if r.computed is None or r.target is None:
                continue
            ae = abs(float(r.computed) - float(r.target))
            re_pct = (r.rel_error or 0.0) * 100.0
            q = ("Excellent" if re_pct < 5 else "Good" if re_pct < 15
                 else "Fair" if re_pct < 35 else "Poor")
            moments.append(ModelMoment(
                moment_name=r.moment_key, model_value=float(r.computed),
                empirical_value=float(r.target), absolute_error=float(ae),
                relative_error=float(re_pct), fit_quality=q))
        n = len(moments)
        mae = sum(m.absolute_error for m in moments) / n if n else 0.0
        mre = sum(m.relative_error for m in moments) / n if n else 0.0
        rmse = (sum(m.absolute_error ** 2 for m in moments) / n) ** 0.5 if n else 0.0
        matched_well = sum(1 for m in moments if m.relative_error < 15.0)

        formal = outcome.fit_score
        score = formal if formal is not None else (outcome.moment_match_score or 0.0)
        formal_str = ("%.3f" % formal) if formal is not None else (
            "None (point-calibrated: df<1, no over-id test)" if outcome.calibration_status == "point_calibrated"
            else "None (df<1, untestable)")
        # The harness ESTIMATED these params (optimized within literature bounds to match targets);
        # they, not the LLM's asserted values, achieve the reported moment_match_score.
        est_str = (f" | ESTIMATED_params={ {k: round(v, 4) for k, v in outcome.estimated_params.items()} }"
                   if getattr(outcome, "estimated", False) and outcome.estimated_params else
                   (" | estimated=True(no-shift)" if getattr(outcome, "estimated", False) else " | estimated=False"))
        assess = (f"deterministic harness | status={outcome.calibration_status} | "
                  f"formal_over_id_fit={formal_str} | "
                  f"moment_match_score={outcome.moment_match_score} (from harness-estimated params) | "
                  f"df={outcome.degrees_of_freedom} | "
                  f"{outcome.uncalibratable_reason or 'over-identified'}{est_str} | coverage={outcome.coverage}")
        if getattr(outcome, "simulator", ""):
            assess += (f" | simulator={outcome.simulator}"
                       + (" (canonical stand-in, not the model's own equations)"
                          if getattr(outcome, "simulator_is_archetype", False) else ""))
        harness_details = {
            "calibration_status": outcome.calibration_status,
            "estimated_params": {k: float(v) for k, v in (outcome.estimated_params or {}).items()},
            "cited_targets_matched": [m.moment_name for m in moments],
            "moment_match_score": outcome.moment_match_score,
            "formal_fit_score": outcome.fit_score,
            "degrees_of_freedom": outcome.degrees_of_freedom,
            "simulator": getattr(outcome, "simulator", "") or None,
            "simulator_is_archetype": bool(getattr(outcome, "simulator_is_archetype", False)),
            "coverage": dict(outcome.coverage or {}),
            "reason": outcome.uncalibratable_reason,
        }
        fit_metrics = FitMetrics(
            total_targets=n, targets_matched=matched_well,
            mean_absolute_error=float(mae), mean_relative_error=float(mre), rmse=float(rmse),
            fit_score=float(score), fit_assessment=assess[:600],
            harness_details=harness_details)
        sandbox_validation = SandboxValidation(
            validated=(outcome.fit_score is not None
                       and outcome.calibration_status in ("calibrated", "simulation_calibrated")
                       and outcome.fit_score >= 0.5),
            verdict=outcome.calibration_status,
            validation_method="deterministic_moment_harness",
            stdout=assess[:1200], error=(outcome.uncalibratable_reason or ""))
        return sandbox_validation, moments, fit_metrics

    def _refine_calibration_loop(
        self,
        formal_model: FormalMathematicalModel,
        empirical_targets: List[EmpiricalTarget],
        calibration_strategy: CalibrationStrategy,
        calibrated_parameters: List[CalibratedParameter],
        max_iterations: int = 3,
    ) -> Tuple[List[CalibratedParameter], SandboxValidation, List[ModelMoment], FitMetrics]:
        """Bounded calibrate->validate->fit refinement.

        Iteration 0 evaluates the supplied single-shot calibration. Each later
        iteration feeds the latest per-moment errors back to the LLM to nudge
        parameters toward their targets (staying within literature ranges) and
        re-evaluates. The best-fit (highest fit_score) iteration is returned.
        Robust by construction: the best result so far is always retained, and
        any exception breaks the loop and returns that best result.
        """
        # Iteration 0: score the single-shot calibration. This is the fallback.
        try:
            sv, mm, fm = self._evaluate_calibration(
                formal_model, calibrated_parameters, empirical_targets
            )
        except Exception as e:
            print(f"    Refinement: initial evaluation failed ({e}); using empty fit.")
            sv = SandboxValidation(validated=False, error=f"evaluation failed: {e}")
            mm = []
            fm = self._compute_fit_metrics(mm, empirical_targets)

        best = (calibrated_parameters, sv, mm, fm)
        best_score = fm.fit_score
        print(f"    Refinement iter 0: fit_score = {best_score:.3f}")

        # Already excellent or nothing to refine -> stop early.
        if best_score >= 0.95 or not calibrated_parameters:
            return best

        current_params = calibrated_parameters
        for it in range(1, max(1, max_iterations)):
            try:
                refined = self._refine_parameters(
                    formal_model, empirical_targets, calibration_strategy,
                    current_params, best[2],  # feed back the latest moments/errors
                )
                if not refined:
                    print(f"    Refinement iter {it}: no refined params returned; stopping.")
                    break
                sv_i, mm_i, fm_i = self._evaluate_calibration(
                    formal_model, refined, empirical_targets
                )
                print(f"    Refinement iter {it}: fit_score = {fm_i.fit_score:.3f}")
                current_params = refined
                if fm_i.fit_score > best_score:
                    best = (refined, sv_i, mm_i, fm_i)
                    best_score = fm_i.fit_score
                if best_score >= 0.95:
                    break
            except Exception as e:
                print(f"    Refinement iter {it} errored ({e}); keeping best so far.")
                break

        print(f"    Refinement complete: best fit_score = {best_score:.3f}")
        return best

    def _refine_parameters(
        self,
        formal_model: FormalMathematicalModel,
        empirical_targets: List[EmpiricalTarget],
        calibration_strategy: CalibrationStrategy,
        current_parameters: List[CalibratedParameter],
        current_moments: List[ModelMoment],
    ) -> List[CalibratedParameter]:
        """Ask the LLM to adjust parameters to close the largest moment gaps,
        staying within each parameter's literature range. Returns a new list of
        CalibratedParameter, or [] on failure (caller keeps the prior set)."""
        if not current_moments:
            return []

        params_text = "\n".join([
            f"- {p.parameter_symbol} ({p.parameter_name}) = {p.calibrated_value} "
            f"[literature range: {p.literature_range or 'unspecified'}]"
            for p in current_parameters
        ])
        # Largest errors first so the LLM focuses on the worst-fitting moments.
        ranked = sorted(current_moments, key=lambda m: m.relative_error, reverse=True)
        gaps_text = "\n".join([
            f"- {m.moment_name}: computed {m.model_value} vs target {m.empirical_value} "
            f"(error {m.relative_error:.1f}%)"
            for m in ranked
        ])

        try:
            result = self.llm.invoke([
                {"role": "system", "content": "You are an expert at calibrating economic models. You adjust parameters to reduce the gap between model-implied moments and empirical targets, while keeping every parameter within its literature range."},
                {"role": "user", "content": """The current calibration leaves these moment gaps. Adjust the parameter
values to CLOSE these gaps. Keep every parameter within its stated literature range —
do NOT move a parameter outside its range. Make targeted, economically sensible changes.

Model: {model_title}

Current Parameters:
{parameters}

Moment Gaps (largest first — computed vs target):
{gaps}

Return ALL parameters (changed and unchanged) as JSON with a "calibrated_parameters"
array, same schema as before:
- parameter_symbol, parameter_name, calibrated_value (number, within literature_range),
  calibration_method, target_matched, literature_range, justification, sensitivity

Respond with ONLY the JSON object, no other text.
""".format(
                    model_title=formal_model.model_title,
                    parameters=params_text,
                    gaps=gaps_text,
                )}
            ])

            data = json.loads(repair_json(result))
            refined: List[CalibratedParameter] = []
            for p in data.get("calibrated_parameters", []):
                if isinstance(p, dict):
                    raw_val = p.get("calibrated_value", 0.0)
                    try:
                        p["calibrated_value"] = float(raw_val)
                    except (ValueError, TypeError):
                        p["calibrated_value"] = 0.0
                # Per-item try/except so one malformed param can't abort refinement
                # (which would return [] and silently freeze the loop at iter 0).
                try:
                    refined.append(CalibratedParameter(**p))
                except Exception as item_err:
                    sym = p.get('parameter_symbol', '?') if isinstance(p, dict) else repr(p)
                    print(f"    Skipping invalid refined parameter {sym}: {item_err}")
                    continue
            return refined
        except Exception as e:
            print(f"    Error refining parameters: {e}")
            return []

    def _validate_with_sandbox(
        self,
        formal_model: FormalMathematicalModel,
        calibrated_parameters: List[CalibratedParameter],
        empirical_targets: List[EmpiricalTarget]
    ) -> SandboxValidation:
        """Generate and execute calibration validation code in sandbox."""

        if not calibrated_parameters:
            return SandboxValidation(
                validated=False,
                error="No calibrated parameters to validate"
            )

        # Build context for code generation using json.dumps for safety
        params_json = json.dumps(
            {p.parameter_symbol: p.calibrated_value for p in calibrated_parameters},
            indent=2
        )
        targets_json = json.dumps(
            {t.target_id: {"name": t.target_name, "value": t.empirical_value, "type": t.target_type}
             for t in empirical_targets},
            indent=2
        )
        equations_text = "\n".join([
            f"- {getattr(eq, 'equation_name', 'eq')}: {getattr(eq, 'equation_latex', getattr(eq, 'description', ''))}"
            for eq in formal_model.equations[:8]
        ])

        try:
            result = self.llm.invoke([
                {"role": "system", "content": "You are an expert computational economist. Generate ONLY executable Python code, no markdown fences."},
                {"role": "user", "content": """Generate a Python script that validates the calibrated parameters for this economic model.

Model: {model_title}
Framework: {framework}

Calibrated Parameters (JSON):
{params_json}

Empirical Targets (JSON) — these are the REFERENCE values you compare AGAINST. They are
NOT allowed to appear inside any moment computation:
{targets_json}

Key Equations (derive every moment from THESE — the model's own equations):
{equations}

Requirements:
1. Use ONLY numpy and json (import numpy as np, import json) — no other imports.
   For the normal CDF / probit, `np.erf` and `np.erfc` ARE available (e.g. normal CDF =
   0.5*(1+np.erf(x/np.sqrt(2)))). Do NOT import scipy or math.
2. Parse the calibrated parameters from the JSON above using json.loads().
3. Compute at least 3 model-implied moments/ratios using ONLY the calibrated PARAMETER
   values and the steady-state relationships implied by the Key Equations above
   (e.g. capital-output ratio = alpha/(delta+growth), labor share = 1-alpha, etc.).
4. CRITICAL — do NOT use, copy, or reference any empirical target VALUE inside a moment
   computation. A moment's `computed` value must be a function of the model parameters
   ONLY. Read the target value ONLY at the final comparison step (to fill `target` and
   `error_pct`). Any moment whose `computed` value equals its `target` because you read
   the target into the formula is INVALID and will be rejected.
5. Compare each computed moment to its matching empirical target and report
   error_pct = 100*abs(computed-target)/abs(target).
6. Print results as JSON: {{"validation": "pass"/"fail", "moments": [{{"name": "...", "computed": X, "target": Y, "error_pct": Z}}], "summary": "..."}}
   Set "validation" to "pass" only if the model-implied moments are economically
   sensible and broadly track the targets; "fail" otherwise.

Keep it simple — focus on steady-state relationships derived from the parameters.

Respond with ONLY the Python code, no explanations or markdown.
""".format(
                    model_title=formal_model.model_title,
                    framework=formal_model.based_on_framework,
                    params_json=params_json,
                    targets_json=targets_json,
                    equations=equations_text
                )}
            ])

            # Clean code (strip markdown fences if present)
            code = CodeSandbox.clean_llm_output(result)
            # numpy has no erf/erfc, but LLMs reliably reach for np.erf for normal-CDF /
            # probit terms -> the script would crash. Prepend a shim that defines them on
            # the (cached) numpy module so the model's later `import numpy as np` sees them.
            code = _SANDBOX_MATH_SHIM + code

            # Execute in sandbox (120s for DSGE calibration code)
            sandbox = CodeSandbox(timeout_sec=120, memory_mb=256)
            exec_result = sandbox.execute(code)

            # `validated` must mean "the validation PASSED", not "the process exited 0".
            # The script prints {"validation": "pass"/"fail", ...}; trust that verdict.
            parsed = _parse_sandbox_json(exec_result.stdout)
            verdict = str((parsed or {}).get("validation", "")).lower()
            passed = bool(exec_result.success) and verdict == "pass"

            # Reject a 'pass' built ENTIRELY on tautological moments (computed ==
            # target, 0 error): a parameter validated against itself is not an
            # empirical fit. If every reported moment is tautological the "pass"
            # is fake -> downgrade to a 'tautological' fail so fit isn't inflated.
            _moments = (parsed or {}).get("moments", []) or []
            if passed and _moments:
                _non_taut = [m for m in _moments if not _is_tautological_moment(m)]
                if not _non_taut:
                    passed = False
                    verdict = "tautological"
                    print("    Sandbox 'pass' REJECTED: all moments are tautological "
                          "(computed == target, 0 error) — not a real empirical fit")

            validation = SandboxValidation(
                validated=passed,
                verdict=(verdict if verdict in ("pass", "fail", "tautological")
                         else ("error" if not exec_result.success else "unknown")),
                code=code,
                stdout=exec_result.stdout,
                stderr=exec_result.stderr,
                error=exec_result.error,
                execution_time_sec=exec_result.execution_time_sec
            )

            if not exec_result.success:
                print(f"    Sandbox validation ERROR: {exec_result.error or exec_result.stderr[:200]}")
            elif passed:
                print(f"    Sandbox validation PASSED ({exec_result.execution_time_sec:.1f}s)")
            else:
                print(f"    Sandbox validation ran but verdict='{verdict or 'unknown'}' -> validated=False")

            return validation

        except Exception as e:
            print(f"    Sandbox validation error: {e}")
            return SandboxValidation(
                validated=False,
                error=f"Code generation failed: {e}"
            )

    def _generate_model_moments(
        self,
        formal_model: FormalMathematicalModel,
        calibrated_parameters: List[CalibratedParameter],
        empirical_targets: List[EmpiricalTarget]
    ) -> List[ModelMoment]:
        """Generate model moments from calibrated model."""
        
        params_text = "\n".join([
            f"- {p.parameter_symbol} = {p.calibrated_value}"
            for p in calibrated_parameters
        ])
        
        targets_text = "\n".join([
            f"- {t.target_id}: {t.target_name} = {t.empirical_value}"
            for t in empirical_targets
        ])
        
        try:
            result = self.llm.invoke([
                {"role": "system", "content": "You are an expert at computing model-implied moments from calibrated economic models."},
                {"role": "user", "content": """Compute model-implied moments for the calibrated model.

Model: {model_title}

Calibrated Parameters:
{parameters}

Empirical Targets:
{targets}

For each target, compute the model-implied value and errors.
Provide:
- moment_name: Name of the moment
- model_value: Value from calibrated model (realistic)
- empirical_value: Empirical target value
- absolute_error: |model_value - empirical_value|
- relative_error: |(model_value - empirical_value) / empirical_value| * 100
- fit_quality: Excellent (<5% error), Good (5-15%), Fair (15-30%), Poor (>30%)

Return as JSON with "moments" array.
Example: {{
  "moments": [
    {{
      "moment_name": "Capital-output ratio",
      "model_value": 2.48,
      "empirical_value": 2.5,
      "absolute_error": 0.02,
      "relative_error": 0.8,
      "fit_quality": "Excellent"
    }}
  ]
}}

Respond with ONLY the JSON object, no other text.
""".format(
                    model_title=formal_model.model_title,
                    parameters=params_text,
                    targets=targets_text
                )}
            ])

            data = json.loads(repair_json(result))
            moments = []
            for m in data.get("moments", []):
                # Coerce numeric fields (LLM sometimes returns strings)
                for fld in ("model_value", "empirical_value", "absolute_error", "relative_error"):
                    raw = m.get(fld)
                    if raw is not None:
                        try:
                            m[fld] = float(raw)
                        except (ValueError, TypeError):
                            m[fld] = 0.0
                moments.append(ModelMoment(**m))

            return moments
        
        except Exception as e:
            print(f"    Error generating moments: {e}")
            return []
    
    def _sandbox_moments_to_model_moments(self, sv: SandboxValidation) -> List[ModelMoment]:
        """Build fit moments from the sandbox's OWN recomputed values, so the reported
        fit_metrics reflect what was actually computed rather than the LLM's narration."""
        parsed = _parse_sandbox_json(sv.stdout) if (sv and sv.stdout) else None
        out: List[ModelMoment] = []
        for m in (parsed or {}).get("moments", []) or []:
            # Drop tautological moments (computed == target, 0 error): they are a
            # parameter validated against itself, not an empirical moment, and would
            # otherwise contribute a fake 0% error and inflate the fit toward 1.0.
            if _is_tautological_moment(m):
                print(f"    Dropping tautological moment '{m.get('name', '?')}' "
                      f"(computed == target, 0 error)")
                continue
            try:
                comp = float(m.get("computed"))
                targ = float(m.get("target"))
            except (TypeError, ValueError):
                continue
            ae = abs(comp - targ)
            ep = m.get("error_pct")
            try:
                rel = float(ep) if ep is not None else (100.0 * ae / abs(targ) if targ else 0.0)
            except (TypeError, ValueError):
                rel = (100.0 * ae / abs(targ) if targ else 0.0)
            out.append(ModelMoment(
                moment_name=str(m.get("name", "moment")),
                model_value=round(comp, 6),
                empirical_value=round(targ, 6),
                absolute_error=round(ae, 4),
                relative_error=round(rel, 2),
                fit_quality=("Excellent" if rel < 5 else "Good" if rel < 15
                             else "Fair" if rel < 30 else "Poor"),
            ))
        return out

    def _compute_fit_metrics(
        self,
        model_moments: List[ModelMoment],
        empirical_targets: List[EmpiricalTarget]
    ) -> FitMetrics:
        """Compute overall fit metrics."""
        
        if not model_moments:
            return FitMetrics(
                total_targets=len(empirical_targets),
                targets_matched=0,
                mean_absolute_error=0.0,
                mean_relative_error=0.0,
                rmse=0.0,
                fit_score=0.0,
                fit_assessment="No moments computed"
            )
        
        # Count well-matched targets (relative error < 15%).
        # NOTE: total_targets must be the SAME set the matches are drawn from
        # (the computed moments), not len(empirical_targets) — the sandbox often
        # computes fewer moments than there are declared targets, which made the
        # ratio (e.g. 2/11) meaningless and dragged the reported fit down.
        targets_matched = sum(1 for m in model_moments if m.relative_error < 15.0)
        total_targets = len(model_moments)

        # Compute mean errors
        mean_abs_error = sum(m.absolute_error for m in model_moments) / len(model_moments)
        mean_rel_error = sum(m.relative_error for m in model_moments) / len(model_moments)

        # Compute RMSE
        rmse = (sum(m.absolute_error ** 2 for m in model_moments) / len(model_moments)) ** 0.5

        # Compute fit score (0-1, higher is better)
        # Based on mean relative error: 0% error = 1.0, 50% error = 0.0
        fit_score = max(0.0, min(1.0, 1.0 - (mean_rel_error / 50.0)))
        
        # Assessment
        if fit_score >= 0.9:
            assessment = "Excellent fit - model matches data very well"
        elif fit_score >= 0.75:
            assessment = "Good fit - model captures key features"
        elif fit_score >= 0.6:
            assessment = "Fair fit - reasonable match with room for improvement"
        else:
            assessment = "Poor fit - significant discrepancies remain"
        
        return FitMetrics(
            total_targets=total_targets,
            targets_matched=targets_matched,
            mean_absolute_error=round(mean_abs_error, 4),
            mean_relative_error=round(mean_rel_error, 2),
            rmse=round(rmse, 4),
            fit_score=round(fit_score, 3),
            fit_assessment=assessment
        )
    
    def _synthesize_calibrated_model(
        self,
        formal_model: FormalMathematicalModel,
        empirical_targets: List[EmpiricalTarget],
        calibration_strategy: CalibrationStrategy,
        calibrated_parameters: List[CalibratedParameter],
        model_moments: List[ModelMoment],
        fit_metrics: FitMetrics,
        sandbox_validation: Optional[SandboxValidation] = None
    ) -> CalibratedModel:
        """Synthesize complete calibrated model."""
        
        model_title = f"Calibrated Model: {formal_model.based_on_framework}"
        
        # Generate calibration summary. The LLM's proposal and the deterministic harness's result
        # are stated separately: counts of proposed parameters/targets never stand in for what the
        # harness estimated and matched.
        hd = getattr(fit_metrics, "harness_details", None) or {}
        _status = hd.get("calibration_status") or (getattr(sandbox_validation, "verdict", "") or "unknown")
        _est = hd.get("estimated_params") or {}
        _est_txt = (", ".join(f"{k} = {v:.4g}" for k, v in _est.items()) if _est else "none")
        _tgts = hd.get("cited_targets_matched") or []
        _sim = hd.get("simulator")
        _sim_txt = ""
        if _sim:
            _sim_txt = (f" It simulated the registered canonical stand-in '{_sim}', not the model's own "
                        "equations; every parameter it did not estimate keeps its proposed value."
                        if hd.get("simulator_is_archetype") else f" It simulated the model with '{_sim}'.")
        _mm = hd.get("moment_match_score")
        calibration_summary = f"""
This calibrated model is based on '{formal_model.model_title}'.
Proposal (language model): {calibration_strategy.strategy_type} strategy, {len(calibrated_parameters)} proposed parameter values, {len(empirical_targets)} candidate empirical targets.
Deterministic harness: verdict {_status}; estimated {len(_est)} parameter(s) ({_est_txt}) against {len(_tgts)} cited target(s){(' (' + ', '.join(_tgts) + ')') if _tgts else ''}{(f'; moment-match score {_mm:.3f}' if isinstance(_mm, (int, float)) else '')}.{_sim_txt}
Harness assessment: {fit_metrics.fit_assessment}
        """.strip()
        
        # Calibration notes
        notes = f"""
Calibration Strategy: {calibration_strategy.description}
Identification: {calibration_strategy.identification_notes}

Proposed parameter values (language model; harness-estimated values are listed in the summary):
{chr(10).join([f"- {p.parameter_symbol} = {p.calibrated_value}: {p.justification}" for p in calibrated_parameters[:5]])}

Model Performance:
- Mean Relative Error: {fit_metrics.mean_relative_error:.2f}%
- RMSE: {fit_metrics.rmse:.4f}
- Targets Well-Matched: {fit_metrics.targets_matched}/{fit_metrics.total_targets}
        """.strip()
        
        # Robustness checks
        robustness_checks = [
            "Sensitivity analysis: vary key parameters by ±10% and check moment stability",
            "Alternative calibration targets: use different data sources or time periods",
            "Specification tests: check if model predictions hold out-of-sample",
            "Monte Carlo analysis: assess parameter uncertainty through bootstrap",
            "Cross-validation: split sample and validate on held-out data"
        ]
        
        # Next steps
        next_steps = [
            "Implement model in computational software (MATLAB/Python/Julia)",
            "Solve for steady state and check stability",
            "Compute impulse response functions to shocks",
            "Perform policy experiments and counterfactuals",
            "Validate model predictions against additional moments",
            "Prepare model for estimation using full likelihood or SMM"
        ]
        
        calibrated_model = CalibratedModel(
            model_title=model_title,
            based_on_model=formal_model.model_title,
            calibration_summary=calibration_summary,
            empirical_targets=empirical_targets,
            calibration_strategy=calibration_strategy,
            calibrated_parameters=calibrated_parameters,
            model_moments=model_moments,
            fit_metrics=fit_metrics,
            sandbox_validation=sandbox_validation,
            calibration_notes=notes,
            robustness_checks=robustness_checks,
            next_steps=next_steps,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "calibration_status": _status,
                "num_candidate_targets": len(empirical_targets),
                "num_proposed_params": len(calibrated_parameters),
                "num_harness_estimated_params": len(_est),
                "num_cited_targets_matched": len(_tgts),
                "harness_estimated_params": _est,
                "simulator": _sim,
                "simulator_is_archetype": bool(hd.get("simulator_is_archetype")),
                "fit_score": fit_metrics.fit_score,
                "sandbox_validated": sandbox_validation.validated if sandbox_validation else False
            }
        )
        
        return calibrated_model


# ========== ORCHESTRATOR ==========

class CalibrationOrchestrator:
    """Orchestrator for the calibration stage."""
    
    def __init__(self, openai_api_key: Optional[str] = None, collector: Optional[MetricsCollector] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        self.collector = collector
        self.calibrator = Calibrator(self.api_key, collector=collector)
        self.calibration_output: Optional[CalibrationOutput] = None
    
    def run_calibration_pipeline(
        self,
        model_design_output: Dict,
        targets_only: bool = False,
    ) -> CalibrationOutput:
        """Run the complete calibration pipeline.

        ``targets_only=True`` extracts only empirical targets per model (skips estimation) —
        for feasibility-loop cycles that need the DRS, not a calibrated model.
        """

        print(f"\n{'='*70}")
        print(f"CALIBRATION PIPELINE{' (targets-only, feasibility cycle)' if targets_only else ''}")
        print(f"{'='*70}")

        # Parse formal models
        formal_models_data = model_design_output.get('formal_models', [])
        formal_models = [FormalMathematicalModel(**m) for m in formal_models_data]

        print(f"Formal Models: {len(formal_models)}")
        print(f"{'='*70}\n")

        # Calibrate models — independent per model, so run them CONCURRENTLY (each is LLM calls +
        # a deterministic harness / isolated sandbox; vLLM batches the cross-model LLM calls for
        # ~7.5x throughput). Sandbox uses per-run NamedTemporaryFiles, so it is thread-safe.
        _results = parallel_map(
            lambda fm: self.calibrator.calibrate_model(fm, targets_only=targets_only),
            formal_models,
        )
        calibrated_models = []
        for fm, r in zip(formal_models, _results):
            if isinstance(r, Exception):
                print(f"  [calibrate] model '{getattr(fm, 'model_title', '?')}' failed: {r}")
                continue
            calibrated_models.append(r)
        
        # average_fit_score must be gated to sandbox-VALIDATED models only — a model
        # whose calibration never passed the sandbox (no params / failed codegen /
        # tautological 'pass') has a meaningless fit and must not inflate the headline
        # average. We report the validated-only average as `average_fit_score`, plus a
        # separate `average_fit_score_all` and a validated count for transparency.
        def _is_validated(m):  # STRICT: an over-identifying test was passed
            sv = getattr(m, "sandbox_validation", None)
            return bool(sv and getattr(sv, "validated", False)) and bool(getattr(m, "calibrated_parameters", []))

        def _is_calibrated(m):  # legitimate calibration: over-id validated OR point-calibrated (estimated)
            sv = getattr(m, "sandbox_validation", None)
            v = getattr(sv, "verdict", "") if sv else ""
            return v in ("calibrated", "point_calibrated") and bool(getattr(m, "calibrated_parameters", []))

        validated_models = [m for m in calibrated_models if _is_validated(m)]
        calibrated_ok = [m for m in calibrated_models if _is_calibrated(m)]
        avg_validated = (sum(m.fit_metrics.fit_score for m in validated_models) / len(validated_models)
                         if validated_models else 0.0)
        # Headline: legitimate calibrations (point-calibrated via estimation OR over-id validated) —
        # reflects the estimated params that reproduce the targets, not the degenerate all-zero.
        avg_calibrated = (sum(m.fit_metrics.fit_score for m in calibrated_ok) / len(calibrated_ok)
                          if calibrated_ok else 0.0)
        avg_all = (sum(m.fit_metrics.fit_score for m in calibrated_models) / len(calibrated_models)
                   if calibrated_models else 0.0)

        # Create output
        self.calibration_output = CalibrationOutput(
            formal_models=[
                m.model_dump() if hasattr(m, "model_dump") else m for m in formal_models
            ],
            calibrated_models=calibrated_models,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "num_models": len(formal_models),
                "num_calibrated": len(calibrated_models),
                "num_sandbox_validated": len(validated_models),
                "num_calibrated_ok": len(calibrated_ok),
                # Headline: legitimate calibrations (point-calibrated via estimation + over-id validated).
                "average_fit_score": avg_calibrated,
                # Stricter: over-identifying-test-passed models only (0.0 when none).
                "average_fit_score_over_id_validated": avg_validated,
                # Transparency: raw average across ALL models.
                "average_fit_score_all": avg_all,
            }
        )

        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Calibrated Models: {len(calibrated_models)}")
        print(f"Sandbox-Validated Models: {len(validated_models)}/{len(calibrated_models)}")
        print(f"Average Fit Score (validated only): {self.calibration_output.metadata['average_fit_score']:.3f}")
        print(f"Average Fit Score (all models): {self.calibration_output.metadata['average_fit_score_all']:.3f}")
        print(f"{'='*70}\n")
        
        return self.calibration_output
    
    def save_calibration_output(self, filename: str = "calibration_output.json"):
        """Save calibration output to JSON."""
        if not self.calibration_output:
            print("No calibration output to save")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.calibration_output.model_dump(), f, indent=2)
        
        print(f"[Orchestrator] Calibration output saved to {filename}")
    
    def save_calibration_text(self, filename: str = "calibrated_models.txt"):
        """Save calibrated models in readable text format."""
        if not self.calibration_output:
            print("No calibration output to save")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("CALIBRATED ECONOMIC MODELS\n")
            f.write("="*70 + "\n\n")
            
            for i, model in enumerate(self.calibration_output.calibrated_models, 1):
                f.write(f"MODEL {i}: {model.model_title}\n")
                f.write("="*70 + "\n\n")
                
                f.write(f"SUMMARY:\n{model.calibration_summary}\n\n")
                
                f.write(f"EMPIRICAL TARGETS ({len(model.empirical_targets)}):\n")
                for target in model.empirical_targets:
                    f.write(f"  {target.target_id}. {target.target_name}\n")
                    f.write(f"     Empirical Value: {target.empirical_value}\n")
                    f.write(f"     Source: {target.empirical_source}\n")
                    f.write(f"     Importance: {target.importance}\n\n")
                
                f.write(f"CALIBRATION STRATEGY:\n")
                f.write(f"  Type: {model.calibration_strategy.strategy_type}\n")
                f.write(f"  {model.calibration_strategy.description}\n\n")
                
                f.write(f"CALIBRATED PARAMETERS ({len(model.calibrated_parameters)}):\n")
                for param in model.calibrated_parameters:
                    f.write(f"  {param.parameter_symbol} = {param.calibrated_value}\n")
                    f.write(f"     {param.parameter_name}\n")
                    f.write(f"     Method: {param.calibration_method}\n")
                    f.write(f"     Justification: {param.justification}\n")
                    f.write(f"     Sensitivity: {param.sensitivity}\n\n")
                
                f.write(f"MODEL MOMENTS ({len(model.model_moments)}):\n")
                for moment in model.model_moments:
                    f.write(f"  {moment.moment_name}\n")
                    f.write(f"     Model: {moment.model_value:.4f} | Empirical: {moment.empirical_value:.4f}\n")
                    f.write(f"     Error: {moment.relative_error:.2f}% | Fit: {moment.fit_quality}\n\n")
                
                f.write(f"FIT METRICS:\n")
                f.write(f"  Fit Score: {model.fit_metrics.fit_score:.3f}\n")
                f.write(f"  Mean Relative Error: {model.fit_metrics.mean_relative_error:.2f}%\n")
                f.write(f"  RMSE: {model.fit_metrics.rmse:.4f}\n")
                f.write(f"  Targets Matched: {model.fit_metrics.targets_matched}/{model.fit_metrics.total_targets}\n")
                f.write(f"  Assessment: {model.fit_metrics.fit_assessment}\n\n")
                
                f.write(f"ROBUSTNESS CHECKS:\n")
                for check in model.robustness_checks:
                    f.write(f"  - {check}\n")
                f.write("\n")
                
                f.write(f"NEXT STEPS:\n")
                for step in model.next_steps:
                    f.write(f"  - {step}\n")
                f.write("\n")
                
                f.write("-"*70 + "\n\n")
        
        print(f"[Orchestrator] Calibrated models saved to {filename}")
    
    def save_parameters_csv(self, filename: str = "calibrated_parameters.csv"):
        """Save calibrated parameters to CSV."""
        if not self.calibration_output:
            print("No calibration output to save")
            return
        
        rows = []
        for model in self.calibration_output.calibrated_models:
            for param in model.calibrated_parameters:
                rows.append({
                    'model': model.model_title,
                    'parameter_symbol': param.parameter_symbol,
                    'parameter_name': param.parameter_name,
                    'calibrated_value': param.calibrated_value,
                    'calibration_method': param.calibration_method,
                    'literature_range': param.literature_range,
                    'sensitivity': param.sensitivity
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"[Orchestrator] Parameters saved to {filename}")


def main():
    """Main function for calibration stage."""
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Load model design output
    design_file = auto_input(
        "Enter path to model design output JSON (from Stage 2): ",
        default=get_default("design_file_path"),
    ).strip()
    
    if not os.path.exists(design_file):
        print(f"Error: {design_file} not found!")
        return
    
    # Load data
    with open(design_file, 'r', encoding='utf-8') as f:
        model_design_output = json.load(f)
    
    # Run pipeline
    orchestrator = CalibrationOrchestrator()
    calibration_output = orchestrator.run_calibration_pipeline(model_design_output)
    
    # Save outputs
    orchestrator.save_calibration_output("calibration_output.json")
    orchestrator.save_calibration_text("calibrated_models.txt")
    orchestrator.save_parameters_csv("calibrated_parameters.csv")
    
    # Print summary
    print("\n" + "="*70)
    print("CALIBRATION SUMMARY")
    print("="*70)
    for i, model in enumerate(calibration_output.calibrated_models, 1):
        print(f"\n{i}. {model.model_title}")
        print(f"   Fit Score: {model.fit_metrics.fit_score:.3f}")
        print(f"   Mean Error: {model.fit_metrics.mean_relative_error:.2f}%")
        print(f"   Targets Matched: {model.fit_metrics.targets_matched}/{model.fit_metrics.total_targets}")
        print(f"   Assessment: {model.fit_metrics.fit_assessment}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
