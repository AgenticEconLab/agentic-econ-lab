# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Estimation Stage — Estimator agent proposes, the deterministic harness disposes.

Pipeline:
1. Load the DataTeam artifact and re-fetch FULL series through the open connectors
2. Estimator (LLM) maps the model specification onto the available series as an EstimationSpec
   (dependent, regressors, transforms, theory-derived sign hypotheses)
3. estim_harness estimates it (statsmodels OLS, HAC default) and issues the honest verdict

Input:  model specification (ModelTeam) + validated dataset artifact (DataTeam)
        + optionally the CodeTeam's executable model: shown to the Estimator, and the
        module variables the specification links to are recorded in metadata
Output: estimation_output.json (EstimationStageOutput)

The verdict is ALWAYS the harness's. An LLM proposal the data cannot support yields
``inestimable`` with a machine-readable reason — never a substituted or fabricated fit.
With a reviewer (WithHITL), ``run_reviewed_pipeline`` puts the review BEFORE estimation:
the reviewer sees the proposed specification, the available series and
the observation budget, never coefficients or fit; one bounded revision is re-reviewed; every
fitted specification is listed in ``metadata.fitted_specifications``.
This file is byte-identical across EstimationTeam mode directories (mode-copy invariant);
which pipeline runs is decided by the per-mode MasterOrchestrator / pipeline runner.
"""

import json
import os
import sys
import time
from pathlib import Path as _Path
from typing import Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv

_agents_dir = _Path(__file__).resolve().parent.parent.parent.parent
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

from shared.auto_input import auto_input, get_default
from shared.json_repair import repair_json
from shared.llm import LLMClient
from shared.observability import MetricsCollector

from EstimationTeam.ael.estim_harness import (
    EstimationOutcome,
    EstimationSpec,
    VERDICT_INESTIMABLE,
    dependent_in_regressors,
    load_panel,
    repair_departures,
    run_estimation,
)
from EstimationTeam.ael.estim_harness.harness import (
    MIN_OBS, MIN_OBS_PER_PARAM, build_design, required_observations)
from EstimationTeam.ael.estim_harness import model_link
from EstimationTeam.ael.schemas.stage_outputs import EstimationStageOutput

load_dotenv()

# Harness verdicts ordered from worst to best: a revision never replaces a better verdict.
_VERDICT_RANK = {"inestimable": 0, "fragile": 1, "estimated": 2}


# ========== AGENT ==========

class Estimator:
    """Econometric specification agent: model spec + available series -> EstimationSpec."""

    SYSTEM_PROMPT = (
        "You are an expert econometrician. You translate an economic model specification "
        "into ONE estimable reduced-form regression on the series actually available. "
        "You respond with STRICT JSON only — no markdown, no commentary."
    )

    def __init__(self, openai_api_key: str = "", collector: Optional[MetricsCollector] = None):
        self.agent_name = "Estimator"
        self.collector = collector
        self.llm = LLMClient(
            temperature=0.2,
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
            collector=collector,
            agent_name=self.agent_name,
        )

    def propose_spec(
        self,
        research_question: str,
        model_context: str,
        variables_desc: str,
        feedback: Optional[str] = None,
        prior_error: Optional[str] = None,
        executable_model_desc: str = "",
    ) -> EstimationSpec:
        """Ask the LLM for an EstimationSpec; pydantic-validate; raise ValueError on failure."""
        guidance = ""
        if feedback:
            guidance += f"\nReviewer feedback to incorporate:\n{feedback[:1500]}\n"
        if prior_error:
            guidance += f"\nYour previous proposal was rejected: {prior_error[:600]}\nFix it.\n"

        code_block = ""
        if executable_model_desc:
            code_block = (
                "\nExecutable model (CodeTeam module derived from the lead model's equations):\n"
                f"{executable_model_desc}\n"
                "For each series that measures one of these module variables, add "
                '"model_variable": "<module symbol>" to its variable object; omit it otherwise.\n')

        prompt = f"""Propose ONE estimable regression specification.

Research question: {research_question[:400]}

Model specification (from the ModelTeam):
{model_context[:2200]}
{code_block}
AVAILABLE SERIES (you may ONLY reference these; use the exact id in series_ref):
{variables_desc[:2200]}
{guidance}
Return STRICT JSON:
{{
  "dependent": {{"name": "<role name>", "series_ref": "<exact series id>", "transform": "level|log|diff|log_diff|pct_change|yoy_pct_change", "lag": 0}},
  "regressors": [{{"name": "...", "series_ref": "...", "transform": "...", "lag": 0}}, ...],
  "method": "ols",
  "cov_type": "HAC",
  "add_constant": true,
  "include_trend": false,
  "endogenous": [],
  "instruments": [],
  "hypotheses": [{{"name": "...", "param": "<a regressor name>", "restriction": "<0|>0|=0", "rationale": "..."}}],
  "rationale": "one sentence on why this specification tests the model"
}}

Rules:
- 1 dependent, 1-4 regressors, all series_ref values from the AVAILABLE SERIES list only.
- Prefer transforms that make macro series stationary (rates/percentages: level; indices/levels: pct_change or log_diff).
- Derive each hypothesis from the model's testable predictions; param must equal a regressor name.
- Set "include_trend": true when regressing slow-moving LEVEL variables over long samples
  (decades) — a shared secular trend otherwise masquerades as a relationship.
- Default to "method": "ols". Use "iv2sls" ONLY when the model implies a regressor is
  endogenous AND a credible excluded instrument exists among the AVAILABLE SERIES: then set
  "endogenous" to the affected regressor name(s) and "instruments" to variable objects (same
  shape as regressors), with at least as many instruments as endogenous regressors.
- For an INTERACTION term (X1 x X2), add "interact_with": "<second series id>" to the
  regressor — its column becomes the product of the two identically-transformed series.
  With "subtract_ref" on the same regressor the difference is built first: (X - W) * Z.
  NEVER re-list an existing regressor's series as a new regressor: duplicate (series,
  transform, lag) columns are perfectly collinear and are dropped by the harness.
- For a DERIVED DIFFERENCE (e.g. real rate = nominal rate minus inflation), add
  "subtract_ref": "<series id to subtract>" — never proxy a real variable with its
  nominal counterpart when both ingredients are in the AVAILABLE SERIES list. When the
  two ingredients need DIFFERENT transforms, also set "subtract_transform": a real rate
  is "transform": "level" on the nominal RATE and "subtract_transform": "yoy_pct_change"
  on the price INDEX — subtracting a price LEVEL from a rate is meaningless.
  EXCEPTION: if the dependent is built from that same price index (e.g. you are explaining
  inflation), a same-period real rate contains the dependent itself and the regression is an
  identity. Then give the real-rate regressor "lag": 1 (the previous period's real rate),
  or use the nominal rate and enter inflation's own lag as a separate regressor.
- More generally, never use ANY series that constructs the dependent (the dependent's
  "series_ref", its "subtract_ref" or its "interact_with") in a same-period regressor or
  instrument (as its series, its "subtract_ref", or its "interact_with"): a dependent
  built as A minus B regressed on contemporaneous B is an identity. Only lagged values
  (a "lag" different from the dependent's) of those series may appear on the right-hand side.
- Respect the OBSERVATION BUDGET line above: fewer regressors, honestly estimated,
  beat a specification the harness must refuse.
"""
        raw = self.llm.invoke([
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ], max_tokens=1500)
        # repair_json returns repaired TEXT (never a parsed object) — parse it ourselves.
        try:
            data = json.loads(repair_json(str(raw)))
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Estimator returned unparseable JSON ({e}): {str(raw)[:200]}")
        if not isinstance(data, dict):
            raise ValueError(f"Estimator returned non-object JSON: {str(raw)[:200]}")
        data.setdefault("method", "ols")
        data.setdefault("cov_type", "HAC")
        return EstimationSpec(**data)


# ========== ORCHESTRATOR ==========

class EstimationOrchestrator:
    """Runs the estimation stage: load series -> propose -> estimate -> honest verdict."""

    def __init__(self, openai_api_key: str = "", collector: Optional[MetricsCollector] = None,
                 quiet: bool = False, output_dir: Optional[str] = None):
        self.collector = collector
        self.quiet = quiet
        # HITL feedback files go to the run's output directory, not to
        # the current working directory (which in the pipeline is the shared code tree).
        self.output_dir = output_dir
        self.estimator = Estimator(openai_api_key, collector=collector)
        self.output: Optional[EstimationStageOutput] = None
        self._panel = None
        self._alias_map: Dict[str, str] = {}
        self._module: Optional[Dict] = None
        self._code_team_ran = False
        self._research_question = ""
        self._loader_notes: List[str] = []
        self._timing: Dict[str, float] = {}
        self._fitted: List[Dict] = []

    def _print(self, msg: str):
        if not self.quiet:
            print(msg)

    @staticmethod
    def describe_variables(series_map, alias_map: Dict[str, str]) -> str:
        from EstimationTeam.ael.estim_harness.transforms import infer_frequency
        names_by_id: Dict[str, List[str]] = {}
        for alias, sid in alias_map.items():
            if alias != sid.lower():
                names_by_id.setdefault(sid, []).append(alias)
        lines = []
        coarse_obs: List[int] = []          # obs counts of the coarsest-frequency series
        coarsest = None
        _ORDER = {"annual": 0, "quarterly": 1, "monthly": 2}
        for sid, s in series_map.items():
            n = int(s.notna().sum())
            freq = infer_frequency(s.index) if n >= 3 else "annual"
            names = ", ".join(names_by_id.get(sid, [])) or sid
            lines.append(f"- id={sid} ({names}): {n} obs, {freq}, "
                         f"{s.index.min().date()}..{s.index.max().date()}")
            rank = _ORDER.get(freq, 0)
            if coarsest is None or rank < coarsest:
                coarsest, coarse_obs = rank, [n]
            elif rank == coarsest:
                coarse_obs.append(n)
        # A spec too large for the overlap (e.g. 7 parameters on 41 annual obs.) is
        # refused — state the deterministic budget so specs are sized
        # to the data BEFORE the harness has to refuse them.
        if coarse_obs:
            cap = min(coarse_obs)
            lines.append(
                f"\nOBSERVATION BUDGET: the aligned sample is capped by the COARSEST-"
                f"frequency series included (annual < quarterly < monthly); mixing in the "
                f"coarsest series above caps it at ~{cap} obs, and transforms/lags cost "
                f"1-4 more. At the required 8 obs per parameter, propose at most "
                f"{max(1, cap // 8)} parameters TOTAL (constant + trend + regressors) "
                f"when using those series — fewer regressors, honestly estimated, beat a "
                f"refused specification.")
        return "\n".join(lines)

    @staticmethod
    def summarize_model_context(model_spec_data: Dict) -> str:
        """Compact, generic summary of whatever ModelTeam artifact shape we were handed."""
        try:
            text = json.dumps(model_spec_data, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(model_spec_data)
        return text[:2200]

    def _inestimable_output(self, research_question: str, reason: str, notes: List[str],
                            timing_sec: Optional[Dict[str, float]] = None):
        outcome = EstimationOutcome(verdict=VERDICT_INESTIMABLE, reason=reason, notes=notes)
        self.output = EstimationStageOutput(
            research_question=research_question, outcome=outcome, loader_notes=notes,
            timing_sec=timing_sec or {},
            metadata={"executable_model_use": model_link.record_use(
                getattr(self, "_module", None), None,
                code_team_ran=getattr(self, "_code_team_ran", False))})
        return self.output

    # ---------------------------------------------------------------- building blocks
    def prepare(self, model_spec_data: Dict, data_artifact: Dict, research_question: str = "",
                executable_model: Optional[Dict] = None,
                model_design: Optional[Dict] = None) -> Optional[EstimationStageOutput]:
        """Load the series and build the proposal context. Returns an inestimable output when
        no usable series exist (nothing to propose on), else None."""
        self._print(f"\n[EstimationStage] Loading full series from the DataTeam artifact...")
        self._research_question = research_question
        self._timing = {}
        self._fitted: List[Dict] = []
        t0 = time.perf_counter()
        series_map, alias_map, notes = load_panel(data_artifact, collector=self.collector)
        self._timing["stage1_data_load_det_sec"] = time.perf_counter() - t0
        self._loader_notes = notes
        module = model_link.select_module(executable_model, model_spec_data, model_design)
        self._module, self._code_team_ran = module, executable_model is not None
        if not series_map:
            self._print("[EstimationStage] No usable real series — honest inestimable verdict")
            return self._inestimable_output(research_question, "no_data", notes,
                                             dict(self._timing))
        self._panel, self._alias_map = series_map, alias_map
        self._print(f"[EstimationStage] Loaded {len(series_map)} series "
                    "(alignment happens per-specification)")
        self._variables_desc = self.describe_variables(series_map, alias_map)
        self._model_context = self.summarize_model_context(model_spec_data)
        self._module_desc = model_link.describe_for_prompt(module)
        return None

    def propose(self, feedback: Optional[str] = None,
                prior_error: Optional[str] = None) -> Tuple[Optional[EstimationSpec], str]:
        """Estimator proposal (two attempts; same-period dependent sent back once). Returns (spec, last_error)."""
        spec: Optional[EstimationSpec] = None
        t0 = time.perf_counter()
        for attempt in (1, 2):
            try:
                proposal = self.estimator.propose_spec(
                    self._research_question, self._model_context, self._variables_desc,
                    feedback=feedback, prior_error=prior_error,
                    executable_model_desc=self._module_desc)
                # The dependent's series inside a same-period regressor puts it on both
                # sides; send it back once, then let the harness refuse it if it persists.
                same_period = dependent_in_regressors(proposal, self._panel, self._alias_map)
                if same_period and attempt == 1:
                    raise ValueError("a series that constructs the dependent enters "
                                     + ", ".join(same_period)
                                     + " in the same period, so it would appear on both sides; "
                                     "use a lagged value or a different series")
                spec = proposal
                break
            except Exception as e:
                prior_error = str(e)
                self._print(f"[EstimationStage] Spec proposal attempt {attempt} rejected: "
                            f"{prior_error[:160]}")
        key = "stage1_llm_proposal_llm_sec"
        self._timing[key] = self._timing.get(key, 0.0) + time.perf_counter() - t0
        return spec, prior_error or ""

    def observation_budget(self, spec: EstimationSpec) -> str:
        """The deterministic budget arithmetic for one spec — overlap after transforms and
        lags vs. the harness's requirement. Builds the design only; nothing is fitted."""
        k, needed = required_observations(spec)
        design, freq, notes = build_design(spec, self._panel, self._alias_map)
        rule = (f"{k} parameters (constant and regressors); the harness requires "
                f"max({MIN_OBS}, {MIN_OBS_PER_PARAM} x {k}) = {needed} observations; ")
        if design is None:
            return rule + ("the design cannot be built: "
                           f"{notes[-1] if notes else 'unknown failure'}")
        return rule + (f"the specification's series overlap on {len(design)} {freq} "
                       "observations after transforms and lags")

    def review_context(self, spec: EstimationSpec, label: str) -> Dict:
        """Specification first, estimation second: what
        the committee sees — the proposed specification, the available series and the
        observation-budget arithmetic. No coefficients, p-values or fit statistics exist yet."""
        return {
            "stage": "EstimationStage",
            "review": "specification before estimation",
            "candidate": label,
            "research_question": self._research_question,
            "specification": spec.model_dump(),
            "available_series": self._variables_desc,
            "observation_budget": self.observation_budget(spec),
            "loader_notes": self._loader_notes,
        }

    def estimate(self, spec: EstimationSpec, label: str = "proposal",
                 committee: str = "not_reviewed") -> EstimationStageOutput:
        """Run the harness on one spec; record it in the fitted-specification list."""
        self._print(f"[EstimationStage] Spec ({label}): {spec.dependent.name} ~ "
                    f"{' + '.join(v.name for v in spec.regressors)} ({spec.method}/{spec.cov_type})")
        t0 = time.perf_counter()
        outcome = run_estimation(spec, self._panel, self._alias_map)
        key = "stage1_estimation_det_sec"
        self._timing[key] = self._timing.get(key, 0.0) + time.perf_counter() - t0
        self._print(f"[EstimationStage] Harness verdict: {outcome.verdict}"
                    + (f" ({outcome.reason})" if outcome.reason else
                       f" — n={outcome.n_obs}, R2={outcome.r_squared:.3f}"))
        self._fitted.append({
            "label": label, "committee": committee, "spec": spec.model_dump(),
            "verdict": outcome.verdict, "reason": outcome.reason, "n_obs": outcome.n_obs,
            "fitted": True, "reported": False, "selection": ""})
        return EstimationStageOutput(
            research_question=self._research_question, outcome=outcome,
            loader_notes=self._loader_notes,
            spec_source={"proposal": "llm", "committee_revision": "llm_revised",
                         "refusal_repair": "llm_repaired"}.get(label, "llm"),
            timing_sec=dict(self._timing),
            metadata={"executable_model_use": model_link.record_use(
                self._module, spec, code_team_ran=self._code_team_ran)})

    def _finalize(self, out: EstimationStageOutput, reported_index: int, why: str,
                  spec_review: Optional[List[Dict]] = None,
                  objections: Optional[List[str]] = None) -> EstimationStageOutput:
        for i, entry in enumerate(self._fitted):
            entry["reported"] = i == reported_index
        if 0 <= reported_index < len(self._fitted):
            self._fitted[reported_index]["selection"] = why
        meta = dict(out.metadata)
        meta["fitted_specifications"] = list(self._fitted)
        if spec_review is not None:
            meta["spec_review"] = spec_review
        if objections:
            meta["committee_objections"] = objections
        self.output = out.model_copy(update={"metadata": meta,
                                             "timing_sec": dict(self._timing)})
        return self.output

    # ---------------------------------------------------------------- pipelines
    def run_estimation_pipeline(
        self,
        model_spec_data: Dict,
        data_artifact: Dict,
        research_question: str = "",
        feedback: Optional[str] = None,
        executable_model: Optional[Dict] = None,
        model_design: Optional[Dict] = None,
    ) -> EstimationStageOutput:
        """One proposal, one harness fit (no review)."""
        early = self.prepare(model_spec_data, data_artifact, research_question,
                             executable_model, model_design)
        if early is not None:
            return early
        spec, err = self.propose(feedback=feedback)
        if spec is None:
            # Honest refusal beats an arbitrary auto-regression passed off as the model's test.
            return self._inestimable_output(
                research_question, "invalid_spec",
                self._loader_notes + [f"Estimator produced no valid specification after 2 "
                                      f"attempts: {err}"], dict(self._timing))
        out = self.estimate(spec, "committee_revision" if feedback else "proposal")
        return self._finalize(out, 0, "the only specification fitted")

    def run_reviewed_pipeline(
        self,
        model_spec_data: Dict,
        data_artifact: Dict,
        review: Callable[[Dict, int], Tuple[bool, str]],
        research_question: str = "",
        executable_model: Optional[Dict] = None,
        model_design: Optional[Dict] = None,
    ) -> EstimationStageOutput:
        """The reviewer votes on the PROPOSED specification before any fit.

        1. The proposal is reviewed (``review(context, round) -> (approved, feedback)``); the
           context has the spec, the available series and the observation budget only.
        2. If rejected with feedback: ONE revision, which is reviewed again. If the revision
           is also rejected (or cannot be produced), the ORIGINAL proposal is estimated and the
           objections are recorded as a limitation. No further rounds.
        3. The chosen spec is estimated. If the harness refuses it (inestimable), ONE repair
           is proposed. It must keep the approved dependent and hypotheses
           (``repair_departures``, deterministic) and is then reviewed with the same
           specification-only context BEFORE fitting; a rejected repair is not fitted and
           the estimation stays inestimable with the objections recorded. An approved
           repair replaces the refused spec only when its verdict is better.
        4. Every proposal considered after the vote - fitted or rejected before fitting
           (verdict 'not_fitted') - is recorded in ``fitted_specifications``."""
        early = self.prepare(model_spec_data, data_artifact, research_question,
                             executable_model, model_design)
        if early is not None:
            return early
        spec0, err = self.propose()
        if spec0 is None:
            return self._inestimable_output(
                research_question, "invalid_spec",
                self._loader_notes + [f"Estimator produced no valid specification after 2 "
                                      f"attempts: {err}"], dict(self._timing))

        reviews: List[Dict] = []
        objections: List[str] = []
        ok0, fb0 = review(self.review_context(spec0, "proposal"), 1)
        reviews.append({"round": 1, "candidate": "proposal", "approved": bool(ok0),
                        "feedback": fb0, "specification": spec0.model_dump()})
        chosen, label, status, revision_feedback = spec0, "proposal", (
            "approved" if ok0 else "rejected"), None
        if not ok0:
            spec1 = None
            if fb0:
                spec1, err1 = self.propose(feedback=fb0)
                if spec1 is None:
                    objections.append(f"Committee objections to the proposed specification "
                                      f"(the revision could not be produced: {err1[:200]}): "
                                      f"{fb0}")
            else:
                objections.append("The committee did not approve the proposed specification "
                                  "and gave no feedback.")
            if spec1 is not None:
                ok1, fb1 = review(self.review_context(spec1, "committee_revision"), 2)
                reviews.append({"round": 2, "candidate": "committee_revision",
                                "approved": bool(ok1), "feedback": fb1,
                                "specification": spec1.model_dump()})
                if ok1:
                    chosen, label, status, revision_feedback = (
                        spec1, "committee_revision", "approved", fb0)
                else:
                    objections.append(f"Committee objections to the proposed specification: "
                                      f"{fb0}")
                    objections.append("Committee objections to the revised specification "
                                      "(rejected; the original proposal was estimated): "
                                      f"{fb1 or '(no feedback)'}")

        out = self.estimate(chosen, label, status)
        reported = 0
        why = (f"{label.replace('_', ' ')} approved by the committee before estimation"
               if status == "approved" else
               "original proposal kept: the committee rejected it and its revision was not "
               "approved; objections are recorded as limitations")
        if out.outcome.verdict == VERDICT_INESTIMABLE:
            refused_reason = out.outcome.reason
            refusal = (f"the harness refused the specification ({out.outcome.reason}): "
                       f"{(out.outcome.notes or [''])[-1]}. Repair THIS specification only: "
                       "keep the dependent variable exactly as it is (series, transform, lag, "
                       "construction) and keep every hypothesis with the same parameter, "
                       "restriction and regressor series; change only what the refusal "
                       "requires")
            spec_r, _err = self.propose(feedback=revision_feedback, prior_error=refusal)
            if spec_r is not None:
                # A repair must not be fitted and published unreviewed, free to change the
                # dependent, sample and hypotheses. It must (1) keep the approved dependent and hypotheses (deterministic) and
                # (2) pass the same specification-only vote as the first proposal BEFORE
                # it is fitted. A rejected repair leaves the estimation inestimable.
                departures = repair_departures(chosen, spec_r, list(self._panel.keys()),
                                               self._alias_map)
                if departures:
                    self._record_unfitted(spec_r, "refusal_repair", "not_reviewed",
                                          "repair_rejected: " + "; ".join(departures))
                    why += ("; a repair after the harness refusal was rejected without "
                            "fitting: " + "; ".join(departures))
                else:
                    rnd = len(reviews) + 1
                    ok_r, fb_r = review(self.review_context(spec_r, "refusal_repair"), rnd)
                    reviews.append({"round": rnd, "candidate": "refusal_repair",
                                    "approved": bool(ok_r), "feedback": fb_r,
                                    "specification": spec_r.model_dump()})
                    if not ok_r:
                        self._record_unfitted(spec_r, "refusal_repair", "rejected",
                                              "repair rejected by the committee before "
                                              "fitting")
                        objections.append(
                            "Committee objections to the repair proposed after the harness "
                            f"refusal (rejected before fitting; the estimation stays "
                            f"inestimable): {fb_r or '(no feedback)'}")
                        why += ("; the committee rejected the repair after the harness "
                                "refusal before it was fitted")
                    else:
                        out_r = self.estimate(spec_r, "refusal_repair", "approved")
                        if (_VERDICT_RANK[out_r.outcome.verdict]
                                > _VERDICT_RANK[out.outcome.verdict]):
                            out, reported = out_r, len(self._fitted) - 1
                            why = (f"repair after the harness refused the "
                                   f"{label.replace('_', ' ')} ({refused_reason}); approved "
                                   "by the committee before estimation")
                        else:
                            why += ("; a repair after the harness refusal was not estimable "
                                    "either")
        return self._finalize(out, reported, why, reviews, objections)

    def _record_unfitted(self, spec: EstimationSpec, label: str, committee: str,
                         reason: str) -> None:
        """A proposal that was rejected before fitting: listed in fitted_specifications with
        verdict 'not_fitted' so every proposal the stage considered is disclosed."""
        self._fitted.append({
            "label": label, "committee": committee, "spec": spec.model_dump(),
            "verdict": "not_fitted", "reason": reason, "n_obs": 0, "fitted": False,
            "reported": False, "selection": ""})

    def collect_human_feedback(self, round_number: int = 1, context: Optional[Dict] = None) -> str:
        """HITL checkpoint: review the proposed specification and estimates (WithHITL modes).
        In pipeline auto mode the committee resolver answers via auto_input's context."""
        concerns = auto_input(
            "Concerns about the estimated specification (variables, transforms, identification): ",
            default=get_default("estimation_spec_feedback"), context=context,
        ).strip()
        changes = auto_input(
            "Requested changes (add/remove regressors, different transforms, sample window): ",
            default=get_default("estimation_change_requests"), context=context,
        ).strip()
        comments = auto_input(
            "General comments: ",
            default=get_default("estimation_general_comments"), context=context,
        ).strip()
        parts = [p for p in (concerns, changes, comments) if p]
        feedback = "\n".join(parts)
        feedback_file = os.path.join(self.output_dir or ".",
                                     f"round{round_number}_estimation_feedback.json")
        with open(feedback_file, "w", encoding="utf-8") as f:
            json.dump({"round_number": round_number, "concerns": concerns,
                       "changes": changes, "comments": comments}, f, indent=2)
        self._print(f"[EstimationStage] Feedback saved to {feedback_file}")
        return feedback

    def save_estimation_output(self, filename: str = "estimation_output.json") -> str:
        if self.output is None:
            raise RuntimeError("run_estimation_pipeline first")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.output.model_dump(), f, indent=2, default=str)
        self._print(f"[EstimationStage] Output saved to {filename}")
        return filename


if __name__ == "__main__":
    spec_path = auto_input("Path to model specification JSON: ",
                           default="../../ModelTeam/ael/ModeNoWcNoHITL/model_design_output.json").strip()
    data_path = auto_input("Path to DataTeam output JSON: ",
                           default="../../DataTeam/ael/open_source_api/api_source_output.json").strip()
    with open(spec_path, "r", encoding="utf-8") as f:
        model_spec_data = json.load(f)
    with open(data_path, "r", encoding="utf-8") as f:
        data_artifact = json.load(f)
    orch = EstimationOrchestrator()
    orch.run_estimation_pipeline(model_spec_data, data_artifact)
    orch.save_estimation_output()
