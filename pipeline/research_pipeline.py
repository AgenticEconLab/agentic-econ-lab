# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Research Pipeline Orchestrator — runs the full 7-team pipeline in two phases.

Phase 1 (linear): IdeationTeam -> LiteratureTeam -> ModelTeam -> DataTeam, with a bounded
Model-Data feasibility cycle (DRS/DAR/MRR) at the Model-Data boundary.
Phase 2 (post-feasibility, _POST_FEASIBILITY_TEAMS): EstimationTeam, ReportingTeam, and the
optional CodeTeam, held back until the model and data have been reconciled.

Each team runs its existing MasterOrchestrator logic, with upstream
artifacts injected programmatically instead of via interactive input.

Usage:
    from pipeline.research_pipeline import ResearchPipelineOrchestrator
    from pipeline.pipeline_config import load_pipeline_config

    config = load_pipeline_config("pipeline/configs/full_research.yaml")
    pipeline = ResearchPipelineOrchestrator(config)
    pipeline.register_team_runner("IdeationTeam", run_ideation_team)
    result = pipeline.run(research_topic="AI in economics")
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from pipeline.artifact_store import ArtifactStore
from pipeline.pipeline_config import PipelineConfig, StageConfig
from shared.observability import MetricsCollector
from shared.instrumentation import WorkflowLogger
from shared.protocols.message_bus import MessageBus, ArtifactMessage
from shared.guardrails.budget_controller import BudgetController, BudgetExceededError
from shared.reliability.state_guard import StateGuard


class TeamResult(BaseModel):
    """Result of a single team's execution."""

    team: str
    success: bool
    outputs: Dict[str, Any] = Field(default_factory=dict)
    duration_sec: float = 0.0
    error: Optional[str] = None


class PipelineResult(BaseModel):
    """Result of the full pipeline execution."""

    pipeline_run_id: str
    success: bool
    teams_completed: List[str] = Field(default_factory=list)
    teams_failed: List[str] = Field(default_factory=list)
    team_results: Dict[str, TeamResult] = Field(default_factory=dict)
    total_duration_sec: float = 0.0
    artifact_names: List[str] = Field(default_factory=list)


# Module-level registry for backward compat — prefer instance-level registration
_TEAM_RUNNERS: Dict[str, Any] = {}

# Teams that consume the FINAL model+data artifacts and therefore run AFTER the
# Model<->Data feasibility loop's reconciliation refresh, not in the linear pass.
# Deferred teams run in execution order, so ReportingTeam still follows EstimationTeam.
_POST_FEASIBILITY_TEAMS = frozenset({"EstimationTeam", "ReportingTeam", "CodeTeam"})


def register_team_runner(team_name: str, runner_fn):
    """
    Register a team runner at module level (backward compat).

    Prefer using orchestrator.register_team_runner() for instance-level
    isolation in production/testing.
    """
    _TEAM_RUNNERS[team_name] = runner_fn


def merge_revision_requests(mrrs) -> Dict[str, Any]:
    """Combine every Model Revision Request of the feasibility loop into one request.

    The design stage regenerates the whole model on each call, so a request that names only
    the latest cycle's unmet requirements lets earlier removals return."""
    dumps = [m.model_dump() if hasattr(m, "model_dump") else dict(m) for m in mrrs if m]
    if not dumps:
        return {}
    unmet: List[str] = []
    for d in dumps:
        for r in d.get("unmet_requirements") or []:
            if r not in unmet:
                unmet.append(r)
    changes = [f"(cycle {d.get('cycle', i + 1)}) {str(d.get('requested_change') or '').strip()}"
               for i, d in enumerate(dumps) if str(d.get("requested_change") or "").strip()]
    reasons = [str(d.get("reason") or "").strip() for d in dumps if str(d.get("reason") or "").strip()]
    merged = dict(dumps[-1])
    merged.update({
        "reason": " | ".join(dict.fromkeys(reasons)),
        "requested_change": "\n".join(changes),
        "unmet_requirements": unmet,
        "decisions": [x for d in dumps for x in (d.get("decisions") or [])],
        "provenance": {**(dumps[-1].get("provenance") or {}), "merged_cycles": [d.get("cycle") for d in dumps]},
    })
    return merged


class ResearchPipelineOrchestrator:
    """
    Orchestrates the full research pipeline across 4 teams.

    Teams are executed in dependency order (topological sort from the
    pipeline config). Artifacts flow between teams via the ArtifactStore.
    """

    def __init__(
        self,
        config: PipelineConfig,
        collector: Optional[MetricsCollector] = None,
        team_runners: Optional[Dict[str, Callable]] = None,
        on_team_complete: Optional[Callable] = None,
    ):
        self.config = config
        self.pipeline_run_id = f"pipeline-{uuid.uuid4().hex[:8]}"
        self.collector = collector or MetricsCollector()
        self._on_team_complete = on_team_complete

        # Instance-level runner registry (inherits from module-level)
        self._team_runners: Dict[str, Callable] = dict(_TEAM_RUNNERS)
        if team_runners:
            self._team_runners.update(team_runners)

        # Resolve output directory
        self.output_dir = os.path.abspath(config.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize sub-components
        self.artifact_store = ArtifactStore(
            base_dir=os.path.join(self.output_dir, "artifacts"),
            pipeline_run_id=self.pipeline_run_id,
        )
        self.message_bus = MessageBus(pipeline_run_id=self.pipeline_run_id)

        # Budget controller from config
        self.budget_controller = BudgetController(
            max_cost_usd=config.budget.max_total_cost_usd,
            max_tokens=config.budget.max_total_tokens,
            per_team_limits={
                s.team: config.budget.max_per_team_cost_usd
                for s in config.stages
            },
        )

        # State guard for inter-team artifact verification
        self.state_guard = StateGuard()

        # Logger for the pipeline itself
        self.logger = WorkflowLogger(
            team="Pipeline",
            mode=config.mode,
            output_dir=Path(self.output_dir),
            framework="ael",
            quiet=True,
        )
        self.logger.set_metrics_collector(self.collector)

    def register_team_runner(self, team_name: str, runner_fn: Callable):
        """Register a team runner on this orchestrator instance."""
        self._team_runners[team_name] = runner_fn

    def run(
        self,
        research_topic: str,
        initial_artifacts: Optional[Dict[str, Any]] = None,
        resume_from: Optional[str] = None,
        skip_teams: Optional[set] = None,
    ) -> PipelineResult:
        """
        Execute the full research pipeline.

        Args:
            research_topic: The research topic to investigate.
            initial_artifacts: Optional pre-loaded artifacts (e.g., for partial runs).

        Returns:
            PipelineResult with outcomes for each team.
        """
        pipeline_start = time.time()
        self.logger.start_execution(metadata={
            "pipeline_run_id": self.pipeline_run_id,
            "research_topic": research_topic,
            "config_name": self.config.name,
            "mode": self.config.mode,
        })

        # Seed initial artifacts
        if initial_artifacts:
            for name, data in initial_artifacts.items():
                self.artifact_store.register(name, data, producer="user")

        # Always register the research topic
        self.artifact_store.register(
            "research_topic",
            {"research_topic": research_topic},
            producer="user",
        )

        # Get execution order from config
        execution_order = self.config.get_execution_order()

        result = PipelineResult(
            pipeline_run_id=self.pipeline_run_id,
            success=True,
        )

        # Determine teams to skip when resuming. ``skip_teams`` (the checkpointed set) is
        # authoritative when given: the order-prefix heuristic wrongly skips OPTIONAL or
        # DEFERRED teams that sit before resume_from in topological order but never ran
        # (e.g. CodeTeam between ModelTeam and the failure point).
        teams_to_skip = set(skip_teams or ())
        if resume_from and not teams_to_skip:
            for t in execution_order:
                if t == resume_from:
                    break
                teams_to_skip.add(t)

        # Teams deferred until after the feasibility loop: (stage_idx, name, stage_config)
        deferred_teams: List[tuple] = []

        for stage_idx, team_name in enumerate(execution_order):
            stage_config = self.config.get_stage(team_name)
            if stage_config is None or not stage_config.enabled:
                continue

            # Skip already-completed teams when resuming
            if team_name in teams_to_skip:
                import logging
                logging.getLogger("ael.pipeline").info(
                    f"Skipping {team_name} (completed in previous run)"
                )
                result.teams_completed.append(team_name)
                continue

            # Check budget before each team
            try:
                self.budget_controller.check(
                    self.collector, current_team=team_name
                )
            except BudgetExceededError as e:
                result.success = False
                result.teams_failed.append(team_name)
                result.team_results[team_name] = TeamResult(
                    team=team_name, success=False,
                    error=f"Budget exceeded before {team_name}: {e}",
                )
                break

            # Post-feasibility teams (e.g. EstimationTeam) are deferred: they estimate on
            # the FINAL model+data, which the feasibility loop may still revise/refresh.
            if team_name in _POST_FEASIBILITY_TEAMS:
                deferred_teams.append((stage_idx, team_name, stage_config))
                continue

            # Run the team
            team_result = self._run_team(
                team_name, stage_config, research_topic,
                stage_number=stage_idx + 1,
            )
            if not self._record_team_outcome(result, team_name, team_result):
                break  # Stop pipeline on first failure

        # Model<->Data feasibility loop (bounded, after the linear pass). Never fails
        # the pipeline — it is a post-hoc reconciliation over already-produced artifacts.
        if (
            getattr(self.config, "feasibility_loop", False)
            and result.success
            and "ModelTeam" in result.teams_completed
            and "DataTeam" in result.teams_completed
        ):
            try:
                self._run_feasibility_loop(research_topic, result)
            except Exception as e:  # pragma: no cover - defensive
                import logging
                logging.getLogger("ael.pipeline").warning(f"Feasibility loop skipped: {e}")

        # Post-feasibility analytic teams (EstimationTeam): estimate on the final,
        # possibly-revised model + refreshed data. Same budget/bookkeeping as the
        # linear pass; a failure here fails the run honestly.
        for stage_idx, team_name, stage_config in deferred_teams:
            if not result.success:
                break
            try:
                self.budget_controller.check(self.collector, current_team=team_name)
            except BudgetExceededError as e:
                result.success = False
                result.teams_failed.append(team_name)
                result.team_results[team_name] = TeamResult(
                    team=team_name, success=False,
                    error=f"Budget exceeded before {team_name}: {e}",
                )
                break
            team_result = self._run_team(
                team_name, stage_config, research_topic, stage_number=stage_idx + 1,
            )
            self._record_team_outcome(result, team_name, team_result)

        result.total_duration_sec = time.time() - pipeline_start
        result.artifact_names = self.artifact_store.list_artifacts()

        # Save pipeline manifest
        self._save_manifest(result, research_topic)

        self.logger.end_execution(success=result.success)
        self.logger.save_execution_log()

        return result

    def _record_team_outcome(self, result: PipelineResult, team_name: str,
                             team_result: TeamResult) -> bool:
        """Record one team's result: artifacts registered+signed+published on success,
        failure marked on the pipeline result. Returns team_result.success."""
        result.team_results[team_name] = team_result
        if team_result.success:
            result.teams_completed.append(team_name)
            if self._on_team_complete:
                self._on_team_complete(team_name, team_result)
            for artifact_name, artifact_data in team_result.outputs.items():
                self.artifact_store.register(
                    artifact_name, artifact_data, producer=team_name
                )
                self.state_guard.sign_output(
                    artifact_data if isinstance(artifact_data, dict) else {"value": artifact_data},
                    stage_name=f"{team_name}:{artifact_name}",
                )
                self.message_bus.publish(ArtifactMessage(
                    source_team=team_name,
                    artifact_name=artifact_name,
                    artifact_data=artifact_data
                    if isinstance(artifact_data, dict) else {"value": artifact_data},
                    pipeline_run_id=self.pipeline_run_id,
                ))
            return True
        result.success = False
        result.teams_failed.append(team_name)
        return False

    def _run_team(
        self,
        team_name: str,
        stage_config: StageConfig,
        research_topic: str,
        stage_number: int = 1,
    ) -> TeamResult:
        """Run a single team with upstream artifacts."""
        self.logger.start_stage(team_name, stage_number=stage_number)
        team_start = time.time()

        # Gather upstream artifacts for this team, verifying state hashes and HMAC
        # signatures: a tampered artifact fails this team instead of flowing on.
        from pipeline.artifact_store import ArtifactIntegrityError
        upstream = {}
        for input_name in stage_config.inputs:
            try:
                data = self.artifact_store.get(input_name)
            except ArtifactIntegrityError as e:
                self.logger.end_stage(team_name, status="error", error_message=str(e))
                return TeamResult(team=team_name, success=False,
                                  duration_sec=time.time() - team_start, error=str(e))
            if data is not None:
                upstream[input_name] = data
                # Verify state hash for upstream artifacts
                artifact_obj = self.artifact_store.get_artifact(input_name)
                if artifact_obj and artifact_obj.producer:
                    hash_key = f"{artifact_obj.producer}:{input_name}"
                    expected_hash = self.state_guard.get_hash(hash_key)
                    if expected_hash is not None:
                        data_for_hash = data if isinstance(data, dict) else {"value": data}
                        if not self.state_guard.verify_input(data_for_hash, expected_hash):
                            import logging
                            logging.getLogger("ael.pipeline").warning(
                                f"State hash mismatch for {hash_key} — artifact may have been modified"
                            )

        # Check if runner is registered
        runner = self._team_runners.get(team_name)
        if runner is None:
            duration = time.time() - team_start
            self.logger.end_stage(team_name, status="skipped")
            return TeamResult(
                team=team_name,
                success=False,
                duration_sec=duration,
                error=f"No runner registered for {team_name}. "
                      f"Register with register_team_runner().",
            )

        try:
            # Create team-specific output directory
            team_output_dir = os.path.join(self.output_dir, team_name)
            os.makedirs(team_output_dir, exist_ok=True)

            # Per-team collector for accurate budget tracking
            team_collector = MetricsCollector()

            outputs = runner(
                upstream_artifacts=upstream,
                mode=stage_config.mode,
                output_dir=team_output_dir,
                collector=team_collector,
            )

            # Per-team budget check with team-scoped collector
            self.budget_controller.check(
                self.collector, current_team=team_name,
                team_collector=team_collector,
            )

            # Merge team metrics into global collector for reporting
            self.collector.merge_from(team_collector)

            duration = time.time() - team_start
            # Record the files the team wrote, so Tier-1 output coverage, reliability and
            # transparency are measured for chained runs (previously always 0).
            output_files = sorted(
                os.path.relpath(os.path.join(dp, f), self.output_dir)
                for dp, _, fs in os.walk(team_output_dir) for f in fs)
            self.logger.end_stage(
                team_name, status="success",
                item_count=len(outputs) if outputs else 0,
                output_files=output_files,
            )
            return TeamResult(
                team=team_name,
                success=True,
                outputs=outputs or {},
                duration_sec=duration,
            )

        except Exception as e:
            duration = time.time() - team_start
            self.logger.end_stage(
                team_name, status="error", error_message=str(e)
            )
            return TeamResult(
                team=team_name,
                success=False,
                duration_sec=duration,
                error=str(e),
            )

    # ------------------------------------------------------------------ #
    # Model<->Data feasibility loop
    # ------------------------------------------------------------------ #
    def _data_tier(self) -> str:
        """Resolve the DataTeam tier from its stage mode."""
        sc = self.config.get_stage("DataTeam")
        m = (sc.mode if sc else "") or ""
        if "remium" in m:
            return "premium_subscribed"
        if "ploaded" in m:
            return "user_uploaded"
        return "open_source_api"

    def _feasibility_research_question(self, research_topic: str) -> str:
        rq_art = self.artifact_store.get("research_questions") or {}
        if isinstance(rq_art, dict):
            for key in ("final_questions", "questions", "prioritized", "prioritized_questions"):
                lst = rq_art.get(key)
                if isinstance(lst, list) and lst:
                    first = lst[0]
                    return first.get("question", str(first)) if isinstance(first, dict) else str(first)
        return research_topic

    def _invoke_for_feasibility(
        self, team_name: str, extra_artifacts: Optional[Dict[str, Any]] = None,
        drs_only: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Re-invoke a team's runner during the loop and register its fresh outputs.

        Distinct from ``_run_team`` (no per-team logging/manifest churn); upstream is
        gathered from the artifact store plus any ``extra_artifacts`` (e.g. the MRR).
        ``drs_only`` (ModelTeam only) runs Calibration in targets-only mode — the fast path
        for a feasibility DRS cycle.
        """
        stage_config = self.config.get_stage(team_name)
        runner = self._team_runners.get(team_name)
        if stage_config is None or runner is None:
            return None
        upstream: Dict[str, Any] = {}
        for input_name in stage_config.inputs:
            data = self.artifact_store.get(input_name)
            if data is not None:
                upstream[input_name] = data
        if extra_artifacts:
            upstream.update(extra_artifacts)

        team_output_dir = os.path.join(self.output_dir, team_name)
        os.makedirs(team_output_dir, exist_ok=True)
        team_collector = MetricsCollector()
        # drs_only only applies to ModelTeam's calibration; other runners don't accept it.
        extra_kwargs = {"drs_only": True} if (drs_only and team_name == "ModelTeam") else {}
        outputs = runner(
            upstream_artifacts=upstream, mode=stage_config.mode,
            output_dir=team_output_dir, collector=team_collector, **extra_kwargs,
        ) or {}
        self.collector.merge_from(team_collector)
        for name, data in outputs.items():
            self.artifact_store.register(name, data, producer=team_name)
        return outputs

    def _run_feasibility_loop(self, research_topic: str, result: PipelineResult) -> None:
        from DataTeam.ael.feasibility import (
            AvailabilityReporter,
            AvailabilityScout,
            FeasibilityLoopController,
            FeasibilityReview,
            available_series_from_source,
            drs_from_model_artifacts,
        )

        tier = self._data_tier()
        rq = self._feasibility_research_question(research_topic)
        # The scout also consults the open-connector universe — a requirement the
        # retrieval plan missed but a connector can supply is FEASIBLE (fetchable), not a
        # reason to ask the model to drop the variable. The resolver is the GENERAL
        # mechanism (provider-index search: FRED/WDI), with the frozen curated table as a
        # small precision layer — field-agnostic by construction, not macro-tuned.
        from shared.tools.econ_connectors import resolve_fetchable_series
        scout = AvailabilityScout(fetchable_lookup=resolve_fetchable_series)
        reporter = AvailabilityReporter()
        review = FeasibilityReview(escalation_available=self.config.feasibility_escalation)
        # committee ballots -> this run's directory
        os.environ.setdefault("AEL_HITL_LOG", os.path.join(self.output_dir, "hitl_committee_ballots.jsonl"))

        applied_mrrs: List[Any] = []

        def design_step(cycle, mrr):
            if cycle > 1 and mrr is not None:
                # Each design call regenerates the model, so the request must carry EVERY
                # revision so far, not only the latest — otherwise variables removed in cycle 1
                # come back in cycle 2.
                applied_mrrs.append(mrr)
                mrr_dict = merge_revision_requests(applied_mrrs)
                self.artifact_store.register("model_revision_request", mrr_dict, producer="FeasibilityReview")
                # DRS-only re-run: ModelDesign (applies the MRR) + target extraction, NO
                # parameter estimation — the DRS needs only variables + targets (perf-opt).
                self._invoke_for_feasibility(
                    "ModelTeam", extra_artifacts={"model_revision_request": mrr_dict}, drs_only=True)
            model_design = self.artifact_store.get("model_design") or {}
            calibration = self.artifact_store.get("model_specification") or {}
            return drs_from_model_artifacts(rq, model_design, calibration, cycle=cycle)

        def availability_step(drs):
            # The available-data UNIVERSE is invariant across model revisions — only the DRS
            # (the model's requirements) changes, and re-grading it is cheap. So REUSE the linear-pass
            # data_source instead of re-running the full DataTeam each cycle (that re-run dominated the
            # loop's wall-clock — ~half of it). The loop's terminal guarantee is unchanged: a revision
            # that would need a genuinely NEW data search just resolves as a documented limitation,
            # exactly as it already does on non-convergence. (The dataset is refreshed ONCE after the
            # loop, if the model was revised — see the post-loop block below.)
            data_source = self.artifact_store.get("data_source") or {}
            available = available_series_from_source(data_source, tier=tier)
            findings = scout.grade(drs, available, tier=tier)
            return reporter.build(rq, tier, findings)

        controller = FeasibilityLoopController(
            design_step, availability_step, review=review,
            max_cycles=self.config.feasibility_max_cycles,
        )
        loop_result = controller.run()
        # Performance: the DRS cycles ran
        # calibration in targets-only mode, so a revised model's `model_specification`
        # is not yet genuinely calibrated. If any revision was applied, run ONE full
        # ModelTeam pass on the final revision (full calibration), then refresh the
        # dataset so downstream sees a calibrated, revised model.
        # Requirements the scout matched via the curated connector universe (absent from
        # the retrieved pool) become a supplemental fetch list for the dataset refresh, so the
        # data downstream teams see actually CONTAINS the series the loop declared feasible.
        supplemental = [
            {"variable_name": f.variable_name, "source": f.matched_source,
             "series_id": f.matched_series}
            for f in getattr(loop_result.final_dar, "findings", [])
            if getattr(f, "fetchable", False)
        ]
        data_extra = {"supplemental_series": supplemental} if supplemental else None
        if supplemental:
            self.artifact_store.register(
                "supplemental_series", {"series": supplemental}, producer="AvailabilityScout")
        if loop_result.mrrs:
            all_mrr = merge_revision_requests(loop_result.mrrs)
            self.artifact_store.register("model_revision_request", all_mrr, producer="FeasibilityReview")
            model_out = self._invoke_for_feasibility(
                "ModelTeam", extra_artifacts={"model_revision_request": all_mrr})  # drs_only=False -> full calibration
            data_out = self._invoke_for_feasibility("DataTeam", extra_artifacts=data_extra)
            # The final design is a NEW model (full pass + committee rounds); grade it again
            # against the refreshed data so the reported DRS/DAR describe the model that is used.
            try:
                final_drs = drs_from_model_artifacts(
                    rq, self.artifact_store.get("model_design") or {},
                    self.artifact_store.get("model_specification") or {},
                    cycle=(loop_result.cycles or 0) + 1)
                self._final_regrade = (final_drs, availability_step(final_drs))
            except Exception as e:  # pragma: no cover - defensive
                import logging
                logging.getLogger("ael.pipeline").warning(f"final feasibility re-grade skipped: {e}")
            # Re-checkpoint the REFRESHED outputs: checkpoints from the linear pass hold the
            # pre-revision artifacts, so a resume would serve stale data (a resumed
            # report could name a different simulated series than the refreshed file).
            if self._on_team_complete:
                for team, outputs in (("ModelTeam", model_out), ("DataTeam", data_out)):
                    if outputs:
                        self._on_team_complete(team, TeamResult(
                            team=team, success=True, outputs=outputs))
        elif supplemental:
            # No model revision, but the scout promised fetchable series — refresh the
            # dataset alone so the promise is kept (and re-checkpoint for resume).
            data_out = self._invoke_for_feasibility("DataTeam", extra_artifacts=data_extra)
            if self._on_team_complete and data_out:
                self._on_team_complete("DataTeam", TeamResult(
                    team="DataTeam", success=True, outputs=data_out))
        self._persist_feasibility(loop_result)

    def _persist_feasibility(self, lr) -> None:
        """Register DRS/DAR as artifacts and write a feasibility report."""
        regrade = getattr(self, "_final_regrade", None)
        final_drs, final_dar = (regrade if regrade else (lr.final_drs, lr.final_dar))
        self.artifact_store.register("data_requirements_spec", final_drs.model_dump(), producer="ModelTeam")
        self.artifact_store.register("data_availability_report", final_dar.model_dump(), producer="DataTeam")
        # The terminal status describes the model that is used. When the final design
        # was re-graded, an unmet essential requirement there overrides a loop that had resolved.
        status = lr.status
        if regrade and (getattr(final_dar, "unmet_essential", None)
                        or getattr(final_dar, "overall_feasible", True) is False):
            status = "unresolved_after_final_design"
        elif regrade and getattr(final_dar, "overall_feasible", False):
            status = "feasible"           # the model actually used is feasible; loop_status keeps history
        report = {
            "status": status,
            "loop_status": lr.status,
            "cycles": lr.cycles,
            "final_data_availability_report": final_dar.model_dump(),
            "final_design_regraded": bool(regrade),
            "loop_data_availability_report": lr.final_dar.model_dump(),
            "decisions": [d.model_dump() for d in lr.decisions],
            "model_revision_requests": [m.model_dump() for m in lr.mrrs],
            "history": [
                {
                    "cycle": h.cycle,
                    "tier": h.tier,
                    "n_requirements": h.n_requirements,
                    "n_unmet_essential": h.n_unmet_essential,
                    "overall_feasible": h.overall_feasible,
                    "decisions": [d.model_dump() for d in h.decisions],
                    "mrr": h.mrr.model_dump() if h.mrr else None,
                }
                for h in lr.history
            ],
        }
        path = os.path.join(self.output_dir, "feasibility_report.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

    def _save_manifest(self, result: PipelineResult, research_topic: str):
        """Save pipeline execution manifest to disk."""
        manifest = {
            "pipeline_run_id": self.pipeline_run_id,
            "research_topic": research_topic,
            "config_name": self.config.name,
            "mode": self.config.mode,
            "success": result.success,
            "teams_completed": result.teams_completed,
            "teams_failed": result.teams_failed,
            "total_duration_sec": result.total_duration_sec,
            "artifacts": self.artifact_store.summary(),
            "artifact_signatures": {
                "verified_reads": self.artifact_store.verified_reads,
                "integrity_failures": self.artifact_store.integrity_failures,
            },
            "messages": self.message_bus.message_count(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = os.path.join(self.output_dir, "pipeline_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
