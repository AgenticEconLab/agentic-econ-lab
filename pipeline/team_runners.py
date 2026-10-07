# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Team Runners — Adapter functions that run each team's 3-stage pipeline
programmatically as part of the cross-team ResearchPipelineOrchestrator.

Each runner wraps the team's existing stage orchestrators, bypassing the
interactive main() function. This keeps the existing MasterOrchestrators
unchanged for standalone execution.

Usage:
    from pipeline.team_runners import register_all_runners

    register_all_runners()  # Must call before pipeline.run()

Canonical Artifact Keys:
    IdeationTeam  -> "research_questions": dict with "final_questions" list
    LiteratureTeam -> "literature_review", "gap_analysis", "knowledge_graph": dicts
    ModelTeam     -> "model_specification": dict with calibrated model
    DataTeam      -> "validated_dataset": dict with documented datasets
"""

import json
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

_log = logging.getLogger("ael.pipeline.runners")

from shared.observability import MetricsCollector
from shared.model_config import load_model_config, stage_model


def _ensure_agents_in_path():
    """Ensure the repository root is on sys.path for team imports."""
    agents_dir = str(Path(__file__).resolve().parent.parent)
    if agents_dir not in sys.path:
        sys.path.insert(0, agents_dir)


@contextmanager
def _team_context(team_base: Path):
    """Context manager for team directory and sys.path manipulation.

    Handles os.chdir and sys.path changes safely using try/finally.
    Note: os.chdir is NOT thread-safe — pipeline must run teams sequentially.
    """
    _ensure_agents_in_path()
    # Register tool wrappers (arxiv_search, openalex_search, fred_get_series, web_search,
    # ...). The standalone MasterOrchestrators do this at startup, but the pipeline imports
    # stage modules directly, so without this the ToolRegistry is empty and every search
    # returns 0 ("tool not found"). Idempotent; best-effort.
    try:
        from shared.tools.register_all import register_all_tools
        register_all_tools()
    except Exception as e:  # pragma: no cover - defensive
        _log.warning("tool registration failed: %s", e)
    original_dir = os.getcwd()
    team_base_str = str(team_base)
    added_to_path = False

    os.chdir(team_base_str)
    if team_base_str not in sys.path:
        sys.path.insert(0, team_base_str)
        added_to_path = True

    try:
        yield
    finally:
        os.chdir(original_dir)
        if added_to_path and team_base_str in sys.path:
            sys.path.remove(team_base_str)


def _load_json_safe(filepath: str) -> Optional[Dict]:
    """Load a JSON file, returning None on failure."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _reject_hitl_mode(team: str, mode: str) -> None:
    """Guard: HITL modes can't run in the headless cross-team pipeline.

    HITL orchestrators expose round-based APIs (``run_search_round``,
    ``run_refinement_round``, etc.) that require a human at the console.
    The cross-team pipeline expects single-shot ``run_automated_*`` APIs.
    Fail fast with a clear message instead of an opaque AttributeError.
    """
    if isinstance(mode, str) and "WithHITL" in mode:
        raise ValueError(
            f"{team} mode '{mode}' is HITL (interactive). The cross-team "
            "pipeline (run_ael_pipeline.py) is headless and cannot answer "
            "HITL prompts. Use a No-HITL mode in ael_config.yaml's "
            "pipeline_modes:, or run the team standalone via "
            f"{team}/ael/{mode}/0-MasterOrchestrator.py."
        )


def _extract_questions_list(upstream_artifacts: Dict[str, Any]) -> list:
    """Extract a flat list of question dicts/strings from upstream artifacts.

    Handles multiple key formats from IdeationTeam output:
    - "final_questions" (from IntegrationStage)
    - "questions", "prioritized", "prioritized_questions" (legacy)
    """
    questions_data = upstream_artifacts.get("research_questions", {})
    if isinstance(questions_data, dict):
        for key in ("final_questions", "questions", "prioritized", "prioritized_questions"):
            if key in questions_data and isinstance(questions_data[key], list):
                if key != "final_questions":
                    _log.warning(
                        "Using fallback key '%s' for research questions; "
                        "canonical key is 'final_questions'", key
                    )
                return questions_data[key]
        # If no recognized key, wrap the whole dict as a single item
        return [questions_data]
    elif isinstance(questions_data, list):
        return questions_data
    return []


# ============================================================================
# IdeationTeam Runner
# ============================================================================

def _hitl_max_rounds() -> int:
    try:
        return max(1, int(os.environ.get("AEL_HITL_MAX_ROUNDS", "3")))
    except ValueError:
        return 3


def _stage_approved(round_number: int, context: Dict[str, Any], question: str,
                    auto_rounds: int = 2) -> bool:
    """Whether to stop refining a WithHITL stage this round.

    In ``llm_economist`` mode the committee votes (develop-until-approved); in
    auto/interactive we keep the team's standard design (``auto_rounds`` iterations) so
    the loop is cheap and does not call the committee. Ideation refines over 2 rounds;
    Literature runs each stage once (auto_rounds=1).
    """
    from shared.auto_input import get_hitl_mode

    if get_hitl_mode() == "llm_economist":
        from shared.llm_economist import LLMEconomistCommittee
        try:
            return LLMEconomistCommittee().vote_proposition(question, context).passed
        except Exception:
            return round_number >= auto_rounds  # committee unavailable -> fall back
    return round_number >= auto_rounds


def _committee_feedback(question: str, context: Dict[str, Any]) -> str:
    """Synthesized committee feedback (llm_economist mode only), else "". Used to drive a
    stage revision when the committee does not approve. Never raises."""
    from shared.auto_input import get_hitl_mode

    if get_hitl_mode() != "llm_economist":
        return ""
    from shared.llm_economist import LLMEconomistCommittee
    try:
        return LLMEconomistCommittee().synthesize_feedback(question, context).feedback
    except Exception:
        return ""


def _collect_feedback(method, context: Dict[str, Any], **kwargs):
    """Call a stage's collect_*_feedback, grounding it with ``context`` when the method
    accepts it. Variants not yet context-aware (e.g. some WithWc files) are called
    without it — the committee still resolves the checkpoint, just less grounded.
    """
    import inspect

    try:
        supports = "context" in inspect.signature(method).parameters
    except (ValueError, TypeError):
        supports = False
    return method(context=context, **kwargs) if supports else method(**kwargs)


@contextmanager
def _hitl_log_in(output_dir: str):
    """Write committee ballots to the run's output directory while a team runs.

    The committee's default ballot log is relative to the cwd, which ``_team_context``
    sets to the team's code directory, so Ideation/Literature ballots landed in the code
    tree. An AEL_HITL_LOG already set by the caller is respected."""
    prev = os.environ.get("AEL_HITL_LOG")
    if not prev:
        os.makedirs(os.path.abspath(output_dir), exist_ok=True)
        os.environ["AEL_HITL_LOG"] = os.path.join(os.path.abspath(output_dir),
                                                  "hitl_committee_ballots.jsonl")
    try:
        yield os.environ["AEL_HITL_LOG"]
    finally:
        if not prev:
            os.environ.pop("AEL_HITL_LOG", None)


def _save_provider_counts(orchestrator: Any, output_dir: str) -> None:
    """Record the SourcingStage's per-search-provider counts (requests, results,
    failures) next to its CSV. Never raises."""
    stats = getattr(orchestrator, "provider_stats", None)
    if stats is None or not hasattr(stats, "as_dict"):
        return
    try:
        with open(os.path.join(output_dir, "sourcing_provider_counts.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(stats.as_dict(), fh, indent=2)
    except Exception as e:
        _log.warning("IdeationTeam: could not save provider counts: %s", e)


def _literature_review_texts(syn_data: Dict[str, Any]) -> list:
    """Review prose blobs from a synthesis_results.json dict (top-level keys plus the
    nested ``literature_review`` sections), as the standalone orchestrators collect them."""
    from shared.verification import extract_review_texts

    texts = extract_review_texts(syn_data) if isinstance(syn_data, dict) else []
    lr = syn_data.get("literature_review") if isinstance(syn_data, dict) else None
    if isinstance(lr, dict):
        for k in ("abstract", "introduction", "synthesis"):
            v = lr.get(k)
            if isinstance(v, str) and v.strip():
                texts.append(v)
        for s in (lr.get("sections") or []):
            if isinstance(s, dict):
                sc = s.get("section_content") or s.get("content")
                if isinstance(sc, str) and sc.strip():
                    texts.append(sc)
    return texts


def _citation_audit(output_file: str, team: str, collector=None) -> Optional[Dict[str, Any]]:
    """Run the CitationVerifier pass the standalone orchestrators run over the team's
    output file and attach the audit to it under ``citation_audit``. Returns the audit,
    or None when the verifier is disabled (feature flag) or fails. Never raises."""
    try:
        from shared.verification import (
            attach_citation_audit,
            extract_ideation_texts,
            run_citation_verifier_pass,
        )
        data = _load_json_safe(output_file) or {}
        texts = (extract_ideation_texts(data) if team == "IdeationTeam"
                 else _literature_review_texts(data))
        audit = run_citation_verifier_pass(texts, collector=collector)
        if not audit:
            return None
        if audit.get("overall_total", 0) == 0:
            audit["overall_hallucination_rate"] = None
            audit["status"] = "N/A"
            audit["note"] = ("No parseable in-text citation markers ([key] or (Author, Year)) "
                             "found; hallucination rate not computable.")
        attach_citation_audit(output_file, audit)
        return audit
    except Exception as e:  # verification must never break the pipeline
        _log.warning("%s: citation verification skipped: %s", team, e)
        return None


def _run_ideation_hitl(
    upstream_artifacts: Dict[str, Any],
    mode: str,
    output_dir: str,
    collector: Optional[MetricsCollector] = None,
) -> Dict[str, Any]:
    with _hitl_log_in(output_dir):
        return _run_ideation_hitl_impl(upstream_artifacts, mode, output_dir, collector)


def _run_ideation_hitl_impl(
    upstream_artifacts: Dict[str, Any],
    mode: str,
    output_dir: str,
    collector: Optional[MetricsCollector] = None,
) -> Dict[str, Any]:
    """Drive the WithHITL IdeationTeam flow headlessly for the pipeline.

    Each stage runs an initial round, then refines until the committee approves (or a
    round cap), with each checkpoint's ``collect_*_feedback`` grounded in the material
    under review. The team's own ``main()`` is untouched — this reuses only its stage
    round methods, so standalone runs are unaffected.
    """
    from importlib import import_module

    team_base = Path(__file__).resolve().parent.parent / "IdeationTeam" / "ael" / mode
    max_rounds = _hitl_max_rounds()

    with _team_context(team_base):
        research_topic = upstream_artifacts.get("research_topic", {}).get(
            "research_topic", "economic research (upstream topic metadata unavailable)"
        )
        collector = collector or MetricsCollector()
        model_config = load_model_config()

        # Stage 1: Sourcing
        lit_file = os.path.join(output_dir, "literature_results_all_rounds.csv")
        with stage_model(model_config, "IdeationTeam", "SourcingStage"):
            o1 = import_module("1-SourcingStage").MultiAgentOrchestrator(quiet=True, collector=collector)
        results = o1.run_search_round(research_topic=research_topic, round_number=1,
                                      feedback=None, max_results_per_agent=8)
        for rnd in range(1, max_rounds + 1):
            ctx = {"papers": [{"title": getattr(r, "title", ""), "source": getattr(r, "source", "")}
                              for r in list(results)[:10]]}
            if rnd == max_rounds or _stage_approved(rnd, ctx, "Is this literature set adequate to proceed?"):
                break
            fb = _collect_feedback(o1.collect_human_feedback, ctx, round_number=rnd)
            results = o1.run_search_round(research_topic=research_topic, round_number=rnd + 1,
                                          feedback=fb, max_results_per_agent=8)
        o1.save_results(lit_file)
        _save_provider_counts(o1, output_dir)
        # Robustness: with no search backend/egress, Stage 1 can find 0 papers and skip
        # writing the CSV — Stage 2's load_literature would then crash. Return cleanly.
        if not os.path.exists(lit_file):
            _log.warning("IdeationTeam WithHITL: Stage 1 found no literature "
                         "(search backend/egress?) — returning empty research_questions")
            return {"research_questions": {"final_questions": []},
                    "note": "no literature found in SourcingStage (search backend/egress unavailable)"}

        # Stage 2: Refinement
        ref_file = os.path.join(output_dir, "refinement_results_all_rounds.json")
        with stage_model(model_config, "IdeationTeam", "RefinementStage"):
            o2 = import_module("2-RefinementStage").RefinementOrchestrator(quiet=True, collector=collector)
        o2.output_dir = output_dir      # feedback files go to the run, not the code tree
        lit_df = o2.load_literature(lit_file)
        concepts, questions = o2.run_refinement_round(literature_df=lit_df, round_number=1,
                                                      feedback=None, num_concepts=8, num_questions=6)
        for rnd in range(1, max_rounds + 1):
            ctx = {"concepts": [getattr(c, "concept_title", "") for c in concepts],
                   "questions": [getattr(q, "question", "") for q in questions]}
            if rnd == max_rounds or _stage_approved(rnd, ctx, "Are these concepts and questions adequate?"):
                break
            fb = _collect_feedback(o2.collect_human_feedback, ctx, round_number=rnd)
            concepts, questions = o2.run_refinement_round(literature_df=lit_df, round_number=rnd + 1,
                                                          feedback=fb, num_concepts=8, num_questions=6)
        o2.save_results(ref_file)

        # Stage 3: Integration
        final_file = os.path.join(output_dir, "finalized_research_questions.json")
        with stage_model(model_config, "IdeationTeam", "IntegrationStage"):
            o3 = import_module("3-IntegrationStage").IntegrationOrchestrator(quiet=True, collector=collector)
        o3.output_dir = output_dir
        o3.research_topic = research_topic  # seed topic: prompts + deterministic topic check
        init_q = o3.load_refinement_results(ref_file)
        contextualized, prioritized = o3.run_integration_round(questions=init_q, round_number=1,
                                                               feedback=None, max_final_questions=5)
        for rnd in range(1, max_rounds + 1):
            ctx = {"prioritized_questions": [getattr(q, "question", "") for q in prioritized]}
            if rnd == max_rounds or _stage_approved(rnd, ctx, "Are these prioritized questions adequate to finalize?"):
                break
            fb = _collect_feedback(o3.collect_integration_feedback, ctx, round_number=rnd, num_questions=len(prioritized))
            q_next = o3.convert_prioritized_to_research_questions(prioritized)
            contextualized, prioritized = o3.run_integration_round(questions=q_next, round_number=rnd + 1,
                                                                   feedback=fb, max_final_questions=5)

        # The committee's quality gates can thin the final set (observed live: 6 formulated
        # -> 2 finalized), starving downstream diversity (2 frameworks -> 2 models). When finals
        # come out < 3, run ONE extra integration round over the FULL refined pool, seeded with
        # the committee's critique, asking for at least three strong but DIVERSE questions.
        if len(prioritized) < 3 and len(init_q) > len(prioritized):
            fb_text = _committee_feedback(
                "The finalized set has fewer than three research questions, which starves the "
                "downstream modeling stage of diversity. From the full refined pool in the "
                "context, which questions should be added or reworked to reach at least three "
                "strong, methodologically DIVERSE questions?",
                {"finalized": [getattr(q, "question", "") for q in prioritized],
                 "refined_pool": [getattr(q, "question", "") for q in init_q]},
            ) or ("Broaden the final set to at least three strong, methodologically diverse "
                  "research questions drawn from the refined pool; keep the existing finalists.")
            # run_integration_round expects a structured IntegrationFeedback, NOT a string
            # (a bare string crashes with: 'str' has no attribute
            # 'questions_to_prioritize') — the critique rides in additional_guidance.
            try:
                from IdeationTeam.ael.schemas.stage_outputs import IntegrationFeedback
                fb_obj = IntegrationFeedback(round_number=max_rounds + 1,
                                             additional_guidance=str(fb_text)[:2000])
            except Exception:
                fb_obj = None      # schemas unavailable -> extra round runs unguided
            _log.info("IdeationTeam: finals < 3 — one extra integration round for diversity")
            contextualized, prioritized = o3.run_integration_round(
                questions=init_q, round_number=max_rounds + 1, feedback=fb_obj, max_final_questions=5)
        o3.save_final_questions(final_file)
        _citation_audit(final_file, "IdeationTeam", collector)

        return {"research_questions": _load_json_safe(final_file) or {}}


def run_ideation_team(
    upstream_artifacts: Dict[str, Any],
    mode: str,
    output_dir: str,
    collector: Optional[MetricsCollector] = None,
) -> Dict[str, Any]:
    """
    Run IdeationTeam's 3-stage pipeline programmatically.

    Inputs (from upstream_artifacts):
        - "research_topic": {"research_topic": str}

    Outputs:
        - "research_questions": dict with prioritized questions
    """
    # WithHITL modes are now driven headlessly with committee-resolved checkpoints.
    if isinstance(mode, str) and "WithHITL" in mode:
        return _run_ideation_hitl(upstream_artifacts, mode, output_dir, collector)
    _reject_hitl_mode("IdeationTeam", mode)
    team_base = Path(__file__).resolve().parent.parent / "IdeationTeam" / "ael" / mode

    with _team_context(team_base):
        research_topic = upstream_artifacts.get("research_topic", {}).get(
            "research_topic", "economic research (upstream topic metadata unavailable)"
        )

        collector = collector or MetricsCollector()
        model_config = load_model_config()

        # Stage 1: Sourcing
        from importlib import import_module
        with stage_model(model_config, "IdeationTeam", "SourcingStage"):
            stage1 = import_module("1-SourcingStage")
            orchestrator1 = stage1.MultiAgentOrchestrator(quiet=True, collector=collector)
            literature_results = orchestrator1.run_automated_search(
                research_topic=research_topic, max_results_per_agent=15
            )
        sourcing_file = os.path.join(output_dir, "literature_results_automated.csv")
        orchestrator1.save_results(sourcing_file)
        _save_provider_counts(orchestrator1, output_dir)

        # Stage 2: Refinement
        with stage_model(model_config, "IdeationTeam", "RefinementStage"):
            stage2 = import_module("2-RefinementStage")
            orchestrator2 = stage2.RefinementOrchestrator(quiet=True, collector=collector)
            literature_df = orchestrator2.load_literature(sourcing_file)
            refined_questions = orchestrator2.run_automated_refinement(
                literature_df=literature_df,
                num_concepts=10,
                num_questions=8,
            )
        refinement_file = os.path.join(output_dir, "refinement_results_automated.json")
        orchestrator2.save_results(refinement_file)

        # Stage 3: Integration
        with stage_model(model_config, "IdeationTeam", "IntegrationStage"):
            stage3 = import_module("3-IntegrationStage")
            orchestrator3 = stage3.IntegrationOrchestrator(quiet=True, collector=collector)
            orchestrator3.output_dir = output_dir
            orchestrator3.research_topic = research_topic  # seed topic
            initial_questions = orchestrator3.load_refinement_results(refinement_file)
            final_questions = orchestrator3.run_automated_integration(
                questions=initial_questions,
                max_final_questions=5,
            )
        final_file = os.path.join(output_dir, "finalized_research_questions_automated.json")
        orchestrator3.save_final_questions(final_file)
        _citation_audit(final_file, "IdeationTeam", collector)

        # Load and return final output
        final_data = _load_json_safe(final_file) or {}
        return {"research_questions": final_data}


# ============================================================================
# LiteratureTeam Runner
# ============================================================================

def _n_items(obj, *keys) -> int:
    """Best-effort count of a list field on a dict-or-object (for gate context)."""
    for k in keys:
        v = obj.get(k) if isinstance(obj, dict) else getattr(obj, k, None)
        if isinstance(v, list):
            return len(v)
    return 0


def _run_literature_hitl(
    upstream_artifacts: Dict[str, Any],
    mode: str,
    output_dir: str,
    collector: Optional[MetricsCollector] = None,
) -> Dict[str, Any]:
    with _hitl_log_in(output_dir):
        return _run_literature_hitl_impl(upstream_artifacts, mode, output_dir, collector)


def _run_literature_hitl_impl(
    upstream_artifacts: Dict[str, Any],
    mode: str,
    output_dir: str,
    collector: Optional[MetricsCollector] = None,
) -> Dict[str, Any]:
    """Drive the WithHITL LiteratureTeam flow headlessly for the pipeline.

    Literature's HITL is an approve-gate between stages (not multi-field feedback), and
    its WithHITL stages expose the SAME pipeline methods as NoHITL — so this reuses them
    and inserts a committee vote gate after each stage. Gathering re-runs with more papers
    if the committee doesn't approve (bounded); gap/synthesis are review gates (the
    committee's verdict is logged, then the pipeline proceeds). main() is untouched.
    """
    from importlib import import_module

    team_base = Path(__file__).resolve().parent.parent / "LiteratureTeam" / "ael" / mode
    max_rounds = _hitl_max_rounds()

    with _team_context(team_base):
        research_questions = _extract_questions_list(upstream_artifacts)
        collector = collector or MetricsCollector()
        model_config = load_model_config()

        # Stage 1: Gathering — committee gate; re-gather more papers on non-approval.
        with stage_model(model_config, "LiteratureTeam", "LiteratureGatheringStage"):
            stage1 = import_module("1-LiteratureGatheringStage")
            fq = []
            for q in research_questions:
                if isinstance(q, dict):
                    fq.append(stage1.ResearchQuestion(**q))
                else:
                    fq.append(stage1.ResearchQuestion(question=str(q), priority_rank=1, priority_score=0.8))
            o1 = stage1.LiteratureGatheringOrchestrator(collector=collector)
        batch_file = os.path.join(output_dir, "literature_batch.json")
        max_papers = 15
        for rnd in range(1, max_rounds + 1):
            batch = o1.run_gathering_pipeline(research_questions=fq, max_papers_per_question=max_papers)
            o1.save_literature_batch(batch_file)
            ctx = {"n_papers": _n_items(batch, "literature_items", "items"), "max_papers": max_papers}
            if rnd == max_rounds or _stage_approved(rnd, ctx, "Is this gathered literature adequate to proceed?", auto_rounds=1):
                break
            max_papers = int(max_papers * 1.5)  # committee wants better coverage -> gather more

        # Stage 2: Gap Detection — committee review gate (verdict logged, then proceed).
        batch_dict = _load_json_safe(batch_file) or {}
        with stage_model(model_config, "LiteratureTeam", "GapDetectionStage"):
            o2 = import_module("2-GapDetectionStage").GapDetectionOrchestrator(collector=collector)
            gap_analysis = o2.run_gap_detection_pipeline(batch_dict)
        gap_file = os.path.join(output_dir, "gap_analysis_results.json")
        graph_file = os.path.join(output_dir, "knowledge_graph.json")
        o2.save_gap_analysis(gap_file)
        o2.save_graph_json(graph_file)
        if not _stage_approved(1, {"n_gaps": _n_items(gap_analysis, "gaps", "research_gaps")},
                               "Are the identified research gaps sound and complete?", auto_rounds=1):
            _log.info("LiteratureTeam GapDetection: committee did not fully approve; proceeding (verdict logged)")

        # Stage 3: Synthesis — committee review gate.
        gap_dict = _load_json_safe(gap_file) or {}
        with stage_model(model_config, "LiteratureTeam", "SynthesisStage"):
            o3 = import_module("3-SynthesisStage").SynthesisOrchestrator(collector=collector)
            # Bibliographic metadata comes from THIS run's batch, not the code dir.
            o3.run_synthesis_pipeline(
                gap_dict, literature_metadata=o3.load_literature_metadata(batch_file))
        review_file = os.path.join(output_dir, "literature_review.txt")
        synthesis_file = os.path.join(output_dir, "synthesis_results.json")
        o3.save_literature_review(review_file)
        o3.save_synthesis_result(synthesis_file)
        _citation_audit(synthesis_file, "LiteratureTeam", collector)
        if not _stage_approved(1, {"has_review": True}, "Is this literature review adequate to finalize?", auto_rounds=1):
            _log.info("LiteratureTeam Synthesis: committee did not fully approve; proceeding (verdict logged)")

        return {
            "literature_review": _load_json_safe(synthesis_file) or {},
            "gap_analysis": _load_json_safe(gap_file) or {},
            "knowledge_graph": _load_json_safe(graph_file) or {},
        }


def run_literature_team(
    upstream_artifacts: Dict[str, Any],
    mode: str,
    output_dir: str,
    collector: Optional[MetricsCollector] = None,
) -> Dict[str, Any]:
    """
    Run LiteratureTeam's 3-stage pipeline programmatically.

    Inputs (from upstream_artifacts):
        - "research_questions": dict with questions list

    Outputs:
        - "literature_review": dict with synthesis result
        - "gap_analysis": dict with gap analysis
        - "knowledge_graph": dict with knowledge graph
    """
    if isinstance(mode, str) and "WithHITL" in mode:
        return _run_literature_hitl(upstream_artifacts, mode, output_dir, collector)
    _reject_hitl_mode("LiteratureTeam", mode)
    team_base = Path(__file__).resolve().parent.parent / "LiteratureTeam" / "ael" / mode

    with _team_context(team_base):
        # Extract research questions from upstream
        research_questions = _extract_questions_list(upstream_artifacts)

        collector = collector or MetricsCollector()
        model_config = load_model_config()

        # Stage 1: Literature Gathering
        from importlib import import_module
        with stage_model(model_config, "LiteratureTeam", "LiteratureGatheringStage"):
            stage1 = import_module("1-LiteratureGatheringStage")

            # Convert dicts to ResearchQuestion Pydantic models
            formatted_questions = []
            for q in research_questions:
                if isinstance(q, dict):
                    formatted_questions.append(stage1.ResearchQuestion(**q))
                else:
                    formatted_questions.append(stage1.ResearchQuestion(
                        question=str(q), priority_rank=1, priority_score=0.8
                    ))

            orchestrator1 = stage1.LiteratureGatheringOrchestrator(collector=collector)
            literature_batch = orchestrator1.run_gathering_pipeline(
                research_questions=formatted_questions,
                max_papers_per_question=15,
            )
        batch_file = os.path.join(output_dir, "literature_batch.json")
        orchestrator1.save_literature_batch(batch_file)

        # Stage 2: Gap Detection
        literature_batch_dict = _load_json_safe(batch_file) or {}
        with stage_model(model_config, "LiteratureTeam", "GapDetectionStage"):
            stage2 = import_module("2-GapDetectionStage")
            orchestrator2 = stage2.GapDetectionOrchestrator(collector=collector)
            gap_analysis = orchestrator2.run_gap_detection_pipeline(literature_batch_dict)
        gap_file = os.path.join(output_dir, "gap_analysis_results.json")
        orchestrator2.save_gap_analysis(gap_file)
        graph_file = os.path.join(output_dir, "knowledge_graph.json")
        orchestrator2.save_graph_json(graph_file)

        # Stage 3: Synthesis
        gap_analysis_dict = _load_json_safe(gap_file) or {}
        with stage_model(model_config, "LiteratureTeam", "SynthesisStage"):
            stage3 = import_module("3-SynthesisStage")
            orchestrator3 = stage3.SynthesisOrchestrator(collector=collector)
            # Bibliographic metadata comes from THIS run's batch, not the code dir.
            synthesis_result = orchestrator3.run_synthesis_pipeline(
                gap_analysis_dict,
                literature_metadata=orchestrator3.load_literature_metadata(batch_file))
        review_file = os.path.join(output_dir, "literature_review.txt")
        orchestrator3.save_literature_review(review_file)
        synthesis_file = os.path.join(output_dir, "synthesis_results.json")
        orchestrator3.save_synthesis_result(synthesis_file)
        _citation_audit(synthesis_file, "LiteratureTeam", collector)

        # Build output artifacts
        outputs = {}
        outputs["literature_review"] = _load_json_safe(synthesis_file) or {}
        outputs["gap_analysis"] = _load_json_safe(gap_file) or {}
        outputs["knowledge_graph"] = _load_json_safe(graph_file) or {}

        return outputs


# ============================================================================
# ModelTeam Runner
# ============================================================================

def _run_model_hitl(
    upstream_artifacts: Dict[str, Any],
    mode: str,
    output_dir: str,
    collector: Optional[MetricsCollector] = None,
    drs_only: bool = False,
) -> Dict[str, Any]:
    """Drive the WithHITL ModelTeam flow headlessly for the pipeline.

    Model's HITL is an approve-gate after each stage (Theory/Design/Calibration) that
    blocks on rejection. Committee gates each; ModelDesign RE-RUNS with the committee's
    feedback as a revision_request — reusing the feasibility-loop revision mechanism (2b)
    — until approved (bounded). An incoming feasibility MRR seeds the first design
    revision. Theory/Calibration are review gates (verdict logged, then proceed). Uses
    the same stage pipeline methods as NoHITL; main() untouched.
    """
    from importlib import import_module

    team_base = Path(__file__).resolve().parent.parent / "ModelTeam" / "ael" / mode
    max_rounds = _hitl_max_rounds()

    with _team_context(team_base):
        questions = _extract_questions_list(upstream_artifacts)
        literature_data = upstream_artifacts.get("literature_review", {})
        if not isinstance(literature_data, dict):
            literature_data = {}
        # a feasibility MRR (if the loop re-entered ModelTeam) seeds the first revision
        revision = upstream_artifacts.get("model_revision_request")
        collector = collector or MetricsCollector()
        model_config = load_model_config()

        # Stage 1: Theory. On a revision cycle (a feasibility MRR is present) the research questions
        # are unchanged — only data availability differs — so REUSE the prior theory instead of
        # regenerating it, and skip the review gate. Fresh theory (+ gate) only on the first,
        # no-revision pass. Mirrors run_model_team's NoHITL reuse; a big saving in the loop.
        theory_file = os.path.join(output_dir, "theory_output.json")
        if revision and os.path.exists(theory_file):
            _log.info("ModelTeam revision cycle: reusing prior theory")
            theory_data = _load_json_safe(theory_file) or {}
        else:
            with stage_model(model_config, "ModelTeam", "TheoryStage"):
                o1 = import_module("1-TheoryStage").TheoryStageOrchestrator(collector=collector)
                o1.run_theory_pipeline(research_questions=questions, literature_batch=literature_data)
            o1.save_theory_output(theory_file)
            theory_data = _load_json_safe(theory_file) or {}
            if not _stage_approved(1, {"has_theory": bool(theory_data)},
                                   "Is this theoretical framework sound and well-motivated?", auto_rounds=1):
                _log.info("ModelTeam Theory: committee did not fully approve; proceeding (verdict logged)")

        # Stage 2: Model Design — committee gate + re-run-until-approved (feasibility-loop mechanism).
        # On a drs_only feasibility cycle the design already passed the committee in the linear pass
        # and the MRR is a MECHANICAL data-availability edit, so apply it ONCE with NO committee
        # re-refinement — the loop only needs the DRS, not a re-approved design.
        design_file = os.path.join(output_dir, "model_design_output.json")
        _design_rounds = 1 if drs_only else max_rounds
        with stage_model(model_config, "ModelTeam", "ModelDesignStage"):
            o2 = import_module("2-ModelDesignStage").ModelDesignOrchestrator(collector=collector)
            for rnd in range(1, _design_rounds + 1):
                o2.run_design_pipeline(theory_output=theory_data, revision_request=revision)
                o2.save_design_output(design_file)
                design_data = _load_json_safe(design_file) or {}
                if rnd == _design_rounds:
                    break
                fm = (design_data.get("formal_models") or [{}])[0] if isinstance(design_data, dict) else {}
                ctx = {"n_variables": _n_items(fm, "variables"), "n_equations": _n_items(fm, "equations")}
                if _stage_approved(rnd, ctx,
                        "Is this model design sound, internally coherent, and identifiable?", auto_rounds=1):
                    break
                fb = _committee_feedback("What must change in the model design to make it sound and identifiable?", ctx)
                # The committee's change is ADDED to the data-feasibility request, not swapped
                # in for it — replacing it let the final design reintroduce the removed variables.
                feas = upstream_artifacts.get("model_revision_request") or {}
                revision = {
                    "reason": " | ".join(x for x in (feas.get("reason"),
                                                     "Committee review did not approve the model design.") if x),
                    "requested_change": "\n".join(x for x in (feas.get("requested_change"),
                                                              f"(committee) {fb}" if fb else "") if x),
                    "unmet_requirements": list(feas.get("unmet_requirements") or []),
                }
        design_data = _load_json_safe(design_file) or {}

        # Stage 3: Calibration — targets-only on a feasibility DRS cycle (skip estimation).
        calibration_file = os.path.join(output_dir, "calibration_output.json")
        with stage_model(model_config, "ModelTeam", "CalibrationStage"):
            o3 = import_module("3-CalibrationStage").CalibrationOrchestrator(collector=collector)
            o3.run_calibration_pipeline(model_design_output=design_data, targets_only=drs_only)
        o3.save_calibration_output(calibration_file)
        if drs_only:
            # DRS-only cycle: no calibration to review; the DRS is what the loop needs.
            return {"model_specification": _load_json_safe(calibration_file) or {}, "model_design": design_data}
        if not _stage_approved(1, {"has_calibration": True},
                               "Is this calibration credible and honestly reported?", auto_rounds=1):
            _log.info("ModelTeam Calibration: committee did not fully approve; proceeding (verdict logged)")

        return {"model_specification": _load_json_safe(calibration_file) or {}, "model_design": design_data}


def run_model_team(
    upstream_artifacts: Dict[str, Any],
    mode: str,
    output_dir: str,
    collector: Optional[MetricsCollector] = None,
    drs_only: bool = False,
) -> Dict[str, Any]:
    """
    Run ModelTeam's 3-stage pipeline programmatically.

    Inputs (from upstream_artifacts):
        - "research_questions": dict with questions
        - "literature_review": dict with literature review (optional)

    Outputs:
        - "model_specification": dict with calibrated models

    ``drs_only=True`` (feasibility-loop cycle) runs Calibration in targets-only mode — extract
    empirical targets, skip parameter estimation — so the loop gets the DRS fast. Full
    calibration runs once at convergence.
    """
    if isinstance(mode, str) and "WithHITL" in mode:
        return _run_model_hitl(upstream_artifacts, mode, output_dir, collector, drs_only=drs_only)
    _reject_hitl_mode("ModelTeam", mode)
    team_base = Path(__file__).resolve().parent.parent / "ModelTeam" / "ael" / mode

    with _team_context(team_base):
        # Extract research questions
        questions = _extract_questions_list(upstream_artifacts)

        # Extract literature data as dict (optional)
        literature_data = upstream_artifacts.get("literature_review", {})
        if not isinstance(literature_data, dict):
            literature_data = {}

        collector = collector or MetricsCollector()
        model_config = load_model_config()

        # Data-feasibility revision (design 2b): on a revision cycle the MRR targets the
        # design — reuse the prior theory rather than regenerating a fresh framework.
        from importlib import import_module
        revision_request = upstream_artifacts.get("model_revision_request")
        theory_file = os.path.join(output_dir, "theory_output.json")

        # Stage 1: Theory (skipped on a revision cycle when prior theory exists)
        if revision_request and os.path.exists(theory_file):
            _log.info("ModelTeam revision cycle: reusing prior theory; applying MRR to design")
        else:
            with stage_model(model_config, "ModelTeam", "TheoryStage"):
                stage1 = import_module("1-TheoryStage")
                orchestrator1 = stage1.TheoryStageOrchestrator(collector=collector)
                orchestrator1.run_theory_pipeline(
                    research_questions=questions,
                    literature_batch=literature_data,
                )
            orchestrator1.save_theory_output(theory_file)

        # Stage 2: Model Design (applies the MRR directive when present)
        theory_data = _load_json_safe(theory_file) or {}
        with stage_model(model_config, "ModelTeam", "ModelDesignStage"):
            stage2 = import_module("2-ModelDesignStage")
            orchestrator2 = stage2.ModelDesignOrchestrator(collector=collector)
            orchestrator2.run_design_pipeline(
                theory_output=theory_data,
                revision_request=revision_request,
            )
        design_file = os.path.join(output_dir, "model_design_output.json")
        orchestrator2.save_design_output(design_file)

        # Stage 3: Calibration (targets-only on a feasibility DRS cycle — skip estimation)
        design_data = _load_json_safe(design_file) or {}
        with stage_model(model_config, "ModelTeam", "CalibrationStage"):
            stage3 = import_module("3-CalibrationStage")
            orchestrator3 = stage3.CalibrationOrchestrator(collector=collector)
            orchestrator3.run_calibration_pipeline(
                model_design_output=design_data,
                targets_only=drs_only,
            )
        calibration_file = os.path.join(output_dir, "calibration_output.json")
        orchestrator3.save_calibration_output(calibration_file)

        # Expose the design output too (feasibility loop derives the DRS from the
        # model's exogenous variables + calibration targets — design decision A).
        return {
            "model_specification": _load_json_safe(calibration_file) or {},
            "model_design": design_data,
        }


# ============================================================================
# DataTeam Runner
# ============================================================================

def run_data_team(
    upstream_artifacts: Dict[str, Any],
    mode: str,
    output_dir: str,
    collector: Optional[MetricsCollector] = None,
) -> Dict[str, Any]:
    """
    Run DataTeam's 3-stage pipeline programmatically.

    Inputs (from upstream_artifacts):
        - "research_questions": dict with questions (extracts first question)
        - "model_specification": dict with model specs (extracts data requirements)

    Outputs:
        - "validated_dataset": dict with documented datasets
    """
    # DataTeam uses different directory naming
    mode_dir_map = {
        "open_source_api": "ModeOpenSourceAPI",
        "premium_subscribed": "ModePremiumSubscribed",
        "user_uploaded": "ModeUserUploaded",
        # Also accept the directory names directly
        "ModeOpenSourceAPI": "ModeOpenSourceAPI",
        "ModePremiumSubscribed": "ModePremiumSubscribed",
        "ModeUserUploaded": "ModeUserUploaded",
    }
    dir_name = mode_dir_map.get(mode, "ModeOpenSourceAPI")

    team_base = Path(__file__).resolve().parent.parent / "DataTeam" / "ael" / dir_name

    with _team_context(team_base):
        # Extract research question (first question from list)
        questions_list = _extract_questions_list(upstream_artifacts)
        if questions_list:
            first_q = questions_list[0]
            research_question = (
                first_q.get("question", str(first_q)) if isinstance(first_q, dict) else str(first_q)
            )
        else:
            # A fallback question must not impose a field; label it as the
            # degenerate no-upstream case explicitly.
            research_question = (
                "(fallback: no upstream research question was provided) Characterize the "
                "key empirical relationships among the core indicators available in the "
                "selected open data sources."
            )

        # The requirements retrieval serves — the feasibility loop's
        # data_requirements_spec, else the DRS derived from model_design + model_specification.
        # (This used to read model_specification["data_requirements"], a key no team writes,
        # so retrieval ran with no requirements in 21/21 runs.)
        from DataTeam.ael.feasibility.adapters import requirements_for_retrieval
        data_requirements = requirements_for_retrieval(upstream_artifacts, research_question)
        print(f"[DataTeam] {len(data_requirements)} retrievable data requirement(s) passed "
              "to the source stage")

        # Available APIs — configurable via AEL_DATA_APIS env var
        api_list_str = os.environ.get("AEL_DATA_APIS", "FRED,World Bank,BLS,Census")
        available_apis = [api.strip() for api in api_list_str.split(",")]

        collector = collector or MetricsCollector()
        model_config = load_model_config()

        # DataTeam's stage HITL (query/quality/final review checkpoints) is gated by
        # enable_hitl. Turn it on when a human or the committee is answering (interactive
        # / llm_economist), off for auto (cheap infra smoke). The Model<->Data feasibility
        # loop is DataTeam's primary committee HITL and runs separately (orchestrator).
        from shared.auto_input import get_hitl_mode
        _data_hitl = get_hitl_mode() in ("llm_economist", "interactive")

        # Stage 1: Data Source. The three workflows expose the SAME class/method but with
        # workflow-specific signatures (open_source_api takes data_requirements/available_apis;
        # premium takes budget/credentials; user_uploaded REQUIRES file_path) — so build the
        # full kwargs set and filter by the actual signature. Blindly passing the open-source
        # kwargs was an instant TypeError in the other two workflows.
        import inspect
        from importlib import import_module
        with stage_model(model_config, "DataTeam", "DataSourceStage"):
            stage1 = import_module("1-DataSourceStage")
            orchestrator1 = stage1.DataSourceOrchestrator(collector=collector)
            _supp = upstream_artifacts.get("supplemental_series") or []
            if isinstance(_supp, dict):   # registered artifact form: {"series": [...]}
                _supp = _supp.get("series") or []
            _desired = {
                "research_question": research_question,
                "data_requirements": data_requirements,
                "available_apis": available_apis,
                "enable_hitl": _data_hitl,
                # user_uploaded: the researcher's dataset; env override, else the sample
                "file_path": os.environ.get(
                    "AEL_USER_DATA_PATH", str(team_base / "example_data.csv")),
                # Curated connector ids the feasibility scout matched OUTSIDE the
                # retrieved pool — open_source_api fetches them deterministically.
                "supplemental_series": _supp if isinstance(_supp, list) else [],
            }
            _sig = inspect.signature(orchestrator1.run_source_pipeline)
            _accepts_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD
                              for p in _sig.parameters.values())
            _kwargs = _desired if _accepts_kw else {
                k: v for k, v in _desired.items() if k in _sig.parameters}
            source_output = orchestrator1.run_source_pipeline(**_kwargs)
        source_file = os.path.join(output_dir, "api_source_output.json")
        orchestrator1.save_source_output(source_file)

        # Stage 2: Data Cleaning
        source_data = _load_json_safe(source_file) or {}
        with stage_model(model_config, "DataTeam", "DataCleaningStage"):
            stage2 = import_module("2-DataCleaningStage")
            orchestrator2 = stage2.DataCleaningOrchestrator(collector=collector)
            orchestrator2.run_cleaning_pipeline(
                data_source_output=source_data,
                research_question=research_question,
                enable_hitl=_data_hitl,
            )
        cleaning_file = os.path.join(output_dir, "api_cleaning_output.json")
        orchestrator2.save_cleaning_output(cleaning_file)

        # Stage 3: Quality Assurance
        cleaning_data = _load_json_safe(cleaning_file) or {}
        with stage_model(model_config, "DataTeam", "QualityAssuranceStage"):
            stage3 = import_module("3-QualityAssuranceStage")
            orchestrator3 = stage3.QualityAssuranceOrchestrator(collector=collector)
            orchestrator3.run_qa_pipeline(
                data_cleaning_output=cleaning_data,
                research_question=research_question,
                enable_hitl=_data_hitl,
            )
        qa_file = os.path.join(output_dir, "api_qa_output.json")
        orchestrator3.save_qa_output(qa_file)

        # Expose the source output too (feasibility scout grades the DRS against the
        # series this tier actually supplied).
        return {
            "validated_dataset": _load_json_safe(qa_file) or {},
            "data_source": source_data,
        }


# ============================================================================
# EstimationTeam Runner
# ============================================================================

def run_estimation_team(
    upstream_artifacts: Dict[str, Any],
    mode: str,
    output_dir: str,
    collector: Optional[MetricsCollector] = None,
) -> Dict[str, Any]:
    """
    Run EstimationTeam's 3-stage pipeline programmatically.

    Runs AFTER the Model<->Data feasibility loop (the orchestrator defers it), so it
    estimates the FINAL model specification on the data the pipeline actually retrieved.

    Inputs (from upstream_artifacts):
        - "research_questions": dict with questions (first question used)
        - "model_specification": the ModelTeam's calibrated specification
        - "data_source": DataTeam stage-1 output carrying retrieved_data provenance
          (falls back to "validated_dataset" if data_source is absent)
        - "executable_model" (optional): the CodeTeam's modules; the lead model's module
          is shown to the Estimator and the part used is recorded
        - "model_design" (optional): names for the module variables

    Outputs:
        - "estimation_results": InferenceStageOutput dict (final verdict + estimates +
          diagnostics-adjusted status + hypothesis/robustness results), plus
          "executable_model_use" recording which module variables the specification used,
          "fitted_specifications" (every spec the harness fitted, its verdict, which one is
          reported and why) and, with a committee, "spec_review" / "committee_objections"

    EstimationTeam has no Wc axis: any WithHITL pipeline mode maps to ModeNoWcWithHITL,
    everything else to ModeNoWcNoHITL. In WithHITL + committee/interactive HITL, the
    committee votes on the proposed specification BEFORE estimation; a rejection triggers
    ONE bounded revision that is re-voted (stage 1 ``run_reviewed_pipeline``) — mirroring the
    standalone WithHITL MasterOrchestrator.
    """
    dir_name = "ModeNoWcWithHITL" if "WithHITL" in str(mode) else "ModeNoWcNoHITL"
    team_base = Path(__file__).resolve().parent.parent / "EstimationTeam" / "ael" / dir_name

    with _team_context(team_base):
        questions_list = _extract_questions_list(upstream_artifacts)
        if questions_list:
            first_q = questions_list[0]
            research_question = (
                first_q.get("question", str(first_q)) if isinstance(first_q, dict) else str(first_q)
            )
        else:
            research_question = ""

        model_spec = upstream_artifacts.get("model_specification") or {}
        data_artifact = (upstream_artifacts.get("data_source")
                         or upstream_artifacts.get("validated_dataset") or {})
        executable_model = upstream_artifacts.get("executable_model") or None
        model_design = upstream_artifacts.get("model_design") or None

        collector = collector or MetricsCollector()
        model_config = load_model_config()

        from shared.auto_input import get_hitl_mode
        _hitl = "WithHITL" in dir_name and get_hitl_mode() in ("llm_economist", "interactive")

        from importlib import import_module

        # Stage 1: Estimation (LLM proposes, deterministic harness disposes)
        with stage_model(model_config, "EstimationTeam", "EstimationStage"):
            stage1 = import_module("1-EstimationStage")
            orchestrator1 = stage1.EstimationOrchestrator(collector=collector,
                                                          output_dir=output_dir)
            if _hitl:
                # Specification first, estimation second:
                # the committee votes on the PROPOSED specification — shown the spec, the
                # available series and the observation-budget arithmetic, never coefficients,
                # p-values or fit. A rejected proposal gets ONE revision, which is re-voted;
                # if that is rejected too, the original is estimated and the objections are
                # recorded. The stage records every fitted specification.
                def _review(ctx: Dict[str, Any], round_number: int):
                    fb = _collect_feedback(orchestrator1.collect_human_feedback, ctx,
                                           round_number=round_number)
                    approved = _stage_approved(
                        round_number, ctx,
                        "Approve this PROPOSED estimation specification (not yet estimated) "
                        f"as a fair empirical test for: {research_question}",
                        auto_rounds=round_number)
                    return approved, str(fb or "")

                out1 = orchestrator1.run_reviewed_pipeline(
                    model_spec_data=model_spec,
                    data_artifact=data_artifact,
                    review=_review,
                    research_question=research_question,
                    executable_model=executable_model,
                    model_design=model_design,
                )
            else:
                out1 = orchestrator1.run_estimation_pipeline(
                    model_spec_data=model_spec,
                    data_artifact=data_artifact,
                    research_question=research_question,
                    executable_model=executable_model,
                    model_design=model_design,
                )
        estimation_file = os.path.join(output_dir, "estimation_output.json")
        orchestrator1.save_estimation_output(estimation_file)

        # Stage 2: Validation & Diagnostics (deterministic battery; may downgrade verdict)
        estimation_data = _load_json_safe(estimation_file) or {}
        with stage_model(model_config, "EstimationTeam", "ValidationDiagnosticsStage"):
            stage2 = import_module("2-ValidationDiagnosticsStage")
            orchestrator2 = stage2.ValidationOrchestrator(collector=collector)
            orchestrator2.run_validation_pipeline(estimation_data)
        validation_file = os.path.join(output_dir, "validation_output.json")
        orchestrator2.save_validation_output(validation_file)

        # Stage 3: Inference & Robustness (hypothesis tests + stability sweeps)
        validation_data = _load_json_safe(validation_file) or {}
        with stage_model(model_config, "EstimationTeam", "InferenceRobustnessStage"):
            stage3 = import_module("3-InferenceRobustnessStage")
            orchestrator3 = stage3.InferenceOrchestrator(collector=collector)
            orchestrator3.run_inference_pipeline(
                validation_data, research_question=research_question,
            )
        inference_file = os.path.join(output_dir, "inference_output.json")
        orchestrator3.save_inference_output(inference_file)

        results = _load_json_safe(inference_file) or {}
        # Stages 2-3 carry the outcome only; the executable-model record comes from stage 1
        stage1_meta = estimation_data.get("metadata") or {}
        results["executable_model_use"] = stage1_meta.get("executable_model_use", {})
        # Every fitted specification (and the committee's review record) reaches the report
        for key in ("fitted_specifications", "spec_review", "committee_objections"):
            if key in stage1_meta:
                results[key] = stage1_meta[key]
        return {"estimation_results": results}


# ============================================================================
# ReportingTeam Runner
# ============================================================================

def run_reporting_team(
    upstream_artifacts: Dict[str, Any],
    mode: str,
    output_dir: str,
    collector: Optional[MetricsCollector] = None,
) -> Dict[str, Any]:
    """
    Run ReportingTeam's 3-stage pipeline programmatically — the pipeline's terminal node.

    Runs AFTER EstimationTeam (both deferred past the feasibility loop), assembling the
    final research report from every upstream artifact, with the deterministic
    number-consistency check annotated on the report itself.

    Inputs (from upstream_artifacts): research_questions, literature_review,
        model_specification, data_source, estimation_results
    Outputs: "research_report": {report_markdown, consistency, report_file, figures}

    No Wc axis: any WithHITL pipeline mode maps to ModeNoWcWithHITL (one bounded narrative
    revision at the committee checkpoint), everything else to ModeNoWcNoHITL.
    """
    dir_name = "ModeNoWcWithHITL" if "WithHITL" in str(mode) else "ModeNoWcNoHITL"
    team_base = Path(__file__).resolve().parent.parent / "ReportingTeam" / "ael" / dir_name

    with _team_context(team_base):
        questions_list = _extract_questions_list(upstream_artifacts)
        research_question = ""
        if questions_list:
            first_q = questions_list[0]
            research_question = (
                first_q.get("question", str(first_q)) if isinstance(first_q, dict) else str(first_q)
            )

        research_questions = upstream_artifacts.get("research_questions") or {}
        literature_review = upstream_artifacts.get("literature_review") or {}
        model_specification = upstream_artifacts.get("model_specification") or {}
        data_source = upstream_artifacts.get("data_source") or {}
        estimation_results = upstream_artifacts.get("estimation_results") or {}
        feasibility_report = _load_json_safe(
            os.path.join(output_dir, "..", "feasibility_report.json"))
        # the reviewed corpus -> References section (+ consistency pool for years/URLs)
        literature_batch = _load_json_safe(
            os.path.join(output_dir, "..", "LiteratureTeam", "literature_batch.json")) or {}
        # Generated-code inventory -> report Code Implementation section (optional —
        # absent when the CodeTeam node is disabled; the section is then omitted)
        code_generation = _load_json_safe(
            os.path.join(output_dir, "..", "CodeTeam", "generation_output.json"))
        code_validation = _load_json_safe(
            os.path.join(output_dir, "..", "CodeTeam", "validation_output.json"))

        collector = collector or MetricsCollector()
        model_config = load_model_config()

        from shared.auto_input import get_hitl_mode
        _hitl = "WithHITL" in dir_name and get_hitl_mode() in ("llm_economist", "interactive")

        from importlib import import_module

        # Stage 1: Interpretation (deterministic magnitudes + figures)
        with stage_model(model_config, "ReportingTeam", "InterpretationStage"):
            stage1 = import_module("1-InterpretationStage")
            orchestrator1 = stage1.InterpretationOrchestrator(collector=collector)
            orchestrator1.run_interpretation_pipeline(
                estimation_results, research_question=research_question,
                figures_dir=os.path.join(output_dir, "figures"))
        interpretation_file = os.path.join(output_dir, "interpretation_output.json")
        orchestrator1.save_interpretation_output(interpretation_file)
        interpretation_output = _load_json_safe(interpretation_file) or {}

        # Stage 2: Drafting (deterministic assembly + narrative; bounded HITL revision)
        report_file = os.path.join(output_dir, "research_report.md")
        with stage_model(model_config, "ReportingTeam", "DraftingStage"):
            stage2 = import_module("2-DraftingStage")
            orchestrator2 = stage2.DraftingOrchestrator(collector=collector,
                                                        output_dir=output_dir)
            out2 = orchestrator2.run_drafting_pipeline(
                research_questions, literature_review, model_specification, data_source,
                estimation_results, interpretation_output,
                feasibility_report=feasibility_report, report_file=report_file,
                literature_batch=literature_batch,
                code_generation=code_generation, code_validation=code_validation)
            if _hitl:
                # A rejected draft gets ONE revision, which is re-voted; if
                # the revision is rejected too, it is published with the committee's
                # objections recorded in Limitations. No further rounds.
                def _draft_vote(out, round_number):
                    ctx = {"stage": "DraftingStage",
                           "research_question": research_question,
                           "report_excerpt": out.report_markdown[:2500],
                           "narratives": out.drafting.narratives}
                    fb = _collect_feedback(orchestrator2.collect_human_feedback, ctx,
                                           round_number=round_number)
                    approved = _stage_approved(
                        round_number, ctx,
                        f"Approve this research report draft for: {research_question}",
                        auto_rounds=round_number)
                    return approved, str(fb or "")

                approved, fb = _draft_vote(out2, 1)
                if not approved and fb:
                    out2 = orchestrator2.run_drafting_pipeline(
                        research_questions, literature_review, model_specification, data_source,
                        estimation_results, interpretation_output,
                        feasibility_report=feasibility_report, report_file=report_file,
                        feedback=fb, literature_batch=literature_batch,
                        code_generation=code_generation, code_validation=code_validation)
                    approved2, fb2 = _draft_vote(out2, 2)
                    if not approved2:
                        out2 = orchestrator2.publish_with_objections(
                            [f"objections to the first draft: {fb}",
                             f"objections to the revised draft: {fb2 or '(no feedback)'}"])
                elif not approved:
                    out2 = orchestrator2.publish_with_objections(
                        ["the committee did not approve the draft and gave no feedback"])
        drafting_file = os.path.join(output_dir, "drafting_output.json")
        orchestrator2.save_drafting_output(drafting_file)

        # Stage 3: Quality (number-consistency check, annotated on the report)
        drafting_output = _load_json_safe(drafting_file) or {}
        with stage_model(model_config, "ReportingTeam", "QualityStage"):
            stage3 = import_module("3-QualityStage")
            orchestrator3 = stage3.QualityOrchestrator(collector=collector)
            out3 = orchestrator3.run_quality_pipeline(
                drafting_output,
                source_artifacts={
                    "research_questions": research_questions,
                    "literature_review": literature_review,
                    "model_specification": model_specification,
                    "data_source": data_source,
                    "estimation_results": estimation_results,
                    "interpretation": interpretation_output,
                    "literature_batch": literature_batch,
                    # The code inventory's counts must be verifiable
                    "code_generation": code_generation or {},
                    "code_validation": code_validation or {}},
                # Without these the availability statement says "No retrieved-data
                # artifact was available" and the formatted report is written into the
                # code tree.
                research_question=research_question,
                data_source=data_source,
                literature_batch=literature_batch,
                formatted_report_file=os.path.join(output_dir, "research_report_formatted.md"))
        quality_file = os.path.join(output_dir, "quality_output.json")
        orchestrator3.save_quality_output(quality_file)

        return {
            "research_report": {
                "report_file": report_file,
                "report_markdown": out3.report_markdown,
                "consistency": out3.quality.consistency.model_dump(),
                "figures": (interpretation_output.get("interpretation") or {}).get(
                    "figure_files", []),
            },
        }


# ============================================================================
# CodeTeam Runner (optional node, disabled by default)
# ============================================================================

def run_code_team(
    upstream_artifacts: Dict[str, Any],
    mode: str,
    output_dir: str,
    collector: Optional[MetricsCollector] = None,
) -> Dict[str, Any]:
    """
    Run CodeTeam's 3-stage pipeline programmatically (optional node).

    Deferred past the feasibility loop with Estimation/Reporting so it derives code from
    the FINAL (possibly revised) model. The module is derived from the parsed sympy system
    — no LLM-written code.

    Inputs (from upstream_artifacts): model_design (+ model_specification for values)
    Outputs: "executable_model": {generation, validation, experimentation}
    """
    dir_name = "ModeNoWcWithHITL" if "WithHITL" in str(mode) else "ModeNoWcNoHITL"
    team_base = Path(__file__).resolve().parent.parent / "CodeTeam" / "ael" / dir_name

    with _team_context(team_base):
        model_design = upstream_artifacts.get("model_design") or {}
        calibration = upstream_artifacts.get("model_specification") or {}
        collector = collector or MetricsCollector()
        model_config = load_model_config()
        modules_dir = os.path.join(output_dir, "generated_models")

        from importlib import import_module

        with stage_model(model_config, "CodeTeam", "CodeGenerationStage"):
            stage1 = import_module("1-CodeGenerationStage")
            orchestrator1 = stage1.CodeGenerationOrchestrator(collector=collector)
            orchestrator1.run_generation_pipeline(model_design, calibration,
                                                  modules_dir=modules_dir)
        generation_file = os.path.join(output_dir, "generation_output.json")
        orchestrator1.save_generation_output(generation_file)
        generation_data = _load_json_safe(generation_file) or {}

        with stage_model(model_config, "CodeTeam", "ValidationStage"):
            stage2 = import_module("2-ValidationStage")
            orchestrator2 = stage2.CodeValidationOrchestrator(collector=collector)
            orchestrator2.run_validation_pipeline(generation_data, workdir=modules_dir)
        validation_file = os.path.join(output_dir, "validation_output.json")
        orchestrator2.save_validation_output(validation_file)
        validation_data = _load_json_safe(validation_file) or {}

        with stage_model(model_config, "CodeTeam", "ExperimentationStage"):
            stage3 = import_module("3-ExperimentationStage")
            orchestrator3 = stage3.ExperimentationOrchestrator(collector=collector)
            orchestrator3.run_experimentation_pipeline(generation_data, validation_data,
                                                       workdir=modules_dir)
        experimentation_file = os.path.join(output_dir, "experimentation_output.json")
        orchestrator3.save_experimentation_output(experimentation_file)

        return {
            "executable_model": {
                "generation": generation_data,
                "validation": validation_data,
                "experimentation": _load_json_safe(experimentation_file) or {},
            },
        }


# ============================================================================
# Registration
# ============================================================================

def register_all_runners():
    """Register all team runners with the pipeline orchestrator."""
    from pipeline.research_pipeline import register_team_runner

    register_team_runner("IdeationTeam", run_ideation_team)
    register_team_runner("LiteratureTeam", run_literature_team)
    register_team_runner("ModelTeam", run_model_team)
    register_team_runner("EstimationTeam", run_estimation_team)
    register_team_runner("ReportingTeam", run_reporting_team)
    register_team_runner("CodeTeam", run_code_team)
    register_team_runner("DataTeam", run_data_team)
