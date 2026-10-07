# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Drafting Stage — deterministic report assembly with LLM narrative blocks.

Pipeline:
1. Reporter (LLM) writes two clearly delimited narrative blocks (intro, discussion),
   grounded on deterministic summaries of the artifacts
2. assemble_report builds the full markdown: every table/number formatted directly from the
   artifacts; narratives inserted at marked slots; Limitations auto-collected

Input:  research_questions, literature_review, model_specification, data_source,
        estimation_results, interpretation_output
Output: drafting_output.json + research_report.md

This file is byte-identical across ReportingTeam mode directories (mode-copy invariant).
"""

import json
import os
import sys
from pathlib import Path as _Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

_agents_dir = _Path(__file__).resolve().parent.parent.parent.parent
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

from shared.auto_input import auto_input, get_default
from shared.llm import LLMClient
from shared.observability import MetricsCollector

from ReportingTeam.ael.report_harness import assemble_report, derived_numbers
from ReportingTeam.ael.report_harness.types import DraftingResult, InterpretationResult
from ReportingTeam.ael.schemas.stage_outputs import DraftingStageOutput

load_dotenv()


# ========== AGENT ==========

class Reporter:
    """Writes narrative blocks; every number it uses is verified by Stage 3."""

    def __init__(self, openai_api_key: str = "", collector: Optional[MetricsCollector] = None):
        self.agent_name = "Reporter"
        self.llm = LLMClient(
            temperature=0.4,
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
            collector=collector,
            agent_name=self.agent_name,
        )

    def write_block(self, slot: str, grounding: str, research_question: str,
                    feedback: Optional[str] = None) -> str:
        guidance = f"\nReviewer feedback to incorporate:\n{feedback[:1200]}\n" if feedback else ""
        raw = self.llm.invoke([
            {"role": "system", "content":
                "You write one section of an economics research report. Ground every claim in "
                "the material provided. If you cite a number, take it from the material and "
                "ROUND it to at most three significant figures (write 0.319, not "
                "0.3185268864789781) — never compute, extrapolate, or invent numbers (a "
                "deterministic checker verifies every number you write, and it accepts "
                "correct roundings)."},
            {"role": "user", "content":
                f"Research question: {research_question[:300]}\n\n"
                f"Section to write: {slot}\n\nMaterial:\n{grounding[:2800]}\n{guidance}\n"
                "Write the section in <=220 words of flowing prose (no headers, no lists)."},
        ], max_tokens=550)
        return str(raw).strip()


# ========== ORCHESTRATOR ==========

class DraftingOrchestrator:
    """Assembles the research report from artifacts + narrative blocks."""

    def __init__(self, openai_api_key: str = "", collector: Optional[MetricsCollector] = None,
                 quiet: bool = False, output_dir: Optional[str] = None):
        self.collector = collector
        self.quiet = quiet
        self.reporter = Reporter(openai_api_key, collector=collector)
        self.output: Optional[DraftingStageOutput] = None
        # Feedback files go to the run's output directory
        self.output_dir = output_dir
        self._last_inputs: Dict = {}

    def _print(self, msg: str):
        if not self.quiet:
            print(msg)

    @staticmethod
    def _question(research_questions: Dict) -> str:
        for key in ("final_questions", "questions", "prioritized", "prioritized_questions"):
            lst = (research_questions or {}).get(key)
            if isinstance(lst, list) and lst:
                q = lst[0]
                return q.get("question", str(q)) if isinstance(q, dict) else str(q)
        return ""

    def run_drafting_pipeline(
        self,
        research_questions: Dict,
        literature_review: Dict,
        model_specification: Dict,
        data_source: Dict,
        estimation_results: Dict,
        interpretation_output: Dict,
        feasibility_report: Optional[Dict] = None,
        pipeline_run_id: str = "",
        report_file: str = "research_report.md",
        feedback: Optional[str] = None,
        literature_batch: Optional[Dict] = None,
        code_generation: Optional[Dict] = None,
        code_validation: Optional[Dict] = None,
        report_objections: Optional[List[str]] = None,
        narratives: Optional[Dict[str, str]] = None,
    ) -> DraftingStageOutput:
        """``narratives`` given: reuse them (no LLM call) — used to re-assemble a draft with
        the committee's unresolved objections added to Limitations."""
        self._last_inputs = dict(
            research_questions=research_questions, literature_review=literature_review,
            model_specification=model_specification, data_source=data_source,
            estimation_results=estimation_results, interpretation_output=interpretation_output,
            feasibility_report=feasibility_report, pipeline_run_id=pipeline_run_id,
            report_file=report_file, literature_batch=literature_batch,
            code_generation=code_generation, code_validation=code_validation)
        question = self._question(research_questions)
        interpretation = InterpretationResult(
            **(interpretation_output.get("interpretation") or {}))

        # LLM narrative blocks, grounded on deterministic summaries (never raw invention)
        reuse = narratives is not None
        narratives = dict(narratives or {})
        outcome = (estimation_results or {}).get("outcome") or {}
        grounding = json.dumps({
            "verdict": outcome.get("verdict"),
            "coefficients": outcome.get("coefficients"),
            "effects": (interpretation_output.get("interpretation") or {}).get("effects"),
            "hypotheses": (estimation_results.get("inference") or {}).get("hypotheses"),
            "n_obs": outcome.get("n_obs"), "r_squared": outcome.get("r_squared"),
        }, default=str)
        for slot in (() if reuse else ("intro", "discussion")):
            try:
                narratives[slot] = self.reporter.write_block(
                    slot, grounding, question, feedback=feedback)
            except Exception as e:
                self._print(f"[DraftingStage] '{slot}' narrative skipped (non-critical): {e}")

        self._print("[DraftingStage] Assembling the report (deterministic skeleton)...")
        markdown = assemble_report(
            research_questions, literature_review, model_specification, data_source,
            estimation_results, interpretation, narratives=narratives,
            feasibility_report=feasibility_report, pipeline_run_id=pipeline_run_id,
            literature_batch=literature_batch,
            code_generation=code_generation, code_validation=code_validation,
            report_objections=report_objections,
        )
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(markdown)
        self._print(f"[DraftingStage] Report written to {report_file} "
                    f"({len(markdown.splitlines())} lines)")

        drafting = DraftingResult(
            report_file=report_file,
            narratives=narratives,
            section_count=markdown.count("\n## "),
            limitation_count=markdown.count("\n- ", markdown.find("## 7. Limitations")),
            # Template-derived numbers with their derivation; the Stage-3 checker reads
            # them as a source artifact with provenance 'derived'
            metadata={"derived_numbers": derived_numbers(code_generation, estimation_results),
                      **({"report_objections": list(report_objections)}
                         if report_objections else {})},
        )
        self.output = DraftingStageOutput(drafting=drafting, report_markdown=markdown)
        return self.output

    def publish_with_objections(self, objections: List[str]) -> DraftingStageOutput:
        """The revised draft was rejected again — publish it (no further LLM rounds) with
        the committee's objections recorded in Limitations."""
        if not self._last_inputs or self.output is None:
            raise RuntimeError("run_drafting_pipeline first")
        return self.run_drafting_pipeline(**self._last_inputs, report_objections=objections,
                                          narratives=self.output.drafting.narratives)

    def collect_human_feedback(self, round_number: int = 1, context: Optional[Dict] = None) -> str:
        """HITL checkpoint: review the drafted report (WithHITL modes). In pipeline auto mode
        the committee resolver answers via auto_input's context."""
        concerns = auto_input(
            "Concerns about the report draft (framing, emphasis, missing caveats): ",
            default=get_default("reporting_draft_feedback"), context=context,
        ).strip()
        comments = auto_input(
            "General comments: ",
            default=get_default("reporting_general_comments"), context=context,
        ).strip()
        parts = [p for p in (concerns, comments) if p]
        feedback = "\n".join(parts)
        feedback_file = os.path.join(self.output_dir or ".",
                                     f"round{round_number}_reporting_feedback.json")
        with open(feedback_file, "w", encoding="utf-8") as f:
            json.dump({"round_number": round_number, "concerns": concerns,
                       "comments": comments}, f, indent=2)
        self._print(f"[DraftingStage] Feedback saved to {feedback_file}")
        return feedback

    def save_drafting_output(self, filename: str = "drafting_output.json") -> str:
        if self.output is None:
            raise RuntimeError("run_drafting_pipeline first")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.output.model_dump(), f, indent=2, default=str)
        self._print(f"[DraftingStage] Output saved to {filename}")
        return filename


if __name__ == "__main__":
    print("DraftingStage expects pipeline artifacts; run via 0-MasterOrchestrator.py")
