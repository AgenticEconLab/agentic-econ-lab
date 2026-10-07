# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Quality Stage — the deterministic number-consistency check over the final draft.

Pipeline:
1. check_consistency: every number in the draft must trace to a source artifact (exactly, as
   a correct rounding, or as a percent rescaling); LLM-narrative hallucinations are flagged
2. A quality footer with the verified/total count is appended to the report — the reader
   sees the verification status ON the report itself
3. Proofreader (LLM) narrates wording/clarity suggestions — narration only, applied never
4. JournalAdvisor (§4.7): deterministic keyword-overlap shortlist against a static,
   illustrative journal-scope table (not live database intelligence), then LLM narrates
   adaptation strategy for the shortlist only — never re-ranks or invents a journal
5. Formatter (§4.7): deterministic re-render of the ALREADY-VERIFIED report into a
   journal-style reference format + a Data/Code Availability statement from real pipeline
   facts; writes a SEPARATE file, never touches the verified original

Input:  drafting_output.json (+ the artifacts the draft was built from)
Output: quality_output.json + the annotated research_report.md + research_report_formatted.md

This file is byte-identical across ReportingTeam mode directories (mode-copy invariant).
"""

import json
import os
import sys
from pathlib import Path as _Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

_agents_dir = _Path(__file__).resolve().parent.parent.parent.parent
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

from shared.auto_input import auto_input
from shared.llm import LLMClient
from shared.observability import MetricsCollector

from ReportingTeam.ael.report_harness import (
    build_availability_statement,
    check_consistency,
    format_for_journal,
    shortlist_journals,
)
from ReportingTeam.ael.report_harness.types import QualityResult
from ReportingTeam.ael.schemas.stage_outputs import QualityStageOutput

load_dotenv()


# ========== AGENT ==========

class Proofreader:
    """Suggests wording improvements; cannot change numbers or verdicts."""

    def __init__(self, openai_api_key: str = "", collector: Optional[MetricsCollector] = None):
        self.agent_name = "Proofreader"
        self.llm = LLMClient(
            temperature=0.3,
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
            collector=collector,
            agent_name=self.agent_name,
        )

    def review(self, report_excerpt: str, consistency_summary: str) -> str:
        raw = self.llm.invoke([
            {"role": "system", "content":
                "You are proofreading an economics research report for clarity and academic "
                "convention. Suggest improvements as a short list; do NOT rewrite numbers."},
            {"role": "user", "content":
                f"Consistency check result: {consistency_summary[:400]}\n\n"
                f"Report (excerpt):\n{report_excerpt[:3000]}\n\n"
                "In <=150 words: the top clarity/convention improvements."},
        ], max_tokens=400)
        return str(raw).strip()


class JournalAdvisor:
    """Narrates adaptation strategy for a deterministically shortlisted set of journals; has
    no authority over which journals are on the shortlist or their scores."""

    def __init__(self, openai_api_key: str = "", collector: Optional[MetricsCollector] = None):
        self.agent_name = "JournalAdvisor"
        self.llm = LLMClient(
            temperature=0.3,
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
            collector=collector,
            agent_name=self.agent_name,
        )

    def advise(self, shortlist_dump: str, research_question: str, report_excerpt: str) -> str:
        raw = self.llm.invoke([
            {"role": "system", "content":
                "You advise on target-journal fit for an economics report. The shortlist is a "
                "static, illustrative reference set of journal scopes, NOT live database "
                "intelligence — never invent acceptance rates, impact factors, or editorial "
                "preferences not stated. Use ONLY the shortlist given."},
            {"role": "user", "content":
                f"Research question: {research_question[:300]}\n\n"
                f"Report excerpt:\n{report_excerpt[:1500]}\n\n"
                f"Deterministic shortlist (name, score, matched scope keywords):\n"
                f"{shortlist_dump[:1500]}\n\n"
                "In <=150 words: for the top 1-2 candidates, what adaptation (framing, "
                "emphasis, methodological presentation) would make the report fit that "
                "journal's scope?"},
        ], max_tokens=400)
        return str(raw).strip()


class Formatter:
    """Deterministic journal-style re-render (references) + Data/Code Availability
    statement, both computed by ``report_harness.formatting``; this class only wires the
    harness output into the stage and writes the separate formatted file. No LLM involved —
    an agent that rewrote the already-verified report's numbers would undo Stage 3's own
    consistency check."""

    def __init__(self, collector: Optional[MetricsCollector] = None):
        self.agent_name = "Formatter"
        self.collector = collector

    def format_report(self, annotated_markdown: str, literature_batch: Dict,
                      data_source: Dict, out_file: str, style: str = "AEA") -> tuple:
        print(f"\n[{self.agent_name}] Re-rendering references in {style} style + "
              "availability statement...")
        formatted = format_for_journal(annotated_markdown, literature_batch, style=style)
        statement = build_availability_statement(data_source, annotated_markdown)
        formatted = formatted.rstrip("\n") + "\n\n---\n## Availability Statement\n" + statement + "\n"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(formatted)
        print(f"  Formatted report -> {out_file}")
        return out_file, statement


# ========== ORCHESTRATOR ==========

class QualityOrchestrator:
    """Runs the consistency check and annotates the report with its verdict."""

    def __init__(self, openai_api_key: str = "", collector: Optional[MetricsCollector] = None,
                 quiet: bool = False):
        self.collector = collector
        self.quiet = quiet
        self.proofreader = Proofreader(openai_api_key, collector=collector)
        self.journal_advisor = JournalAdvisor(openai_api_key, collector=collector)
        self.formatter = Formatter(collector=collector)
        self.output: Optional[QualityStageOutput] = None

    def _print(self, msg: str):
        if not self.quiet:
            print(msg)

    def run_quality_pipeline(self, drafting_output: Dict,
                             source_artifacts: Any,
                             research_question: str = "",
                             data_source: Optional[Dict] = None,
                             literature_batch: Optional[Dict] = None,
                             formatted_report_file: str = "research_report_formatted.md",
                             ) -> QualityStageOutput:
        markdown = drafting_output.get("report_markdown", "") or ""
        report_file = (drafting_output.get("drafting") or {}).get("report_file",
                                                                  "research_report.md")
        # Numbers the template derived (with their recorded derivation) are a source
        # artifact of their own, provenance 'derived'
        derived = ((drafting_output.get("drafting") or {}).get("metadata") or {}).get(
            "derived_numbers") or {}
        if derived:
            if isinstance(source_artifacts, dict):
                source_artifacts = {**source_artifacts, "derived": derived}
            else:
                source_artifacts = list(source_artifacts or []) + [derived]
        self._print("[QualityStage] Running the number-consistency check...")
        consistency = check_consistency(markdown, source_artifacts)
        self._print(f"[QualityStage] {consistency.verified}/{consistency.total_numbers} numbers "
                    f"verified ({consistency.skipped_small_ints} small-int enumerations "
                    f"skipped) -> {consistency.verdict}")
        for u in consistency.unverified[:8]:
            self._print(f"  UNVERIFIED {u.value}: ...{u.context}...")

        _kinds = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in
                           sorted(consistency.match_kinds.items())) or "none"
        footer = ["\n---",
                  f"**Number-consistency check:** {consistency.verified} of "
                  f"{consistency.total_numbers} numbers match a value in the source artifacts "
                  f"({_kinds}); {consistency.skipped_small_ints} small integers in enumeration, "
                  f"ordinal or count positions and confidence levels next to 'CI' are exempt "
                  f"(verdict: `{consistency.verdict}`). Numbers inside free text upstream "
                  f"(literature prose, rationales) do not count as sources. The check verifies "
                  "that each value occurs upstream, not the claim made around it."]
        if consistency.unverified:
            footer.append("Unverified numbers (flagged, likely from narrative): "
                          + ", ".join(f"`{u.value}`" for u in consistency.unverified[:10]))
        annotated = markdown + "\n".join(footer) + "\n"
        if report_file:
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(annotated)

        notes = ""
        try:
            notes = self.proofreader.review(
                annotated[:3000],
                f"{consistency.verified}/{consistency.total_numbers} verified, "
                f"verdict {consistency.verdict}")
        except Exception as e:
            self._print(f"[QualityStage] Proofreader narration skipped (non-critical): {e}")

        # JournalAdvisor: deterministic shortlist, then LLM adaptation-strategy narration.
        dependent_name = ""
        _arts = (list(source_artifacts.values()) if isinstance(source_artifacts, dict)
                 else list(source_artifacts or []))
        for artifact in _arts:
            outcome = (artifact or {}).get("outcome") if isinstance(artifact, dict) else None
            if outcome and outcome.get("dependent_name"):
                dependent_name = outcome["dependent_name"]
                break
        shortlist = shortlist_journals(research_question, dependent_name)
        self._print(f"[JournalAdvisor] Shortlisted {len(shortlist)} journal(s) by keyword "
                    "overlap (static reference table)")
        journal_narrative = ""
        if shortlist:
            try:
                dump = json.dumps([m.model_dump() for m in shortlist], default=str)
                journal_narrative = self.journal_advisor.advise(
                    dump, research_question, annotated[:1500])
            except Exception as e:
                self._print(f"[JournalAdvisor] Narration skipped (non-critical): {e}")

        # Formatter: deterministic re-render of the already-verified report.
        _, availability_statement = self.formatter.format_report(
            annotated, literature_batch or {}, data_source or {}, formatted_report_file)

        quality = QualityResult(consistency=consistency, report_file=report_file,
                                proofreader_notes=notes,
                                journal_shortlist=shortlist, journal_narrative=journal_narrative,
                                formatted_report_file=formatted_report_file,
                                availability_statement=availability_statement)
        self.output = QualityStageOutput(quality=quality, report_markdown=annotated)
        return self.output

    def save_quality_output(self, filename: str = "quality_output.json") -> str:
        if self.output is None:
            raise RuntimeError("run_quality_pipeline first")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.output.model_dump(), f, indent=2, default=str)
        self._print(f"[QualityStage] Output saved to {filename}")
        return filename


if __name__ == "__main__":
    draft_path = auto_input("Path to drafting_output.json: ",
                            default="drafting_output.json").strip()
    with open(draft_path, "r", encoding="utf-8") as f:
        drafting_output = json.load(f)
    orch = QualityOrchestrator()
    orch.run_quality_pipeline(drafting_output, source_artifacts=[])
    orch.save_quality_output()
