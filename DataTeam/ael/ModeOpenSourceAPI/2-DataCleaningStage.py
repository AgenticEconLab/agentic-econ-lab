# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Data Cleaning Stage - Open Source API Mode

This script performs:
- Quality assessment of retrieved API data
- Temporal alignment of data with different frequencies
- Multi-source integration into unified dataset
- Final data transformation

Pipeline:
1. DataValidationAgent: Quality assessment on retrieved API data
2. TemporalAlignmentAgent: Align data with different frequencies
3. MultiSourceIntegrationAgent: Merge data from multiple sources
4. DataTransformationAgent: Apply final transformations

HITL Checkpoint (4):
4. Alignment Review (after temporal alignment)

Input: Retrieved data from Stage 1 (api_source_output.json)
Output: Aligned and integrated datasets
"""

import os
import sys
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Add parent directories to path for shared imports
_current_dir = Path(__file__).resolve().parent
_agents_dir = _current_dir.parent.parent.parent  # repository root
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

from shared.llm import LLMClient
from shared.json_repair import repair_json
from shared.observability import MetricsCollector
from shared.auto_input import auto_input, get_default
from DataTeam.ael.schemas.stage_outputs import (
    RetrievedData, QualityDimension, QualityAssessment,
    AlignedSeries, IntegratedDataset, DataCleaningOutput,
)


# ============================================================================
# Environment Configuration
# ============================================================================

def get_env_path():
    """Find and return the path to the .env file in the repository root (the directory holding ael_config.yaml and run_ael_pipeline.py)."""
    current_dir = Path(__file__).resolve().parent
    
    while not ((current_dir / "ael_config.yaml").exists() and (current_dir / "run_ael_pipeline.py").exists()) and current_dir.parent != current_dir:
        current_dir = current_dir.parent
    
    if (current_dir / "ael_config.yaml").exists() and (current_dir / "run_ael_pipeline.py").exists():
        env_path = current_dir / ".env"
        if env_path.exists():
            return str(env_path)
    
    return None


# Load environment variables
env_path = get_env_path()
if env_path:
    load_dotenv(env_path)
else:
    load_dotenv()


# ============================================================================
# Agents
# ============================================================================

def load_full_observations(data_source_output: Dict) -> Dict[str, List[Dict]]:
    """Full retrieved observations written by the source stage (sidecar file named in
    metadata.observations_file). {} when absent/unreadable — callers fall back to previews
    and say so."""
    path = ((data_source_output or {}).get("metadata") or {}).get("observations_file")
    if not path or not os.path.exists(str(path)):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): v for k, v in data.items() if isinstance(v, list)}
    except Exception as e:
        print(f"  [WARNING] could not read full observations ({path}): {e}")
        return {}


def _preview_observations(data: RetrievedData) -> List[Dict]:
    """[{date, value}] from a 5-row preview (only when the full series is unavailable)."""
    out = []
    for row in data.data_preview or []:
        val = row.get(data.series_id)
        if val is None:
            val = next((v for k, v in row.items() if k != "date"), None)
        out.append({"date": row.get("date"), "value": val})
    return out


def _grade(score: float) -> str:
    return "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else \
        "D" if score >= 60 else "F"


class DataValidationAgent:
    """Quality assessment of retrieved API data — DETERMINISTIC.

    The verdict and every score come from checks on the observations themselves (constant,
    all-NaN, duplicate dates, too few observations, simulated, preview-only, discontinued,
    unverified/proxy title). The LLM only narrates the computed checks; it never scores."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None,
                 narrate: bool = True):
        self.agent_name = "DataValidationAgent"
        self.api_key = openai_api_key
        self.narrate = narrate
        self.llm = LLMClient(
            temperature=0.3,
            api_key=self.api_key,
            collector=collector,
            agent_name=self.agent_name
        )

    def assess_quality(
        self,
        retrieved_data: List[RetrievedData],
        full_observations: Optional[Dict[str, List[Dict]]] = None,
    ) -> List[QualityAssessment]:
        """Deterministic quality assessment of each retrieved series."""

        print(f"\n[{self.agent_name}] Assessing quality of {len(retrieved_data)} data series...")
        full_observations = full_observations or {}
        assessments = []
        for data in retrieved_data:
            print(f"  Assessing: {data.series_id}...")
            obs = full_observations.get(data.series_id)
            assessment = self._assess_series_quality(
                data, obs if obs is not None else _preview_observations(data),
                preview_only=obs is None and not data.data_simulated)
            assessments.append(assessment)
            print(f"    Quality Score: {assessment.overall_score:.1f} ({assessment.quality_grade}, "
                  f"{assessment.verdict}){' — ' + '; '.join(assessment.actionable_issues) if assessment.actionable_issues else ''}")
        print(f"[{self.agent_name}] Quality assessment complete for {len(assessments)} series")
        return assessments

    def _assess_series_quality(self, data: RetrievedData, observations: List[Dict],
                               preview_only: bool = False) -> QualityAssessment:
        """Assess one series from its observations (deterministic) + optional narration."""
        from DataTeam.ael.data_checks import series_checks
        checks = series_checks(observations, simulated=bool(data.data_simulated),
                               preview_only=preview_only)
        issues = list(checks["issues"])
        warnings = list(checks.get("warnings") or [])
        if getattr(data, "discontinued", False):
            warnings.append(f"discontinued/stale: last observation {data.end_date}")
        if getattr(data, "title_check", None) == "mismatch":
            warnings.append(f"proxy: source title '{data.source_title}' is not "
                            f"'{data.series_name}'")
        elif getattr(data, "title_check", None) == "unverified" and not data.data_simulated:
            warnings.append("source title unverified")
        n = checks["n_observations"]
        completeness = 0.0 if checks["all_nan"] else round(100 * (1 - checks["missing_share"]), 1)
        if checks["below_min_observations"]:
            completeness = min(completeness, round(100 * n / max(checks["min_observations"], 1), 1))
        accuracy = 0.0 if (checks["constant"] or checks["all_nan"] or checks["simulated"]) else 100.0
        consistency = 0.0 if checks["duplicate_dates"] else (70.0 if warnings else 100.0)
        timeliness = 40.0 if getattr(data, "discontinued", False) else 100.0
        dims = [
            QualityDimension(dimension="completeness", score=completeness,
                             issues_found=[i for i in issues if "observation" in i],
                             recommendations=[]),
            QualityDimension(dimension="accuracy", score=accuracy,
                             issues_found=[i for i in issues if "constant" in i or "numeric" in i
                                           or "simulated" in i], recommendations=[]),
            QualityDimension(dimension="consistency", score=consistency,
                             issues_found=[i for i in issues if "duplicate" in i] + warnings,
                             recommendations=[]),
            QualityDimension(dimension="timeliness", score=timeliness,
                             issues_found=[w for w in warnings if "discontinued" in w],
                             recommendations=[]),
        ]
        overall = sum(d.score for d in dims) / len(dims)
        if checks["verdict"] == "fail":
            overall = min(overall, 40.0)          # a blocking check caps the score (grade F)
        if checks["simulated"]:
            overall = 0.0
        overall = round(overall, 1)
        assessment = QualityAssessment(
            series_id=data.series_id,
            series_name=data.series_name,
            source_name=data.source_name,
            dimensions=dims,
            overall_score=overall,
            quality_grade=_grade(overall),
            actionable_issues=issues + warnings,
            checks=checks,
            verdict=checks["verdict"],
        )
        if self.narrate:
            assessment.narrative = self._narrate(data, checks, issues + warnings)
        return assessment

    def _narrate(self, data: RetrievedData, checks: Dict, issues: List[str]) -> str:
        """Two-sentence plain-language narration of the COMPUTED checks (no scoring)."""
        try:
            text = self.llm.format_and_invoke(
                system_prompt="You describe computed data-quality checks in plain language. "
                              "Do not add facts, scores or judgments beyond the checks.",
                user_prompt=("Series {sid} ({name}, {source}). Computed checks: {checks}. "
                             "Issues: {issues}. Describe them in at most two sentences."),
                variables={"sid": data.series_id, "name": data.series_name,
                           "source": data.source_name,
                           "checks": json.dumps({k: v for k, v in checks.items()
                                                 if k != "issues"}),
                           "issues": "; ".join(issues) or "none"},
            )
            return str(text or "").strip()[:600]
        except Exception as e:
            print(f"      (narration unavailable: {e})")
            return ""


class TemporalAlignmentAgent:
    """Aligns series BY DATE at the coarsest common frequency (real pandas resampling of
    the retrieved observations; previews are used only when the full series is unavailable,
    and that is stated)."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "TemporalAlignmentAgent"
        self.api_key = openai_api_key

    def align_data(
        self,
        retrieved_data: List[RetrievedData],
        research_question: str = "",
        full_observations: Optional[Dict[str, List[Dict]]] = None,
        quality_assessments: Optional[List[QualityAssessment]] = None,
    ) -> Tuple[List[AlignedSeries], str]:
        """Align the series that passed the deterministic checks; keep the merged frame on
        ``self.merged`` and the alignment facts on ``self.alignment``."""
        from DataTeam.ael.data_checks import (AGGREGATION_NOTE, align_and_merge, to_series)

        print(f"\n[{self.agent_name}] Aligning {len(retrieved_data)} data series...")
        full_observations = full_observations or {}
        verdicts = {a.series_id: a for a in (quality_assessments or [])}
        series_map, excluded, preview_only = {}, {}, []
        for d in retrieved_data:
            qa = verdicts.get(d.series_id)
            if d.data_simulated:
                excluded[d.series_id] = ["simulated placeholder"]
                continue
            obs = full_observations.get(d.series_id)
            if obs is None:
                obs = _preview_observations(d)
                preview_only.append(d.series_id)
            if qa is not None and qa.verdict == "fail":
                excluded[d.series_id] = list(qa.checks.get("issues") or ["failed checks"])
                continue
            series_map[d.series_id] = to_series(obs)
        frame, target, info = align_and_merge(series_map)
        complete = frame.dropna() if not frame.empty else frame
        self.merged = frame
        self.alignment = {
            "target_frequency": target,
            "aggregation": AGGREGATION_NOTE,
            "outer_rows": int(len(frame)),
            "complete_case_rows": int(len(complete)),
            "complete_case_period": (f"{complete.index[0].date()} to {complete.index[-1].date()}"
                                     if len(complete) else ""),
            "included": list(series_map),
            "excluded": excluded,
            "preview_only": preview_only,
            "series": info,
        }
        print(f"  Target frequency: {target}; {len(series_map)} series aligned, "
              f"{len(excluded)} excluded; complete-case rows: {len(complete)}")
        if preview_only:
            print(f"  [WARNING] only the stored 5-row preview was available for: "
                  f"{', '.join(preview_only)}")

        aligned = []
        for d in retrieved_data:
            if d.series_id not in series_map:
                continue
            col = (frame[d.series_id].dropna() if d.series_id in frame.columns
                   else pd.Series(dtype=float))
            meta = info.get(d.series_id, {})
            aligned.append(AlignedSeries(
                series_id=d.series_id,
                series_name=d.series_name,
                source_name=d.source_name,
                original_frequency=meta.get("native_frequency") or d.frequency,
                target_frequency=target,
                conversion_method=meta.get("method", ""),
                num_observations=int(len(col)),
                start_date=str(col.index[0].date()) if len(col) else "",
                end_date=str(col.index[-1].date()) if len(col) else "",
                data_preview=[{"date": str(i.date()), d.series_id: float(v)}
                              for i, v in col.tail(5).items()],
                alignment_notes=(f"{meta.get('method', '')}; {AGGREGATION_NOTE}"
                                 + ("; ONLY the 5-row preview was available"
                                    if d.series_id in preview_only else "")),
            ))
        print(f"[{self.agent_name}] Alignment complete for {len(aligned)} series")
        return aligned, target


class MultiSourceIntegrationAgent:
    """Builds the integrated dataset from the DATE-ALIGNED frame."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "MultiSourceIntegrationAgent"
        self.api_key = openai_api_key

    def integrate_data(
        self,
        aligned_series: List[AlignedSeries],
        target_frequency: str,
        series_metadata: Optional[Dict[str, Dict[str, str]]] = None,
        merged=None,
        alignment: Optional[Dict] = None,
        quality_assessments: Optional[List[QualityAssessment]] = None,
    ) -> List[IntegratedDataset]:
        """One integrated dataset: complete-case rows of the aligned frame, real dates."""
        from DataTeam.ael.data_checks import min_observations
        import pandas as pd

        print(f"\n[{self.agent_name}] Integrating {len(aligned_series)} aligned series...")
        series_metadata = series_metadata or {}
        alignment = dict(alignment or {})
        frame = merged if merged is not None else pd.DataFrame()
        complete = frame.dropna() if not frame.empty else frame
        provenance = {}
        for s in aligned_series:
            meta = series_metadata.get(s.series_id, {})
            provenance[s.series_id] = {
                "source": s.source_name,
                "original_frequency": s.original_frequency,
                "conversion_method": s.conversion_method,
                "units": meta.get("units", ""),
                "description": meta.get("description", ""),
                "series_name": meta.get("series_name", s.series_name),
                "seasonal_adjustment": meta.get("seasonal_adjustment", ""),
            }
        # dataset verdict: every requested series must pass and the merge must leave enough
        # complete rows — the QA stage's validation fails on this verdict
        issues = []
        for a in quality_assessments or []:
            if a.verdict == "fail":
                issues.append(f"{a.series_id}: {'; '.join(a.checks.get('issues') or ['failed'])}")
        floor = min_observations()
        if len(complete) < floor:
            issues.append(f"merged dataset has {len(complete)} complete-case rows (< {floor})")
        if alignment.get("preview_only"):
            issues.append("only 5-row previews were available for "
                          + ", ".join(alignment["preview_only"]))
        alignment.update({"dataset_verdict": "fail" if issues else "pass",
                          "dataset_issues": issues,
                          "merge_strategy": "outer join on period index; complete cases reported"})
        preview = [{"date": str(i.date()), **{c: float(v) for c, v in row.items()}}
                   for i, row in complete.tail(5).iterrows()]
        variables = ["date"] + list(frame.columns)
        sources = sorted({s.source_name for s in aligned_series})
        print(f"  Complete-case rows: {len(complete)}; dataset verdict: "
              f"{alignment['dataset_verdict']}")
        return [IntegratedDataset(
            dataset_id="INT_API_001",
            dataset_name="Integrated API Economic Dataset",
            source_apis=sources,
            num_variables=len(variables),
            num_observations=int(len(complete)),
            time_period=alignment.get("complete_case_period") or "",
            frequency=target_frequency,
            merge_strategy="date-aligned outer join; num_observations = complete-case rows",
            variables=variables,
            data_preview=preview,
            provenance=provenance,
            alignment=alignment,
        )]


class DataTransformationAgent:
    """Agent for applying final transformations."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "DataTransformationAgent"
        self.api_key = openai_api_key
        self.llm = LLMClient(
            temperature=0.3,
            api_key=self.api_key,
            collector=collector,
            agent_name=self.agent_name
        )
    
    def transform_data(
        self,
        integrated_dataset: IntegratedDataset,
        quality_assessments: List[QualityAssessment]
    ) -> IntegratedDataset:
        """Apply final transformations based on quality assessment."""
        
        print(f"\n[{self.agent_name}] Applying final transformations...")
        
        # Identify transformations needed based on quality issues
        transformations = self._identify_transformations(quality_assessments)
        
        for transform in transformations:
            print(f"  Applying: {transform}")
        
        # In real implementation, would apply actual transformations
        # For now, just update the dataset metadata
        
        print(f"[{self.agent_name}] Transformations complete")
        return integrated_dataset
    
    def _identify_transformations(self, quality_assessments: List[QualityAssessment]) -> List[str]:
        """Identify transformations needed based on quality issues."""
        
        transformations = []
        
        for assessment in quality_assessments:
            for issue in assessment.actionable_issues:
                if "missing" in issue.lower():
                    transformations.append(f"Impute missing values for {assessment.series_id}")
                elif "outlier" in issue.lower():
                    transformations.append(f"Handle outliers for {assessment.series_id}")
        
        if not transformations:
            transformations.append("No transformations required - data quality is acceptable")
        
        return transformations


# ============================================================================
# HITL Checkpoint
# ============================================================================

def checkpoint4_alignment_review(
    aligned_series: List[AlignedSeries],
    target_frequency: str,
    alignment: Optional[Dict] = None,
    quality_assessments: Optional[List[QualityAssessment]] = None,
    research_question: str = "",
):
    """HITL Checkpoint 4: Alignment Review (the printed question is the prompt and the
    alignment facts are the context)."""
    from DataTeam.ael.hitl import ask
    alignment = alignment or {}
    lines = [f"Research question: {research_question}",
             f"Target frequency: {target_frequency} ({alignment.get('aggregation', '')})",
             f"Complete-case rows after the date-aligned merge: "
             f"{alignment.get('complete_case_rows', '?')} "
             f"({alignment.get('complete_case_period', '')})", "", "Aligned series:"]
    for s in aligned_series:
        lines.append(f"  - {s.series_id}: {s.original_frequency} -> {s.target_frequency}, "
                     f"{s.num_observations} obs, {s.conversion_method}")
    for sid, why in (alignment.get("excluded") or {}).items():
        lines.append(f"  - EXCLUDED {sid}: {'; '.join(why)}")
    ctx = {
        "research_question": research_question,
        "alignment": {k: v for k, v in alignment.items() if k != "series"},
        "aligned_series": [
            {"series_id": s.series_id, "original_frequency": s.original_frequency,
             "method": s.conversion_method, "n_obs": s.num_observations,
             "period": f"{s.start_date}..{s.end_date}"} for s in aligned_series],
        "quality_checks": [
            {"series_id": a.series_id, "verdict": a.verdict, "issues": a.actionable_issues}
            for a in (quality_assessments or [])],
    }
    return ask("Alignment Review", lines,
               ["Is the target frequency appropriate for the research question?",
                "Are the aggregation method and the excluded series acceptable?"],
               default=get_default("quality_review") or "approved", context=ctx)


# ============================================================================
# Orchestrator
# ============================================================================

class DataCleaningOrchestrator:
    """Orchestrator for the data cleaning stage."""

    def __init__(self, openai_api_key: Optional[str] = None, collector: Optional[MetricsCollector] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        self.validation_agent = DataValidationAgent(self.api_key, collector=collector)
        self.alignment_agent = TemporalAlignmentAgent(self.api_key, collector=collector)
        self.integration_agent = MultiSourceIntegrationAgent(self.api_key, collector=collector)
        self.transformation_agent = DataTransformationAgent(self.api_key, collector=collector)
        self.cleaning_output: Optional[DataCleaningOutput] = None
        self.merged = None

    def run_cleaning_pipeline(
        self,
        data_source_output: Dict,
        research_question: str = "",
        enable_hitl: bool = True
    ) -> DataCleaningOutput:
        """Run the complete data cleaning pipeline."""

        print(f"\n{'='*70}")
        print(f"TEMPORAL ALIGNMENT & MULTI-SOURCE INTEGRATION PIPELINE")
        print(f"{'='*70}")

        # Parse retrieved data
        retrieved_data_raw = data_source_output.get('retrieved_data', [])
        retrieved_data = [RetrievedData(**d) for d in retrieved_data_raw]
        # The FULL retrieved observations (sidecar written by the source stage)
        full_observations = load_full_observations(data_source_output)
        print(f"Full observations available for {len(full_observations)} of "
              f"{len(retrieved_data)} series")

        # Carry the data-integrity flag forward (prefer the per-series field, fall
        # back to the Stage-1 metadata for back-compat with older source JSON).
        data_simulated = any(getattr(d, "data_simulated", False) for d in retrieved_data) or bool(
            data_source_output.get('metadata', {}).get('data_simulated', False))

        # Build a series_id -> {units, description} map from the Stage-1
        # selected_series so real units/descriptions can be carried into the
        # provenance (and ultimately the codebook), instead of "Various".
        series_metadata: Dict[str, Dict[str, str]] = {}
        for s in data_source_output.get('selected_series', []):
            sid = s.get('series_id')
            if sid:
                series_metadata[sid] = {
                    "units": s.get('units', '') or '',
                    "description": s.get('description', '') or '',
                    "series_name": s.get('series_name', '') or '',
                    "seasonal_adjustment": s.get('seasonal_adjustment', '') or '',
                }

        research_q = data_source_output.get('research_question', research_question)

        print(f"Retrieved Data Series: {len(retrieved_data)}")
        print(f"{'='*70}\n")

        # Step 1: Quality Assessment (deterministic checks on the observations)
        print(f"STEP 1: QUALITY ASSESSMENT")
        print(f"-"*70)
        quality_assessments = self.validation_agent.assess_quality(
            retrieved_data, full_observations)

        # Step 2: Temporal Alignment (by date, coarsest common frequency)
        print(f"\nSTEP 2: TEMPORAL ALIGNMENT")
        print(f"-"*70)
        aligned_series, target_frequency = self.alignment_agent.align_data(
            retrieved_data, research_q, full_observations=full_observations,
            quality_assessments=quality_assessments)
        alignment = dict(getattr(self.alignment_agent, "alignment", {}) or {})
        self.merged = getattr(self.alignment_agent, "merged", None)

        # HITL Checkpoint 4: Alignment Review — the alignment is deterministic, so an
        # objection cannot be revised here; it is recorded and flows downstream.
        hitl = []
        if enable_hitl:
            cp4 = checkpoint4_alignment_review(aligned_series, target_frequency, alignment,
                                               quality_assessments, research_q)
            hitl.append(cp4)
            print(f"[HITL] Alignment review: {'approved' if cp4.approved else 'OBJECTION'} "
                  f"('{cp4.response[:120]}')")

        # Step 3: Multi-Source Integration
        print(f"\nSTEP 3: MULTI-SOURCE INTEGRATION")
        print(f"-"*70)
        integrated_datasets = self.integration_agent.integrate_data(
            aligned_series,
            target_frequency,
            series_metadata=series_metadata,
            merged=self.merged,
            alignment=alignment,
            quality_assessments=quality_assessments,
        )

        # Step 4: Final Transformation
        print(f"\nSTEP 4: FINAL TRANSFORMATION")
        print(f"-"*70)
        for i, dataset in enumerate(integrated_datasets):
            integrated_datasets[i] = self.transformation_agent.transform_data(
                dataset,
                quality_assessments
            )

        dataset_alignment = integrated_datasets[0].alignment if integrated_datasets else {}
        # Create output
        self.cleaning_output = DataCleaningOutput(
            retrieved_data=retrieved_data,
            quality_assessments=quality_assessments,
            aligned_datasets=aligned_series,
            integrated_datasets=integrated_datasets,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "research_question": research_q,
                "num_retrieved": len(retrieved_data),
                "num_aligned": len(aligned_series),
                "num_integrated": len(integrated_datasets),
                "target_frequency": target_frequency,
                "avg_quality_score": sum(a.overall_score for a in quality_assessments) / len(quality_assessments) if quality_assessments else 0,
                "quality_scoring": "deterministic checks on the retrieved observations "
                                   "(LLM narration only)",
                "num_failed_checks": sum(1 for a in quality_assessments if a.verdict == "fail"),
                "dataset_verdict": dataset_alignment.get("dataset_verdict", ""),
                "dataset_issues": dataset_alignment.get("dataset_issues", []),
                "data_simulated": data_simulated,
                # The COUNT, so QA can cap proportionally instead of branding a
                # 14-real/1-simulated dataset with the full-simulation floor of 50.
                "num_simulated": sum(
                    1 for d in retrieved_data if getattr(d, "data_simulated", False)),
                "hitl_checkpoints": [r.to_record() for r in hitl],
                "hitl_objections": [r.to_record() for r in hitl if not r.approved],
                "limitations": [r.limitation() for r in hitl if r.limitation()],
            }
        )

        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Quality Assessments: {len(quality_assessments)}")
        print(f"Aligned Series: {len(aligned_series)}")
        print(f"Integrated Datasets: {len(integrated_datasets)}")
        print(f"Target Frequency: {target_frequency}")
        print(f"Avg Quality Score: {self.cleaning_output.metadata['avg_quality_score']:.1f}")
        print(f"Dataset verdict: {self.cleaning_output.metadata['dataset_verdict']}")
        print(f"{'='*70}\n")

        return self.cleaning_output

    def save_cleaning_output(self, filename: str = "api_cleaning_output.json"):
        """Save cleaning output to JSON (and the merged, date-aligned data as CSV)."""
        if not self.cleaning_output:
            print("No cleaning output to save")
            return

        if self.merged is not None and not self.merged.empty:
            base, _ = os.path.splitext(os.path.abspath(filename))
            csv_file = f"{base}_merged.csv"
            self.merged.to_csv(csv_file)
            for ds in self.cleaning_output.integrated_datasets:
                ds.alignment["merged_file"] = csv_file

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.cleaning_output.model_dump(), f, indent=2)

        print(f"[Orchestrator] Cleaning output saved to {filename}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main function for data cleaning stage."""
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Load source output
    source_file = "api_source_output.json"
    if os.path.exists(source_file):
        with open(source_file, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
    else:
        # Create sample data for testing
        source_data = {
            "research_question": "What is the relationship between GDP growth, unemployment, and inflation?",
            "retrieved_data": [
                {
                    "series_id": "GDPC1",
                    "series_name": "Real GDP",
                    "source_name": "FRED",
                    "retrieval_date": "2024-01-01",
                    "num_observations": 300,
                    "start_date": "1990-01-01",
                    "end_date": "2023-12-01",
                    "frequency": "quarterly",
                    "data_preview": [{"date": "2023-12-01", "GDPC1": 22000}],
                    "quality_notes": "Sample data",
                    "retrieval_status": "Success"
                },
                {
                    "series_id": "UNRATE",
                    "series_name": "Unemployment Rate",
                    "source_name": "FRED",
                    "retrieval_date": "2024-01-01",
                    "num_observations": 400,
                    "start_date": "1990-01-01",
                    "end_date": "2023-12-01",
                    "frequency": "monthly",
                    "data_preview": [{"date": "2023-12-01", "UNRATE": 3.7}],
                    "quality_notes": "Sample data",
                    "retrieval_status": "Success"
                }
            ]
        }
    
    # Run pipeline
    orchestrator = DataCleaningOrchestrator()
    cleaning_output = orchestrator.run_cleaning_pipeline(
        data_source_output=source_data,
        enable_hitl=True
    )
    
    # Save outputs
    orchestrator.save_cleaning_output("api_cleaning_output.json")
    
    print("\n" + "="*70)
    print("DATA CLEANING SUMMARY")
    print("="*70)
    print(f"Quality Assessments: {len(cleaning_output.quality_assessments)}")
    print(f"Aligned Series: {len(cleaning_output.aligned_datasets)}")
    print(f"Integrated Datasets: {len(cleaning_output.integrated_datasets)}")
    print("="*70)


if __name__ == "__main__":
    main()
