# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Quality Assurance Stage - Open Source API Mode

This script performs:
- Final validation of integrated dataset
- Comprehensive documentation generation
- Data codebook creation
- Report generation with proper citations
- Replication manifest generation (environment, code fingerprint, audit trail)

Pipeline:
1. FinalValidationAgent: Final validation of integrated dataset
2. DocumentationAgent: Generate comprehensive report and codebook
3. Archivist: Build the replication manifest (deterministic, no LLM)

HITL Checkpoint (5):
5. Final Approval (after report generation)

Input: Integrated data from Stage 2 (api_cleaning_output.json)
Output: Documented datasets with report, codebook, and replication manifest
"""

import os
import sys
import json
import hashlib
import platform
from importlib import metadata as importlib_metadata
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
# Add parent directories to path for shared imports
_current_dir = Path(__file__).resolve().parent
_agents_dir = _current_dir.parent.parent.parent  # repository root
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

from shared.llm import LLMClient
from shared.json_repair import repair_json
from shared.observability import MetricsCollector
from shared.tools.sandbox_tool import CodeSandbox, ExecutionResult
from shared.auto_input import auto_input, get_default
from shared.reliability.state_guard import StateGuard
from DataTeam.ael.schemas.stage_outputs import (
    IntegratedDataset, QualityAssessment, VariableEntry,
    DataCodebook, DataReport, DocumentedDataset, QualityAssuranceOutput,
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

def dataset_check_verdict(dataset: IntegratedDataset) -> Tuple[str, List[str]]:
    """Deterministic dataset verdict. Uses the cleaning stage's alignment verdict; for
    an older artifact without one, recomputes the blocking checks from the preview rows
    (constant / all-missing variable, too few rows). ('', []) when there is no dataset."""
    if dataset is None:
        return "", []
    alignment = getattr(dataset, "alignment", None) or {}
    if alignment.get("dataset_verdict"):
        return alignment["dataset_verdict"], list(alignment.get("dataset_issues") or [])
    from DataTeam.ael.data_checks import min_observations
    issues = []
    variables = [v for v in dataset.variables if v != "date"]
    rows = dataset.data_preview or []
    for v in variables:
        vals = [r.get(v) for r in rows if isinstance(r.get(v), (int, float))]
        if not vals:
            issues.append(f"{v}: no numeric values in the dataset preview")
        elif len(vals) > 1 and max(vals) == min(vals):
            issues.append(f"{v}: constant in the dataset preview")
    if dataset.num_observations < min_observations():
        issues.append(f"{dataset.num_observations} observations (< {min_observations()})")
    return ("fail" if issues else "pass"), issues


class FinalValidationAgent:
    """Agent for final validation of integrated dataset."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "FinalValidationAgent"
        self.api_key = openai_api_key
        self.llm = LLMClient(
            temperature=0.3,
            api_key=self.api_key,
            collector=collector,
            agent_name=self.agent_name
        )
    
    def validate_dataset(
        self,
        integrated_dataset: IntegratedDataset,
        quality_assessments: List[QualityAssessment],
        data_simulated: bool = False,
        sim_share: float = 0.0,
    ) -> Tuple[float, str]:
        """Perform final validation of integrated dataset.

        When simulated series are present, certification is CAPPED — never Gold — and
        the score cap is PROPORTIONAL to the simulated share (so one disclosed simulated
        series out of 15 does not brand the whole dataset with the full-simulation floor
        of 50). A share >= 0.5 keeps the hard floor.
        """

        print(f"\n[{self.agent_name}] Performing final validation...")

        # Calculate overall quality score
        if quality_assessments:
            avg_score = sum(a.overall_score for a in quality_assessments) / len(quality_assessments)
        else:
            avg_score = 75.0

        # Determine certification level
        if avg_score >= 90:
            certification = "Gold"
        elif avg_score >= 75:
            certification = "Silver"
        else:
            certification = "Bronze"

        # Data-integrity cap: simulated data must never be certified Gold / 95+.
        if data_simulated:
            share = sim_share if 0.0 < sim_share <= 1.0 else 1.0
            cap = 50.0 if share >= 0.5 else 90.0 - 80.0 * share
            avg_score = min(avg_score, cap)
            certification = "Bronze" if avg_score < 75 else "Silver"   # NEVER Gold
            print(f"  [DATA INTEGRITY] {share:.0%} of series are SIMULATED — score "
                  f"capped to {avg_score:.1f}, certification {certification} "
                  f"(never Gold).")

        # The deterministic dataset verdict (cleaning stage) bounds certification — a
        # dataset with failing series or too few complete rows is never above Bronze
        verdict, issues = dataset_check_verdict(integrated_dataset)
        if verdict == "fail":
            avg_score = min(avg_score, 60.0)
            certification = "Bronze"
            print(f"  [DATA CHECKS] dataset verdict FAIL — {'; '.join(issues)[:300]}")

        print(f"  Quality Score: {avg_score:.1f}")
        print(f"  Certification: {certification}")

        return avg_score, certification


class DocumentationAgent:
    """Agent for generating comprehensive report and codebook."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "DocumentationAgent"
        self.api_key = openai_api_key
        self.llm = LLMClient(
            temperature=0.3,
            api_key=self.api_key,
            collector=collector,
            agent_name=self.agent_name
        )

    def generate_codebook(
        self,
        integrated_dataset: IntegratedDataset,
        quality_assessments: List[QualityAssessment]
    ) -> DataCodebook:
        """Generate comprehensive data codebook."""
        
        print(f"\n[{self.agent_name}] Generating data codebook...")

        # Build a series_id -> QA assessment map so per-variable notes can be
        # populated from the real quality issues found during cleaning.
        qa_by_series: Dict[str, QualityAssessment] = {}
        for a in quality_assessments:
            qa_by_series[a.series_id] = a

        # Create variable entries with REAL descriptions, units, and per-series
        # notes (previously these were "Economic indicator: X" / "Various" / "").
        variables = []
        for var in integrated_dataset.variables:
            if var == "date":
                continue

            prov = integrated_dataset.provenance.get(var, {})
            source = prov.get("source", "Unknown")
            conversion = prov.get("conversion_method", "none")

            # Real description / units from Stage-1 series metadata (carried via
            # provenance). Fall back to a sensible default only if absent.
            series_name = prov.get("series_name", "") or var
            description = prov.get("description", "") or f"Economic indicator: {var}"
            units = prov.get("units", "") or "Not specified"
            seasonal = prov.get("seasonal_adjustment", "")

            # Per-series notes: surface the actionable QA issues for this series.
            note_parts = []
            if series_name and series_name != var:
                note_parts.append(f"Series: {series_name} ({var}).")
            if seasonal:
                note_parts.append(f"Seasonal adjustment: {seasonal}.")
            assessment = qa_by_series.get(var)
            if assessment:
                note_parts.append(
                    f"Quality: {assessment.overall_score:.0f}/100 "
                    f"(grade {assessment.quality_grade})."
                )
                issues = list(assessment.actionable_issues or [])
                for dim in assessment.dimensions:
                    issues.extend(dim.issues_found or [])
                # Deduplicate while preserving order; cap to keep notes readable.
                seen = set()
                uniq_issues = []
                for it in issues:
                    if it and it not in seen:
                        seen.add(it)
                        uniq_issues.append(it)
                if uniq_issues:
                    note_parts.append("QA issues: " + "; ".join(uniq_issues[:4]) + ".")
            if conversion and conversion not in ("none", "no_conversion"):
                note_parts.append(
                    f"Frequency-aligned from {prov.get('original_frequency', 'source frequency')} "
                    f"to {integrated_dataset.frequency} via {conversion}."
                )
            notes = " ".join(note_parts)

            variables.append(VariableEntry(
                variable_name=var,
                description=description,
                source=source,
                units=units,
                frequency=integrated_dataset.frequency,
                time_coverage=integrated_dataset.time_period,
                transformations=conversion,
                notes=notes
            ))

        # Create data sources info
        data_sources = []
        for api in integrated_dataset.source_apis:
            data_sources.append({
                "name": api,
                "url": self._get_api_url(api),
                "description": self._get_api_description(api),
                "citation": self._get_api_citation(api)
            })

        # LLM-generated methodology narrative explaining source/series choices,
        # the frequency-alignment method + its consequences, and what the QA
        # scores mean. Falls back to a deterministic narrative on any error.
        methodology_narrative = self._generate_methodology_narrative(
            integrated_dataset, quality_assessments, variables)

        codebook = DataCodebook(
            title=f"Data Codebook: {integrated_dataset.dataset_name}",
            version="1.0.0",
            created_date=datetime.now().strftime('%Y-%m-%d'),
            variables=variables,
            summary_statistics={
                "num_variables": integrated_dataset.num_variables,
                "num_observations": integrated_dataset.num_observations,
                "time_period": integrated_dataset.time_period,
                "frequency": integrated_dataset.frequency
            },
            data_sources=data_sources,
            methodology=f"Data retrieved from open-source APIs ({', '.join(integrated_dataset.source_apis)}), "
                       f"temporally aligned to {integrated_dataset.frequency} frequency, "
                       f"and merged using {integrated_dataset.merge_strategy} strategy.",
            methodology_narrative=methodology_narrative,
            limitations=[
                "Data may contain measurement errors from original sources",
                "Some variables may have been interpolated or aggregated",
                "Coverage may vary across variables",
                "Real-time data may have revisions"
            ],
            usage_notes="This dataset is suitable for economic research and analysis. "
                       "Users should review variable definitions and transformations before use."
        )

        print(f"  Created codebook with {len(variables)} variables")
        return codebook

    def _generate_methodology_narrative(
        self,
        integrated_dataset: IntegratedDataset,
        quality_assessments: List[QualityAssessment],
        variables: List[VariableEntry]
    ) -> str:
        """Generate a methodology narrative for the codebook.

        Explains (1) source/series choices, (2) the frequency-alignment method
        and its consequences, and (3) what the QA scores mean. Uses the LLM when
        available; falls back to a deterministic, fully-populated narrative.
        """
        avg_score = (
            sum(a.overall_score for a in quality_assessments) / len(quality_assessments)
            if quality_assessments else 0.0
        )

        # Compact, factual context for the model.
        series_lines = []
        for v in variables:
            series_lines.append(
                f"- {v.variable_name} (source {v.source}, units {v.units}, "
                f"transform {v.transformations})"
            )
        conv_methods = sorted({
            p.get("conversion_method", "none")
            for p in integrated_dataset.provenance.values()
        })
        context = (
            f"Dataset: {integrated_dataset.dataset_name}\n"
            f"Sources: {', '.join(integrated_dataset.source_apis)}\n"
            f"Target frequency: {integrated_dataset.frequency}\n"
            f"Merge strategy: {integrated_dataset.merge_strategy}\n"
            f"Time period: {integrated_dataset.time_period}\n"
            f"Observations: {integrated_dataset.num_observations}\n"
            f"Frequency-conversion methods used: {', '.join(conv_methods)}\n"
            f"Average QA score: {avg_score:.1f}/100\n"
            f"Variables:\n" + "\n".join(series_lines)
        )

        prompt = (
            "Write a concise (180-260 word) methodology narrative for an economic "
            "data codebook. Cover three things explicitly: (1) why these sources "
            "and series were chosen for the dataset; (2) the frequency-alignment "
            "method used to bring all series to the target frequency and its "
            "analytical consequences (e.g. aggregation averaging smooths "
            "within-period variation; interpolation can introduce serial "
            "correlation); and (3) how to interpret the QA scores (the 0-100 "
            "scale and letter grades) when judging fitness for analysis. Write in "
            "plain prose, no headings, no bullet lists.\n\n"
            f"Context:\n{context}"
        )

        try:
            response = self.llm.invoke([
                {"role": "system", "content": "You are an economics data documentation specialist."},
                {"role": "user", "content": prompt},
            ])
            narrative = (response or "").strip()
            if narrative:
                return narrative
        except Exception as e:  # pragma: no cover - network/LLM failure path
            print(f"  [WARN] methodology narrative LLM call failed ({e}); using fallback")

        # Deterministic fallback narrative (always fully populated).
        conv_desc = ", ".join(conv_methods) if conv_methods else "no conversion"
        return (
            f"This dataset integrates {integrated_dataset.num_variables - 1} economic "
            f"series from {', '.join(integrated_dataset.source_apis)}, selected because "
            f"they directly measure the quantities required by the research question and "
            f"are available at consistent, well-documented frequencies over "
            f"{integrated_dataset.time_period}. To create a single analysable panel, all "
            f"series were temporally aligned to {integrated_dataset.frequency} frequency "
            f"using {conv_desc}; aggregation by averaging smooths within-period variation "
            f"(reducing high-frequency noise but masking short-lived shocks), while any "
            f"interpolation can introduce mild serial correlation that should be accounted "
            f"for in time-series estimation. Sources were merged using the "
            f"'{integrated_dataset.merge_strategy}' strategy on the common date index. "
            f"Data quality was scored on a 0-100 scale across completeness, accuracy, "
            f"consistency, and timeliness, summarised by letter grades (A>=90, B>=80, "
            f"C>=70, D>=60, F<60); the average quality score here is {avg_score:.1f}/100. "
            f"Scores at or above 80 indicate the series is suitable for analysis with "
            f"routine care, while lower scores flag variables whose issues (listed in the "
            f"per-variable notes) should be reviewed before use."
        )
    
    def generate_report(
        self,
        integrated_dataset: IntegratedDataset,
        quality_assessments: List[QualityAssessment],
        research_question: str
    ) -> DataReport:
        """Generate comprehensive API data acquisition report."""
        
        print(f"\n[{self.agent_name}] Generating data acquisition report...")
        
        # Generate report sections
        report = DataReport(
            title="Open Source API Data Acquisition Report",
            research_question=research_question,
            executive_summary=self._generate_executive_summary(integrated_dataset, quality_assessments),
            api_sources_section=self._generate_sources_section(integrated_dataset),
            series_retrieval_section=self._generate_retrieval_section(integrated_dataset),
            quality_assessment_section=self._generate_quality_section(quality_assessments),
            temporal_alignment_section=self._generate_alignment_section(integrated_dataset),
            integration_section=self._generate_integration_section(integrated_dataset),
            transformations_section=self._generate_transformations_section(integrated_dataset),
            citations=self._generate_citations(integrated_dataset.source_apis)
        )
        
        print(f"  Generated report with {len(report.citations)} citations")
        return report
    
    def _generate_executive_summary(
        self,
        dataset: IntegratedDataset,
        assessments: List[QualityAssessment]
    ) -> str:
        """Generate executive summary."""
        
        avg_score = sum(a.overall_score for a in assessments) / len(assessments) if assessments else 75
        
        return f"""This report documents the acquisition and processing of economic data from open-source APIs 
for the research project. Data was retrieved from {len(dataset.source_apis)} API sources 
({', '.join(dataset.source_apis)}), resulting in a unified dataset with {dataset.num_variables} variables 
and {dataset.num_observations} observations covering {dataset.time_period} at {dataset.frequency} frequency.

The overall data quality score is {avg_score:.1f}/100, indicating {'high' if avg_score >= 80 else 'acceptable'} 
quality for research purposes. All data has been temporally aligned and integrated following 
best practices for economic data processing."""
    
    def _generate_sources_section(self, dataset: IntegratedDataset) -> str:
        """Generate API sources section."""
        
        section = "## API Sources Used\n\n"
        
        for api in dataset.source_apis:
            section += f"### {api}\n"
            section += f"- **URL**: {self._get_api_url(api)}\n"
            section += f"- **Description**: {self._get_api_description(api)}\n"
            section += f"- **Access**: {'API key required' if api == 'FRED' else 'Public (no key required)'}\n\n"
        
        return section
    
    def _generate_retrieval_section(self, dataset: IntegratedDataset) -> str:
        """Generate series retrieval section."""
        
        section = "## Series Retrieval\n\n"
        section += f"Retrieved {len(dataset.variables) - 1} data series from {len(dataset.source_apis)} sources.\n\n"
        section += "### Variables Retrieved\n"
        
        for var in dataset.variables:
            if var != "date":
                source = dataset.provenance.get(var, {}).get("source", "Unknown")
                section += f"- **{var}**: Retrieved from {source}\n"
        
        return section
    
    def _generate_quality_section(self, assessments: List[QualityAssessment]) -> str:
        """Generate quality assessment section."""
        
        section = "## Quality Assessment\n\n"
        
        if assessments:
            avg_score = sum(a.overall_score for a in assessments) / len(assessments)
            section += f"**Average Quality Score**: {avg_score:.1f}/100\n\n"
            
            section += "### Quality by Series\n"
            for a in assessments:
                section += f"- **{a.series_id}**: {a.overall_score:.1f}/100 ({a.quality_grade})\n"
        else:
            section += "Quality assessments not available.\n"
        
        return section
    
    def _generate_alignment_section(self, dataset: IntegratedDataset) -> str:
        """Generate temporal alignment section."""
        
        section = "## Temporal Alignment\n\n"
        section += f"**Target Frequency**: {dataset.frequency}\n\n"
        section += "### Conversion Methods\n"
        
        for var, info in dataset.provenance.items():
            method = info.get("conversion_method", "none")
            orig_freq = info.get("original_frequency", "unknown")
            section += f"- **{var}**: {orig_freq} → {dataset.frequency} ({method})\n"
        
        return section
    
    def _generate_integration_section(self, dataset: IntegratedDataset) -> str:
        """Generate multi-source integration section."""
        
        section = "## Multi-Source Integration\n\n"
        section += f"**Merge Strategy**: {dataset.merge_strategy}\n"
        section += f"**Sources Integrated**: {', '.join(dataset.source_apis)}\n"
        section += f"**Final Dataset**: {dataset.num_variables} variables, {dataset.num_observations} observations\n"
        section += f"**Time Period**: {dataset.time_period}\n"
        
        return section
    
    def _generate_transformations_section(self, dataset: IntegratedDataset) -> str:
        """Generate final transformations section."""
        
        section = "## Final Transformations\n\n"
        section += "The following transformations were applied to create the final dataset:\n\n"
        section += "1. **Date Standardization**: All dates converted to YYYY-MM-DD format\n"
        section += "2. **Variable Naming**: Standardized variable names using series IDs\n"
        section += "3. **Missing Value Handling**: Applied appropriate imputation methods\n"
        section += "4. **Frequency Alignment**: Converted all series to common frequency\n"
        
        return section
    
    def _generate_citations(self, source_apis: List[str]) -> List[str]:
        """Generate proper citations for data sources."""
        
        citations = []
        
        for api in source_apis:
            citations.append(self._get_api_citation(api))
        
        return citations
    
    def generate_innovation_analysis(
        self,
        integrated_dataset: IntegratedDataset,
        research_question: str
    ) -> Dict:
        """Generate innovative analytical suggestions for the collected data."""

        print(f"\n[{self.agent_name}] Generating innovation analysis...")

        variables_text = ", ".join(v for v in integrated_dataset.variables if v != "date")
        sources_text = ", ".join(integrated_dataset.source_apis)

        system_prompt = """You are a creative economics researcher who specializes in finding
novel analytical approaches and unconventional research methodologies."""
        user_prompt = """Given the following economic dataset, suggest INNOVATIVE analytical approaches
that could yield novel economic insights.

Research Question: {research_question}

Dataset Information:
- Variables: {variables}
- Sources: {sources}
- Time Period: {time_period}
- Frequency: {frequency}
- Observations: {num_observations}

Provide your analysis as JSON with these sections:
{{{{
  "innovative_methods": [
    {{{{
      "method_name": "Name of the innovative method",
      "description": "Detailed description of the approach",
      "novelty_justification": "Why this is innovative and what new insights it could provide",
      "required_tools": ["tool1", "tool2"]
    }}}}
  ],
  "cross_domain_connections": [
    {{{{
      "connection": "Description of cross-domain insight",
      "domains": ["domain1", "domain2"],
      "potential_impact": "What this could reveal"
    }}}}
  ],
  "unconventional_data_combinations": [
    {{{{
      "combination": "Description of novel data combination",
      "variables_involved": ["var1", "var2"],
      "expected_insight": "What combining these could reveal"
    }}}}
  ],
  "emerging_techniques": [
    {{{{
      "technique": "Name of emerging analytical technique",
      "application": "How to apply it to this dataset",
      "advantage_over_traditional": "Why this is better than traditional approaches"
    }}}}
  ]
}}}}

Generate at least 3 innovative methods, 2 cross-domain connections, 2 unconventional
data combinations, and 2 emerging techniques. Be SPECIFIC and CREATIVE.

Respond with ONLY the JSON object, no other text."""

        try:
            result = self.llm.format_and_invoke(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                variables={
                    "research_question": research_question,
                    "variables": variables_text,
                    "sources": sources_text,
                    "time_period": integrated_dataset.time_period,
                    "frequency": integrated_dataset.frequency,
                    "num_observations": str(integrated_dataset.num_observations)
                }
            )

            innovation_data = json.loads(repair_json(result))
            num_items = (len(innovation_data.get("innovative_methods", [])) +
                        len(innovation_data.get("cross_domain_connections", [])) +
                        len(innovation_data.get("unconventional_data_combinations", [])) +
                        len(innovation_data.get("emerging_techniques", [])))
            print(f"  Generated {num_items} innovation suggestions")
            return innovation_data

        except Exception as e:
            print(f"    Error generating innovation analysis: {e}")
            return {
                "innovative_methods": [
                    {"method_name": "Causal Machine Learning", "description": "Apply causal forests or double ML to identify heterogeneous treatment effects in economic policy", "novelty_justification": "Combines ML flexibility with causal identification", "required_tools": ["econml", "sklearn"]},
                    {"method_name": "Spectral Analysis of Economic Cycles", "description": "Use wavelet decomposition to identify multi-scale cyclical patterns", "novelty_justification": "Reveals frequency-dependent relationships invisible to standard time-series methods", "required_tools": ["pywt", "scipy"]},
                    {"method_name": "Network-based Contagion Modeling", "description": "Model economic variable interdependencies as a dynamic network", "novelty_justification": "Captures nonlinear spillover effects between economic indicators", "required_tools": ["networkx", "pyvis"]}
                ],
                "cross_domain_connections": [
                    {"connection": "Apply information theory (entropy measures) to quantify uncertainty in economic forecasts", "domains": ["Information Theory", "Macroeconomics"], "potential_impact": "Better uncertainty quantification than traditional confidence intervals"}
                ],
                "unconventional_data_combinations": [
                    {"combination": "Combine macroeconomic time series with high-frequency financial data", "variables_involved": ["GDP", "S&P 500 intraday volatility"], "expected_insight": "Real-time economic nowcasting"}
                ],
                "emerging_techniques": [
                    {"technique": "Transformer-based Time Series Forecasting", "application": "Apply temporal attention mechanisms to economic indicator prediction", "advantage_over_traditional": "Captures long-range dependencies without explicit lag specification"}
                ]
            }

    def _get_api_url(self, api: str) -> str:
        """Get API URL."""
        urls = {
            "FRED": "https://fred.stlouisfed.org/",
            "Yahoo Finance": "https://finance.yahoo.com/",
            "World Bank": "https://data.worldbank.org/",
            "OECD": "https://data.oecd.org/"
        }
        return urls.get(api, "Unknown")
    
    def _get_api_description(self, api: str) -> str:
        """Get API description."""
        descriptions = {
            "FRED": "Federal Reserve Economic Data - comprehensive economic time series from the Federal Reserve Bank of St. Louis",
            "Yahoo Finance": "Financial market data including stock prices, indices, and company financials",
            "World Bank": "World Bank Open Data - development indicators and global economic statistics",
            "OECD": "OECD Data - economic statistics from OECD member countries"
        }
        return descriptions.get(api, "Unknown data source")
    
    def _get_api_citation(self, api: str) -> str:
        """Get proper citation for API."""
        citations = {
            "FRED": "Federal Reserve Bank of St. Louis, Federal Reserve Economic Data (FRED). Retrieved from https://fred.stlouisfed.org/",
            "Yahoo Finance": "Yahoo Finance. Retrieved from https://finance.yahoo.com/",
            "World Bank": "World Bank. World Development Indicators. Retrieved from https://data.worldbank.org/",
            "OECD": "OECD. OECD Data. Retrieved from https://data.oecd.org/"
        }
        return citations.get(api, f"{api}. Retrieved from unknown source.")



class Archivist:
    """Replication-assurance agent. Deterministic (no LLM): consolidates the
    audit trail already produced by Stage 1/2/3 into one manifest, and adds
    what nothing upstream captures — the environment the pipeline ran in and
    a content fingerprint of the exact pipeline code that ran — so a dataset
    can be reproduced independent of the run that first produced it."""

    ARCHIVIST_VERSION = "1.0.0"
    KEY_PACKAGES = ["pandas", "numpy", "pydantic", "requests", "python-dotenv", "openai", "fredapi"]
    STAGE_FILES = ["1-DataSourceStage.py", "2-DataCleaningStage.py", "3-QualityAssuranceStage.py"]

    def __init__(self, collector: Optional[MetricsCollector] = None):
        self.agent_name = "Archivist"
        self.collector = collector
        self._guard = StateGuard()

    def _capture_environment(self) -> Dict:
        """Python version, platform, and installed versions of the packages
        this pipeline actually depends on."""
        packages = {}
        for pkg in self.KEY_PACKAGES:
            try:
                packages[pkg] = importlib_metadata.version(pkg)
            except importlib_metadata.PackageNotFoundError:
                packages[pkg] = "not installed"
        return {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
        }

    def _fingerprint_pipeline_code(self, stage_dir: Path) -> Dict:
        """SHA-256 fingerprint of the exact stage scripts that ran, so the
        code version that produced a dataset can be verified independent of
        git history or timestamps."""
        file_hashes = {}
        for fname in self.STAGE_FILES:
            fpath = stage_dir / fname
            if fpath.exists():
                file_hashes[fname] = hashlib.sha256(fpath.read_bytes()).hexdigest()[:16]
            else:
                file_hashes[fname] = "file_not_found"
        combined = hashlib.sha256("".join(file_hashes.values()).encode()).hexdigest()[:16]
        return {"stage_file_hashes": file_hashes, "pipeline_fingerprint": combined}

    def _build_audit_trail(self, doc: "DocumentedDataset") -> List[str]:
        """Consolidate the per-variable provenance already tracked across
        Stage 1 (retrieval), Stage 2 (cleaning/alignment/integration), and
        Stage 3 (QA/certification) into one readable trail."""
        ds = doc.integrated_dataset
        trail = [
            f"Retrieved from: {', '.join(ds.source_apis)}",
            f"Merged via: {ds.merge_strategy} "
            f"({ds.num_variables} variables, {ds.num_observations} observations)",
        ]
        conv_methods = sorted({
            p.get("conversion_method", "none")
            for p in ds.provenance.values() if isinstance(p, dict)
        })
        if conv_methods:
            trail.append(f"Frequency-aligned to {ds.frequency} via: {', '.join(conv_methods)}")
        trail.append(f"Certified {doc.certification_level} (quality score {doc.quality_score:.1f}/100)")
        if doc.data_simulated:
            trail.append("WARNING: contains simulated series (certification capped)")
        sandbox = (doc.documentation or {}).get("sandbox_validation", {})
        if sandbox:
            trail.append(f"Sandbox validation: {'passed' if sandbox.get('validated') else 'failed'}")
        return trail

    def build_manifest(
        self,
        documented_datasets: List["DocumentedDataset"],
        data_cleaning_output: Dict,
        stage_dir: Path,
    ) -> Dict:
        """Build the replication manifest for this run."""
        print(f"\n[{self.agent_name}] Building replication manifest...")

        environment = self._capture_environment()
        pipeline_fingerprint = self._fingerprint_pipeline_code(stage_dir)
        cleaning_output_hash = self._guard.sign_output(
            data_cleaning_output, stage_name="DataCleaningStage_input_to_QA")

        dataset_entries = []
        for doc in documented_datasets:
            content_hash = self._guard.sign_output(
                doc.integrated_dataset.model_dump(), stage_name=f"dataset:{doc.dataset_id}")
            dataset_entries.append({
                "dataset_id": doc.dataset_id,
                "dataset_name": doc.dataset_name,
                "content_hash": content_hash,
                "certification_level": doc.certification_level,
                "quality_score": doc.quality_score,
                "data_simulated": doc.data_simulated,
                "audit_trail": self._build_audit_trail(doc),
            })

        manifest = {
            "archivist_version": self.ARCHIVIST_VERSION,
            "generated_at": datetime.now().isoformat(),
            "environment": environment,
            "pipeline_code_fingerprint": pipeline_fingerprint,
            "upstream_cleaning_output_hash": cleaning_output_hash,
            "datasets": dataset_entries,
        }

        print(f"  Pipeline fingerprint: {pipeline_fingerprint['pipeline_fingerprint']}")
        print(f"  Datasets archived: {len(dataset_entries)}")
        return manifest


# ============================================================================
# HITL Checkpoint
# ============================================================================

def checkpoint5_final_approval(
    documented_dataset: DocumentedDataset,
    research_question: str = "",
):
    """HITL Checkpoint 5: Final Approval (the printed question is the prompt; the package
    under review — including the deterministic checks — is the context)."""
    from DataTeam.ael.hitl import ask
    ds = documented_dataset.integrated_dataset
    sandbox = (documented_dataset.documentation or {}).get("sandbox_validation", {}) or {}
    alignment = getattr(ds, "alignment", None) or {}
    lines = [f"Research question: {research_question}",
             f"Dataset: {documented_dataset.dataset_name}",
             f"Variables: {', '.join(v for v in ds.variables if v != 'date')}",
             f"Observations (complete cases): {ds.num_observations} — {ds.time_period} "
             f"({ds.frequency})",
             f"Quality score: {documented_dataset.quality_score:.1f}; certification: "
             f"{documented_dataset.certification_level}",
             f"Deterministic checks: {sandbox.get('deterministic_verdict', '?')}"
             + (f" — {'; '.join(sandbox.get('deterministic_issues') or [])[:300]}"
                if sandbox.get('deterministic_issues') else ""),
             f"Simulated data present: {documented_dataset.data_simulated}",
             f"Codebook: {len(documented_dataset.codebook.variables)} variables; citations: "
             f"{len(documented_dataset.report.citations)}"]
    ctx = {
        "research_question": research_question,
        "variables": ds.variables, "num_observations": ds.num_observations,
        "time_period": ds.time_period, "frequency": ds.frequency,
        "quality_score": documented_dataset.quality_score,
        "certification": documented_dataset.certification_level,
        "data_simulated": documented_dataset.data_simulated,
        "deterministic_verdict": sandbox.get("deterministic_verdict"),
        "deterministic_issues": sandbox.get("deterministic_issues"),
        "excluded_series": alignment.get("excluded"),
        "data_preview": ds.data_preview[-5:],
        "codebook_limitations": documented_dataset.codebook.limitations,
    }
    return ask("Final Approval", lines,
               ["Is the dataset ready for the analysis the research question needs?",
                "Are the documentation, limitations and citations complete and accurate?"],
               default=get_default("final_approval") or "approved", context=ctx)


# ============================================================================
# Orchestrator
# ============================================================================

class QualityAssuranceOrchestrator:
    """Orchestrator for the quality assurance stage."""

    def __init__(self, openai_api_key: Optional[str] = None, collector: Optional[MetricsCollector] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        self.validation_agent = FinalValidationAgent(self.api_key, collector=collector)
        self.documentation_agent = DocumentationAgent(self.api_key, collector=collector)
        self.archivist = Archivist(collector=collector)
        self.qa_output: Optional[QualityAssuranceOutput] = None
        self.report_content: str = ""
        self.replication_manifest: Optional[Dict] = None
    
    def run_qa_pipeline(
        self,
        data_cleaning_output: Dict,
        research_question: str = "",
        enable_hitl: bool = True
    ) -> QualityAssuranceOutput:
        """Run the complete quality assurance pipeline."""
        
        print(f"\n{'='*70}")
        print(f"REPORT & CODEBOOK GENERATION PIPELINE")
        print(f"{'='*70}")
        
        # Parse integrated datasets
        integrated_data_raw = data_cleaning_output.get('integrated_datasets', [])
        integrated_datasets = [IntegratedDataset(**d) for d in integrated_data_raw]
        
        # Parse quality assessments
        qa_raw = data_cleaning_output.get('quality_assessments', [])
        quality_assessments = [QualityAssessment(**a) for a in qa_raw]
        
        research_q = data_cleaning_output.get('metadata', {}).get('research_question', research_question)
        if not research_q:
            research_q = "Economic data analysis"

        # Data-integrity flag propagated from Stage 1/2 (simulated/synthetic data).
        data_simulated = bool(data_cleaning_output.get('metadata', {}).get('data_simulated', False))
        _meta = data_cleaning_output.get('metadata', {}) or {}
        _n_sim = int(_meta.get('num_simulated', 0) or 0)
        _n_all = int(_meta.get('num_retrieved', 0) or 0)
        # Proportional share; older cleaning JSON without the count falls back to 1.0
        # (the original whole-dataset behavior) so the cap can never LOOSEN silently.
        sim_share = (_n_sim / _n_all) if (_n_all and _n_sim) else (1.0 if data_simulated else 0.0)
        if data_simulated:
            print(f"[DATA INTEGRITY] Upstream data contains SIMULATED series "
                  f"(share {sim_share:.0%}) — certification will be capped.")

        print(f"Integrated Datasets: {len(integrated_datasets)}")
        print(f"Quality Assessments: {len(quality_assessments)}")
        print(f"{'='*70}\n")
        
        documented_datasets = []
        hitl_results = []
        upstream_limitations = list(_meta.get("limitations") or [])

        for dataset in integrated_datasets:
            print(f"\nProcessing: {dataset.dataset_name}")
            print(f"-"*70)
            
            # Step 1: Final Validation
            print(f"\nSTEP 1: FINAL VALIDATION")
            quality_score, certification = self.validation_agent.validate_dataset(
                dataset,
                quality_assessments,
                data_simulated=data_simulated,
                sim_share=sim_share,
            )
            
            # Step 2: Generate Codebook
            print(f"\nSTEP 2: GENERATE CODEBOOK")
            codebook = self.documentation_agent.generate_codebook(
                dataset,
                quality_assessments
            )
            
            # Step 3: Generate Report
            print(f"\nSTEP 3: GENERATE REPORT")
            report = self.documentation_agent.generate_report(
                dataset,
                quality_assessments,
                research_q
            )
            
            # Step 4: Innovation Analysis
            print(f"\nSTEP 4: INNOVATION ANALYSIS")
            innovation_analysis = self.documentation_agent.generate_innovation_analysis(
                dataset, research_q
            )

            # Step 5: Sandbox Data Validation
            print(f"\nSTEP 5: SANDBOX DATA VALIDATION")
            sandbox_result = self._validate_data_with_sandbox(dataset, quality_assessments)
            # deterministic findings and upstream reviewer objections reach the codebook
            for issue in sandbox_result.get("deterministic_issues") or []:
                codebook.limitations.append(f"Deterministic data check failed: {issue}")
            codebook.limitations.extend(upstream_limitations)

            # Create citation
            citation = self._generate_dataset_citation(dataset)

            # Create documented dataset
            documented = DocumentedDataset(
                dataset_id=dataset.dataset_id,
                dataset_name=dataset.dataset_name,
                integrated_dataset=dataset,
                codebook=codebook,
                report=report,
                quality_score=quality_score,
                certification_level=certification,
                citation=citation,
                data_simulated=data_simulated,
                documentation={
                    "sections": [
                        {"title": "Overview", "content": report.executive_summary},
                        {"title": "Sources", "content": report.api_sources_section},
                        {"title": "Quality", "content": report.quality_assessment_section},
                        {"title": "Innovation Analysis", "content": json.dumps(innovation_analysis, indent=2)}
                    ],
                    "innovation_suggestions": innovation_analysis,
                    "sandbox_validation": sandbox_result,
                    "data_simulated": data_simulated
                }
            )
            
            # HITL Checkpoint 5: Final Approval — an objection is recorded in the dataset's
            # limitations and the stage metadata, never only printed
            if enable_hitl:
                cp5 = checkpoint5_final_approval(documented, research_q)
                hitl_results.append(cp5)
                print(f"[HITL] Final approval: {'approved' if cp5.approved else 'OBJECTION'} "
                      f"('{cp5.response[:120]}')")
                if not cp5.approved:
                    documented.codebook.limitations.append(cp5.limitation())
            
            documented_datasets.append(documented)

        # Step 6: Archivist builds the replication manifest for this run
        print(f"\nSTEP 6: REPLICATION MANIFEST")
        stage_dir = Path(__file__).resolve().parent
        self.replication_manifest = self.archivist.build_manifest(
            documented_datasets, data_cleaning_output, stage_dir
        )

        # Create output
        self.qa_output = QualityAssuranceOutput(
            integrated_datasets=integrated_datasets,
            quality_assessments=quality_assessments,
            documented_datasets=documented_datasets,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "num_integrated": len(integrated_datasets),
                "num_documented": len(documented_datasets),
                "research_question": research_q,
                "data_simulated": data_simulated,
                "hitl_checkpoints": [r.to_record() for r in hitl_results],
                "hitl_objections": [r.to_record() for r in hitl_results if not r.approved],
                "limitations": upstream_limitations + [
                    r.limitation() for r in hitl_results if r.limitation()],
            },
            replication_manifest=self.replication_manifest,
        )
        
        # Store report content for saving
        if documented_datasets:
            self.report_content = self._compile_full_report(documented_datasets[0], research_q)
        
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Documented Datasets: {len(documented_datasets)}")
        if documented_datasets:
            print(f"Quality Score: {documented_datasets[0].quality_score:.1f}")
            print(f"Certification: {documented_datasets[0].certification_level}")
        print(f"{'='*70}\n")
        
        return self.qa_output
    
    def _validate_data_with_sandbox(
        self,
        dataset: IntegratedDataset,
        quality_assessments: List[QualityAssessment]
    ) -> Dict:
        """Generate and execute data validation code in sandbox."""

        variables = [v for v in dataset.variables if v != "date"]
        preview_json = json.dumps(dataset.data_preview[:5], indent=2)
        qa_text = "\n".join([
            f"- {a.series_id}: score={a.overall_score}, grade={a.quality_grade}"
            for a in quality_assessments
        ])

        try:
            result = self.documentation_agent.llm.invoke([
                {"role": "system", "content": "You are a data quality engineer. Generate ONLY executable Python code, no markdown fences."},
                {"role": "user", "content": """Generate a Python script that validates this economic dataset.

Dataset: {dataset_name}
Variables: {variables}
Observations: {num_obs}
Time Period: {time_period}
Frequency: {frequency}

Data Preview (first rows):
{preview}

Quality Assessments:
{qa_text}

Requirements:
1. Use ONLY numpy and json (import numpy as np, import json) — no other imports
2. Define the data preview as a list of dicts
3. Compute summary statistics: count, min, max, mean for each numeric variable
4. Check for basic quality issues: missing values, outliers (>3 std from mean)
5. Print results as JSON: {{"validated": true/false, "statistics": [{{"variable": "...", "count": N, "mean": X, "min": Y, "max": Z}}], "issues": ["..."], "summary": "..."}}

Respond with ONLY the Python code, no explanations or markdown.
""".format(
                    dataset_name=dataset.dataset_name,
                    variables=", ".join(variables),
                    num_obs=dataset.num_observations,
                    time_period=dataset.time_period,
                    frequency=dataset.frequency,
                    preview=preview_json,
                    qa_text=qa_text or "None available"
                )}
            ])

            code = CodeSandbox.clean_llm_output(result)

            sandbox = CodeSandbox(timeout_sec=60, memory_mb=256)
            exec_result = sandbox.execute(code)

            # "validated" requires both that the generated script ran and that the
            # deterministic dataset checks pass; the script's own output is kept as a descriptive record.
            verdict, issues = dataset_check_verdict(dataset)
            validation = {
                "validated": bool(exec_result.success) and verdict == "pass",
                "validation_method": "sandbox_execution",
                "requires": "deterministic dataset checks pass AND the script runs",
                "deterministic_verdict": verdict,
                "deterministic_issues": issues,
                "sandbox_executed": bool(exec_result.success),
                "code": code,
                "stdout": exec_result.stdout,
                "stderr": exec_result.stderr,
                "error": exec_result.error,
                "execution_time_sec": exec_result.execution_time_sec
            }

            if validation["validated"]:
                print(f"  Validation PASSED (deterministic checks + sandbox, "
                      f"{exec_result.execution_time_sec:.1f}s)")
            elif verdict != "pass":
                print(f"  Validation FAILED on deterministic checks: {'; '.join(issues)[:300]}")
            else:
                print(f"  Sandbox validation FAILED: {exec_result.error or exec_result.stderr[:200]}")

            return validation

        except Exception as e:
            print(f"  Sandbox validation error: {e}")
            verdict, issues = dataset_check_verdict(dataset)
            return {
                "validated": False,
                "validation_method": "sandbox_execution",
                "requires": "deterministic dataset checks pass AND the script runs",
                "deterministic_verdict": verdict,
                "deterministic_issues": issues,
                "error": f"Code generation failed: {e}"
            }

    def _generate_dataset_citation(self, dataset: IntegratedDataset) -> str:
        """Generate citation for the dataset."""

        return f"""{dataset.dataset_name}. Version 1.0.0.
Generated: {datetime.now().strftime('%Y-%m-%d')}.
Sources: {', '.join(dataset.source_apis)}.
Time Period: {dataset.time_period}. Frequency: {dataset.frequency}."""
    
    def _compile_full_report(self, documented: DocumentedDataset, research_question: str) -> str:
        """Compile the full report as markdown."""
        
        report = documented.report
        codebook = documented.codebook
        
        content = f"""# {report.title}

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Research Question**: {research_question}

---

## Executive Summary

{report.executive_summary}

---

{report.api_sources_section}

---

{report.series_retrieval_section}

---

{report.quality_assessment_section}

---

{report.temporal_alignment_section}

---

{report.integration_section}

---

{report.transformations_section}

---

## Data Codebook

### Dataset Information
- **Title**: {codebook.title}
- **Version**: {codebook.version}
- **Created**: {codebook.created_date}
- **Variables**: {codebook.summary_statistics.get('num_variables', 'N/A')}
- **Observations**: {codebook.summary_statistics.get('num_observations', 'N/A')}
- **Time Period**: {codebook.summary_statistics.get('time_period', 'N/A')}
- **Frequency**: {codebook.summary_statistics.get('frequency', 'N/A')}

### Variable Descriptions

| Variable | Description | Source | Units | Frequency |
|----------|-------------|--------|-------|-----------|
"""
        
        for var in codebook.variables:
            content += f"| {var.variable_name} | {var.description} | {var.source} | {var.units} | {var.frequency} |\n"
        
        content += f"""
### Methodology

{codebook.methodology}

### Known Limitations

"""
        for limitation in codebook.limitations:
            content += f"- {limitation}\n"
        
        content += f"""
### Usage Notes

{codebook.usage_notes}

---

## Citations

"""
        for citation in report.citations:
            content += f"- {citation}\n"
        
        # Add Innovation Analysis section
        innovation = documented.documentation.get("innovation_suggestions", {})
        content += "\n---\n\n## Innovation Analysis\n\n"
        content += "### Innovative Analytical Methods\n\n"
        for method in innovation.get("innovative_methods", []):
            content += f"**{method.get('method_name', 'N/A')}**: {method.get('description', '')}\n"
            content += f"- *Novelty*: {method.get('novelty_justification', '')}\n\n"

        content += "### Cross-Domain Connections\n\n"
        for conn in innovation.get("cross_domain_connections", []):
            content += f"- **{conn.get('connection', '')}** (Domains: {', '.join(conn.get('domains', []))})\n"
            content += f"  - Potential Impact: {conn.get('potential_impact', '')}\n\n"

        content += "### Unconventional Data Combinations\n\n"
        for combo in innovation.get("unconventional_data_combinations", []):
            content += f"- **{combo.get('combination', '')}**\n"
            content += f"  - Expected Insight: {combo.get('expected_insight', '')}\n\n"

        content += "### Emerging Techniques\n\n"
        for tech in innovation.get("emerging_techniques", []):
            content += f"- **{tech.get('technique', '')}**: {tech.get('application', '')}\n"
            content += f"  - Advantage: {tech.get('advantage_over_traditional', '')}\n\n"

        simulated_note = ""
        if getattr(documented, "data_simulated", False):
            simulated_note = (
                "\n> ⚠️ **DATA INTEGRITY WARNING**: This dataset is SIMULATED/synthetic "
                "(no live API retrieval succeeded). Certification is capped at Bronze and "
                "the data must NOT be used for real analysis or treated as authoritative.\n"
            )

        content += f"""
---

## How to Cite This Dataset

{documented.citation}

---
{simulated_note}
**Data Provenance**: {'SIMULATED (synthetic, not real)' if getattr(documented, 'data_simulated', False) else 'Live API retrieval'}

**Quality Score**: {documented.quality_score:.1f}/100

**Certification Level**: {documented.certification_level}

---

*Report generated by DataTeam Open Source API Pipeline*
"""
        
        return content
    
    def save_qa_output(self, filename: str = "api_qa_output.json"):
        """Save QA output to JSON."""
        if not self.qa_output:
            print("No QA output to save")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.qa_output.model_dump(), f, indent=2)
        
        print(f"[Orchestrator] QA output saved to {filename}")
    
    def save_report_codebook(self, filename: str = "api_data_report.md"):
        """Save report and codebook as markdown file."""
        if not self.report_content:
            print("No report content to save")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.report_content)

        print(f"[Orchestrator] Report and codebook saved to {filename}")

    def save_replication_manifest(self, filename: str = "replication_manifest.json"):
        """Save the Archivist's replication manifest to JSON."""
        if not self.replication_manifest:
            print("No replication manifest to save")
            return

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.replication_manifest, f, indent=2)

        print(f"[Orchestrator] Replication manifest saved to {filename}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main function for quality assurance stage."""
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Load cleaning output
    cleaning_file = "api_cleaning_output.json"
    if os.path.exists(cleaning_file):
        with open(cleaning_file, 'r', encoding='utf-8') as f:
            cleaning_data = json.load(f)
    else:
        # Create sample data for testing
        cleaning_data = {
            "integrated_datasets": [
                {
                    "dataset_id": "INT_API_001",
                    "dataset_name": "Integrated API Economic Dataset",
                    "source_apis": ["FRED", "Yahoo Finance"],
                    "num_variables": 5,
                    "num_observations": 300,
                    "time_period": "1990-01-01 to 2023-12-01",
                    "frequency": "quarterly",
                    "merge_strategy": "outer_join_with_interpolation",
                    "variables": ["date", "GDPC1", "UNRATE", "CPIAUCSL"],
                    "data_preview": [{"date": "2023-12-01", "GDPC1": 22000, "UNRATE": 3.7}],
                    "provenance": {
                        "GDPC1": {"source": "FRED", "original_frequency": "quarterly", "conversion_method": "none"},
                        "UNRATE": {"source": "FRED", "original_frequency": "monthly", "conversion_method": "aggregation_average"}
                    }
                }
            ],
            "quality_assessments": [
                {
                    "series_id": "GDPC1",
                    "series_name": "Real GDP",
                    "source_name": "FRED",
                    "dimensions": [{"dimension": "completeness", "score": 90, "issues_found": [], "recommendations": []}],
                    "overall_score": 88,
                    "quality_grade": "B",
                    "actionable_issues": []
                }
            ],
            "metadata": {
                "research_question": "What is the relationship between GDP growth, unemployment, and inflation?"
            }
        }
    
    # Run pipeline
    orchestrator = QualityAssuranceOrchestrator()
    qa_output = orchestrator.run_qa_pipeline(
        data_cleaning_output=cleaning_data,
        enable_hitl=True
    )
    
    # Save outputs
    orchestrator.save_qa_output("api_qa_output.json")
    orchestrator.save_report_codebook(f"api_data_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.md")
    orchestrator.save_replication_manifest("replication_manifest.json")
    
    print("\n" + "="*70)
    print("QUALITY ASSURANCE SUMMARY")
    print("="*70)
    print(f"Documented Datasets: {len(qa_output.documented_datasets)}")
    if qa_output.documented_datasets:
        print(f"Quality Score: {qa_output.documented_datasets[0].quality_score:.1f}")
        print(f"Certification: {qa_output.documented_datasets[0].certification_level}")
    print("="*70)


if __name__ == "__main__":
    main()
