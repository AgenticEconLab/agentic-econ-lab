# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Quality Assurance Stage - User-Uploaded Data Mode

This script handles:
- Final validation of transformed dataset
- Comprehensive report generation
- Data codebook creation

Pipeline:
1. FinalValidationAgent: Final validation of transformed dataset
2. DocumentationAgent: Generate comprehensive report and codebook

HITL Checkpoint (5):
5. Final Approval (after report generation)

Input: Cleaning output from Stage 2 (user_data_cleaning_output.json)
Output: Documented dataset with report and codebook
"""

import os
import json
import hashlib
import platform
from importlib import metadata as importlib_metadata
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
# Add parent directories to path for shared imports
import sys
current_dir = Path(__file__).resolve().parent
agents_dir = current_dir.parent.parent.parent  # repository root
if str(agents_dir) not in sys.path:
    sys.path.insert(0, str(agents_dir))

from shared.auto_input import auto_input, get_default
from shared.llm import LLMClient
from shared.json_repair import repair_json
from shared.observability import MetricsCollector
from shared.tools.sandbox_tool import CodeSandbox, ExecutionResult
from shared.reliability.state_guard import StateGuard
from DataTeam.ael.schemas.stage_outputs import (
    UserUploadedVariableEntry as VariableEntry,
    UserUploadedDataCodebook as DataCodebook,
    UserUploadedDataReport as DataReport,
    UserUploadedDocumentedDataset as DocumentedDataset,
    UserUploadedQualityAssuranceOutput as QualityAssuranceOutput,
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

class FinalValidationAgent:
    """Agent for final validation of transformed dataset."""
    
    def __init__(self, openai_api_key: str):
        self.agent_name = "FinalValidationAgent"
        self.api_key = openai_api_key
    
    def validate_dataset(
        self,
        cleaning_output: Dict
    ) -> tuple:
        """Perform final validation of transformed dataset."""
        
        print(f"\n[{self.agent_name}] Performing final validation...")
        
        # Get quality assessment from source stage
        quality_score = 75  # Default
        
        # Try to get from nested structure
        if 'quality_assessment' in cleaning_output:
            qa = cleaning_output['quality_assessment']
            if isinstance(qa, dict):
                quality_score = qa.get('overall_score', 75)
        
        # Adjust based on privacy and transformation results
        privacy = cleaning_output.get('privacy_screening', {})
        if privacy.get('risk_level') == 'High':
            quality_score -= 10
        elif privacy.get('risk_level') == 'Medium':
            quality_score -= 5
        
        # Determine certification level
        if quality_score >= 90:
            certification = "Gold"
        elif quality_score >= 75:
            certification = "Silver"
        else:
            certification = "Bronze"
        
        ready_for_analysis = quality_score >= 60 and privacy.get('risk_level') != 'High'
        
        print(f"  Quality Score: {quality_score:.1f}")
        print(f"  Certification: {certification}")
        print(f"  Ready for Analysis: {ready_for_analysis}")
        
        return quality_score, certification, ready_for_analysis


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
        cleaning_output: Dict,
        file_path: str
    ) -> DataCodebook:
        """Generate comprehensive data codebook."""
        
        print(f"\n[{self.agent_name}] Generating data codebook...")
        
        # Extract file info. file_info is NOT a field on the cleaning-output
        # schema, so Stage 2 now propagates it via metadata. Try both the direct
        # key (back-compat) and metadata, so the codebook is no longer built from
        # an empty dict (the "0 variables" bug).
        file_info = {}
        if cleaning_output.get('file_info'):
            file_info = cleaning_output['file_info']
        elif cleaning_output.get('metadata', {}).get('file_info'):
            file_info = cleaning_output['metadata']['file_info']

        # Get column information
        column_names = file_info.get('column_names', [])
        column_types = file_info.get('column_types', {})

        # Get structure inference
        structure = cleaning_output.get('structure_inference', {})
        suggested_roles = structure.get('suggested_roles', {})
        structure_type = structure.get('structure_type', 'unknown')

        # Fallback: if file_info still lacks columns, recover column names from
        # the structure inference (id/temporal/grouping/key variables) so the
        # codebook is never empty when any column information exists upstream.
        if not column_names:
            recovered = []
            for v in structure.get('id_variables', []) or []:
                recovered.append(v)
            tv = structure.get('temporal_variable')
            if tv:
                recovered.append(tv)
            for v in structure.get('grouping_variables', []) or []:
                recovered.append(v)
            for vlist in (structure.get('key_variables', {}) or {}).values():
                if isinstance(vlist, list):
                    recovered.extend(vlist)
            recovered.extend(suggested_roles.keys())
            # Deduplicate preserving order.
            seen = set()
            column_names = [c for c in recovered if c and not (c in seen or seen.add(c))]
            if column_names:
                print(f"  [INFO] file_info had no columns; recovered "
                      f"{len(column_names)} variable names from structure inference")

        # Missing-data summary (from cleaning QA) for per-variable notes.
        missing_summary = cleaning_output.get('quality_assessment', {}).get('missing_data_summary', {})
        if isinstance(missing_summary, dict):
            per_col_missing = missing_summary.get('by_column', {}) or missing_summary.get('columns', {})
        else:
            per_col_missing = {}

        # Create variable entries
        variables = []
        for col in column_names:
            role = suggested_roles.get(col, 'other')
            dtype = column_types.get(col, 'unknown')

            miss = per_col_missing.get(col, {}) if isinstance(per_col_missing, dict) else {}
            missing_count = int(miss.get('count', 0)) if isinstance(miss, dict) else 0
            missing_pct = float(miss.get('pct', miss.get('percent', 0.0))) if isinstance(miss, dict) else 0.0

            note_parts = [f"Inferred role: {role}.", f"Data type: {dtype}."]
            if missing_count:
                note_parts.append(f"{missing_count} missing values ({missing_pct:.1f}%).")
            notes = " ".join(note_parts)

            variables.append(VariableEntry(
                variable_name=col,
                description=f"Column '{col}' ({dtype}); role in analysis: {role}.",
                data_type=dtype,
                role=role,
                value_range="See summary statistics",
                missing_count=missing_count,
                missing_pct=missing_pct,
                notes=notes
            ))
        
        # LLM-generated methodology narrative explaining the inferred structure /
        # variable roles, the privacy+cleaning transformations and their
        # consequences, and how to read the QA score. Falls back to a
        # deterministic narrative on any error (parity with OpenSourceAPI mode).
        methodology_narrative = self._generate_methodology_narrative(
            cleaning_output, variables, structure_type)

        # Create codebook
        codebook = DataCodebook(
            title=f"Data Codebook: {os.path.basename(file_path)}",
            version="1.0.0",
            created_date=datetime.now().strftime('%Y-%m-%d'),
            file_info={
                "file_name": file_info.get('file_name', os.path.basename(file_path)),
                "file_size_mb": file_info.get('file_size_mb', 0),
                "num_rows": file_info.get('num_rows', 0),
                "num_columns": file_info.get('num_columns', len(column_names))
            },
            variables=variables,
            summary_statistics={
                "total_variables": len(variables),
                "numeric_variables": sum(1 for v in variables if 'int' in v.data_type or 'float' in v.data_type),
                "categorical_variables": sum(1 for v in variables if 'object' in v.data_type or 'category' in v.data_type)
            },
            data_structure=structure_type,
            methodology="Data processed through automated pipeline with quality assessment, "
                       "privacy screening, structure inference, and transformation.",
            methodology_narrative=methodology_narrative,
            limitations=self._identify_limitations(cleaning_output),
            usage_notes="Review variable descriptions and roles before analysis. "
                       "Check for any remaining data quality issues."
        )
        
        print(f"  Created codebook with {len(variables)} variables")
        return codebook

    def _generate_methodology_narrative(
        self,
        cleaning_output: Dict,
        variables: List[VariableEntry],
        structure_type: str
    ) -> str:
        """Generate a methodology narrative for the user-uploaded codebook.

        Explains (1) the inferred data structure and variable roles, (2) the
        privacy/cleaning transformations applied and their analytical
        consequences, and (3) how to interpret the QA score. Uses the LLM when
        available; falls back to a deterministic, fully-populated narrative.
        """
        quality = cleaning_output.get('quality_assessment', {}) or {}
        privacy = cleaning_output.get('privacy_screening', {}) or {}
        transformations = cleaning_output.get('transformations', {}) or {}

        overall_score = quality.get('overall_score', 'N/A')
        grade = quality.get('grade', 'N/A')
        risk_level = privacy.get('risk_level', 'N/A')
        pii_detected = bool(privacy.get('pii_detected'))
        columns_removed = transformations.get('columns_removed', []) or []
        rows_before = transformations.get('rows_before', 'N/A')
        rows_after = transformations.get('rows_after', 'N/A')
        transform_log = transformations.get('transformation_log', []) or []

        role_lines = []
        for v in variables[:25]:
            role_lines.append(f"- {v.variable_name} (type {v.data_type}, role {v.role})")

        context = (
            f"Data structure: {structure_type}\n"
            f"Variables ({len(variables)}):\n" + "\n".join(role_lines) + "\n"
            f"QA overall score: {overall_score}/100 (grade {grade})\n"
            f"Privacy risk level: {risk_level}; PII detected: {pii_detected}\n"
            f"Rows before/after cleaning: {rows_before} -> {rows_after}\n"
            f"Columns removed: {', '.join(columns_removed) if columns_removed else 'none'}\n"
            f"Transformation log: {'; '.join(str(t) for t in transform_log[:8]) if transform_log else 'none'}"
        )

        prompt = (
            "Write a concise (180-260 word) methodology narrative for a codebook "
            "documenting a USER-UPLOADED dataset. Cover three things explicitly: "
            "(1) the inferred data structure and the suggested variable roles, and why "
            "they fit the dataset; (2) the privacy screening and cleaning "
            "transformations applied (PII handling, column/row changes) and their "
            "analytical consequences (e.g. removing identifier columns protects privacy "
            "but precludes record linkage; dropping rows can bias the sample); and (3) "
            "how to interpret the QA score (0-100 scale and letter grade) when judging "
            "fitness for analysis. Write in plain prose, no headings, no bullet lists.\n\n"
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
        removed_desc = (
            f"the columns {', '.join(columns_removed)} were removed"
            if columns_removed else "no columns were removed"
        )
        return (
            f"This dataset was inferred to have a {structure_type} structure, and its "
            f"{len(variables)} variables were assigned analytical roles (outcome, treatment, "
            f"control, identifier, or other) based on their names, types, and inferred "
            f"relationships, so that downstream modelling can map columns to research "
            f"constructs without re-inspecting the raw file. During cleaning the data passed "
            f"a privacy screen (risk level {risk_level}; PII "
            f"{'was' if pii_detected else 'was not'} detected), after which {removed_desc} and "
            f"the row count moved from {rows_before} to {rows_after}; removing identifier or "
            f"PII columns protects privacy but precludes later record linkage, while any row "
            f"reduction can shift the sample composition and should be checked for selection "
            f"bias before inference. Data quality was summarised by an overall score of "
            f"{overall_score}/100 (grade {grade}) on a 0-100 scale where higher is better; "
            f"scores at or above 80 indicate the dataset is suitable for analysis with routine "
            f"care, while lower scores flag issues (missingness, inconsistent types, outliers) "
            f"listed in the per-variable notes that should be resolved before drawing "
            f"conclusions."
        )

    def generate_report(
        self,
        cleaning_output: Dict,
        file_path: str,
        research_question: str
    ) -> DataReport:
        """Generate comprehensive data acquisition report."""
        
        print(f"\n[{self.agent_name}] Generating data acquisition report...")
        
        # Extract components. file_info is propagated via metadata (it is not a
        # field on the cleaning-output schema), so check both locations.
        file_info = cleaning_output.get('file_info') or \
            cleaning_output.get('metadata', {}).get('file_info', {})
        quality = cleaning_output.get('quality_assessment', {})
        privacy = cleaning_output.get('privacy_screening', {})
        structure = cleaning_output.get('structure_inference', {})
        alignment = cleaning_output.get('research_alignment', {})
        transformations = cleaning_output.get('transformations', {})
        
        # Generate report sections
        report = DataReport(
            title="User-Uploaded Data Acquisition Report",
            file_path=file_path,
            research_question=research_question,
            executive_summary=self._generate_executive_summary(file_info, quality, privacy, structure),
            file_characteristics_section=self._generate_file_section(file_info),
            quality_assessment_section=self._generate_quality_section(quality),
            privacy_screening_section=self._generate_privacy_section(privacy),
            structure_analysis_section=self._generate_structure_section(structure),
            research_alignment_section=self._generate_alignment_section(alignment),
            transformations_section=self._generate_transformations_section(transformations),
            recommendations=self._generate_recommendations(quality, privacy, alignment)
        )
        
        print(f"  Generated report with all sections")
        return report
    
    def generate_innovation_analysis(
        self,
        research_question: str,
        data_summary: Dict
    ) -> Dict:
        """Generate innovative analytical suggestions for the uploaded data."""

        print(f"\n[{self.agent_name}] Generating innovation analysis...")

        columns = data_summary.get("columns", [])
        num_rows = data_summary.get("num_rows", 0)

        system_prompt = """You are a creative economics researcher who specializes in finding
novel analytical approaches from user-provided datasets."""

        user_prompt = """Given the following user-uploaded dataset, suggest INNOVATIVE analytical approaches
that could yield novel economic insights.

Research Question: {research_question}

Dataset Information:
- Columns: {columns}
- Number of Rows: {num_rows}
- File Type: {file_type}

Provide your analysis as JSON with these sections:
{{{{
  "innovative_methods": [
    {{{{
      "method_name": "Name of the innovative method",
      "description": "Detailed description",
      "novelty_justification": "Why this is innovative",
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
Consider what UNIQUE insights this particular dataset structure could provide.

Respond with ONLY the JSON object, no other text."""

        try:
            result = self.llm.format_and_invoke(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                variables={
                    "research_question": research_question,
                    "columns": ", ".join(columns[:15]) if columns else "various columns",
                    "num_rows": str(num_rows),
                    "file_type": data_summary.get("file_type", "CSV")
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
                    {"method_name": "Automated Feature Engineering", "description": "Apply tsfresh or featuretools to automatically extract meaningful features from raw data", "novelty_justification": "Discovers non-obvious variable transformations that domain experts might miss", "required_tools": ["tsfresh", "featuretools"]},
                    {"method_name": "Topological Data Analysis", "description": "Use persistent homology to identify structural patterns in the data manifold", "novelty_justification": "Reveals hidden geometric structure in high-dimensional economic data", "required_tools": ["giotto-tda", "ripser"]},
                    {"method_name": "Synthetic Control Methods", "description": "Construct counterfactual scenarios from the uploaded data for causal inference", "novelty_justification": "Enables causal claims from observational data without randomization", "required_tools": ["synth", "SparseSC"]}
                ],
                "cross_domain_connections": [
                    {"connection": "Apply natural language processing to any text fields for sentiment extraction", "domains": ["NLP", "Economics"], "potential_impact": "Convert qualitative data into quantitative sentiment indicators"}
                ],
                "unconventional_data_combinations": [
                    {"combination": "Link uploaded data with publicly available geographic/demographic data", "variables_involved": ["location fields", "census data"], "expected_insight": "Spatial economic analysis revealing regional heterogeneity"}
                ],
                "emerging_techniques": [
                    {"technique": "Conformal Prediction", "application": "Generate prediction intervals with guaranteed coverage probability", "advantage_over_traditional": "Distribution-free uncertainty quantification without parametric assumptions"}
                ]
            }

    def _generate_executive_summary(
        self,
        file_info: Dict,
        quality: Dict,
        privacy: Dict,
        structure: Dict
    ) -> str:
        """Generate executive summary."""
        
        return f"""This report documents the processing of a user-uploaded data file for research analysis.

**File Overview:**
- File: {file_info.get('file_name', 'Unknown')}
- Size: {file_info.get('file_size_mb', 0):.2f} MB
- Rows: {file_info.get('num_rows', 0):,}
- Columns: {file_info.get('num_columns', 0)}

**Key Findings:**
- Quality Score: {quality.get('overall_score', 'N/A')}/100 ({quality.get('grade', 'N/A')})
- Privacy Risk: {privacy.get('risk_level', 'N/A')}
- Data Structure: {structure.get('structure_type', 'N/A')}

The dataset has been processed through quality assessment, privacy screening, structure inference, 
and data transformation stages. See detailed sections below for complete analysis."""
    
    def _generate_file_section(self, file_info: Dict) -> str:
        """Generate file characteristics section."""
        
        section = "## File Characteristics\n\n"
        section += f"- **File Name**: {file_info.get('file_name', 'Unknown')}\n"
        section += f"- **File Extension**: {file_info.get('file_extension', 'Unknown')}\n"
        section += f"- **File Size**: {file_info.get('file_size_mb', 0):.2f} MB\n"
        section += f"- **Number of Rows**: {file_info.get('num_rows', 0):,}\n"
        section += f"- **Number of Columns**: {file_info.get('num_columns', 0)}\n"
        
        if file_info.get('column_names'):
            section += f"\n**Columns**: {', '.join(file_info['column_names'][:10])}"
            if len(file_info['column_names']) > 10:
                section += f"... (+{len(file_info['column_names']) - 10} more)"
        
        return section
    
    def _generate_quality_section(self, quality: Dict) -> str:
        """Generate quality assessment section."""
        
        section = "## Quality Assessment\n\n"
        section += f"**Overall Score**: {quality.get('overall_score', 'N/A')}/100\n"
        section += f"**Grade**: {quality.get('grade', 'N/A')}\n\n"
        
        if quality.get('dimensions'):
            section += "### Quality Dimensions\n\n"
            for dim in quality['dimensions']:
                if isinstance(dim, dict):
                    section += f"- **{dim.get('dimension', 'Unknown').title()}**: {dim.get('score', 'N/A')}/100\n"
                    if dim.get('issues'):
                        for issue in dim['issues'][:2]:
                            section += f"  - Issue: {issue}\n"
        
        if quality.get('actionable_issues'):
            section += "\n### Actionable Issues\n\n"
            for issue in quality['actionable_issues'][:5]:
                section += f"- {issue}\n"
        
        return section
    
    def _generate_privacy_section(self, privacy: Dict) -> str:
        """Generate privacy screening section."""
        
        section = "## Privacy Screening\n\n"
        section += f"**Risk Level**: {privacy.get('risk_level', 'N/A')}\n"
        section += f"**PII Detected**: {'Yes' if privacy.get('pii_detected') else 'No'}\n\n"
        
        if privacy.get('pii_detections'):
            section += "### PII Detections\n\n"
            for detection in privacy['pii_detections'][:5]:
                if isinstance(detection, dict):
                    section += f"- **{detection.get('column_name', 'Unknown')}**: {detection.get('pii_type', 'Unknown')}\n"
                    section += f"  - Confidence: {detection.get('confidence', 0):.2f}\n"
                    section += f"  - Recommendation: {detection.get('recommendation', 'N/A')}\n"
        
        if privacy.get('sensitive_columns'):
            section += f"\n**Sensitive Columns**: {', '.join(privacy['sensitive_columns'])}\n"
        
        if privacy.get('remediation_actions'):
            section += "\n### Remediation Actions\n\n"
            for action in privacy['remediation_actions'][:5]:
                section += f"- {action}\n"
        
        return section
    
    def _generate_structure_section(self, structure: Dict) -> str:
        """Generate structure analysis section."""
        
        section = "## Structure Analysis\n\n"
        section += f"**Data Structure**: {structure.get('structure_type', 'N/A')}\n"
        section += f"**Confidence**: {structure.get('confidence', 0):.2f}\n\n"
        
        if structure.get('key_variables'):
            section += "### Key Variables\n\n"
            for var_type, vars in structure['key_variables'].items():
                if vars:
                    section += f"- **{var_type.title()}**: {', '.join(vars)}\n"
        
        if structure.get('suggested_roles'):
            section += "\n### Suggested Variable Roles\n\n"
            for var, role in list(structure['suggested_roles'].items())[:10]:
                section += f"- **{var}**: {role}\n"
        
        return section
    
    def _generate_alignment_section(self, alignment: Dict) -> str:
        """Generate research alignment section."""
        
        section = "## Research Alignment\n\n"
        section += f"**Alignment Score**: {alignment.get('alignment_score', 'N/A')}/100\n"
        section += f"**Suitability**: {alignment.get('suitability', 'N/A')}\n"
        section += f"**Sample Size Adequate**: {'Yes' if alignment.get('sample_size_adequate') else 'No'}\n\n"
        
        if alignment.get('missing_elements'):
            section += "### Missing Elements\n\n"
            for element in alignment['missing_elements']:
                section += f"- {element}\n"
        
        if alignment.get('recommendations'):
            section += "\n### Recommendations\n\n"
            for rec in alignment['recommendations']:
                section += f"- {rec}\n"
        
        return section
    
    def _generate_transformations_section(self, transformations: Dict) -> str:
        """Generate transformations section."""
        
        section = "## Transformations Applied\n\n"
        section += f"**Rows**: {transformations.get('rows_before', 'N/A')} → {transformations.get('rows_after', 'N/A')}\n"
        section += f"**Columns**: {transformations.get('columns_before', 'N/A')} → {transformations.get('columns_after', 'N/A')}\n\n"
        
        if transformations.get('columns_removed'):
            section += f"**Columns Removed**: {', '.join(transformations['columns_removed'])}\n\n"
        
        if transformations.get('transformation_log'):
            section += "### Transformation Log\n\n"
            for log_entry in transformations['transformation_log']:
                section += f"- {log_entry}\n"
        
        if transformations.get('transformations_applied'):
            section += "\n### Python Code\n\n```python\n"
            for t in transformations['transformations_applied'][:5]:
                if isinstance(t, dict):
                    section += f"# {t.get('description', '')}\n"
                    section += f"{t.get('python_code', '')}\n\n"
            section += "```\n"
        
        return section
    
    def _generate_recommendations(
        self,
        quality: Dict,
        privacy: Dict,
        alignment: Dict
    ) -> List[str]:
        """Generate recommendations."""
        
        recommendations = []
        
        # Quality-based recommendations
        if quality.get('overall_score', 100) < 80:
            recommendations.append("Address data quality issues before analysis")
        
        # Privacy-based recommendations
        if privacy.get('risk_level') == 'High':
            recommendations.append("CRITICAL: Remove or anonymize high-risk PII before sharing")
        elif privacy.get('risk_level') == 'Medium':
            recommendations.append("Review and address medium-risk privacy concerns")
        
        # Alignment-based recommendations
        if alignment.get('alignment_score', 100) < 70:
            recommendations.append("Consider supplementing data to improve research alignment")
        
        if alignment.get('missing_elements'):
            recommendations.append(f"Consider adding: {', '.join(alignment['missing_elements'][:3])}")
        
        if not recommendations:
            recommendations.append("Dataset is ready for analysis")
        
        return recommendations
    
    def _identify_limitations(self, cleaning_output: Dict) -> List[str]:
        """Identify dataset limitations."""
        
        limitations = []
        
        quality = cleaning_output.get('quality_assessment', {})
        privacy = cleaning_output.get('privacy_screening', {})
        transformations = cleaning_output.get('transformations', {})
        
        if quality.get('overall_score', 100) < 80:
            limitations.append("Data quality score below optimal threshold")
        
        if privacy.get('pii_detected'):
            limitations.append("Dataset contained PII that was removed/anonymized")
        
        if transformations.get('columns_removed'):
            limitations.append(f"Columns removed during processing: {', '.join(transformations['columns_removed'])}")
        
        limitations.append("User-uploaded data may have undocumented collection methodology")

        return limitations


class Archivist:
    """Replication-assurance agent. Deterministic (no LLM): consolidates the
    audit trail already produced by Stage 1/2/3 into one manifest, and adds
    what nothing upstream captures — the environment the pipeline ran in and
    a content fingerprint of the exact pipeline code that ran — so a dataset
    can be reproduced independent of the run that first produced it."""

    ARCHIVIST_VERSION = "1.0.0"
    KEY_PACKAGES = ["pandas", "numpy", "pydantic", "requests", "python-dotenv", "openai"]
    STAGE_FILES = ["1-DataSourceStage.py", "2-DataCleaningStage.py", "3-QualityAssuranceStage.py"]

    def __init__(self, collector: Optional[MetricsCollector] = None):
        self.agent_name = "Archivist"
        self.collector = collector
        self._guard = StateGuard()

    def _capture_environment(self) -> Dict:
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
        """Consolidate the file/privacy/transformation provenance already
        tracked across Stage 1 (upload intake), Stage 2 (privacy screening
        and cleaning), and Stage 3 (QA/certification) into one readable
        trail."""
        trail = [f"Original file: {doc.file_path}"]
        data_structure = getattr(doc.codebook, "data_structure", "") if doc.codebook else ""
        if data_structure:
            trail.append(f"Inferred structure: {data_structure}")
        trail.append(f"Certified {doc.certification_level} (quality score {doc.quality_score:.1f}/100)")
        trail.append(f"Ready for analysis: {doc.ready_for_analysis}")
        sandbox = (doc.documentation or {}).get("sandbox_validation", {})
        if sandbox:
            trail.append(f"Sandbox validation: {'passed' if sandbox.get('validated') else 'failed'}")
        return trail

    def build_manifest(
        self,
        documented_dataset: "DocumentedDataset",
        data_cleaning_output: Dict,
        stage_dir: Path,
    ) -> Dict:
        """Build the replication manifest for this run."""
        print(f"\n[{self.agent_name}] Building replication manifest...")

        environment = self._capture_environment()
        pipeline_fingerprint = self._fingerprint_pipeline_code(stage_dir)
        cleaning_output_hash = self._guard.sign_output(
            data_cleaning_output, stage_name="DataCleaningStage_input_to_QA")
        content_hash = self._guard.sign_output(
            documented_dataset.codebook.model_dump() if documented_dataset.codebook else {},
            stage_name=f"dataset:{documented_dataset.dataset_id}")

        manifest = {
            "archivist_version": self.ARCHIVIST_VERSION,
            "generated_at": datetime.now().isoformat(),
            "environment": environment,
            "pipeline_code_fingerprint": pipeline_fingerprint,
            "upstream_cleaning_output_hash": cleaning_output_hash,
            "datasets": [{
                "dataset_id": documented_dataset.dataset_id,
                "dataset_name": documented_dataset.dataset_name,
                "content_hash": content_hash,
                "certification_level": documented_dataset.certification_level,
                "quality_score": documented_dataset.quality_score,
                "audit_trail": self._build_audit_trail(documented_dataset),
            }],
        }

        print(f"  Pipeline fingerprint: {pipeline_fingerprint['pipeline_fingerprint']}")
        return manifest


# ============================================================================
# HITL Checkpoint
# ============================================================================

def checkpoint5_final_approval(documented_dataset: DocumentedDataset) -> str:
    """HITL Checkpoint 5: Final Approval."""
    
    print("\n" + "="*70)
    print("🛑 HITL CHECKPOINT 5: Final Approval")
    print("="*70)
    print(f"\nDataset: {documented_dataset.dataset_name}")
    print(f"Quality Score: {documented_dataset.quality_score:.1f}/100")
    print(f"Certification: {documented_dataset.certification_level}")
    print(f"Ready for Analysis: {documented_dataset.ready_for_analysis}")
    
    print(f"\nCodebook: {len(documented_dataset.codebook.variables)} variables documented")
    print(f"Report Sections: File, Quality, Privacy, Structure, Alignment, Transformations")
    
    if documented_dataset.report.recommendations:
        print("\nRecommendations:")
        for rec in documented_dataset.report.recommendations[:5]:
            print(f"  - {rec}")
    
    print("\nFinal decision:")
    print("- Is the documentation complete and accurate?")
    print("- Is the dataset ready for analysis?")
    print("- Should any sections be revised?")

    response = auto_input("\n> ", default=get_default("approve_stage")).strip()
    return response if response else "approved"


# ============================================================================
# Orchestrator
# ============================================================================

class QualityAssuranceOrchestrator:
    """Orchestrator for the quality assurance stage."""

    def __init__(self, openai_api_key: Optional[str] = None, collector: Optional[MetricsCollector] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        self.validation_agent = FinalValidationAgent(self.api_key)
        self.documentation_agent = DocumentationAgent(self.api_key, collector=collector)
        self.archivist = Archivist(collector=collector)
        self.qa_output: Optional[QualityAssuranceOutput] = None
        self.report_content: str = ""
        self.replication_manifest: Optional[Dict] = None
    
    def run_qa_pipeline(
        self,
        data_cleaning_output: Dict,
        research_question: str = "",
        file_path: str = "",
        enable_hitl: bool = True
    ) -> QualityAssuranceOutput:
        """Run the complete quality assurance pipeline."""
        
        print(f"\n{'='*70}")
        print(f"REPORT & CODEBOOK GENERATION PIPELINE")
        print(f"{'='*70}")
        
        # Get file path and research question
        file_path = data_cleaning_output.get('file_path', file_path)
        research_q = data_cleaning_output.get('research_question', research_question)
        
        print(f"File: {os.path.basename(file_path)}")
        print(f"{'='*70}\n")
        
        # Step 1: Final Validation
        print(f"STEP 1: FINAL VALIDATION")
        print(f"-"*70)
        quality_score, certification, ready_for_analysis = self.validation_agent.validate_dataset(
            data_cleaning_output
        )
        
        # Step 2: Generate Codebook
        print(f"\nSTEP 2: GENERATE CODEBOOK")
        print(f"-"*70)
        codebook = self.documentation_agent.generate_codebook(
            data_cleaning_output,
            file_path
        )
        
        # Step 3: Generate Report
        print(f"\nSTEP 3: GENERATE REPORT")
        print(f"-"*70)
        report = self.documentation_agent.generate_report(
            data_cleaning_output,
            file_path,
            research_q
        )
        
        # Step 4: Innovation Analysis
        print(f"\nSTEP 4: INNOVATION ANALYSIS")
        print(f"-"*70)
        # file_info is propagated via metadata by Stage 2 (no file_info field on the
        # cleaning schema), so check both locations to avoid empty columns.
        file_info = data_cleaning_output.get("file_info") or \
            data_cleaning_output.get("metadata", {}).get("file_info", {}) or {}
        innovation_analysis = self.documentation_agent.generate_innovation_analysis(
            research_question=research_q,
            data_summary={
                "columns": file_info.get("column_names", []),
                "num_rows": file_info.get("num_rows", 0),
                "file_type": file_info.get("file_extension", "CSV")
            }
        )

        # Step 5: Sandbox Data Validation
        print(f"\nSTEP 5: SANDBOX DATA VALIDATION")
        print(f"-"*70)
        sandbox_result = self._validate_data_with_sandbox(
            data_cleaning_output
        )

        # Gate certification on validation: a dataset that did not pass the
        # (now deterministic) validation must NOT be certified Silver/Gold. This
        # closes the "validated=false yet cert=Silver/75" gap.
        sandbox_validated = bool(sandbox_result.get("validated", False))
        if not sandbox_validated:
            if certification in ("Gold", "Silver"):
                certification = "Bronze"
            quality_score = min(quality_score, 60.0)
            ready_for_analysis = False
            print(f"  [VALIDATION GATE] Validation did NOT pass — capping certification to "
                  f"'{certification}' (score {quality_score:.1f}), ready_for_analysis=False.")

        # Create documented dataset
        documented_dataset = DocumentedDataset(
            dataset_id=f"USER_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            dataset_name=os.path.basename(file_path),
            file_path=file_path,
            codebook=codebook,
            report=report,
            quality_score=quality_score,
            certification_level=certification,
            ready_for_analysis=ready_for_analysis,
            documentation={
                "generated_date": datetime.now().isoformat(),
                "pipeline_version": "1.0.0",
                "innovation_suggestions": innovation_analysis,
                "sandbox_validation": sandbox_result
            }
        )
        
        # HITL Checkpoint 5: Final Approval
        if enable_hitl:
            feedback = checkpoint5_final_approval(documented_dataset)
            print(f"[HITL] Final approval feedback: {feedback}")

        # Step 6: Archivist builds the replication manifest for this run
        print(f"\nSTEP 6: REPLICATION MANIFEST")
        stage_dir = Path(__file__).resolve().parent
        self.replication_manifest = self.archivist.build_manifest(
            documented_dataset, data_cleaning_output, stage_dir
        )

        # Create output
        self.qa_output = QualityAssuranceOutput(
            file_path=file_path,
            research_question=research_q,
            documented_dataset=documented_dataset,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "pipeline_stage": "quality_assurance",
                "hitl_enabled": enable_hitl,
                "innovation_suggestions": innovation_analysis
            },
            replication_manifest=self.replication_manifest,
        )
        
        # Store report content for saving
        self.report_content = self._compile_full_report(documented_dataset, research_q)
        
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Quality Score: {quality_score:.1f}/100")
        print(f"Certification: {certification}")
        print(f"Ready for Analysis: {ready_for_analysis}")
        print(f"{'='*70}\n")
        
        return self.qa_output

    def _validate_data_with_sandbox(
        self,
        data_cleaning_output: Dict
    ) -> Dict:
        """Validate the dataset deterministically in the sandbox.

        Previously this asked the LLM to GENERATE the validation code, which often
        emitted ``open()``/``os``/``sys`` — all blocked by the sandbox — so it
        returned ``validated=False`` with empty stdout on every run regardless of
        data quality. Here we author a fixed, sandbox-safe script (only ``json``,
        with the dataset facts INLINED so no file I/O is needed). The result is
        therefore deterministic and reflects the actual checks. If the sandbox
        itself cannot run, we fall back to the identical checks in-process so
        ``validated`` is still meaningful.
        """
        # file_info is propagated via metadata by Stage 2 (the cleaning schema has
        # no file_info field), so check both locations — the direct read alone was
        # always empty, which is part of why validation never passed.
        file_info = data_cleaning_output.get("file_info") or \
            data_cleaning_output.get("metadata", {}).get("file_info", {}) or {}
        columns = list(file_info.get("column_names", []) or [])
        try:
            num_rows = int(file_info.get("num_rows", 0) or 0)
        except (TypeError, ValueError):
            num_rows = 0
        file_type = file_info.get("file_extension", "CSV")

        # Deterministic, in-process checks (used both to author expectations and as
        # the fallback when the sandbox subprocess is unavailable).
        def _run_checks():
            column_checks = []
            issues = []
            for col in columns:
                valid = isinstance(col, str) and col == col.strip() and " " not in col
                column_checks.append({"column": col, "valid_name": bool(valid)})
                if not valid:
                    issues.append(f"Column name not clean: {col!r}")
            if not columns:
                issues.append("No columns found in file_info")
            if num_rows <= 0:
                issues.append("Row count is not positive")
            validated = bool(columns) and num_rows > 0
            return validated, column_checks, issues

        # Sandbox-safe authored script: only the `json` module (whitelisted) and
        # inlined data — no open()/os/sys, so it always passes the safety check.
        code = (
            "import json\n"
            f"columns = {json.dumps(columns)}\n"
            f"num_rows = {json.dumps(num_rows)}\n"
            f"file_type = {json.dumps(file_type)}\n"
            "column_checks = []\n"
            "issues = []\n"
            "for col in columns:\n"
            "    valid = isinstance(col, str) and col == col.strip() and ' ' not in col\n"
            "    column_checks.append({'column': col, 'valid_name': bool(valid)})\n"
            "    if not valid:\n"
            "        issues.append('Column name not clean: ' + repr(col))\n"
            "if not columns:\n"
            "    issues.append('No columns found in file_info')\n"
            "if num_rows <= 0:\n"
            "    issues.append('Row count is not positive')\n"
            "validated = bool(columns) and num_rows > 0\n"
            "print(json.dumps({\n"
            "    'validated': validated,\n"
            "    'column_checks': column_checks,\n"
            "    'issues': issues,\n"
            "    'summary': str(len(columns)) + ' columns, ' + str(num_rows) + ' rows; ' + str(len(issues)) + ' issue(s)'\n"
            "}))\n"
        )

        try:
            sandbox = CodeSandbox(timeout_sec=60, memory_mb=256)
            exec_result = sandbox.execute(code)

            if exec_result.success and exec_result.stdout.strip():
                try:
                    parsed = json.loads(exec_result.stdout.strip().splitlines()[-1])
                except (json.JSONDecodeError, IndexError):
                    parsed = {}
                validated = bool(parsed.get("validated", False))
                validation = {
                    "validated": validated,
                    "validation_method": "sandbox_execution",
                    "code": code,
                    "stdout": exec_result.stdout,
                    "stderr": exec_result.stderr,
                    "error": exec_result.error,
                    "column_checks": parsed.get("column_checks", []),
                    "issues": parsed.get("issues", []),
                    "execution_time_sec": exec_result.execution_time_sec
                }
                print(f"  Sandbox validation {'PASSED' if validated else 'FAILED checks'} "
                      f"({exec_result.execution_time_sec:.1f}s)")
                return validation

            # Sandbox ran but produced no usable output — fall back in-process.
            print(f"  Sandbox unavailable ({exec_result.error or exec_result.stderr[:120]}); "
                  f"using in-process deterministic validation")
        except Exception as e:
            print(f"  Sandbox validation error: {e}; using in-process deterministic validation")

        validated, column_checks, issues = _run_checks()
        return {
            "validated": validated,
            "validation_method": "in_process_deterministic",
            "column_checks": column_checks,
            "issues": issues,
            "summary": f"{len(columns)} columns, {num_rows} rows; {len(issues)} issue(s)"
        }

    def _compile_full_report(self, documented: DocumentedDataset, research_question: str) -> str:
        """Compile the full report as markdown."""
        
        report = documented.report
        codebook = documented.codebook
        
        content = f"""# {report.title}

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**File**: {report.file_path}

**Research Question**: {research_question}

---

## Executive Summary

{report.executive_summary}

---

{report.file_characteristics_section}

---

{report.quality_assessment_section}

---

{report.privacy_screening_section}

---

{report.structure_analysis_section}

---

{report.research_alignment_section}

---

{report.transformations_section}

---

## Data Codebook

### Dataset Information
- **Title**: {codebook.title}
- **Version**: {codebook.version}
- **Created**: {codebook.created_date}
- **Data Structure**: {codebook.data_structure}

### File Information
- **File Name**: {codebook.file_info.get('file_name', 'N/A')}
- **Rows**: {codebook.file_info.get('num_rows', 'N/A'):,}
- **Columns**: {codebook.file_info.get('num_columns', 'N/A')}

### Variable Descriptions

| Variable | Type | Role | Description |
|----------|------|------|-------------|
"""
        
        for var in codebook.variables:
            content += f"| {var.variable_name} | {var.data_type} | {var.role} | {var.description} |\n"
        
        content += f"""
### Summary Statistics
- **Total Variables**: {codebook.summary_statistics.get('total_variables', 'N/A')}
- **Numeric Variables**: {codebook.summary_statistics.get('numeric_variables', 'N/A')}
- **Categorical Variables**: {codebook.summary_statistics.get('categorical_variables', 'N/A')}

### Methodology

{codebook.methodology}

### Methodology Narrative

{codebook.methodology_narrative or 'Not available.'}

### Known Limitations

"""
        for limitation in codebook.limitations:
            content += f"- {limitation}\n"
        
        content += f"""
### Usage Notes

{codebook.usage_notes}

---

## Recommendations

"""
        for rec in report.recommendations:
            content += f"- {rec}\n"
        
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

        content += f"""
---

## Certification

**Quality Score**: {documented.quality_score:.1f}/100

**Certification Level**: {documented.certification_level}

**Ready for Analysis**: {'Yes' if documented.ready_for_analysis else 'No'}

---

*Report generated by DataTeam User-Uploaded Data Pipeline*
"""
        
        return content
    
    def save_qa_output(self, filename: str = "user_data_qa_output.json"):
        """Save QA output to JSON."""
        if not self.qa_output:
            print("No QA output to save")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.qa_output.model_dump(), f, indent=2)
        
        print(f"[Orchestrator] QA output saved to {filename}")
    
    def save_report_codebook(self, filename: str = "user_data_report.md"):
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
    cleaning_file = "user_data_cleaning_output.json"
    if os.path.exists(cleaning_file):
        with open(cleaning_file, 'r', encoding='utf-8') as f:
            cleaning_data = json.load(f)
    else:
        # Create sample data for testing
        cleaning_data = {
            "file_path": "example_data.csv",
            "research_question": "Analyze the relationship between education and income",
            "file_info": {
                "file_name": "example_data.csv",
                "file_size_mb": 0.5,
                "num_rows": 100,
                "num_columns": 6,
                "column_names": ["id", "name", "age", "income", "education_years", "employed"],
                "column_types": {"id": "int64", "name": "object", "age": "int64"}
            },
            "quality_assessment": {
                "overall_score": 85,
                "grade": "B",
                "dimensions": [
                    {"dimension": "completeness", "score": 90, "issues": []},
                    {"dimension": "accuracy", "score": 85, "issues": []}
                ],
                "actionable_issues": []
            },
            "privacy_screening": {
                "risk_level": "Medium",
                "pii_detected": True,
                "pii_detections": [
                    {"column_name": "name", "pii_type": "name", "confidence": 0.8, "recommendation": "Remove"}
                ],
                "sensitive_columns": ["name"],
                "remediation_actions": ["name: Remove or anonymize"]
            },
            "structure_inference": {
                "structure_type": "cross_sectional",
                "confidence": 0.8,
                "key_variables": {"identifier": ["id"]},
                "suggested_roles": {"income": "outcome", "education_years": "treatment"}
            },
            "research_alignment": {
                "alignment_score": 80,
                "suitability": "Medium",
                "sample_size_adequate": True,
                "missing_elements": [],
                "recommendations": []
            },
            "transformations": {
                "transformations_applied": [
                    {"description": "Remove PII column: name", "python_code": "df = df.drop(columns=['name'])"}
                ],
                "rows_before": 100,
                "rows_after": 100,
                "columns_before": 6,
                "columns_after": 5,
                "columns_removed": ["name"],
                "transformation_log": ["Removed PII column: name"]
            }
        }
    
    # Run pipeline
    orchestrator = QualityAssuranceOrchestrator()
    qa_output = orchestrator.run_qa_pipeline(
        data_cleaning_output=cleaning_data,
        enable_hitl=True
    )
    
    # Save outputs
    orchestrator.save_qa_output("user_data_qa_output.json")
    orchestrator.save_report_codebook(f"user_data_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.md")
    orchestrator.save_replication_manifest("replication_manifest.json")
    
    print("\n" + "="*70)
    print("QUALITY ASSURANCE SUMMARY")
    print("="*70)
    print(f"Quality Score: {qa_output.documented_dataset.quality_score:.1f}")
    print(f"Certification: {qa_output.documented_dataset.certification_level}")
    print(f"Ready for Analysis: {qa_output.documented_dataset.ready_for_analysis}")
    print("="*70)


if __name__ == "__main__":
    main()
