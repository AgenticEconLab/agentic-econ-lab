# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Data Source Stage - User-Uploaded Data Mode

This script handles:
- File detection and parsing (CSV, Excel, Stata, SPSS, SAS)
- Quality assessment (completeness, accuracy, consistency, structure)

Pipeline:
1. FileParserAgent: Detect file format and parse uploaded file
2. DataValidationAgent: Assess data quality across multiple dimensions

HITL Checkpoint (1):
1. Quality Review (after quality assessment)

Input: User-uploaded file path
Output: Parsed file info and quality assessment
"""

import os
import json
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
# Add parent directories to path for shared imports
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
agents_dir = current_dir.parent.parent.parent  # repository root
if str(agents_dir) not in sys.path:
    sys.path.insert(0, str(agents_dir))

from shared.auto_input import auto_input, get_default
from shared.llm import LLMClient
from shared.observability import MetricsCollector, tracked_file_parse
from DataTeam.ael.schemas.stage_outputs import (
    FileInfo,
    UserUploadedQualityDimension as QualityDimension,
    UserUploadedQualityAssessment as QualityAssessment,
    UserUploadedDataRequirement as DataRequirement,
    UserUploadedDataSourceOutput as DataSourceOutput,
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

class FileParserAgent:
    """Agent for detecting file format and parsing uploaded files."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "FileParserAgent"
        self.api_key = openai_api_key
        self.collector = collector

    def parse_file(self, file_path: str) -> FileInfo:
        """Detect file format and parse the uploaded file."""

        print(f"\n[{self.agent_name}] Parsing file: {file_path}")

        # Get file info
        file_name = os.path.basename(file_path)
        file_extension = os.path.splitext(file_path)[1].lower()
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        print(f"  File: {file_name}")
        print(f"  Extension: {file_extension}")
        print(f"  Size: {file_size_mb:.2f} MB")

        # Parse based on extension using tracked wrapper
        format_names = {
            '.csv': 'CSV', '.xlsx': 'Excel', '.xls': 'Excel', '.dta': 'Stata',
            '.sav': 'SPSS', '.sas7bdat': 'SAS', '.json': 'JSON', '.parquet': 'Parquet',
        }
        df = None
        parsing_notes = ""
        format_name = format_names.get(file_extension, f"Unknown ({file_extension})")

        try:
            df = tracked_file_parse(
                file_path,
                collector=self.collector,
                agent=self.agent_name,
            )
            if file_extension in format_names:
                parsing_notes = f"{format_name} file parsed successfully"
            else:
                print(f"  [WARNING] Unknown format, attempted CSV parsing...")
                parsing_notes = f"Unknown format ({file_extension}), parsed as CSV"

            print(f"  ✓ {parsing_notes}")

        except Exception as e:
            print(f"  [ERROR] Parsing failed: {e}")
            parsing_notes = f"Parsing failed: {str(e)}"
            # Create empty DataFrame
            df = pd.DataFrame()
        
        # Extract file info
        if df is not None and len(df) > 0:
            column_names = list(df.columns)
            column_types = {col: str(df[col].dtype) for col in df.columns}
            
            # Create preview (first 5 rows)
            data_preview = df.head(5).to_dict(orient='records')
            
            # Convert any non-serializable types
            for row in data_preview:
                for key, value in row.items():
                    if pd.isna(value):
                        row[key] = None
                    elif isinstance(value, (np.integer, np.floating)):
                        row[key] = float(value)
            
            num_rows = len(df)
            num_columns = len(df.columns)
        else:
            column_names = []
            column_types = {}
            data_preview = []
            num_rows = 0
            num_columns = 0
        
        print(f"  Rows: {num_rows}")
        print(f"  Columns: {num_columns}")
        
        # Store DataFrame for later use
        self._df = df
        
        return FileInfo(
            file_path=file_path,
            file_name=file_name,
            file_extension=file_extension,
            file_size_mb=round(file_size_mb, 2),
            num_rows=num_rows,
            num_columns=num_columns,
            column_names=column_names,
            column_types=column_types,
            data_preview=data_preview,
            parsing_notes=parsing_notes
        )
    
    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the parsed DataFrame."""
        return getattr(self, '_df', None)


class DataValidationAgent:
    """Agent for assessing data quality across multiple dimensions."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "DataValidationAgent"
        self.api_key = openai_api_key
        self.llm = LLMClient(
            temperature=0.3,
            api_key=self.api_key,
            collector=collector,
            agent_name=self.agent_name
        )
    
    def assess_quality(
        self,
        file_info: FileInfo,
        df: Optional[pd.DataFrame] = None
    ) -> QualityAssessment:
        """Assess data quality across completeness, accuracy, consistency, structure, and semantic correctness."""

        print(f"\n[{self.agent_name}] Assessing data quality...")

        dimensions = []

        # 1. Completeness Assessment
        print("  Assessing completeness...")
        completeness = self._assess_completeness(file_info, df)
        dimensions.append(completeness)

        # 2. Accuracy Assessment (outliers)
        print("  Assessing accuracy...")
        accuracy = self._assess_accuracy(file_info, df)
        dimensions.append(accuracy)

        # 3. Consistency Assessment
        print("  Assessing consistency...")
        consistency = self._assess_consistency(file_info, df)
        dimensions.append(consistency)

        # 4. Structure Assessment
        print("  Assessing structure...")
        structure = self._assess_structure(file_info, df)
        dimensions.append(structure)

        # 5. Semantic Correctness Assessment (LLM-based)
        print("  Assessing semantic correctness...")
        semantic = self._assess_semantic_correctness(file_info, df)
        dimensions.append(semantic)
        
        # Calculate overall score
        overall_score = sum(d.score for d in dimensions) / len(dimensions)
        
        # Determine grade
        if overall_score >= 90:
            grade = "A"
        elif overall_score >= 80:
            grade = "B"
        elif overall_score >= 70:
            grade = "C"
        elif overall_score >= 60:
            grade = "D"
        else:
            grade = "F"
        
        # Compile summaries
        missing_data_summary = completeness.details.get("missing_summary", {})
        outlier_summary = accuracy.details.get("outlier_summary", {})
        duplicate_summary = structure.details.get("duplicate_summary", {})
        
        # Collect actionable issues
        actionable_issues = []
        for dim in dimensions:
            actionable_issues.extend(dim.issues[:2])  # Top 2 issues per dimension
        
        print(f"  Overall Score: {overall_score:.1f} ({grade})")
        
        return QualityAssessment(
            overall_score=round(overall_score, 1),
            grade=grade,
            dimensions=dimensions,
            missing_data_summary=missing_data_summary,
            outlier_summary=outlier_summary,
            duplicate_summary=duplicate_summary,
            actionable_issues=actionable_issues
        )
    
    def _assess_completeness(
        self,
        file_info: FileInfo,
        df: Optional[pd.DataFrame]
    ) -> QualityDimension:
        """Assess data completeness (missing values)."""
        
        issues = []
        recommendations = []
        details = {}
        
        if df is not None and len(df) > 0:
            # Calculate missing values
            missing_counts = df.isnull().sum()
            missing_pcts = (missing_counts / len(df) * 100).round(2)
            
            total_missing = missing_counts.sum()
            total_cells = df.size
            overall_missing_pct = (total_missing / total_cells * 100)
            
            # Score based on missing percentage
            score = max(0, 100 - overall_missing_pct * 2)
            
            # Identify columns with high missing rates
            high_missing_cols = missing_pcts[missing_pcts > 10].to_dict()
            
            if high_missing_cols:
                issues.append(f"High missing rates in: {', '.join(high_missing_cols.keys())}")
                recommendations.append("Consider imputation or removal for columns with >10% missing")
            
            details = {
                "missing_summary": {
                    "total_missing_cells": int(total_missing),
                    "overall_missing_pct": round(overall_missing_pct, 2),
                    "columns_with_missing": {k: float(v) for k, v in missing_pcts[missing_pcts > 0].to_dict().items()}
                }
            }
        else:
            score = 0
            issues.append("Unable to assess completeness - no data available")
        
        return QualityDimension(
            dimension="completeness",
            score=round(score, 1),
            issues=issues,
            recommendations=recommendations,
            details=details
        )
    
    def _assess_accuracy(
        self,
        file_info: FileInfo,
        df: Optional[pd.DataFrame]
    ) -> QualityDimension:
        """Assess data accuracy (outliers)."""
        
        issues = []
        recommendations = []
        details = {}
        
        if df is not None and len(df) > 0:
            # Check numeric columns for outliers using IQR method
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outlier_counts = {}
            
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers > 0:
                    outlier_counts[col] = int(outliers)
            
            total_outliers = sum(outlier_counts.values())
            outlier_pct = (total_outliers / (len(df) * len(numeric_cols)) * 100) if len(numeric_cols) > 0 else 0
            
            # Score based on outlier percentage
            score = max(0, 100 - outlier_pct * 5)
            
            if outlier_counts:
                issues.append(f"Outliers detected in {len(outlier_counts)} columns")
                recommendations.append("Review outliers and consider winsorization or removal")
            
            details = {
                "outlier_summary": {
                    "total_outliers": total_outliers,
                    "outlier_pct": round(outlier_pct, 2),
                    "columns_with_outliers": outlier_counts
                }
            }
        else:
            score = 0
            issues.append("Unable to assess accuracy - no data available")
        
        return QualityDimension(
            dimension="accuracy",
            score=round(score, 1),
            issues=issues,
            recommendations=recommendations,
            details=details
        )
    
    def _assess_consistency(
        self,
        file_info: FileInfo,
        df: Optional[pd.DataFrame]
    ) -> QualityDimension:
        """Assess data consistency."""
        
        issues = []
        recommendations = []
        details = {}
        score = 85  # Default score
        
        if df is not None and len(df) > 0:
            # Check for mixed types in columns
            mixed_type_cols = []
            for col in df.columns:
                if df[col].apply(type).nunique() > 1:
                    mixed_type_cols.append(col)
            
            if mixed_type_cols:
                issues.append(f"Mixed data types in: {', '.join(mixed_type_cols[:5])}")
                recommendations.append("Standardize data types for affected columns")
                score -= len(mixed_type_cols) * 2
            
            # Check for inconsistent string formats (e.g., case variations)
            string_cols = df.select_dtypes(include=['object']).columns
            for col in string_cols[:5]:  # Check first 5 string columns
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) > 0:
                    # Check for case variations
                    lower_vals = set(str(v).lower() for v in unique_vals)
                    if len(lower_vals) < len(unique_vals):
                        issues.append(f"Possible case inconsistencies in '{col}'")
                        score -= 3
            
            details = {
                "mixed_type_columns": mixed_type_cols,
                "string_columns_checked": list(string_cols[:5])
            }
        else:
            score = 0
            issues.append("Unable to assess consistency - no data available")
        
        return QualityDimension(
            dimension="consistency",
            score=max(0, round(score, 1)),
            issues=issues,
            recommendations=recommendations,
            details=details
        )
    
    def _assess_structure(
        self,
        file_info: FileInfo,
        df: Optional[pd.DataFrame]
    ) -> QualityDimension:
        """Assess data structure (duplicates, data types)."""
        
        issues = []
        recommendations = []
        details = {}
        score = 90  # Default score
        
        if df is not None and len(df) > 0:
            # Check for duplicate rows
            duplicate_count = df.duplicated().sum()
            duplicate_pct = (duplicate_count / len(df) * 100)
            
            if duplicate_count > 0:
                issues.append(f"{duplicate_count} duplicate rows ({duplicate_pct:.1f}%)")
                recommendations.append("Review and remove duplicate rows")
                score -= min(20, duplicate_pct * 2)
            
            # Check for constant columns
            constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
            if constant_cols:
                issues.append(f"Constant columns: {', '.join(constant_cols[:5])}")
                recommendations.append("Consider removing constant columns")
                score -= len(constant_cols) * 2
            
            details = {
                "duplicate_summary": {
                    "duplicate_rows": int(duplicate_count),
                    "duplicate_pct": round(duplicate_pct, 2)
                },
                "constant_columns": constant_cols
            }
        else:
            score = 0
            issues.append("Unable to assess structure - no data available")
        
        return QualityDimension(
            dimension="structure",
            score=max(0, round(score, 1)),
            issues=issues,
            recommendations=recommendations,
            details=details
        )


    def _assess_semantic_correctness(
        self,
        file_info: FileInfo,
        df: Optional[pd.DataFrame]
    ) -> QualityDimension:
        """Assess semantic correctness using LLM to check column meanings, units, and plausible value ranges."""

        issues = []
        recommendations = []
        details = {}
        score = 85  # Default score

        if df is not None and len(df) > 0:
            # Build a summary of columns with statistics
            col_summaries = []
            for col in df.columns[:15]:  # Limit to first 15 columns
                summary = {"name": col, "dtype": str(df[col].dtype)}
                if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    summary["min"] = float(df[col].min()) if not pd.isna(df[col].min()) else None
                    summary["max"] = float(df[col].max()) if not pd.isna(df[col].max()) else None
                    summary["mean"] = float(df[col].mean()) if not pd.isna(df[col].mean()) else None
                elif df[col].dtype == 'object':
                    unique_vals = df[col].dropna().unique()
                    summary["unique_count"] = len(unique_vals)
                    summary["sample_values"] = [str(v) for v in unique_vals[:5]]
                col_summaries.append(summary)

            system_prompt = "You are a data quality expert specializing in economic and financial datasets. Assess semantic correctness."
            user_prompt = """Analyze these column statistics from an uploaded dataset and identify potential CORRECTNESS issues.

File: {file_name} ({num_rows} rows, {num_cols} columns)

Column summaries:
{col_summaries}

Check for:
1. UNIT PLAUSIBILITY: Are numeric ranges plausible for what the column name implies? (e.g., an "age" column with max=500 is suspicious, a "GDP" column in billions should not have values < 1)
2. SEMANTIC MISMATCHES: Do column names match their apparent content? (e.g., a column named "date" containing numeric values)
3. ECONOMIC VALIDITY: For economic/financial data, are values in expected ranges? (e.g., unemployment rate should be 0-100%, interest rates should not be negative unless intentional)
4. MISSING CONTEXT: Are there columns that need units/metadata to be interpretable?

Return as JSON:
{{{{
  "issues_found": [
    {{{{"column": "col_name", "issue": "description", "severity": "high/medium/low"}}}}
  ],
  "recommendations": ["recommendation1", "recommendation2"],
  "columns_needing_documentation": ["col1", "col2"],
  "overall_assessment": "brief overall assessment"
}}}}

Respond with ONLY the JSON object.
"""

            try:
                result = self.llm.format_and_invoke(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    variables={
                        "file_name": file_info.file_name,
                        "num_rows": str(file_info.num_rows),
                        "num_cols": str(file_info.num_columns),
                        "col_summaries": json.dumps(col_summaries, indent=2, default=str)
                    }
                )

                analysis = _extract_json(result)
                issues_found = analysis.get("issues_found", [])
                recs = analysis.get("recommendations", [])
                cols_needing_docs = analysis.get("columns_needing_documentation", [])

                for issue in issues_found:
                    severity = issue.get("severity", "medium")
                    issues.append(f"[{severity.upper()}] {issue.get('column', '?')}: {issue.get('issue', 'unknown issue')}")
                    if severity == "high":
                        score -= 10
                    elif severity == "medium":
                        score -= 5
                    else:
                        score -= 2

                recommendations.extend(recs[:3])
                if cols_needing_docs:
                    recommendations.append(f"Add documentation/units for: {', '.join(cols_needing_docs[:5])}")

                details = {
                    "semantic_issues": issues_found,
                    "columns_needing_documentation": cols_needing_docs,
                    "overall_assessment": analysis.get("overall_assessment", "")
                }

            except Exception as e:
                print(f"    Semantic correctness check error: {e}")
                issues.append("Semantic correctness check could not be completed")
                details = {"error": str(e)}
        else:
            score = 0
            issues.append("Unable to assess semantic correctness - no data available")

        return QualityDimension(
            dimension="semantic_correctness",
            score=max(0, round(score, 1)),
            issues=issues,
            recommendations=recommendations,
            details=details
        )


# ============================================================================
# HITL Checkpoint
# ============================================================================

def checkpoint1_quality_review(quality_assessment: QualityAssessment) -> str:
    """HITL Checkpoint 1: Quality Review."""
    
    print("\n" + "="*70)
    print("🛑 HITL CHECKPOINT 1: Quality Review")
    print("="*70)
    print(f"\nOverall Quality Score: {quality_assessment.overall_score}/100 ({quality_assessment.grade})")
    
    print("\nQuality Dimensions:")
    for dim in quality_assessment.dimensions:
        print(f"  - {dim.dimension.title()}: {dim.score}/100")
        if dim.issues:
            for issue in dim.issues[:2]:
                print(f"      Issue: {issue}")
    
    if quality_assessment.actionable_issues:
        print("\nActionable Issues:")
        for issue in quality_assessment.actionable_issues[:5]:
            print(f"  - {issue}")
    
    print("\nDecision needed:")
    print("- Is the data quality acceptable for your research purpose?")
    print("- Should we proceed with this dataset?")
    print("- Are there specific quality issues that need addressing first?")

    response = auto_input("\n> ", default=get_default("approve_stage")).strip()
    return response if response else "approved"


# ============================================================================
# Orchestrator
# ============================================================================

def _extract_json(text: str):
    """Extract JSON from text that may contain markdown code blocks."""
    stripped = text.strip()
    # Try direct parse first
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Try to extract from ```json ... ``` blocks
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', stripped, re.DOTALL)
    if match:
        return json.loads(match.group(1).strip())
    # Find the first [ or { and parse from there
    first_bracket = len(stripped)
    first_brace = len(stripped)
    if '[' in stripped:
        first_bracket = stripped.index('[')
    if '{' in stripped:
        first_brace = stripped.index('{')
    start = min(first_bracket, first_brace)
    if start < len(stripped):
        return json.loads(stripped[start:])
    return json.loads(stripped)


class DataSourceOrchestrator:
    """Orchestrator for the data source stage."""

    def __init__(self, openai_api_key: Optional[str] = None, collector: Optional[MetricsCollector] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.collector = collector

        self.file_parser = FileParserAgent(self.api_key, collector=collector)
        self.data_validator = DataValidationAgent(self.api_key, collector=collector)
        self.source_output: Optional[DataSourceOutput] = None
    
    def _generate_data_requirements(
        self,
        research_question: str,
        file_info: FileInfo,
    ) -> List[DataRequirement]:
        """Use LLM to derive data requirements from the research question and map them to uploaded columns."""
        if not research_question:
            return []

        llm = LLMClient(
            temperature=0.2,
            api_key=self.api_key,
            collector=self.collector,
            agent_name="DataSourceOrchestrator",
        )

        system_prompt = "You are a quantitative economics research assistant. Given a research question and the columns available in an uploaded dataset, identify the key data requirements."
        user_prompt = """Research question: {research_question}

Available columns in the uploaded file: {columns}

For each variable required to answer the research question, provide:
1. variable_name -- a short descriptive name
2. role -- one of: dependent, independent, control, instrument
3. mapped_column -- the uploaded column name that best satisfies this requirement, or "MISSING" if none matches
4. justification -- why this variable is needed

Return a JSON array of objects with these four fields. Include 3-6 requirements.

Respond with ONLY the JSON array, no other text."""

        try:
            result = llm.format_and_invoke(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                variables={
                    "research_question": research_question,
                    "columns": ", ".join(file_info.column_names),
                }
            )
            items = _extract_json(result)
            if isinstance(items, dict):
                items = items.get("requirements", items.get("data_requirements", [items]))
            reqs = []
            for item in items:
                reqs.append(DataRequirement(
                    variable_name=item.get("variable_name", ""),
                    role=item.get("role", "independent"),
                    mapped_column=item.get("mapped_column", "MISSING"),
                    justification=item.get("justification", ""),
                ))
            return reqs
        except Exception as e:
            print(f"  [WARNING] Data requirements generation failed: {e}")
            return []

    def run_source_pipeline(
        self,
        file_path: str,
        research_question: str = "",
        enable_hitl: bool = True
    ) -> DataSourceOutput:
        """Run the complete data source pipeline."""

        print(f"\n{'='*70}")
        print(f"FILE PARSING & QUALITY ASSESSMENT PIPELINE")
        print(f"{'='*70}")
        print(f"File: {file_path}")
        print(f"Research Question: {research_question[:60]}..." if len(research_question) > 60 else f"Research Question: {research_question}")
        print(f"{'='*70}\n")

        # Step 1: Parse file
        print(f"STEP 1: FILE PARSING")
        print(f"-"*70)
        file_info = self.file_parser.parse_file(file_path)
        df = self.file_parser.get_dataframe()

        # Step 2: Quality assessment
        print(f"\nSTEP 2: QUALITY ASSESSMENT")
        print(f"-"*70)
        quality_assessment = self.data_validator.assess_quality(file_info, df)

        # Step 2b: Generate data requirements from research question
        print(f"\nSTEP 2b: DATA REQUIREMENTS MAPPING")
        print(f"-"*70)
        data_requirements = self._generate_data_requirements(research_question, file_info)
        print(f"  Mapped {len(data_requirements)} requirements to uploaded columns")

        # HITL Checkpoint 1: Quality Review
        if enable_hitl:
            feedback = checkpoint1_quality_review(quality_assessment)
            print(f"[HITL] Quality review feedback: {feedback}")

        # Build cross-domain opportunities based on uploaded file contents
        col_names = file_info.column_names
        cross_domain_opps = []
        if len(col_names) >= 2:
            cross_domain_opps.append(
                f"Enrich uploaded dataset columns ({', '.join(col_names[:3])}) with external macroeconomic indicators from FRED or World Bank for contextual analysis"
            )
        else:
            cross_domain_opps.append(
                "Limited columns restrict cross-domain integration; consider merging with external datasets"
            )
        cross_domain_opps.append(
            "Link uploaded microdata with geographic or demographic panel data to capture spatial or cohort heterogeneity"
        )
        cross_domain_opps.append(
            "Combine user-provided survey data with administrative records for validation and triangulation"
        )

        rq_alignment = (
            f"Uploaded file '{file_info.file_name}' contains {file_info.num_rows} observations across "
            f"{file_info.num_columns} variables. The dataset provides direct empirical content for the "
            f"research question with an overall quality score of {quality_assessment.overall_score}/100 ({quality_assessment.grade})."
        )

        # Generate assumptions and limitations based on uploaded file context
        mapped_count = sum(1 for r in data_requirements if r.mapped_column != "MISSING")
        missing_count = sum(1 for r in data_requirements if r.mapped_column == "MISSING")
        assumptions = [
            f"Assumes the uploaded file '{file_info.file_name}' is representative of the population of interest",
            f"Assumes {file_info.num_rows} observations provide sufficient statistical power for the research question",
            f"Assumes the {file_info.num_columns} variables capture the relevant dimensions of the research question",
            f"Assumes data quality score of {quality_assessment.overall_score}/100 ({quality_assessment.grade}) is acceptable for the intended analysis",
            "Assumes the data collection methodology (unknown for user-uploaded data) does not introduce systematic biases",
            "Assumes column names and units are correctly documented and interpretable"
        ]
        limitations = [
            "Data provenance and collection methodology are unknown for user-uploaded files",
            f"Quality assessment identified {len(quality_assessment.actionable_issues)} actionable issues that may affect analysis validity",
            f"{'All' if missing_count == 0 else str(mapped_count) + ' of ' + str(len(data_requirements))} required variables are mapped to uploaded columns" + (f"; {missing_count} required variable(s) are MISSING" if missing_count > 0 else ""),
            "External validation of data accuracy is not possible without original source documentation",
            "Results may not generalize beyond the sample represented in the uploaded file",
            "User-uploaded data may not be updated or maintained, limiting temporal relevance"
        ]

        # Create output
        self.source_output = DataSourceOutput(
            file_path=file_path,
            research_question=research_question,
            file_info=file_info,
            quality_assessment=quality_assessment,
            data_requirements=data_requirements,
            cross_domain_opportunities=cross_domain_opps,
            research_question_alignment=rq_alignment,
            assumptions=assumptions,
            limitations=limitations,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "pipeline_stage": "data_source",
                "hitl_enabled": enable_hitl
            }
        )
        
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"File: {file_info.file_name}")
        print(f"Rows: {file_info.num_rows}")
        print(f"Columns: {file_info.num_columns}")
        print(f"Quality Score: {quality_assessment.overall_score} ({quality_assessment.grade})")
        print(f"{'='*70}\n")
        
        return self.source_output
    
    def save_source_output(self, filename: str = "user_data_source_output.json"):
        """Save source output to JSON."""
        if not self.source_output:
            print("No source output to save")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.source_output.model_dump(), f, indent=2)
        
        print(f"[Orchestrator] Source output saved to {filename}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main function for data source stage."""
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Get file path from user
    print("Enter the path to your data file (or press Enter for example):")
    file_path = auto_input("> ", default=get_default("user_uploaded_file_path")).strip()
    
    if not file_path:
        # Create example file
        file_path = "example_data.csv"
        if not os.path.exists(file_path):
            print("\nCreating example data file...")
            example_data = {
                "id": range(1, 101),
                "name": [f"Person_{i}" for i in range(1, 101)],
                "age": [25 + (i % 40) for i in range(100)],
                "income": [30000 + (i * 500) for i in range(100)],
                "education_years": [12 + (i % 10) for i in range(100)],
                "employed": [i % 2 == 0 for i in range(100)]
            }
            df = pd.DataFrame(example_data)
            df.to_csv(file_path, index=False)
            print(f"Created: {file_path}")
    
    # Run pipeline
    orchestrator = DataSourceOrchestrator()
    source_output = orchestrator.run_source_pipeline(
        file_path=file_path,
        research_question="Analyze the relationship between education and income",
        enable_hitl=True
    )
    
    # Save outputs
    orchestrator.save_source_output("user_data_source_output.json")
    
    print("\n" + "="*70)
    print("DATA SOURCE SUMMARY")
    print("="*70)
    print(f"File: {source_output.file_info.file_name}")
    print(f"Rows: {source_output.file_info.num_rows}")
    print(f"Columns: {source_output.file_info.num_columns}")
    print(f"Quality: {source_output.quality_assessment.overall_score} ({source_output.quality_assessment.grade})")
    print("="*70)


if __name__ == "__main__":
    main()
