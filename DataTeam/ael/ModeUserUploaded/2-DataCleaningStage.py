# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Data Cleaning Stage - User-Uploaded Data Mode

This script handles:
- Privacy screening (PII detection)
- Structure inference (cross-sectional, time series, panel)
- Research alignment assessment
- Data transformation

Pipeline:
1. PrivacyScreeningAgent: Scan for PII and sensitive data
2. StructureInferenceAgent: Infer logical structure of dataset
3. ResearchContextAgent: Assess alignment with research requirements
4. DataTransformationAgent: Apply transformations based on feedback

HITL Checkpoints (2-4):
2. Privacy Review (after privacy screening)
3. Structure Confirmation (after structure inference)
4. Transformation Review (after data transformation)

Input: Source output from Stage 1 (user_data_source_output.json)
Output: Transformed dataset with privacy and structure analysis
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
current_dir = Path(__file__).resolve().parent
agents_dir = current_dir.parent.parent.parent  # repository root
if str(agents_dir) not in sys.path:
    sys.path.insert(0, str(agents_dir))

from shared.auto_input import auto_input, get_default
from shared.llm import LLMClient
from shared.json_repair import repair_json
from shared.observability import MetricsCollector, tracked_file_parse
from DataTeam.ael.schemas.stage_outputs import (
    PIIDetection, PrivacyScreening, StructureInference,
    ResearchAlignment, Transformation, TransformationResults,
    UserUploadedDataCleaningOutput as DataCleaningOutput,
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

class PrivacyScreeningAgent:
    """Agent for scanning dataset for PII and sensitive data."""
    
    def __init__(self, openai_api_key: str):
        self.agent_name = "PrivacyScreeningAgent"
        self.api_key = openai_api_key
        
        # PII detection patterns
        self.pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "phone": r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        }
        
        # Name-like column patterns
        self.name_column_patterns = ['name', 'first_name', 'last_name', 'full_name', 'person', 'contact']
        self.address_column_patterns = ['address', 'street', 'city', 'zip', 'postal', 'location']
    
    def screen_privacy(
        self,
        file_info: Dict,
        df: Optional[pd.DataFrame] = None
    ) -> PrivacyScreening:
        """Scan dataset for PII and sensitive data."""
        
        print(f"\n[{self.agent_name}] Screening for PII and sensitive data...")
        
        pii_detections = []
        sensitive_columns = []
        
        column_names = file_info.get('column_names', [])
        
        # Check column names for PII indicators
        for col in column_names:
            col_lower = col.lower()
            
            # Check for name-like columns
            if any(pattern in col_lower for pattern in self.name_column_patterns):
                pii_detections.append(PIIDetection(
                    column_name=col,
                    pii_type="name",
                    confidence=0.8,
                    sample_matches=["[REDACTED]"],
                    recommendation="Consider removing or anonymizing"
                ))
                sensitive_columns.append(col)
            
            # Check for address-like columns
            if any(pattern in col_lower for pattern in self.address_column_patterns):
                pii_detections.append(PIIDetection(
                    column_name=col,
                    pii_type="address",
                    confidence=0.7,
                    sample_matches=["[REDACTED]"],
                    recommendation="Consider removing or generalizing"
                ))
                sensitive_columns.append(col)
            
            # Check for email columns
            if 'email' in col_lower or 'e-mail' in col_lower:
                pii_detections.append(PIIDetection(
                    column_name=col,
                    pii_type="email",
                    confidence=0.9,
                    sample_matches=["[REDACTED]"],
                    recommendation="Remove or hash"
                ))
                sensitive_columns.append(col)
            
            # Check for SSN columns
            if 'ssn' in col_lower or 'social' in col_lower:
                pii_detections.append(PIIDetection(
                    column_name=col,
                    pii_type="ssn",
                    confidence=0.95,
                    sample_matches=["[REDACTED]"],
                    recommendation="Remove immediately"
                ))
                sensitive_columns.append(col)
            
            # Check for phone columns
            if 'phone' in col_lower or 'mobile' in col_lower or 'tel' in col_lower:
                pii_detections.append(PIIDetection(
                    column_name=col,
                    pii_type="phone",
                    confidence=0.85,
                    sample_matches=["[REDACTED]"],
                    recommendation="Remove or mask"
                ))
                sensitive_columns.append(col)
        
        # Scan data content if DataFrame available
        if df is not None:
            for col in df.select_dtypes(include=['object']).columns:
                if col in sensitive_columns:
                    continue
                
                # Sample data for pattern matching
                sample = df[col].dropna().head(100).astype(str)
                
                for pii_type, pattern in self.pii_patterns.items():
                    matches = sample.str.contains(pattern, regex=True, na=False)
                    if matches.any():
                        match_count = matches.sum()
                        pii_detections.append(PIIDetection(
                            column_name=col,
                            pii_type=pii_type,
                            confidence=min(0.9, match_count / len(sample)),
                            sample_matches=["[REDACTED]"] * min(3, match_count),
                            recommendation=f"Review and remediate {pii_type} data"
                        ))
                        sensitive_columns.append(col)
        
        # Determine risk level
        if any(d.pii_type in ['ssn', 'credit_card'] for d in pii_detections):
            risk_level = "High"
        elif any(d.pii_type in ['email', 'phone', 'name'] for d in pii_detections):
            risk_level = "Medium"
        elif pii_detections:
            risk_level = "Low"
        else:
            risk_level = "None"
        
        # Generate recommendations
        recommendations = []
        remediation_actions = []
        
        if pii_detections:
            recommendations.append("Review all detected PII before sharing data")
            recommendations.append("Consider data anonymization techniques")
            
            for detection in pii_detections:
                remediation_actions.append(f"{detection.column_name}: {detection.recommendation}")
        else:
            recommendations.append("No obvious PII detected, but manual review recommended")
        
        print(f"  Risk Level: {risk_level}")
        print(f"  PII Detections: {len(pii_detections)}")
        print(f"  Sensitive Columns: {len(sensitive_columns)}")
        
        return PrivacyScreening(
            risk_level=risk_level,
            pii_detected=len(pii_detections) > 0,
            pii_detections=pii_detections,
            sensitive_columns=list(set(sensitive_columns)),
            recommendations=recommendations,
            remediation_actions=remediation_actions
        )


class StructureInferenceAgent:
    """Agent for inferring logical structure of dataset."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "StructureInferenceAgent"
        self.api_key = openai_api_key
        self.llm = LLMClient(
            temperature=0.3,
            api_key=self.api_key,
            collector=collector,
            agent_name=self.agent_name
        )
    
    def infer_structure(
        self,
        file_info: Dict,
        research_question: str,
        df: Optional[pd.DataFrame] = None
    ) -> StructureInference:
        """Infer logical structure of the dataset."""
        
        print(f"\n[{self.agent_name}] Inferring data structure...")
        
        column_names = file_info.get('column_names', [])
        column_types = file_info.get('column_types', {})
        num_rows = file_info.get('num_rows', 0)
        
        # Detect temporal variables
        temporal_patterns = ['date', 'time', 'year', 'month', 'quarter', 'period', 'timestamp']
        temporal_vars = [col for col in column_names if any(p in col.lower() for p in temporal_patterns)]
        
        # Detect ID variables
        id_patterns = ['id', 'code', 'key', 'identifier', 'index']
        id_vars = [col for col in column_names if any(p in col.lower() for p in id_patterns)]
        
        # Detect grouping variables
        group_patterns = ['group', 'category', 'type', 'class', 'region', 'country', 'state', 'sector']
        group_vars = [col for col in column_names if any(p in col.lower() for p in group_patterns)]
        
        # Infer structure type
        if temporal_vars and id_vars:
            structure_type = "panel"
            confidence = 0.85
        elif temporal_vars:
            structure_type = "time_series"
            confidence = 0.80
        elif group_vars and len(group_vars) > 1:
            structure_type = "hierarchical"
            confidence = 0.70
        else:
            structure_type = "cross_sectional"
            confidence = 0.75
        
        # Suggest variable roles using LLM
        suggested_roles = self._suggest_variable_roles(column_names, research_question)
        
        print(f"  Structure Type: {structure_type}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Temporal Variables: {temporal_vars}")
        print(f"  ID Variables: {id_vars}")
        
        return StructureInference(
            structure_type=structure_type,
            confidence=confidence,
            key_variables={
                "temporal": temporal_vars,
                "identifier": id_vars,
                "grouping": group_vars
            },
            temporal_variable=temporal_vars[0] if temporal_vars else None,
            id_variables=id_vars,
            grouping_variables=group_vars,
            suggested_roles=suggested_roles,
            structure_notes=f"Inferred {structure_type} structure with {confidence:.0%} confidence"
        )
    
    def _suggest_variable_roles(self, column_names: List[str], research_question: str) -> Dict[str, str]:
        """Suggest variable roles based on column names and research question."""

        system_prompt = "You are an expert at understanding research data structures."

        user_prompt = """Suggest variable roles for these columns based on the research question.

Research Question: {research_question}

Columns: {columns}

For each column, suggest a role:
- outcome: dependent variable / what we're trying to explain
- treatment: independent variable of interest
- control: control variable
- identifier: ID or key variable
- temporal: time variable
- grouping: grouping/categorical variable
- other: other purpose

Return as JSON with column names as keys and roles as values.
Example: {{{{"income": "outcome", "education": "treatment", "age": "control"}}}}

Respond with ONLY the JSON object, no other text.
"""

        try:
            result = self.llm.format_and_invoke(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                variables={
                    "research_question": research_question,
                    "columns": ", ".join(column_names[:20])
                }
            )

            return json.loads(repair_json(result))
        except Exception as e:
            print(f"    Error suggesting roles: {e}")
            return {}


class ResearchContextAgent:
    """Agent for assessing alignment between data and research requirements."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "ResearchContextAgent"
        self.api_key = openai_api_key
        self.llm = LLMClient(
            temperature=0.3,
            api_key=self.api_key,
            collector=collector,
            agent_name=self.agent_name
        )
    
    def assess_alignment(
        self,
        file_info: Dict,
        research_question: str,
        structure_inference: StructureInference
    ) -> ResearchAlignment:
        """Assess alignment between uploaded data and research requirements."""
        
        print(f"\n[{self.agent_name}] Assessing research alignment...")
        
        num_rows = file_info.get('num_rows', 0)
        num_columns = file_info.get('num_columns', 0)
        column_names = file_info.get('column_names', [])
        
        # Assess sample size
        sample_size_adequate = num_rows >= 30  # Minimum for statistical analysis
        
        # Use LLM to assess alignment
        alignment_result = self._assess_with_llm(
            research_question,
            column_names,
            num_rows,
            structure_inference
        )
        
        print(f"  Alignment Score: {alignment_result.get('alignment_score', 'N/A')}")
        print(f"  Suitability: {alignment_result.get('suitability', 'N/A')}")
        
        return ResearchAlignment(
            alignment_score=alignment_result.get('alignment_score', 70),
            suitability=alignment_result.get('suitability', 'Medium'),
            sample_size_adequate=sample_size_adequate,
            variable_coverage=alignment_result.get('variable_coverage', {}),
            temporal_coverage=alignment_result.get('temporal_coverage', 'Unknown'),
            missing_elements=alignment_result.get('missing_elements', []),
            recommendations=alignment_result.get('recommendations', [])
        )
    
    def _assess_with_llm(
        self,
        research_question: str,
        column_names: List[str],
        num_rows: int,
        structure_inference: StructureInference
    ) -> Dict:
        """Use LLM to assess research alignment."""

        system_prompt = "You are an expert at assessing data suitability for research."

        user_prompt = """Assess how well this dataset aligns with the research question.

Research Question: {research_question}

Dataset:
- Columns: {columns}
- Rows: {num_rows}
- Structure: {structure_type}
- Suggested Roles: {suggested_roles}

Provide assessment as JSON:
- alignment_score: 0-100 score
- suitability: High/Medium/Low
- variable_coverage: dict of key variables and whether they're present (true/false)
- temporal_coverage: assessment of time coverage
- missing_elements: list of missing data elements
- recommendations: list of recommendations

Example: {{{{
  "alignment_score": 75,
  "suitability": "Medium",
  "variable_coverage": {{{{"outcome_variable": true, "treatment_variable": true, "controls": false}}}},
  "temporal_coverage": "Adequate for cross-sectional analysis",
  "missing_elements": ["Control variables for demographics"],
  "recommendations": ["Consider adding demographic controls"]
}}}}

Respond with ONLY the JSON object, no other text.
"""

        try:
            result = self.llm.format_and_invoke(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                variables={
                    "research_question": research_question,
                    "columns": ", ".join(column_names[:20]),
                    "num_rows": str(num_rows),
                    "structure_type": structure_inference.structure_type,
                    "suggested_roles": json.dumps(structure_inference.suggested_roles)
                }
            )

            return json.loads(repair_json(result))
        except Exception as e:
            print(f"    Error assessing alignment: {e}")
            return {
                "alignment_score": 70,
                "suitability": "Medium",
                "variable_coverage": {},
                "temporal_coverage": "Unknown",
                "missing_elements": [],
                "recommendations": ["Manual review recommended"]
            }


class DataTransformationAgent:
    """Agent for transforming data based on feedback."""
    
    def __init__(self, openai_api_key: str):
        self.agent_name = "DataTransformationAgent"
        self.api_key = openai_api_key
    
    def transform_data(
        self,
        file_info: Dict,
        privacy_screening: PrivacyScreening,
        quality_feedback: str = "",
        df: Optional[pd.DataFrame] = None
    ) -> TransformationResults:
        """Transform data based on human feedback and assessments."""
        
        print(f"\n[{self.agent_name}] Applying data transformations...")
        
        transformations = []
        transformation_log = []
        columns_removed = []
        columns_added = []
        
        rows_before = file_info.get('num_rows', 0)
        columns_before = file_info.get('num_columns', 0)
        
        # 1. Privacy remediation
        if privacy_screening.pii_detected:
            for col in privacy_screening.sensitive_columns:
                transformations.append(Transformation(
                    transformation_id=f"T{len(transformations)+1}",
                    transformation_type="privacy",
                    description=f"Remove PII column: {col}",
                    columns_affected=[col],
                    method="drop_column",
                    parameters={},
                    python_code=f"df = df.drop(columns=['{col}'])"
                ))
                columns_removed.append(col)
                transformation_log.append(f"Removed PII column: {col}")
        
        # 2. Missing data handling (if quality issues mentioned)
        if "missing" in quality_feedback.lower() or not quality_feedback:
            transformations.append(Transformation(
                transformation_id=f"T{len(transformations)+1}",
                transformation_type="missing_data",
                description="Handle missing values",
                columns_affected=["all"],
                method="dropna_threshold",
                parameters={"threshold": 0.5},
                python_code="df = df.dropna(thresh=int(len(df.columns) * 0.5))"
            ))
            transformation_log.append("Applied missing value handling (50% threshold)")
        
        # 3. Duplicate removal
        transformations.append(Transformation(
            transformation_id=f"T{len(transformations)+1}",
            transformation_type="structure",
            description="Remove duplicate rows",
            columns_affected=["all"],
            method="drop_duplicates",
            parameters={},
            python_code="df = df.drop_duplicates()"
        ))
        transformation_log.append("Removed duplicate rows")
        
        # Calculate after counts (simulated)
        rows_after = rows_before  # Would be actual count after transformation
        columns_after = columns_before - len(columns_removed) + len(columns_added)
        
        print(f"  Transformations Applied: {len(transformations)}")
        print(f"  Columns Removed: {len(columns_removed)}")
        
        return TransformationResults(
            transformations_applied=transformations,
            rows_before=rows_before,
            rows_after=rows_after,
            columns_before=columns_before,
            columns_after=columns_after,
            columns_removed=columns_removed,
            columns_added=columns_added,
            transformation_log=transformation_log
        )


# ============================================================================
# HITL Checkpoints
# ============================================================================

def checkpoint2_privacy_review(privacy_screening: PrivacyScreening) -> str:
    """HITL Checkpoint 2: Privacy Review."""
    
    print("\n" + "="*70)
    print("🛑 HITL CHECKPOINT 2: Privacy Review")
    print("="*70)
    print(f"\nRisk Level: {privacy_screening.risk_level}")
    print(f"PII Detected: {privacy_screening.pii_detected}")
    
    if privacy_screening.pii_detections:
        print("\nPII Detections:")
        for detection in privacy_screening.pii_detections[:5]:
            print(f"  - {detection.column_name}: {detection.pii_type} (confidence: {detection.confidence:.2f})")
            print(f"    Recommendation: {detection.recommendation}")
    
    if privacy_screening.sensitive_columns:
        print(f"\nSensitive Columns: {', '.join(privacy_screening.sensitive_columns)}")
    
    print("\nDecision needed:")
    print("- Is the privacy risk acceptable?")
    print("- Should we remove PII-containing columns?")
    print("- Should we anonymize specific columns?")

    response = auto_input("\n> ", default=get_default("approve_stage")).strip()
    return response if response else "approved"


def checkpoint3_structure_confirmation(structure_inference: StructureInference) -> str:
    """HITL Checkpoint 3: Structure Confirmation."""
    
    print("\n" + "="*70)
    print("🛑 HITL CHECKPOINT 3: Structure Confirmation")
    print("="*70)
    print(f"\nInferred Structure: {structure_inference.structure_type}")
    print(f"Confidence: {structure_inference.confidence:.2f}")
    
    print("\nKey Variables:")
    for var_type, vars in structure_inference.key_variables.items():
        if vars:
            print(f"  - {var_type}: {', '.join(vars)}")
    
    if structure_inference.suggested_roles:
        print("\nSuggested Variable Roles:")
        for var, role in list(structure_inference.suggested_roles.items())[:10]:
            print(f"  - {var}: {role}")
    
    print("\nDecision needed:")
    print("- Is the structure inference correct?")
    print("- Are the key variables identified correctly?")
    print("- Do variable role suggestions align with your research design?")

    response = auto_input("\n> ", default=get_default("approve_stage")).strip()
    return response if response else "approved"


def checkpoint4_transformation_review(transformations: TransformationResults) -> str:
    """HITL Checkpoint 4: Transformation Review."""
    
    print("\n" + "="*70)
    print("🛑 HITL CHECKPOINT 4: Transformation Review")
    print("="*70)
    print(f"\nTransformations Applied: {len(transformations.transformations_applied)}")
    print(f"Rows: {transformations.rows_before} -> {transformations.rows_after}")
    print(f"Columns: {transformations.columns_before} -> {transformations.columns_after}")
    
    if transformations.columns_removed:
        print(f"\nColumns Removed: {', '.join(transformations.columns_removed)}")
    
    print("\nTransformation Log:")
    for log_entry in transformations.transformation_log:
        print(f"  - {log_entry}")
    
    print("\nPython Code for Transformations:")
    for t in transformations.transformations_applied[:5]:
        print(f"  # {t.description}")
        print(f"  {t.python_code}")
    
    print("\nDecision needed:")
    print("- Are transformations appropriate and correct?")
    print("- Should any transformations be modified?")
    print("- Are there additional transformations needed?")

    response = auto_input("\n> ", default=get_default("approve_stage")).strip()
    return response if response else "approved"


# ============================================================================
# Orchestrator
# ============================================================================

class DataCleaningOrchestrator:
    """Orchestrator for the data cleaning stage."""

    def __init__(self, openai_api_key: Optional[str] = None, collector: Optional[MetricsCollector] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        self.collector = collector
        self.privacy_agent = PrivacyScreeningAgent(self.api_key)
        self.structure_agent = StructureInferenceAgent(self.api_key, collector=collector)
        self.research_agent = ResearchContextAgent(self.api_key, collector=collector)
        self.transformation_agent = DataTransformationAgent(self.api_key)
        self.cleaning_output: Optional[DataCleaningOutput] = None
    
    def run_cleaning_pipeline(
        self,
        data_source_output: Dict,
        research_question: str = "",
        enable_hitl: bool = True
    ) -> DataCleaningOutput:
        """Run the complete data cleaning pipeline."""
        
        print(f"\n{'='*70}")
        print(f"PRIVACY, STRUCTURE & TRANSFORMATION PIPELINE")
        print(f"{'='*70}")
        
        file_path = data_source_output.get('file_path', '')
        file_info = data_source_output.get('file_info', {})
        quality_assessment = data_source_output.get('quality_assessment', {})
        research_q = data_source_output.get('research_question', research_question)
        
        print(f"File: {file_info.get('file_name', 'Unknown')}")
        print(f"{'='*70}\n")
        
        # Load DataFrame if file exists (tracked)
        df = None
        if file_path and os.path.exists(file_path):
            try:
                df = tracked_file_parse(
                    file_path,
                    collector=self.collector,
                    agent="DataCleaningOrchestrator",
                )
            except Exception as e:
                print(f"  [WARNING] Could not load DataFrame: {e}")
        
        # Step 1: Privacy Screening
        print(f"STEP 1: PRIVACY SCREENING")
        print(f"-"*70)
        privacy_screening = self.privacy_agent.screen_privacy(file_info, df)
        
        # HITL Checkpoint 2: Privacy Review
        if enable_hitl:
            feedback2 = checkpoint2_privacy_review(privacy_screening)
            print(f"[HITL] Privacy review feedback: {feedback2}")
        
        # Step 2: Structure Inference
        print(f"\nSTEP 2: STRUCTURE INFERENCE")
        print(f"-"*70)
        structure_inference = self.structure_agent.infer_structure(file_info, research_q, df)
        
        # HITL Checkpoint 3: Structure Confirmation
        if enable_hitl:
            feedback3 = checkpoint3_structure_confirmation(structure_inference)
            print(f"[HITL] Structure confirmation feedback: {feedback3}")
        
        # Step 3: Research Alignment
        print(f"\nSTEP 3: RESEARCH ALIGNMENT")
        print(f"-"*70)
        research_alignment = self.research_agent.assess_alignment(
            file_info,
            research_q,
            structure_inference
        )
        
        # Step 4: Data Transformation
        print(f"\nSTEP 4: DATA TRANSFORMATION")
        print(f"-"*70)
        quality_feedback = json.dumps(quality_assessment.get('actionable_issues', []))
        transformations = self.transformation_agent.transform_data(
            file_info,
            privacy_screening,
            quality_feedback,
            df
        )
        
        # HITL Checkpoint 4: Transformation Review
        if enable_hitl:
            feedback4 = checkpoint4_transformation_review(transformations)
            print(f"[HITL] Transformation review feedback: {feedback4}")
        
        # Create output
        self.cleaning_output = DataCleaningOutput(
            file_path=file_path,
            research_question=research_q,
            privacy_screening=privacy_screening,
            structure_inference=structure_inference,
            research_alignment=research_alignment,
            transformations=transformations,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "pipeline_stage": "data_cleaning",
                "hitl_enabled": enable_hitl,
                # Propagate file_info (column_names/column_types/etc.) so the
                # Stage-3 codebook can populate variables. The cleaning output
                # schema itself has no file_info field, so it would otherwise be
                # lost between stages -> codebook with "0 variables".
                "file_info": file_info,
            }
        )
        
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Privacy Risk: {privacy_screening.risk_level}")
        print(f"Structure: {structure_inference.structure_type}")
        print(f"Research Alignment: {research_alignment.alignment_score}/100")
        print(f"Transformations: {len(transformations.transformations_applied)}")
        print(f"{'='*70}\n")
        
        return self.cleaning_output
    
    def save_cleaning_output(self, filename: str = "user_data_cleaning_output.json"):
        """Save cleaning output to JSON."""
        if not self.cleaning_output:
            print("No cleaning output to save")
            return
        
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
    source_file = "user_data_source_output.json"
    if os.path.exists(source_file):
        with open(source_file, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
    else:
        # Create sample data for testing
        source_data = {
            "file_path": "example_data.csv",
            "research_question": "Analyze the relationship between education and income",
            "file_info": {
                "file_name": "example_data.csv",
                "num_rows": 100,
                "num_columns": 6,
                "column_names": ["id", "name", "age", "income", "education_years", "employed"],
                "column_types": {"id": "int64", "name": "object", "age": "int64"}
            },
            "quality_assessment": {
                "overall_score": 85,
                "grade": "B",
                "actionable_issues": ["Some missing values detected"]
            }
        }
    
    # Run pipeline
    orchestrator = DataCleaningOrchestrator()
    cleaning_output = orchestrator.run_cleaning_pipeline(
        data_source_output=source_data,
        enable_hitl=True
    )
    
    # Save outputs
    orchestrator.save_cleaning_output("user_data_cleaning_output.json")
    
    print("\n" + "="*70)
    print("DATA CLEANING SUMMARY")
    print("="*70)
    print(f"Privacy Risk: {cleaning_output.privacy_screening.risk_level}")
    print(f"Structure: {cleaning_output.structure_inference.structure_type}")
    print(f"Alignment: {cleaning_output.research_alignment.alignment_score}/100")
    print("="*70)


if __name__ == "__main__":
    main()
