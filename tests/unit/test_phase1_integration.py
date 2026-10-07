# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Tests for AEL V0.4 Phase 1: Schema-Driven Workflows + Guardrails.

Verifies:
1. All stage files import from canonical schemas (no inline models)
2. Canonical schemas contain all necessary models
3. BudgetController is imported in all MasterOrchestrators
4. PIIDetector is imported in DataTeam MasterOrchestrators
5. SchemaValidator can validate outputs from all teams/stages
"""

import ast
import os
import sys
from pathlib import Path

import pytest


# ============================================================================
# Helpers
# ============================================================================

def _get_stage_files():
    """Return all AEL stage files (1-*, 2-*, 3-*Stage.py) across all teams."""
    agents_dir = Path(__file__).resolve().parent.parent.parent
    patterns = [
        "IdeationTeam/ael/Mode*/*-*Stage.py",
        "LiteratureTeam/ael/Mode*/*-*Stage.py",
        "DataTeam/ael/Mode*/*-*Stage.py",
        "ModelTeam/ael/Mode*/*-*Stage.py",
    ]
    files = []
    for pattern in patterns:
        files.extend(agents_dir.glob(pattern))
    return sorted(files)


def _get_master_orchestrators():
    """Return all MasterOrchestrator files."""
    agents_dir = Path(__file__).resolve().parent.parent.parent
    patterns = [
        "IdeationTeam/ael/Mode*/0-MasterOrchestrator.py",
        "LiteratureTeam/ael/Mode*/0-MasterOrchestrator.py",
        "DataTeam/ael/Mode*/0-MasterOrchestrator.py",
        "ModelTeam/ael/Mode*/0-MasterOrchestrator.py",
    ]
    files = []
    for pattern in patterns:
        files.extend(agents_dir.glob(pattern))
    return sorted(files)


def _get_datateam_orchestrators():
    """Return DataTeam MasterOrchestrator files."""
    agents_dir = Path(__file__).resolve().parent.parent.parent
    return sorted(agents_dir.glob("DataTeam/ael/Mode*/0-MasterOrchestrator.py"))


def _parse_ast(filepath: Path):
    """Parse a Python file and return its AST."""
    source = filepath.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(filepath))


def _find_class_defs(tree: ast.AST):
    """Find all class definitions that inherit from BaseModel."""
    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = None
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name == "BaseModel":
                    classes.append(node.name)
    return classes


def _find_imports(tree: ast.AST, module_substring: str):
    """Check if any import statement contains the given module substring."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if module_substring in node.module:
                return True
    return False


# ============================================================================
# Test 1: No inline BaseModel definitions in stage files
# ============================================================================

class TestNoInlineModels:
    """Verify all stage files import from canonical schemas (zero inline models)."""

    def test_stage_files_exist(self):
        """Verify we found stage files to test."""
        files = _get_stage_files()
        assert len(files) >= 44, f"Expected ≥44 stage files, found {len(files)}"

    def test_no_inline_basemodel_in_stage_files(self):
        """No stage file should define its own BaseModel subclass."""
        files = _get_stage_files()
        violations = []
        for f in files:
            tree = _parse_ast(f)
            inline_models = _find_class_defs(tree)
            if inline_models:
                rel = f.relative_to(f.parent.parent.parent.parent)
                violations.append(f"{rel}: {inline_models}")
        assert violations == [], (
            f"Inline BaseModel definitions found in {len(violations)} files:\n"
            + "\n".join(violations)
        )


# ============================================================================
# Test 2: Canonical schemas contain required models
# ============================================================================

class TestCanonicalSchemas:
    """Verify canonical schemas export all necessary models."""

    def test_ideation_schema_exports(self):
        from IdeationTeam.ael.schemas.stage_outputs import (
            LiteratureItem, SearchQuery, SourcingFeedback, SourcingStageOutput,
            ResearchConcept, EvolutionTrace, ResearchQuestion,
            ConceptList, QuestionList, HumanFeedback, StageTransitionSummary,
            RefinementStageOutput,
            ContextualizedQuestion, PrioritizedQuestion,
            IntegrationFeedback, ContextualizedQuestionList, PrioritizedQuestionList,
            IntegrationStageOutput,
        )
        assert LiteratureItem is not None
        assert SourcingFeedback is not None
        assert ConceptList is not None
        assert IntegrationFeedback is not None

    def test_literature_schema_exports(self):
        from LiteratureTeam.ael.schemas.stage_outputs import (
            LiteratureItem, TrendAnalysis, CitationEntry, KnowledgeInsight,
            LiteratureBatch, GatheringStageOutput,
            PaperStructure, ResearchGap, KnowledgeNode, KnowledgeEdge,
            KnowledgeGraph, GapAnalysisResult, GapDetectionStageOutput,
            LiteratureSection, LiteratureReview, ResearchObjective, ResearchPlan,
            FormattedReference, Bibliography, CrossDomainConnection,
            SynthesisResult, SynthesisStageOutput,
        )
        assert LiteratureItem is not None
        assert SynthesisResult is not None

    def test_model_schema_exports(self):
        from ModelTeam.ael.schemas.stage_outputs import (
            TheoreticalAssumption, MathematicalFormulation, ConceptualComponent,
            TheoreticalFramework, TheoryStageOutput,
            ModelVariable, ModelParameter, ModelEquation, ModelConstraint,
            ModelDynamics, EquilibriumDefinition, SolutionMethod,
            FormalMathematicalModel, ModelDesignStageOutput, ModelDesignOutput,
            EmpiricalTarget, CalibrationStrategy, CalibratedParameter,
            ModelMoment, FitMetrics, SandboxValidation, CalibratedModel,
            CalibrationStageOutput, CalibrationOutput,
        )
        assert SandboxValidation is not None
        assert ModelDesignOutput is ModelDesignStageOutput
        assert CalibrationOutput is CalibrationStageOutput

    def test_data_schema_exports(self):
        from DataTeam.ael.schemas.stage_outputs import (
            DataRequirement, APISource, DataSeries, RetrievedData,
            DataSourceStageOutput, DataSourceOutput,
            QualityDimension, QualityAssessment, AlignedSeries, IntegratedDataset,
            DataCleaningStageOutput, DataCleaningOutput,
            VariableEntry, DataCodebook, DataReport, DocumentedDataset,
            QualityAssuranceStageOutput, QualityAssuranceOutput,
        )
        assert DataSourceOutput is DataSourceStageOutput
        assert DataCleaningOutput is DataCleaningStageOutput
        assert QualityAssuranceOutput is QualityAssuranceStageOutput

    def test_data_premium_models(self):
        from DataTeam.ael.schemas.stage_outputs import (
            VendorCredential, CredentialValidation,
            VendorCostEstimate, CostEstimation,
            OptimizedQuery, QueryOptimization,
            RetrievedDataset, DataRetrieval, VendorQualityCheck,
            LicenseRestriction, LicenseCompliance,
            IdentifierMapping, FieldMapping, DataStandardization,
        )
        assert VendorCredential is not None
        assert DataStandardization is not None

    def test_data_user_uploaded_models(self):
        from DataTeam.ael.schemas.stage_outputs import (
            FileInfo,
            PIIDetection, PrivacyScreening,
            StructureInference, ResearchAlignment,
            Transformation, TransformationResults,
        )
        assert FileInfo is not None
        assert TransformationResults is not None


# ============================================================================
# Test 3: BudgetController in all MasterOrchestrators
# ============================================================================

class TestBudgetControllerIntegration:
    """Verify BudgetController is integrated in all 15 MasterOrchestrators."""

    def test_all_orchestrators_found(self):
        files = _get_master_orchestrators()
        assert len(files) == 15, f"Expected 15 MasterOrchestrators, found {len(files)}"

    def test_budget_controller_imported(self):
        """Every MasterOrchestrator imports BudgetController."""
        files = _get_master_orchestrators()
        missing = []
        for f in files:
            tree = _parse_ast(f)
            has_import = _find_imports(tree, "budget_controller")
            if not has_import:
                rel = f.relative_to(f.parent.parent.parent.parent)
                missing.append(str(rel))
        assert missing == [], f"BudgetController import missing in: {missing}"

    def test_budget_exceeded_error_imported(self):
        """Every MasterOrchestrator imports BudgetExceededError."""
        files = _get_master_orchestrators()
        missing = []
        for f in files:
            source = f.read_text(encoding="utf-8")
            if "BudgetExceededError" not in source:
                rel = f.relative_to(f.parent.parent.parent.parent)
                missing.append(str(rel))
        assert missing == [], f"BudgetExceededError missing in: {missing}"

    def test_budget_instantiation(self):
        """Every MasterOrchestrator creates a BudgetController instance."""
        files = _get_master_orchestrators()
        missing = []
        for f in files:
            source = f.read_text(encoding="utf-8")
            if "BudgetController(" not in source:
                rel = f.relative_to(f.parent.parent.parent.parent)
                missing.append(str(rel))
        assert missing == [], f"BudgetController() instantiation missing in: {missing}"

    def test_budget_check_calls(self):
        """Every MasterOrchestrator calls budget.check()."""
        files = _get_master_orchestrators()
        missing = []
        for f in files:
            source = f.read_text(encoding="utf-8")
            if "budget.check(" not in source:
                rel = f.relative_to(f.parent.parent.parent.parent)
                missing.append(str(rel))
        assert missing == [], f"budget.check() call missing in: {missing}"


# ============================================================================
# Test 4: PIIDetector in DataTeam MasterOrchestrators
# ============================================================================

class TestPIIDetectorIntegration:
    """Verify PIIDetector is integrated in DataTeam MasterOrchestrators."""

    def test_datateam_orchestrators_found(self):
        files = _get_datateam_orchestrators()
        assert len(files) == 3, f"Expected 3 DataTeam MasterOrchestrators, found {len(files)}"

    def test_pii_detector_imported(self):
        """Every DataTeam MasterOrchestrator imports PIIDetector."""
        files = _get_datateam_orchestrators()
        missing = []
        for f in files:
            tree = _parse_ast(f)
            has_import = _find_imports(tree, "pii_detector")
            if not has_import:
                rel = f.relative_to(f.parent.parent.parent.parent)
                missing.append(str(rel))
        assert missing == [], f"PIIDetector import missing in: {missing}"

    def test_pii_detector_instantiation(self):
        """Every DataTeam MasterOrchestrator creates a PIIDetector instance."""
        files = _get_datateam_orchestrators()
        missing = []
        for f in files:
            source = f.read_text(encoding="utf-8")
            if "PIIDetector()" not in source:
                rel = f.relative_to(f.parent.parent.parent.parent)
                missing.append(str(rel))
        assert missing == [], f"PIIDetector() instantiation missing in: {missing}"

    def test_pii_scan_call(self):
        """Every DataTeam MasterOrchestrator calls pii_detector.scan_dict()."""
        files = _get_datateam_orchestrators()
        missing = []
        for f in files:
            source = f.read_text(encoding="utf-8")
            if "pii_detector.scan_dict(" not in source:
                rel = f.relative_to(f.parent.parent.parent.parent)
                missing.append(str(rel))
        assert missing == [], f"pii_detector.scan_dict() call missing in: {missing}"

    def test_pii_not_in_non_datateam(self):
        """Non-DataTeam MasterOrchestrators should NOT import PIIDetector."""
        all_orch = _get_master_orchestrators()
        data_orch = set(_get_datateam_orchestrators())
        non_data = [f for f in all_orch if f not in data_orch]
        unexpected = []
        for f in non_data:
            tree = _parse_ast(f)
            if _find_imports(tree, "pii_detector"):
                rel = f.relative_to(f.parent.parent.parent.parent)
                unexpected.append(str(rel))
        assert unexpected == [], f"PIIDetector unexpectedly imported in: {unexpected}"


# ============================================================================
# Test 5: SchemaValidator compatibility
# ============================================================================

class TestSchemaValidatorCompatibility:
    """Verify SchemaValidator can find schemas for all team/stage pairs."""

    def test_schema_validator_lists_all_teams(self):
        from shared.guardrails.schema_validator import SchemaValidator
        SchemaValidator.reset()
        schemas = SchemaValidator.list_schemas()
        assert "IdeationTeam" in schemas
        assert "LiteratureTeam" in schemas
        assert "ModelTeam" in schemas
        assert "DataTeam" in schemas

    def test_schema_validator_ideation_stages(self):
        from shared.guardrails.schema_validator import SchemaValidator
        SchemaValidator.reset()
        schemas = SchemaValidator.list_schemas()
        assert "SourcingStage" in schemas["IdeationTeam"]
        assert "RefinementStage" in schemas["IdeationTeam"]
        assert "IntegrationStage" in schemas["IdeationTeam"]

    def test_schema_validator_literature_stages(self):
        from shared.guardrails.schema_validator import SchemaValidator
        SchemaValidator.reset()
        schemas = SchemaValidator.list_schemas()
        assert "LiteratureGatheringStage" in schemas["LiteratureTeam"]
        assert "GapDetectionStage" in schemas["LiteratureTeam"]
        assert "SynthesisStage" in schemas["LiteratureTeam"]

    def test_schema_validator_model_stages(self):
        from shared.guardrails.schema_validator import SchemaValidator
        SchemaValidator.reset()
        schemas = SchemaValidator.list_schemas()
        assert "TheoryStage" in schemas["ModelTeam"]
        assert "ModelDesignStage" in schemas["ModelTeam"]
        assert "CalibrationStage" in schemas["ModelTeam"]

    def test_schema_validator_data_stages(self):
        from shared.guardrails.schema_validator import SchemaValidator
        SchemaValidator.reset()
        schemas = SchemaValidator.list_schemas()
        assert "DataSourceStage" in schemas["DataTeam"]
        assert "DataCleaningStage" in schemas["DataTeam"]
        assert "QualityAssuranceStage" in schemas["DataTeam"]

    def test_validate_valid_ideation_output(self):
        """Validate a sample IdeationTeam SourcingStage output."""
        from shared.guardrails.schema_validator import SchemaValidator
        SchemaValidator.reset()
        sample = {
            "literature_items": [
                {
                    "title": "Test Paper",
                    "authors": ["Author A"],
                    "url": "https://example.com",
                    "source": "arXiv",
                    "agent": "TrendSurfer",
                }
            ],
            "search_queries": [],
            "metadata": {},
        }
        result = SchemaValidator.validate("IdeationTeam", "SourcingStage", sample)
        assert result.valid, f"Validation failed: {result.errors}"

    def test_validate_invalid_ideation_output(self):
        """Invalid output should fail validation."""
        from shared.guardrails.schema_validator import SchemaValidator
        SchemaValidator.reset()
        sample = {"missing_field": True}
        result = SchemaValidator.validate("IdeationTeam", "SourcingStage", sample)
        assert not result.valid


# ============================================================================
# Test 6: SchemaValidator in all MasterOrchestrators
# ============================================================================

class TestSchemaValidatorIntegration:
    """Verify SchemaValidator.validate_or_raise() is wired into all MasterOrchestrators."""

    def test_schema_validator_imported(self):
        """Every MasterOrchestrator imports SchemaValidator."""
        files = _get_master_orchestrators()
        missing = []
        for f in files:
            tree = _parse_ast(f)
            has_import = _find_imports(tree, "schema_validator")
            if not has_import:
                rel = f.relative_to(f.parent.parent.parent.parent)
                missing.append(str(rel))
        assert missing == [], f"SchemaValidator import missing in: {missing}"

    def test_validate_or_raise_calls(self):
        """Every MasterOrchestrator calls validate_or_raise() exactly 3 times (once per stage)."""
        files = _get_master_orchestrators()
        incorrect = []
        for f in files:
            source = f.read_text(encoding="utf-8")
            count = source.count("validate_or_raise(")
            if count != 3:
                rel = f.relative_to(f.parent.parent.parent.parent)
                incorrect.append(f"{rel}: {count} calls (expected 3)")
        assert incorrect == [], (
            f"validate_or_raise() call count wrong in:\n" + "\n".join(incorrect)
        )

    def test_total_validate_calls(self):
        """Total validate_or_raise() calls across all 15 orchestrators should be 45."""
        files = _get_master_orchestrators()
        total = 0
        for f in files:
            source = f.read_text(encoding="utf-8")
            total += source.count("validate_or_raise(")
        assert total == 45, f"Expected 45 total validate_or_raise() calls, got {total}"


# ============================================================================
# Test 7: Integration test — schema validation on sample workflow outputs
# ============================================================================

class TestSchemaValidationIntegration:
    """End-to-end: verify SchemaValidator validates real-format stage outputs."""

    def test_ideation_sourcing_output(self):
        from shared.guardrails.schema_validator import SchemaValidator
        SchemaValidator.reset()
        data = {
            "literature_items": [
                {"title": "Paper A", "authors": ["Auth1"], "url": "http://x.com",
                 "source": "arXiv", "agent": "TrendSurfer"}
            ],
            "search_queries": [
                {"queries": ["ABM macro"], "keywords": ["ABM"], "focus_areas": ["macro"]}
            ],
            "metadata": {"stage": "SourcingStage", "item_count": 1},
        }
        result = SchemaValidator.validate("IdeationTeam", "SourcingStage", data)
        assert result.valid, f"Validation failed: {result.errors}"

    def test_literature_gathering_output(self):
        from shared.guardrails.schema_validator import SchemaValidator
        SchemaValidator.reset()
        data = {
            "literature_batch": {
                "research_questions": [],
                "literature_items": [],
                "trend_analyses": [],
                "citations": [],
                "insights": [],
            },
            "metadata": {},
        }
        result = SchemaValidator.validate("LiteratureTeam", "LiteratureGatheringStage", data)
        assert result.valid, f"Validation failed: {result.errors}"

    def test_model_theory_output(self):
        from shared.guardrails.schema_validator import SchemaValidator
        SchemaValidator.reset()
        data = {
            "research_questions": [],
            "theoretical_frameworks": [],
            "metadata": {},
        }
        result = SchemaValidator.validate("ModelTeam", "TheoryStage", data)
        assert result.valid, f"Validation failed: {result.errors}"

    def test_data_source_output(self):
        from shared.guardrails.schema_validator import SchemaValidator
        SchemaValidator.reset()
        data = {
            "research_question": "Test question",
            "data_requirements": [],
            "discovered_sources": [],
            "selected_series": [],
            "retrieved_data": [],
            "metadata": {},
        }
        result = SchemaValidator.validate("DataTeam", "DataSourceStage", data)
        assert result.valid, f"Validation failed: {result.errors}"

    def test_validate_or_raise_success(self):
        """validate_or_raise returns dict on valid data."""
        from shared.guardrails.schema_validator import SchemaValidator
        SchemaValidator.reset()
        data = {
            "research_questions": [],
            "theoretical_frameworks": [],
            "metadata": {},
        }
        result = SchemaValidator.validate_or_raise("ModelTeam", "TheoryStage", data)
        assert isinstance(result, dict)

    def test_validate_or_raise_failure(self):
        """validate_or_raise raises ValueError on invalid data."""
        from shared.guardrails.schema_validator import SchemaValidator
        SchemaValidator.reset()
        with pytest.raises(ValueError, match="Schema validation failed"):
            SchemaValidator.validate_or_raise("ModelTeam", "TheoryStage", {"bad": True})


# ============================================================================
# Test 8: Backward-compatible aliases
# ============================================================================

class TestBackwardCompatAliases:
    """Verify backward-compatible aliases work correctly."""

    def test_model_design_output_alias(self):
        from ModelTeam.ael.schemas.stage_outputs import (
            ModelDesignOutput, ModelDesignStageOutput,
        )
        assert ModelDesignOutput is ModelDesignStageOutput

    def test_calibration_output_alias(self):
        from ModelTeam.ael.schemas.stage_outputs import (
            CalibrationOutput, CalibrationStageOutput,
        )
        assert CalibrationOutput is CalibrationStageOutput

    def test_data_source_output_alias(self):
        from DataTeam.ael.schemas.stage_outputs import (
            DataSourceOutput, DataSourceStageOutput,
        )
        assert DataSourceOutput is DataSourceStageOutput

    def test_data_cleaning_output_alias(self):
        from DataTeam.ael.schemas.stage_outputs import (
            DataCleaningOutput, DataCleaningStageOutput,
        )
        assert DataCleaningOutput is DataCleaningStageOutput

    def test_quality_assurance_output_alias(self):
        from DataTeam.ael.schemas.stage_outputs import (
            QualityAssuranceOutput, QualityAssuranceStageOutput,
        )
        assert QualityAssuranceOutput is QualityAssuranceStageOutput
