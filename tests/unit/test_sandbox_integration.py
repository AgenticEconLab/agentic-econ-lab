# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Tests for sandbox integration in ModelTeam CalibrationStage and DataTeam QualityAssuranceStage.

Verifies:
- CodeSandbox import in all stage files
- SandboxValidation data model in ModelTeam
- _validate_with_sandbox method in ModelTeam Calibrator
- _validate_data_with_sandbox method in DataTeam orchestrators
- Sandbox validation result structure
"""

import sys
import importlib
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# repository root for file path lookups (sys.path handled by conftest.py)
agents_dir = Path(__file__).resolve().parent.parent.parent


# ============================================================================
# ModelTeam CalibrationStage Tests
# ============================================================================

class TestModelTeamSandboxIntegration:
    """Tests for ModelTeam CalibrationStage sandbox integration."""

    @pytest.fixture
    def calibration_module(self):
        """Import the calibration stage module."""
        spec = importlib.util.spec_from_file_location(
            "calibration_stage",
            agents_dir / "ModelTeam" / "ael" / "ModeNoWcNoHITL" / "3-CalibrationStage.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_sandbox_import_exists(self, calibration_module):
        """CodeSandbox and ExecutionResult are imported."""
        assert hasattr(calibration_module, "CodeSandbox")
        assert hasattr(calibration_module, "ExecutionResult")

    def test_sandbox_validation_model_exists(self, calibration_module):
        """SandboxValidation Pydantic model is defined."""
        SandboxValidation = calibration_module.SandboxValidation
        assert hasattr(SandboxValidation, "model_fields")
        fields = set(SandboxValidation.model_fields.keys())
        assert "validated" in fields
        assert "validation_method" in fields
        assert "code" in fields
        assert "stdout" in fields
        assert "error" in fields
        assert "execution_time_sec" in fields

    def test_sandbox_validation_defaults(self, calibration_module):
        """SandboxValidation has correct default values."""
        sv = calibration_module.SandboxValidation(validated=True)
        assert sv.validated is True
        assert sv.validation_method == "sandbox_execution"
        assert sv.code == ""
        assert sv.stdout == ""
        assert sv.error == ""
        assert sv.execution_time_sec == 0.0

    def test_calibrated_model_has_sandbox_field(self, calibration_module):
        """CalibratedModel includes optional sandbox_validation field."""
        fields = calibration_module.CalibratedModel.model_fields
        assert "sandbox_validation" in fields
        # Should be Optional (default None)
        assert fields["sandbox_validation"].default is None

    def test_calibrator_has_validate_method(self, calibration_module):
        """Calibrator class has _validate_with_sandbox method."""
        assert hasattr(calibration_module.Calibrator, "_validate_with_sandbox")

    @patch("shared.llm.LLMClient")
    def test_validate_with_sandbox_no_params(self, mock_llm_cls, calibration_module):
        """_validate_with_sandbox returns failed validation when no parameters."""
        calibrator = calibration_module.Calibrator.__new__(calibration_module.Calibrator)
        calibrator.agent_name = "Calibrator"
        calibrator.llm = MagicMock()

        result = calibrator._validate_with_sandbox(
            MagicMock(),  # formal_model
            [],           # empty calibrated_parameters
            []            # empty empirical_targets
        )

        assert isinstance(result, calibration_module.SandboxValidation)
        assert result.validated is False
        assert "No calibrated parameters" in result.error

    @patch("shared.llm.LLMClient")
    def test_validate_with_sandbox_llm_error(self, mock_llm_cls, calibration_module):
        """_validate_with_sandbox handles LLM errors gracefully."""
        calibrator = calibration_module.Calibrator.__new__(calibration_module.Calibrator)
        calibrator.agent_name = "Calibrator"
        calibrator.llm = MagicMock()
        calibrator.llm.invoke.side_effect = Exception("LLM API error")

        param = calibration_module.CalibratedParameter(
            parameter_symbol="beta",
            parameter_name="Discount factor",
            calibrated_value=0.96,
            calibration_method="direct",
            literature_range="[0.95, 0.99]",
            justification="Standard",
            sensitivity="Medium"
        )
        formal_model = MagicMock()
        formal_model.model_title = "Test Model"
        formal_model.based_on_framework = "DSGE"
        formal_model.equations = []

        result = calibrator._validate_with_sandbox(
            formal_model,
            [param],
            []
        )

        assert isinstance(result, calibration_module.SandboxValidation)
        assert result.validated is False
        assert "Code generation failed" in result.error

    @patch("shared.llm.LLMClient")
    def test_validate_with_sandbox_success(self, mock_llm_cls, calibration_module):
        """_validate_with_sandbox executes safe code successfully."""
        calibrator = calibration_module.Calibrator.__new__(calibration_module.Calibrator)
        calibrator.agent_name = "Calibrator"
        calibrator.llm = MagicMock()
        # Return simple safe code
        calibrator.llm.invoke.return_value = 'import numpy as np\nimport json\nprint(json.dumps({"validation": "pass", "moments": [], "summary": "OK"}))'

        param = calibration_module.CalibratedParameter(
            parameter_symbol="beta",
            parameter_name="Discount factor",
            calibrated_value=0.96,
            calibration_method="direct",
            literature_range="[0.95, 0.99]",
            justification="Standard",
            sensitivity="Medium"
        )
        formal_model = MagicMock()
        formal_model.model_title = "Test Model"
        formal_model.based_on_framework = "DSGE"
        formal_model.equations = []

        result = calibrator._validate_with_sandbox(
            formal_model,
            [param],
            []
        )

        assert isinstance(result, calibration_module.SandboxValidation)
        assert result.validated is True
        assert result.execution_time_sec > 0
        assert "pass" in result.stdout

    @patch("shared.llm.LLMClient")
    def test_validate_with_sandbox_unsafe_code(self, mock_llm_cls, calibration_module):
        """_validate_with_sandbox rejects unsafe code."""
        calibrator = calibration_module.Calibrator.__new__(calibration_module.Calibrator)
        calibrator.agent_name = "Calibrator"
        calibrator.llm = MagicMock()
        # Return unsafe code (os import)
        calibrator.llm.invoke.return_value = 'import os\nos.system("echo hacked")'

        param = calibration_module.CalibratedParameter(
            parameter_symbol="beta",
            parameter_name="Discount factor",
            calibrated_value=0.96,
            calibration_method="direct",
            literature_range="[0.95, 0.99]",
            justification="Standard",
            sensitivity="Medium"
        )
        formal_model = MagicMock()
        formal_model.model_title = "Test Model"
        formal_model.based_on_framework = "DSGE"
        formal_model.equations = []

        result = calibrator._validate_with_sandbox(
            formal_model,
            [param],
            []
        )

        assert isinstance(result, calibration_module.SandboxValidation)
        assert result.validated is False
        # Should fail safety check — error starts with "Unsafe code:"
        assert result.error.startswith("Unsafe code:")

    @patch("shared.llm.LLMClient")
    def test_validate_strips_markdown_fences(self, mock_llm_cls, calibration_module):
        """_validate_with_sandbox strips markdown code fences from LLM output."""
        calibrator = calibration_module.Calibrator.__new__(calibration_module.Calibrator)
        calibrator.agent_name = "Calibrator"
        calibrator.llm = MagicMock()
        calibrator.llm.invoke.return_value = '```python\nimport json\nprint(json.dumps({"validation": "pass"}))\n```'

        param = calibration_module.CalibratedParameter(
            parameter_symbol="alpha",
            parameter_name="Capital share",
            calibrated_value=0.33,
            calibration_method="direct",
            literature_range="[0.30, 0.40]",
            justification="Standard",
            sensitivity="Low"
        )
        formal_model = MagicMock()
        formal_model.model_title = "Test"
        formal_model.based_on_framework = "RBC"
        formal_model.equations = []

        result = calibrator._validate_with_sandbox(formal_model, [param], [])
        assert result.validated is True

    def test_synthesize_accepts_sandbox_validation(self, calibration_module):
        """_synthesize_calibrated_model accepts sandbox_validation parameter."""
        import inspect
        sig = inspect.signature(calibration_module.Calibrator._synthesize_calibrated_model)
        assert "sandbox_validation" in sig.parameters

    def test_all_mode_variants_identical(self):
        """All 4 ModelTeam mode variants have same CalibrationStage."""
        import hashlib
        base = agents_dir / "ModelTeam" / "ael" / "ModeNoWcNoHITL" / "3-CalibrationStage.py"
        base_hash = hashlib.md5(base.read_bytes()).hexdigest()

        for mode in ["ModeNoWcWithHITL", "ModeWithWcNoHITL", "ModeWithWcWithHITL"]:
            path = agents_dir / "ModelTeam" / "ael" / mode / "3-CalibrationStage.py"
            assert hashlib.md5(path.read_bytes()).hexdigest() == base_hash, \
                f"{mode} differs from ModeNoWcNoHITL"


# ============================================================================
# DataTeam QualityAssuranceStage Tests
# ============================================================================

class TestDataTeamSandboxIntegration:
    """Tests for DataTeam QualityAssuranceStage sandbox integration."""

    @pytest.fixture
    def open_source_module(self):
        """Import the OpenSourceAPI QA stage module."""
        spec = importlib.util.spec_from_file_location(
            "qa_open_source",
            agents_dir / "DataTeam" / "ael" / "ModeOpenSourceAPI" / "3-QualityAssuranceStage.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    @pytest.fixture
    def premium_module(self):
        """Import the Premium QA stage module."""
        spec = importlib.util.spec_from_file_location(
            "qa_premium",
            agents_dir / "DataTeam" / "ael" / "ModePremiumSubscribed" / "3-QualityAssuranceStage.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    @pytest.fixture
    def user_uploaded_module(self):
        """Import the UserUploaded QA stage module."""
        spec = importlib.util.spec_from_file_location(
            "qa_user_uploaded",
            agents_dir / "DataTeam" / "ael" / "ModeUserUploaded" / "3-QualityAssuranceStage.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_sandbox_import_open_source(self, open_source_module):
        """OpenSourceAPI variant imports CodeSandbox."""
        assert hasattr(open_source_module, "CodeSandbox")
        assert hasattr(open_source_module, "ExecutionResult")

    def test_sandbox_import_premium(self, premium_module):
        """Premium variant imports CodeSandbox."""
        assert hasattr(premium_module, "CodeSandbox")
        assert hasattr(premium_module, "ExecutionResult")

    def test_sandbox_import_user_uploaded(self, user_uploaded_module):
        """UserUploaded variant imports CodeSandbox."""
        assert hasattr(user_uploaded_module, "CodeSandbox")
        assert hasattr(user_uploaded_module, "ExecutionResult")

    def test_orchestrator_has_validate_method_open_source(self, open_source_module):
        """OpenSourceAPI orchestrator has _validate_data_with_sandbox."""
        assert hasattr(open_source_module.QualityAssuranceOrchestrator, "_validate_data_with_sandbox")

    def test_orchestrator_has_validate_method_premium(self, premium_module):
        """Premium orchestrator has _validate_data_with_sandbox."""
        assert hasattr(premium_module.QualityAssuranceOrchestrator, "_validate_data_with_sandbox")

    def test_orchestrator_has_validate_method_user_uploaded(self, user_uploaded_module):
        """UserUploaded orchestrator has _validate_data_with_sandbox."""
        assert hasattr(user_uploaded_module.QualityAssuranceOrchestrator, "_validate_data_with_sandbox")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_validate_data_open_source_success(self, open_source_module):
        """OpenSourceAPI sandbox validation executes safe code."""
        with patch.object(open_source_module.LLMClient, "invoke",
                          return_value='import json\nprint(json.dumps({"validated": true, "statistics": [], "issues": [], "summary": "OK"}))'):
            orch = open_source_module.QualityAssuranceOrchestrator(openai_api_key="test-key")
            dataset = open_source_module.IntegratedDataset(
                dataset_id="TEST",
                dataset_name="Test Dataset",
                source_apis=["FRED"],
                num_variables=3,
                num_observations=100,
                time_period="2020-2023",
                frequency="quarterly",
                merge_strategy="outer_join",
                variables=["date", "GDP", "CPI"],
                data_preview=[{"date": "2023-01-01", "GDP": 22000, "CPI": 300}],
                provenance={"GDP": {"source": "FRED"}}
            )
            result = orch._validate_data_with_sandbox(dataset, [])

        assert isinstance(result, dict)
        assert "validated" in result
        assert "validation_method" in result
        assert result["validation_method"] == "sandbox_execution"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_validate_data_open_source_llm_error(self, open_source_module):
        """OpenSourceAPI handles LLM errors gracefully."""
        with patch.object(open_source_module.LLMClient, "invoke",
                          side_effect=Exception("API error")):
            orch = open_source_module.QualityAssuranceOrchestrator(openai_api_key="test-key")
            dataset = open_source_module.IntegratedDataset(
                dataset_id="TEST",
                dataset_name="Test",
                source_apis=["FRED"],
                num_variables=2,
                num_observations=50,
                time_period="2020-2023",
                frequency="quarterly",
                merge_strategy="outer_join",
                variables=["date", "GDP"],
                data_preview=[],
                provenance={}
            )
            result = orch._validate_data_with_sandbox(dataset, [])

        assert result["validated"] is False
        assert "Code generation failed" in result["error"]

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_validate_data_open_source_unsafe_code(self, open_source_module):
        """OpenSourceAPI rejects unsafe sandbox code."""
        with patch.object(open_source_module.LLMClient, "invoke",
                          return_value='import os\nprint(os.listdir("."))'):
            orch = open_source_module.QualityAssuranceOrchestrator(openai_api_key="test-key")
            dataset = open_source_module.IntegratedDataset(
                dataset_id="TEST",
                dataset_name="Test",
                source_apis=["FRED"],
                num_variables=2,
                num_observations=50,
                time_period="2020-2023",
                frequency="quarterly",
                merge_strategy="outer_join",
                variables=["date", "GDP"],
                data_preview=[],
                provenance={}
            )
            result = orch._validate_data_with_sandbox(dataset, [])

        assert result["validated"] is False


# ============================================================================
# CodeSandbox.clean_llm_output Tests
# ============================================================================

class TestCleanLlmOutput:
    """Tests for CodeSandbox.clean_llm_output static method."""

    def test_plain_code_unchanged(self):
        from shared.tools.sandbox_tool import CodeSandbox
        code = "import json\nprint(json.dumps({'a': 1}))"
        assert CodeSandbox.clean_llm_output(code) == code

    def test_strips_python_fence(self):
        from shared.tools.sandbox_tool import CodeSandbox
        raw = "```python\nimport json\nprint('hello')\n```"
        assert CodeSandbox.clean_llm_output(raw) == "import json\nprint('hello')"

    def test_strips_bare_fence(self):
        from shared.tools.sandbox_tool import CodeSandbox
        raw = "```\nprint(42)\n```"
        assert CodeSandbox.clean_llm_output(raw) == "print(42)"

    def test_strips_with_whitespace(self):
        from shared.tools.sandbox_tool import CodeSandbox
        raw = "  \n```python\nx = 1\n```\n  "
        assert CodeSandbox.clean_llm_output(raw) == "x = 1"

    def test_no_fence_language_leak(self):
        from shared.tools.sandbox_tool import CodeSandbox
        raw = "```python\nprint(1)\n```"
        result = CodeSandbox.clean_llm_output(raw)
        assert not result.startswith("python")


# ============================================================================
# Sandbox Environment Security Tests
# ============================================================================

class TestSandboxEnvironmentSecurity:
    """Tests for sandbox subprocess environment isolation."""

    def test_env_does_not_leak_via_blocked_import(self):
        from shared.tools.sandbox_tool import CodeSandbox
        sandbox = CodeSandbox(timeout_sec=10)
        code = "import os\nprint(os.environ.get('OPENAI_API_KEY', 'none'))"
        result = sandbox.execute(code)
        assert not result.success
        assert result.error.startswith("Unsafe code:")

    def test_minimal_env_passes_safe_code(self):
        from shared.tools.sandbox_tool import CodeSandbox
        sandbox = CodeSandbox(timeout_sec=10)
        result = sandbox.execute("import json\nprint(json.dumps({'status': 'ok'}))")
        assert result.success
        assert "ok" in result.stdout


# ============================================================================
# Premium & UserUploaded Functional Tests
# ============================================================================

class TestPremiumAndUserUploadedFunctional:
    """Functional tests for Premium and UserUploaded sandbox validation."""

    @pytest.fixture
    def premium_module(self):
        spec = importlib.util.spec_from_file_location(
            "qa_premium_func",
            agents_dir / "DataTeam" / "ael" / "ModePremiumSubscribed" / "3-QualityAssuranceStage.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    @pytest.fixture
    def user_uploaded_module(self):
        spec = importlib.util.spec_from_file_location(
            "qa_user_uploaded_func",
            agents_dir / "DataTeam" / "ael" / "ModeUserUploaded" / "3-QualityAssuranceStage.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_premium_sandbox_success(self, premium_module):
        """Premium variant sandbox validation runs safe code."""
        with patch.object(premium_module.LLMClient, "invoke",
                          return_value='import json\nprint(json.dumps({"validated": true, "field_checks": [], "issues": [], "summary": "OK"}))'):
            orch = premium_module.QualityAssuranceOrchestrator(openai_api_key="test-key")
            standardization = MagicMock()
            standardization.unified_fields = ["price", "volume"]
            standardization.vendors_merged = ["Bloomberg"]
            standardization.final_record_count = 1000
            result = orch._validate_data_with_sandbox(standardization, {})

        assert isinstance(result, dict)
        assert result["validation_method"] == "sandbox_execution"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_premium_sandbox_llm_error(self, premium_module):
        """Premium variant handles LLM errors gracefully."""
        with patch.object(premium_module.LLMClient, "invoke",
                          side_effect=Exception("API error")):
            orch = premium_module.QualityAssuranceOrchestrator(openai_api_key="test-key")
            standardization = MagicMock()
            standardization.unified_fields = []
            standardization.vendors_merged = []
            standardization.final_record_count = 0
            result = orch._validate_data_with_sandbox(standardization, {})

        assert result["validated"] is False
        assert "Code generation failed" in result["error"]

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_user_uploaded_sandbox_success(self, user_uploaded_module):
        """UserUploaded variant sandbox validation runs safe code."""
        with patch.object(user_uploaded_module.LLMClient, "invoke",
                          return_value='import json\nprint(json.dumps({"validated": true, "column_checks": [], "issues": [], "summary": "OK"}))'):
            orch = user_uploaded_module.QualityAssuranceOrchestrator(openai_api_key="test-key")
            data_cleaning = {"file_info": {"column_names": ["date", "value"], "num_rows": 100, "file_extension": "CSV"}}
            result = orch._validate_data_with_sandbox(data_cleaning)

        assert isinstance(result, dict)
        assert result["validation_method"] == "sandbox_execution"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_user_uploaded_sandbox_is_llm_independent(self, user_uploaded_module):
        """UserUploaded sandbox validation is now DETERMINISTIC / LLM-independent.

        It authors a fixed, sandbox-safe script instead of asking the LLM to generate the
        validation code (which previously emitted blocked open()/os/sys -> validated=False
        on every run). So an LLM error during the run must NOT break validation, and the old
        'Code generation failed' path is gone.
        """
        with patch.object(user_uploaded_module.LLMClient, "invoke",
                          side_effect=Exception("API error")):
            orch = user_uploaded_module.QualityAssuranceOrchestrator(openai_api_key="test-key")
            data_cleaning = {"file_info": {"column_names": [], "num_rows": 0}}
            result = orch._validate_data_with_sandbox(data_cleaning)

        # validation ran independently of the failing LLM (no codegen)
        assert isinstance(result["validated"], bool)
        assert "Code generation failed" not in (result.get("error") or "")
