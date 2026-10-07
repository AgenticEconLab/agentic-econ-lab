# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Schema Validator — Enforces Pydantic schemas on inter-stage data.

Prevents malformed data from propagating through the AEL pipeline.
Each team/stage pair maps to a canonical Pydantic output model defined
in {Team}/ael/schemas/stage_outputs.py.

Usage:
    from shared.guardrails import SchemaValidator

    validator = SchemaValidator()
    result = validator.validate("IdeationTeam", "SourcingStage", raw_data)
    if result.valid:
        validated_data = result.data
    else:
        print(result.errors)
"""

from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, ValidationError


class ValidationResult(BaseModel):
    """Result of a schema validation attempt."""
    valid: bool
    data: Optional[Dict] = None
    errors: Optional[List[Dict]] = None
    team: str = ""
    stage: str = ""


class SchemaValidator:
    """
    Enforces Pydantic schemas on all inter-stage data.

    The SCHEMAS dict maps (team, stage) to the canonical output model.
    Validation is strict by default: extra fields are ignored but missing
    required fields cause validation failure.
    """

    # Lazy-loaded schema registry to avoid circular imports at module level.
    _schemas: Optional[Dict[str, Dict[str, Type[BaseModel]]]] = None

    @classmethod
    def _load_schemas(cls) -> Dict[str, Dict[str, Type[BaseModel]]]:
        """Load schema classes on first use."""
        if cls._schemas is not None:
            return cls._schemas

        # Import canonical output schemas from each team
        from IdeationTeam.ael.schemas.stage_outputs import (
            SourcingStageOutput,
            RefinementStageOutput,
            IntegrationStageOutput,
        )
        from LiteratureTeam.ael.schemas.stage_outputs import (
            GatheringStageOutput,
            GapDetectionStageOutput,
            SynthesisStageOutput,
        )
        from ModelTeam.ael.schemas.stage_outputs import (
            TheoryStageOutput,
            ModelDesignStageOutput,
            CalibrationStageOutput,
        )
        from DataTeam.ael.schemas.stage_outputs import (
            DataSourceStageOutput,
            DataCleaningStageOutput,
            QualityAssuranceStageOutput,
        )

        cls._schemas = {
            "IdeationTeam": {
                "SourcingStage": SourcingStageOutput,
                "RefinementStage": RefinementStageOutput,
                "IntegrationStage": IntegrationStageOutput,
            },
            "LiteratureTeam": {
                "LiteratureGatheringStage": GatheringStageOutput,
                "GapDetectionStage": GapDetectionStageOutput,
                "SynthesisStage": SynthesisStageOutput,
            },
            "ModelTeam": {
                "TheoryStage": TheoryStageOutput,
                "ModelDesignStage": ModelDesignStageOutput,
                "CalibrationStage": CalibrationStageOutput,
            },
            "DataTeam": {
                "DataSourceStage": DataSourceStageOutput,
                "DataCleaningStage": DataCleaningStageOutput,
                "QualityAssuranceStage": QualityAssuranceStageOutput,
            },
        }
        return cls._schemas

    @classmethod
    def get_schema(cls, team: str, stage: str) -> Optional[Type[BaseModel]]:
        """Look up the canonical schema for a team/stage pair."""
        schemas = cls._load_schemas()
        team_schemas = schemas.get(team)
        if team_schemas is None:
            return None
        return team_schemas.get(stage)

    @classmethod
    def list_schemas(cls) -> Dict[str, List[str]]:
        """List all registered team/stage pairs."""
        schemas = cls._load_schemas()
        return {team: list(stages.keys()) for team, stages in schemas.items()}

    @classmethod
    def validate(
        cls,
        team: str,
        stage: str,
        data: Any,
        strict: bool = False,
    ) -> ValidationResult:
        """
        Validate stage output data against its canonical schema.

        Args:
            team: Team name (e.g., "IdeationTeam").
            stage: Stage name (e.g., "SourcingStage").
            data: Raw data dict to validate.
            strict: If True, extra fields cause an error.

        Returns:
            ValidationResult with valid=True and validated data, or
            valid=False and error details.
        """
        schema = cls.get_schema(team, stage)
        if schema is None:
            return ValidationResult(
                valid=False,
                team=team,
                stage=stage,
                errors=[{
                    "type": "schema_not_found",
                    "msg": f"No schema registered for {team}/{stage}",
                }],
            )

        try:
            if strict:
                validated = schema.model_validate(data, strict=True)
            else:
                validated = schema.model_validate(data)
            return ValidationResult(
                valid=True,
                data=validated.model_dump(),
                team=team,
                stage=stage,
            )
        except ValidationError as e:
            return ValidationResult(
                valid=False,
                team=team,
                stage=stage,
                errors=[err for err in e.errors()],
            )

    @classmethod
    def validate_or_raise(
        cls,
        team: str,
        stage: str,
        data: Any,
    ) -> Dict:
        """
        Validate and return the data dict, or raise ValidationError.

        Convenience method for pipeline code that should halt on invalid data.
        """
        result = cls.validate(team, stage, data)
        if not result.valid:
            compact = cls._compact_errors(result.errors or [])
            raise ValueError(
                f"Schema validation failed for {team}/{stage}: {compact}"
            )
        return result.data

    @staticmethod
    def _compact_errors(errors: List[Dict]) -> str:
        """Render pydantic errors without dumping the full failing input."""
        parts = []
        for err in errors[:5]:
            loc = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
            msg = err.get("msg", "")
            etype = err.get("type", "")
            parts.append(f"{loc}: {msg} ({etype})")
        if len(errors) > 5:
            parts.append(f"... and {len(errors) - 5} more")
        return "; ".join(parts)

    @classmethod
    def register_schema(
        cls,
        team: str,
        stage: str,
        schema: Type[BaseModel],
    ):
        """
        Register or override a schema at runtime.

        Useful for testing or for adding custom stages.
        """
        schemas = cls._load_schemas()
        if team not in schemas:
            schemas[team] = {}
        schemas[team][stage] = schema

    @classmethod
    def reset(cls):
        """Reset schema cache. Forces re-import on next use. For testing."""
        cls._schemas = None
