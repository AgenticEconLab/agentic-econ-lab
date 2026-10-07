# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Pipeline package — Cross-team research pipeline orchestration.

Provides:
- ArtifactStore: Typed artifact persistence across teams
- ResearchPipelineOrchestrator: Central pipeline orchestrator
- PipelineConfig: YAML-based pipeline configuration
"""

from pipeline.artifact_store import ArtifactStore, Artifact
from pipeline.research_pipeline import ResearchPipelineOrchestrator

__all__ = [
    "ArtifactStore",
    "Artifact",
    "ResearchPipelineOrchestrator",
]
