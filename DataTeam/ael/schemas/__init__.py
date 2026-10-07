# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
DataTeam AEL Schemas — Canonical Pydantic models for stage inputs/outputs.
"""

from DataTeam.ael.schemas.stage_inputs import (
    DataSourceStageInput,
    DataCleaningStageInput,
    QualityAssuranceStageInput,
)
from DataTeam.ael.schemas.stage_outputs import (
    # Open Source API — Stage 1
    DataRequirement,
    APISource,
    DataSeries,
    RetrievedData,
    DataSourceStageOutput,
    # Open Source API — Stage 2
    QualityDimension,
    QualityAssessment,
    AlignedSeries,
    IntegratedDataset,
    DataCleaningStageOutput,
    # Open Source API — Stage 3
    VariableEntry,
    DataCodebook,
    DataReport,
    DocumentedDataset,
    QualityAssuranceStageOutput,
    # Backward-compatible aliases
    DataSourceOutput,
    DataCleaningOutput,
    QualityAssuranceOutput,
    # Premium Subscribed — Stage 1
    VendorCredential,
    CredentialValidation,
    VendorCostEstimate,
    CostEstimation,
    # Premium Subscribed — Stage 2
    OptimizedQuery,
    QueryOptimization,
    RetrievedDataset,
    DataRetrieval,
    VendorQualityCheck,
    LicenseRestriction,
    LicenseCompliance,
    PremiumQualityAssessment,
    PremiumDataCleaningOutput,
    # Premium Subscribed — Stage 3
    IdentifierMapping,
    FieldMapping,
    DataStandardization,
    PremiumVariableEntry,
    PremiumDataCodebook,
    PremiumDataReport,
    PremiumDocumentedDataset,
    PremiumQualityAssuranceOutput,
    # User Uploaded — Stage 1
    FileInfo,
    UserUploadedQualityDimension,
    UserUploadedQualityAssessment,
    UserUploadedDataRequirement,
    UserUploadedDataSourceOutput,
    # User Uploaded — Stage 2
    PIIDetection,
    PrivacyScreening,
    StructureInference,
    ResearchAlignment,
    Transformation,
    TransformationResults,
    UserUploadedDataCleaningOutput,
    # User Uploaded — Stage 3
    UserUploadedVariableEntry,
    UserUploadedDataCodebook,
    UserUploadedDataReport,
    UserUploadedDocumentedDataset,
    UserUploadedQualityAssuranceOutput,
)

__all__ = [
    # Inputs
    "DataSourceStageInput",
    "DataCleaningStageInput",
    "QualityAssuranceStageInput",
    # Open Source API — Stage 1
    "DataRequirement",
    "APISource",
    "DataSeries",
    "RetrievedData",
    "DataSourceStageOutput",
    # Open Source API — Stage 2
    "QualityDimension",
    "QualityAssessment",
    "AlignedSeries",
    "IntegratedDataset",
    "DataCleaningStageOutput",
    # Open Source API — Stage 3
    "VariableEntry",
    "DataCodebook",
    "DataReport",
    "DocumentedDataset",
    "QualityAssuranceStageOutput",
    # Backward-compatible aliases
    "DataSourceOutput",
    "DataCleaningOutput",
    "QualityAssuranceOutput",
    # Premium Subscribed — Stage 1
    "VendorCredential",
    "CredentialValidation",
    "VendorCostEstimate",
    "CostEstimation",
    # Premium Subscribed — Stage 2
    "OptimizedQuery",
    "QueryOptimization",
    "RetrievedDataset",
    "DataRetrieval",
    "VendorQualityCheck",
    "LicenseRestriction",
    "LicenseCompliance",
    "PremiumQualityAssessment",
    "PremiumDataCleaningOutput",
    # Premium Subscribed — Stage 3
    "IdentifierMapping",
    "FieldMapping",
    "DataStandardization",
    "PremiumVariableEntry",
    "PremiumDataCodebook",
    "PremiumDataReport",
    "PremiumDocumentedDataset",
    "PremiumQualityAssuranceOutput",
    # User Uploaded — Stage 1
    "FileInfo",
    "UserUploadedQualityDimension",
    "UserUploadedQualityAssessment",
    "UserUploadedDataRequirement",
    "UserUploadedDataSourceOutput",
    # User Uploaded — Stage 2
    "PIIDetection",
    "PrivacyScreening",
    "StructureInference",
    "ResearchAlignment",
    "Transformation",
    "TransformationResults",
    "UserUploadedDataCleaningOutput",
    # User Uploaded — Stage 3
    "UserUploadedVariableEntry",
    "UserUploadedDataCodebook",
    "UserUploadedDataReport",
    "UserUploadedDocumentedDataset",
    "UserUploadedQualityAssuranceOutput",
]
