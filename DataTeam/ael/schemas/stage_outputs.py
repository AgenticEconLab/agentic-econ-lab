# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
DataTeam stage output schemas.

Defines the validated output contracts for each stage in the Data pipeline.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ============================================================================
# Stage 1: Data Source — Output Models
# ============================================================================

class DataRequirement(BaseModel):
    """A data requirement for the research question."""
    requirement_id: str = Field(description="Unique requirement identifier")
    variable_name: str = Field(description="Name of variable/indicator needed")
    description: str = Field(description="Description of what data is needed")
    frequency: str = Field(description="Required frequency: annual, quarterly, monthly, daily")
    time_period: str = Field(description="Required time period (e.g., '1990-2023')")
    geographic_coverage: str = Field(description="Geographic coverage (e.g., 'US', 'Global')")
    unit_of_measurement: str = Field(description="Unit of measurement")
    priority: str = Field(description="High/Medium/Low priority")
    suggested_sources: List[str] = Field(default_factory=list, description="Suggested API sources")


class APISource(BaseModel):
    """An API data source discovered by agents."""
    source_id: str = Field(description="Unique source identifier")
    source_name: str = Field(description="Name of API source (FRED, Yahoo Finance, etc.)")
    source_type: str = Field(description="Type: central_bank, financial, international, government")
    url: str = Field(description="API URL or access point")
    description: str = Field(description="Description of the source")
    coverage: str = Field(description="Geographic and temporal coverage")
    data_quality: str = Field(description="High/Medium/Low quality assessment")
    requires_api_key: bool = Field(description="Whether API key is required")
    api_key_available: bool = Field(description="Whether API key is configured")
    update_frequency: str = Field(description="How often data is updated")
    series_available: List[str] = Field(description="Key series/indicators available")
    series_count: int = Field(default=0, description="Number of series retrieved")
    research_alignment: str = Field(default="", description="Why appropriate for the question")


class DataSeries(BaseModel):
    """A data series from an API."""
    series_id: str = Field(description="Unique series identifier (e.g., FRED series ID)")
    series_name: str = Field(description="Human-readable series name")
    source_name: str = Field(description="Source API name")
    description: str = Field(description="Series description")
    frequency: str = Field(description="Data frequency: daily, weekly, monthly, quarterly, annual")
    units: str = Field(description="Units of measurement")
    seasonal_adjustment: str = Field(description="Seasonally adjusted or not")
    start_date: str = Field(description="Start date of available data")
    end_date: str = Field(description="End date of available data")
    last_updated: str = Field(description="Last update date")
    variable_role: str = Field(description="Role: outcome, predictor, control")


class RetrievedData(BaseModel):
    """Retrieved data from an API."""
    series_id: str = Field(description="Series identifier")
    series_name: str = Field(description="Series name")
    source_name: str = Field(description="Source API name")
    # The source's OWN title for the series, and
    # the requested variable it stands in for when that title differs materially.
    source_title: Optional[str] = Field(default=None, description="The source's own series title")
    proxy_for: Optional[str] = Field(
        default=None,
        description="Requested variable this series is a proxy for (set when the source title "
                    "differs materially from the requested name)")
    retrieval_date: str = Field(description="Date data was retrieved")
    num_observations: int = Field(description="Number of observations retrieved")
    start_date: str = Field(description="Start date of retrieved data")
    end_date: str = Field(description="End date of retrieved data")
    frequency: str = Field(description="Data frequency")
    data_preview: List[Dict] = Field(description="First few rows as preview")
    quality_notes: str = Field(description="Initial quality assessment notes")
    retrieval_status: str = Field(description="Success/Failed/Partial/Simulated")
    data_simulated: bool = Field(
        default=False,
        description="True when the values are synthetic/simulated (no live API call "
                    "succeeded). Downstream cleaning/QA must propagate this flag and must "
                    "NOT certify simulated data as Gold/real.")
    # When the id-repair ladder resolved a different id/source than requested, the
    # CANONICAL fetch coordinates are first-class fields (so the estimation loader does not
    # have to parse them out of quality_notes prose).
    fetch_id: Optional[str] = Field(
        default=None, description="Canonical id actually fetched (None = series_id as-is)")
    fetch_source: Optional[str] = Field(
        default=None, description="Source actually fetched from (None = source_name as-is)")
    # Deterministic facts about what was actually retrieved
    title_check: Optional[str] = Field(
        default=None,
        description="Concept check of source_title vs the requested name: match / mismatch / "
                    "unverified (no informative title — disclosed, not a match)")
    title_check_reason: str = Field(default="", description="Why the check did not match")
    declared_frequency: Optional[str] = Field(
        default=None, description="Frequency the selection step declared (LLM label); "
                                  "`frequency` is inferred from the observation dates")
    discontinued: bool = Field(
        default=False, description="Last observation predates the requested sample end by "
                                   "more than the frequency's publication-lag allowance")
    projections_excluded: int = Field(
        default=0, description="Observations after the retrieval/vintage date dropped as "
                               "projections")


class DataSourceStageOutput(BaseModel):
    """Complete output of Stage 1: Data Source."""
    research_question: str = Field(description="Research question")
    data_requirements: List[DataRequirement] = Field(description="Data requirements")
    discovered_sources: List[APISource] = Field(description="Discovered API sources")
    selected_series: List[DataSeries] = Field(description="Selected data series")
    retrieved_data: List[RetrievedData] = Field(description="Retrieved data")
    cross_domain_opportunities: List[str] = Field(
        default_factory=list, description="Cross-domain data opportunities")
    research_question_alignment: str = Field(
        default="", description="Source-question alignment explanation")
    assumptions: List[str] = Field(
        default_factory=list, description="Data source selection assumptions")
    limitations: List[str] = Field(
        default_factory=list, description="Limitations of selected data")
    metadata: Dict = Field(default_factory=dict, description="Output metadata")


# ============================================================================
# Stage 2: Data Cleaning — Output Models
# ============================================================================

class QualityDimension(BaseModel):
    """A quality dimension assessment for a data series."""
    dimension: str = Field(description="Dimension: completeness, accuracy, consistency, timeliness")
    score: float = Field(description="Score 0-100")
    issues_found: List[str] = Field(description="Issues found")
    recommendations: List[str] = Field(description="Recommendations")


class QualityAssessment(BaseModel):
    """Quality assessment of a data series."""
    series_id: str = Field(description="Series identifier")
    series_name: str = Field(description="Series name")
    source_name: str = Field(description="Source API name")
    dimensions: List[QualityDimension] = Field(description="Quality dimensions")
    overall_score: float = Field(description="Overall quality score 0-100")
    quality_grade: str = Field(description="A/B/C/D/F grade")
    actionable_issues: List[str] = Field(description="Issues requiring action")
    # The verdict and score come from deterministic checks on the observations; an LLM
    # may only narrate them
    checks: Dict = Field(default_factory=dict, description="Deterministic check results")
    verdict: str = Field(default="", description="pass / fail from the deterministic checks")
    narrative: str = Field(default="", description="LLM narration of the checks (no scoring)")


class AlignedSeries(BaseModel):
    """A temporally aligned data series."""
    series_id: str = Field(description="Series identifier")
    series_name: str = Field(description="Series name")
    source_name: str = Field(description="Source API name")
    original_frequency: str = Field(description="Original frequency")
    target_frequency: str = Field(description="Target frequency after alignment")
    conversion_method: str = Field(description="Method used for conversion")
    num_observations: int = Field(description="Number of observations after alignment")
    start_date: str = Field(description="Start date after alignment")
    end_date: str = Field(description="End date after alignment")
    data_preview: List[Dict] = Field(description="Preview of aligned data")
    alignment_notes: str = Field(description="Notes on alignment process")


class IntegratedDataset(BaseModel):
    """An integrated multi-source dataset."""
    dataset_id: str = Field(description="Dataset identifier")
    dataset_name: str = Field(description="Dataset name")
    source_apis: List[str] = Field(description="Source APIs used")
    num_variables: int = Field(description="Number of variables")
    num_observations: int = Field(description="Number of observations")
    time_period: str = Field(description="Time period covered")
    frequency: str = Field(description="Data frequency")
    merge_strategy: str = Field(description="Merge strategy used")
    variables: List[str] = Field(description="Variable names")
    data_preview: List[Dict] = Field(description="Preview of integrated data")
    provenance: Dict = Field(description="Source provenance tracking")
    alignment: Dict = Field(
        default_factory=dict,
        description="How the merge was computed (target frequency, aggregation rule, "
                    "outer/complete-case row counts, excluded series, merged-data file)")


class DataCleaningStageOutput(BaseModel):
    """Complete output of Stage 2: Data Cleaning."""
    retrieved_data: List[RetrievedData] = Field(description="Input retrieved data")
    quality_assessments: List[QualityAssessment] = Field(description="Quality assessments")
    aligned_datasets: List[AlignedSeries] = Field(description="Aligned data series")
    integrated_datasets: List[IntegratedDataset] = Field(description="Integrated datasets")
    metadata: Dict = Field(default_factory=dict, description="Output metadata")


# ============================================================================
# Stage 3: Quality Assurance — Output Models
# ============================================================================

class VariableEntry(BaseModel):
    """A variable entry in the codebook."""
    variable_name: str = Field(description="Variable name")
    description: str = Field(description="Variable description")
    source: str = Field(description="Source API")
    units: str = Field(description="Units of measurement")
    frequency: str = Field(description="Data frequency")
    time_coverage: str = Field(description="Time period covered")
    transformations: str = Field(description="Transformations applied")
    notes: str = Field(description="Additional notes")


class DataCodebook(BaseModel):
    """A data codebook documenting the dataset."""
    title: str = Field(description="Codebook title")
    version: str = Field(description="Version number")
    created_date: str = Field(description="Creation date")
    variables: List[VariableEntry] = Field(description="Variable entries")
    summary_statistics: Dict = Field(description="Summary statistics")
    data_sources: List[Dict] = Field(description="Data source information")
    methodology: str = Field(description="Methodology description")
    methodology_narrative: str = Field(
        default="",
        description="LLM-generated narrative explaining source/series choices, the "
                    "frequency-alignment method and its consequences, and how to "
                    "interpret the QA scores")
    limitations: List[str] = Field(description="Known limitations")
    usage_notes: str = Field(description="Usage notes")


class DataReport(BaseModel):
    """A data acquisition report."""
    title: str = Field(description="Report title")
    research_question: str = Field(description="Research question")
    executive_summary: str = Field(description="Executive summary")
    api_sources_section: str = Field(description="API sources used section")
    series_retrieval_section: str = Field(description="Series retrieval section")
    quality_assessment_section: str = Field(description="Quality assessment section")
    temporal_alignment_section: str = Field(description="Temporal alignment section")
    integration_section: str = Field(description="Multi-source integration section")
    transformations_section: str = Field(description="Final transformations section")
    citations: List[str] = Field(description="Proper citations for data sources")


class DocumentedDataset(BaseModel):
    """A fully documented dataset with codebook and report."""
    dataset_id: str = Field(description="Dataset identifier")
    dataset_name: str = Field(description="Dataset name")
    integrated_dataset: IntegratedDataset = Field(description="Integrated dataset")
    codebook: DataCodebook = Field(description="Data codebook")
    report: DataReport = Field(description="Data acquisition report")
    quality_score: float = Field(description="Overall quality score")
    certification_level: str = Field(description="Bronze/Silver/Gold certification")
    citation: str = Field(description="How to cite this dataset")
    data_simulated: bool = Field(
        default=False,
        description="True when the underlying data is simulated (no live API). When True the "
                    "certification is capped (never Gold) and the score is held below 95.")
    documentation: Dict = Field(description="Additional documentation")


class QualityAssuranceStageOutput(BaseModel):
    """Complete output of Stage 3: Quality Assurance."""
    integrated_datasets: List[IntegratedDataset] = Field(description="Input integrated datasets")
    quality_assessments: List[QualityAssessment] = Field(description="Quality assessments")
    documented_datasets: List[DocumentedDataset] = Field(description="Documented datasets")
    metadata: Dict = Field(default_factory=dict, description="Output metadata")
    replication_manifest: Optional[Dict] = Field(
        default=None,
        description="Archivist output: environment spec, pipeline code fingerprint, "
                    "per-dataset content hash, and consolidated audit trail, enabling "
                    "exact reproduction independent of the run that produced the dataset.")


# Backward-compatible aliases (inline stage files used shorter names)
DataSourceOutput = DataSourceStageOutput
DataCleaningOutput = DataCleaningStageOutput
QualityAssuranceOutput = QualityAssuranceStageOutput


# ============================================================================
# ModePremiumSubscribed — Stage 1: Credential / Cost Models
# ============================================================================

class VendorCredential(BaseModel):
    """Model for vendor credential status."""
    vendor: str = Field(description="Vendor name: Bloomberg, Refinitiv, WRDS")
    available: bool = Field(description="Whether credentials are available")
    status: str = Field(description="Status: Configured, Not configured, Expired")
    subscription_tier: str = Field(description="Subscription tier if known")
    data_entitlements: List[str] = Field(description="Data entitlements")
    notes: str = Field(description="Additional notes")
    research_alignment: str = Field(description="Why this vendor is appropriate for the research question", default="")


class CredentialValidation(BaseModel):
    """Model for credential validation results."""
    validation_timestamp: str = Field(description="Validation timestamp")
    vendors: List[VendorCredential] = Field(description="Vendor credentials")
    accessible_sources: List[str] = Field(description="List of accessible sources")
    recommendations: List[str] = Field(description="Recommendations")


class VendorCostEstimate(BaseModel):
    """Model for vendor-specific cost estimate."""
    vendor: str = Field(description="Vendor name")
    data_type: str = Field(description="Type of data")
    estimated_records: int = Field(description="Estimated number of records")
    estimated_api_calls: int = Field(description="Estimated API calls")
    cost_per_unit: float = Field(description="Cost per unit (call/record)")
    estimated_cost: float = Field(description="Total estimated cost")
    cost_breakdown: Dict = Field(description="Detailed cost breakdown")
    notes: str = Field(description="Cost notes")


class CostEstimation(BaseModel):
    """Model for cost estimation results."""
    research_question: str = Field(description="Research question")
    budget_limit: Optional[float] = Field(description="Budget limit if specified")
    vendor_estimates: List[VendorCostEstimate] = Field(description="Per-vendor estimates")
    total_estimated_cost: float = Field(description="Total estimated cost")
    total_actual_cost: float = Field(default=0, description="Total actual cost after retrieval")
    cost_saving_recommendations: List[str] = Field(description="Cost saving recommendations")
    within_budget: bool = Field(description="Whether within budget")


class PremiumDataSourceOutput(BaseModel):
    """Stage 1 output for ModePremiumSubscribed (credential validation + cost estimation).

    Distinct from ``DataSourceStageOutput`` (open-source API mode): premium Stage 1
    produces credential/cost artifacts, not retrieved data series, so it must NOT be
    built from the open-source schema (doing so omitted the 4 required open-source
    fields and crashed every rep with a ValidationError). ``status``/``data_available``
    let the pipeline emit a clean N/A result (e.g. "N/A (no premium terminal)") when no
    Bloomberg/Refinitiv/WRDS access exists, instead of crashing.
    """
    research_question: str = Field(description="Research question")
    status: str = Field(
        default="ok",
        description="Pipeline status, e.g. 'ok' or 'N/A (no premium terminal: "
                    "Bloomberg/Refinitiv/WRDS not accessible)'")
    data_available: bool = Field(
        default=True,
        description="Whether at least one premium data source is actually accessible")
    budget_limit: Optional[float] = Field(
        default=None, description="Budget limit if specified")
    credential_validation: CredentialValidation = Field(
        description="Credential validation results")
    cost_estimation: CostEstimation = Field(description="Cost estimation results")
    cross_domain_opportunities: List[str] = Field(
        default_factory=list, description="Cross-domain data opportunities")
    research_question_alignment: str = Field(
        default="", description="Source-question alignment explanation")
    assumptions: List[str] = Field(
        default_factory=list, description="Assumptions underlying premium data use")
    limitations: List[str] = Field(
        default_factory=list, description="Limitations of premium data access")
    metadata: Dict = Field(default_factory=dict, description="Output metadata")


# ============================================================================
# ModePremiumSubscribed — Stage 2: Query / Retrieval / License Models
# ============================================================================

class OptimizedQuery(BaseModel):
    """Model for an optimized query."""
    query_id: str = Field(description="Query identifier")
    vendor: str = Field(description="Target vendor")
    query_type: str = Field(description="Query type: SQL, API, bulk")
    query_specification: str = Field(description="Query specification")
    fields_requested: List[str] = Field(description="Fields requested")
    date_range: str = Field(description="Date range")
    estimated_cost: float = Field(description="Estimated cost")
    optimization_notes: str = Field(description="Optimization notes")


class QueryOptimization(BaseModel):
    """Model for query optimization results."""
    optimized_queries: List[OptimizedQuery] = Field(description="Optimized queries")
    total_estimated_cost: float = Field(description="Total estimated cost after optimization")
    cost_savings: float = Field(description="Cost savings from optimization")
    optimization_techniques: List[str] = Field(description="Techniques applied")


class RetrievedDataset(BaseModel):
    """Model for a retrieved dataset."""
    dataset_id: str = Field(description="Dataset identifier")
    vendor: str = Field(description="Source vendor")
    query_id: str = Field(description="Query that retrieved this data")
    num_records: int = Field(description="Number of records")
    num_fields: int = Field(description="Number of fields")
    date_range: str = Field(description="Actual date range")
    actual_cost: float = Field(description="Actual cost incurred")
    retrieval_status: str = Field(description="Success/Partial/Failed")
    retrieval_notes: str = Field(description="Retrieval notes")


class DataRetrieval(BaseModel):
    """Model for data retrieval results."""
    retrieved_datasets: List[RetrievedDataset] = Field(description="Retrieved datasets")
    total_records: int = Field(description="Total records retrieved")
    total_actual_cost: float = Field(description="Total actual cost")
    retrieval_success_rate: float = Field(description="Success rate 0-1")
    failed_queries: List[str] = Field(description="Failed query IDs")


class VendorQualityCheck(BaseModel):
    """Model for vendor-specific quality check."""
    vendor: str = Field(description="Vendor name")
    check_type: str = Field(description="Type of check")
    passed: bool = Field(description="Whether check passed")
    score: float = Field(description="Score 0-100")
    issues: List[str] = Field(description="Issues found")
    notes: str = Field(description="Check notes")


class LicenseRestriction(BaseModel):
    """Model for a license restriction."""
    vendor: str = Field(description="Vendor name")
    restriction_type: str = Field(description="Type: redistribution, display, derived, attribution")
    description: str = Field(description="Restriction description")
    applies_to: List[str] = Field(description="Data types this applies to")
    severity: str = Field(description="High/Medium/Low")


class LicenseCompliance(BaseModel):
    """Model for license compliance results."""
    compliance_status: str = Field(description="Compliant/Partial/Non-compliant")
    risk_level: str = Field(description="High/Medium/Low/None")
    restrictions: List[LicenseRestriction] = Field(description="License restrictions")
    required_citations: Dict[str, str] = Field(description="Required citations by vendor")
    required_disclaimers: List[str] = Field(description="Required disclaimers")
    usage_recommendations: List[str] = Field(description="Usage recommendations")
    compliance_risks: List[str] = Field(description="Identified compliance risks")


# ============================================================================
# ModePremiumSubscribed — Stage 3: Standardization Models
# ============================================================================

class IdentifierMapping(BaseModel):
    """Model for identifier mapping."""
    original_id: str = Field(description="Original vendor-specific identifier")
    vendor: str = Field(description="Source vendor")
    universal_id: str = Field(description="Universal identifier")
    id_type: str = Field(description="ID type: CUSIP, ISIN, PERMNO, etc.")


class FieldMapping(BaseModel):
    """Model for field name mapping."""
    vendor_field: str = Field(description="Vendor-specific field name")
    vendor: str = Field(description="Source vendor")
    standard_field: str = Field(description="Standardized field name")
    transformation: str = Field(description="Transformation applied")


class DataStandardization(BaseModel):
    """Model for data standardization results."""
    identifier_mappings: List[IdentifierMapping] = Field(description="Identifier mappings")
    field_mappings: List[FieldMapping] = Field(description="Field mappings")
    vendors_merged: List[str] = Field(description="Vendors merged")
    overlap_handling: str = Field(description="How overlapping data was handled")
    transformations_applied: List[str] = Field(description="Transformations applied")
    final_record_count: int = Field(description="Final record count")
    final_field_count: int = Field(description="Final field count")
    unified_fields: List[str] = Field(description="Unified field names", default_factory=list)


# ============================================================================
# ModeUserUploaded — Stage 1: File Info Model
# ============================================================================

class FileInfo(BaseModel):
    """Model for parsed file information."""
    file_path: str = Field(description="Path to the file")
    file_name: str = Field(description="File name")
    file_extension: str = Field(description="File extension")
    file_size_mb: float = Field(description="File size in MB")
    num_rows: int = Field(description="Number of rows")
    num_columns: int = Field(description="Number of columns")
    column_names: List[str] = Field(description="Column names")
    column_types: Dict[str, str] = Field(description="Column data types")
    data_preview: List[Dict] = Field(description="First few rows as preview")
    parsing_notes: str = Field(description="Notes from parsing process")


# ============================================================================
# ModeUserUploaded — Stage 2: PII / Privacy / Transformation Models
# ============================================================================

class PIIDetection(BaseModel):
    """Model for PII detection result."""
    column_name: str = Field(description="Column name")
    pii_type: str = Field(description="Type of PII: email, ssn, phone, credit_card, address, name")
    confidence: float = Field(description="Confidence score 0-1")
    sample_matches: List[str] = Field(description="Sample matches (redacted)")
    recommendation: str = Field(description="Recommended action")


class PrivacyScreening(BaseModel):
    """Model for privacy screening results."""
    risk_level: str = Field(description="Risk level: High/Medium/Low/None")
    pii_detected: bool = Field(description="Whether PII was detected")
    pii_detections: List[PIIDetection] = Field(description="PII detections")
    sensitive_columns: List[str] = Field(description="Columns with sensitive data")
    recommendations: List[str] = Field(description="Privacy recommendations")
    remediation_actions: List[str] = Field(description="Suggested remediation actions")


class StructureInference(BaseModel):
    """Model for structure inference results."""
    structure_type: str = Field(description="Type: cross_sectional, time_series, panel, hierarchical")
    confidence: float = Field(description="Confidence in inference 0-1")
    key_variables: Dict[str, List[str]] = Field(description="Key variables by type")
    temporal_variable: Optional[str] = Field(description="Temporal variable if time series/panel")
    id_variables: List[str] = Field(description="Identifier variables")
    grouping_variables: List[str] = Field(description="Grouping/hierarchical variables")
    suggested_roles: Dict[str, str] = Field(description="Suggested variable roles")
    structure_notes: str = Field(description="Notes on structure")


class ResearchAlignment(BaseModel):
    """Model for research alignment assessment."""
    alignment_score: float = Field(description="Alignment score 0-100")
    suitability: str = Field(description="High/Medium/Low suitability")
    sample_size_adequate: bool = Field(description="Whether sample size is adequate")
    variable_coverage: Dict[str, bool] = Field(description="Variable coverage assessment")
    temporal_coverage: str = Field(description="Temporal coverage assessment")
    missing_elements: List[str] = Field(description="Missing data elements")
    recommendations: List[str] = Field(description="Recommendations for research")


class Transformation(BaseModel):
    """Model for a data transformation."""
    transformation_id: str = Field(description="Transformation identifier")
    transformation_type: str = Field(description="Type: privacy, missing_data, outlier, derived, restructure, filter")
    description: str = Field(description="Description of transformation")
    columns_affected: List[str] = Field(description="Columns affected")
    method: str = Field(description="Method used")
    parameters: Dict = Field(default_factory=dict, description="Parameters")
    python_code: str = Field(description="Python code for transformation")


class TransformationResults(BaseModel):
    """Model for transformation results."""
    transformations_applied: List[Transformation] = Field(description="Transformations applied")
    rows_before: int = Field(description="Rows before transformation")
    rows_after: int = Field(description="Rows after transformation")
    columns_before: int = Field(description="Columns before transformation")
    columns_after: int = Field(description="Columns after transformation")
    columns_removed: List[str] = Field(description="Columns removed")
    columns_added: List[str] = Field(description="Columns added")
    transformation_log: List[str] = Field(description="Transformation log")


# ============================================================================
# ModePremiumSubscribed — Stage 2: Quality Assessment & Cleaning Output
# ============================================================================

class PremiumQualityAssessment(BaseModel):
    """Quality assessment for premium subscribed data (vendor-specific checks)."""
    overall_score: float = Field(description="Overall quality score 0-100")
    grade: str = Field(description="A/B/C/D/F grade")
    completeness_score: float = Field(description="Completeness score")
    accuracy_score: float = Field(description="Accuracy score")
    consistency_score: float = Field(description="Consistency score")
    timeliness_score: float = Field(description="Timeliness score")
    vendor_checks: List[VendorQualityCheck] = Field(description="Vendor-specific checks")
    cross_vendor_validation: Dict = Field(description="Cross-vendor validation results")
    issues: List[str] = Field(description="Quality issues")


class PremiumDataCleaningOutput(BaseModel):
    """Complete output of Stage 2 for ModePremiumSubscribed."""
    research_question: str = Field(description="Research question")
    query_optimization: QueryOptimization = Field(description="Query optimization results")
    data_retrieval: DataRetrieval = Field(description="Data retrieval results")
    quality_assessment: PremiumQualityAssessment = Field(description="Quality assessment")
    license_compliance: LicenseCompliance = Field(description="License compliance")
    metadata: Dict = Field(description="Output metadata", default_factory=dict)


# ============================================================================
# ModePremiumSubscribed — Stage 3: Documentation Models
# ============================================================================

class PremiumVariableEntry(BaseModel):
    """A variable entry in the codebook for premium data."""
    variable_name: str = Field(description="Variable name")
    description: str = Field(description="Variable description")
    source_vendor: str = Field(description="Source vendor")
    original_field: str = Field(description="Original field name")
    data_type: str = Field(description="Data type")
    unit: str = Field(description="Unit of measurement")
    value_range: str = Field(description="Range of values")
    license_notes: str = Field(description="License-specific usage notes")


class PremiumDataCodebook(BaseModel):
    """Data codebook for premium subscribed data."""
    title: str = Field(description="Codebook title")
    version: str = Field(description="Version number")
    created_date: str = Field(description="Creation date")
    variables: List[PremiumVariableEntry] = Field(description="Variable entries")
    vendor_attribution: Dict[str, str] = Field(description="Vendor attribution")
    license_compliance_notes: List[str] = Field(description="License compliance notes")
    required_citations: Dict[str, str] = Field(description="Required citations")
    usage_restrictions: List[str] = Field(description="Usage restrictions")


class PremiumDataReport(BaseModel):
    """Data acquisition report for premium subscribed data."""
    title: str = Field(description="Report title")
    research_question: str = Field(description="Research question")
    executive_summary: str = Field(description="Executive summary")
    vendors_section: str = Field(description="Vendors used section")
    credential_section: str = Field(description="Credential validation section")
    cost_section: str = Field(description="Cost estimation and actual costs section")
    query_section: str = Field(description="Query optimization section")
    retrieval_section: str = Field(description="Data retrieval section")
    quality_section: str = Field(description="Quality assessment section")
    compliance_section: str = Field(description="License compliance section")
    standardization_section: str = Field(description="Data standardization section")
    recommendations: List[str] = Field(description="Recommendations")


class PremiumDocumentedDataset(BaseModel):
    """Fully documented dataset for premium subscribed data."""
    dataset_id: str = Field(description="Dataset identifier")
    dataset_name: str = Field(description="Dataset name")
    standardization: DataStandardization = Field(description="Standardization results")
    codebook: PremiumDataCodebook = Field(description="Data codebook")
    report: PremiumDataReport = Field(description="Data acquisition report")
    quality_score: float = Field(description="Overall quality score")
    certification_level: str = Field(description="Bronze/Silver/Gold certification")
    compliance_status: str = Field(description="Compliance status")
    total_cost: float = Field(description="Total cost incurred")
    ready_for_analysis: bool = Field(description="Whether dataset is ready for analysis")


class PremiumQualityAssuranceOutput(BaseModel):
    """Complete output of Stage 3 for ModePremiumSubscribed."""
    research_question: str = Field(description="Research question")
    documented_dataset: PremiumDocumentedDataset = Field(description="Documented dataset")
    metadata: Dict = Field(description="Output metadata", default_factory=dict)
    replication_manifest: Optional[Dict] = Field(
        default=None,
        description="Archivist output: environment spec, pipeline code fingerprint, "
                    "dataset content hash, and consolidated audit trail, enabling "
                    "exact reproduction independent of the run that produced the dataset.")


# ============================================================================
# ModeUserUploaded — Stage 1: Quality Assessment & Source Output
# ============================================================================

class UserUploadedQualityDimension(BaseModel):
    """A quality dimension assessment for user-uploaded data."""
    dimension: str = Field(description="Dimension: completeness, accuracy, consistency, structure")
    score: float = Field(description="Score 0-100")
    issues: List[str] = Field(description="Issues found")
    recommendations: List[str] = Field(description="Recommendations")
    details: Dict = Field(default_factory=dict, description="Additional details")


class UserUploadedQualityAssessment(BaseModel):
    """Comprehensive quality assessment for user-uploaded data."""
    overall_score: float = Field(description="Overall quality score 0-100")
    grade: str = Field(description="A/B/C/D/F grade")
    dimensions: List[UserUploadedQualityDimension] = Field(description="Quality dimensions")
    missing_data_summary: Dict = Field(description="Missing data analysis")
    outlier_summary: Dict = Field(description="Outlier analysis")
    duplicate_summary: Dict = Field(description="Duplicate analysis")
    actionable_issues: List[str] = Field(description="Issues requiring action")


class UserUploadedDataRequirement(BaseModel):
    """A data requirement mapped to an uploaded column."""
    variable_name: str = Field(description="Required variable name")
    role: str = Field(description="Role in analysis: dependent, independent, control, instrument")
    mapped_column: str = Field(description="Uploaded column that satisfies this requirement, or 'MISSING' if none")
    justification: str = Field(description="Why this variable is needed for the research question")


class UserUploadedDataSourceOutput(BaseModel):
    """Complete output of Stage 1 for ModeUserUploaded."""
    file_path: str = Field(description="Input file path")
    research_question: str = Field(description="Research question")
    file_info: FileInfo = Field(description="Parsed file information")
    quality_assessment: UserUploadedQualityAssessment = Field(description="Quality assessment")
    data_requirements: List[UserUploadedDataRequirement] = Field(
        description="Research-question-driven data requirements mapped to uploaded columns",
        default_factory=list)
    cross_domain_opportunities: List[str] = Field(
        description="Cross-domain data integration opportunities", default_factory=list)
    research_question_alignment: str = Field(
        description="Explanation of why this data source is appropriate for the research question",
        default="")
    assumptions: List[str] = Field(
        description="Assumptions underlying the use of this uploaded dataset",
        default_factory=list)
    limitations: List[str] = Field(
        description="Limitations of the uploaded dataset for the research question",
        default_factory=list)
    metadata: Dict = Field(description="Output metadata", default_factory=dict)


# ============================================================================
# ModeUserUploaded — Stage 2: Cleaning Output
# ============================================================================

class UserUploadedDataCleaningOutput(BaseModel):
    """Complete output of Stage 2 for ModeUserUploaded."""
    file_path: str = Field(description="Input file path")
    research_question: str = Field(description="Research question")
    privacy_screening: PrivacyScreening = Field(description="Privacy screening results")
    structure_inference: StructureInference = Field(description="Structure inference results")
    research_alignment: ResearchAlignment = Field(description="Research alignment assessment")
    transformations: TransformationResults = Field(description="Transformation results")
    metadata: Dict = Field(description="Output metadata", default_factory=dict)


# ============================================================================
# ModeUserUploaded — Stage 3: Documentation Models
# ============================================================================

class UserUploadedVariableEntry(BaseModel):
    """A variable entry in the codebook for user-uploaded data."""
    variable_name: str = Field(description="Variable name")
    description: str = Field(description="Variable description")
    data_type: str = Field(description="Data type")
    role: str = Field(description="Variable role: outcome, treatment, control, identifier, other")
    value_range: str = Field(description="Range of values")
    missing_count: int = Field(description="Number of missing values")
    missing_pct: float = Field(description="Percentage missing")
    notes: str = Field(description="Additional notes")


class UserUploadedDataCodebook(BaseModel):
    """Data codebook for user-uploaded data."""
    title: str = Field(description="Codebook title")
    version: str = Field(description="Version number")
    created_date: str = Field(description="Creation date")
    file_info: Dict = Field(description="Original file information")
    variables: List[UserUploadedVariableEntry] = Field(description="Variable entries")
    summary_statistics: Dict = Field(description="Summary statistics")
    data_structure: str = Field(description="Data structure type")
    methodology: str = Field(description="Methodology description")
    methodology_narrative: str = Field(
        default="",
        description="LLM-generated narrative explaining the inferred data structure and "
                    "variable roles, the privacy/cleaning transformations applied and their "
                    "consequences, and how to interpret the QA score when judging fitness "
                    "for analysis")
    limitations: List[str] = Field(description="Known limitations")
    usage_notes: str = Field(description="Usage notes")


class UserUploadedDataReport(BaseModel):
    """Data acquisition report for user-uploaded data."""
    title: str = Field(description="Report title")
    file_path: str = Field(description="Original file path")
    research_question: str = Field(description="Research question")
    executive_summary: str = Field(description="Executive summary")
    file_characteristics_section: str = Field(description="File characteristics section")
    quality_assessment_section: str = Field(description="Quality assessment section")
    privacy_screening_section: str = Field(description="Privacy screening section")
    structure_analysis_section: str = Field(description="Structure analysis section")
    research_alignment_section: str = Field(description="Research alignment section")
    transformations_section: str = Field(description="Transformations section")
    recommendations: List[str] = Field(description="Recommendations")


class UserUploadedDocumentedDataset(BaseModel):
    """Fully documented dataset for user-uploaded data."""
    dataset_id: str = Field(description="Dataset identifier")
    dataset_name: str = Field(description="Dataset name")
    file_path: str = Field(description="Original file path")
    codebook: UserUploadedDataCodebook = Field(description="Data codebook")
    report: UserUploadedDataReport = Field(description="Data acquisition report")
    quality_score: float = Field(description="Overall quality score")
    certification_level: str = Field(description="Bronze/Silver/Gold certification")
    ready_for_analysis: bool = Field(description="Whether dataset is ready for analysis")
    documentation: Dict = Field(description="Additional documentation")


class UserUploadedQualityAssuranceOutput(BaseModel):
    """Complete output of Stage 3 for ModeUserUploaded."""
    file_path: str = Field(description="Input file path")
    research_question: str = Field(description="Research question")
    documented_dataset: UserUploadedDocumentedDataset = Field(description="Documented dataset")
    metadata: Dict = Field(description="Output metadata", default_factory=dict)
    replication_manifest: Optional[Dict] = Field(
        default=None,
        description="Archivist output: environment spec, pipeline code fingerprint, "
                    "dataset content hash, and consolidated audit trail, enabling "
                    "exact reproduction independent of the run that produced the dataset.")
