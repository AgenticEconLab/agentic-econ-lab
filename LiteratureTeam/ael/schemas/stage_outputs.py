# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
LiteratureTeam stage output schemas.

Defines the validated output contracts for each stage in the Literature pipeline.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

# LLMCoercedModel coerces richer LLM JSON shapes (e.g. a scalar str where List[str]
# is declared) to the declared field type BEFORE pydantic type-checks them. Used for
# models built from LLM output so a stray str->List[str] (issue #12) no longer raises
# a ValidationError that silently drops the item. (codes is on sys.path because
# this module is imported as LiteratureTeam.ael.schemas.stage_outputs.)
from shared.schema_coerce import LLMCoercedModel


# ============================================================================
# Stage 1: Literature Gathering — Output Models
# ============================================================================

class LiteratureItem(BaseModel):
    """A literature item retrieved by gathering agents."""
    title: str = Field(description="Title of the paper/document")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    abstract: str = Field(default="No abstract available", description="Abstract or summary")
    url: str = Field(default="", description="URL or identifier")
    source: str = Field(default="unknown", description="Source database/repository")
    year: Optional[int] = Field(default=None, description="Publication year")
    citation_count: Optional[int] = Field(default=None, description="Number of citations")
    literature_type: str = Field(default="academic", description="Type: academic, grey, preprint")
    research_question: str = Field(default="", description="Associated research question")
    relevance_score: Optional[float] = Field(default=None, description="Relevance score (0-1)")
    paper_id: Optional[str] = Field(default=None, description="Unique identifier from source")
    venue: Optional[str] = Field(default=None, description="Publication venue")
    metadata_source: Optional[str] = Field(default=None, description="Provenance of metadata")


class TrendAnalysis(BaseModel):
    """Trend analysis results from TrendTracker agent."""
    trend_name: str = Field(description="Name of the trend")
    description: str = Field(description="Description of the trend")
    key_papers: List[str] = Field(description="Key papers related to this trend")
    emergence_year: Optional[int] = Field(default=None, description="Year trend emerged")
    growth_trajectory: str = Field(default="Growing", description="Growing/Stable/Declining")
    related_questions: List[str] = Field(description="Related research questions")


class CitationEntry(BaseModel):
    """A formatted citation entry."""
    citation_key: str = Field(description="Unique citation key (e.g., Author2023)")
    formatted_citation: str = Field(description="Formatted citation string")
    paper_title: str = Field(description="Paper title")
    authors: List[str] = Field(description="Authors")
    year: Optional[int] = Field(default=None, description="Publication year")
    url: str = Field(description="URL or DOI")
    citation_count: Optional[int] = Field(default=None, description="Number of citations")
    related_trends: List[str] = Field(description="Related trend names")


class KnowledgeInsight(LLMCoercedModel):
    """Extracted knowledge insight from literature.

    Built from LLM JSON. Inherits LLMCoercedModel so a List[str] field returned as a
    plain string (esp. ``related_questions``) is coerced to a 1-element list instead of
    raising a ValidationError that drops the whole insight (issue #12)."""
    insight_title: str = Field(description="Title of the insight")
    insight_description: str = Field(description="Detailed description")
    supporting_papers: List[str] = Field(description="Papers supporting this insight")
    key_findings: List[str] = Field(description="Key findings (3-5 points)")
    methodologies_used: List[str] = Field(description="Methodologies mentioned")
    research_gaps: List[str] = Field(description="Identified research gaps")
    related_questions: List[str] = Field(description="Related research questions")


class LiteratureBatch(BaseModel):
    """Complete literature batch output (aggregation of all gathering results)."""
    research_questions: List[Dict] = Field(description="Input research questions")
    literature_items: List[LiteratureItem] = Field(description="Retrieved literature")
    trend_analyses: List[TrendAnalysis] = Field(description="Trend analysis results")
    citations: List[CitationEntry] = Field(description="Bibliography entries")
    insights: List[KnowledgeInsight] = Field(description="Extracted insights")
    metadata: Dict = Field(default_factory=dict, description="Batch metadata")


class GatheringStageOutput(BaseModel):
    """Complete output of Stage 1: Literature Gathering."""
    literature_batch: LiteratureBatch = Field(description="Complete literature batch")
    metadata: Dict = Field(default_factory=dict, description="Stage metadata")


# ============================================================================
# Stage 2: Gap Detection — Output Models
# ============================================================================

class PaperStructure(BaseModel):
    """Decomposed structure of an analyzed paper."""
    paper_title: str = Field(description="Title of the paper")
    research_objectives: List[str] = Field(description="Main research objectives")
    methodologies: List[str] = Field(description="Methodologies used")
    key_findings: List[str] = Field(description="Key findings (3-5 points)")
    theoretical_contributions: List[str] = Field(description="Theoretical contributions")
    limitations: List[str] = Field(description="Stated limitations")
    future_work_suggestions: List[str] = Field(description="Suggested future work")
    data_sources: List[str] = Field(default_factory=list, description="Data sources used")
    assumptions: List[str] = Field(default_factory=list, description="Key assumptions")
    decomposition_mode: str = Field(default="metadata", description="Mode: metadata or deep")
    full_text_available: bool = Field(default=False, description="Whether full PDF was extracted")
    methodology_details: Optional[Dict] = Field(default=None, description="Detailed methodology")
    data_description: Optional[Dict] = Field(default=None, description="Detailed data description")
    model_equations: List[str] = Field(default_factory=list, description="Key equations/models")
    empirical_strategy: Optional[str] = Field(default=None, description="Empirical strategy")
    robustness_checks: List[str] = Field(default_factory=list, description="Robustness checks")
    variable_definitions: List[Dict] = Field(default_factory=list, description="Variable definitions")
    sample_description: Optional[str] = Field(default=None, description="Sample description")
    sections_extracted: List[str] = Field(default_factory=list, description="Sections extracted")
    relevance_score: Optional[float] = Field(default=None, description="Relevance (1-5)")
    domain_tags: List[str] = Field(default_factory=list, description="Research domain tags")


class ResearchGap(BaseModel):
    """An identified research gap."""
    gap_id: str = Field(default="", description="Unique gap identifier")
    gap_type: str = Field(default="methodological", description="Type: methodological, theoretical, empirical, data")
    gap_title: str = Field(default="Unnamed Gap", description="Concise title")
    gap_description: str = Field(default="", description="Detailed description")
    evidence_papers: List[str] = Field(default_factory=list, description="Papers revealing this gap")
    severity: str = Field(default="Medium", description="Critical/High/Medium/Low")
    addressability: str = Field(default="Moderate", description="Easy/Moderate/Difficult")
    related_trends: List[str] = Field(default_factory=list, description="Related trend names")
    potential_impact: str = Field(default="Moderate impact expected", description="Expected impact")
    suggested_approaches: List[str] = Field(default_factory=list, description="Suggested approaches")


class KnowledgeNode(BaseModel):
    """A node in the knowledge graph."""
    node_id: str = Field(description="Unique node identifier")
    node_type: str = Field(description="Type: paper, concept, method, gap, trend")
    label: str = Field(description="Node label/name")
    properties: Dict = Field(default_factory=dict, description="Additional properties")


class KnowledgeEdge(BaseModel):
    """An edge in the knowledge graph."""
    source_id: str = Field(description="Source node ID")
    target_id: str = Field(description="Target node ID")
    relationship: str = Field(description="Relationship type: cites, uses_method, etc.")
    weight: Optional[float] = Field(default=1.0, description="Edge weight/strength")
    properties: Dict = Field(default_factory=dict, description="Additional properties")


class KnowledgeGraph(BaseModel):
    """Complete knowledge graph of the literature."""
    nodes: List[KnowledgeNode] = Field(description="Graph nodes")
    edges: List[KnowledgeEdge] = Field(description="Graph edges")
    metadata: Dict = Field(default_factory=dict, description="Graph metadata")


class GapAnalysisResult(BaseModel):
    """Complete gap analysis output."""
    paper_structures: List[PaperStructure] = Field(description="Decomposed paper structures")
    research_gaps: List[ResearchGap] = Field(description="Identified research gaps")
    knowledge_graph: KnowledgeGraph = Field(description="Knowledge graph")
    gap_summary: Dict = Field(default_factory=dict, description="Summary statistics")
    metadata: Dict = Field(default_factory=dict, description="Analysis metadata")
    reasoning_trace: Dict = Field(default_factory=dict, description="Decision reasoning trace")
    assumptions: List[str] = Field(default_factory=list, description="Assumptions made")
    limitations: List[str] = Field(default_factory=list, description="Known limitations")


class GapDetectionStageOutput(BaseModel):
    """Complete output of Stage 2: Gap Detection."""
    gap_analysis: GapAnalysisResult = Field(description="Complete gap analysis")
    metadata: Dict = Field(default_factory=dict, description="Stage metadata")


# ============================================================================
# Stage 3: Synthesis — Output Models
# ============================================================================

class LiteratureSection(BaseModel):
    """A section of the literature review."""
    section_title: str = Field(description="Section title")
    section_content: str = Field(description="Section content (2-4 paragraphs)")
    key_papers: List[str] = Field(description="Key papers cited in this section")
    related_gaps: List[str] = Field(description="Related gap IDs")


class LiteratureReview(BaseModel):
    """Complete literature review document."""
    title: str = Field(description="Review title")
    abstract: str = Field(description="Abstract (150-250 words)")
    introduction: str = Field(description="Introduction section")
    sections: List[LiteratureSection] = Field(description="Main review sections")
    synthesis: str = Field(description="Synthesis and conclusions")
    metadata: Dict = Field(default_factory=dict, description="Review metadata")


class ResearchObjective(BaseModel):
    """A research objective derived from gap analysis."""
    objective_id: str = Field(description="Unique objective ID")
    objective_title: str = Field(description="Objective title")
    description: str = Field(description="Detailed description")
    addresses_gaps: List[str] = Field(description="Gap IDs this addresses")
    methodology: List[str] = Field(description="Proposed methodologies")
    expected_outcomes: List[str] = Field(description="Expected outcomes")
    timeline: str = Field(description="Estimated timeline")
    priority: str = Field(description="High/Medium/Low")


class ResearchPlan(BaseModel):
    """Research plan with objectives and methodology."""
    plan_title: str = Field(description="Plan title")
    overview: str = Field(description="Plan overview (2-3 paragraphs)")
    research_objectives: List[ResearchObjective] = Field(description="Research objectives")
    methodology_overview: str = Field(description="Overall methodology approach")
    expected_contributions: List[str] = Field(description="Expected contributions")
    timeline_summary: str = Field(description="Timeline summary")
    resource_requirements: List[str] = Field(description="Required resources")


class FormattedReference(BaseModel):
    """A formatted bibliographic reference."""
    citation_key: str = Field(description="Citation key (e.g., Smith2023)")
    formatted_citation: str = Field(description="Full formatted citation")
    reference_type: str = Field(description="Type: journal, conference, preprint, etc.")
    importance: str = Field(description="High/Medium/Low")
    title: str = Field(default="", description="Paper title")
    url: str = Field(default="", description="Source URL / DOI carried from gathering, so the reference stays verifiable")


class Bibliography(BaseModel):
    """Complete bibliography."""
    references: List[FormattedReference] = Field(description="All references")
    total_references: int = Field(description="Total number of references")
    by_type: Dict = Field(default_factory=dict, description="Count by reference type")
    by_importance: Dict = Field(default_factory=dict, description="Count by importance")


class CrossDomainConnection(BaseModel):
    """A cross-domain connection identified in the literature."""
    domain_a: str = Field(description="First research domain")
    domain_b: str = Field(description="Second research domain")
    connection_type: str = Field(description="Nature of the connection")
    description: str = Field(description="How these domains connect")
    significance: str = Field(description="Why this connection matters")
    bridging_papers: List[str] = Field(default_factory=list, description="Papers spanning both")
    research_opportunity: str = Field(default="", description="Research opportunity")


class SynthesisResult(BaseModel):
    """Complete synthesis output."""
    literature_review: LiteratureReview = Field(description="Literature review")
    research_plan: ResearchPlan = Field(description="Research plan")
    bibliography: Bibliography = Field(description="Bibliography")
    cross_domain_connections: List[CrossDomainConnection] = Field(
        default_factory=list, description="Cross-domain connections")
    metadata: Dict = Field(default_factory=dict, description="Synthesis metadata")
    decision_log: Dict = Field(default_factory=dict, description="Decision transparency log")
    assumptions_and_limitations: Dict = Field(
        default_factory=dict, description="Assumptions and limitations")


class SynthesisStageOutput(BaseModel):
    """Complete output of Stage 3: Synthesis."""
    synthesis: SynthesisResult = Field(description="Complete synthesis result")
    metadata: Dict = Field(default_factory=dict, description="Stage metadata")
