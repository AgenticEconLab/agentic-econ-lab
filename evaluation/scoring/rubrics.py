# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Evaluation rubrics for LLM-as-Reviewer content quality assessment.

Defines team-specific rubrics for each Tier-2 evaluated dimension.
Each rubric contains 3-4 sub-criteria scored on a 1-5 Likert scale.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class LevelDescriptor:
    """Description for each score level (1-5)."""
    score: int
    label: str
    description: str


@dataclass
class SubCriterion:
    """A single evaluation sub-criterion within a rubric."""
    name: str
    description: str
    levels: List[LevelDescriptor] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        """Format sub-criterion for inclusion in an LLM prompt."""
        lines = [f"### {self.name}", f"{self.description}", ""]
        for level in sorted(self.levels, key=lambda l: l.score):
            lines.append(f"  {level.score} ({level.label}): {level.description}")
        return "\n".join(lines)


@dataclass
class EvaluationRubric:
    """A complete rubric for evaluating one dimension of one team."""
    dimension: str
    team: str
    sub_criteria: List[SubCriterion]

    def to_prompt_text(self) -> str:
        """Format the full rubric for inclusion in an LLM prompt."""
        lines = [
            f"## Evaluation Rubric: {self.dimension.upper()} — {self.team}",
            "",
            "Score each sub-criterion on a 1-5 scale using the level descriptors below.",
            "",
        ]
        for sc in self.sub_criteria:
            lines.append(sc.to_prompt_text())
            lines.append("")
        return "\n".join(lines)


# =============================================================================
# STANDARD LEVEL DESCRIPTORS (reused across rubrics)
# =============================================================================

def _standard_levels(
    poor: str, below: str, adequate: str, good: str, excellent: str
) -> List[LevelDescriptor]:
    return [
        LevelDescriptor(1, "Poor", poor),
        LevelDescriptor(2, "Below Average", below),
        LevelDescriptor(3, "Adequate", adequate),
        LevelDescriptor(4, "Good", good),
        LevelDescriptor(5, "Excellent", excellent),
    ]


# =============================================================================
# CORRECTNESS RUBRICS (team-specific)
# =============================================================================

CORRECTNESS_RUBRICS: Dict[str, EvaluationRubric] = {
    "IdeationTeam": EvaluationRubric(
        dimension="correctness",
        team="IdeationTeam",
        sub_criteria=[
            SubCriterion(
                name="Domain Relevance",
                description="Are the research questions grounded in economics and relevant to the stated research topic?",
                levels=_standard_levels(
                    "Questions are unrelated to economics or the stated topic",
                    "Questions are tangentially related with significant gaps in relevance",
                    "Questions are generally relevant but some lack clear economic grounding",
                    "Questions are well-grounded in economics with clear topical relevance",
                    "All questions are deeply rooted in economic theory and highly relevant to the topic",
                ),
            ),
            SubCriterion(
                name="Factual Grounding",
                description="Do the theoretical frameworks, rationales, and methodologies reference real, established economic concepts and theories?",
                levels=_standard_levels(
                    "References are fabricated or fundamentally incorrect",
                    "Some references are real but many are inaccurate or misattributed",
                    "Most references are real but some lack precision or specificity",
                    "References are accurate and well-attributed with minor gaps",
                    "All references are verified, accurate, and properly attributed",
                ),
            ),
            SubCriterion(
                name="Internal Consistency",
                description="Are the proposed methodologies consistent with the stated theoretical frameworks and research questions?",
                levels=_standard_levels(
                    "Methodologies contradict the theoretical frameworks",
                    "Significant mismatches between methods and frameworks",
                    "Generally consistent with some misalignment",
                    "Methods are well-aligned with frameworks, minor issues only",
                    "Perfect alignment between questions, frameworks, and methodologies",
                ),
            ),
            SubCriterion(
                name="Citation Validity",
                description="Do the referenced theories, authors, and frameworks exist and are they correctly attributed?",
                levels=_standard_levels(
                    "Most citations appear fabricated or are incorrectly attributed",
                    "Several citations are questionable or misattributed",
                    "Most citations are plausible but some cannot be verified",
                    "Citations are largely verifiable with minor attribution issues",
                    "All citations are verifiable and correctly attributed",
                ),
            ),
        ],
    ),
    "LiteratureTeam": EvaluationRubric(
        dimension="correctness",
        team="LiteratureTeam",
        sub_criteria=[
            SubCriterion(
                name="Literature Relevance",
                description="Are the collected papers relevant to the stated research question?",
                levels=_standard_levels(
                    "Most papers are irrelevant to the research question",
                    "Many papers are tangentially relevant or from wrong domains",
                    "Majority are relevant but some off-topic items included",
                    "Papers are well-targeted with few irrelevant inclusions",
                    "All papers are highly relevant and well-curated for the research question",
                ),
            ),
            SubCriterion(
                name="Gap Identification Accuracy",
                description="Are the identified research gaps genuine and not already addressed in the collected literature?",
                levels=_standard_levels(
                    "Identified gaps are not genuine or are already well-addressed",
                    "Some gaps are valid but others are redundant or trivial",
                    "Most gaps are genuine but some overlap with existing work",
                    "Gaps are well-identified and mostly unaddressed in the literature",
                    "All gaps are genuine, significant, and clearly unaddressed",
                ),
            ),
            SubCriterion(
                name="Synthesis Accuracy",
                description="Does the synthesis accurately represent the findings of the reviewed papers?",
                levels=_standard_levels(
                    "Synthesis misrepresents or contradicts the reviewed literature",
                    "Several inaccuracies in representing paper findings",
                    "Generally accurate with some oversimplifications",
                    "Accurate representation with nuanced understanding",
                    "Synthesis faithfully and comprehensively represents all reviewed work",
                ),
            ),
            SubCriterion(
                name="Citation Integrity",
                description="Are paper titles, authors, and claims accurately represented?",
                levels=_standard_levels(
                    "Paper metadata is largely fabricated or incorrect",
                    "Several papers have incorrect titles, authors, or claims",
                    "Most metadata is correct but some inconsistencies exist",
                    "Metadata is largely accurate with very minor errors",
                    "All paper metadata is accurate and verifiable",
                ),
            ),
        ],
    ),
    "ModelTeam": EvaluationRubric(
        dimension="correctness",
        team="ModelTeam",
        sub_criteria=[
            SubCriterion(
                name="Mathematical Validity",
                description="Are the equations dimensionally consistent and mathematically well-formed?",
                levels=_standard_levels(
                    "Equations contain fundamental mathematical errors",
                    "Some equations are inconsistent or poorly formed",
                    "Equations are generally valid but with minor issues",
                    "Equations are well-formed and dimensionally consistent",
                    "All mathematical formulations are rigorous and formally correct",
                ),
            ),
            SubCriterion(
                name="Assumption Soundness",
                description="Are model assumptions clearly stated, justified, and appropriate for the model type?",
                levels=_standard_levels(
                    "Assumptions are unstated, unjustified, or contradictory",
                    "Some assumptions lack justification or are inappropriate",
                    "Assumptions are stated and generally reasonable",
                    "Assumptions are well-justified with supporting literature",
                    "All assumptions are rigorously justified, with criticality assessed and relaxation paths identified",
                ),
            ),
            SubCriterion(
                name="Literature Reference Accuracy",
                description="Do cited works in the theoretical framework exist and are they correctly attributed?",
                levels=_standard_levels(
                    "Most cited works appear fabricated",
                    "Several citations are questionable or misattributed",
                    "Most citations are plausible but some unverifiable",
                    "Citations are largely verifiable and accurately attributed",
                    "All citations are real, correctly attributed, and appropriately used",
                ),
            ),
            SubCriterion(
                name="Model-Theory Alignment",
                description="Does the formal model faithfully operationalize the stated theoretical framework?",
                levels=_standard_levels(
                    "Model has no clear connection to the theoretical framework",
                    "Significant gaps between theory and model specification",
                    "Model partially captures the theoretical framework",
                    "Model closely operationalizes the theory with minor gaps",
                    "Model is a faithful and comprehensive operationalization of the theory",
                ),
            ),
        ],
    ),
    "DataTeam": EvaluationRubric(
        dimension="correctness",
        team="DataTeam",
        sub_criteria=[
            SubCriterion(
                name="Source Appropriateness",
                description="Are the identified data sources appropriate for the research question?",
                levels=_standard_levels(
                    "Data sources are irrelevant to the research question",
                    "Some sources are appropriate but major gaps exist",
                    "Most sources are appropriate for the research needs",
                    "Sources are well-chosen and cover key data requirements",
                    "All sources are optimal for the research question with comprehensive coverage",
                ),
            ),
            SubCriterion(
                name="Variable Specification Accuracy",
                description="Are variable names, units, frequencies, and time periods correctly specified?",
                levels=_standard_levels(
                    "Variable specifications contain major errors",
                    "Several variables have incorrect units or frequencies",
                    "Most specifications are correct with minor issues",
                    "Specifications are accurate and well-documented",
                    "All variable specifications are precise, correct, and well-documented",
                ),
            ),
            SubCriterion(
                name="Coverage Adequacy",
                description="Do the identified data requirements adequately cover what the research question needs?",
                levels=_standard_levels(
                    "Major data requirements are missing",
                    "Several important variables are not covered",
                    "Core requirements are covered but some gaps remain",
                    "Requirements cover the research question well with minor gaps",
                    "Requirements comprehensively cover all aspects of the research question",
                ),
            ),
            SubCriterion(
                name="API/Source Validity",
                description="Do the suggested data sources (FRED, BLS, etc.) actually provide the claimed data series?",
                levels=_standard_levels(
                    "Most suggested API endpoints or series IDs are invalid",
                    "Several data series are not available from the claimed sources",
                    "Most sources are valid but some series may not exist as specified",
                    "Sources are valid and most series are correctly identified",
                    "All sources and series are verified as available and correctly specified",
                ),
            ),
        ],
    ),
}


# =============================================================================
# SOUNDNESS RUBRICS (team-specific)
# =============================================================================

SOUNDNESS_RUBRICS: Dict[str, EvaluationRubric] = {
    "IdeationTeam": EvaluationRubric(
        dimension="soundness",
        team="IdeationTeam",
        sub_criteria=[
            SubCriterion(
                name="Logical Coherence",
                description="Does the reasoning from research topic to generated research questions follow logically?",
                levels=_standard_levels(
                    "No logical connection between inputs and generated questions",
                    "Weak logical connections with significant gaps in reasoning",
                    "Generally logical but some steps lack clear justification",
                    "Clear logical progression with well-supported reasoning",
                    "Impeccable logical chain from topic analysis to question formulation",
                ),
            ),
            SubCriterion(
                name="Methodological Appropriateness",
                description="Are the proposed methodologies suitable for answering the research questions?",
                levels=_standard_levels(
                    "Proposed methods cannot address the research questions",
                    "Methods are partially suitable but major mismatches exist",
                    "Methods are generally appropriate with some concerns",
                    "Methods are well-suited to the research questions",
                    "Methods are optimally chosen and thoroughly justified for each question",
                ),
            ),
            SubCriterion(
                name="Theoretical Integration",
                description="Are multiple theoretical perspectives coherently integrated rather than merely juxtaposed?",
                levels=_standard_levels(
                    "Theories are listed without any integration",
                    "Theories are juxtaposed but not meaningfully connected",
                    "Some integration attempted but connections are superficial",
                    "Theories are meaningfully integrated with clear connections",
                    "Deep, sophisticated integration showing how theories complement and extend each other",
                ),
            ),
            SubCriterion(
                name="Cross-Stage Consistency",
                description="Is there logical progression and consistency from sourcing through refinement to finalized questions?",
                levels=_standard_levels(
                    "No discernible progression across stages",
                    "Stages appear disconnected with inconsistent outputs",
                    "Some progression visible but with gaps between stages",
                    "Clear progression with consistent refinement across stages",
                    "Seamless logical progression showing clear value-added at each stage",
                ),
            ),
        ],
    ),
    "LiteratureTeam": EvaluationRubric(
        dimension="soundness",
        team="LiteratureTeam",
        sub_criteria=[
            SubCriterion(
                name="Thematic Organization",
                description="Is the literature logically organized by relevant themes or research dimensions?",
                levels=_standard_levels(
                    "No discernible organizational structure",
                    "Organization is inconsistent or uses inappropriate categories",
                    "Basic thematic structure present but some papers miscategorized",
                    "Well-organized by relevant themes with clear rationale",
                    "Sophisticated thematic organization revealing meaningful research landscape",
                ),
            ),
            SubCriterion(
                name="Gap-Literature Connection",
                description="Do the identified gaps logically follow from the reviewed literature?",
                levels=_standard_levels(
                    "Gaps have no connection to the reviewed literature",
                    "Gaps are loosely connected but not well-supported by the review",
                    "Most gaps follow from the literature but some are unsupported",
                    "Gaps are well-grounded in the reviewed literature",
                    "Each gap is clearly and convincingly derived from systematic analysis of the literature",
                ),
            ),
            SubCriterion(
                name="Synthesis Logical Structure",
                description="Does the synthesis follow a coherent argumentative structure?",
                levels=_standard_levels(
                    "Synthesis is incoherent or contradictory",
                    "Some logical structure but significant argumentative gaps",
                    "Generally coherent with some structural weaknesses",
                    "Well-structured synthesis with clear argumentation",
                    "Rigorous argumentative structure building a compelling narrative from the literature",
                ),
            ),
        ],
    ),
    "ModelTeam": EvaluationRubric(
        dimension="soundness",
        team="ModelTeam",
        sub_criteria=[
            SubCriterion(
                name="Assumption-to-Model Logic",
                description="Do the model equations follow logically from the stated assumptions?",
                levels=_standard_levels(
                    "Equations are disconnected from assumptions",
                    "Some equations follow from assumptions but major gaps exist",
                    "Most equations are traceable to assumptions",
                    "Clear logical derivation from assumptions to equations",
                    "Rigorous derivation chain from every assumption to corresponding model components",
                ),
            ),
            SubCriterion(
                name="Calibration-Theory Consistency",
                description="Does the calibration approach appropriately test the theoretical predictions?",
                levels=_standard_levels(
                    "Calibration is unrelated to the theoretical predictions",
                    "Calibration partially addresses theoretical predictions",
                    "Calibration generally aligned with theory but some gaps",
                    "Calibration well-designed to test key theoretical predictions",
                    "Calibration comprehensively and rigorously tests all theoretical predictions",
                ),
            ),
            SubCriterion(
                name="Component Interaction Logic",
                description="Are the interactions between model components logically specified?",
                levels=_standard_levels(
                    "Component interactions are undefined or contradictory",
                    "Some interactions specified but major logical gaps",
                    "Most interactions are logically specified",
                    "Interactions are well-specified and internally consistent",
                    "All interactions are rigorously defined with clear theoretical justification",
                ),
            ),
        ],
    ),
    "DataTeam": EvaluationRubric(
        dimension="soundness",
        team="DataTeam",
        sub_criteria=[
            SubCriterion(
                name="Requirement-Question Alignment",
                description="Do the data requirements logically follow from the research question?",
                levels=_standard_levels(
                    "Data requirements are unrelated to the research question",
                    "Some requirements are relevant but major gaps exist",
                    "Most requirements are aligned with the research question",
                    "Requirements are well-aligned and cover key aspects",
                    "Requirements are comprehensive and precisely derived from the research question",
                ),
            ),
            SubCriterion(
                name="Source Selection Logic",
                description="Is the choice of data sources logically justified given the requirements?",
                levels=_standard_levels(
                    "Source choices are arbitrary or unjustified",
                    "Some source choices are logical but others are questionable",
                    "Most source selections are reasonable",
                    "Source selections are well-justified and appropriate",
                    "Each source selection is optimally justified with clear reasoning",
                ),
            ),
            SubCriterion(
                name="Pipeline Logic",
                description="Does the data pipeline (source -> clean -> QA) follow a logically sound progression?",
                levels=_standard_levels(
                    "Pipeline stages are disordered or redundant",
                    "Some logical issues in the pipeline progression",
                    "Pipeline is generally logical with minor issues",
                    "Pipeline follows a clear and logical progression",
                    "Pipeline is optimally structured with each stage building logically on the previous",
                ),
            ),
        ],
    ),
}


# =============================================================================
# INNOVATION POTENTIAL RUBRIC (generic, used across all teams)
# =============================================================================

INNOVATION_RUBRIC = EvaluationRubric(
    dimension="innovation_potential",
    team="generic",
    sub_criteria=[
        SubCriterion(
            name="Novelty",
            description=(
                "Does the output introduce genuinely new combinations of ideas, perspectives, "
                "or approaches not commonly found in existing literature? "
                "Calibration note: A score of 3 indicates output that synthesizes known ideas "
                "in standard ways. A score of 4 requires at least one combination that a domain "
                "expert would find non-obvious. A score of 5 requires connections that could "
                "seed a genuinely new research program."
            ),
            levels=_standard_levels(
                "Output is entirely derivative with no novel elements",
                "Minor variations on well-established ideas",
                "Some novel elements but largely follows established patterns",
                "Contains meaningfully novel combinations or perspectives",
                "Introduces genuinely original ideas or innovative cross-domain connections",
            ),
        ),
        SubCriterion(
            name="Cross-Domain Integration",
            description=(
                "Does the output draw meaningful connections across different economic subfields "
                "or between economics and other disciplines? "
                "Calibration note: Count distinct subfields referenced (e.g., macroeconomics, "
                "behavioral economics, computational economics). A score of 3 requires 2-3 "
                "subfields with surface-level connections. A score of 4 requires 3+ subfields "
                "with substantive integration. A score of 5 requires deep connections that "
                "reveal non-obvious complementarities."
            ),
            levels=_standard_levels(
                "Output is confined to a single narrow subfield",
                "Mentions multiple fields but connections are superficial",
                "Some cross-domain connections are drawn",
                "Meaningful integration across multiple fields",
                "Deep, sophisticated cross-domain integration revealing non-obvious connections",
            ),
        ),
        SubCriterion(
            name="Research Significance",
            description=(
                "Would the output, if pursued as research, make a meaningful contribution "
                "to the field? "
                "Calibration note: Consider whether the output identifies gaps that are "
                "(a) genuinely unaddressed and (b) addressable with existing methods. A score "
                "of 3 corresponds to a solid but incremental contribution. A score of 5 "
                "corresponds to a potentially field-shaping direction."
            ),
            levels=_standard_levels(
                "Output has no discernible research value",
                "Output addresses a trivial or already-resolved question",
                "Output has modest potential contribution",
                "Output addresses a significant gap with clear potential impact",
                "Output identifies a high-impact research direction with transformative potential",
            ),
        ),
        SubCriterion(
            name="Specificity and Actionability",
            description=(
                "Are the novel elements specific enough to guide actual research implementation? "
                "Calibration note: A score of 3 means a researcher could identify the topic area "
                "but would need substantial additional work to formulate hypotheses. A score of 5 "
                "means the output contains testable hypotheses, suggested data sources, and a "
                "methodological sketch."
            ),
            levels=_standard_levels(
                "Output is too vague to guide any research",
                "Some specificity but major gaps in actionability",
                "Moderately specific with a general sense of direction",
                "Specific enough to design a research study",
                "Highly specific with clear, actionable research directions and methodology",
            ),
        ),
    ],
)


# =============================================================================
# TRANSPARENCY RUBRIC (generic, used across all teams)
# =============================================================================

TRANSPARENCY_RUBRIC = EvaluationRubric(
    dimension="transparency",
    team="generic",
    sub_criteria=[
        SubCriterion(
            name="Decision Rationale Clarity",
            description="Can a reader understand WHY particular choices were made (e.g., why certain questions were prioritized, why certain sources were selected)?",
            levels=_standard_levels(
                "No rationale provided for any decisions",
                "Rationale is vague or generic for most decisions",
                "Some decisions have clear rationale, others do not",
                "Most decisions have clear, understandable rationale",
                "Every decision is accompanied by clear, specific reasoning",
            ),
        ),
        SubCriterion(
            name="Intermediate Step Visibility",
            description="Are intermediate reasoning steps and data transformations visible in the outputs?",
            levels=_standard_levels(
                "Only final outputs are visible with no intermediate information",
                "Minimal intermediate information available",
                "Some intermediate steps are visible but many are hidden",
                "Most intermediate steps are documented and traceable",
                "Complete visibility of all intermediate steps from input to output",
            ),
        ),
        SubCriterion(
            name="Assumption and Limitation Disclosure",
            description="Are assumptions, limitations, and uncertainties acknowledged in the outputs?",
            levels=_standard_levels(
                "No assumptions or limitations are disclosed",
                "Few assumptions are stated, limitations ignored",
                "Some assumptions and limitations are noted",
                "Key assumptions and limitations are clearly stated",
                "Comprehensive disclosure of all assumptions, limitations, and uncertainty levels",
            ),
        ),
    ],
)


# =============================================================================
# DECISION QUALITY RUBRIC (generic, used across all teams)
# =============================================================================

DECISION_QUALITY_RUBRIC = EvaluationRubric(
    dimension="decision_quality",
    team="generic",
    sub_criteria=[
        SubCriterion(
            name="Decision Coherence",
            description=(
                "Do sequential decisions in the workflow follow logically from prior context? "
                "Consider whether each stage's approach builds on what was learned in previous stages."
            ),
            levels=_standard_levels(
                "Decisions appear random with no logical connection to prior stages",
                "Some decisions follow from context but many seem arbitrary",
                "Generally coherent decision sequence with occasional gaps",
                "Decisions clearly build on prior context with minor deviations",
                "Every decision demonstrably follows from accumulated evidence and prior outputs",
            ),
        ),
        SubCriterion(
            name="Information Utilization",
            description=(
                "Does each stage effectively use information produced by prior stages? "
                "Check whether findings from early stages inform later stage decisions."
            ),
            levels=_standard_levels(
                "Later stages ignore outputs from earlier stages entirely",
                "Minimal use of cross-stage information",
                "Some cross-stage information flow but key findings are underutilized",
                "Good utilization of prior stage outputs with minor missed opportunities",
                "Comprehensive integration of all prior outputs into subsequent decisions",
            ),
        ),
        SubCriterion(
            name="Scope Management",
            description=(
                "Does the workflow maintain appropriate scope throughout execution? "
                "Check for scope creep, premature narrowing, or loss of focus."
            ),
            levels=_standard_levels(
                "Scope shifts dramatically between stages with no justification",
                "Significant scope drift from the original research question",
                "Generally on-scope with some tangential diversions",
                "Well-managed scope with justified expansions or narrowing",
                "Precise scope management throughout, with explicit justification for any changes",
            ),
        ),
    ],
)


# =============================================================================
# ECONOMIC RIGOR RUBRICS (team-specific)
# =============================================================================

ECONOMIC_RIGOR_RUBRICS: Dict[str, EvaluationRubric] = {
    "IdeationTeam": EvaluationRubric(
        dimension="economic_rigor",
        team="IdeationTeam",
        sub_criteria=[
            SubCriterion(
                name="Theoretical Framework Validity",
                description="Are the referenced economic theories and frameworks correctly characterized and appropriately applied?",
                levels=_standard_levels(
                    "Economic theories are mischaracterized or inappropriately applied",
                    "Some theories are correctly referenced but others are misapplied",
                    "Theories are generally correct but applications are sometimes shallow",
                    "Theories are accurately characterized with appropriate application",
                    "Deep, nuanced understanding of economic theories with sophisticated application",
                ),
            ),
            SubCriterion(
                name="Methodological Appropriateness",
                description="Are the proposed research methodologies standard for economics and appropriate for the research questions?",
                levels=_standard_levels(
                    "Proposed methods are not standard economics methodology",
                    "Some methods are appropriate but others are mismatched",
                    "Methods are generally standard but some choices are suboptimal",
                    "Methods are well-chosen and standard for the research questions",
                    "Methods are optimal, demonstrating deep understanding of economics methodology",
                ),
            ),
            SubCriterion(
                name="Economics Terminology",
                description="Is economics terminology used correctly and precisely?",
                levels=_standard_levels(
                    "Terminology is frequently misused or confused",
                    "Several terms are used imprecisely or in non-standard ways",
                    "Most terminology is correct with occasional imprecision",
                    "Terminology is precise and correctly used throughout",
                    "Terminology use reflects expert-level precision and nuance",
                ),
            ),
        ],
    ),
    "LiteratureTeam": EvaluationRubric(
        dimension="economic_rigor",
        team="LiteratureTeam",
        sub_criteria=[
            SubCriterion(
                name="Field Coverage",
                description="Does the literature review cover the appropriate subfields and methodological approaches?",
                levels=_standard_levels(
                    "Review misses major relevant subfields entirely",
                    "Several important subfields or approaches are underrepresented",
                    "Core subfields covered but some secondary areas missed",
                    "Comprehensive coverage of relevant subfields and approaches",
                    "Exhaustive coverage revealing deep understanding of the field landscape",
                ),
            ),
            SubCriterion(
                name="Methodological Classification",
                description="Are reviewed papers correctly classified by their methodological approach?",
                levels=_standard_levels(
                    "Papers are frequently misclassified methodologically",
                    "Several classification errors for non-obvious methodologies",
                    "Most papers correctly classified with minor errors",
                    "Papers are accurately classified by methodology",
                    "Precise methodological classification with nuanced distinctions",
                ),
            ),
            SubCriterion(
                name="Contribution Assessment",
                description="Are the contributions of reviewed papers accurately assessed within the field?",
                levels=_standard_levels(
                    "Paper contributions are fundamentally mischaracterized",
                    "Some contributions are overstated or understated",
                    "Generally accurate assessment with occasional misjudgments",
                    "Contributions are accurately assessed in field context",
                    "Nuanced assessment showing deep understanding of each paper's place in the literature",
                ),
            ),
        ],
    ),
    "ModelTeam": EvaluationRubric(
        dimension="economic_rigor",
        team="ModelTeam",
        sub_criteria=[
            SubCriterion(
                name="Model Specification Standards",
                description="Does the model specification follow established economics modeling conventions?",
                levels=_standard_levels(
                    "Model specification violates basic economics modeling conventions",
                    "Some conventions are followed but significant departures exist",
                    "Specification is generally conventional with minor deviations",
                    "Specification follows established conventions accurately",
                    "Specification demonstrates expert-level adherence to modeling best practices",
                ),
            ),
            SubCriterion(
                name="Parameter Calibration Rigor",
                description="Are calibration targets, data sources, and methods consistent with standard economics practice?",
                levels=_standard_levels(
                    "Calibration approach has no basis in economics practice",
                    "Some calibration choices are standard but others are ad hoc",
                    "Calibration is generally standard with some non-standard choices",
                    "Calibration follows standard economics practice throughout",
                    "Calibration demonstrates rigorous adherence to state-of-the-art methodology",
                ),
            ),
            SubCriterion(
                name="Equilibrium and Dynamics",
                description="Are equilibrium concepts and dynamic specifications appropriate for the model type?",
                levels=_standard_levels(
                    "Equilibrium or dynamics concepts are fundamentally incorrect",
                    "Some concepts are correct but others are misapplied",
                    "Generally appropriate but with some simplifications that may affect validity",
                    "Equilibrium and dynamics are appropriately specified",
                    "Sophisticated treatment demonstrating deep understanding of model dynamics",
                ),
            ),
        ],
    ),
    "DataTeam": EvaluationRubric(
        dimension="economic_rigor",
        team="DataTeam",
        sub_criteria=[
            SubCriterion(
                name="Data Source Standards",
                description="Are the identified data sources standard for economics research?",
                levels=_standard_levels(
                    "Data sources are non-standard or inappropriate for economics",
                    "Some sources are standard but others are questionable",
                    "Most sources are standard economics data providers",
                    "Sources are well-established economics data providers",
                    "Sources represent the gold standard for the specific research domain",
                ),
            ),
            SubCriterion(
                name="Variable Construction",
                description="Are variables constructed following standard economics conventions?",
                levels=_standard_levels(
                    "Variable construction violates economics conventions",
                    "Some variables follow conventions but others are non-standard",
                    "Variables are generally conventional with minor issues",
                    "Variables are correctly constructed per economics standards",
                    "Variable construction reflects expert-level precision and field norms",
                ),
            ),
            SubCriterion(
                name="Temporal and Cross-Section Validity",
                description="Are time periods, frequencies, and cross-sectional units appropriate?",
                levels=_standard_levels(
                    "Temporal or cross-sectional choices are inappropriate",
                    "Some choices are valid but others raise concerns",
                    "Generally appropriate with some suboptimal choices",
                    "Time periods and units are well-chosen and justified",
                    "Optimal temporal and cross-sectional specifications with clear justification",
                ),
            ),
        ],
    ),
}


# =============================================================================
# RUBRIC REGISTRY
# =============================================================================

def get_rubric(dimension: str, team: str) -> EvaluationRubric:
    """Get the appropriate rubric for a dimension+team combination."""
    if dimension == "correctness":
        return CORRECTNESS_RUBRICS.get(team, CORRECTNESS_RUBRICS["IdeationTeam"])
    elif dimension == "soundness":
        return SOUNDNESS_RUBRICS.get(team, SOUNDNESS_RUBRICS["IdeationTeam"])
    elif dimension == "innovation_potential":
        return INNOVATION_RUBRIC
    elif dimension == "transparency":
        return TRANSPARENCY_RUBRIC
    elif dimension == "decision_quality":
        return DECISION_QUALITY_RUBRIC
    elif dimension == "economic_rigor":
        return ECONOMIC_RIGOR_RUBRICS.get(team, ECONOMIC_RIGOR_RUBRICS["IdeationTeam"])
    else:
        raise ValueError(f"No Tier-2 rubric defined for dimension: {dimension}")


# Dimensions that have Tier-2 LLM evaluation
LLM_EVALUATED_DIMENSIONS = [
    "correctness", "soundness", "innovation_potential", "transparency",
    "decision_quality", "economic_rigor",
]


# ---------------------------------------------------------------------------------------------
# Without these the Section-4 execution teams would fall through to the IdeationTeam rubrics —
# the judges would score generated CODE against research-question criteria. Team-specific rubrics
# for the three dimensions that key on team; the four shared rubrics apply as designed.
# ---------------------------------------------------------------------------------------------

CORRECTNESS_RUBRICS["CodeTeam"] = EvaluationRubric(
    dimension="correctness", team="CodeTeam",
    sub_criteria=[
        SubCriterion(
            name="Mathematical Fidelity",
            description="Do the emitted residual functions correspond one-to-one to the model's stated equations, with traceable provenance?",
            levels=_standard_levels(
                "Residuals bear no recognizable relation to the model's equations",
                "Several residuals are wrong or unattributable to any equation",
                "Most residuals match their equations; some provenance is unclear",
                "Residuals match the equations with clear provenance and minor gaps",
                "Every residual is a faithful, documented transcription of its equation",
            )),
        SubCriterion(
            name="Executable Integrity",
            description="Do the modules execute cleanly (finite residuals, convergence where the system is square, no runtime errors)?",
            levels=_standard_levels(
                "Modules crash or produce non-finite output at any evaluation point",
                "Modules import but fail most validation checks",
                "Modules execute with mixed validation results not fully explained",
                "Modules execute cleanly with validation results consistent with their verdicts",
                "Modules execute cleanly, converge where solvable, and every check outcome is explained",
            )),
        SubCriterion(
            name="Verdict Accuracy",
            description="Do the generated/partial/ungenerable verdicts match what the artifacts actually contain?",
            levels=_standard_levels(
                "Verdicts contradict the artifacts (e.g. 'generated' with no usable code)",
                "Verdicts frequently overstate or understate the artifact state",
                "Verdicts are broadly right with some unexplained mismatches",
                "Verdicts match the artifacts with reasons stated",
                "Verdicts are precise, reasoned, and fully consistent with the artifacts",
            )),
    ])

SOUNDNESS_RUBRICS["CodeTeam"] = EvaluationRubric(
    dimension="soundness", team="CodeTeam",
    sub_criteria=[
        SubCriterion(
            name="System Coherence",
            description="Are the unknowns genuine model variables (no expression-blobs as pseudo-variables; functional relations preserved)?",
            levels=_standard_levels(
                "Unknowns include expression-blobs or contradictory duplicates of the same quantity",
                "Several unknowns are artifacts of parsing rather than model variables",
                "Unknowns are mostly genuine with isolated incoherences",
                "Unknowns are genuine model variables; system dimensionality is reasoned",
                "The system is fully coherent: every unknown is a model variable and the equation/unknown accounting is explicit",
            )),
        SubCriterion(
            name="Validation Rigor",
            description="Does the executed check battery meaningfully cover compilation, finiteness, solvability, and parameter sensitivity?",
            levels=_standard_levels(
                "No meaningful validation was executed",
                "Only trivial checks ran; failures are unexplained",
                "The battery ran with partial coverage or shallow reporting",
                "The battery covers the key properties with clear per-check reporting",
                "Full battery with per-check detail, and every failure is diagnosed honestly",
            )),
        SubCriterion(
            name="Honest Scoping",
            description="Are non-algebraic constructs (expectations, integrals, inequalities) declined explicitly rather than silently approximated?",
            levels=_standard_levels(
                "Unsupported constructs are silently mangled into wrong algebra",
                "Some unsupported constructs leak through without disclosure",
                "Refusals happen but reasons are vague",
                "Refusals are explicit with per-equation reasons",
                "Scoping is exemplary: every exclusion is stated, reasoned, and reflected in the verdict",
            )),
    ])

ECONOMIC_RIGOR_RUBRICS["CodeTeam"] = EvaluationRubric(
    dimension="economic_rigor", team="CodeTeam",
    sub_criteria=[
        SubCriterion(
            name="Economic Interpretability",
            description="Do variables, parameters, and any solved steady state carry clear economic meaning?",
            levels=_standard_levels(
                "Symbols and solutions are economically meaningless or unlabeled",
                "Economic meaning is mostly obscured by machine naming",
                "Key quantities are interpretable; others are opaque",
                "Quantities are labeled and economically interpretable",
                "The module reads as an economic model: every quantity named, units and roles clear",
            )),
        SubCriterion(
            name="Comparative-Statics Sense",
            description="Where computed, do parameter sweeps have economically plausible signs and magnitudes, honestly reported?",
            levels=_standard_levels(
                "Sweeps are absent where promised, or report nonsense without comment",
                "Sweeps run but implausible results pass unremarked",
                "Sweeps are partially interpreted",
                "Sweeps are interpreted with plausible signs or honest flags where not",
                "Sweeps are complete, economically interpreted, and anomalies are diagnosed",
            )),
        SubCriterion(
            name="Model-Code Alignment",
            description="Does the code reflect this model's economics (its actual equations and calibration) rather than a generic template?",
            levels=_standard_levels(
                "The code is generic boilerplate detached from the model",
                "Alignment is superficial (titles only)",
                "The core block aligns; calibration or structure drifts",
                "Equations and calibration align with the model artifacts",
                "Full alignment: equations, calibration, and scoping decisions all trace to the model",
            )),
    ])

CORRECTNESS_RUBRICS["EstimationTeam"] = EvaluationRubric(
    dimension="correctness", team="EstimationTeam",
    sub_criteria=[
        SubCriterion(
            name="Specification-Data Fit",
            description="Does the specification use genuinely available series with appropriate transforms (stationarity, units, frequency)?",
            levels=_standard_levels(
                "The specification references unavailable data or economically meaningless constructions",
                "Several variables are mis-transformed or mismatched to their concepts",
                "The specification is usable with some questionable transform choices",
                "Variables and transforms are appropriate with minor caveats",
                "Every variable is well-chosen, correctly transformed, and matched to its concept",
            )),
        SubCriterion(
            name="Statistical Accuracy",
            description="Are estimates, standard errors, fit statistics, and test results internally consistent and correctly reported?",
            levels=_standard_levels(
                "Reported statistics are inconsistent or fabricated",
                "Multiple statistical inconsistencies",
                "Statistics are broadly consistent with isolated discrepancies",
                "Statistics are consistent and complete",
                "Statistics are consistent, complete, and reported with appropriate precision",
            )),
        SubCriterion(
            name="Verdict Accuracy",
            description="Does the estimated/fragile/inestimable verdict match the diagnostics and sample evidence?",
            levels=_standard_levels(
                "The verdict contradicts the evidence",
                "The verdict overstates the reliability of the fit",
                "The verdict is defensible but weakly justified",
                "The verdict follows from the diagnostics with stated reasons",
                "The verdict is precisely scoped to the evidence, with machine-readable reasons",
            )),
    ])

SOUNDNESS_RUBRICS["EstimationTeam"] = EvaluationRubric(
    dimension="soundness", team="EstimationTeam",
    sub_criteria=[
        SubCriterion(
            name="Identification Discipline",
            description="Is causal language proportionate to the design (association for OLS time series; instruments justified where used)?",
            levels=_standard_levels(
                "Causal claims are made from designs that cannot support them",
                "Causal language frequently outruns the design",
                "Language is mostly proportionate with slips",
                "Language is proportionate; endogeneity concerns are acknowledged",
                "Exemplary discipline: every claim names the variation that identifies it",
            )),
        SubCriterion(
            name="Diagnostics Rigor",
            description="Is a meaningful diagnostics battery executed and heeded (not just reported)?",
            levels=_standard_levels(
                "No diagnostics, or failures ignored in the verdict",
                "Diagnostics are token; severe issues unaddressed",
                "Diagnostics run; consequences partially drawn",
                "Diagnostics run and the verdict responds to them",
                "Full battery with verdict downgrades/flags exactly where the tests demand",
            )),
        SubCriterion(
            name="Robustness Coverage",
            description="Are robustness variants (leave-one-out, subsamples, covariance swaps) executed and stability honestly scored?",
            levels=_standard_levels(
                "No robustness analysis",
                "A single variant, or unstable results presented as stable",
                "Several variants with partial reporting",
                "A standard suite with honest stability scoring",
                "A full suite; instabilities and inestimable variants are reported as such",
            )),
    ])

ECONOMIC_RIGOR_RUBRICS["EstimationTeam"] = EvaluationRubric(
    dimension="economic_rigor", team="EstimationTeam",
    sub_criteria=[
        SubCriterion(
            name="Theory Linkage",
            description="Are the tested hypotheses genuinely derived from the model carried through the pipeline?",
            levels=_standard_levels(
                "Hypotheses are unrelated to the model",
                "Hypotheses are loosely inspired by the model",
                "Most hypotheses trace to the model",
                "Hypotheses trace to specific model mechanisms",
                "Every hypothesis names its mechanism and the restriction it implies",
            )),
        SubCriterion(
            name="Magnitude Interpretation",
            description="Are effect sizes interpreted economically (units, one-SD effects, elasticities), not just star-counted?",
            levels=_standard_levels(
                "No economic interpretation of magnitudes",
                "Interpretation is limited to significance stars",
                "Some magnitudes are interpreted",
                "Magnitudes are interpreted in economic units",
                "Magnitudes are interpreted and benchmarked against meaningful comparators",
            )),
        SubCriterion(
            name="Honest Negatives",
            description="Are unsupported hypotheses and weak fits reported as findings rather than buried or spun?",
            levels=_standard_levels(
                "Negative results are hidden or reframed as support",
                "Negatives are mentioned but minimized",
                "Negatives are reported without interpretation",
                "Negatives are reported and interpreted",
                "Negatives are treated as first-class findings with their implications drawn",
            )),
    ])

CORRECTNESS_RUBRICS["ReportingTeam"] = EvaluationRubric(
    dimension="correctness", team="ReportingTeam",
    sub_criteria=[
        SubCriterion(
            name="Number Fidelity",
            description="Do the report's numbers trace to upstream artifacts (consistency check clean; no unexplained figures)?",
            levels=_standard_levels(
                "Numbers are fabricated or contradict the artifacts",
                "Multiple numbers lack artifact support",
                "Numbers are mostly supported with isolated unverified values",
                "Numbers are supported; the consistency check is clean or explained",
                "Every number traces to an artifact and the check verifies it",
            )),
        SubCriterion(
            name="Citation Integrity",
            description="Are references real, drawn from the reviewed corpus, and actually used by the text?",
            levels=_standard_levels(
                "References are fabricated",
                "Many references are unverifiable or unused",
                "References are real; usage is uneven",
                "References are real, verified, and mostly load-bearing",
                "The bibliography is fully verified and every entry earns its place",
            )),
        SubCriterion(
            name="Coverage",
            description="Does the report represent every upstream stage (question, literature, model, data, code, estimation, limitations)?",
            levels=_standard_levels(
                "Whole stages are missing without acknowledgment",
                "Several stages are missing or token",
                "Most stages are covered; some thinly",
                "All stages are covered proportionately",
                "Coverage is complete, proportionate, and cross-referenced",
            )),
    ])

SOUNDNESS_RUBRICS["ReportingTeam"] = EvaluationRubric(
    dimension="soundness", team="ReportingTeam",
    sub_criteria=[
        SubCriterion(
            name="Evidence-Claim Alignment",
            description="Do narrative claims match the tables and verdicts they summarize?",
            levels=_standard_levels(
                "The narrative contradicts the evidence",
                "Claims frequently outrun the tables",
                "Claims broadly align with occasional overreach",
                "Claims align with the evidence",
                "Every claim is anchored to a specific table, verdict, or artifact",
            )),
        SubCriterion(
            name="Limitation Honesty",
            description="Are limitations specific, sourced from the stages that raised them, and proportionate?",
            levels=_standard_levels(
                "No limitations, or boilerplate disclaimers",
                "Generic limitations detached from the run",
                "Some run-specific limitations",
                "Limitations are run-specific and sourced",
                "Limitations are collected from every stage, specific, and honestly weighted",
            )),
        SubCriterion(
            name="Structural Discipline",
            description="Does the report follow method-results-interpretation discipline without interleaving or forward-reference clutter?",
            levels=_standard_levels(
                "The report is disorganized to the point of obscuring findings",
                "Structure is weak; interpretation and results interleave confusingly",
                "Structure is serviceable",
                "Structure is clean and conventional",
                "Structure is exemplary: a reader can audit any claim in one hop",
            )),
    ])

ECONOMIC_RIGOR_RUBRICS["ReportingTeam"] = EvaluationRubric(
    dimension="economic_rigor", team="ReportingTeam",
    sub_criteria=[
        SubCriterion(
            name="Economic Narrative",
            description="Are results interpreted through economic mechanisms rather than restated statistically?",
            levels=_standard_levels(
                "No economic interpretation",
                "Interpretation restates coefficients in words",
                "Some mechanism-level interpretation",
                "Results are interpreted through the model's mechanisms",
                "Interpretation connects results, mechanisms, and the literature coherently",
            )),
        SubCriterion(
            name="Caveat Proportionality",
            description="Do hedges match the strength of the evidence (neither overclaiming nor drowning results in disclaimers)?",
            levels=_standard_levels(
                "Systematic overclaiming or self-nullifying hedging",
                "Hedging is frequently disproportionate",
                "Hedging is mostly proportionate",
                "Hedging is proportionate throughout",
                "Calibration of confidence is exemplary, claim by claim",
            )),
        SubCriterion(
            name="Policy Restraint",
            description="Are policy statements scoped to what the design supports (metrics named; no general recommendations from prototypes)?",
            levels=_standard_levels(
                "Sweeping policy claims from unsupportive designs",
                "Policy language regularly exceeds the evidence",
                "Policy language is mostly scoped",
                "Policy language is scoped and metric-named",
                "Policy content is precisely scoped, metric-named, and flagged as prototype-grade",
            )),
    ])
