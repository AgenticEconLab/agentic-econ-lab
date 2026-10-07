# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Research Question Refinement Stage with Human-in-the-Loop
This script uses two specialized agents to refine research questions based on literature.

Agents:
- Ideator: Generates research concepts and ideas from literature
- Refiner: Formulates precise, actionable research questions

Input: Literature results from 1-SourcingStage.py (CSV file)
Output: Refined research questions with two-round feedback process
"""

import os
import sys
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
# Add parent directories to path for shared imports
from pathlib import Path as _Path
_agents_dir = _Path(__file__).resolve().parent.parent.parent.parent
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

from shared.llm import LLMClient
from shared.observability import MetricsCollector
from shared.auto_input import auto_input, get_default
from IdeationTeam.ael.schemas.stage_outputs import ResearchConcept, EvolutionTrace, ResearchQuestion, ConceptList, QuestionList, HumanFeedback, StageTransitionSummary

# Load environment variables
load_dotenv()


class BaseRefinementAgent:
    """Base class for refinement agents."""

    def __init__(self, agent_name: str, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = agent_name
        self.api_key = openai_api_key
        self.collector = collector
        self.llm = LLMClient(
            temperature=0.7,
            api_key=self.api_key,
            collector=collector,
            agent_name=agent_name,
        )
    
    def process(self, literature_df: pd.DataFrame, feedback: Optional[HumanFeedback] = None):
        """Process literature and generate output. To be implemented by subclasses."""
        raise NotImplementedError


class Ideator(BaseRefinementAgent):
    """Agent that generates research concepts from literature."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        super().__init__("Ideator", openai_api_key, collector=collector)
    
    def generate_concepts(
        self, 
        literature_df: pd.DataFrame, 
        feedback: Optional[HumanFeedback] = None,
        num_concepts: int = 8
    ) -> List[ResearchConcept]:
        """Generate research concepts from literature."""
        
        print(f"\n[{self.agent_name}] Generating research concepts from {len(literature_df)} papers...")
        
        # Prepare literature summary
        lit_summary = self._prepare_literature_summary(literature_df)
        
        # Prepare feedback context
        feedback_context = ""
        if feedback:
            feedback_context = f"""
            Previous feedback:
            - Promising concepts: {', '.join(feedback.promising_items[:5])}
            - Weak concepts: {', '.join(feedback.weak_items[:3])}
            - Missing angles: {', '.join(feedback.missing_angles)}
            - Focus directions: {', '.join(feedback.focus_directions)}
            - Comments: {feedback.comments}
            """
        
        result = self.llm.format_and_invoke(
            system_prompt="You are Ideator, an expert at identifying novel research concepts from academic literature. Before your conclusion, populate `reasoning` with the evidence you weighed and `alternatives_considered` with >=2 framings you rejected and why. You must respond with valid JSON only.",
            user_prompt="""Based on the following literature, generate innovative research concepts.

            Literature Summary:
            {literature_summary}

            {feedback_context}

            Generate {num_concepts} research concepts that:
            - Identify gaps in current research
            - Synthesize insights across multiple papers
            - Propose novel angles or perspectives
            - Are grounded in the literature provided
            - Have potential for significant contribution

            IMPORTANT INNOVATION REQUIREMENTS:
            - At least 2 concepts MUST combine insights from DIFFERENT economic subfields
              (e.g., behavioral economics + international trade, labor economics + environmental economics)
            - At least 1 concept MUST challenge conventional economic wisdom or propose a counterintuitive hypothesis;
              for that concept, populate "novelty_claim" stating EXPLICITLY what is novel and what RISK it carries
            - For each concept, explicitly state what makes it NOVEL compared to existing literature

            For each concept, provide:
            - A clear, concise title
            - Detailed description (2-3 sentences)
            - Key themes (3-5 themes)
            - Supporting literature (list 2-4 paper titles from the summary)

            Respond with a JSON object containing:
            - "concepts": array of concept objects, each with "concept_title", "description", "key_themes", "literature_support", "reasoning" (evidence weighed), "alternatives_considered" (>=2 rejected framings, each with a reason), and "novelty_claim" (what is novel + the risk it carries)

            Example format:
            {{
              "concepts": [
                {{
                  "concept_title": "Concept Title",
                  "description": "Detailed description...",
                  "key_themes": ["theme1", "theme2", "theme3"],
                  "literature_support": ["Paper A", "Paper B"]
                }}
              ]
            }}

            Respond with ONLY the JSON object, no other text.
            """,
            variables={
                "literature_summary": lit_summary,
                "feedback_context": feedback_context,
                "num_concepts": str(num_concepts),
            },
            parse_as=ConceptList,
        )

        # Score novelty
        concepts = result.concepts
        for i, concept in enumerate(concepts):
            concept.novelty_score = self._score_novelty(concept, literature_df)
        
        print(f"[{self.agent_name}] Generated {len(concepts)} concepts")
        return concepts
    
    def _prepare_literature_summary(self, df: pd.DataFrame, max_papers: int = 30) -> str:
        """Prepare a summary of literature for the prompt."""
        # Sort by relevance score if available
        if 'relevance_score' in df.columns:
            df_sorted = df.sort_values('relevance_score', ascending=False).head(max_papers)
        else:
            df_sorted = df.head(max_papers)
        
        summary_parts = []
        for idx, row in df_sorted.iterrows():
            # Coerce possibly-missing/NaN (float) cells to safe strings before slicing.
            _abs = row.get('abstract')
            _abstract = str(_abs)[:300] if pd.notna(_abs) and str(_abs).strip() else 'N/A'
            paper_summary = f"""
Paper {idx + 1}: {row.get('title', 'N/A')}
Authors: {row.get('authors', 'N/A')}
Year: {row.get('year', 'N/A')}
Abstract: {_abstract}...
"""
            summary_parts.append(paper_summary)
        
        return "\n".join(summary_parts)
    
    def _score_novelty(self, concept: ResearchConcept, literature_df: pd.DataFrame) -> float:
        """Score concept novelty from real signals and record auditable sub-scores.

        Components (each 0-1, then averaged into novelty_score):
        - theme_breadth: how many distinct themes the concept spans (proxy for
          cross-domain synthesis), normalized to 5 themes.
        - lexical_novelty: fraction of concept terms (title + themes) that do NOT
          already appear in the retrieved-literature corpus — uncommon vocabulary
          relative to the prior art is a proxy for genuine novelty.
        - synthesis_breadth: how many distinct supporting papers the concept draws
          on (synthesizing across more sources is harder/more novel), capped at 4.
        Sub-scores are written to concept.novelty_components for auditability.
        """
        import re

        # Build a vocabulary of terms already present in the retrieved literature.
        corpus_tokens: set = set()
        for col in ("title", "abstract", "keywords"):
            if col in literature_df.columns:
                for val in literature_df[col].tolist():
                    text = str(val if val is not None else "").lower()
                    corpus_tokens.update(re.findall(r"[a-z]{4,}", text))

        # Theme breadth: distinct themes, normalized.
        distinct_themes = {t.strip().lower() for t in (concept.key_themes or []) if t and t.strip()}
        theme_breadth = min(len(distinct_themes) / 5.0, 1.0)

        # Lexical novelty: concept terms not seen in the literature corpus.
        concept_text = " ".join([concept.concept_title or ""] + list(concept.key_themes or []))
        concept_tokens = set(re.findall(r"[a-z]{4,}", concept_text.lower()))
        if concept_tokens and corpus_tokens:
            unseen = concept_tokens - corpus_tokens
            lexical_novelty = len(unseen) / len(concept_tokens)
        elif concept_tokens:
            # No corpus to compare against -> treat as moderately novel.
            lexical_novelty = 0.5
        else:
            lexical_novelty = 0.0

        # Synthesis breadth: distinct supporting papers, capped.
        distinct_support = {s.strip().lower() for s in (concept.literature_support or []) if s and s.strip()}
        synthesis_breadth = min(len(distinct_support) / 4.0, 1.0)

        components = {
            "theme_breadth": round(theme_breadth, 3),
            "lexical_novelty": round(lexical_novelty, 3),
            "synthesis_breadth": round(synthesis_breadth, 3),
        }
        # Weighted: lexical novelty is the strongest signal of true novelty.
        score = 0.5 * lexical_novelty + 0.3 * theme_breadth + 0.2 * synthesis_breadth
        concept.novelty_components = components
        return round(min(max(score, 0.0), 1.0), 2)


class Refiner(BaseRefinementAgent):
    """Agent that formulates precise research questions from concepts."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        super().__init__("Refiner", openai_api_key, collector=collector)
    
    def formulate_questions(
        self,
        concepts: List[ResearchConcept],
        literature_df: pd.DataFrame,
        feedback: Optional[HumanFeedback] = None,
        num_questions: int = 6
    ) -> List[ResearchQuestion]:
        """Formulate research questions from concepts."""
        
        print(f"\n[{self.agent_name}] Formulating research questions from {len(concepts)} concepts...")
        
        # Prepare concepts summary
        concepts_summary = self._prepare_concepts_summary(concepts)
        
        # Prepare feedback context
        feedback_context = ""
        if feedback:
            feedback_context = f"""
            Previous feedback:
            - Promising questions: {', '.join(feedback.promising_items[:5])}
            - Weak questions: {', '.join(feedback.weak_items[:3])}
            - Missing angles: {', '.join(feedback.missing_angles)}
            - Focus directions: {', '.join(feedback.focus_directions)}
            - Comments: {feedback.comments}
            """
        
        result = self.llm.format_and_invoke(
            system_prompt="You are Refiner, an expert at formulating precise, actionable research questions. Before your conclusion, populate `reasoning` with the evidence you weighed and `alternatives_considered` with >=2 question formulations you rejected and why. You must respond with valid JSON only.",
            user_prompt="""Based on the following research concepts, formulate specific research questions.

            Research Concepts:
            {concepts_summary}

            {feedback_context}

            Generate {num_questions} research questions that are:
            - Specific and well-defined
            - Answerable through empirical or theoretical research
            - Novel and contribute to the field
            - Feasible with available methods
            - Grounded in the concepts provided

            For each question:
            - State the question clearly and concisely
            - Provide rationale (why it's important)
            - Suggest 2-4 methodological approaches
            - Link to related concept titles

            TRANSPARENCY REQUIREMENTS (critical for reproducibility):
            - For each question, explain WHY this question was prioritized over alternative formulations
            - Describe what ALTERNATIVE questions were considered and why they were less suitable
            - State the ASSUMPTIONS underlying the question and conditions under which it might not apply
            - Clearly identify the DECISION CRITERIA used to rank question importance

            EVOLUTION TRACE (critical for intermediate step visibility):
            For each question, provide an evolution_trace documenting:
            - source_ideas: Which specific concept titles from the input this question builds upon (use exact concept titles)
            - refinement_steps: 2-4 steps describing HOW the source ideas were transformed into this question (e.g., "Combined concept A and B around shared theme of X", "Narrowed scope from broad Y to specific Z", "Added methodological grounding in agent-based modeling")
            - alternatives_considered: 1-3 alternative question formulations that were considered but rejected, with brief reasons (e.g., "How does X impact Z? (rejected: too broad, lacks methodological specificity)")

            CITATION CORRECTNESS: Do NOT fabricate specific author names or paper titles. Instead of citing "Smith et al. (2020)", reference established theoretical frameworks and fields (e.g., "building on endogenous growth theory" or "extending the New Keynesian framework"). This ensures all references are verifiable.

            Respond with a JSON object containing:
            - "questions": array of question objects, each with "question", "rationale", "methodology_hints", "related_concepts", "selection_rationale", "assumptions", "limitations", "evolution_trace", "reasoning" (evidence weighed), and "alternatives_considered" (>=2 rejected formulations, each with a reason)

            Example format:
            {{
              "questions": [
                {{
                  "question": "How does X affect Y under conditions Z?",
                  "rationale": "This question addresses a gap...",
                  "methodology_hints": ["Agent-based modeling", "Econometric analysis"],
                  "related_concepts": ["Concept Title 1", "Concept Title 2"],
                  "selection_rationale": "Prioritized over broader formulations because...",
                  "assumptions": ["Assumes rational agent behavior", "Assumes data availability for Z"],
                  "limitations": ["Limited to developed economies", "Does not account for structural breaks"],
                  "evolution_trace": {{
                    "source_ideas": ["Concept Title 1", "Concept Title 2"],
                    "refinement_steps": [
                      "Combined insights from Concept 1 (theme X) and Concept 2 (theme Y)",
                      "Narrowed scope from general impact to specific mechanism under conditions Z",
                      "Added agent-based modeling as primary methodology based on complexity of interactions"
                    ],
                    "alternatives_considered": [
                      "What is the relationship between X and Y? (rejected: too vague, lacks causal direction)",
                      "Does X cause Y in all economies? (rejected: too broad, infeasible to test universally)"
                    ]
                  }}
                }}
              ]
            }}

            Respond with ONLY the JSON object, no other text.
            """,
            variables={
                "concepts_summary": concepts_summary,
                "feedback_context": feedback_context,
                "num_questions": str(num_questions),
            },
            parse_as=QuestionList,
        )

        # Score feasibility
        questions = result.questions
        for question in questions:
            question.feasibility_score = self._score_feasibility(question)
        
        print(f"[{self.agent_name}] Formulated {len(questions)} questions")
        return questions
    
    def _prepare_concepts_summary(self, concepts: List[ResearchConcept]) -> str:
        """Prepare a summary of concepts for the prompt."""
        summary_parts = []
        for i, concept in enumerate(concepts):
            concept_summary = f"""
Concept {i + 1}: {concept.concept_title}
Description: {concept.description}
Key Themes: {', '.join(concept.key_themes)}
Literature Support: {', '.join(concept.literature_support)}
Novelty Score: {concept.novelty_score}
"""
            summary_parts.append(concept_summary)
        
        return "\n".join(summary_parts)
    
    def _score_feasibility(self, question: ResearchQuestion) -> float:
        """Score question feasibility from real signals and record sub-scores.

        Components (each 0-1, averaged into feasibility_score):
        - method_count: number of distinct methodological approaches proposed,
          normalized to 4 (more concrete routes = more feasible to execute).
        - method_specificity: average descriptiveness of the methodology hints
          (multi-word, named methods are more actionable than one-word stubs).
        - scoping_clarity: whether the question states assumptions and limitations
          (a well-scoped question with stated boundary conditions is more feasible).
        Sub-scores are written to question.feasibility_components for auditability.
        """
        methods = [m.strip() for m in (question.methodology_hints or []) if m and m.strip()]
        distinct_methods = list({m.lower(): m for m in methods}.values())

        method_count = min(len(distinct_methods) / 4.0, 1.0)

        # Specificity: reward named/multi-word methods over single-word stubs.
        if distinct_methods:
            specific = sum(1 for m in distinct_methods if len(m.split()) >= 2)
            method_specificity = specific / len(distinct_methods)
        else:
            method_specificity = 0.0

        # Scoping clarity: assumptions + limitations stated => better-bounded study.
        scoping_signals = 0
        if question.assumptions:
            scoping_signals += 1
        if question.limitations:
            scoping_signals += 1
        scoping_clarity = scoping_signals / 2.0

        components = {
            "method_count": round(method_count, 3),
            "method_specificity": round(method_specificity, 3),
            "scoping_clarity": round(scoping_clarity, 3),
        }
        score = 0.4 * method_count + 0.35 * method_specificity + 0.25 * scoping_clarity
        question.feasibility_components = components
        return round(min(max(score, 0.0), 1.0), 2)


class RefinementOrchestrator:
    """Orchestrates the two-agent refinement process with human feedback."""

    def __init__(self, quiet: bool = False, collector: Optional[MetricsCollector] = None):
        self.api_key = os.getenv("OPENAI_API_KEY")

        self.collector = collector

        # Quiet mode suppresses verbose output (use with ConsoleUI)
        self.quiet = quiet or os.environ.get("AGENT_QUIET_MODE", "").lower() == "true"

        self.ideator = Ideator(self.api_key, collector=collector)
        self.refiner = Refiner(self.api_key, collector=collector)

        self.all_concepts = []
        self.all_questions = []
        self.round_results = {}
        self.concepts = []  # For compatibility
        self.stage_transition_summaries = {}  # Track per-round transition summaries

    def load_literature(self, csv_path: str) -> pd.DataFrame:
        """Load literature from CSV file."""
        print(f"\n[Orchestrator] Loading literature from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"[Orchestrator] Loaded {len(df)} papers")
        return df
    
    def run_refinement_round(
        self,
        literature_df: pd.DataFrame,
        round_number: int,
        feedback: Optional[HumanFeedback] = None,
        num_concepts: int = 8,
        num_questions: int = 6
    ) -> Tuple[List[ResearchConcept], List[ResearchQuestion]]:
        """Run one round of refinement (Ideator → Refiner)."""
        
        print(f"\n{'='*70}")
        print(f"ROUND {round_number}: Research Question Refinement")
        print(f"{'='*70}")
        
        # Step 1: Ideator generates concepts
        concepts = self.ideator.generate_concepts(
            literature_df=literature_df,
            feedback=feedback,
            num_concepts=num_concepts
        )
        
        # Step 2: Refiner formulates questions from concepts
        questions = self.refiner.formulate_questions(
            concepts=concepts,
            literature_df=literature_df,
            feedback=feedback,
            num_questions=num_questions
        )
        
        # Store results
        self.round_results[round_number] = {
            'concepts': concepts,
            'questions': questions
        }

        # Compute stage transition summary
        concept_titles = [c.concept_title for c in concepts]
        cluster_preview = ', '.join(concept_titles[:4])
        if len(concept_titles) > 4:
            cluster_preview += f", ... (+{len(concept_titles) - 4} more)"
        transition_summary = StageTransitionSummary(
            sourcing_input_count=len(literature_df),
            refinement_clusters=len(concepts),
            final_questions=len(questions),
            filtering_rationale=(
                f"Processed {len(literature_df)} source papers into {len(concepts)} thematic "
                f"concept clusters ({cluster_preview}); refined into {len(questions)} research "
                f"questions based on novelty, feasibility, and alignment with research topic. "
                f"Filtering criteria: empirical testability, methodological feasibility, "
                f"gap significance, and cross-domain novelty."
            )
        )
        self.stage_transition_summaries[round_number] = transition_summary

        # Add to all results
        self.all_concepts.extend(concepts)
        self.all_questions.extend(questions)

        return concepts, questions

    def print_concepts(self, round_number: int, top_n: int = 10):
        """Print concepts from a specific round."""
        if round_number not in self.round_results:
            print(f"No results for round {round_number}")
            return
        
        concepts = self.round_results[round_number]['concepts']
        
        print(f"\n{'='*70}")
        print(f"ROUND {round_number} - RESEARCH CONCEPTS (Top {min(top_n, len(concepts))})")
        print(f"{'='*70}\n")
        
        # Sort by novelty score
        sorted_concepts = sorted(concepts, key=lambda x: x.novelty_score or 0, reverse=True)
        
        for i, concept in enumerate(sorted_concepts[:top_n], 1):
            print(f"{i}. {concept.concept_title}")
            print(f"   Novelty: {concept.novelty_score}")
            print(f"   Description: {concept.description}")
            print(f"   Key Themes: {', '.join(concept.key_themes)}")
            print(f"   Literature Support: {', '.join(concept.literature_support[:3])}")
            print()
    
    def print_questions(self, round_number: int, top_n: int = 10):
        """Print questions from a specific round."""
        if round_number not in self.round_results:
            print(f"No results for round {round_number}")
            return
        
        questions = self.round_results[round_number]['questions']
        
        print(f"\n{'='*70}")
        print(f"ROUND {round_number} - RESEARCH QUESTIONS (Top {min(top_n, len(questions))})")
        print(f"{'='*70}\n")
        
        # Sort by feasibility score
        sorted_questions = sorted(questions, key=lambda x: x.feasibility_score or 0, reverse=True)
        
        for i, question in enumerate(sorted_questions[:top_n], 1):
            print(f"{i}. {question.question}")
            print(f"   Feasibility: {question.feasibility_score}")
            print(f"   Rationale: {question.rationale}")
            print(f"   Methodologies: {', '.join(question.methodology_hints)}")
            print(f"   Related Concepts: {', '.join(question.related_concepts)}")
            print()
    
    def collect_human_feedback(self, round_number: int) -> HumanFeedback:
        """Collect human feedback interactively."""
        print(f"\n{'='*70}")
        print(f"FEEDBACK COLLECTION - Round {round_number}")
        print(f"{'='*70}\n")
        
        print("Please provide feedback on the concepts and questions generated.")
        print("Press Enter to skip any field.\n")
        
        promising = auto_input(
            "Enter promising concepts/questions (comma-separated): ",
            default=get_default("promising_items"),
        ).strip()
        promising_items = [x.strip() for x in promising.split(",")] if promising else []

        weak = auto_input(
            "Enter weak or unfeasible items (comma-separated): ",
            default=get_default("weak_items"),
        ).strip()
        weak_items = [x.strip() for x in weak.split(",")] if weak else []

        missing = auto_input(
            "Enter missing angles or perspectives (comma-separated): ",
            default=get_default("missing_angles"),
        ).strip()
        missing_angles = [x.strip() for x in missing.split(",")] if missing else []

        focus = auto_input(
            "Enter directions to focus on (comma-separated): ",
            default=get_default("focus_directions"),
        ).strip()
        focus_directions = [x.strip() for x in focus.split(",")] if focus else []

        comments = auto_input(
            "General comments: ",
            default=get_default("general_comments"),
        ).strip()
        
        feedback = HumanFeedback(
            round_number=round_number,
            promising_items=promising_items,
            weak_items=weak_items,
            missing_angles=missing_angles,
            focus_directions=focus_directions,
            comments=comments
        )
        
        # Save feedback to JSON
        feedback_file = os.path.join(getattr(self, "output_dir", None) or ".", f"round{round_number}_refinement_feedback.json")
        with open(feedback_file, 'w') as f:
            json.dump(feedback.model_dump(), f, indent=2)
        
        print(f"\n[Orchestrator] Feedback saved to {feedback_file}")
        
        return feedback
    
    def run_automated_refinement(
        self,
        literature_df: pd.DataFrame,
        num_concepts: int = 10,
        num_questions: int = 8
    ) -> List[ResearchQuestion]:
        """Run automated single-round refinement without human feedback."""
        print(f"\n{'='*70}")
        print(f"AUTOMATED RESEARCH QUESTION REFINEMENT")
        print(f"{'='*70}")
        
        # Step 1: Ideator generates concepts
        concepts = self.ideator.generate_concepts(
            literature_df=literature_df,
            feedback=None,
            num_concepts=num_concepts
        )
        
        # Step 2: Refiner formulates questions from concepts
        questions = self.refiner.formulate_questions(
            concepts=concepts,
            literature_df=literature_df,
            feedback=None,
            num_questions=num_questions
        )
        
        # Store results
        self.all_concepts = concepts
        self.all_questions = questions

        # Compute stage transition summary
        concept_titles = [c.concept_title for c in concepts]
        cluster_preview = ', '.join(concept_titles[:4])
        if len(concept_titles) > 4:
            cluster_preview += f", ... (+{len(concept_titles) - 4} more)"
        self.stage_transition_summaries[0] = StageTransitionSummary(
            sourcing_input_count=len(literature_df),
            refinement_clusters=len(concepts),
            final_questions=len(questions),
            filtering_rationale=(
                f"Processed {len(literature_df)} source papers into {len(concepts)} thematic "
                f"concept clusters ({cluster_preview}); refined into {len(questions)} research "
                f"questions based on novelty, feasibility, and alignment with research topic. "
                f"Filtering criteria: empirical testability, methodological feasibility, "
                f"gap significance, and cross-domain novelty."
            )
        )

        print(f"\n[Orchestrator] Generated {len(concepts)} concepts and {len(questions)} questions")

        return questions

    def print_concepts(self, top_n: int = 10):
        """Print concepts."""
        concepts = self.all_concepts
        
        print(f"\n{'='*70}")
        print(f"RESEARCH CONCEPTS (Top {min(top_n, len(concepts))})")
        print(f"{'='*70}\n")
        
        # Sort by novelty score
        sorted_concepts = sorted(concepts, key=lambda x: x.novelty_score or 0, reverse=True)
        
        for i, concept in enumerate(sorted_concepts[:top_n], 1):
            print(f"{i}. {concept.concept_title}")
            print(f"   Novelty: {concept.novelty_score}")
            print(f"   Description: {concept.description}")
            print(f"   Key Themes: {', '.join(concept.key_themes)}")
            print(f"   Literature Support: {', '.join(concept.literature_support[:3])}")
            print()
    
    def print_questions(self, top_n: int = 10):
        """Print questions."""
        questions = self.all_questions
        
        print(f"\n{'='*70}")
        print(f"RESEARCH QUESTIONS (Top {min(top_n, len(questions))})")
        print(f"{'='*70}\n")
        
        # Sort by feasibility score
        sorted_questions = sorted(questions, key=lambda x: x.feasibility_score or 0, reverse=True)
        
        for i, question in enumerate(sorted_questions[:top_n], 1):
            print(f"{i}. {question.question}")
            print(f"   Feasibility: {question.feasibility_score}")
            print(f"   Rationale: {question.rationale}")
            print(f"   Methodologies: {', '.join(question.methodology_hints)}")
            print(f"   Related Concepts: {', '.join(question.related_concepts)}")
            print()
    
    def save_results(self, filename: str, round_number: Optional[int] = None):
        """Save concepts and questions to JSON file."""
        if round_number:
            # Save specific round
            if round_number not in self.round_results:
                print(f"No results for round {round_number}")
                return
            
            data = {
                'round': round_number,
                'concepts': [c.model_dump() for c in self.round_results[round_number]['concepts']],
                'questions': [q.model_dump() for q in self.round_results[round_number]['questions']]
            }
            output_file = f"round{round_number}_{filename}"
        else:
            # Save all rounds
            _concepts_dump = [c.model_dump() for c in self.all_concepts]
            _questions_dump = [q.model_dump() for q in self.all_questions]
            # Emit both canonical schema keys ('concepts'/'questions' per
            # RefinementStageOutput) and legacy 'all_*' keys used by existing
            # downstream readers (e.g. 3-IntegrationStage.load_refinement_results).
            data = {
                'concepts': _concepts_dump,
                'questions': _questions_dump,
                'all_concepts': _concepts_dump,
                'all_questions': _questions_dump,
                'by_round': {
                    str(rnd): {
                        'concepts': [c.model_dump() for c in results['concepts']],
                        'questions': [q.model_dump() for q in results['questions']]
                    }
                    for rnd, results in self.round_results.items()
                }
            }
            output_file = filename

        # Add stage transition summaries
        if self.stage_transition_summaries:
            if round_number and round_number in self.stage_transition_summaries:
                data['stage_transition_summary'] = self.stage_transition_summaries[round_number].model_dump()
            elif not round_number:
                data['stage_transition_summaries'] = {
                    str(rnd): summary.model_dump()
                    for rnd, summary in self.stage_transition_summaries.items()
                }

        # Add process-level transparency metadata
        data['process_assumptions'] = [
            "Research topics provided are within the scope of current economics research",
            "Available literature is representative of the state of the field",
            "LLM-generated questions reflect genuine research opportunities, not artifacts of training data",
            "Feasibility scores assume standard academic resources and data access"
        ]
        data['process_limitations'] = [
            "Question generation is bounded by the LLM's training knowledge cutoff",
            "Novelty assessment may not account for very recent unpublished work",
            "Cross-disciplinary questions may favor economics-adjacent fields represented in training data",
            "Human-in-the-loop feedback (if used) reflects individual reviewer preferences"
        ]

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[Orchestrator] Results saved to {output_file}")


def main():
    """Main function with two-round refinement process."""
    
    # Configuration
    literature_csv = "literature_results_all_rounds.csv"  # Output from Stage 1
    
    print("="*70)
    print("STARTING TWO-ROUND RESEARCH QUESTION REFINEMENT")
    print("="*70)
    
    # Initialize orchestrator
    orchestrator = RefinementOrchestrator()
    
    # Load literature from Stage 1
    try:
        literature_df = orchestrator.load_literature(literature_csv)
    except FileNotFoundError:
        print(f"\nError: {literature_csv} not found!")
        print("Please run 1-SourcingStage.py first to generate literature results.")
        return
    
    # ROUND 1: Initial concept generation and question formulation
    concepts1, questions1 = orchestrator.run_refinement_round(
        literature_df=literature_df,
        round_number=1,
        feedback=None,
        num_concepts=8,
        num_questions=6
    )
    
    # Display Round 1 results
    orchestrator.print_concepts(round_number=1, top_n=8)
    orchestrator.print_questions(round_number=1, top_n=6)
    
    # Save Round 1 results
    orchestrator.save_results("refinement_results.json", round_number=1)
    
    # Collect human feedback
    feedback = orchestrator.collect_human_feedback(round_number=1)
    
    # ROUND 2: Refined concepts and questions based on feedback
    concepts2, questions2 = orchestrator.run_refinement_round(
        literature_df=literature_df,
        round_number=2,
        feedback=feedback,
        num_concepts=8,
        num_questions=6
    )
    
    # Display Round 2 results
    orchestrator.print_concepts(round_number=2, top_n=8)
    orchestrator.print_questions(round_number=2, top_n=6)
    
    # Save Round 2 results
    orchestrator.save_results("refinement_results.json", round_number=2)
    
    # Save all results
    orchestrator.save_results("refinement_results_all_rounds.json")
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Round 1: {len(concepts1)} concepts, {len(questions1)} questions")
    print(f"Round 2: {len(concepts2)} concepts, {len(questions2)} questions")
    print(f"Total unique concepts: {len(orchestrator.all_concepts)}")
    print(f"Total unique questions: {len(orchestrator.all_questions)}")
    
    print(f"\n{'='*70}")
    print("PROCESS COMPLETE")
    print(f"{'='*70}")
    print("Files created:")
    print("  - round1_refinement_results.json")
    print("  - round1_refinement_feedback.json")
    print("  - round2_refinement_results.json")
    print("  - refinement_results_all_rounds.json")


if __name__ == "__main__":
    main()
