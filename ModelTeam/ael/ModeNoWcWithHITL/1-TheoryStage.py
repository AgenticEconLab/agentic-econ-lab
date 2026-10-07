# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Theory Stage - Automated Mode (No Firecrawl, No HITL)
This script develops theoretical frameworks based on research questions and literature.

Pipeline:
1. Theorist: Conceptual architecture development (assumptions, basic formulae, theoretical framework)

Input: Research questions + Literature review/batch from LiteratureTeam
Output: Theoretical framework with assumptions, equations, and conceptual model
"""

import os
import json
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
# Add parent directories to path for shared imports
import sys
from pathlib import Path as _Path
_agents_dir = _Path(__file__).resolve().parent.parent.parent.parent
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

from shared.llm import LLMClient
from shared.observability import MetricsCollector
from shared.tools.register_all import register_all_tools
from shared.tools.tool_registry import ToolRegistry
from shared.json_repair import repair_json
from shared.auto_input import auto_input, get_default
from ModelTeam.ael.schemas.stage_inputs import ResearchQuestion, LiteratureItem
from ModelTeam.ael.schemas.stage_outputs import (
    TheoreticalAssumption, MathematicalFormulation, ConceptualComponent,
    TheoreticalFramework, TheoryStageOutput,
)

# Load environment variables
load_dotenv()

# Register all tools in the ToolRegistry
register_all_tools()


# ========== AGENT ==========

class Theorist:
    """Agent for developing theoretical frameworks and conceptual architectures."""
    
    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "Theorist"
        self.api_key = openai_api_key
        self.collector = collector
        self.llm = LLMClient(
            temperature=0.4,
            api_key=self.api_key,
            collector=collector,
            agent_name=self.agent_name,
        )
    
    def develop_framework(
        self,
        research_question: ResearchQuestion,
        literature_items: List[Dict],
        literature_insights: Optional[List[Dict]] = None
    ) -> TheoreticalFramework:
        """Develop a theoretical framework for a research question."""
        
        print(f"\n[{self.agent_name}] Developing theoretical framework for:")
        print(f"  {research_question.question[:80]}...")
        
        # Step 1: Identify theoretical assumptions
        print(f"  Step 1: Identifying theoretical assumptions...")
        assumptions = self._identify_assumptions(research_question, literature_items)
        
        # Step 2: Define conceptual components
        print(f"  Step 2: Defining conceptual components...")
        components = self._define_components(research_question, literature_items, assumptions)
        
        # Step 3: Formulate mathematical structure
        print(f"  Step 3: Formulating mathematical structure...")
        formulations = self._formulate_mathematics(research_question, assumptions, components)
        
        # Step 4: Synthesize complete framework
        print(f"  Step 4: Synthesizing complete framework...")
        framework = self._synthesize_framework(
            research_question,
            assumptions,
            components,
            formulations,
            literature_items,
            literature_insights
        )
        
        print(f"[{self.agent_name}] Framework developed: {framework.framework_title}")
        return framework
    
    def _identify_assumptions(
        self,
        research_question: ResearchQuestion,
        literature_items: List[Dict]
    ) -> List[TheoreticalAssumption]:
        """Identify key theoretical assumptions."""
        
        # Prepare literature context
        lit_context = self._prepare_literature_context(literature_items, max_papers=10)

        # Search arXiv for relevant theoretical papers
        arxiv_context = ""
        try:
            query_text = research_question.question[:100]
            _tool_result = ToolRegistry.invoke("arxiv_search", {
                "query": query_text, "max_results": 5, "sort_by": "relevance"
            }, collector=self.collector, agent=self.agent_name)
            arxiv_results = _tool_result.data if _tool_result.success else []
            if arxiv_results:
                arxiv_parts = []
                for i, paper in enumerate(arxiv_results, 1):
                    title = paper.get('title', 'Unknown')
                    summary = paper.get('abstract', '')[:200]
                    arxiv_parts.append(f"{i}. {title}\n   {summary}...")
                arxiv_context = "\n\n".join(arxiv_parts)
        except Exception as e:
            print(f"    arXiv search failed (non-critical): {e}")

        try:
            result = self.llm.invoke([
                {"role": "system", "content": "You are an expert economic theorist specializing in formal modeling and theoretical frameworks."},
                {"role": "user", "content": """Identify 5-8 key theoretical assumptions needed to address this research question.

Research Question: {question}
Theoretical Framework: {framework}
Methodologies: {methodologies}

Literature Context:
{literature_context}

arXiv Papers:
{arxiv_context}

For each assumption, provide:
- assumption_id: A short ID (e.g., "A1", "A2")
- assumption_statement: Clear, precise statement
- justification: Why this assumption is needed
- type: behavioral, structural, parametric, or distributional
- criticality: Critical, Important, or Standard
- supporting_literature: Paper titles that support this
- potential_relaxations: How it could be relaxed in extensions

Return as a JSON object with an "assumptions" array.
Example: {{
  "assumptions": [
    {{
      "assumption_id": "A1",
      "assumption_statement": "Agents are rational and forward-looking",
      "justification": "...",
      "type": "behavioral",
      "criticality": "Critical",
      "supporting_literature": ["Paper 1", "Paper 2"],
      "potential_relaxations": ["Bounded rationality", "Adaptive expectations"]
    }}
  ]
}}

Respond with ONLY the JSON object, no other text.
""".format(
                    question=research_question.question,
                    framework=research_question.theoretical_framework or "Not specified",
                    methodologies=", ".join(research_question.methodology) if research_question.methodology else "Not specified",
                    literature_context=lit_context,
                    arxiv_context=arxiv_context or "None available"
                )}
            ])

            data = json.loads(repair_json(result))
            assumptions = [TheoreticalAssumption(**a) for a in data.get("assumptions", [])]
            
            return assumptions
        
        except Exception as e:
            print(f"    Error identifying assumptions: {e}")
            return []
    
    def _define_components(
        self,
        research_question: ResearchQuestion,
        literature_items: List[Dict],
        assumptions: List[TheoreticalAssumption]
    ) -> List[ConceptualComponent]:
        """Define conceptual components of the model."""
        
        # Prepare assumptions context
        assumptions_text = "\n".join([
            f"- {a.assumption_id}: {a.assumption_statement}"
            for a in assumptions
        ])
        
        lit_context = self._prepare_literature_context(literature_items, max_papers=10)
        
        try:
            result = self.llm.invoke([
                {"role": "system", "content": "You are an expert at designing conceptual architectures for economic models."},
                {"role": "user", "content": """Define 4-7 key conceptual components for a model addressing this research question.

Research Question: {question}
Theoretical Framework: {framework}

Assumptions:
{assumptions}

Literature Context:
{literature_context}

For each component, provide:
- component_id: Short ID (e.g., "C1", "C2")
- component_name: Name of the component
- component_type: agent, market, institution, mechanism, or constraint
- description: Detailed description
- key_features: 3-5 key features
- interactions: How it interacts with other components
- literature_basis: Papers supporting this component design

Return as a JSON object with a "components" array.
Example: {{
  "components": [
    {{
      "component_id": "C1",
      "component_name": "Heterogeneous Households",
      "component_type": "agent",
      "description": "...",
      "key_features": ["Feature 1", "Feature 2"],
      "interactions": ["Interacts with labor market", "Receives transfers"],
      "literature_basis": ["Paper 1", "Paper 2"]
    }}
  ]
}}

Respond with ONLY the JSON object, no other text.
""".format(
                    question=research_question.question,
                    framework=research_question.theoretical_framework or "Not specified",
                    assumptions=assumptions_text,
                    literature_context=lit_context
                )}
            ])

            data = json.loads(repair_json(result))
            components = [ConceptualComponent(**c) for c in data.get("components", [])]
            
            return components
        
        except Exception as e:
            print(f"    Error defining components: {e}")
            return []
    
    def _formulate_mathematics(
        self,
        research_question: ResearchQuestion,
        assumptions: List[TheoreticalAssumption],
        components: List[ConceptualComponent]
    ) -> List[MathematicalFormulation]:
        """Formulate key mathematical equations."""
        
        assumptions_text = "\n".join([
            f"- {a.assumption_id}: {a.assumption_statement}"
            for a in assumptions
        ])
        
        components_text = "\n".join([
            f"- {c.component_id} ({c.component_type}): {c.component_name}"
            for c in components
        ])
        
        try:
            result = self.llm.invoke([
                {"role": "system", "content": "You are an expert at mathematical formalization of economic models."},
                {"role": "user", "content": """Formulate 5-8 key equations for a model addressing this research question.

Research Question: {question}
Theoretical Framework: {framework}

Assumptions:
{assumptions}

Components:
{components}

For each equation, provide:
- equation_id: Short ID (e.g., "E1", "E2")
- equation_name: Name/description
- equation_latex: LaTeX representation (use standard LaTeX syntax)
- equation_plain: Plain text version
- variables: List of variables
- parameters: List of parameters
- interpretation: Economic interpretation
- derivation_notes: Brief notes on derivation

Return as a JSON object with an "equations" array.
Example: {{
  "equations": [
    {{
      "equation_id": "E1",
      "equation_name": "Household Utility Function",
      "equation_latex": "U(c_t, l_t) = \\\\frac{{c_t^{{1-\\\\sigma}}}}{{1-\\\\sigma}} - \\\\frac{{l_t^{{1+\\\\phi}}}}{{1+\\\\phi}}",
      "equation_plain": "U(c_t, l_t) = (c_t^(1-sigma))/(1-sigma) - (l_t^(1+phi))/(1+phi)",
      "variables": ["c_t (consumption)", "l_t (labor supply)"],
      "parameters": ["sigma (risk aversion)", "phi (Frisch elasticity)"],
      "interpretation": "CRRA utility over consumption with disutility of labor",
      "derivation_notes": "Standard separable utility specification"
    }}
  ]
}}

Respond with ONLY the JSON object, no other text.
""".format(
                    question=research_question.question,
                    framework=research_question.theoretical_framework or "Not specified",
                    assumptions=assumptions_text,
                    components=components_text
                )}
            ])

            data = json.loads(repair_json(result))
            formulations = [MathematicalFormulation(**e) for e in data.get("equations", [])]
            
            return formulations
        
        except Exception as e:
            print(f"    Error formulating mathematics: {e}")
            return []
    
    def _synthesize_framework(
        self,
        research_question: ResearchQuestion,
        assumptions: List[TheoreticalAssumption],
        components: List[ConceptualComponent],
        formulations: List[MathematicalFormulation],
        literature_items: List[Dict],
        literature_insights: Optional[List[Dict]]
    ) -> TheoreticalFramework:
        """Synthesize complete theoretical framework."""
        
        # Prepare context
        assumptions_text = "\n".join([f"- {a.assumption_id}: {a.assumption_statement}" for a in assumptions])
        components_text = "\n".join([f"- {c.component_id}: {c.component_name}" for c in components])
        equations_text = "\n".join([f"- {e.equation_id}: {e.equation_name}" for e in formulations])
        
        base_messages = [
            {"role": "system", "content": "You are an expert at synthesizing comprehensive theoretical frameworks."},
            {"role": "user", "content": """Synthesize a complete theoretical framework description.

Research Question: {question}

Assumptions:
{assumptions}

Components:
{components}

Equations:
{equations}

Provide:
- framework_title: Concise title for the framework
- framework_overview: 2-3 paragraph overview
- theoretical_approach: Main approach (e.g., "Dynamic Stochastic General Equilibrium", "Agent-Based Model")
- model_structure: Detailed description of overall structure
- equilibrium_concept: Equilibrium concept if applicable
- solution_approach: How the model would be solved/estimated
- literature_connections: How framework connects to existing work (5-7 points)
- innovation_points: Novel aspects (3-5 points)
- testable_predictions: Testable predictions (3-5 points)

Return as a JSON object.
Example: {{
  "framework_title": "...",
  "framework_overview": "...",
  "theoretical_approach": "...",
  "model_structure": "...",
  "equilibrium_concept": "...",
  "solution_approach": "...",
  "literature_connections": ["Connection 1", "Connection 2"],
  "innovation_points": ["Innovation 1", "Innovation 2"],
  "testable_predictions": ["Prediction 1", "Prediction 2"]
}}

Respond with ONLY the JSON object, no other text.
""".format(
                question=research_question.question,
                assumptions=assumptions_text,
                components=components_text,
                equations=equations_text
            )}
        ]

        try:
            # Parse with repair_json; on failure, retry once with a stricter
            # "valid JSON only" reminder before falling back to the minimal
            # framework. This recovers the ~1/3 of frameworks that previously
            # collapsed to an empty fallback on a transient JSON parse error.
            data = None
            last_err: Optional[Exception] = None
            for attempt in range(2):
                messages = base_messages
                if attempt == 1:
                    messages = base_messages + [{
                        "role": "user",
                        "content": ("Your previous reply was not valid JSON. Reply again with "
                                    "ONLY a single valid JSON object matching the requested keys. "
                                    "No markdown fences, no commentary, no trailing text."),
                    }]
                result = self.llm.invoke(messages)
                try:
                    data = json.loads(repair_json(result))
                    break
                except Exception as pe:
                    last_err = pe
                    print(f"    Framework JSON parse failed (attempt {attempt + 1}/2): {pe}")
            if data is None:
                raise last_err or ValueError("framework synthesis returned no parseable JSON")

            framework = TheoreticalFramework(
                framework_title=data.get("framework_title", "Theoretical Framework"),
                framework_overview=data.get("framework_overview", ""),
                research_question=research_question.question,
                theoretical_approach=data.get("theoretical_approach", ""),
                assumptions=assumptions,
                conceptual_components=components,
                mathematical_formulations=formulations,
                model_structure=data.get("model_structure", ""),
                equilibrium_concept=data.get("equilibrium_concept", ""),
                solution_approach=data.get("solution_approach", ""),
                literature_connections=data.get("literature_connections", []),
                innovation_points=data.get("innovation_points", []),
                testable_predictions=data.get("testable_predictions", []),
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "num_assumptions": len(assumptions),
                    "num_components": len(components),
                    "num_equations": len(formulations)
                }
            )
            
            return framework
        
        except Exception as e:
            print(f"    Error synthesizing framework: {e}")
            # Return minimal framework
            return TheoreticalFramework(
                framework_title="Theoretical Framework",
                framework_overview="Framework synthesis failed",
                research_question=research_question.question,
                theoretical_approach="Unknown",
                assumptions=assumptions,
                conceptual_components=components,
                mathematical_formulations=formulations,
                model_structure="",
                solution_approach="",
                literature_connections=[],
                innovation_points=[],
                testable_predictions=[]
            )
    
    def _prepare_literature_context(self, literature_items: List[Dict], max_papers: int = 10) -> str:
        """Prepare literature context for prompts."""
        context_parts = []
        
        for i, paper in enumerate(literature_items[:max_papers], 1):
            title = paper.get('title', 'Unknown')
            authors = ', '.join(paper.get('authors', [])[:2])
            abstract = paper.get('abstract', '')[:200]
            
            context_parts.append(f"{i}. {title} ({authors})\n   {abstract}...")
        
        return "\n\n".join(context_parts)


# ========== ORCHESTRATOR ==========

class TheoryStageOrchestrator:
    """Orchestrator for the theory development stage."""
    
    def __init__(self, openai_api_key: Optional[str] = None, collector: Optional[MetricsCollector] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        self.collector = collector
        self.theorist = Theorist(self.api_key, collector=collector)
        self.theory_output: Optional[TheoryStageOutput] = None
    
    def run_theory_pipeline(
        self,
        research_questions: List[ResearchQuestion],
        literature_batch: Dict,
        max_frameworks: int = 3
    ) -> TheoryStageOutput:
        """Run the complete theory development pipeline."""
        
        print(f"\n{'='*70}")
        print(f"THEORY DEVELOPMENT PIPELINE")
        print(f"{'='*70}")
        print(f"Research Questions: {len(research_questions)}")
        print(f"Literature Items: {len(literature_batch.get('literature_items', []))}")
        print(f"{'='*70}\n")
        
        # Extract literature items and insights
        literature_items = literature_batch.get('literature_items', [])
        literature_insights = literature_batch.get('insights', [])
        
        # Convert research questions to ResearchQuestion objects if they're dicts
        rq_objects = []
        for rq in research_questions[:max_frameworks]:
            if isinstance(rq, dict):
                rq_obj = ResearchQuestion(**rq)
            else:
                rq_obj = rq
            rq_objects.append(rq_obj)
        
        # Develop frameworks for top research questions
        frameworks = []
        for rq in rq_objects:
            framework = self.theorist.develop_framework(
                research_question=rq,
                literature_items=literature_items,
                literature_insights=literature_insights
            )
            frameworks.append(framework)
        
        # Create output
        self.theory_output = TheoryStageOutput(
            research_questions=research_questions,
            theoretical_frameworks=frameworks,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "num_questions": len(research_questions),
                "num_frameworks": len(frameworks),
                "num_literature_items": len(literature_items)
            }
        )
        
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Frameworks Developed: {len(frameworks)}")
        print(f"{'='*70}\n")
        
        return self.theory_output
    
    def save_theory_output(self, filename: str = "theory_output.json"):
        """Save theory output to JSON."""
        if not self.theory_output:
            print("No theory output to save")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.theory_output.model_dump(), f, indent=2)
        
        print(f"[Orchestrator] Theory output saved to {filename}")
    
    def save_frameworks_text(self, filename: str = "theoretical_frameworks.txt"):
        """Save frameworks in readable text format."""
        if not self.theory_output:
            print("No theory output to save")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("THEORETICAL FRAMEWORKS\n")
            f.write("="*70 + "\n\n")
            
            for i, framework in enumerate(self.theory_output.theoretical_frameworks, 1):
                f.write(f"FRAMEWORK {i}: {framework.framework_title}\n")
                f.write("="*70 + "\n\n")
                
                f.write(f"RESEARCH QUESTION:\n{framework.research_question}\n\n")
                
                f.write(f"THEORETICAL APPROACH:\n{framework.theoretical_approach}\n\n")
                
                f.write(f"OVERVIEW:\n{framework.framework_overview}\n\n")
                
                f.write(f"ASSUMPTIONS ({len(framework.assumptions)}):\n")
                for assumption in framework.assumptions:
                    f.write(f"  {assumption.assumption_id}. {assumption.assumption_statement}\n")
                    f.write(f"     Type: {assumption.type} | Criticality: {assumption.criticality}\n")
                    f.write(f"     Justification: {assumption.justification}\n\n")
                
                f.write(f"CONCEPTUAL COMPONENTS ({len(framework.conceptual_components)}):\n")
                for component in framework.conceptual_components:
                    f.write(f"  {component.component_id}. {component.component_name} ({component.component_type})\n")
                    f.write(f"     {component.description}\n\n")
                
                f.write(f"KEY EQUATIONS ({len(framework.mathematical_formulations)}):\n")
                for eq in framework.mathematical_formulations:
                    f.write(f"  {eq.equation_id}. {eq.equation_name}\n")
                    f.write(f"     LaTeX: {eq.equation_latex}\n")
                    f.write(f"     Plain: {eq.equation_plain}\n")
                    f.write(f"     Interpretation: {eq.interpretation}\n\n")
                
                f.write(f"MODEL STRUCTURE:\n{framework.model_structure}\n\n")
                
                if framework.equilibrium_concept:
                    f.write(f"EQUILIBRIUM CONCEPT:\n{framework.equilibrium_concept}\n\n")
                
                f.write(f"SOLUTION APPROACH:\n{framework.solution_approach}\n\n")
                
                f.write(f"INNOVATION POINTS:\n")
                for point in framework.innovation_points:
                    f.write(f"  - {point}\n")
                f.write("\n")
                
                f.write(f"TESTABLE PREDICTIONS:\n")
                for pred in framework.testable_predictions:
                    f.write(f"  - {pred}\n")
                f.write("\n")
                
                f.write("-"*70 + "\n\n")
        
        print(f"[Orchestrator] Frameworks saved to {filename}")


def main():
    """Main function for theory stage."""
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Load inputs
    questions_file = auto_input(
        "Enter path to research questions JSON: ",
        default=get_default("model_research_questions_path"),
    ).strip()
    literature_file = auto_input(
        "Enter path to literature batch JSON: ",
        default=get_default("literature_file_path"),
    ).strip()
    
    if not os.path.exists(questions_file):
        print(f"Error: {questions_file} not found!")
        return
    
    if not os.path.exists(literature_file):
        print(f"Error: {literature_file} not found!")
        return
    
    # Load data
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    
    with open(literature_file, 'r', encoding='utf-8') as f:
        literature_batch = json.load(f)
    
    # Parse research questions
    if isinstance(questions_data, list):
        research_questions = [ResearchQuestion(**q) for q in questions_data]
    elif isinstance(questions_data, dict) and 'questions' in questions_data:
        research_questions = [ResearchQuestion(**q) for q in questions_data['questions']]
    else:
        print("Error: Unexpected questions format")
        return
    
    # Run pipeline
    orchestrator = TheoryStageOrchestrator()
    theory_output = orchestrator.run_theory_pipeline(
        research_questions=research_questions,
        literature_batch=literature_batch,
        max_frameworks=3
    )
    
    # Save outputs
    orchestrator.save_theory_output("theory_output.json")
    orchestrator.save_frameworks_text("theoretical_frameworks.txt")
    
    # Print summary
    print("\n" + "="*70)
    print("THEORY STAGE SUMMARY")
    print("="*70)
    for i, framework in enumerate(theory_output.theoretical_frameworks, 1):
        print(f"\n{i}. {framework.framework_title}")
        print(f"   Approach: {framework.theoretical_approach}")
        print(f"   Assumptions: {len(framework.assumptions)}")
        print(f"   Components: {len(framework.conceptual_components)}")
        print(f"   Equations: {len(framework.mathematical_formulations)}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
