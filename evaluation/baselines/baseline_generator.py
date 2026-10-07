# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Single-prompt LLM baseline generator.

Generates "single-prompt" baseline outputs that replicate the task of each
agent team in a single LLM call. This provides a lower-bound baseline for
comparison against the multi-agent agentic workflow outputs.

The baseline outputs are saved in the same JSON format as workflow outputs
so the same evaluation pipeline (Tier-2 LLM-as-reviewer) can score them.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Team-specific single-prompt templates
# ---------------------------------------------------------------------------

TEAM_PROMPTS: Dict[str, str] = {
    "IdeationTeam": """\
You are an economics research assistant. Given a research topic, generate \
a set of innovative research questions suitable for academic investigation.

Research Topic: {research_topic}

Produce a JSON object with the following structure:
{{
  "research_questions": [
    {{
      "question": "The research question text",
      "theoretical_framework": "The theoretical basis for this question",
      "methodology": "Suggested research methodology",
      "rationale": "Why this question is important and novel",
      "key_references": ["Author (Year) - Brief description"]
    }}
  ],
  "topic_analysis": {{
    "key_themes": ["theme1", "theme2"],
    "trending_areas": ["area1", "area2"],
    "cross_domain_connections": ["connection1"]
  }}
}}

Generate 5-8 high-quality research questions that are:
- Grounded in economic theory
- Novel and not already well-addressed in the literature
- Specific enough to guide actual research
- Diverse in methodology and theoretical perspective

Respond ONLY with valid JSON.""",

    "LiteratureTeam": """\
You are an economics research assistant performing a literature review. \
Given a research topic, produce a structured literature review with gap analysis.

Research Topic: {research_topic}

Produce a JSON object with the following structure:
{{
  "literature_review": {{
    "papers": [
      {{
        "title": "Paper title",
        "authors": "Author names",
        "year": 2024,
        "journal": "Journal name",
        "summary": "Brief summary of key findings",
        "methodology": "Research methodology used",
        "relevance": "How this paper relates to the research topic"
      }}
    ],
    "thematic_categories": [
      {{
        "theme": "Theme name",
        "description": "Theme description",
        "paper_count": 5
      }}
    ]
  }},
  "gap_analysis": {{
    "identified_gaps": [
      {{
        "gap": "Description of the research gap",
        "supporting_evidence": "Why this gap exists",
        "potential_approach": "How this gap could be addressed"
      }}
    ]
  }},
  "synthesis": {{
    "key_findings": "Summary of key findings from the literature",
    "research_landscape": "Overview of the research landscape",
    "future_directions": "Suggested future research directions"
  }}
}}

Identify 10-15 relevant papers and 3-5 research gaps.
Respond ONLY with valid JSON.""",

    "ModelTeam": """\
You are an economics research assistant specializing in economic modeling. \
Given a research question, develop a theoretical framework and formal model.

Research Question: {research_topic}

Produce a JSON object with the following structure:
{{
  "theory": {{
    "theoretical_framework": "Description of the theoretical basis",
    "key_assumptions": ["assumption1", "assumption2"],
    "core_mechanisms": ["mechanism1", "mechanism2"],
    "literature_basis": ["Author (Year) - contribution"]
  }},
  "model_design": {{
    "model_type": "e.g., DSGE, Agent-Based, Partial Equilibrium",
    "equations": [
      {{
        "name": "Equation name",
        "latex": "LaTeX representation",
        "description": "What this equation captures"
      }}
    ],
    "variables": [
      {{
        "symbol": "Variable symbol",
        "name": "Variable name",
        "type": "endogenous/exogenous/parameter",
        "description": "Description"
      }}
    ],
    "equilibrium_conditions": ["condition1"]
  }},
  "calibration": {{
    "parameters": [
      {{
        "symbol": "Parameter symbol",
        "value": 0.0,
        "source": "Calibration source",
        "justification": "Why this value"
      }}
    ],
    "sensitivity_analysis": ["parameter1 range: [low, high]"],
    "key_predictions": ["prediction1"]
  }}
}}

Develop a rigorous economic model with:
- Clear theoretical grounding
- Well-specified mathematical formulation
- Realistic calibration with cited parameter values
Respond ONLY with valid JSON.""",

    "DataTeam": """\
You are an economics research assistant specializing in data collection. \
Given a research question, identify required data and appropriate sources.

Research Question: {research_topic}

Produce a JSON object with the following structure:
{{
  "data_requirements": [
    {{
      "variable": "Variable name",
      "description": "What this variable measures",
      "frequency": "e.g., monthly, quarterly, annual",
      "time_period": "e.g., 2000-2024",
      "unit": "Unit of measurement",
      "source": "Data source name",
      "api_endpoint": "API endpoint or series ID if applicable"
    }}
  ],
  "data_sources": [
    {{
      "name": "Source name (e.g., FRED, World Bank, BLS)",
      "type": "open_source/premium/subscription",
      "url": "Source URL",
      "access_method": "API/download/web scraping",
      "variables_available": ["var1", "var2"]
    }}
  ],
  "data_pipeline": {{
    "cleaning_steps": ["step1", "step2"],
    "transformations": ["transformation1"],
    "quality_checks": ["check1", "check2"]
  }}
}}

Identify 5-10 key variables with specific, real data sources.
Respond ONLY with valid JSON.""",
}

# Map team output filenames to match workflow output format.
# Uses names compatible with LLMEvaluator._load_outputs() fallback list
# so the same evaluation pipeline can score baseline outputs.
TEAM_OUTPUT_FILENAMES: Dict[str, Dict[str, str]] = {
    "IdeationTeam": {
        "primary": "finalized_research_questions.json",
    },
    "LiteratureTeam": {
        "primary": "synthesis_results.json",
    },
    "ModelTeam": {
        "primary": "theory_output.json",
    },
    "DataTeam": {
        "primary": "api_data_requirements.json",
    },
}


class BaselineGenerator:
    """
    Generates single-prompt LLM baselines for comparison against
    multi-agent workflow outputs.

    Each baseline represents what a single LLM call can produce for
    the same research task, without the multi-agent orchestration,
    stage decomposition, or specialized agent roles.
    """

    def __init__(
        self,
        model: str = os.environ.get("AEL_JUDGE_MODEL", "vllm/mistral-small-3.2-24b-fp8"),
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        collector: "Optional[Any]" = None,
    ):
        """
        Initialize BaselineGenerator.

        Args:
            model: LLM model for baseline generation. Open-weight vLLM by default
                (env AEL_JUDGE_MODEL); commercial models allowed as fallback.
            temperature: Sampling temperature.
            api_key: commercial API key (only needed for OpenAI/Anthropic models).
                If None, reads from env.
            collector: Optional MetricsCollector for observability tracking.
        """
        self.model = model
        self.temperature = temperature
        self.collector = collector

        if api_key:
            self.api_key = api_key
        else:
            load_dotenv()
            self.api_key = os.getenv("OPENAI_API_KEY", "")

        # Open-weight local judges (vllm/, ollama/) need no API key; only require a
        # commercial key when a commercial model is actually requested.
        _is_local = self.model.startswith(("vllm/", "ollama/"))
        if not self.api_key and not _is_local:
            raise ValueError(
                f"OPENAI_API_KEY not found for commercial model '{self.model}'. "
                "Set it, pass api_key, or use a vllm/ollama model."
            )

        # Create LLMClient for httpx-based calls with automatic observability
        from shared.llm import LLMClient
        self._llm_client = LLMClient(
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key,
            collector=self.collector,
            agent_name="BaselineGenerator",
            allow_env_override=False,  # pin the baseline model; AEL_MODEL must not override it
        )

    def generate_single_prompt_baseline(
        self,
        team: str,
        research_topic: str,
    ) -> Dict[str, Any]:
        """
        Generate a single-prompt baseline for a given team and topic.

        Sends one LLM call that mimics the overall task of the agentic
        workflow team, producing output in a comparable JSON format.

        Args:
            team: Team name (IdeationTeam, LiteratureTeam, etc.)
            research_topic: The research topic or question.

        Returns:
            Dictionary containing the baseline output.

        Raises:
            ValueError: If team is not recognized.
        """
        if team not in TEAM_PROMPTS:
            raise ValueError(
                f"Unknown team: {team}. "
                f"Valid teams: {list(TEAM_PROMPTS.keys())}"
            )

        prompt_template = TEAM_PROMPTS[team]
        prompt = prompt_template.format(research_topic=research_topic)

        response = self._call_llm(prompt)

        try:
            output = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            output = {"raw_response": response, "parse_error": True}

        # Add metadata
        output["_baseline_metadata"] = {
            "generator": "single_prompt_baseline",
            "model": self.model,
            "temperature": self.temperature,
            "team": team,
            "research_topic": research_topic,
            "generated_at": datetime.now().isoformat(),
        }

        return output

    def save_baseline(
        self,
        team: str,
        outputs: Dict[str, Any],
        output_dir: Path,
    ) -> Path:
        """
        Save baseline outputs in the same format as workflow outputs.

        Args:
            team: Team name.
            outputs: Baseline output dictionary.
            output_dir: Directory to save outputs.

        Returns:
            Path to saved baseline file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        filenames = TEAM_OUTPUT_FILENAMES.get(team, {})
        primary_name = filenames.get("primary", f"{team}_baseline.json")
        output_path = output_dir / primary_name

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2, default=str)

        # Also save an execution_log.json for compatibility
        log_path = output_dir / "execution_log.json"
        execution_log = {
            "run_id": f"baseline_{team}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "team": team,
            "mode": "SinglePromptBaseline",
            "framework": "baseline",
            "metadata": {
                "research_topic": outputs.get("_baseline_metadata", {}).get(
                    "research_topic", ""
                ),
                "baseline_model": self.model,
            },
            "stages": [
                {
                    "stage_name": "SinglePromptGeneration",
                    "stage_number": 1,
                    "status": "success",
                    "item_count": self._count_items(outputs, team),
                }
            ],
            "success": True,
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(execution_log, f, indent=2)

        return output_path

    def generate_and_save(
        self,
        team: str,
        research_topic: str,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """
        Generate baseline and save to disk in one step.

        Args:
            team: Team name.
            research_topic: Research topic or question.
            output_dir: Directory to save outputs.

        Returns:
            Baseline output dictionary.
        """
        outputs = self.generate_single_prompt_baseline(team, research_topic)
        save_path = self.save_baseline(team, outputs, output_dir)
        print(f"  Baseline saved: {save_path}")
        return outputs

    def generate_all_teams(
        self,
        research_topic: str,
        output_base_dir: Path,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate baselines for all teams.

        Args:
            research_topic: Research topic or question.
            output_base_dir: Base output directory (team subdirs created).

        Returns:
            Dictionary mapping team name to baseline output.
        """
        results = {}
        for team in TEAM_PROMPTS:
            print(f"Generating baseline for {team}...")
            team_dir = output_base_dir / team / "ael" / "SinglePromptBaseline"
            try:
                outputs = self.generate_and_save(team, research_topic, team_dir)
                results[team] = outputs
            except Exception as e:
                print(f"  ERROR: {e}")
                results[team] = {"error": str(e)}
        return results

    def _call_llm(self, prompt: str) -> str:
        """Call the OpenAI API via LLMClient (httpx-based with observability)."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert economics researcher. "
                    "Respond only with valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return self._llm_client.invoke(
            messages, response_format={"type": "json_object"}
        )

    def _count_items(self, outputs: Dict[str, Any], team: str) -> int:
        """Count output items for execution log."""
        if team == "IdeationTeam":
            return len(outputs.get("research_questions", []))
        elif team == "LiteratureTeam":
            review = outputs.get("literature_review", {})
            return len(review.get("papers", []))
        elif team == "ModelTeam":
            design = outputs.get("model_design", {})
            return len(design.get("equations", []))
        elif team == "DataTeam":
            return len(outputs.get("data_requirements", []))
        return 0
