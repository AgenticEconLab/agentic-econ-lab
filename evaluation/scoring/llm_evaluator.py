# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
LLM-as-Reviewer evaluator for content quality assessment (Tier 2).

Routes through shared.llm.LLMClient. The default judge is an open-weight
vLLM model (env AEL_JUDGE_MODEL, default vllm/mistral-small-3.2-24b-fp8);
"vllm/"/"ollama/" go to the local server (no key), a "claude" name routes to
Anthropic and a plain name to OpenAI as commercial fallback.

Evaluates actual content quality of workflow outputs for dimensions
requiring expert judgment: Correctness, Soundness, Innovation Potential,
Transparency.
"""

import json
import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from evaluation.scoring.rubrics import (
    EvaluationRubric,
    LLM_EVALUATED_DIMENSIONS,
    get_rubric,
)
from shared.json_repair import repair_json

from evaluation.schemas.llm_scores import (
    EnsembleMetadata,
    EnsemblePassResult,
    LLMDimensionScore,
    SubCriterionScore,
    TwoTierResult,
)


# Output files to load per team (excluding execution_log.json)
TEAM_OUTPUT_FILES: Dict[str, List[str]] = {
    "IdeationTeam": [
        "finalized_research_questions_automated.json",
        "refinement_results_automated.json",
    ],
    "LiteratureTeam": [
        "consolidated_review.json",
        "literature_batch.json",
        "gap_analysis_results.json",
        "synthesis_results.json",
        "knowledge_graph.json",
    ],
    "ModelTeam": [
        "theory_output.json",
        "model_design_output.json",
        "calibration_output.json",
    ],
    "DataTeam": [
        # All four DataTeam stage outputs (per workflow mode), not just the first
        # two. The cleaning + QA files carry the provenance, QA reasoning,
        # methodology narrative, and innovation suggestions that drive the
        # Transparency and Innovation scores — omitting them blinded the judge.
        # OpenSourceAPI:
        "api_data_requirements.json",
        "api_source_output.json",
        "api_cleaning_output.json",
        "api_qa_output.json",
        # UserUploaded:
        "user_data_source_output.json",
        "user_data_cleaning_output.json",
        "user_data_qa_output.json",
        # PremiumSubscribed:
        "premium_data_source_output.json",
        "premium_data_cleaning_output.json",
        "premium_data_qa_output.json",
    ],
    # Section-4 execution teams: without entries in this map Tier-2 silently skips them
    # (the judge has nothing to read).
    "EstimationTeam": [
        "estimation_output.json",
        "validation_output.json",
        "inference_output.json",
    ],
    "ReportingTeam": [
        "research_report.md",          # terminal deliverable (Markdown, loaded as text)
        "quality_output.json",
        "interpretation_output.json",
        "drafting_output.json",
    ],
    "CodeTeam": [
        "generation_output.json",
        "validation_output.json",
        "experimentation_output.json",
    ],
}

# Fallback output files for modes that use different naming
TEAM_OUTPUT_FALLBACKS: Dict[str, List[str]] = {
    "IdeationTeam": [
        "finalized_research_questions.json",
        "refinement_results.json",
        "integration_results_all_rounds.json",
    ],
    "DataTeam": [
        "data_source_output.json",
        "data_cleaning_output.json",
        "quality_assurance_output.json",
        "api_cleaning_output.json",
        "api_qa_output.json",
        "premium_data_source_output.json",
        "premium_data_cleaning_output.json",
        "premium_data_qa_output.json",
        "premium_quality_assurance_output.json",
        "user_data_source_output.json",
        "user_data_cleaning_output.json",
        "user_data_qa_output.json",
        "user_quality_assurance_output.json",
    ],
}

# Maximum characters of output content to include in prompt
MAX_OUTPUT_CHARS = 24000

# Per-team budget overrides. DataTeam now loads four stage files (the QA file
# alone can reach ~36KB), so it gets a larger budget; large numeric sample
# arrays (data_preview) are also stripped before serialization (see
# _strip_data_samples) so the real reasoning fits.
TEAM_MAX_OUTPUT_CHARS: Dict[str, int] = {
    "DataTeam": 42000,
    # the full research report (~20KB Markdown) plus three stage JSONs
    "ReportingTeam": 36000,
}

# Keys holding bulky row-level sample arrays that crowd out reasoning text.
# Stripped before serialization so provenance/QA/innovation narrative survives the token
# budget. Applied to all teams: otherwise EstimationTeam's analysis_data rows (repeated in
# all three stage files) fill the 24k budget and inference_output.json (hypotheses,
# robustness) never reaches the judge; the consistency checker's per-number list does the
# same for ReportingTeam.
SAMPLE_ARRAY_KEYS = {"data_preview", "sample_matches", "data_sample", "analysis_data",
                     "verified_numbers"}


def _allocate(sizes: List[int], available: int) -> List[int]:
    """Per-file character allowance: every file is represented. If everything fits it
    is kept whole; otherwise the budget is water-filled — files shorter than the common cap
    keep their full length, the rest are truncated to the same cap c, where
    sum(min(size, c)) = available."""
    if sum(sizes) <= available:
        return list(sizes)
    remaining, left = max(0, available), len(sizes)
    caps = [0] * len(sizes)
    for i in sorted(range(len(sizes)), key=lambda j: sizes[j]):
        share = remaining // left if left else 0
        caps[i] = min(sizes[i], share)
        remaining -= caps[i]
        left -= 1
    return caps


def _strip_data_samples(obj: Any) -> Any:
    """Recursively drop bulky row-level sample arrays (e.g. data_preview).

    Replaces each sample array with a compact "<N rows omitted>" marker so the
    judge still knows data was present without spending the budget on raw rows.
    Returns a new structure; the input is not mutated.
    """
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if k in SAMPLE_ARRAY_KEYS and isinstance(v, list):
                out[k] = f"<{len(v)} sample rows omitted for token budget>"
            else:
                out[k] = _strip_data_samples(v)
        return out
    if isinstance(obj, list):
        return [_strip_data_samples(x) for x in obj]
    return obj


def _extract_nested(obj: Any, target_keys: set, found: Dict[str, Any], path: str = "") -> None:
    """Collect values for target_keys found anywhere in a nested structure.

    Used to surface DataTeam transparency/innovation fields (provenance,
    merge_strategy, innovation_suggestions, quality dimension issues/
    recommendations, research_alignment, series descriptions) that live deep
    inside documented_datasets[...] and would otherwise be lost to truncation.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in target_keys and v not in (None, "", [], {}):
                key = f"{path}.{k}" if path else k
                # Avoid clobbering: suffix duplicates with an index.
                if key in found:
                    i = 2
                    while f"{key}#{i}" in found:
                        i += 1
                    key = f"{key}#{i}"
                found[key] = v
            _extract_nested(v, target_keys, found, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            _extract_nested(item, target_keys, found, f"{path}[{idx}]")


class LLMEvaluator:
    """Evaluates workflow output content quality using LLM-as-reviewer.

    Supports OpenAI and Anthropic backends. The provider is auto-detected
    from the model name: models starting with "claude" use the Anthropic
    API; all others use the OpenAI API.
    """

    def __init__(
        self,
        model: str = os.environ.get("AEL_JUDGE_MODEL", "vllm/mistral-small-3.2-24b-fp8"),
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        collector: "Optional[Any]" = None,
    ):
        self.model = model
        self.temperature = temperature
        self.provider = self._detect_provider(model)
        self.collector = collector

        # Load API key based on provider (local servers vllm/ollama need no key)
        if api_key:
            self.api_key = api_key
        elif self.provider in ("ollama", "vllm"):
            self.api_key = ""
        else:
            load_dotenv()
            if self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
            else:
                self.api_key = os.getenv("OPENAI_API_KEY", "")

        if not self.api_key and self.provider not in ("ollama", "vllm"):
            key_name = "ANTHROPIC_API_KEY" if self.provider == "anthropic" else "OPENAI_API_KEY"
            raise ValueError(
                f"{key_name} not found. Set it in environment or pass api_key."
            )

        # OpenAI + local (vllm/ollama) judges route via LLMClient (observability + the
        # local base_url); LLMClient.detect_provider handles the vllm/ and ollama/ prefixes.
        if self.provider in ("openai", "ollama", "vllm"):
            from shared.llm import LLMClient
            self._llm_client = LLMClient(
                model=self.model,
                temperature=self.temperature,
                api_key=self.api_key or None,
                collector=self.collector,
                agent_name="LLMEvaluator",
                # Pin the judge model: AEL_MODEL (the subject/agents' model) must NOT override
                # it, else a multi-model consensus (e.g. Mistral+Gemma) collapses onto one model
                # — potentially the subject (self-evaluation). Replaces the fragile manual
                # `AEL_MODEL=<judge>` alignment in eval.sbatch.
                allow_env_override=False,
            )
        else:
            self._llm_client = None

    @staticmethod
    def _detect_provider(model: str) -> str:
        """Detect the API provider from the model name."""
        if model.startswith("claude"):
            return "anthropic"
        if model.startswith("vllm/"):
            return "vllm"            # Layer 1 — HPC vLLM (no API key)
        if model.startswith("ollama/"):
            return "ollama"          # Layer 2 — local laptop Ollama (no API key)
        return "openai"

    def evaluate_workflow(
        self,
        team: str,
        mode: str,
        base_path: Path,
        output_dir: "Optional[Path]" = None,
    ) -> Dict[str, LLMDimensionScore]:
        """Evaluate all Tier-2 dimensions for a single workflow configuration.

        If ``output_dir`` is given, output files are read directly from it (e.g. a
        multirun ``run_NNN/`` dir); otherwise from ``base_path/team/ael/mode``.
        """
        outputs = self._load_outputs(team, mode, base_path, output_dir)
        execution_log = self._load_execution_log(team, mode, base_path, output_dir)

        if not outputs:
            print(f"  WARNING: No output files found for {team}/{mode}")
            return {}

        results = {}
        for dimension in LLM_EVALUATED_DIMENSIONS:
            try:
                score = self.evaluate_dimension(
                    dimension, team, mode, outputs, execution_log
                )
                results[dimension] = score
                avg = score.raw_average
                print(f"    {dimension}: {score.overall_score:.3f} (raw avg: {avg:.2f}/5)")
            except Exception as e:
                print(f"    {dimension}: ERROR - {e}")

        return results

    def evaluate_workflow_ensemble(
        self,
        team: str,
        mode: str,
        base_path: Path,
        repeats: int = 2,
    ) -> List[Dict[str, "LLMDimensionScore"]]:
        """Run Tier-2 evaluation multiple times for ensemble averaging.

        Returns:
            List of per-pass dimension score dicts, where each element
            is the dict returned by evaluate_workflow().
        """
        all_passes: List[Dict[str, LLMDimensionScore]] = []
        for i in range(repeats):
            print(f"      Pass {i + 1}/{repeats} [{self.model}]...")
            scores = self.evaluate_workflow(team, mode, base_path)
            all_passes.append(scores)
        return all_passes

    def evaluate_dimension(
        self,
        dimension: str,
        team: str,
        mode: str,
        outputs: Dict[str, Any],
        execution_log: Optional[Dict[str, Any]] = None,
    ) -> LLMDimensionScore:
        """Evaluate a single dimension for a single workflow."""
        if self.collector is not None:
            self.collector.set_context(stage=f"eval_{dimension}")
        rubric = get_rubric(dimension, team)
        prompt = self._build_prompt(rubric, outputs, team, mode, execution_log)
        response = self._call_llm(prompt)
        sub_criteria = self._parse_response(response, rubric)

        return LLMDimensionScore.from_sub_criteria(
            dimension=dimension,
            team=team,
            mode=mode,
            sub_criteria=sub_criteria,
            evaluator_model=self.model,
        )

    def _build_prompt(
        self,
        rubric: EvaluationRubric,
        outputs: Dict[str, Any],
        team: str,
        mode: str,
        execution_log: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build the structured evaluation prompt."""
        # Extract research topic from execution log metadata
        research_topic = "Not specified"
        if execution_log and "metadata" in execution_log:
            meta = execution_log["metadata"]
            research_topic = meta.get(
                "research_topic",
                meta.get("research_question", "Not specified"),
            )

        # Format output content (truncated)
        output_text = self._format_outputs(outputs, team)

        # Build the rubric text
        rubric_text = rubric.to_prompt_text()

        # Construct the criterion names list for the JSON format
        criteria_names = [sc.name for sc in rubric.sub_criteria]
        json_template = json.dumps(
            {
                "sub_criteria": [
                    {
                        "criterion_name": name,
                        "score": "<1-5>",
                        "justification": "<2-3 sentences>",
                    }
                    for name in criteria_names
                ],
                "overall_notes": "<Brief summary>",
            },
            indent=2,
        )

        prompt = f"""You are an expert evaluator of AI-generated economics research outputs.
You will score the quality of outputs from an agentic workflow system on specific criteria.

IMPORTANT INSTRUCTIONS:
- Score each sub-criterion independently on a 1-5 Likert scale.
- Provide a brief justification (2-3 sentences) for each score.
- Be critical and honest. Do not inflate scores.
- Base your evaluation ONLY on the content provided below.
- Respond ONLY with the JSON structure specified. No other text.

{rubric_text}

---

## Workflow Context

Team: {team}
Mode: {mode}
Research Topic: {research_topic}

## Workflow Outputs to Evaluate

{output_text}

---

## Required JSON Output Format

{json_template}
"""
        return prompt

    # Fields that carry transparency-relevant metadata (reasoning, decisions, assumptions)
    TRANSPARENCY_FIELDS = {
        "reasoning_trace", "decision_log", "assumptions_and_limitations",
        "assumptions", "limitations", "process_assumptions",
        "process_limitations", "selection_rationale",
        "cross_domain_opportunities", "research_question_alignment",
        "data_requirements", "citation_integrity_check",
    }

    # DataTeam's real rationale is buried inside nested stage structures
    # (documented_datasets[...], integrated_datasets[...], quality_assessments
    # [...].dimensions[...]). These keys are pulled out by a nested extractor so
    # they survive truncation and the judge can see provenance, QA reasoning,
    # methodology, and the (strong) innovation suggestions.
    DATATEAM_TRANSPARENCY_KEYS = {
        "research_alignment",        # discovered_sources[].research_alignment
        "description",               # selected_series[].description (+ codebook var descriptions)
        "issues_found",              # quality_assessments[].dimensions[].issues_found
        "recommendations",           # quality_assessments[].dimensions[].recommendations
        "actionable_issues",         # quality_assessments[].actionable_issues
        "provenance",                # integrated dataset source/conversion provenance
        "merge_strategy",            # multi-source merge strategy
        "conversion_method",         # per-series frequency conversion method
        "innovation_suggestions",    # documentation.innovation_suggestions (the strong part)
        "methodology",               # codebook methodology
        "methodology_narrative",     # LLM methodology narrative (new in QA stage)
        "usage_notes",               # codebook usage notes
    }

    def _format_outputs(self, outputs: Dict[str, Any], team: str) -> str:
        """Format workflow outputs for inclusion in the prompt, with truncation.

        Transparency-relevant fields (reasoning traces, decision logs,
        assumptions/limitations) are extracted from all output files and
        presented first so the evaluator sees them even when later content
        is truncated.
        """
        sections = []
        total_chars = 0
        budget = TEAM_MAX_OUTPUT_CHARS.get(team, MAX_OUTPUT_CHARS)
        is_datateam = team == "DataTeam"

        # --- Pass 0: the deterministic number-consistency result, compact, first ---
        consistency: Dict[str, Any] = {}
        for filename, content in outputs.items():
            if isinstance(content, (dict, list)):
                _extract_nested(content, {"consistency"}, consistency, path=filename)
        compact = {}
        for k, v in consistency.items():
            if isinstance(v, dict) and "verdict" in v:
                compact[k] = {
                    "verdict": v.get("verdict"), "verified": v.get("verified"),
                    "total_numbers": v.get("total_numbers"),
                    "skipped_small_ints": v.get("skipped_small_ints"),
                    "match_kinds": v.get("match_kinds"),
                    "verified_by_source": v.get("verified_by_source"),
                    "unverified": (v.get("unverified") or [])[:10]}
        if compact:
            text = json.dumps(compact, indent=2, default=str)[: budget // 8]
            sections.append("### Number-consistency check result (deterministic)\n"
                            f"```json\n{text}\n```")
            total_chars += len(text)

        # --- Pass 1: extract transparency metadata across all files ---
        transparency_entries: Dict[str, Any] = {}
        for filename, content in outputs.items():
            if isinstance(content, dict):
                for key in self.TRANSPARENCY_FIELDS:
                    if key in content and content[key]:
                        transparency_entries[f"{filename}::{key}"] = content[key]
            # DataTeam: also pull the deeply-nested rationale/innovation fields.
            if is_datateam:
                nested: Dict[str, Any] = {}
                _extract_nested(content, self.DATATEAM_TRANSPARENCY_KEYS, nested)
                for k, v in nested.items():
                    transparency_entries[f"{filename}::{k}"] = v

        if transparency_entries:
            header = "### Transparency Metadata (extracted from outputs)"
            text = json.dumps(_strip_data_samples(transparency_entries), indent=2, default=str)
            if len(text) > budget // 2:
                text = text[: budget // 2] + "\n... [truncated]"
            sections.append(f"{header}\n```json\n{text}\n```")
            total_chars += len(text)

        # --- Pass 2: file contents, bulky row arrays stripped (all teams), each file given
        # a share of the remaining budget so every output file is represented ---
        texts = []
        for filename, content in outputs.items():
            dump_content = _strip_data_samples(content)
            if isinstance(dump_content, (dict, list)):
                texts.append((filename, json.dumps(dump_content, indent=2, default=str)))
            else:
                texts.append((filename, str(dump_content)))
        caps = _allocate([len(t) for _, t in texts], budget - total_chars)
        for (filename, text), cap in zip(texts, caps):
            if len(text) > cap:
                text = text[:cap] + f"\n... [truncated: {cap} of {len(text)} characters shown]"
            sections.append(f"### File: {filename}\n```json\n{text}\n```")
            total_chars += min(cap, len(text))

        return "\n\n".join(sections)

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API and return the response text.

        Routes to the appropriate provider backend (OpenAI or Anthropic)
        based on self.provider, which is auto-detected from the model name.
        """
        if self.provider == "anthropic":
            return self._call_anthropic(prompt)
        return self._call_openai(prompt)

    def _call_openai(self, prompt: str) -> str:
        """Call the OpenAI API via LLMClient (httpx-based with observability)."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert academic reviewer evaluating AI-generated "
                    "economics research outputs. Respond only with valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return self._llm_client.invoke(
            messages, response_format={"type": "json_object"}
        )

    def _call_anthropic(self, prompt: str) -> str:
        """Call the Anthropic API with manual observability tracking."""
        import time
        from anthropic import Anthropic

        client = Anthropic(api_key=self.api_key)

        start = time.time()
        error_msg = None
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=self.temperature,
                system=(
                    "You are an expert academic reviewer evaluating AI-generated "
                    "economics research outputs. Respond only with valid JSON."
                ),
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            latency = time.time() - start

            # Record metrics if collector is available
            if self.collector is not None:
                from shared.observability import LLMCallRecord, estimate_cost

                usage = response.usage
                prompt_tokens = usage.input_tokens if usage else 0
                completion_tokens = usage.output_tokens if usage else 0
                total_tokens = prompt_tokens + completion_tokens
                cost = estimate_cost(self.model, prompt_tokens, completion_tokens)

                record = LLMCallRecord(
                    agent="LLMEvaluator",
                    model=self.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost_usd=cost,
                    latency_seconds=latency,
                )
                self.collector.record_llm_call(record)

            return response.content[0].text

        except Exception as e:
            latency = time.time() - start
            if self.collector is not None:
                from shared.observability import LLMCallRecord

                record = LLMCallRecord(
                    agent="LLMEvaluator",
                    model=self.model,
                    latency_seconds=latency,
                    error=str(e),
                )
                self.collector.record_llm_call(record)
            raise

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from text that may contain markdown code blocks."""
        import re
        # Try to extract JSON from ```json ... ``` blocks
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Try to find raw JSON object
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        return text

    def _dump_unparseable_response(self, raw: str, repaired: str, error: Exception) -> None:
        """Best-effort diagnostic dump when a judge response survives every repair
        attempt unparseable -- so the exact malformation can be inspected offline
        instead of only ever seeing the exception message. Never raises itself."""
        try:
            out_dir = Path(__file__).resolve().parents[1] / "_unparseable_responses"
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
            (out_dir / f"{stamp}.txt").write_text(
                f"error: {error}\nmodel: {getattr(self, 'model', '?')}\n\n"
                f"--- raw response ---\n{raw}\n\n"
                f"--- after repair_json ---\n{repaired}\n"
            )
        except Exception as dump_exc:
            print(f"    [diagnostic dump failed: {type(dump_exc).__name__}: {dump_exc}]")

    def _parse_response(
        self, response_text: str, rubric: EvaluationRubric
    ) -> List[SubCriterionScore]:
        """Parse LLM JSON response into SubCriterionScore objects."""
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            # Try extracting JSON from markdown code blocks
            cleaned = self._extract_json(response_text)
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError as e:
                # Heuristic repair (shared/json_repair.py): a missing comma between
                # sibling members is a deterministic judge-formatting quirk, not
                # transient flakiness -- retrying an identical temperature=0 call
                # reproduces the exact same malformed JSON every time (observed for
                # CodeTeam/correctness under Gemma: always "Expecting ',' delimiter" at the
                # same offset).
                repaired = repair_json(cleaned)
                try:
                    data = json.loads(repaired)
                except json.JSONDecodeError:
                    print(f"    [repair_json did not fix it -- raw len={len(response_text)}, "
                          f"repaired len={len(repaired)}, changed={repaired != cleaned}]")
                    self._dump_unparseable_response(response_text, repaired, e)
                    raise ValueError(f"Failed to parse LLM response as JSON: {e}")

        sub_criteria_data = data.get("sub_criteria", [])
        expected_names = {sc.name for sc in rubric.sub_criteria}

        scores = []
        for item in sub_criteria_data:
            name = item.get("criterion_name", "")
            raw_score = item.get("score", 3)
            justification = item.get("justification", "No justification provided")

            # Clamp score to 1-5 range
            score = max(1, min(5, int(raw_score)))

            scores.append(
                SubCriterionScore(
                    criterion_name=name,
                    score=score,
                    justification=justification,
                )
            )

        # If LLM missed some criteria, fill with neutral scores
        returned_names = {s.criterion_name for s in scores}
        for expected in expected_names:
            if expected not in returned_names:
                scores.append(
                    SubCriterionScore(
                        criterion_name=expected,
                        score=3,
                        justification="Not evaluated by LLM (default neutral score)",
                    )
                )

        return scores

    def _load_outputs(
        self, team: str, mode: str, base_path: Path, output_dir: "Optional[Path]" = None
    ) -> Dict[str, Any]:
        """Load workflow output JSON files for a team/mode."""
        output_dir = Path(output_dir) if output_dir else (base_path / team / "ael" / mode)
        outputs = {}

        # Try primary output files
        primary_files = TEAM_OUTPUT_FILES.get(team, [])
        for filename in primary_files:
            filepath = output_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        if filename.endswith(".md"):
                            outputs[filename] = f.read()   # Markdown deliverable as text
                        else:
                            outputs[filename] = json.load(f)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

        # Try fallback files if few primary files found
        if len(outputs) < 2:
            fallback_files = TEAM_OUTPUT_FALLBACKS.get(team, [])
            for filename in fallback_files:
                filepath = output_dir / filename
                if filepath.exists() and filename not in outputs:
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            outputs[filename] = json.load(f)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass

        return outputs

    def _load_execution_log(
        self, team: str, mode: str, base_path: Path, output_dir: "Optional[Path]" = None
    ) -> Optional[Dict[str, Any]]:
        """Load execution log for context."""
        _dir = Path(output_dir) if output_dir else (base_path / team / "ael" / mode)
        log_path = _dir / "execution_log.json"
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        return None


# ---------------------------------------------------------------------------
# Ensemble aggregation helpers (module-level)
# ---------------------------------------------------------------------------

def average_ensemble_scores(
    all_passes: List[Dict[str, LLMDimensionScore]],
    team: str,
    mode: str,
) -> Dict[str, LLMDimensionScore]:
    """Average LLMDimensionScore objects across multiple ensemble passes.

    For each dimension, averages the raw sub-criterion scores across passes
    and produces a single consolidated LLMDimensionScore.

    Args:
        all_passes: List of per-pass dimension score dicts.
        team: Team name.
        mode: Mode name.

    Returns:
        Dict of averaged LLMDimensionScore per dimension.
    """
    if not all_passes:
        return {}

    # Collect all dimension names across passes
    all_dims = set()
    for pass_scores in all_passes:
        all_dims.update(pass_scores.keys())

    averaged = {}
    for dim in all_dims:
        # Gather sub-criteria across passes for this dimension
        passes_with_dim = [p[dim] for p in all_passes if dim in p]
        if not passes_with_dim:
            continue

        # Collect scores per sub-criterion name
        sc_scores: Dict[str, List[int]] = {}
        sc_justifications: Dict[str, List[str]] = {}
        for pass_score in passes_with_dim:
            for sc in pass_score.sub_criteria:
                sc_scores.setdefault(sc.criterion_name, []).append(sc.score)
                sc_justifications.setdefault(sc.criterion_name, []).append(
                    sc.justification
                )

        # Average each sub-criterion
        avg_sub_criteria = []
        for sc_name, scores_list in sc_scores.items():
            if not scores_list:
                continue
            avg_raw = sum(scores_list) / len(scores_list)
            # Round to nearest int for the 1-5 scale, but keep float precision
            # in the normalized score via from_sub_criteria
            avg_int = max(1, min(5, round(avg_raw)))
            # Pick the justification from the pass closest to the average
            best_idx = min(
                range(len(scores_list)),
                key=lambda i: abs(scores_list[i] - avg_raw),
            )
            avg_sub_criteria.append(
                SubCriterionScore(
                    criterion_name=sc_name,
                    score=avg_int,
                    justification=sc_justifications[sc_name][best_idx],
                )
            )

        models_used = list({p.evaluator_model for p in passes_with_dim})
        averaged[dim] = LLMDimensionScore.from_sub_criteria(
            dimension=dim,
            team=team,
            mode=mode,
            sub_criteria=avg_sub_criteria,
            evaluator_model=f"ensemble({','.join(models_used)})",
            notes=f"Averaged from {len(passes_with_dim)} passes",
        )

    return averaged


def compute_ensemble_metadata(
    model_passes: Dict[str, List[Dict[str, LLMDimensionScore]]],
    repeats_per_model: int,
) -> EnsembleMetadata:
    """Compute ensemble statistics from per-model pass results.

    Args:
        model_passes: {model_name: [pass1_scores, pass2_scores, ...]}
        repeats_per_model: Number of repeats per model.

    Returns:
        EnsembleMetadata with within-model std, between-model gap, etc.
    """
    import statistics

    models = sorted(model_passes.keys())
    total_passes = sum(len(v) for v in model_passes.values())

    # Build per-pass results
    per_pass_results = []
    pass_counter = 0
    for model_name in models:
        for idx, pass_scores in enumerate(model_passes[model_name]):
            per_pass_results.append(
                EnsemblePassResult(
                    evaluator_model=model_name,
                    pass_index=idx,
                    dimension_scores=pass_scores,
                )
            )
            pass_counter += 1

    # Compute within-model std per dimension
    all_dims = set()
    for passes_list in model_passes.values():
        for pass_scores in passes_list:
            all_dims.update(pass_scores.keys())

    within_std: Dict[str, Dict[str, float]] = {}
    model_dim_means: Dict[str, Dict[str, float]] = {}

    for model_name in models:
        within_std[model_name] = {}
        model_dim_means[model_name] = {}
        passes_list = model_passes[model_name]

        for dim in all_dims:
            scores = [
                p[dim].overall_score
                for p in passes_list
                if dim in p
            ]
            if len(scores) >= 2:
                within_std[model_name][dim] = round(statistics.stdev(scores), 4)
            elif len(scores) == 1:
                within_std[model_name][dim] = 0.0

            if scores:
                model_dim_means[model_name][dim] = sum(scores) / len(scores)

    # Compute between-model gap (first model - second model)
    between_gap: Dict[str, float] = {}
    if len(models) >= 2:
        for dim in all_dims:
            m0 = model_dim_means.get(models[0], {}).get(dim)
            m1 = model_dim_means.get(models[1], {}).get(dim)
            if m0 is not None and m1 is not None:
                between_gap[dim] = round(m0 - m1, 4)

    return EnsembleMetadata(
        models=models,
        repeats_per_model=repeats_per_model,
        total_passes=total_passes,
        per_pass_results=per_pass_results,
        within_model_std=within_std,
        between_model_gap=between_gap,
    )
