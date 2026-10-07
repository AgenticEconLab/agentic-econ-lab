# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Research Question Integration Stage with Human-in-the-Loop
This script integrates refined research questions through theoretical framing and synthesis.

Agents:
- Contextualizer: Provides theoretical framing for research questions
- Finalizer: Synthesizes and prioritizes research questions

Input: Refined questions from 2-RefinementStage.py (JSON file)
Output: Finalized, prioritized research questions after two-round feedback process
"""

import os
import sys
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
# Add parent directories to path for shared imports
from pathlib import Path as _Path
_agents_dir = _Path(__file__).resolve().parent.parent.parent.parent
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

from shared.llm import LLMClient
from shared.observability import MetricsCollector
from shared.auto_input import auto_input, get_default
from IdeationTeam.ael.schemas.stage_outputs import ResearchQuestion, ContextualizedQuestion, PrioritizedQuestion, IntegrationFeedback, ContextualizedQuestionList, PrioritizedQuestionList
from shared.json_repair import safe_json_loads, repair_json
from shared.selection_aggregation import plan_selection, topic_overlap, is_on_topic
from shared.question_ancestry import validate_ancestry, source_label, ACCEPTED, CARRIED_FORWARD

# Load environment variables
load_dotenv()


# ---------------------------------------------------------------------------
# Resilient LLM-JSON parsing for the Integration stage.
#
# The Contextualizer/Finalizer can return a large (~20k-char) JSON payload that
# fails json.loads outright — e.g. a response truncated at max_tokens, or one
# carrying unescaped LaTeX backslashes. Letting LLMClient's parse_as raise on
# that crashed the whole stage (~25% of reps). Instead we request JSON mode,
# then parse defensively with the repo's safe_json_loads/repair_json and build
# each item under its own try/except (the LLMCoercedModel mixin coerces shapes),
# so one bad or partial entry never loses the rest.
# ---------------------------------------------------------------------------

def _salvage_question_dicts(raw):
    """Extract individual balanced ``{...}`` objects from a malformed/truncated
    LLM payload, parsing each independently. Recovers the complete question
    entries that precede a truncation point."""
    if not raw or not isinstance(raw, str):
        return []
    text = repair_json(raw)
    # Prefer the questions array; fall back to scanning the whole string.
    key = text.find('"questions"')
    start = text.find('[', key) if key != -1 else text.find('[')
    scan = text[start + 1:] if start != -1 else text
    out = []
    n = len(scan)
    i = 0
    while i < n:
        if scan[i] != '{':
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        end = -1
        j = i
        while j < n:
            c = scan[j]
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == '"':
                in_str = not in_str
            elif not in_str:
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end = j
                        break
            j += 1
        if end == -1:
            break  # remaining text is truncated mid-object — stop
        obj = safe_json_loads(scan[i:end + 1], fallback=None)
        if isinstance(obj, dict):
            out.append(obj)
        i = end + 1
    return out


def _coerce_question_items(raw, item_model):
    """Parse an LLM ``{"questions": [...]}`` payload into ``item_model`` instances.

    Never raises: returns whatever could be recovered (possibly empty) so one
    malformed/truncated response degrades gracefully instead of crashing the
    stage. ``raw`` may be a JSON string (preferred) or an already-parsed object.
    """
    data = raw if isinstance(raw, (dict, list)) else safe_json_loads(raw, fallback=None)
    entries = None
    if isinstance(data, dict):
        cand = data.get("questions")
        if isinstance(cand, list):
            entries = cand
        elif isinstance(cand, dict):
            entries = [cand]
        else:
            entries = next((v for v in data.values() if isinstance(v, list)), None)
    elif isinstance(data, list):
        entries = data
    if entries is None:
        # Top-level JSON unparseable — salvage individual objects from the text.
        entries = _salvage_question_dicts(raw if isinstance(raw, str) else "")
    items = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        try:
            items.append(item_model(**entry))
        except Exception:
            continue
    return items


def _topic_block(research_topic: Optional[str]) -> str:
    """Prompt block carrying the seed research topic (without it the questions drift
    off the seed's theme across rounds)."""
    if not research_topic:
        return ""
    return (f"Seed research topic (every question must stay within it): {research_topic}\n"
            "Do not replace the topic with a different subject.")


def _merge_research_questions(q1: ResearchQuestion, q2: ResearchQuestion) -> ResearchQuestion:
    """Combine two questions chosen for merging into one (the Contextualizer reformulates it)."""
    return ResearchQuestion(
        question=f"{q1.question} (merged with: {q2.question})",
        rationale=f"{q1.rationale} Additionally, {q2.rationale}",
        methodology_hints=list(dict.fromkeys((q1.methodology_hints or []) + (q2.methodology_hints or []))),
        related_concepts=list(dict.fromkeys((q1.related_concepts or []) + (q2.related_concepts or []))),
        feasibility_score=max(q1.feasibility_score or 0, q2.feasibility_score or 0),
    )


def apply_selection_feedback(
    questions: List[ResearchQuestion],
    feedback: Optional[IntegrationFeedback],
    min_keep: int = 3,
) -> Tuple[List[ResearchQuestion], Dict]:
    """Apply merge/discard feedback to the questions the reviewer saw, in ONE numbering.

    The feedback indices refer to the list under review (the previous round's prioritized
    questions, in rank order). Discards never leave fewer than ``min_keep`` questions (the
    highest-ranked discards are restored first); merges apply only between surviving
    indices and never reuse an index. Returns (questions, record)."""
    if not feedback or not questions:
        return list(questions), {}
    plan = plan_selection(len(questions), feedback.questions_to_discard,
                          feedback.questions_to_merge, min_keep=min_keep)
    merged = {i: _merge_research_questions(questions[i], questions[j]) for i, j in plan["merge"]}
    # Questions the reviewer asked to prioritize move to the front (stable otherwise).
    prio = {int(i) for i in (feedback.questions_to_prioritize or [])}
    keep = sorted(plan["keep"], key=lambda i: 0 if i in prio else 1)
    out = [merged.get(i, questions[i]) for i in keep]
    record = {k: plan[k] for k in ("discard", "merge", "restored", "dropped_merges")}
    record["prioritized_first"] = [i for i in keep if i in prio]
    record["n_in"], record["n_out"] = len(questions), len(out)
    return out, record


def topic_check(questions: List[PrioritizedQuestion], research_topic: Optional[str]) -> List[Dict]:
    """Deterministic on-topic check of final questions against the seed topic's content
    words; on-topic questions are ranked ahead of off-topic ones (stable), and each
    question carries ``topic_overlap``/``on_topic``. Returns the off-topic records."""
    if not research_topic or not questions:
        return []
    for q in questions:
        text = " ".join([q.question or "", q.theoretical_framework or "", q.rationale or ""])
        q.topic_overlap = topic_overlap(research_topic, text)
        q.on_topic = is_on_topic(research_topic, text)
    if all(q.on_topic is None for q in questions):
        return []
    questions.sort(key=lambda q: 0 if q.on_topic is not False else 1)
    for rank, q in enumerate(questions, 1):
        q.priority_rank = rank
    return [{"question": q.question, "topic_overlap": q.topic_overlap}
            for q in questions if q.on_topic is False]


def _unframed(q: ResearchQuestion) -> ContextualizedQuestion:
    """A given question carried forward without theoretical framing."""
    return ContextualizedQuestion(
        question=q.question, theoretical_framework="", literature_gaps=[],
        contribution=q.rationale or "", related_theories=[],
        priority_score=q.feasibility_score)


def _ancestry_record(step: str, entries: List[Dict], rejected: List[Dict], n_in: int) -> Dict:
    counts: Dict[str, int] = {}
    for e in entries:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    return {"step": step, "n_in": n_in, "n_out": len(entries), "counts": counts,
            "questions": [{k: e.get(k) for k in ("source_id", "status", "similarity", "text",
                                                 "proposed_wording")} for e in entries],
            "rejected": rejected}


def check_contextualizer_ancestry(
    questions: List[ResearchQuestion],
    framed: List[ContextualizedQuestion],
    research_topic: Optional[str] = None,
) -> Tuple[List[ContextualizedQuestion], Dict]:
    """Keep only framings that cite a given question's id (once) and reformulate
    it; every given question comes back exactly once, in input order, carried forward
    unframed when no accepted framing exists for it. Never adds questions."""
    entries, rejected = validate_ancestry(
        [q.question for q in questions],
        [(c.question, c.source_id) for c in framed],
        topic=research_topic, slots=len(questions))
    out = []
    for e in sorted(entries, key=lambda e: e["source_index"]):
        # only a framing that kept the question verbatim is used; a reworded framing describes
        # a different question, so the source is carried forward and the wording only recorded
        if e["status"] == ACCEPTED:
            item = framed[e["output_index"]]
            item.question = questions[e["source_index"]].question   # the source's exact text
        else:
            item = _unframed(questions[e["source_index"]])
        item.source_id = e["source_id"]
        item.ancestry = e["status"]
        item.ancestry_similarity = e["similarity"]
        out.append(item)
    return out, _ancestry_record("contextualizer", entries, rejected, len(questions))


def check_finalizer_ancestry(
    sources: List[ContextualizedQuestion],
    finals: List[PrioritizedQuestion],
    max_questions: int,
    research_topic: Optional[str] = None,
) -> Tuple[List[PrioritizedQuestion], Dict]:
    """A final question is kept only if it cites an unused source id and
    reformulates that source; otherwise its source replaces it unchanged (valid id) or it
    is dropped and an uncited source is carried forward (no valid id). Output count is
    min(#outputs, max_questions, #sources); ranks are renumbered 1..k in output order."""
    finals = sorted(finals, key=lambda x: (x.priority_rank if x.priority_rank is not None else 9999))
    entries, rejected = validate_ancestry(
        [c.question for c in sources],
        [(p.question, p.source_id) for p in finals],
        topic=research_topic, slots=min(len(finals), max_questions))
    out = []
    for e in entries:
        if e["status"] != ACCEPTED:     # reworded or uncited: publish the source
            src = sources[e["source_index"]]
            item = PrioritizedQuestion(
                question=src.question,
                theoretical_framework=src.theoretical_framework or "",
                rationale=src.contribution or "",
                methodology=[],
                expected_impact=src.contribution or "",
                feasibility="",
                priority_score=src.priority_score,
            )
        else:
            item = finals[e["output_index"]]
            item.question = sources[e["source_index"]].question      # the source's exact text
        item.source_id = e["source_id"]
        item.ancestry = e["status"]
        item.ancestry_similarity = e["similarity"]
        out.append(item)
    for rank, item in enumerate(out, 1):
        item.priority_rank = rank
    return out, _ancestry_record("finalizer", entries, rejected, len(sources))


class Contextualizer:
    """Agent that provides theoretical framing for research questions."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "Contextualizer"
        self.api_key = openai_api_key
        self.collector = collector
        self.llm = LLMClient(
            temperature=0.5,
            api_key=openai_api_key,
            collector=collector,
            agent_name="Contextualizer",
        )
    
    def contextualize_questions(
        self,
        questions: List[ResearchQuestion],
        feedback: Optional[IntegrationFeedback] = None,
        research_topic: Optional[str] = None
    ) -> List[ContextualizedQuestion]:
        """Provide theoretical framing for research questions."""
        
        print(f"\n[{self.agent_name}] Contextualizing {len(questions)} research questions...")
        
        # Prepare questions summary
        questions_summary = self._prepare_questions_summary(questions)
        
        # Prepare feedback context
        feedback_context = ""
        if feedback:
            feedback_context = f"""
            Previous feedback:
            - Merges and discards have already been applied to the list below.
            - Additional guidance: {feedback.additional_guidance}
            """
        
        result = self.llm.format_and_invoke(
            system_prompt="You are Contextualizer, an expert at providing theoretical framing and positioning for research questions. Before your conclusion, populate `reasoning` with the evidence you weighed and `alternatives_considered` with >=2 framings you rejected and why. You must respond with valid JSON only.",
            user_prompt="""Provide theoretical framing for the following research questions. Frame exactly the questions given, copying each question's text verbatim into "question"; do not add, drop, reword, or replace questions. Each question has an id (Q1, Q2, ...); every output object must carry "source_id" set to the id of the question it frames, and each id is used exactly once.

            {topic_block}

            Research Questions:
            {questions_summary}

            {feedback_context}

            For each question, provide:
            - Theoretical framework: Position the question within existing theoretical perspectives
            - Literature gaps: Identify specific gaps this question addresses
            - Contribution: Articulate the expected contribution to the field
            - Related theories: List relevant theoretical perspectives (3-5)

            Respond with a JSON object containing:
            - "questions": array of question objects, each with "source_id", "question", "theoretical_framework", "literature_gaps", "contribution", "related_theories", "gap_severity" (one of low|medium|high — how severe/underexplored the addressed gap is), "reasoning" (evidence weighed), and "alternatives_considered" (>=2 rejected framings, each with a reason)

            Example format:
            {{
              "questions": [
                {{
                  "source_id": "Q1",
                  "question": "How does X affect Y?",
                  "theoretical_framework": "This question builds on institutional theory...",
                  "literature_gaps": ["Gap 1", "Gap 2"],
                  "contribution": "This research will contribute by...",
                  "related_theories": ["Theory 1", "Theory 2", "Theory 3"]
                }}
              ]
            }}

            Respond with ONLY the JSON object, no other text.
            """,
            variables={
                "questions_summary": questions_summary,
                "feedback_context": feedback_context,
                "topic_block": _topic_block(research_topic),
            },
            response_format={"type": "json_object"},
        )

        # Resilient parse (safe_json_loads/repair_json + per-item try/except)
        # so a malformed/truncated response degrades gracefully, not a crash.
        contextualized_questions = _coerce_question_items(result, ContextualizedQuestion)

        # Score priority based on theoretical richness
        for question in contextualized_questions:
            question.priority_score = self._score_priority(question)
        # Ancestry: one framing per given question, citing its id.
        self.last_ancestry = None
        if contextualized_questions:
            contextualized_questions, self.last_ancestry = check_contextualizer_ancestry(
                questions, contextualized_questions, research_topic)
        
        print(f"[{self.agent_name}] Contextualized {len(contextualized_questions)} questions")
        return contextualized_questions
    
    def _prepare_questions_summary(self, questions: List[ResearchQuestion]) -> str:
        """Prepare a summary of questions for the prompt."""
        summary_parts = []
        for i, q in enumerate(questions):
            summary = f"""
Question {source_label(i)}: {q.question}
Rationale: {q.rationale}
Methodologies: {', '.join(q.methodology_hints)}
Related Concepts: {', '.join(q.related_concepts)}
Feasibility: {q.feasibility_score}
"""
            summary_parts.append(summary)
        return "\n".join(summary_parts)
    
    def _score_priority(self, question: ContextualizedQuestion) -> float:
        """Score priority from gap severity + evidence and record sub-scores.

        Components (each 0-1, combined into priority_score):
        - gap_severity: how severe/underexplored the addressed gap is, taken from
          the model's own low/medium/high judgement (defaults to medium if absent).
        - gap_evidence: how many distinct literature gaps the question targets
          (more identified gaps = stronger evidence the question matters), to 5.
        - theory_grounding: how many related theories anchor the question, to 5
          (well-anchored questions are higher-value and more defensible).
        Sub-scores are written to question.priority_components for auditability.
        """
        severity_map = {"low": 0.33, "medium": 0.66, "high": 1.0}
        sev_raw = (question.gap_severity or "medium").strip().lower()
        gap_severity = severity_map.get(sev_raw, 0.66)

        distinct_gaps = {g.strip().lower() for g in (question.literature_gaps or []) if g and g.strip()}
        gap_evidence = min(len(distinct_gaps) / 5.0, 1.0)

        distinct_theories = {t.strip().lower() for t in (question.related_theories or []) if t and t.strip()}
        theory_grounding = min(len(distinct_theories) / 5.0, 1.0)

        components = {
            "gap_severity": round(gap_severity, 3),
            "gap_evidence": round(gap_evidence, 3),
            "theory_grounding": round(theory_grounding, 3),
        }
        # Severity dominates: a severe, well-evidenced gap is the top priority.
        score = 0.5 * gap_severity + 0.3 * gap_evidence + 0.2 * theory_grounding
        question.priority_components = components
        return round(min(max(score, 0.0), 1.0), 2)


class Finalizer:
    """Agent that synthesizes and prioritizes research questions."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "Finalizer"
        self.api_key = openai_api_key
        self.collector = collector
        self.llm = LLMClient(
            temperature=0.3,
            api_key=openai_api_key,
            collector=collector,
            agent_name="Finalizer",
        )
    
    def synthesize_questions(
        self,
        contextualized_questions: List[ContextualizedQuestion],
        feedback: Optional[IntegrationFeedback] = None,
        max_questions: int = 5,
        research_topic: Optional[str] = None
    ) -> List[PrioritizedQuestion]:
        """Synthesize and prioritize research questions."""
        
        print(f"\n[{self.agent_name}] Synthesizing {len(contextualized_questions)} contextualized questions...")
        
        # Apply feedback: merge and discard questions
        processed_questions = self._apply_feedback(contextualized_questions, feedback)
        
        if not processed_questions:
            print(f"[{self.agent_name}] No questions to synthesize; skipping (no new questions are generated)")
            return []
        # The Finalizer reformulates and ranks the questions it is given; it may not add new ones.
        max_questions = min(max_questions, len(processed_questions))

        # Prepare questions summary
        questions_summary = self._prepare_contextualized_summary(processed_questions)
        
        # Prepare feedback context
        feedback_context = ""
        if feedback:
            feedback_context = f"""
            Previous feedback:
            - Questions merged: {len(feedback.questions_to_merge)} pairs
            - Questions discarded: {len(feedback.questions_to_discard)}
            - Additional guidance: {feedback.additional_guidance}
            """
        
        result = self.llm.format_and_invoke(
            system_prompt="You are Finalizer, an expert at synthesizing and prioritizing research questions. Before your conclusion, populate `reasoning` with the evidence you weighed when ranking and `alternatives_considered` with >=2 questions or rankings you rejected and why. You must respond with valid JSON only.",
            user_prompt="""Synthesize and prioritize the following contextualized research questions.

            {topic_block}

            Contextualized Questions:
            {questions_summary}

            {feedback_context}

            Return at most {max_questions} prioritized research questions. Each must be exactly one of the questions listed above, with its text copied VERBATIM, and must carry "source_id" set to that question's id (Q1, Q2, ...); use each id at most once, do not merge, reword or add questions. Your task is to select and rank them and to write the rationale, methodology, impact and feasibility for each. An output without a valid, unused source_id, or whose text differs from its source, is replaced by the source question without your annotations. They should:
            - Sharpen each question into a coherent research direction
            - Prioritize based on impact, feasibility, and theoretical contribution
            - Provide clear, actionable formulations
            - Include comprehensive rationale and methodology

            For each prioritized question:
            - Source id: id of the listed question it reformulates (e.g. "Q2")
            - Question: the listed question's text, copied verbatim
            - Theoretical framework: Integrated theoretical positioning
            - Rationale: Comprehensive justification
            - Methodology: Recommended research methods (3-5)
            - Expected impact: Anticipated contribution and impact
            - Feasibility: Assessment of research feasibility
            - Priority rank: Ranking from 1 (highest) to {max_questions}
            - Priority score: Numerical score (0-1)

            Respond with a JSON object containing:
            - "questions": array of question objects with all fields above, plus "reasoning" (evidence weighed when ranking) and "alternatives_considered" (>=2 rejected questions/rankings, each with a reason)

            Example format:
            {{
              "questions": [
                {{
                  "source_id": "Q2",
                  "question": "<the text of Q2, verbatim>",
                  "theoretical_framework": "Framework...",
                  "rationale": "Rationale...",
                  "methodology": ["Method 1", "Method 2"],
                  "expected_impact": "Impact...",
                  "feasibility": "Feasibility assessment...",
                  "priority_rank": 1,
                  "priority_score": 0.95
                }}
              ]
            }}

            Respond with ONLY the JSON object, no other text.
            """,
            variables={
                "questions_summary": questions_summary,
                "feedback_context": feedback_context,
                "topic_block": _topic_block(research_topic),
                "max_questions": str(max_questions),
            },
            response_format={"type": "json_object"},
        )

        # Resilient parse (safe_json_loads/repair_json + per-item try/except).
        prioritized_questions = _coerce_question_items(result, PrioritizedQuestion)
        
        # Ancestry: sort by rank, keep only reformulations of a cited, unused
        # source; never more questions than were given.
        self.last_ancestry = None
        if prioritized_questions:
            prioritized_questions, self.last_ancestry = check_finalizer_ancestry(
                processed_questions, prioritized_questions, max_questions, research_topic)
        
        print(f"[{self.agent_name}] Synthesized into {len(prioritized_questions)} prioritized questions")
        return prioritized_questions
    
    def _apply_feedback(
        self,
        questions: List[ContextualizedQuestion],
        feedback: Optional[IntegrationFeedback]
    ) -> List[ContextualizedQuestion]:
        """Pass-through. Merge/discard feedback is applied before contextualization by
        ``apply_selection_feedback`` against the numbering the reviewer saw; applying the
        same indices here would hit a re-generated list."""
        return list(questions)
    
    def _prepare_contextualized_summary(self, questions: List[ContextualizedQuestion]) -> str:
        """Prepare a summary of contextualized questions for the prompt."""
        summary_parts = []
        for i, q in enumerate(questions):
            summary = f"""
Question {source_label(i)}: {q.question}
Theoretical Framework: {q.theoretical_framework}
Literature Gaps: {', '.join(q.literature_gaps)}
Contribution: {q.contribution}
Related Theories: {', '.join(q.related_theories)}
Priority Score: {q.priority_score}
"""
            summary_parts.append(summary)
        return "\n".join(summary_parts)


class IntegrationOrchestrator:
    """Orchestrates the integration process with two-round human feedback."""

    def __init__(self, quiet: bool = False, collector: Optional[MetricsCollector] = None):
        self.api_key = os.getenv("OPENAI_API_KEY")

        self.collector = collector

        # Quiet mode suppresses verbose output (use with ConsoleUI)
        self.quiet = quiet or os.environ.get("AGENT_QUIET_MODE", "").lower() == "true"

        self.contextualizer = Contextualizer(self.api_key, collector=collector)
        self.finalizer = Finalizer(self.api_key, collector=collector)

        self.round_results = {}
        self.research_topic: Optional[str] = None   # seed topic (set by the caller)
        self.output_dir: Optional[str] = None       # where feedback files go
        self.selection_log: List[Dict] = []
        self.off_topic: List[Dict] = []
        self.ancestry_log: List[Dict] = []          # per-step ancestry checks

    def get_agent_stats(self) -> dict:
        """Get agent statistics for ConsoleUI."""
        return {}
    
    def load_refinement_results(self, json_path: str) -> List[ResearchQuestion]:
        """Load refined questions from Stage 2."""
        print(f"\n[Orchestrator] Loading refinement results from {json_path}...")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract all questions from the refinement stage
        questions = []
        if 'all_questions' in data:
            for q_dict in data['all_questions']:
                questions.append(ResearchQuestion(**q_dict))
        elif 'questions' in data:
            for q_dict in data['questions']:
                questions.append(ResearchQuestion(**q_dict))
        
        print(f"[Orchestrator] Loaded {len(questions)} refined questions")
        return questions
    
    def _integrate(
        self,
        questions: List[ResearchQuestion],
        feedback: Optional[IntegrationFeedback],
        max_final_questions: int,
        round_number: int,
    ) -> Tuple[List[ContextualizedQuestion], List[PrioritizedQuestion]]:
        """Contextualize -> finalize, with these guards: merge/discard feedback is applied
        to the list the reviewer saw (one numbering, survivor floor); the Finalizer is never
        called with an empty list (the previous round's set is kept); final questions are
        checked against the seed topic."""
        questions, record = apply_selection_feedback(questions, feedback)
        if record:
            record["round"] = round_number
            self.selection_log.append(record)
        self.contextualizer.last_ancestry = None
        self.finalizer.last_ancestry = None
        contextualized = self.contextualizer.contextualize_questions(
            questions=questions, feedback=feedback, research_topic=self.research_topic)
        if not contextualized and questions:
            # The Contextualizer returned nothing parseable: carry the given questions
            # forward unframed instead of handing the Finalizer an empty list.
            contextualized = [_unframed(q) for q in questions]
            for i, c in enumerate(contextualized):
                c.source_id, c.ancestry = source_label(i), CARRIED_FORWARD
        prioritized = []
        if contextualized:
            prioritized = self.finalizer.synthesize_questions(
                contextualized_questions=contextualized,
                feedback=feedback,
                max_questions=max_final_questions,
                research_topic=self.research_topic,
            )
        for agent in (self.contextualizer, self.finalizer):
            rec = getattr(agent, "last_ancestry", None)
            if rec:
                self.ancestry_log.append(dict(rec, round=round_number))
        if not prioritized:
            previous = [r for _, r in sorted(self.round_results.items()) if r.get("prioritized")]
            if previous:
                print(f"[Orchestrator] Round {round_number} produced no questions; "
                      "keeping the previous round's set")
                contextualized = previous[-1]["contextualized"]
                prioritized = previous[-1]["prioritized"]
        self.off_topic = topic_check(prioritized, self.research_topic)
        return contextualized, prioritized

    def run_integration_round(
        self,
        questions: List[ResearchQuestion],
        round_number: int,
        feedback: Optional[IntegrationFeedback] = None,
        max_final_questions: int = 5
    ) -> Tuple[List[ContextualizedQuestion], List[PrioritizedQuestion]]:
        """Run one round of integration (Contextualizer → Finalizer)."""
        
        print(f"\n{'='*70}")
        print(f"INTEGRATION ROUND {round_number}")
        print(f"{'='*70}")
        
        contextualized, prioritized = self._integrate(
            questions, feedback, max_final_questions, round_number)
        
        # Store results
        self.round_results[round_number] = {
            'contextualized': contextualized,
            'prioritized': prioritized
        }
        
        return contextualized, prioritized
    
    def print_contextualized_questions(self, round_number: int):
        """Print contextualized questions from a specific round."""
        if round_number not in self.round_results:
            print(f"No results for round {round_number}")
            return
        
        questions = self.round_results[round_number]['contextualized']
        
        print(f"\n{'='*70}")
        print(f"ROUND {round_number} - CONTEXTUALIZED QUESTIONS")
        print(f"{'='*70}\n")
        
        for i, q in enumerate(questions, 1):
            print(f"{i}. {q.question}")
            print(f"   Priority Score: {q.priority_score}")
            print(f"   Theoretical Framework: {q.theoretical_framework[:200]}...")
            print(f"   Literature Gaps: {', '.join(q.literature_gaps)}")
            print(f"   Related Theories: {', '.join(q.related_theories)}")
            print()
    
    def print_prioritized_questions(self, round_number: int):
        """Print prioritized questions from a specific round."""
        if round_number not in self.round_results:
            print(f"No results for round {round_number}")
            return
        
        questions = self.round_results[round_number]['prioritized']
        
        print(f"\n{'='*70}")
        print(f"ROUND {round_number} - PRIORITIZED RESEARCH QUESTIONS")
        print(f"{'='*70}\n")
        
        for q in questions:
            print(f"RANK {q.priority_rank} (Score: {q.priority_score})")
            print(f"Question: {q.question}")
            print(f"Theoretical Framework: {q.theoretical_framework[:200]}...")
            print(f"Rationale: {q.rationale[:200]}...")
            print(f"Methodology: {', '.join(q.methodology)}")
            print(f"Expected Impact: {q.expected_impact[:150]}...")
            print(f"Feasibility: {q.feasibility[:150]}...")
            print()
    
    def collect_integration_feedback(
        self,
        round_number: int,
        num_questions: int
    ) -> IntegrationFeedback:
        """Collect human feedback for integration stage."""
        print(f"\n{'='*70}")
        print(f"INTEGRATION FEEDBACK - Round {round_number}")
        print(f"{'='*70}\n")
        
        print("Please provide feedback on the prioritized research questions.")
        print("Enter question numbers (1-indexed) for each action.")
        print("Press Enter to skip any field.\n")
        
        # Questions to merge
        merge_input = auto_input(
            "Enter pairs of questions to merge (e.g., '1,2 3,4'): ",
            default=get_default("questions_to_merge"),
        ).strip()
        questions_to_merge = []
        if merge_input:
            pairs = merge_input.split()
            for pair in pairs:
                try:
                    idx1, idx2 = pair.split(',')
                    questions_to_merge.append((int(idx1) - 1, int(idx2) - 1))
                except:
                    print(f"Invalid pair format: {pair}")
        
        # Questions to discard
        discard_input = auto_input(
            "Enter questions to discard (comma-separated, e.g., '2,5'): ",
            default=get_default("questions_to_discard"),
        ).strip()
        questions_to_discard = []
        if discard_input:
            try:
                questions_to_discard = [int(x.strip()) - 1 for x in discard_input.split(',')]
            except:
                print("Invalid discard format")
        
        # Questions to prioritize
        prioritize_input = auto_input(
            "Enter questions to prioritize (comma-separated, e.g., '1,3'): ",
            default=get_default("questions_to_prioritize"),
        ).strip()
        questions_to_prioritize = []
        if prioritize_input:
            try:
                questions_to_prioritize = [int(x.strip()) - 1 for x in prioritize_input.split(',')]
            except:
                print("Invalid prioritize format")
        
        # Additional guidance
        guidance = auto_input(
            "Additional guidance for refinement: ",
            default=get_default("additional_guidance"),
        ).strip()
        
        feedback = IntegrationFeedback(
            round_number=round_number,
            questions_to_merge=questions_to_merge,
            questions_to_discard=questions_to_discard,
            questions_to_prioritize=questions_to_prioritize,
            additional_guidance=guidance
        )
        
        # Save feedback to JSON
        feedback_file = os.path.join(getattr(self, "output_dir", None) or ".", f"round{round_number}_integration_feedback.json")
        with open(feedback_file, 'w') as f:
            json.dump(feedback.model_dump(), f, indent=2)
        
        print(f"\n[Orchestrator] Feedback saved to {feedback_file}")
        
        return feedback
    
    def convert_prioritized_to_research_questions(
        self,
        prioritized: List[PrioritizedQuestion]
    ) -> List[ResearchQuestion]:
        """Convert prioritized questions back to ResearchQuestion format for next round."""
        questions = []
        for pq in prioritized:
            q = ResearchQuestion(
                question=pq.question,
                rationale=pq.rationale,
                methodology_hints=pq.methodology,
                related_concepts=[],  # Not applicable in this stage
                feasibility_score=pq.priority_score
            )
            questions.append(q)
        return questions
    
    def run_automated_integration(
        self,
        questions: List[ResearchQuestion],
        max_final_questions: int = 5
    ) -> List[PrioritizedQuestion]:
        """Run automated single-round integration without human feedback."""
        print(f"\n{'='*70}")
        print(f"AUTOMATED QUESTION INTEGRATION & PRIORITIZATION")
        print(f"{'='*70}")
        
        contextualized, prioritized = self._integrate(
            questions, None, max_final_questions, 1)
        
        # Store results
        self.round_results[1] = {
            'contextualized': contextualized,
            'prioritized': prioritized
        }
        
        print(f"\n[Orchestrator] Generated {len(prioritized)} prioritized questions")
        
        return prioritized
    
    def save_results(self, filename: str, round_number: Optional[int] = None):
        """Save integration results to JSON file."""
        if round_number:
            # Save specific round
            if round_number not in self.round_results:
                print(f"No results for round {round_number}")
                return
            
            data = {
                'round': round_number,
                'contextualized': [q.model_dump() for q in self.round_results[round_number]['contextualized']],
                'prioritized': [q.model_dump() for q in self.round_results[round_number]['prioritized']]
            }
            output_file = f"round{round_number}_{filename}"
        else:
            # Save all rounds
            data = {
                'by_round': {
                    str(rnd): {
                        'contextualized': [q.model_dump() for q in results['contextualized']],
                        'prioritized': [q.model_dump() for q in results['prioritized']]
                    }
                    for rnd, results in self.round_results.items()
                }
            }
            output_file = filename
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[Orchestrator] Results saved to {output_file}")
    
    def save_final_questions(self, filename: str = "finalized_research_questions.json"):
        """Save the final prioritized questions from the last round."""
        if not self.round_results:
            print("No results to save")
            return
        
        # Get the last round's prioritized questions
        last_round = max(self.round_results.keys())
        final_questions = self.round_results[last_round]['prioritized']
        
        _final_dump = [q.model_dump() for q in final_questions]
        # Emit both canonical 'prioritized_questions' (IntegrationStageOutput)
        # and legacy 'final_questions' — pipeline/team_runners.py accepts either.
        data = {
            'timestamp': datetime.now().isoformat(),
            'total_rounds': len(self.round_results),
            'prioritized_questions': _final_dump,
            'final_questions': _final_dump,
            'research_topic': self.research_topic,
            'off_topic_questions': self.off_topic,
            'selection_feedback_applied': self.selection_log,
            'ancestry_checks': self.ancestry_log,
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n[Orchestrator] Final questions saved to {filename}")
        
        # Also save as readable text
        text_filename = filename.replace('.json', '.txt')
        with open(text_filename, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("FINALIZED RESEARCH QUESTIONS\n")
            f.write("="*70 + "\n\n")
            
            for q in final_questions:
                f.write(f"RANK {q.priority_rank} (Priority Score: {q.priority_score})\n")
                f.write(f"{'='*70}\n\n")
                f.write(f"QUESTION:\n{q.question}\n\n")
                f.write(f"THEORETICAL FRAMEWORK:\n{q.theoretical_framework}\n\n")
                f.write(f"RATIONALE:\n{q.rationale}\n\n")
                f.write(f"METHODOLOGY:\n{', '.join(q.methodology)}\n\n")
                f.write(f"EXPECTED IMPACT:\n{q.expected_impact}\n\n")
                f.write(f"FEASIBILITY:\n{q.feasibility}\n\n")
                f.write("-"*70 + "\n\n")
        
        print(f"[Orchestrator] Final questions also saved to {text_filename}")


def main():
    """Main function with two-round integration process."""
    
    # Configuration
    refinement_results = "refinement_results_all_rounds.json"  # Output from Stage 2
    max_final_questions = 5
    
    print("="*70)
    print("STARTING TWO-ROUND RESEARCH QUESTION INTEGRATION")
    print("="*70)
    
    # Initialize orchestrator
    orchestrator = IntegrationOrchestrator()
    
    # Load refined questions from Stage 2
    try:
        initial_questions = orchestrator.load_refinement_results(refinement_results)
    except FileNotFoundError:
        print(f"\nError: {refinement_results} not found!")
        print("Please run 2-RefinementStage.py first to generate refinement results.")
        return
    
    # ROUND 1: Initial contextualization and synthesis
    contextualized1, prioritized1 = orchestrator.run_integration_round(
        questions=initial_questions,
        round_number=1,
        feedback=None,
        max_final_questions=max_final_questions
    )
    
    # Display Round 1 results
    orchestrator.print_contextualized_questions(round_number=1)
    orchestrator.print_prioritized_questions(round_number=1)
    
    # Save Round 1 results
    orchestrator.save_results("integration_results.json", round_number=1)
    
    # Collect human feedback for Round 1
    feedback1 = orchestrator.collect_integration_feedback(
        round_number=1,
        num_questions=len(prioritized1)
    )
    
    # Convert prioritized questions back to ResearchQuestion format for Round 2
    questions_for_round2 = orchestrator.convert_prioritized_to_research_questions(prioritized1)
    
    # ROUND 2: Refined contextualization and synthesis based on feedback
    contextualized2, prioritized2 = orchestrator.run_integration_round(
        questions=questions_for_round2,
        round_number=2,
        feedback=feedback1,
        max_final_questions=max_final_questions
    )
    
    # Display Round 2 results
    orchestrator.print_contextualized_questions(round_number=2)
    orchestrator.print_prioritized_questions(round_number=2)
    
    # Save Round 2 results
    orchestrator.save_results("integration_results.json", round_number=2)
    
    # Save all results
    orchestrator.save_results("integration_results_all_rounds.json")
    
    # Save final questions
    orchestrator.save_final_questions("finalized_research_questions.json")
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Initial questions from Stage 2: {len(initial_questions)}")
    print(f"Round 1 prioritized questions: {len(prioritized1)}")
    print(f"Round 2 prioritized questions: {len(prioritized2)}")
    
    print(f"\n{'='*70}")
    print("INTEGRATION PROCESS COMPLETE")
    print(f"{'='*70}")
    print("Files created:")
    print("  - round1_integration_results.json")
    print("  - round1_integration_feedback.json")
    print("  - round2_integration_results.json")
    print("  - integration_results_all_rounds.json")
    print("  - finalized_research_questions.json")
    print("  - finalized_research_questions.txt")
    
    print(f"\n{'='*70}")
    print("FINALIZED RESEARCH QUESTIONS")
    print(f"{'='*70}\n")
    
    for q in prioritized2:
        print(f"RANK {q.priority_rank}: {q.question}")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
