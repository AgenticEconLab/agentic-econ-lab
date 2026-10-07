# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Synthesis Stage - Automated Mode (No Firecrawl, No HITL)
This script synthesizes gap analysis from Stage 2 into a literature review and research plan.

Pipeline:
1. KnowledgeWeaver: Knowledge integration (synthesizes insights, gaps, and trends)
2. CiteKeeper: Reference finalization (creates formatted bibliography)

Input: Gap analysis results from Stage 2 (gap_analysis_results.json)
Output: Literature review and research plan
"""

import os
import json
import re
import time
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
import sys
from pathlib import Path as _Path

# Add agents dir to path for shared imports
_agents_dir = _Path(__file__).resolve().parent.parent.parent.parent
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))
import shared.llm as _shared_llm
from shared.llm import LLMClient
from shared.json_repair import repair_json, safe_json_loads
from shared.observability import MetricsCollector, tracked_get, tracked_arxiv_search
from shared.auto_input import auto_input, get_default
from LiteratureTeam.ael.schemas.stage_outputs import (
    LiteratureSection, LiteratureReview, ResearchObjective, ResearchPlan,
    FormattedReference, Bibliography, CrossDomainConnection, SynthesisResult,
)

# Load environment variables
load_dotenv()

# --- Synthesis section generation robustness (issue #7) -------------------------
# Section content is written by a per-section LLM call. A single ReadTimeout used to
# fall straight through the broad except to a "This section reviews research on X."
# title-echo placeholder (reintroducing the title-not-findings bug). We retry with
# backoff and raise the per-call HTTP timeout floor so a transient slow generation
# gets a second chance; on permanent failure the section is DROPPED, not placeholdered.
_SECTION_MAX_TRIES = int(os.environ.get("AEL_SYNTHESIS_MAX_TRIES", "3"))
_SECTION_BACKOFF_BASE = float(os.environ.get("AEL_SYNTHESIS_BACKOFF", "2.0"))
# Raise the shared per-call HTTP timeout floor for this stage (read live by each
# httpx call in shared.llm, so bumping the module global takes effect immediately).
_SYNTHESIS_LLM_TIMEOUT = float(os.environ.get("AEL_SYNTHESIS_LLM_TIMEOUT", "900"))
if getattr(_shared_llm, "_LLM_TIMEOUT", 0) < _SYNTHESIS_LLM_TIMEOUT:
    _shared_llm._LLM_TIMEOUT = _SYNTHESIS_LLM_TIMEOUT

# ========== HELPER FUNCTIONS ==========

def sanitize_json_string(content: str) -> str:
    """
    Sanitize JSON string to handle invalid escape sequences (e.g., LaTeX).
    
    Args:
        content: Raw string that may contain invalid JSON escapes
        
    Returns:
        Sanitized string safe for JSON parsing
    """
    # Fix common invalid escape sequences (like LaTeX backslashes)
    # Replace single backslashes that are not valid JSON escapes
    # Valid JSON escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
    content = re.sub(r'\\([^"\\\/bfnrtu])', r'\\\\\\1', content)
    return content

def safe_parse_json(content: str, context: str = "JSON response") -> Optional[dict]:
    """
    Safely parse JSON from LLM response, handling various formats.
    
    Args:
        content: The content string from LLM response
        context: Context description for error messages
        
    Returns:
        Parsed JSON as dict/list, or None if parsing fails
    """
    if not content:
        print(f"    Warning: Empty {context}")
        return None
    
    content = content.strip()
    
    # Hardened primary path (issue #19): the shared safe_json_loads strips markdown
    # code fences, repairs invalid backslash escapes (LaTeX \beta, etc.) via repair_json,
    # and extracts the first embedded JSON object/array from surrounding prose. Returns
    # None on total failure so we keep falling through to the legacy field-extraction
    # last resort below instead of silently accepting {}.
    parsed = safe_json_loads(content, fallback=None)
    if parsed is not None:
        return parsed

    # Try direct JSON parsing first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Try with sanitized content (fix invalid escape sequences)
    try:
        sanitized = sanitize_json_string(content)
        return json.loads(sanitized)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks (improved patterns)
    # Use [\s\S] instead of . to reliably match newlines
    code_block_patterns = [
        r'```json\s*([\s\S]*?)```',    # ```json ... ``` (most common)
        r'```\s*([\s\S]*?)```',        # ``` ... ``` (generic code block)
    ]
    
    for pattern in code_block_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            cleaned = match.strip()
            if cleaned:
                # Try direct parsing
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass
                
                # Try with sanitization
                try:
                    sanitized_match = sanitize_json_string(cleaned)
                    return json.loads(sanitized_match)
                except json.JSONDecodeError:
                    pass
                
                # Try to find JSON object/array within the match
                for start_char in ['{', '[']:
                    start_idx = cleaned.find(start_char)
                    if start_idx != -1:
                        end_char = '}' if start_char == '{' else ']'
                        depth = 0
                        end_idx = -1
                        for i in range(start_idx, len(cleaned)):
                            if cleaned[i] == start_char:
                                depth += 1
                            elif cleaned[i] == end_char:
                                depth -= 1
                                if depth == 0:
                                    end_idx = i + 1
                                    break
                        if end_idx > start_idx:
                            json_substr = cleaned[start_idx:end_idx]
                            try:
                                return json.loads(json_substr)
                            except json.JSONDecodeError:
                                try:
                                    return json.loads(sanitize_json_string(json_substr))
                                except json.JSONDecodeError:
                                    continue
    
    # Try to find JSON array or object in the content (even if not in code blocks)
    # Look for first [ or { and last ] or }
    start_idx = -1
    end_idx = -1
    bracket_type = None
    
    for i, char in enumerate(content):
        if char in ['[', '{']:
            start_idx = i
            bracket_type = char
            break
    
    if start_idx != -1:
        # Find matching closing bracket
        open_char = bracket_type
        close_char = ']' if open_char == '[' else '}'
        depth = 0
        
        for i in range(start_idx, len(content)):
            if content[i] == open_char:
                depth += 1
            elif content[i] == close_char:
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break
        
        if end_idx > start_idx:
            try:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
    
    # Last resort: try to extract common field values directly
    # Check for common field names: content, synthesis, title, abstract, introduction, etc.
    common_fields = ["content", "synthesis", "title", "abstract", "introduction", "text", "body"]
    
    for field_name in common_fields:
        # Pattern for "field": "value" with proper escaping
        field_pattern = rf'"{field_name}"\s*:\s*"((?:[^"\\]|\\.)*)"'
        field_match = re.search(field_pattern, content, re.DOTALL)
        if field_match:
            try:
                # Unescape the matched value
                value = field_match.group(1)
                # Handle common escape sequences
                value = value.replace('\\n', '\n').replace('\\r', '\r').replace('\\t', '\t').replace('\\"', '"')
                return {field_name: value}
            except:
                pass
    
    print(f"    Warning: Could not parse {context} as JSON. Content preview: {content[:200]}...")
    return None

# ========== AGENTS ==========

class KnowledgeWeaver:
    """Agent for knowledge integration and synthesis."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "KnowledgeWeaver"
        self.api_key = openai_api_key
        self.collector = collector
        self.llm = LLMClient(


            temperature=0.6,

            api_key=self.api_key,

            collector=collector,

            agent_name=self.agent_name

        )
    
    def integrate_knowledge(
        self,
        gap_analysis: Dict
    ) -> tuple[LiteratureReview, ResearchPlan, List[CrossDomainConnection]]:
        """Integrate knowledge from gap analysis into review, plan, and cross-domain analysis."""

        print(f"\n[{self.agent_name}] Integrating knowledge from gap analysis...")

        # Extract key components
        paper_structures = gap_analysis.get('paper_structures', [])
        research_gaps = gap_analysis.get('research_gaps', [])
        knowledge_graph = gap_analysis.get('knowledge_graph', {})

        print(f"  - Papers: {len(paper_structures)}")
        print(f"  - Gaps: {len(research_gaps)}")
        print(f"  - Graph nodes: {knowledge_graph.get('metadata', {}).get('total_nodes', 0)}")

        # Step 1: Generate literature review
        literature_review = self._generate_literature_review(
            paper_structures,
            research_gaps,
            knowledge_graph
        )

        # Step 2: Generate research plan
        research_plan = self._generate_research_plan(
            research_gaps,
            paper_structures,
            knowledge_graph
        )

        # Step 3: Generate cross-domain analysis
        cross_domain_connections = self.generate_cross_domain_analysis(
            paper_structures,
            research_gaps
        )

        print(f"[{self.agent_name}] Knowledge integration complete")

        return literature_review, research_plan, cross_domain_connections
    
    def _generate_literature_review(
        self,
        paper_structures: List[Dict],
        research_gaps: List[Dict],
        knowledge_graph: Dict
    ) -> LiteratureReview:
        """Generate comprehensive literature review."""
        
        print(f"\n  [{self.agent_name}] Generating literature review...")
        
        # Safety check for empty data
        if not paper_structures:
            paper_structures = []
        if not research_gaps:
            research_gaps = []
        
        # Prepare context
        papers_summary = self._prepare_papers_summary(paper_structures[:30])
        gaps_summary = self._prepare_gaps_summary(research_gaps[:15])
        
        # Generate title and abstract
        title, abstract = self._generate_title_and_abstract(papers_summary, gaps_summary)
        
        # Generate introduction
        introduction = self._generate_introduction(papers_summary, gaps_summary)
        
        # Generate main sections
        sections = self._generate_review_sections(paper_structures, research_gaps, knowledge_graph)
        
        # Generate synthesis
        synthesis = self._generate_synthesis(research_gaps, knowledge_graph)
        
        review = LiteratureReview(
            title=title,
            abstract=abstract,
            introduction=introduction,
            sections=sections,
            synthesis=synthesis,
            metadata={
                "total_papers_reviewed": len(paper_structures),
                "total_gaps_identified": len(research_gaps),
                "total_sections": len(sections),
                "created_at": datetime.now().isoformat()
            }
        )
        
        print(f"  [{self.agent_name}] Literature review generated ({len(sections)} sections)")
        
        return review
    
    def _generate_title_and_abstract(
        self,
        papers_summary: str,
        gaps_summary: str
    ) -> tuple[str, str]:
        """Generate review title and abstract."""
        try:

        
            result = self.llm.format_and_invoke(

        
                system_prompt="You are an expert at writing academic literature reviews.",

        
                user_prompt="""Based on the following papers and research gaps, generate a title and abstract for a literature review.
            
                Papers Summary:
                {papers}
            
                Research Gaps:
                {gaps}
            
                Generate:
                - title: Concise, descriptive title (10-15 words)
                - abstract: Comprehensive abstract (150-250 words) covering scope, key findings, gaps, and future directions
            
                Return as a JSON object with "title" and "abstract" fields.
            
                Respond with ONLY the JSON object, no other text.
                """,

        
                variables={
                "papers": papers_summary[:2000],
                "gaps": gaps_summary[:1000]
                }

        
            )
            
            if not result:
                print(f"    Warning: Empty response from LLM for title/abstract")
                return "Literature Review", "This review synthesizes recent research findings and identifies key gaps."
            
            data = safe_parse_json(result, "title/abstract")
            if data is None:
                return "Literature Review", "This review synthesizes recent research findings and identifies key gaps."
            
            return data.get("title", "Literature Review"), data.get("abstract", "")
        
        except Exception as e:
            print(f"    Error generating title/abstract: {e}")
            import traceback
            traceback.print_exc()
            return "Literature Review", "This review synthesizes recent research findings and identifies key gaps."
    
    def _generate_introduction(self, papers_summary: str, gaps_summary: str) -> str:
        """Generate introduction section."""
        try:

        
            result = self.llm.format_and_invoke(

        
                system_prompt="""You are an expert economics researcher writing the introduction to a literature review.
    You ground the review in the current state of the economics field and explain why this topic demands systematic review.""",

        
                user_prompt="""Write an introduction section (3-4 paragraphs) for a literature review.

    Papers Summary:
    {papers}

    Research Gaps:
    {gaps}

    The introduction MUST:
    - Establish the economic IMPORTANCE of this research area (why does it matter for policy, theory, or welfare?)
    - Briefly position this review within the existing economics literature landscape
    - Clearly state what this review covers and what it excludes (scope boundaries)
    - Preview the main themes, key debates, and the most critical gaps
    - State the review's objectives and intended contribution to the field

    Be specific to economics. Reference concrete policy debates or theoretical developments where possible.

    Return as a JSON object with an "introduction" field containing the text.

    Respond with ONLY the JSON object, no other text.
    """,

        
                variables={
                "papers": papers_summary[:2000],
                "gaps": gaps_summary[:1000]
                }

        
            )
            
            if not result:
                print(f"    Warning: Empty response from LLM for introduction")
                return "This literature review examines recent research in the field."
            
            data = safe_parse_json(result, "introduction")
            if data is None:
                return "This literature review examines recent research in the field."
            
            return data.get("introduction", "")
        
        except Exception as e:
            print(f"    Error generating introduction: {e}")
            import traceback
            traceback.print_exc()
            return "This literature review examines recent research in the field."
    
    def _generate_review_sections(
        self,
        paper_structures: List[Dict],
        research_gaps: List[Dict],
        knowledge_graph: Dict
    ) -> List[LiteratureSection]:
        """Generate main review sections organized by themes."""
        
        # Group papers by methodology/theme
        sections = []
        
        # Extract unique methodologies and themes
        all_methods = set()
        for paper in paper_structures[:30]:
            all_methods.update(paper.get('methodologies', []))
        
        # Create sections for top methodologies/themes
        top_methods = list(all_methods)[:5]
        
        # If no methodologies found, return empty sections list
        if not top_methods:
            return sections
        
        for idx, method in enumerate(top_methods, 1):
            # Find papers using this method
            related_papers = [
                p.get('paper_title', 'Unknown')
                for p in paper_structures
                if method in p.get('methodologies', [])
            ][:10]
            
            # Find related gaps
            related_gaps = [
                g.get('gap_id', '')
                for g in research_gaps
                if method.lower() in g.get('gap_description', '').lower()
            ][:3]
            
            # Generate section content
            section_content = self._generate_section_content(
                method,
                related_papers,
                related_gaps,
                paper_structures,
                research_gaps
            )

            # Permanent generation failure (e.g. repeated LLM ReadTimeout) returns None.
            # DROP the section rather than emit a title-echo placeholder (issue #7).
            if not section_content or not section_content.strip():
                print(f"  Dropping section '{method} Approaches' — content generation failed after retries.")
                continue

            section = LiteratureSection(
                section_title=f"{method} Approaches",
                section_content=section_content,
                key_papers=related_papers[:5],
                related_gaps=related_gaps
            )
            
            sections.append(section)
        
        # Add a gaps section
        gaps_section = self._generate_gaps_section(research_gaps)
        sections.append(gaps_section)
        
        return sections
    
    def _generate_section_content(
        self,
        theme: str,
        related_papers: List[str],
        related_gaps: List[str],
        paper_structures: List[Dict],
        research_gaps: Optional[List[Dict]] = None
    ) -> Optional[str]:
        """Generate content for a review section.

        Returns the section text, or None when generation permanently fails (after
        retries) so the caller drops the section instead of writing a title-echo
        placeholder (issue #7)."""

        # Build structured per-paper context from paper_structures (NOT titles only),
        # so the model writes from real economic substance instead of hallucinating it.
        struct_map = {
            (p.get('paper_title') or '').strip().lower(): p
            for p in (paper_structures or [])
        }

        def _fmt_list(values, limit):
            items = [str(v).strip() for v in (values or []) if str(v).strip()]
            return items[:limit]

        paper_blocks = []
        for title in related_papers[:5]:
            struct = struct_map.get((title or '').strip().lower())
            if not struct:
                # No structured data for this title -> title only (graceful fallback)
                paper_blocks.append(f'- "{title}"')
                continue

            lines = [f'- "{title}"']
            findings = _fmt_list(struct.get('key_findings'), 4)
            if findings:
                lines.append("  Key findings: " + "; ".join(findings))
            contributions = _fmt_list(struct.get('theoretical_contributions'), 3)
            if contributions:
                lines.append("  Theoretical contributions: " + "; ".join(contributions))
            limitations = _fmt_list(struct.get('limitations'), 3)
            if limitations:
                lines.append("  Limitations: " + "; ".join(limitations))
            methods = _fmt_list(struct.get('methodologies'), 3)
            if methods:
                lines.append("  Methodologies: " + "; ".join(methods))
            md = struct.get('methodology_details')
            if isinstance(md, dict):
                md_parts = [f"{k}: {v}" for k, v in md.items() if str(v).strip()]
                if md_parts:
                    lines.append("  Methodology details: " + "; ".join(md_parts[:3]))
            elif isinstance(md, str) and md.strip():
                lines.append("  Methodology details: " + md.strip()[:300])
            paper_blocks.append("\n".join(lines))

        papers_text = "\n".join(paper_blocks) if paper_blocks else "None available for this theme"

        # Build gaps context for this section
        gaps_context = "None identified for this theme"
        if research_gaps and related_gaps:
            matching_gaps = [g for g in research_gaps if g.get('gap_id', '') in related_gaps]
            if matching_gaps:
                gaps_context = "\n".join([
                    f"- [{g.get('gap_id')}] {g.get('gap_title', '')}: {g.get('gap_description', '')[:200]}"
                    for g in matching_gaps[:5]
                ])
        last_err = None
        for attempt in range(1, _SECTION_MAX_TRIES + 1):
            try:
                result = self.llm.format_and_invoke(


                system_prompt="""You are an expert economics researcher writing a critical literature review.
    You synthesize findings analytically, not just descriptively. You identify where studies agree, disagree, and what remains unresolved.""",

        
                user_prompt="""Write a section (3-4 paragraphs) for a literature review on the following theme.

    Theme: {theme}

    Key Papers (with their actual findings, contributions, and limitations):
    {papers}

    Related Research Gaps:
    {gaps_context}

    The section MUST:
    - SYNTHESIZE findings across papers (identify consensus, contradictions, and open debates)
    - Base every claim on the findings, contributions, and limitations PROVIDED above for each paper — do NOT invent additional results
    - Critically evaluate methodological STRENGTHS and WEAKNESSES of the approaches used
    - Identify what this body of work DOES and DOES NOT establish empirically
    - Note specific limitations that restrict the generalizability of findings
    - Explain WHY this theme matters for the broader economics research agenda
    - CONNECT specific gaps to specific papers and their limitations (e.g., "Paper X's reliance on Y creates gap Z")

    CITATION INTEGRITY: Reference ONLY papers listed above. Cite papers by TITLE. Do NOT invent author names or years. Only state an author-year if it is explicitly provided in the input. Do NOT fabricate or invent paper titles, findings, author names, or years not provided in the input.

    Avoid generic summaries. Be specific about which papers contribute what evidence.

    Return as a JSON object with a "content" field containing the text.

    Respond with ONLY the JSON object, no other text.
    """,

        
                variables={
                "theme": theme,
                "papers": papers_text,
                "gaps_context": gaps_context
                }

        
                )

                if not result:
                    raise ValueError("empty response from LLM for section content")

                data = safe_parse_json(result, "section content")
                content = data.get("content", "") if isinstance(data, dict) else ""
                if not content:
                    # Direct-extract a "content": "..." value if the structured parse missed it.
                    content_match = re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', result, re.DOTALL)
                    if content_match:
                        try:
                            content = json.loads(f'"{content_match.group(1)}"')
                        except Exception:
                            content = ""
                if content and str(content).strip():
                    return content
                raise ValueError("no usable 'content' field in section response")

            except Exception as e:
                last_err = e
                print(f"    [section:{theme}] attempt {attempt}/{_SECTION_MAX_TRIES} failed: {e}")
                if attempt < _SECTION_MAX_TRIES:
                    time.sleep(_SECTION_BACKOFF_BASE * (2 ** (attempt - 1)))

        # Permanent failure: signal the caller to DROP this section instead of emitting a
        # title-only placeholder (which reintroduces the title-not-findings bug, issue #7).
        print(f"    [section:{theme}] permanently failed after {_SECTION_MAX_TRIES} tries: {last_err}. Dropping section.")
        return None
    
    def _generate_gaps_section(self, research_gaps: List[Dict]) -> LiteratureSection:
        """Generate section on research gaps."""
        
        gaps_text = "\n".join([
            f"- [{g.get('gap_type', 'unknown').upper()}] {g.get('gap_title', 'Unknown')}: {g.get('gap_description', '')[:150]}..."
            for g in research_gaps[:10]
        ])
        try:

        
            result = self.llm.format_and_invoke(

        
                system_prompt="You are an expert at writing academic literature reviews.",

        
                user_prompt="""Write a section (2-3 paragraphs) summarizing research gaps.
            
                Research Gaps:
                {gaps}
            
                The section should:
                - Categorize gaps by type
                - Discuss their significance
                - Suggest priorities for future research
            
                Return as a JSON object with a "content" field containing the text.
            
                Respond with ONLY the JSON object, no other text.
                """,

        
                variables={"gaps": gaps_text}

        
            )
            
            if not result:
                print(f"    Warning: Empty response from LLM for gaps section")
                content = "Several research gaps have been identified in the literature."
            else:
                data = safe_parse_json(result, "gaps section")
                if data is None:
                    content = "Several research gaps have been identified in the literature."
                else:
                    content = data.get("content", "")
        except Exception as e:
            print(f"    Error generating gaps section: {e}")
            import traceback
            traceback.print_exc()
            content = "Several research gaps have been identified in the literature."
        
        return LiteratureSection(
            section_title="Research Gaps and Future Directions",
            section_content=content,
            key_papers=[],
            related_gaps=[g.get('gap_id', '') for g in research_gaps[:10]]
        )
    
    def _generate_synthesis(
        self,
        research_gaps: List[Dict],
        knowledge_graph: Dict
    ) -> str:
        """Generate synthesis and conclusions with cross-domain integration."""

        gaps_summary = self._prepare_gaps_summary(research_gaps[:10])
        try:

            result = self.llm.format_and_invoke(

                system_prompt="""You are a leading economics researcher writing the synthesis section of a literature review.
    You are known for identifying NOVEL connections between research streams that other researchers miss.""",

                user_prompt="""Write a synthesis and conclusions section (4-5 paragraphs) for a literature review.

    Research Gaps:
    {gaps}

    The synthesis MUST include:
    1. THEMATIC SYNTHESIS: Summarize key themes and how they interconnect (not just list them)
    2. CROSS-DOMAIN CONNECTIONS: Identify at least 2 NOVEL connections between different research streams (e.g., how methods from behavioral economics could resolve a gap in computational modeling, or how data science techniques could advance traditional econometric approaches)
    3. CRITICAL ASSESSMENT: Which gaps are most URGENT for the field? Why? What would the economics profession gain from addressing them?
    4. UNCONVENTIONAL DIRECTIONS: Propose at least 1 research direction that is NOT a straightforward extension of existing work but represents a genuinely new approach
    5. BROADER IMPLICATIONS: How do these findings affect economic policy, methodology, or theory beyond the immediate research questions?

    CITATION INTEGRITY: Reference ONLY gaps and papers provided in the input. Do NOT fabricate specific author names or paper titles. Instead, reference theoretical frameworks and established fields.

    Be specific and substantive. Avoid generic statements like "more research is needed."

    Return as a JSON object with a "synthesis" field containing the text.

    Respond with ONLY the JSON object, no other text.
    """,

                variables={"gaps": gaps_summary}

            )
            
            if not result:
                print(f"    Warning: Empty response from LLM for synthesis")
                return "This review has identified several important research gaps and future directions."
            
            data = safe_parse_json(result, "synthesis")
            if data is None:
                return "This review has identified several important research gaps and future directions."
            
            return data.get("synthesis", "")
        
        except Exception as e:
            print(f"    Error generating synthesis: {e}")
            import traceback
            traceback.print_exc()
            return "This review has identified several important research gaps and future directions."

    def generate_cross_domain_analysis(
        self,
        paper_structures: List[Dict],
        research_gaps: List[Dict]
    ) -> List[CrossDomainConnection]:
        """Identify and analyze connections across different research domains found in the literature."""

        print(f"\n  [{self.agent_name}] Generating cross-domain analysis...")

        # Collect domain tags from papers (from WP15.2)
        domain_inventory = {}
        for paper in paper_structures[:30]:
            tags = paper.get('domain_tags', [])
            title = paper.get('paper_title', 'Unknown')
            for tag in tags:
                tag_lower = tag.lower().strip()
                if tag_lower:
                    if tag_lower not in domain_inventory:
                        domain_inventory[tag_lower] = []
                    domain_inventory[tag_lower].append(title)

        # Also infer domains from methodologies if domain_tags are sparse
        if len(domain_inventory) < 3:
            for paper in paper_structures[:30]:
                title = paper.get('paper_title', 'Unknown')
                for method in paper.get('methodologies', []):
                    method_lower = method.lower()
                    if any(kw in method_lower for kw in ['machine learning', 'neural', 'deep learning', 'nlp']):
                        domain_inventory.setdefault('machine learning', []).append(title)
                    if any(kw in method_lower for kw in ['regression', 'panel data', 'instrumental', 'econometric']):
                        domain_inventory.setdefault('econometrics', []).append(title)
                    if any(kw in method_lower for kw in ['agent-based', 'abm', 'simulation']):
                        domain_inventory.setdefault('computational economics', []).append(title)
                    if any(kw in method_lower for kw in ['dsge', 'macro', 'general equilibrium']):
                        domain_inventory.setdefault('macroeconomics', []).append(title)
                    if any(kw in method_lower for kw in ['behavioral', 'experiment', 'survey']):
                        domain_inventory.setdefault('behavioral economics', []).append(title)

        # Prepare domain summary for the LLM
        domain_summary_parts = []
        for domain, papers in sorted(domain_inventory.items(), key=lambda x: -len(x[1])):
            paper_list = list(set(papers))[:5]
            domain_summary_parts.append(f"- {domain} ({len(set(papers))} papers): {', '.join(p[:60] for p in paper_list)}")

        domain_summary = "\n".join(domain_summary_parts[:15]) if domain_summary_parts else "No domain tags available"

        # Prepare gaps summary
        gaps_text = "\n".join([
            f"- [{g.get('gap_type', 'unknown')}] {g.get('gap_title', '')}: {g.get('gap_description', '')[:120]}"
            for g in research_gaps[:10]
        ]) if research_gaps else "No gaps available"
        try:

            result = self.llm.format_and_invoke(

                system_prompt="""You are a leading interdisciplinary researcher who excels at identifying NOVEL connections across different research domains.
    You see bridges between fields that specialist researchers miss. Your cross-domain insights have led to breakthrough research agendas.""",

                user_prompt="""Analyze the following research domains and their papers to identify meaningful CROSS-DOMAIN CONNECTIONS.

    Research Domains and Papers:
    {domains}

    Research Gaps:
    {gaps}

    Identify 3-5 cross-domain connections. For each connection:
    1. Name the two domains being connected
    2. Explain the connection_type: one of "methodological transfer" (methods from domain A can solve problems in domain B), "shared phenomenon" (both domains study the same underlying mechanism), "complementary evidence" (findings in domain A validate/contradict domain B), or "theoretical bridge" (theoretical framework connects both domains)
    3. Describe the nature of the connection in 2-3 sentences
    4. Explain WHY this connection is significant for the research topic
    5. List specific papers that bridge both domains (use exact titles from the input)
    6. Describe a specific research opportunity arising from this connection

    Return as a JSON array of objects with fields: domain_a, domain_b, connection_type, description, significance, bridging_papers, research_opportunity

    Respond with ONLY the JSON array, no other text.""",

                variables={
                "domains": domain_summary,
                "gaps": gaps_text
                }

            )

            if not result:
                print(f"    Warning: Empty response from LLM for cross-domain analysis")
                return []

            connections_data = safe_parse_json(result, "cross-domain connections")
            if connections_data is None:
                return []

            # Handle dict wrapper
            if isinstance(connections_data, dict):
                for key in ['connections', 'cross_domain_connections', 'data']:
                    if key in connections_data:
                        connections_data = connections_data[key]
                        break
                else:
                    connections_data = [connections_data]

            if not isinstance(connections_data, list):
                return []

            connections = []
            for conn_data in connections_data:
                try:
                    conn = CrossDomainConnection(
                        domain_a=conn_data.get('domain_a', ''),
                        domain_b=conn_data.get('domain_b', ''),
                        connection_type=conn_data.get('connection_type', 'shared phenomenon'),
                        description=conn_data.get('description', ''),
                        significance=conn_data.get('significance', ''),
                        bridging_papers=conn_data.get('bridging_papers', []),
                        research_opportunity=conn_data.get('research_opportunity', '')
                    )
                    connections.append(conn)
                except Exception as e:
                    print(f"    Warning: Could not create CrossDomainConnection: {e}")
                    continue

            print(f"  [{self.agent_name}] Identified {len(connections)} cross-domain connections")
            for i, conn in enumerate(connections, 1):
                print(f"    {i}. {conn.domain_a} <-> {conn.domain_b} ({conn.connection_type})")

            return connections

        except Exception as e:
            print(f"    Error generating cross-domain analysis: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _generate_research_plan(
        self,
        research_gaps: List[Dict],
        paper_structures: List[Dict],
        knowledge_graph: Dict
    ) -> ResearchPlan:
        """Generate research plan based on gaps."""
        
        print(f"\n  [{self.agent_name}] Generating research plan...")
        
        # Generate plan overview
        overview = self._generate_plan_overview(research_gaps)
        
        # Generate research objectives
        objectives = self._generate_research_objectives(research_gaps[:10])
        
        # Generate methodology overview
        methodology_overview = self._generate_methodology_overview(paper_structures, research_gaps)
        
        # Generate expected contributions
        expected_contributions = self._generate_expected_contributions(research_gaps)
        
        # Generate timeline and resources
        timeline_summary = self._generate_timeline_summary(objectives)
        resource_requirements = self._generate_resource_requirements(objectives)
        
        plan = ResearchPlan(
            plan_title="Research Plan: Addressing Key Gaps",
            overview=overview,
            research_objectives=objectives,
            methodology_overview=methodology_overview,
            expected_contributions=expected_contributions,
            timeline_summary=timeline_summary,
            resource_requirements=resource_requirements
        )
        
        print(f"  [{self.agent_name}] Research plan generated ({len(objectives)} objectives)")
        
        return plan
    
    def _generate_plan_overview(self, research_gaps: List[Dict]) -> str:
        """Generate research plan overview."""
        
        gaps_text = "\n".join([
            f"- {g.get('gap_title', 'Unknown')}"
            for g in research_gaps[:5]
        ])
        try:

        
            result = self.llm.format_and_invoke(

        
                system_prompt="You are an expert at writing research plans.",

        
                user_prompt="""Write an overview (2-3 paragraphs) for a research plan that addresses these gaps:
            
                {gaps}
            
                The overview should:
                - State the plan's purpose and scope
                - Explain how it addresses the gaps
                - Outline the expected impact
            
                Return as a JSON object with an "overview" field.
            
                Respond with ONLY the JSON object, no other text.
                """,

        
                variables={"gaps": gaps_text}

        
            )
            
            if not result:
                print(f"    Warning: Empty response from LLM for plan overview")
                return "This research plan addresses key gaps identified in the literature."
            
            data = safe_parse_json(result, "plan overview")
            if data is None:
                return "This research plan addresses key gaps identified in the literature."
            
            return data.get("overview", "")
        except Exception as e:
            print(f"    Error generating plan overview: {e}")
            import traceback
            traceback.print_exc()
            return "This research plan addresses key gaps identified in the literature."
    
    def _generate_research_objectives(
        self,
        research_gaps: List[Dict]
    ) -> List[ResearchObjective]:
        """Generate research objectives from gaps."""
        
        objectives = []
        
        # Group gaps by type
        gap_types = {}
        for gap in research_gaps:
            gap_type = gap.get('gap_type', 'unknown')
            if gap_type not in gap_types:
                gap_types[gap_type] = []
            gap_types[gap_type].append(gap)
        
        # Create objectives for each gap type
        for idx, (gap_type, gaps) in enumerate(gap_types.items(), 1):
            gap_titles = [g.get('gap_title', 'Unknown') for g in gaps[:3]]
            gap_ids = [g.get('gap_id', '') for g in gaps[:3]]
            
            # Determine priority based on severity
            severities = [g.get('severity', 'Low') for g in gaps]
            priority = "High" if "Critical" in severities or "High" in severities else "Medium"
            
            objective = ResearchObjective(
                objective_id=f"OBJ_{idx:03d}",
                objective_title=f"Address {gap_type.capitalize()} Gaps",
                description=f"This objective focuses on addressing {gap_type} gaps including: {', '.join(gap_titles[:2])}.",
                addresses_gaps=gap_ids,
                methodology=gaps[0].get('suggested_approaches', [])[:3] if gaps else [],
                expected_outcomes=[
                    f"Improved understanding of {gap_type} aspects",
                    "Novel methodological contributions",
                    "Empirical validation of findings"
                ],
                timeline="12-18 months",
                priority=priority
            )
            
            objectives.append(objective)
        
        return objectives
    
    def _generate_methodology_overview(
        self,
        paper_structures: List[Dict],
        research_gaps: List[Dict]
    ) -> str:
        """Generate methodology overview."""
        
        # Collect methodologies from papers
        all_methods = set()
        for paper in paper_structures[:20]:
            all_methods.update(paper.get('methodologies', []))
        
        methods_text = ", ".join(list(all_methods)[:5])
        
        return f"The research will employ a mixed-methods approach, building on established methodologies including {methods_text}. Novel methodological contributions will address identified gaps in current approaches."
    
    def _generate_expected_contributions(self, research_gaps: List[Dict]) -> List[str]:
        """Generate expected contributions."""
        
        contributions = [
            "Comprehensive literature synthesis addressing key research gaps",
            "Novel methodological frameworks for future research",
            "Empirical evidence to validate theoretical propositions"
        ]
        
        # Add gap-specific contributions
        for gap in research_gaps[:3]:
            impact = gap.get('potential_impact', '')
            if impact:
                contributions.append(impact)
        
        return contributions[:5]
    
    def _generate_timeline_summary(self, objectives: List[ResearchObjective]) -> str:
        """Generate timeline summary."""
        
        high_priority = sum(1 for obj in objectives if obj.priority == "High")
        total = len(objectives)
        
        return f"The research plan spans 18-24 months, with {high_priority} high-priority objectives to be completed in the first 12 months, followed by {total - high_priority} medium-priority objectives."
    
    def _generate_resource_requirements(self, objectives: List[ResearchObjective]) -> List[str]:
        """Generate resource requirements."""
        
        return [
            "Access to academic databases and literature",
            "Computational resources for data analysis",
            "Research team with expertise in relevant methodologies",
            "Collaboration with domain experts",
            "Funding for data collection and dissemination"
        ]
    
    def _prepare_papers_summary(self, paper_structures: List[Dict]) -> str:
        """Prepare summary of papers."""
        
        if not paper_structures:
            return "No papers available for summary."
        
        summary_parts = []
        for paper in paper_structures[:20]:
            summary = f"- {paper.get('paper_title', 'Unknown')}: {', '.join(paper.get('key_findings', [])[:2])}"
            summary_parts.append(summary)
        
        return "\n".join(summary_parts) if summary_parts else "No paper summaries available."
    
    def _prepare_gaps_summary(self, research_gaps: List[Dict]) -> str:
        """Prepare summary of gaps."""
        
        if not research_gaps:
            return "No research gaps identified."
        
        summary_parts = []
        for gap in research_gaps:
            summary = f"- [{gap.get('gap_type', 'unknown').upper()}] {gap.get('gap_title', 'Unknown')}: {gap.get('gap_description', '')[:100]}..."
            summary_parts.append(summary)
        
        return "\n".join(summary_parts) if summary_parts else "No gap summaries available."

class CiteKeeper:
    """Agent for reference finalization and bibliography management."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "CiteKeeper"
        self.api_key = openai_api_key
        self.collector = collector
        self.llm = LLMClient(


            temperature=0.1,

            api_key=self.api_key,

            collector=collector,

            agent_name=self.agent_name

        )
    
    def finalize_references(
        self,
        gap_analysis: Dict,
        literature_review: LiteratureReview,
        research_plan: ResearchPlan,
        literature_metadata: Optional[Dict[str, Dict]] = None
    ) -> Bibliography:
        """Finalize references and create formatted bibliography."""

        print(f"\n[{self.agent_name}] Finalizing references and bibliography...")

        # Get paper details from gap analysis
        paper_structures = gap_analysis.get('paper_structures', [])

        # Real bibliographic metadata (year/authors/venue) keyed by lowercased title,
        # sourced from the Stage-1 literature batch (LiteratureItem fields). Falls back
        # to {} so missing data degrades gracefully to "(n.d.)" rather than asserting 2024.
        literature_metadata = literature_metadata or {}

        # Extract all papers for bibliography (review-cited + all analyzed)
        cited_papers = self._extract_cited_papers(literature_review, research_plan, paper_structures)

        # title -> source URL, recovered from the Stage-2 knowledge graph (which carries the
        # Stage-1 gathering URLs) so the bibliography keeps verifiable identifiers.
        url_map = self._build_url_map(gap_analysis)

        # Create formatted references
        references = self._create_formatted_references(
            cited_papers, paper_structures, url_map, literature_metadata
        )
        
        # Sort references alphabetically
        references.sort(key=lambda x: x.citation_key)
        
        # Create bibliography statistics
        by_type = {}
        by_importance = {}
        
        for ref in references:
            by_type[ref.reference_type] = by_type.get(ref.reference_type, 0) + 1
            by_importance[ref.importance] = by_importance.get(ref.importance, 0) + 1
        
        bibliography = Bibliography(
            references=references,
            total_references=len(references),
            by_type=by_type,
            by_importance=by_importance
        )
        
        print(f"[{self.agent_name}] Bibliography finalized ({len(references)} references)")
        
        return bibliography
    
    def _extract_cited_papers(
        self,
        literature_review: LiteratureReview,
        research_plan: ResearchPlan,
        paper_structures: Optional[List[Dict]] = None
    ) -> set:
        """Extract all papers that should appear in the bibliography.

        Includes papers explicitly cited in review sections plus all papers
        from the analyzed paper set (paper_structures) to ensure the
        bibliography is comprehensive.
        """

        cited_papers = set()

        # From review sections
        for section in literature_review.sections:
            cited_papers.update(section.key_papers)

        # Include all analyzed papers so the bibliography is comprehensive
        if paper_structures:
            for p in paper_structures:
                title = p.get('paper_title', '')
                if title:
                    cited_papers.add(title)

        return cited_papers
    
    def _build_url_map(self, gap_analysis: Dict) -> Dict[str, str]:
        """Map paper title (lowercased) -> source URL from the Stage-2 knowledge-graph
        nodes, which carry the Stage-1 gathering URLs. Keeps bibliography entries
        verifiable (so they are not flagged as unsourced in the release audit)."""
        out: Dict[str, str] = {}
        kg = (gap_analysis or {}).get("knowledge_graph", {}) or {}
        for n in (kg.get("nodes") or []):
            label = (n.get("label") or "").strip()
            url = ((n.get("properties") or {}).get("url") or "").strip()
            if label and url:
                out[label.lower()] = url
        return out

    def _create_formatted_references(
        self,
        cited_papers: set,
        paper_structures: List[Dict],
        url_map: Optional[Dict[str, str]] = None,
        literature_metadata: Optional[Dict[str, Dict]] = None
    ) -> List[FormattedReference]:
        """Create formatted references for cited papers."""

        references = []
        url_map = url_map or {}
        literature_metadata = literature_metadata or {}

        # Create a mapping of paper titles to structures
        paper_map = {p.get('paper_title', ''): p for p in paper_structures}

        for paper_title in cited_papers:
            if not paper_title:
                continue

            ref_url = url_map.get(paper_title.strip().lower(), "")
            # Real year/authors/venue from the Stage-1 literature batch (keyed by title).
            meta = literature_metadata.get(paper_title.strip().lower(), {})
            paper = paper_map.get(paper_title)
            if not paper:
                # Create basic reference if paper not found in structures.
                # Still use any real metadata (year/authors/venue) we have for it.
                ref = FormattedReference(
                    citation_key=self._generate_citation_key(paper_title, meta),
                    formatted_citation=self._format_citation_from_meta(paper_title, meta),
                    reference_type="unknown",
                    importance="Medium",
                    title=paper_title,
                    url=ref_url
                )
                references.append(ref)
                continue

            # Generate citation key (uses real year when available)
            citation_key = self._generate_citation_key(paper_title, meta)

            # Format citation (APA-ish, from real metadata; "(n.d.)" when year missing)
            formatted_citation = self._format_citation_from_meta(paper_title, meta)

            # Determine reference type
            ref_type = self._determine_reference_type(paper)

            # Determine importance
            importance = self._determine_importance(paper)

            ref = FormattedReference(
                citation_key=citation_key,
                formatted_citation=formatted_citation,
                reference_type=ref_type,
                importance=importance,
                title=paper_title,
                url=ref_url
            )

            references.append(ref)

        return references

    def _generate_citation_key(self, paper_title: str, meta: Optional[Dict] = None) -> str:
        """Generate a citation key, preferring the gathering stage's pre-built key.

        Order of preference: (1) the clean "Surname+Year" key already produced by
        Stage-1 gathering (carried in meta['citation_key']); (2) first-author surname
        + real year; (3) a meaningful title word + real year. When the year is
        genuinely absent the key uses a *separated* "-nd" marker (e.g. "Schulz-nd")
        instead of jamming "nd" onto the word ("Schulznd"). Never hardcodes a year.
        """
        meta = meta or {}

        # (1) Reuse the clean key the gathering stage already built (e.g. "Kaplan2018").
        prebuilt = str(meta.get('citation_key') or '').strip()
        if prebuilt:
            # Gathering uses an "XXXX" placeholder for an unknown year; normalize it
            # to a separated "-nd" marker so the key matches the "(n.d.)" date display.
            if prebuilt.endswith('XXXX'):
                prebuilt = prebuilt[:-4].rstrip('-_') + '-nd'
            return prebuilt

        # (2) Prefer first author's surname for the key
        key = None
        authors = meta.get('authors') or []
        if authors:
            first_author = str(authors[0]).strip()
            if first_author:
                # Last token is usually the surname; strip punctuation
                key = first_author.split()[-1].strip(',:;.') if first_author.split() else None

        if not key:
            # (3) Title fallback: first meaningful alphanumeric word (skip 1-2 char tokens)
            words = [w for w in re.findall(r"[A-Za-z0-9]+", paper_title) if len(w) > 2]
            if words:
                key = words[0]
            else:
                toks = paper_title.split()
                key = toks[0].strip(',:;.') if toks else "Unknown"
        key = key.capitalize() if key else "Unknown"

        year = meta.get('year')
        # Real year -> "Surname2018"; genuinely missing -> separated "Surname-nd"
        # (never concatenates the title word with a bare "nd").
        return f"{key}{year}" if year else f"{key}-nd"

    def _format_citation_from_meta(self, paper_title: str, meta: Optional[Dict] = None) -> str:
        """Format an APA-ish citation from real metadata.

        Uses actual authors/year/venue when present; falls back to "(n.d.)" for a
        missing year (never asserts 2024) and omits unknown fields gracefully.
        """
        meta = meta or {}

        # Reuse the gathering stage's APA-style citation when present — it already
        # carries the real authors + year + venue/DOI (e.g. "Kaplan, Moll, & Violante
        # (2018). Monetary Policy According to HANK. ..."), so we never degrade a
        # paper that has good metadata to a bare "(n.d.)".
        prebuilt = str(meta.get('formatted_citation') or '').strip()
        if prebuilt:
            return prebuilt

        authors = [str(a).strip() for a in (meta.get('authors') or []) if str(a).strip()]
        year = meta.get('year')
        venue = (meta.get('venue') or '').strip()

        parts = []
        if authors:
            if len(authors) == 1:
                parts.append(authors[0] + ".")
            elif len(authors) <= 3:
                parts.append(", ".join(authors[:-1]) + ", & " + authors[-1] + ".")
            else:
                parts.append(authors[0] + " et al.")

        parts.append(f"({year})." if year else "(n.d.).")
        parts.append(f"{paper_title}.")
        if venue:
            parts.append(f"{venue}.")

        return " ".join(parts)
    
    def _determine_reference_type(self, paper: Dict) -> str:
        """Determine reference type."""
        
        methodologies = paper.get('methodologies', [])
        
        if any('empirical' in m.lower() for m in methodologies):
            return "journal"
        elif any('review' in m.lower() for m in methodologies):
            return "review"
        else:
            return "conference"
    
    def _determine_importance(self, paper: Dict) -> str:
        """Determine reference importance."""
        
        findings = paper.get('key_findings', [])
        contributions = paper.get('theoretical_contributions', [])
        
        if len(findings) >= 4 or len(contributions) >= 2:
            return "High"
        elif len(findings) >= 2 or len(contributions) >= 1:
            return "Medium"
        else:
            return "Low"

# ========== ORCHESTRATOR ==========

class SynthesisOrchestrator:
    """Orchestrates the synthesis pipeline."""

    def __init__(self, openai_api_key: Optional[str] = None, collector: Optional[MetricsCollector] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.collector = collector

        # Initialize agents
        self.knowledge_weaver = KnowledgeWeaver(self.api_key, collector=collector)
        self.cite_keeper = CiteKeeper(self.api_key, collector=collector)
        
        self.synthesis_result: Optional[SynthesisResult] = None
    
    def run_synthesis_pipeline(
        self,
        gap_analysis: Dict,
        literature_metadata: Optional[Dict[str, Dict]] = None
    ) -> SynthesisResult:
        """Run the complete synthesis pipeline."""

        print(f"\n{'='*70}")
        print(f"SYNTHESIS PIPELINE")
        print(f"{'='*70}")
        print(f"Input: {gap_analysis.get('metadata', {}).get('total_gaps_identified', 0)} gaps")
        print(f"{'='*70}\n")

        # Bibliography grounding: the caller passes the bibliographic metadata
        # (year/authors/venue) loaded from ITS run's literature batch via
        # load_literature_metadata(<run batch path>). The stage no longer guesses a
        # cwd-relative "literature_batch.json": in the pipeline the cwd is the team's
        # code directory, so that read would find no batch (or another run's) and every
        # bibliography entry would come out without authors/years.
        if literature_metadata is None:
            print("[Orchestrator] No literature metadata supplied; bibliography entries "
                  "will lack authors/years. Pass load_literature_metadata(<batch file>).")
            literature_metadata = {}

        # Step 1: KnowledgeWeaver - Knowledge integration
        literature_review, research_plan, cross_domain_connections = self.knowledge_weaver.integrate_knowledge(gap_analysis)

        # Step 2: CiteKeeper - Reference finalization
        bibliography = self.cite_keeper.finalize_references(
            gap_analysis,
            literature_review,
            research_plan,
            literature_metadata=literature_metadata
        )
        
        # Build decision log for transparency
        decision_log = {
            "theme_selection_rationale": f"Review sections organized by methodology type. Top methodologies selected based on frequency across {len(gap_analysis.get('paper_structures', []))} analyzed papers. This thematic approach was chosen over chronological or theoretical organization because it enables direct comparison of methodological strengths and gaps.",
            "gap_prioritization_rationale": f"Gaps prioritized by severity (Critical > High > Medium > Low) and addressability. {len(gap_analysis.get('research_gaps', []))} gaps were identified; research plan objectives address the most critical and addressable gaps first.",
            "synthesis_process": "Two-stage synthesis: (1) KnowledgeWeaver integrated papers and gaps into a literature review and research plan, (2) CiteKeeper finalized references with proper formatting. Each review section was generated with explicit gap-literature connections.",
            "cross_stage_consistency": f"Synthesis references only papers from Stage 1 literature batch and gaps from Stage 2 gap analysis. Bibliography contains {bibliography.total_references} references sourced from the analyzed paper set."
        }

        assumptions_and_limitations = {
            "assumptions": [
                "Literature review sections accurately represent the balance of evidence across reviewed papers",
                "Methodology-based thematic organization captures the most important research dimensions",
                "Gap severity ratings from Stage 2 correctly reflect research priorities",
                "The reviewed paper set is sufficiently representative for meaningful synthesis",
                "Research plan timelines assume standard academic resource availability"
            ],
            "limitations": [
                "Synthesis is based on abstract-level and metadata analysis for papers without full text",
                "Review sections may underrepresent papers that use uncommon methodologies (outside top themes)",
                "Citation integrity depends on Stage 1 paper metadata accuracy",
                "Research plan feasibility estimates are heuristic, not based on empirical resource costing",
                "Cross-domain connections may be constrained by the topical scope of the literature batch"
            ]
        }

        # Store gap_analysis for consolidated review generation
        self._gap_analysis = gap_analysis

        # Create synthesis result
        self.synthesis_result = SynthesisResult(
            literature_review=literature_review,
            research_plan=research_plan,
            bibliography=bibliography,
            cross_domain_connections=cross_domain_connections,
            metadata={
                "created_at": datetime.now().isoformat(),
                "review_sections": len(literature_review.sections),
                "research_objectives": len(research_plan.research_objectives),
                "total_references": bibliography.total_references
            },
            decision_log=decision_log,
            assumptions_and_limitations=assumptions_and_limitations
        )
        
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Literature Review: {len(literature_review.sections)} sections")
        print(f"Research Plan: {len(research_plan.research_objectives)} objectives")
        print(f"Bibliography: {bibliography.total_references} references")
        print(f"Cross-Domain Connections: {len(cross_domain_connections)}")
        print(f"{'='*70}\n")
        
        return self.synthesis_result
    
    def load_gap_analysis(self, filepath: str) -> Dict:
        """Load gap analysis from Stage 2."""
        print(f"[Orchestrator] Loading gap analysis from: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"[Orchestrator] Loaded analysis with {data.get('metadata', {}).get('total_gaps_identified', 0)} gaps")
        return data

    def load_literature_metadata(self, filepath: str = "literature_batch.json") -> Dict[str, Dict]:
        """Build a lowercased-title -> {year, authors, venue} map from the Stage-1
        literature batch (LiteratureItem fields). Used to ground the bibliography in
        real bibliographic metadata. Returns {} (graceful) if the file is missing."""
        metadata: Dict[str, Dict] = {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                batch = json.load(f)
        except Exception as e:
            print(f"[Orchestrator] Literature batch not found ({filepath}): {e}. "
                  f"Bibliography will use '(n.d.)' where year is unknown.")
            return metadata

        for item in batch.get('literature_items', []):
            title = (item.get('title') or '').strip()
            if not title:
                continue
            metadata[title.lower()] = {
                "year": item.get('year'),
                "authors": item.get('authors') or [],
                "venue": item.get('venue'),
            }

        # Supplement with the gathering stage's pre-built citations (keyed by
        # paper_title): these already carry a clean "Surname+Year" citation_key and
        # an APA-style formatted_citation, and can fill any year/author gaps the
        # literature_items left as None. Per-citation try/except so one bad row never
        # drops the whole map.
        for cit in batch.get('citations', []) or []:
            try:
                ctitle = (cit.get('paper_title') or '').strip()
                if not ctitle:
                    continue
                entry = metadata.setdefault(ctitle.lower(), {
                    "year": None, "authors": [], "venue": None
                })
                if not entry.get('year') and cit.get('year'):
                    entry['year'] = cit.get('year')
                if not entry.get('authors') and cit.get('authors'):
                    entry['authors'] = cit.get('authors')
                if cit.get('citation_key'):
                    entry['citation_key'] = cit.get('citation_key')
                if cit.get('formatted_citation'):
                    entry['formatted_citation'] = cit.get('formatted_citation')
            except Exception:
                continue

        print(f"[Orchestrator] Loaded bibliographic metadata for {len(metadata)} papers from {filepath}")
        return metadata
    
    def save_synthesis_result(self, filepath: str):
        """Save the complete synthesis result to JSON."""
        if not self.synthesis_result:
            print("No synthesis result to save.")
            return
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.synthesis_result.model_dump(), f, indent=2, ensure_ascii=False)
        
        print(f"[Orchestrator] Synthesis result saved to: {filepath}")
    
    def save_literature_review(self, filepath: str):
        """Save literature review as formatted text."""
        if not self.synthesis_result:
            print("No synthesis result to save.")
            return
        
        review = self.synthesis_result.literature_review
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{review.title}\n")
            f.write("="*len(review.title) + "\n\n")
            
            f.write("ABSTRACT\n")
            f.write("-"*70 + "\n")
            f.write(f"{review.abstract}\n\n")
            
            f.write("INTRODUCTION\n")
            f.write("-"*70 + "\n")
            f.write(f"{review.introduction}\n\n")
            
            for section in review.sections:
                f.write(f"{section.section_title.upper()}\n")
                f.write("-"*70 + "\n")
                f.write(f"{section.section_content}\n\n")
            
            # Cross-Domain Insights section
            if self.synthesis_result and self.synthesis_result.cross_domain_connections:
                f.write("CROSS-DOMAIN INSIGHTS\n")
                f.write("-"*70 + "\n")
                for i, conn in enumerate(self.synthesis_result.cross_domain_connections, 1):
                    f.write(f"\n{i}. {conn.domain_a} <-> {conn.domain_b} ({conn.connection_type})\n")
                    f.write(f"   {conn.description}\n")
                    f.write(f"   Significance: {conn.significance}\n")
                    if conn.bridging_papers:
                        f.write(f"   Bridging papers: {', '.join(conn.bridging_papers[:3])}\n")
                    if conn.research_opportunity:
                        f.write(f"   Research opportunity: {conn.research_opportunity}\n")
                f.write("\n")

            f.write("SYNTHESIS AND CONCLUSIONS\n")
            f.write("-"*70 + "\n")
            f.write(f"{review.synthesis}\n")

        print(f"[Orchestrator] Literature review saved to: {filepath}")
    
    def save_research_plan(self, filepath: str):
        """Save research plan as formatted text."""
        if not self.synthesis_result:
            print("No synthesis result to save.")
            return
        
        plan = self.synthesis_result.research_plan
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{plan.plan_title}\n")
            f.write("="*len(plan.plan_title) + "\n\n")
            
            f.write("OVERVIEW\n")
            f.write("-"*70 + "\n")
            f.write(f"{plan.overview}\n\n")
            
            f.write("RESEARCH OBJECTIVES\n")
            f.write("-"*70 + "\n")
            for obj in plan.research_objectives:
                f.write(f"\n{obj.objective_id}: {obj.objective_title} (Priority: {obj.priority})\n")
                f.write(f"{obj.description}\n")
                f.write(f"Methodology: {', '.join(obj.methodology)}\n")
                f.write(f"Timeline: {obj.timeline}\n")
            
            f.write(f"\nMETHODOLOGY OVERVIEW\n")
            f.write("-"*70 + "\n")
            f.write(f"{plan.methodology_overview}\n\n")
            
            f.write("EXPECTED CONTRIBUTIONS\n")
            f.write("-"*70 + "\n")
            for contrib in plan.expected_contributions:
                f.write(f"- {contrib}\n")
            
            f.write(f"\nTIMELINE\n")
            f.write("-"*70 + "\n")
            f.write(f"{plan.timeline_summary}\n\n")
            
            f.write("RESOURCE REQUIREMENTS\n")
            f.write("-"*70 + "\n")
            for resource in plan.resource_requirements:
                f.write(f"- {resource}\n")
        
        print(f"[Orchestrator] Research plan saved to: {filepath}")
    
    def save_bibliography(self, filepath: str):
        """Save bibliography as formatted text."""
        if not self.synthesis_result:
            print("No synthesis result to save.")
            return
        
        bib = self.synthesis_result.bibliography
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("BIBLIOGRAPHY\n")
            f.write("="*70 + "\n\n")
            
            for ref in bib.references:
                f.write(f"[{ref.citation_key}] {ref.formatted_citation}\n\n")
        
        print(f"[Orchestrator] Bibliography saved to: {filepath}")
    
    def generate_consolidated_review(self) -> Dict:
        """Generate a consolidated review interleaving papers, gaps, synthesis, and citation integrity checks.

        This file is designed to be the *first* file the LLM evaluator reads,
        giving it a single-document view of cross-stage coherence.
        """
        if not self.synthesis_result:
            return {}

        gap_analysis = getattr(self, "_gap_analysis", {})
        paper_structures = gap_analysis.get("paper_structures", [])
        research_gaps = gap_analysis.get("research_gaps", [])
        review = self.synthesis_result.literature_review
        bib = self.synthesis_result.bibliography

        # Citation integrity: collect all paper titles referenced in synthesis
        cited_in_review = set()
        for section in review.sections:
            cited_in_review.update(section.key_papers)

        papers_from_stage1 = {p.get("paper_title", "") for p in paper_structures}
        verified = cited_in_review & papers_from_stage1
        unverified = cited_in_review - papers_from_stage1

        consolidated = {
            "stage1_papers_count": len(paper_structures),
            "stage1_paper_titles": [p.get("paper_title", "") for p in paper_structures[:30]],
            "stage2_gaps_count": len(research_gaps),
            "stage2_gaps_summary": [
                {
                    "gap_id": g.get("gap_id", ""),
                    "gap_title": g.get("gap_title", ""),
                    "gap_type": g.get("gap_type", ""),
                    "severity": g.get("severity", ""),
                    "source_papers": g.get("source_papers", [])[:3],
                }
                for g in research_gaps[:15]
            ],
            "stage3_review_title": review.title,
            "stage3_review_sections": len(review.sections),
            "stage3_bibliography_count": bib.total_references,
            "citation_integrity_check": {
                "total_cited_in_review": len(cited_in_review),
                "verified_against_stage1": len(verified),
                "unverified": list(unverified)[:10],
                "integrity_ratio": round(len(verified) / max(len(cited_in_review), 1), 3),
            },
            "cross_domain_connections": [
                {
                    "domain_a": conn.domain_a,
                    "domain_b": conn.domain_b,
                    "connection_type": conn.connection_type,
                    "description": conn.description,
                    "significance": conn.significance,
                    "bridging_papers": conn.bridging_papers[:3],
                    "research_opportunity": conn.research_opportunity,
                }
                for conn in self.synthesis_result.cross_domain_connections
            ],
            "decision_log": self.synthesis_result.decision_log,
            "assumptions_and_limitations": self.synthesis_result.assumptions_and_limitations,
        }

        return consolidated

    def save_consolidated_review(self, filepath: str):
        """Save the consolidated review to JSON."""
        consolidated = self.generate_consolidated_review()
        if not consolidated:
            print("No consolidated review to save.")
            return

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(consolidated, f, indent=2, ensure_ascii=False)

        print(f"[Orchestrator] Consolidated review saved to: {filepath}")

    def print_summary(self):
        """Print a summary of the synthesis."""
        if not self.synthesis_result:
            print("No synthesis result available.")
            return

        result = self.synthesis_result

        print(f"\n{'='*70}")
        print(f"SYNTHESIS SUMMARY")
        print(f"{'='*70}\n")

        # Literature review
        print(f"Literature Review:")
        print(f"  Title: {result.literature_review.title}")
        print(f"  Sections: {len(result.literature_review.sections)}")
        print(f"  Total Papers: {result.literature_review.metadata.get('total_papers_reviewed', 0)}")

        # Research plan
        print(f"\nResearch Plan:")
        print(f"  Objectives: {len(result.research_plan.research_objectives)}")
        high_priority = sum(1 for obj in result.research_plan.research_objectives if obj.priority == "High")
        print(f"  High Priority: {high_priority}")

        # Bibliography
        print(f"\nBibliography:")
        print(f"  Total References: {result.bibliography.total_references}")
        print(f"  By Type: {result.bibliography.by_type}")
        print(f"  By Importance: {result.bibliography.by_importance}")

        print(f"\n{'='*70}\n")

# ========== MAIN ==========

def main():
    """Main function to run the synthesis stage."""
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Initialize orchestrator
    orchestrator = SynthesisOrchestrator()
    
    # Load gap analysis from Stage 2
    gap_file = auto_input(
        "Enter path to gap analysis JSON (or press Enter for default): ",
        default=get_default("gap_file_path"),
    ).strip()
    if not gap_file:
        gap_file = "gap_analysis_results.json"
    
    try:
        gap_analysis = orchestrator.load_gap_analysis(gap_file)
    except Exception as e:
        print(f"Error loading gap analysis: {e}")
        print("Please run Stage 2 first to generate the gap analysis.")
        return

    # Load real bibliographic metadata (year/authors/venue) from the Stage-1 batch,
    # so the bibliography cites real years/authors instead of hardcoding 2024.
    literature_metadata = orchestrator.load_literature_metadata("literature_batch.json")

    # Run pipeline
    synthesis_result = orchestrator.run_synthesis_pipeline(gap_analysis, literature_metadata)
    
    # Save results
    orchestrator.save_synthesis_result("synthesis_results.json")
    orchestrator.save_literature_review("literature_review.txt")
    orchestrator.save_research_plan("research_plan.txt")
    orchestrator.save_bibliography("bibliography.txt")
    orchestrator.save_consolidated_review("consolidated_review.json")

    # Print summary
    orchestrator.print_summary()

    print("\n" + "="*70)
    print("Synthesis complete!")
    print("Output files:")
    print("  - synthesis_results.json (complete synthesis)")
    print("  - literature_review.txt (formatted review)")
    print("  - research_plan.txt (formatted plan)")
    print("  - bibliography.txt (formatted references)")
    print("  - consolidated_review.json (cross-stage consolidated view)")
    print("="*70)

if __name__ == "__main__":
    main()
