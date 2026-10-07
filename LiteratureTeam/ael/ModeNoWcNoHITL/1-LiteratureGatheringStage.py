# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Literature Gathering Stage - Automated Mode (No Firecrawl, No HITL)
This script gathers and processes literature based on research questions from IdeationTeam.

Pipeline:
1. TopicCrawler: Retrieves literature based on research questions
2. TrendTracker: Monitors field trends and emerging patterns
3. CiteKeeper: Manages bibliography (receives TrendTracker results)
4. InsightSummarizer: Extracts key knowledge from literature

Input: Research questions (JSON from IdeationTeam Stage 3)
Output: Initial literature batch with metadata, trends, citations, and insights
"""

import os
import json
import re
import time
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import requests
import sys
from pathlib import Path as _Path

# Add agents dir to path for shared imports
_agents_dir = _Path(__file__).resolve().parent.parent.parent.parent
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))
from shared.llm import LLMClient
from shared.observability import MetricsCollector
from shared.tools.tool_registry import ToolRegistry
from shared.tools.scholarly_search import (
    DEFAULT_RETRIES, ProviderStats, clean_venue, crossref_headers, crossref_select,
    crossref_venue, crossref_year, dedupe_papers, fetch_json,
)
from shared.auto_input import auto_input, get_default
from LiteratureTeam.ael.schemas.stage_inputs import ResearchQuestionInput as ResearchQuestion
from LiteratureTeam.ael.schemas.stage_outputs import (
    LiteratureItem, TrendAnalysis, CitationEntry, KnowledgeInsight, LiteratureBatch,
)

# Load environment variables
load_dotenv()

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
    
    # Try to extract JSON from markdown code blocks
    json_patterns = [
        r'```json\s*(.*?)\s*```',  # ```json ... ```
        r'```\s*(.*?)\s*```',      # ``` ... ```
        r'\[(.*)\]',                # Array in brackets
        r'\{(.*)\}',                # Object in braces
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    # Try to find JSON array or object in the content
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
    
    print(f"    Warning: Could not parse {context} as JSON. Content preview: {content[:200]}...")
    return None

# ========== AGENTS ==========

class TopicCrawler:
    """Agent for literature retrieval based on research questions."""

    def __init__(self, openai_api_key: str, semantic_scholar_api_key: Optional[str] = None, collector: Optional[MetricsCollector] = None):
        self.agent_name = "TopicCrawler"
        self.api_key = openai_api_key
        self.collector = collector
        self.semantic_scholar_api_key = semantic_scholar_api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.provider_stats = ProviderStats()   # per-provider counts
        self.llm = LLMClient(
            temperature=0.3,
            api_key=self.api_key,
            collector=collector,
            agent_name=self.agent_name
        )

    def retrieve_literature(
        self,
        research_questions: List[ResearchQuestion],
        max_papers_per_question: int = 15
    ) -> List[LiteratureItem]:
        """Retrieve literature for each research question."""
        print(f"\n[{self.agent_name}] Retrieving literature for {len(research_questions)} research questions...")

        all_literature = []

        for rq in research_questions:
            print(f"\n  Processing: {rq.question[:80]}...")

            # Generate search queries from research question
            search_queries = self._generate_search_queries(rq)

            # Search multiple sources using top 3 queries for better coverage
            for query in search_queries[:3]:
                # Search Semantic Scholar
                papers = self._search_semantic_scholar(query, max_papers_per_question // 2)

                # Search arXiv
                _arx = self._search_arxiv(query, max_papers_per_question // 2)
                self.provider_stats.record("arXiv", len(_arx))
                papers.extend(_arx)

                # Search OpenAlex (open, keyed) — broad econ coverage WITH abstracts
                papers.extend(self._search_openalex(query, max_papers_per_question // 2))

                # Associate with research question
                for paper in papers:
                    paper.research_question = rq.question

                all_literature.extend(papers)
                print(f"    Found {len(papers)} papers for query: {query[:50]}...")

                # Add delay between queries: 1.0s with API key (1 RPS limit), 5.0s without
                time.sleep(1.0 if self.semantic_scholar_api_key else 2.0)

        # Deduplicate
        unique_literature = self._deduplicate(all_literature)
        print(f"\n[{self.agent_name}] Total unique papers before filtering: {len(unique_literature)}")

        # Abstract cascade: fill missing abstracts from the optional local cache (user CSV,
        # ABSTRACT_CACHE) then the optional Elsevier API, BEFORE the filter drops abstract-less
        # papers. No-op (keyless path) when neither is configured.
        unique_literature = self._fill_missing_abstracts(unique_literature)

        # Filter out papers with no meaningful abstract
        filtered_literature = [
            p for p in unique_literature
            if p.abstract and p.abstract.strip() not in ("", "No abstract available")
            and len(p.abstract.strip()) > 50
        ]
        print(f"[{self.agent_name}] Papers after abstract filter: {len(filtered_literature)}")

        # LLM-based relevance filtering
        if filtered_literature and research_questions:
            filtered_literature = self._filter_by_relevance(filtered_literature, research_questions)

        print(f"\n[{self.agent_name}] Total papers after relevance filtering: {len(filtered_literature)}")

        return filtered_literature

    def _search_openalex(self, query: str, max_results: int = 10) -> List[LiteratureItem]:
        """Search OpenAlex (open, keyed) — returns econ papers WITH abstracts.

        Failures are logged and counted per provider."""
        try:
            res = ToolRegistry.invoke("openalex_search",
                {"query": query, "max_results": max_results, "econ_only": True},
                collector=self.collector, agent=self.agent_name)
        except Exception as e:
            print(f"    [OpenAlex] search failed: {e}")
            self.provider_stats.record("OpenAlex", 0, error=str(e))
            return []
        if not res.success:
            print(f"    [OpenAlex] search failed: {res.error}")
            self.provider_stats.record("OpenAlex", 0, error=res.error or "request failed")
            return []
        items = []
        for p in (res.data or []):
            try:
                items.append(LiteratureItem(
                    title=p.get("title") or "Untitled",
                    authors=p.get("authors") or [],
                    abstract=p.get("abstract") or "No abstract available",
                    url=p.get("url") or "",
                    source="OpenAlex",
                    year=p.get("year"),
                    literature_type="academic",
                    citation_count=p.get("citation_count"),
                    venue=clean_venue(p.get("venue")),
                    metadata_source="OpenAlex API",
                ))
            except Exception as e:
                print(f"    [OpenAlex] skipped a malformed record: {e}")
        self.provider_stats.record("OpenAlex", len(items))
        return items

    @staticmethod
    def _extract_doi(url: str) -> str:
        """Pull a DOI out of a URL/identifier (for the abstract cascade)."""
        if not url:
            return ""
        u = url.strip()
        low = u.lower()
        if "doi.org/" in low:
            return u[low.index("doi.org/") + len("doi.org/"):].strip()
        if low.startswith("10."):
            return u
        return ""

    def _fill_missing_abstracts(self, papers: List[LiteratureItem]) -> List[LiteratureItem]:
        """Fill abstracts for papers lacking one via the optional cascade: local cache
        (user CSV, ABSTRACT_CACHE) -> Elsevier API (key+insttoken). No-op without those."""
        from shared.tools.abstract_cache import abstract_from_cache, abstract_coverage
        filled = 0
        for p in papers:
            ab = (p.abstract or "").strip()
            if ab and ab != "No abstract available" and len(ab) > 50:
                continue
            doi = self._extract_doi(p.url)
            if not doi:
                continue
            new_ab = abstract_from_cache(doi)
            if not new_ab:
                try:
                    r = ToolRegistry.invoke("elsevier_abstract", {"doi": doi},
                                            collector=self.collector, agent=self.agent_name)
                    new_ab = (r.data or {}).get("abstract", "") if r.success else ""
                except Exception:
                    new_ab = ""
            if new_ab:
                p.abstract = new_ab
                filled += 1
        if filled:
            print(f"[{self.agent_name}] Abstract cascade filled {filled} missing abstracts (cache/Elsevier)")
        # Abstract-coverage report + reminder — printed to console AND stored for output files.
        _kept = sum(1 for p in papers
                    if (p.abstract or "").strip() not in ("", "No abstract available")
                    and len((p.abstract or "").strip()) > 50)
        self._abstract_coverage = abstract_coverage(len(papers), _kept)
        print(f"[{self.agent_name}] Abstract coverage: {self._abstract_coverage['summary']}")
        if self._abstract_coverage['reminder']:
            print(f"[{self.agent_name}] {self._abstract_coverage['reminder']}")
        return papers

    def _generate_search_queries(self, research_question: ResearchQuestion) -> List[str]:
        """Generate search queries from a research question."""
        try:
            result = self.llm.format_and_invoke(
                system_prompt="""You are an expert at formulating academic search queries for economics and social science databases.
You specialize in crafting precise queries that retrieve HIGHLY RELEVANT papers from Semantic Scholar and arXiv.""",
                user_prompt="""Generate 5 effective search queries for academic databases based on this research question.

Research Question: {question}
Theoretical Framework: {framework}
Methodologies: {methodologies}

IMPORTANT GUIDELINES:
- Each query MUST include economics-specific terminology (e.g., "monetary policy", "fiscal", "macroeconomic", "agent-based", "DSGE", "heterogeneous agents")
- Include key theorists or seminal model names when relevant (e.g., "Smets-Wouters", "HANK model", "Bewley-Aiyagari")
- Use Boolean-style phrasing suitable for academic search (e.g., "agent-based model macroeconomic monetary policy")
- Make queries SPECIFIC, not generic. Do NOT use broad terms like "machine learning applications" or "AI in science"
- Vary queries: one broad survey-style, one methodology-specific, one theory-specific, one empirical, one recent-trends

BAD examples (too generic): "computational models", "policy analysis methods", "AI applications"
GOOD examples: "agent-based macroeconomic model monetary policy transmission", "heterogeneous agent New Keynesian HANK fiscal policy", "calibration DSGE Bayesian estimation"

Return queries as a JSON array of 5 strings.
Example: ["query 1", "query 2", "query 3", "query 4", "query 5"]

Respond with ONLY the JSON array, no other text.
""",
                variables={
                    "question": research_question.question,
                    "framework": research_question.theoretical_framework or "N/A",
                    "methodologies": ", ".join(research_question.methodology) if research_question.methodology else "N/A"
                }
            )

            queries = json.loads(result)
            if isinstance(queries, list) and len(queries) > 0:
                return queries[:5]
            return [research_question.question]
        except Exception as e:
            print(f"    Error generating queries: {e}")
            q = research_question.question
            return [q, f"economics {q}"]

    def _search_semantic_scholar(self, query: str, max_results: int = 10) -> List[LiteratureItem]:
        """Search Semantic Scholar API with retry and fallback."""
        results = []
        # Keyless Semantic Scholar is rate-limited to near-uselessness (429) and always
        # ends up in the Crossref fallback anyway. Skip the futile S2 call when unkeyed and
        # go straight to Crossref (which returns results). S2 is used only with a valid key.
        if not self.semantic_scholar_api_key:
            return self._search_crossref_fallback(query, max_results)
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,authors,abstract,url,year,citationCount,publicationTypes,paperId,venue"
        }

        # Add API key header if available
        headers = {}
        if self.semantic_scholar_api_key:
            headers["x-api-key"] = self.semantic_scholar_api_key

        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                _tool_result = ToolRegistry.invoke("web_fetch", {
                    "url": base_url,
                    "params": params,
                    "headers": headers,
                    "retries": DEFAULT_RETRIES,
                }, collector=self.collector, agent=self.agent_name)
                if not _tool_result.success:
                    raise Exception(_tool_result.error)
                data = _tool_result.data.get("data", {})
                status_code = _tool_result.data.get("status_code", 0)

                if status_code == 429:
                    raise requests.exceptions.HTTPError(response=type('R', (), {'status_code': 429})())

                for paper in data.get("data", []):
                    item = LiteratureItem(
                        title=paper.get("title") or "Untitled",
                        authors=[author.get("name", "") for author in paper.get("authors", [])],
                        abstract=paper.get("abstract") or "No abstract available",
                        url=paper.get("url") or "",
                        source="Semantic Scholar",
                        year=paper.get("year"),
                        literature_type="academic",
                        citation_count=paper.get("citationCount"),
                        paper_id=paper.get("paperId"),
                        venue=paper.get("venue"),
                        metadata_source="Semantic Scholar API",
                    )
                    results.append(item)

                break  # Success, exit retry loop

            except requests.exceptions.HTTPError as e:
                if getattr(getattr(e, 'response', None), 'status_code', None) == 429:
                    # Rate limited - wait and retry
                    wait_time = 2 * (attempt + 1)  # 2s, 4s with API key (1 RPS)
                    print(f"    Rate limited by Semantic Scholar, waiting {wait_time}s (attempt {attempt+1}/{max_retries+1})")
                    time.sleep(wait_time)
                else:
                    print(f"    Error searching Semantic Scholar (attempt {attempt+1}): {e}")
                    time.sleep(1.0)
            except Exception as e:
                is_rate_limit = "429" in str(e)
                if not is_rate_limit or attempt == max_retries:
                    print(f"    Error searching Semantic Scholar (attempt {attempt+1}): {e}")
                time.sleep(2 * (attempt + 1) if is_rate_limit else 2.0)

        # If still no results after retries, try a broader query
        if not results and " " in query:
            try:
                # Simplify query: take first 3-4 significant words
                words = [w for w in query.split() if len(w) > 3][:4]
                simplified_query = " ".join(words)
                print(f"    Retrying with simplified query: {simplified_query}")
                params["query"] = simplified_query
                _tool_result = ToolRegistry.invoke("web_fetch", {
                    "url": base_url,
                    "params": params,
                    "headers": headers,
                    "retries": DEFAULT_RETRIES,
                }, collector=self.collector, agent=self.agent_name)
                if _tool_result.success:
                    data = _tool_result.data.get("data", {})
                    for paper in data.get("data", []):
                        item = LiteratureItem(
                            title=paper.get("title") or "Untitled",
                            authors=[author.get("name", "") for author in paper.get("authors", [])],
                            abstract=paper.get("abstract") or "No abstract available",
                            url=paper.get("url") or "",
                            source="Semantic Scholar",
                            year=paper.get("year"),
                            literature_type="academic",
                            citation_count=paper.get("citationCount"),
                            paper_id=paper.get("paperId"),
                            venue=paper.get("venue"),
                            metadata_source="Semantic Scholar API",
                        )
                        results.append(item)
            except Exception as e:
                print(f"    Simplified query also failed: {e}")

        self.provider_stats.record(
            "Semantic Scholar", len(results),
            error=None if results else "no results (HTTP error, rate limit, or empty)")

        # Crossref fallback if Semantic Scholar returned no results
        if not results:
            results = self._search_crossref_fallback(query, max_results)

        return results

    def _search_crossref_fallback(self, query: str, max_results: int = 10) -> List[LiteratureItem]:
        """Crossref search (fallback when Semantic Scholar is unavailable).

        Requests every date field and the container title: year = first present of
        issued / published-print / published-online / posted; venue = container-title.
        429/5xx are retried with backoff; failures are logged and counted."""
        print(f"    Trying Crossref fallback for: {query[:50]}...")
        data = fetch_json(
            "https://api.crossref.org/works", provider="Crossref",
            params={"query": query, "rows": max_results, "select": crossref_select()},
            headers=crossref_headers(), stats=self.provider_stats,
            collector=self.collector, agent=self.agent_name,
        )
        if data is None:
            return []
        results = []
        for item_data in (data.get("message", {}) or {}).get("items", []) if isinstance(data, dict) else []:
            try:
                title = (item_data.get("title") or ["Untitled"])[0]
                authors = [
                    f"{a.get('given', '')} {a.get('family', '')}".strip()
                    for a in item_data.get("author", []) or []
                ]
                abstract_clean = re.sub(r"<[^>]+>", "", item_data.get("abstract") or "No abstract available")
                results.append(LiteratureItem(
                    title=title,
                    authors=[a for a in authors if a],
                    abstract=abstract_clean,
                    url=item_data.get("URL") or f"https://doi.org/{item_data.get('DOI', '')}",
                    source="Crossref",
                    year=crossref_year(item_data),
                    literature_type="academic",
                    citation_count=item_data.get("is-referenced-by-count"),
                    venue=crossref_venue(item_data),
                    metadata_source="Crossref API",
                ))
            except Exception as e:
                print(f"    [Crossref] skipped a malformed record: {e}")
        self.provider_stats.record("Crossref", len(results))
        print(f"    Crossref fallback returned {len(results)} papers")
        return results

    def _search_arxiv(self, query: str, max_results: int = 10) -> List[LiteratureItem]:
        """Search arXiv API with retry logic."""
        results = []

        # Add economics category filter for better relevance
        econ_query = query
        if "econ" not in query.lower() and "q-fin" not in query.lower():
            econ_query = f"({query}) AND (cat:econ.* OR cat:q-fin.* OR cat:cs.MA)"

        for attempt, search_query in enumerate([econ_query, query]):
            try:
                _tool_result = ToolRegistry.invoke("arxiv_search", {
                    "query": search_query,
                    "max_results": max_results,
                    "sort_by": "relevance",
                }, collector=self.collector, agent=self.agent_name)
                arxiv_results = _tool_result.data if _tool_result.success else []

                for paper in arxiv_results:
                    pub_str = paper.get("published", "")
                    try:
                        pub_year = int(pub_str[:4]) if pub_str and len(pub_str) >= 4 else None
                    except (ValueError, TypeError):
                        pub_year = None
                    item = LiteratureItem(
                        title=paper.get("title", "") or "Untitled",
                        authors=paper.get("authors", []),
                        abstract=paper.get("abstract", "") or "No abstract available",
                        url=paper.get("url", ""),
                        source="arXiv",
                        year=pub_year,
                        literature_type="preprint",
                        paper_id=paper.get("url", "") or None,
                        metadata_source="arXiv API",
                    )
                    results.append(item)

                if results:
                    break  # Got results, no need for fallback query

            except Exception as e:
                print(f"    Error searching arXiv (attempt {attempt+1}): {e}")
                time.sleep(2.0)

        return results
    
    def _deduplicate(self, papers: List[LiteratureItem]) -> List[LiteratureItem]:
        """Remove duplicates by normalized title + first author (so the same paper under
        two DOIs, e.g. two SSRN postings, is listed once)."""
        return dedupe_papers(papers)

    def _filter_by_relevance(
        self,
        papers: List[LiteratureItem],
        research_questions: List[ResearchQuestion],
        batch_size: int = 20
    ) -> List[LiteratureItem]:
        """Filter papers by relevance using LLM-based scoring."""
        print(f"\n  [{self.agent_name}] Filtering {len(papers)} papers for relevance...")

        # Build the research context string
        rq_text = "\n".join([f"- {rq.question}" for rq in research_questions])

        relevant_papers = []
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i+batch_size]
            paper_list = "\n".join([
                f"[{idx}] Title: {p.title}\n    Abstract: {p.abstract[:300]}"
                for idx, p in enumerate(batch)
            ])

            try:
                result = self.llm.format_and_invoke(
                    system_prompt="""You are an expert economics research librarian. Your job is to assess whether papers are RELEVANT to specific economics research questions. Include papers that provide useful background or methodology, not just direct matches.""",
                    user_prompt="""Rate each paper's relevance to the research questions below. Return the indices of papers that score 3, 4, or 5 (at least moderately relevant).

Research Questions:
{questions}

Papers:
{papers}

For each paper, consider:
- Is this paper about ECONOMICS or a closely related social science?
- Does it address the research questions, their methodology, or useful background?
- Would an economics researcher find this useful for the stated topic?

Scoring:
- 5: Directly addresses the research question
- 4: Strongly related, provides important context or methods
- 3: Moderately related, useful background
- 2: Tangentially related
- 1: Not relevant

Return a JSON array of indices (0-based) for papers rated 3-5.
Example: [0, 2, 5]

If NO papers are relevant, return an empty array: []

Respond with ONLY the JSON array, no other text.""",
                    variables={
                        "questions": rq_text,
                        "papers": paper_list
                    }
                )

                indices = safe_parse_json(result, "relevance filter")
                if isinstance(indices, list):
                    for idx in indices:
                        if isinstance(idx, int) and 0 <= idx < len(batch):
                            batch[idx].relevance_score = 1.0
                            relevant_papers.append(batch[idx])
                else:
                    # If parsing fails, keep all papers in batch (conservative)
                    relevant_papers.extend(batch)
            except Exception as e:
                print(f"    Warning: Relevance filtering failed for batch: {e}")
                # On failure, keep all papers (conservative fallback)
                relevant_papers.extend(batch)

        print(f"  [{self.agent_name}] Kept {len(relevant_papers)} relevant papers out of {len(papers)}")

        # If filtering was too aggressive (kept less than 30%), fall back to all papers
        if len(relevant_papers) < max(3, len(papers) * 0.3):
            if filtered_papers:
                # Keep the RELEVANT subset even if small — a thin CLEAN corpus beats a
                # polluted one (falling back to ALL re-admitted off-topic papers that
                # cascaded into gaps/knowledge-graph/synthesis).
                print(f"    [relevance] kept only {len(filtered_papers)}/{len(papers)}; keeping the filtered subset (NOT all)")
            else:
                print(f"    [relevance] filter kept 0 papers; keeping all {len(papers)} as last resort")
                return papers

        return relevant_papers

class TrendTracker:
    """Agent for field monitoring and trend analysis."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "TrendTracker"
        self.api_key = openai_api_key
        self.collector = collector
        self.llm = LLMClient(
            temperature=0.5,
            api_key=self.api_key,
            collector=collector,
            agent_name=self.agent_name
        )
    
    def analyze_trends(
        self,
        literature: List[LiteratureItem],
        research_questions: List[ResearchQuestion]
    ) -> List[TrendAnalysis]:
        """Analyze trends in the literature."""
        print(f"\n[{self.agent_name}] Analyzing trends in {len(literature)} papers...")
        
        # Prepare literature summary
        lit_summary = self._prepare_literature_summary(literature)
        questions_text = "\n".join([f"- {rq.question}" for rq in research_questions])
        
        try:
            result = self.llm.format_and_invoke(
                system_prompt="You are an expert at identifying research trends and patterns in academic literature.",
                user_prompt="""Analyze the following literature and identify 5-7 major research trends.

            Research Questions:
            {questions}

            Literature Summary:
            {literature}

            For each trend, provide:
            - trend_name: Clear, concise name
            - description: 2-3 sentence description
            - key_papers: List of 3-5 paper titles from the summary
            - emergence_year: Approximate year the trend emerged
            - growth_trajectory: "Growing", "Stable", or "Declining"
            - related_questions: Which research questions relate to this trend

            Return as a JSON array of trend objects.
            Example: [{{"trend_name": "...", "description": "...", "key_papers": [...], ...}}]

            Respond with ONLY the JSON array, no other text.
            """,
                variables={
                    "questions": questions_text,
                    "literature": lit_summary
                }
            )

            if not result:
                print(f"[{self.agent_name}] Warning: Empty response from LLM")
                return []

            trends_data = safe_parse_json(result, "trends analysis")
            if trends_data is None:
                return []
            
            # Handle both list and dict formats
            if isinstance(trends_data, dict):
                if 'trends' in trends_data:
                    trends_data = trends_data['trends']
                elif 'data' in trends_data:
                    trends_data = trends_data['data']
                else:
                    trends_data = [trends_data]
            elif not isinstance(trends_data, list):
                print(f"[{self.agent_name}] Warning: Unexpected trends data format")
                return []
            
            trends = []
            for trend in trends_data:
                try:
                    trends.append(TrendAnalysis(**trend))
                except Exception as e:
                    print(f"[{self.agent_name}] Warning: Skipping invalid trend: {e}")
                    continue
            
            print(f"[{self.agent_name}] Identified {len(trends)} trends")
            return trends
        
        except Exception as e:
            print(f"[{self.agent_name}] Error analyzing trends: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _prepare_literature_summary(self, literature: List[LiteratureItem], max_papers: int = 50) -> str:
        """Prepare a summary of literature for analysis."""
        summary_parts = []
        
        for idx, paper in enumerate(literature[:max_papers], 1):
            summary = f"""
Paper {idx}: {paper.title}
Authors: {', '.join(paper.authors[:3])}
Year: {paper.year or 'N/A'}
Citations: {paper.citation_count or 'N/A'}
Abstract: {paper.abstract[:250]}...
"""
            summary_parts.append(summary)
        
        return "\n".join(summary_parts)

class CiteKeeper:
    """Agent for bibliography management."""

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
    
    def manage_bibliography(
        self,
        literature: List[LiteratureItem],
        trends: List[TrendAnalysis]
    ) -> List[CitationEntry]:
        """Create bibliography entries with trend associations."""
        print(f"\n[{self.agent_name}] Managing bibliography for {len(literature)} papers...")
        
        citations = []
        
        for paper in literature:
            # Generate citation key
            citation_key = self._generate_citation_key(paper)
            
            # Format citation
            formatted_citation = self._format_citation(paper)
            
            # Associate with trends
            related_trends = self._associate_with_trends(paper, trends)
            
            citation = CitationEntry(
                citation_key=citation_key,
                formatted_citation=formatted_citation,
                paper_title=paper.title,
                authors=paper.authors,
                year=paper.year,
                url=paper.url,
                citation_count=paper.citation_count,
                related_trends=related_trends
            )
            
            citations.append(citation)
        
        # Sort by citation count (descending)
        citations.sort(key=lambda x: x.citation_count or 0, reverse=True)
        
        print(f"[{self.agent_name}] Created {len(citations)} bibliography entries")
        return citations
    
    def _generate_citation_key(self, paper: LiteratureItem) -> str:
        """Generate a citation key (e.g., Smith2023)."""
        if not paper.authors:
            author_part = "Unknown"
        else:
            # Get first author's last name
            first_author = paper.authors[0]
            author_part = first_author.split()[-1] if first_author else "Unknown"
        
        year_part = str(paper.year) if paper.year else "XXXX"
        
        return f"{author_part}{year_part}"
    
    def _format_citation(self, paper: LiteratureItem) -> str:
        """Format citation in APA style."""
        authors_str = ", ".join(paper.authors[:3])
        if len(paper.authors) > 3:
            authors_str += " et al."
        
        year_str = f"({paper.year})" if paper.year else "(n.d.)"
        
        citation = f"{authors_str} {year_str}. {paper.title}. {paper.source}. {paper.url}"
        
        return citation
    
    def _associate_with_trends(self, paper: LiteratureItem, trends: List[TrendAnalysis]) -> List[str]:
        """Associate paper with relevant trends."""
        related_trends = []
        
        paper_text = f"{paper.title} {paper.abstract}".lower()
        
        for trend in trends:
            # Check if paper is mentioned in trend's key papers
            if paper.title in trend.key_papers:
                related_trends.append(trend.trend_name)
            # Or check for keyword overlap
            elif any(keyword.lower() in paper_text for keyword in trend.trend_name.split()):
                related_trends.append(trend.trend_name)
        
        return related_trends

class InsightSummarizer:
    """Agent for extracting knowledge insights from literature."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "InsightSummarizer"
        self.api_key = openai_api_key
        self.collector = collector
        self.llm = LLMClient(
            temperature=0.6,
            api_key=self.api_key,
            collector=collector,
            agent_name=self.agent_name
        )
    
    def extract_insights(
        self,
        literature: List[LiteratureItem],
        research_questions: List[ResearchQuestion],
        trends: List[TrendAnalysis]
    ) -> List[KnowledgeInsight]:
        """Extract key insights from literature."""
        print(f"\n[{self.agent_name}] Extracting insights from {len(literature)} papers...")
        
        # Group literature by research question
        insights = []
        
        for rq in research_questions:
            # Get papers related to this question
            related_papers = [p for p in literature if p.research_question == rq.question]
            
            if not related_papers:
                continue
            
            print(f"\n  Analyzing {len(related_papers)} papers for: {rq.question[:60]}...")
            
            # Extract insights for this question
            question_insights = self._extract_insights_for_question(rq, related_papers, trends)
            insights.extend(question_insights)
        
        print(f"\n[{self.agent_name}] Extracted {len(insights)} insights")
        return insights
    
    def _extract_insights_for_question(
        self,
        research_question: ResearchQuestion,
        papers: List[LiteratureItem],
        trends: List[TrendAnalysis]
    ) -> List[KnowledgeInsight]:
        """Extract insights for a specific research question."""
        
        # Prepare papers summary
        papers_summary = "\n\n".join([
            f"Paper: {p.title}\nAuthors: {', '.join(p.authors[:3])}\nAbstract: {p.abstract[:300]}..."
            for p in papers[:20]  # Top 20 papers
        ])
        
        # Prepare trends context
        trends_text = "\n".join([f"- {t.trend_name}: {t.description}" for t in trends[:5]])
        
        try:
            result = self.llm.format_and_invoke(
                system_prompt="You are an expert at synthesizing research insights from academic literature.",
                user_prompt="""Extract 2-3 key insights from the following papers related to the research question.

            Research Question: {question}

            Relevant Trends:
            {trends}

            Papers:
            {papers}

            For each insight, provide:
            - insight_title: Clear, concise title
            - insight_description: Detailed description (3-4 sentences)
            - supporting_papers: List of 2-4 paper titles
            - key_findings: List of 3-5 key findings
            - methodologies_used: List of methodologies mentioned
            - research_gaps: List of 2-3 identified gaps
            - related_questions: [The research question]

            Return as a JSON array of insight objects.
            Example: [{{"insight_title": "...", "insight_description": "...", ...}}]

            Respond with ONLY the JSON array, no other text.
            """,
                variables={
                    "question": research_question.question,
                    "trends": trends_text,
                    "papers": papers_summary
                }
            )

            if not result:
                print(f"    Warning: Empty response from LLM for insights")
                return []

            insights_data = safe_parse_json(result, "insights extraction")
            if insights_data is None:
                return []
            
            # Handle both list and dict formats
            if isinstance(insights_data, dict):
                if 'insights' in insights_data:
                    insights_data = insights_data['insights']
                elif 'data' in insights_data:
                    insights_data = insights_data['data']
                else:
                    insights_data = [insights_data]
            elif not isinstance(insights_data, list):
                print(f"    Warning: Unexpected insights data format")
                return []
            
            insights = []
            for insight in insights_data:
                try:
                    insights.append(KnowledgeInsight(**insight))
                except Exception as e:
                    print(f"    Warning: Skipping invalid insight: {e}")
                    continue
            
            return insights
        
        except Exception as e:
            print(f"    Error extracting insights: {e}")
            import traceback
            traceback.print_exc()
            return []

# ========== ORCHESTRATOR ==========

class LiteratureGatheringOrchestrator:
    """Orchestrates the literature gathering pipeline."""

    def __init__(self, openai_api_key: Optional[str] = None, collector: Optional[MetricsCollector] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.collector = collector

        # Initialize agents
        self.topic_crawler = TopicCrawler(self.api_key, collector=collector)
        self.trend_tracker = TrendTracker(self.api_key, collector=collector)
        self.cite_keeper = CiteKeeper(self.api_key, collector=collector)
        self.insight_summarizer = InsightSummarizer(self.api_key, collector=collector)
        
        self.literature_batch: Optional[LiteratureBatch] = None
    
    def run_gathering_pipeline(
        self,
        research_questions: List[ResearchQuestion],
        max_papers_per_question: int = 15
    ) -> LiteratureBatch:
        """Run the complete literature gathering pipeline."""
        
        print(f"\n{'='*70}")
        print(f"LITERATURE GATHERING PIPELINE")
        print(f"{'='*70}")
        print(f"Research Questions: {len(research_questions)}")
        print(f"{'='*70}\n")
        
        # Step 1: TopicCrawler - Retrieve literature
        literature = self.topic_crawler.retrieve_literature(
            research_questions=research_questions,
            max_papers_per_question=max_papers_per_question
        )
        
        # Step 2: TrendTracker - Analyze trends
        trends = self.trend_tracker.analyze_trends(
            literature=literature,
            research_questions=research_questions
        )
        
        # Step 3: CiteKeeper - Manage bibliography (receives TrendTracker results)
        citations = self.cite_keeper.manage_bibliography(
            literature=literature,
            trends=trends
        )
        
        # Step 4: InsightSummarizer - Extract insights
        insights = self.insight_summarizer.extract_insights(
            literature=literature,
            research_questions=research_questions,
            trends=trends
        )
        
        # Create literature batch
        self.literature_batch = LiteratureBatch(
            research_questions=[
                rq.model_dump() if hasattr(rq, "model_dump") else rq
                for rq in research_questions
            ],
            literature_items=literature,
            trend_analyses=trends,
            citations=citations,
            insights=insights,
            metadata={
                "created_at": datetime.now().isoformat(),
                "total_papers": len(literature),
                "total_trends": len(trends),
                "total_citations": len(citations),
                "total_insights": len(insights),
                "abstract_coverage": getattr(self.topic_crawler, "_abstract_coverage", {}),
                "provider_counts": self.topic_crawler.provider_stats.as_dict(),
            }
        )
        
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Literature Items: {len(literature)}")
        print(f"Trends Identified: {len(trends)}")
        print(f"Citations Created: {len(citations)}")
        print(f"Insights Extracted: {len(insights)}")
        print(f"{'='*70}\n")
        
        return self.literature_batch
    
    def load_research_questions(self, filepath: str) -> List[ResearchQuestion]:
        """Load research questions from IdeationTeam output."""
        print(f"[Orchestrator] Loading research questions from: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different formats
        if isinstance(data, list):
            questions = [ResearchQuestion(**q) for q in data]
        elif isinstance(data, dict) and 'questions' in data:
            questions = [ResearchQuestion(**q) for q in data['questions']]
        else:
            raise ValueError("Unexpected JSON format for research questions")
        
        print(f"[Orchestrator] Loaded {len(questions)} research questions")
        return questions
    
    def save_literature_batch(self, filepath: str):
        """Save the complete literature batch to JSON."""
        if not self.literature_batch:
            print("No literature batch to save.")
            return
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.literature_batch.model_dump(), f, indent=2, ensure_ascii=False)
        
        print(f"[Orchestrator] Literature batch saved to: {filepath}")
    
    def save_literature_csv(self, filepath: str):
        """Save literature items to CSV for easy viewing."""
        if not self.literature_batch:
            print("No literature batch to save.")
            return
        
        data = [item.model_dump() for item in self.literature_batch.literature_items]
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"[Orchestrator] Literature CSV saved to: {filepath}")
    
    def print_summary(self):
        """Print a summary of the literature batch."""
        if not self.literature_batch:
            print("No literature batch available.")
            return
        
        batch = self.literature_batch
        
        print(f"\n{'='*70}")
        print(f"LITERATURE BATCH SUMMARY")
        print(f"{'='*70}\n")
        
        # Research Questions
        print(f"Research Questions ({len(batch.research_questions)}):")
        for i, rq in enumerate(batch.research_questions, 1):
            print(f"  {i}. {rq.question}")
        
        # Top Papers
        print(f"\nTop 10 Papers (by citations):")
        sorted_papers = sorted(
            batch.literature_items,
            key=lambda x: x.citation_count or 0,
            reverse=True
        )
        for i, paper in enumerate(sorted_papers[:10], 1):
            print(f"  {i}. {paper.title}")
            print(f"     Citations: {paper.citation_count or 'N/A'} | Year: {paper.year or 'N/A'}")
        
        # Trends
        print(f"\nIdentified Trends ({len(batch.trend_analyses)}):")
        for i, trend in enumerate(batch.trend_analyses, 1):
            print(f"  {i}. {trend.trend_name} ({trend.growth_trajectory})")
            print(f"     {trend.description[:100]}...")
        
        # Insights
        print(f"\nKey Insights ({len(batch.insights)}):")
        for i, insight in enumerate(batch.insights, 1):
            print(f"  {i}. {insight.insight_title}")
            print(f"     {insight.insight_description[:100]}...")
        
        print(f"\n{'='*70}\n")

# ========== MAIN ==========

def main():
    """Main function to run the literature gathering stage."""
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Initialize orchestrator
    orchestrator = LiteratureGatheringOrchestrator()
    
    # Load research questions from IdeationTeam output
    # Default path - adjust as needed
    questions_file = auto_input(
        "Enter path to research questions JSON (or press Enter for default): ",
        default=get_default("questions_file_path"),
    ).strip()
    if not questions_file:
        questions_file = "../../1-IdeationTeam/ModeNoWcNoHITL/finalized_research_questions_automated.json"
    
    try:
        research_questions = orchestrator.load_research_questions(questions_file)
    except Exception as e:
        print(f"Error loading research questions: {e}")
        print("Using sample research questions for demonstration...")
        research_questions = [
            ResearchQuestion(
                question="How do agent-based models improve monetary policy analysis?",
                priority_rank=1,
                priority_score=0.95
            ),
            ResearchQuestion(
                question="What are the computational challenges in large-scale ABM simulations?",
                priority_rank=2,
                priority_score=0.88
            )
        ]
    
    # Run pipeline
    literature_batch = orchestrator.run_gathering_pipeline(
        research_questions=research_questions,
        max_papers_per_question=15
    )
    
    # Save results
    orchestrator.save_literature_batch("literature_batch.json")
    orchestrator.save_literature_csv("literature_items.csv")
    
    # Print summary
    orchestrator.print_summary()
    
    print("\n" + "="*70)
    print("Literature gathering complete!")
    print("Output files:")
    print("  - literature_batch.json (complete batch with all data)")
    print("  - literature_items.csv (literature items only)")
    print("="*70)

if __name__ == "__main__":
    main()
