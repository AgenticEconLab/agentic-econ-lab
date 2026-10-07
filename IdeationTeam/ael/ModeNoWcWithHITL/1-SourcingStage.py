# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Multi-Agent Literature Sourcing Stage with Human-in-the-Loop
This script uses specialized AEL agents to gather literature with feedback rounds.

Agents:
- TrendSurfer: Identifies emerging trends and recent developments
- CorpusScout: Searches academic databases for peer-reviewed literature
- ScholarSearcher: Finds highly-cited foundational papers
- GreyScout: Discovers grey literature (reports, working papers, policy docs)
"""

import os
import sys
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

# Add parent directories to path for shared imports
from pathlib import Path
current_dir = Path(__file__).resolve().parent
agents_dir = current_dir.parent.parent.parent  # repository root
if str(agents_dir) not in sys.path:
    sys.path.insert(0, str(agents_dir))

from shared.llm import LLMClient
from shared.auto_input import auto_input, get_default
from shared.observability import MetricsCollector
from shared.tools.tool_registry import ToolRegistry
from shared.tools.scholarly_search import (
    DEFAULT_RETRIES, ProviderStats, arxiv_econ_query, clean_venue, crossref_items,
    dedupe_papers,
)
from IdeationTeam.ael.schemas.stage_outputs import LiteratureItem, SearchQuery, SourcingFeedback as HumanFeedback

# Load environment variables
load_dotenv()

# Get Semantic Scholar API key
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")


class BaseAgent:
    """Base class for all literature sourcing agents."""

    def __init__(self, name: str, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.name = name
        self.collector = collector
        self.llm = LLMClient(
            temperature=0.3,
            api_key=openai_api_key,
            collector=collector,
            agent_name=name,
        )
        self.results: List[LiteratureItem] = []
    
    def refine_query(self, research_topic: str, feedback: Optional[HumanFeedback] = None) -> SearchQuery:
        """Refine research query based on agent specialty and feedback."""
        raise NotImplementedError("Subclasses must implement refine_query")
    
    def search(self, query: str, max_results: int = 10) -> List[LiteratureItem]:
        """Execute search based on agent specialty."""
        raise NotImplementedError("Subclasses must implement search")


class TrendSurfer(BaseAgent):
    """Agent focused on identifying emerging trends and recent developments."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        super().__init__("TrendSurfer", openai_api_key, collector=collector)
    
    def refine_query(self, research_topic: str, feedback: Optional[HumanFeedback] = None) -> SearchQuery:
        """Generate queries focused on recent trends and developments."""
        feedback_context = ""
        if feedback:
            feedback_context = f"""
            Previous feedback:
            - Missing topics: {', '.join(feedback.missing_topics)}
            - Additional keywords: {', '.join(feedback.additional_keywords)}
            - Comments: {feedback.comments}
            """

        return self.llm.format_and_invoke(
            system_prompt="You are TrendSurfer, an expert at identifying emerging trends and recent developments in research. You must respond with valid JSON only.",
            user_prompt="""Generate search queries focused on RECENT (last 2-3 years) and EMERGING trends.

            Research Topic: {topic}
            {feedback_context}

            Focus on:
            - New methodologies and approaches
            - Recent empirical findings
            - Emerging debates and controversies
            - Latest technological applications

            Respond with a JSON object containing:
            - "queries": array of 3-5 search query strings
            - "keywords": array of 5-10 keyword strings
            - "focus_areas": array of 3-5 focus area strings

            Example: {{"queries": ["query1", "query2"], "keywords": ["kw1", "kw2"], "focus_areas": ["area1", "area2"]}}

            Respond with ONLY the JSON object, no other text.
            """,
            variables={"topic": research_topic, "feedback_context": feedback_context},
            parse_as=SearchQuery,
        )
    
    def search(self, query: str, max_results: int = 10) -> List[LiteratureItem]:
        """Search for recent papers on arXiv and Semantic Scholar."""
        print(f"[TrendSurfer] Searching for recent trends: {query}")
        results = []

        # Search arXiv (preprints - recent trends)
        try:
            _tool_result = ToolRegistry.invoke("arxiv_search", {
                "query": arxiv_econ_query(query),  # econ/q-fin categories
                "max_results": max_results,
                "sort_by": "submittedDate",
            }, collector=self.collector, agent=self.name)
            arxiv_results = _tool_result.data if _tool_result.success else []

            for paper in arxiv_results:
                # Parse year from published string (e.g. "2024-01-15 ...")
                pub_str = paper.get("published", "")
                try:
                    pub_year = int(pub_str[:4]) if len(pub_str) >= 4 else None
                except (ValueError, TypeError):
                    pub_year = None

                # Only include papers from last 3 years
                if pub_year and pub_year >= datetime.now().year - 3:
                    item = LiteratureItem(
                        title=paper.get("title", ""),
                        authors=paper.get("authors", []),
                        abstract=paper.get("abstract", ""),
                        url=paper.get("url", ""),
                        source="arXiv",
                        agent="TrendSurfer",
                        year=pub_year,
                        literature_type="preprint"
                    )
                    results.append(item)
        except Exception as e:
            print(f"[TrendSurfer] Error searching arXiv: {e}")

        return results


class CorpusScout(BaseAgent):
    """Agent focused on comprehensive academic literature search."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        super().__init__("CorpusScout", openai_api_key, collector=collector)
    
    def refine_query(self, research_topic: str, feedback: Optional[HumanFeedback] = None) -> SearchQuery:
        """Generate comprehensive academic search queries."""
        feedback_context = ""
        if feedback:
            feedback_context = f"""
            Previous feedback:
            - Relevant papers found: {', '.join(feedback.relevant_papers[:3])}
            - Missing topics: {', '.join(feedback.missing_topics)}
            - Additional keywords: {', '.join(feedback.additional_keywords)}
            """

        return self.llm.format_and_invoke(
            system_prompt="You are CorpusScout, an expert at comprehensive academic literature searches. You must respond with valid JSON only.",
            user_prompt="""Generate broad and comprehensive search queries for peer-reviewed academic literature.

            Research Topic: {topic}
            {feedback_context}

            Focus on:
            - Core theoretical frameworks
            - Empirical studies and methodologies
            - Review articles and meta-analyses
            - Cross-disciplinary connections

            Respond with a JSON object containing:
            - "queries": array of 3-5 search query strings
            - "keywords": array of 5-10 keyword strings
            - "focus_areas": array of 3-5 focus area strings

            Example format:
            {{
              "queries": ["query 1", "query 2", "query 3"],
              "keywords": ["keyword1", "keyword2", "keyword3"],
              "focus_areas": ["area1", "area2", "area3"]
            }}

            Respond with ONLY the JSON object, no other text.
            """,
            variables={"topic": research_topic, "feedback_context": feedback_context},
            parse_as=SearchQuery,
        )
    
    def search(self, query: str, max_results: int = 15) -> List[LiteratureItem]:
        """Search academic databases comprehensively."""
        print(f"[CorpusScout] Comprehensive academic search: {query}")
        results = []

        # Search Semantic Scholar
        try:
            base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": query,
                "limit": max_results,
                "fields": "title,authors,abstract,url,year,citationCount,publicationTypes"
            }

            # Add API key header if available
            headers = {}
            if SEMANTIC_SCHOLAR_API_KEY:
                headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

            _tool_result = ToolRegistry.invoke("web_fetch", {
                "url": base_url,
                "params": params,
                "headers": headers,
                "retries": DEFAULT_RETRIES,
            }, collector=self.collector, agent=self.name)

            _status = (_tool_result.data or {}).get("status_code", 200) if _tool_result.success else None
            if _tool_result.success and not (200 <= int(_status or 200) < 300):
                raise RuntimeError(f"HTTP {_status}")
            if _tool_result.success:
                data = _tool_result.data.get("data", {})
            else:
                raise RuntimeError(_tool_result.error)

            for paper in data.get("data", []):
                item = LiteratureItem(
                    title=paper.get("title", ""),
                    authors=[author.get("name", "") for author in paper.get("authors", [])],
                    abstract=paper.get("abstract") or "No abstract available",
                    url=paper.get("url") or "",
                    source="Semantic Scholar",
                    agent="CorpusScout",
                    year=paper.get("year"),
                    literature_type="academic",
                    citation_count=paper.get("citationCount")
                )
                results.append(item)

            _st = getattr(self, "provider_stats", None)
            if _st is not None:
                _st.record("Semantic Scholar", len(results))
            # Rate limit delay
            time.sleep(1.0)

        except Exception as e:
            print(f"[CorpusScout] Semantic Scholar failed: {e}. Trying Crossref fallback...")
            _st = getattr(self, "provider_stats", None)
            if _st is not None:
                _st.record("Semantic Scholar", 0, error=str(e))
            results.extend(self._search_crossref(query, max_results))

        return results

    def _search_crossref(self, query: str, max_results: int = 10) -> List[LiteratureItem]:
        """Crossref search (fallback when Semantic Scholar is unavailable).

        Year = first present of issued / published-print / published-online / posted, venue =
        container-title; 429/5xx retried with backoff, failures logged and counted."""
        items = crossref_items(query, max_results, stats=getattr(self, "provider_stats", None),
                               collector=self.collector, agent=self.name,
                               sort_by_citations=False)
        results = []
        for it in items:
            try:
                results.append(LiteratureItem(
                    title=it["title"], authors=it["authors"], abstract=it["abstract"],
                    url=it["url"], source="Crossref", agent="CorpusScout", year=it["year"],
                    venue=it["venue"], literature_type="academic",
                    citation_count=it["citation_count"],
                ))
            except Exception as e:
                print(f"[CorpusScout] Crossref: skipped a malformed record: {e}")
        print(f"[CorpusScout] Crossref fallback returned {len(results)} papers")
        return results

class ScholarSearcher(BaseAgent):
    """Agent focused on finding highly-cited foundational papers."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        super().__init__("ScholarSearcher", openai_api_key, collector=collector)

    def refine_query(self, research_topic: str, feedback: Optional[HumanFeedback] = None) -> SearchQuery:
        """Generate queries for foundational and highly-cited work."""
        feedback_context = ""
        if feedback:
            feedback_context = f"""
            Previous feedback:
            - Additional keywords: {', '.join(feedback.additional_keywords)}
            - Comments: {feedback.comments}
            """

        return self.llm.format_and_invoke(
            system_prompt="You are ScholarSearcher, an expert at finding seminal and highly-cited foundational papers. You must respond with valid JSON only.",
            user_prompt="""Generate search queries for FOUNDATIONAL and HIGHLY-CITED papers.

            Research Topic: {topic}
            {feedback_context}

            Focus on:
            - Seminal theoretical contributions
            - Landmark empirical studies
            - Highly-cited review articles
            - Classic papers that defined the field

            Respond with a JSON object containing:
            - "queries": array of 3-5 search query strings
            - "keywords": array of 5-10 keyword strings
            - "focus_areas": array of 3-5 focus area strings

            Example: {{"queries": ["query1", "query2"], "keywords": ["kw1", "kw2"], "focus_areas": ["area1", "area2"]}}

            Respond with ONLY the JSON object, no other text.
            """,
            variables={"topic": research_topic, "feedback_context": feedback_context},
            parse_as=SearchQuery,
        )
    
    def search(self, query: str, max_results: int = 10) -> List[LiteratureItem]:
        """Search for highly-cited papers."""
        print(f"[ScholarSearcher] Searching for foundational papers: {query}")
        results = []

        # Search Semantic Scholar sorted by citations via bulk endpoint
        try:
            base_url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
            params = {
                "query": query,
                "fields": "title,authors,abstract,url,year,citationCount",
                "sort": "citationCount:desc",
                "minCitationCount": 50,
            }

            # Add API key header if available
            headers = {}
            if SEMANTIC_SCHOLAR_API_KEY:
                headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

            _tool_result = ToolRegistry.invoke("web_fetch", {
                "url": base_url,
                "params": params,
                "headers": headers,
                "retries": DEFAULT_RETRIES,
            }, collector=self.collector, agent=self.name)

            _status = (_tool_result.data or {}).get("status_code", 200) if _tool_result.success else None
            if _tool_result.success and not (200 <= int(_status or 200) < 300):
                raise RuntimeError(f"HTTP {_status}")
            if _tool_result.success:
                data = _tool_result.data.get("data", {})
            else:
                raise RuntimeError(_tool_result.error)

            for paper in data.get("data", [])[:max_results]:
                item = LiteratureItem(
                    title=paper.get("title", ""),
                    authors=[author.get("name", "") for author in paper.get("authors", [])],
                    abstract=paper.get("abstract") or "No abstract available",
                    url=paper.get("url") or "",
                    source="Semantic Scholar",
                    agent="ScholarSearcher",
                    year=paper.get("year"),
                    literature_type="academic",
                    citation_count=paper.get("citationCount", 0)
                )
                results.append(item)

            _st = getattr(self, "provider_stats", None)
            if _st is not None:
                _st.record("Semantic Scholar", len(results))
            # Rate limit delay
            time.sleep(1.0)

        except Exception as e:
            print(f"[ScholarSearcher] Semantic Scholar failed: {e}. Trying Crossref fallback...")
            _st = getattr(self, "provider_stats", None)
            if _st is not None:
                _st.record("Semantic Scholar", 0, error=str(e))
            results.extend(self._search_crossref(query, max_results))

        return results

    def _search_crossref(self, query: str, max_results: int = 10) -> List[LiteratureItem]:
        """Crossref search (fallback when Semantic Scholar is unavailable).

        Year = first present of issued / published-print / published-online / posted, venue =
        container-title; 429/5xx retried with backoff, failures logged and counted."""
        items = crossref_items(query, max_results, stats=getattr(self, "provider_stats", None),
                               collector=self.collector, agent=self.name,
                               sort_by_citations=True)
        items = [it for it in items if (it.get("citation_count") or 0) > 50]
        results = []
        for it in items:
            try:
                results.append(LiteratureItem(
                    title=it["title"], authors=it["authors"], abstract=it["abstract"],
                    url=it["url"], source="Crossref", agent="ScholarSearcher", year=it["year"],
                    venue=it["venue"], literature_type="academic",
                    citation_count=it["citation_count"],
                ))
            except Exception as e:
                print(f"[ScholarSearcher] Crossref: skipped a malformed record: {e}")
        print(f"[ScholarSearcher] Crossref fallback returned {len(results)} papers")
        return results

class GreyScout(BaseAgent):
    """Agent focused on grey literature (reports, working papers, policy documents)."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        super().__init__("GreyScout", openai_api_key, collector=collector)

    def refine_query(self, research_topic: str, feedback: Optional[HumanFeedback] = None) -> SearchQuery:
        """Generate queries for grey literature."""
        feedback_context = ""
        if feedback:
            feedback_context = f"""
            Previous feedback:
            - Missing topics: {', '.join(feedback.missing_topics)}
            - Additional keywords: {', '.join(feedback.additional_keywords)}
            """

        return self.llm.format_and_invoke(
            system_prompt="You are GreyScout, an expert at finding grey literature like working papers, reports, and policy documents. You must respond with valid JSON only.",
            user_prompt="""Generate search queries for GREY LITERATURE (non-peer-reviewed but authoritative).

            Research Topic: {topic}
            {feedback_context}

            Focus on:
            - Working papers and preprints
            - Policy reports and white papers
            - Technical reports from institutions
            - Conference proceedings
            - Think tank publications

            Respond with a JSON object containing:
            - "queries": array of 3-5 search query strings
            - "keywords": array of 5-10 keyword strings
            - "focus_areas": array of 3-5 focus area strings

            Example: {{"queries": ["query1", "query2"], "keywords": ["kw1", "kw2"], "focus_areas": ["area1", "area2"]}}

            Respond with ONLY the JSON object, no other text.
            """,
            variables={"topic": research_topic, "feedback_context": feedback_context},
            parse_as=SearchQuery,
        )

    def search(self, query: str, max_results: int = 10) -> List[LiteratureItem]:
        """Search for grey literature."""
        print(f"[GreyScout] Searching for grey literature: {query}")
        results = []

        # Search arXiv for working papers
        try:
            _tool_result = ToolRegistry.invoke("arxiv_search", {
                "query": query,
                "max_results": max_results,
                "sort_by": "relevance",
            }, collector=self.collector, agent=self.name)
            arxiv_results = _tool_result.data if _tool_result.success else []

            for paper in arxiv_results:
                pub_str = paper.get("published", "")
                try:
                    pub_year = int(pub_str[:4]) if len(pub_str) >= 4 else None
                except (ValueError, TypeError):
                    pub_year = None

                item = LiteratureItem(
                    title=paper.get("title", ""),
                    authors=paper.get("authors", []),
                    abstract=paper.get("abstract", ""),
                    url=paper.get("url", ""),
                    source="arXiv (Working Paper)",
                    agent="GreyScout",
                    year=pub_year,
                    literature_type="grey"
                )
                results.append(item)

        except Exception as e:
            print(f"[GreyScout] Error searching: {e}")

        return results


class MultiAgentOrchestrator:
    """Orchestrates multiple agents with human-in-the-loop feedback."""
    
    def __init__(self, openai_api_key: Optional[str] = None, quiet: bool = False, collector: Optional[MetricsCollector] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.quiet = quiet
        self.collector = collector

        # Initialize all agents with collector
        self.agents = {
            "TrendSurfer": TrendSurfer(self.api_key, collector=collector),
            "CorpusScout": CorpusScout(self.api_key, collector=collector),
            "ScholarSearcher": ScholarSearcher(self.api_key, collector=collector),
            "GreyScout": GreyScout(self.api_key, collector=collector)
        }

        self.llm = LLMClient(
            temperature=0.3,
            api_key=self.api_key,
            collector=collector,
            agent_name="Orchestrator",
        )
        self.all_results: List[LiteratureItem] = []
        self.provider_stats = ProviderStats()   # per-provider counts
        for _agent in self.agents.values():
            _agent.provider_stats = self.provider_stats
        self.round_results: Dict[int, List[LiteratureItem]] = {}
    
    def _search_openalex(self, query, max_results=15):
        """OpenAlex (open, keyed) — broad econ coverage WITH abstracts. Failures are logged
        and counted per provider."""
        try:
            res = ToolRegistry.invoke("openalex_search",
                {"query": query, "max_results": max_results, "econ_only": True},
                collector=getattr(self, "collector", None), agent="OpenAlex")
        except Exception as e:
            print(f"  [OpenAlex] search failed: {e}")
            self.provider_stats.record("OpenAlex", 0, error=str(e))
            return []
        if not res.success:
            print(f"  [OpenAlex] search failed: {res.error}")
            self.provider_stats.record("OpenAlex", 0, error=res.error or "request failed")
            return []
        items = []
        for p in (res.data or []):
            try:
                items.append(LiteratureItem(
                    title=p.get("title") or "", authors=p.get("authors") or [],
                    abstract=p.get("abstract") or "", url=p.get("url") or "",
                    source="OpenAlex", agent="OpenAlex", year=p.get("year"),
                    venue=clean_venue(p.get("venue")), literature_type="article"))
            except Exception as e:
                print(f"  [OpenAlex] skipped a malformed record: {e}")
        self.provider_stats.record("OpenAlex", len(items))
        return items

    def run_search_round(
        self,
        research_topic: str,
        round_number: int,
        feedback: Optional[HumanFeedback] = None,
        max_results_per_agent: int = 10
    ) -> List[LiteratureItem]:
        """Run one round of multi-agent search."""
        print(f"\n{'='*70}")
        print(f"ROUND {round_number}: Multi-Agent Literature Search")
        print(f"Topic: {research_topic}")
        print(f"{'='*70}\n")
        
        round_results = []
        
        for agent_name, agent in self.agents.items():
            print(f"\n--- {agent_name} ---")
            
            # Refine query based on agent specialty and feedback
            search_query = agent.refine_query(research_topic, feedback)
            print(f"Generated {len(search_query.queries)} queries")
            print(f"Keywords: {', '.join(search_query.keywords[:5])}...")
            
            # Execute searches
            for query in search_query.queries[:2]:  # Use top 2 queries per agent
                results = agent.search(query, max_results_per_agent)
                round_results.extend(results)
                print(f"  Found {len(results)} items for query: {query[:50]}...")
        
        # Deduplicate
        unique_results = self._deduplicate(round_results)
        # OpenAlex (open, keyed) + abstract cascade + coverage reminder (open-first)
        try:
            _oa = self._search_openalex(research_topic, 15)
            if _oa:
                unique_results = self._deduplicate(unique_results + _oa)
        except Exception:
            pass
        from shared.tools.abstract_cache import fill_missing_abstracts, report_and_remind
        fill_missing_abstracts(unique_results, "Sourcing", getattr(self, "collector", None))
        self.abstract_coverage = report_and_remind(unique_results, "Sourcing")
        print(f"\n[Orchestrator] Total unique items found: {len(unique_results)}")
        
        # Rank by relevance
        ranked_results = self._rank_by_relevance(unique_results, research_topic, feedback)
        
        self.round_results[round_number] = ranked_results
        self.all_results.extend(ranked_results)
        
        return ranked_results
    
    def _deduplicate(self, papers: List[LiteratureItem]) -> List[LiteratureItem]:
        """Remove duplicates by normalized title + first author (so the same paper under
        two DOIs, e.g. two SSRN postings, is listed once)."""
        return dedupe_papers(papers)

    def _rank_by_relevance(
        self,
        papers: List[LiteratureItem],
        research_topic: str,
        feedback: Optional[HumanFeedback] = None
    ) -> List[LiteratureItem]:
        """Rank papers by relevance using LLM."""
        if not papers:
            return []
        
        print(f"[Orchestrator] Ranking {len(papers)} papers...")
        
        feedback_context = ""
        if feedback:
            feedback_context = f"""
            Consider this feedback:
            - Relevant papers: {', '.join(feedback.relevant_papers[:3])}
            - Avoid topics like: {', '.join(feedback.irrelevant_papers[:3])}
            - Focus more on: {', '.join(feedback.missing_topics)}
            """
        
        batch_size = 5
        ranked_papers = []
        
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i+batch_size]
            
            papers_text = "\n\n".join([
                f"Paper {idx+1} (by {p.agent}):\nTitle: {p.title}\nAbstract: {p.abstract[:250]}..."
                for idx, p in enumerate(batch)
            ])
            
            try:
                result = self.llm.format_and_invoke(
                    system_prompt="You are an expert research evaluator.",
                    user_prompt="""Rate each paper's relevance to the research topic (0-1 scale).

                Research Topic: {topic}
                {feedback_context}

                Papers:
                {papers}

                Provide scores as comma-separated (e.g., 0.9, 0.7, 0.5, 0.3, 0.8).
                Only numbers, no other text.
                """,
                    variables={"topic": research_topic, "feedback_context": feedback_context, "papers": papers_text},
                )

                scores = [float(s.strip()) for s in result.split(",")]
                
                for paper, score in zip(batch, scores):
                    paper.relevance_score = score
                    ranked_papers.append(paper)
            
            except Exception as e:
                print(f"[Orchestrator] Error ranking batch: {e}")
                ranked_papers.extend(batch)
        
        ranked_papers.sort(key=lambda x: x.relevance_score or 0, reverse=True)
        return ranked_papers
    
    def collect_human_feedback(self, round_number: int, context: Optional[dict] = None) -> HumanFeedback:
        """Collect feedback from human researcher (interactive with auto-input support).

        ``context`` (the papers under review) is optional and additive: it is forwarded
        to ``auto_input`` so that, in ``llm_economist`` mode, the committee's feedback is
        grounded in the actual results. Interactive/auto modes ignore it, so standalone
        behavior is unchanged.
        """
        print(f"\n{'='*70}")
        print(f"HUMAN FEEDBACK - Round {round_number}")
        print(f"{'='*70}\n")

        print("Please provide feedback on the results:")
        print("(Press Enter to skip any field)\n")

        from shared.console_ui import resolve_feedback_entries

        relevant = auto_input(
            "Relevant paper titles OR numbers from list above (comma-separated): ",
            default=get_default("relevant_papers"), context=context
        ).strip()
        irrelevant = auto_input(
            "Irrelevant paper titles OR numbers (comma-separated): ",
            default=get_default("irrelevant_papers"), context=context
        ).strip()
        missing = auto_input(
            "Missing topics to explore (comma-separated): ",
            default=get_default("missing_topics"), context=context
        ).strip()
        keywords = auto_input(
            "Additional keywords (comma-separated): ",
            default=get_default("additional_keywords"), context=context
        ).strip()
        comments = auto_input(
            "General comments: ",
            default=get_default("general_comments"), context=context
        ).strip()

        feedback = HumanFeedback(
            round_number=round_number,
            relevant_papers=resolve_feedback_entries(relevant, keys=["title"]),
            irrelevant_papers=resolve_feedback_entries(irrelevant, keys=["title"]),
            missing_topics=[t.strip() for t in missing.split(",") if t.strip()],
            additional_keywords=[k.strip() for k in keywords.split(",") if k.strip()],
            comments=comments
        )

        return feedback
    
    def save_results(self, filename: str, round_number: Optional[int] = None):
        """Save results to CSV."""
        if round_number:
            results = self.round_results.get(round_number, [])
            filename = f"round{round_number}_{filename}"
        else:
            results = self.all_results
        
        if not results:
            print("No results to save.")
            return
        
        data = [item.model_dump() for item in results]
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"[Orchestrator] Results saved to {filename}")
    
    def print_summary(self, round_number: Optional[int] = None, top_n: int = 10):
        """Print summary of results."""
        if round_number:
            results = self.round_results.get(round_number, [])
            title = f"ROUND {round_number} - TOP {top_n} PAPERS"
        else:
            results = self.all_results
            title = f"ALL ROUNDS - TOP {top_n} PAPERS"
        
        print(f"\n{'='*70}")
        print(title)
        print(f"{'='*70}\n")
        
        for idx, paper in enumerate(results[:top_n], 1):
            print(f"{idx}. [{paper.agent}] {paper.title}")
            print(f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
            print(f"   Source: {paper.source} | Year: {paper.year or 'N/A'} | Type: {paper.literature_type}")
            if paper.citation_count:
                print(f"   Citations: {paper.citation_count}")
            if paper.relevance_score:
                print(f"   Relevance: {paper.relevance_score:.2f}")
            print(f"   URL: {paper.url}")
            print(f"   Abstract: {paper.abstract[:150]}...")
            print()
        
        # Agent statistics
        agent_counts = {}
        for paper in results:
            agent_counts[paper.agent] = agent_counts.get(paper.agent, 0) + 1
        
        print(f"\nAgent Contributions:")
        for agent, count in sorted(agent_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {agent}: {count} papers")


def main():
    """Main function with two-round human-in-the-loop process."""
    # example
    research_topic = "Agent-based modeling in macroeconomics and monetary policy"
    
    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator()
    
    # ROUND 1: Initial search
    print("\n" + "="*70)
    print("STARTING TWO-ROUND LITERATURE SEARCH WITH HUMAN FEEDBACK")
    print("="*70)
    
    round1_results = orchestrator.run_search_round(
        research_topic=research_topic,
        round_number=1,
        feedback=None,
        max_results_per_agent=8
    )
    
    # Display Round 1 results
    orchestrator.print_summary(round_number=1, top_n=15)
    orchestrator.save_results("literature_results.csv", round_number=1)
    
    # Collect human feedback
    feedback = orchestrator.collect_human_feedback(round_number=1)
    
    # Save feedback
    with open("round1_feedback.json", "w") as f:
        json.dump(feedback.model_dump(), f, indent=2)
    print("[Orchestrator] Feedback saved to round1_feedback.json")
    
    # ROUND 2: Refined search with feedback
    round2_results = orchestrator.run_search_round(
        research_topic=research_topic,
        round_number=2,
        feedback=feedback,
        max_results_per_agent=8
    )
    
    # Display Round 2 results
    orchestrator.print_summary(round_number=2, top_n=15)
    orchestrator.save_results("literature_results.csv", round_number=2)
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Round 1: {len(round1_results)} papers")
    print(f"Round 2: {len(round2_results)} papers")
    print(f"Total unique papers: {len(orchestrator.all_results)}")
    
    # Save all results
    orchestrator.save_results("literature_results_all_rounds.csv")


if __name__ == "__main__":
    main()
