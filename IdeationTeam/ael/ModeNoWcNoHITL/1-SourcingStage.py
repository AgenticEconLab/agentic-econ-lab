# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Automated Multi-Agent Literature Sourcing Stage (No Human-in-the-Loop)
This script uses specialized AEL agents to gather literature automatically.

Agents:
- TrendSurfer: Identifies emerging trends and recent developments
- CorpusScout: Searches academic databases for peer-reviewed literature
- ScholarSearcher: Finds highly-cited foundational papers
- GreyScout: Discovers grey literature (reports, working papers, policy docs)

Input: Research keywords or basic research ideas
Output: Ranked literature results (CSV file)
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
from pathlib import Path as _Path
_agents_dir = _Path(__file__).resolve().parent.parent.parent.parent
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

from shared.llm import LLMClient
from shared.observability import MetricsCollector
from shared.tools.tool_registry import ToolRegistry
from shared.tools.scholarly_search import (
    DEFAULT_RETRIES, ProviderStats, arxiv_econ_query, clean_venue, crossref_items,
    dedupe_papers,
)
from IdeationTeam.ael.schemas.stage_outputs import LiteratureItem, SearchQuery

# Load environment variables from parent directories
from pathlib import Path

def find_env_file():
    """Find .env file in current or parent directories."""
    current = Path(__file__).resolve().parent
    for _ in range(5):  # Check up to 5 levels
        env_path = current / ".env"
        if env_path.exists():
            return str(env_path)
        current = current.parent
    return None

env_path = find_env_file()
if env_path:
    load_dotenv(env_path)
else:
    load_dotenv()

# Get Semantic Scholar API key (displayed by ConsoleUI in MasterOrchestrator)
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
    
    def refine_query(self, research_topic: str) -> SearchQuery:
        """Refine research query based on agent specialty and feedback."""
        raise NotImplementedError("Subclasses must implement refine_query")
    
    def search(self, query: str, max_results: int = 10) -> List[LiteratureItem]:
        """Execute search based on agent specialty."""
        raise NotImplementedError("Subclasses must implement search")


class TrendSurfer(BaseAgent):
    """Agent focused on identifying emerging trends and recent developments."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        super().__init__("TrendSurfer", openai_api_key, collector=collector)
    
    def refine_query(self, research_topic: str) -> SearchQuery:
        """Generate queries focused on recent trends and developments."""
        return self.llm.format_and_invoke(
            system_prompt="You are TrendSurfer, an expert at identifying emerging trends and recent developments in research. You must respond with valid JSON only.",
            user_prompt="""Generate search queries focused on RECENT (last 2-3 years) and EMERGING trends.

            Research Topic: {topic}

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
            variables={"topic": research_topic},
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

    def refine_query(self, research_topic: str) -> SearchQuery:
        """Generate comprehensive academic search queries."""
        return self.llm.format_and_invoke(
            system_prompt="You are CorpusScout, an expert at comprehensive academic literature searches. You must respond with valid JSON only.",
            user_prompt="""Generate broad and comprehensive search queries for peer-reviewed academic literature.

            Research Topic: {topic}

            Focus on:
            - Core theoretical frameworks
            - Empirical studies and methodologies
            - Review articles and meta-analyses
            - Cross-disciplinary connections

            Respond with a JSON object containing:
            - "queries": array of 3-5 search query strings
            - "keywords": array of 5-10 keyword strings
            - "focus_areas": array of 3-5 focus area strings

            Respond with ONLY the JSON object, no other text.
            """,
            variables={"topic": research_topic},
            parse_as=SearchQuery,
        )
    
    def search(self, query: str, max_results: int = 15) -> List[LiteratureItem]:
        """Search academic databases comprehensively."""
        print(f"[CorpusScout] Comprehensive academic search: {query}")
        results = []

        # Search Semantic Scholar (with retry on 5xx)
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
    
    def refine_query(self, research_topic: str) -> SearchQuery:
        """Generate queries for foundational and highly-cited work."""
        return self.llm.format_and_invoke(
            system_prompt="You are ScholarSearcher, an expert at finding seminal and highly-cited foundational papers. You must respond with valid JSON only.",
            user_prompt="""Generate search queries for FOUNDATIONAL and HIGHLY-CITED papers.

            Research Topic: {topic}

            Focus on:
            - Seminal theoretical contributions
            - Landmark empirical studies
            - Highly-cited review articles
            - Classic papers that defined the field

            Respond with a JSON object containing:
            - "queries": array of 3-5 search query strings
            - "keywords": array of 5-10 keyword strings
            - "focus_areas": array of 3-5 focus area strings

            Respond with ONLY the JSON object, no other text.
            """,
            variables={"topic": research_topic},
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
    
    def refine_query(self, research_topic: str) -> SearchQuery:
        """Generate queries for grey literature."""
        return self.llm.format_and_invoke(
            system_prompt="You are GreyScout, an expert at finding grey literature like working papers, reports, and policy documents. You must respond with valid JSON only.",
            user_prompt="""Generate search queries for GREY LITERATURE (non-peer-reviewed but authoritative).

            Research Topic: {topic}

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

            Respond with ONLY the JSON object, no other text.
            """,
            variables={"topic": research_topic},
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
    """Orchestrates multiple agents for automated literature sourcing."""

    def __init__(self, openai_api_key: Optional[str] = None, quiet: bool = False, collector: Optional[MetricsCollector] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        self.collector = collector

        # Quiet mode suppresses verbose output (use with ConsoleUI)
        self.quiet = quiet or os.environ.get("AGENT_QUIET_MODE", "").lower() == "true"

        # Initialize all agents with collector
        self.agents = {
            "TrendSurfer": TrendSurfer(self.api_key, collector=collector),
            "CorpusScout": CorpusScout(self.api_key, collector=collector),
            "ScholarSearcher": ScholarSearcher(self.api_key, collector=collector),
            "GreyScout": GreyScout(self.api_key, collector=collector)
        }

        # Orchestrator's own LLM (for ranking) with observability
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

        # Track agent statistics for ConsoleUI
        self.agent_stats: Dict[str, Dict] = {
            name: {"items": 0, "errors": 0} for name in self.agents.keys()
        }

    def get_agent_stats(self) -> Dict[str, Dict]:
        """Get statistics for each agent (for ConsoleUI)."""
        return self.agent_stats
    
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

    def run_automated_search(
        self,
        research_topic: str,
        max_results_per_agent: int = 15
    ) -> List[LiteratureItem]:
        """Run automated single-round search without human feedback."""
        if not self.quiet:
            print(f"\n{'='*70}")
            print(f"AUTOMATED LITERATURE SEARCH")
            print(f"Topic: {research_topic}")
            print(f"{'='*70}\n")

        round_results = []

        for agent_name, agent in self.agents.items():
            if not self.quiet:
                print(f"\n--- {agent_name} ---")

            agent_items = 0
            agent_errors = 0

            # Refine query based on agent specialty
            try:
                search_query = agent.refine_query(research_topic)
                if not self.quiet:
                    print(f"Generated {len(search_query.queries)} queries")
                    print(f"Keywords: {', '.join(search_query.keywords[:5])}...")
            except Exception as e:
                agent_errors += 1
                if not self.quiet:
                    print(f"  Error refining query: {e}")
                continue

            # Execute searches
            for query in search_query.queries[:2]:  # Use top 2 queries per agent
                try:
                    results = agent.search(query, max_results_per_agent)
                    round_results.extend(results)
                    agent_items += len(results)
                    if not self.quiet:
                        print(f"  Found {len(results)} items for query: {query[:50]}...")
                except Exception as e:
                    agent_errors += 1
                    if not self.quiet:
                        print(f"  Error searching: {e}")

            # Update stats
            self.agent_stats[agent_name] = {"items": agent_items, "errors": agent_errors}

        # OpenAlex (open, keyed) — broad econ coverage WITH abstracts
        try:
            _oa = self._search_openalex(research_topic, max_results_per_agent)
            if _oa:
                round_results.extend(_oa)
                if not self.quiet:
                    print(f"\n--- OpenAlex ---\n  Found {len(_oa)} items")
        except Exception as _e:
            if not self.quiet:
                print(f"  OpenAlex error: {_e}")

        # Deduplicate
        unique_results = self._deduplicate(round_results)
        if not self.quiet:
            print(f"\n[Orchestrator] Total unique items found: {len(unique_results)}")
        
        # Rank by relevance
        # Abstract cascade + coverage reminder (open-first; fill from optional cache/Elsevier)
        from shared.tools.abstract_cache import fill_missing_abstracts, report_and_remind
        fill_missing_abstracts(unique_results, "Sourcing", getattr(self, "collector", None))
        self.abstract_coverage = report_and_remind(unique_results, "Sourcing")

        ranked_results = self._rank_by_relevance(unique_results, research_topic)
        
        self.all_results = ranked_results
        
        return ranked_results
    
    def _deduplicate(self, papers: List[LiteratureItem]) -> List[LiteratureItem]:
        """Remove duplicates by normalized title + first author (so the same paper under
        two DOIs, e.g. two SSRN postings, is listed once)."""
        return dedupe_papers(papers)

    def _rank_by_relevance(
        self,
        papers: List[LiteratureItem],
        research_topic: str
    ) -> List[LiteratureItem]:
        """Rank papers by relevance using LLM."""
        if not papers:
            return []

        if not self.quiet:
            print(f"[Orchestrator] Ranking {len(papers)} papers...")
        
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

                Papers:
                {papers}

                Provide scores as comma-separated (e.g., 0.9, 0.7, 0.5, 0.3, 0.8).
                Only numbers, no other text.
                """,
                    variables={"topic": research_topic, "papers": papers_text},
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
    
    def save_results(self, filename: str):
        """Save results to CSV."""
        results = self.all_results

        if not results:
            if not self.quiet:
                print("No results to save.")
            return

        data = [item.model_dump() for item in results]
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        if not self.quiet:
            print(f"[Orchestrator] Results saved to {filename}")
    
    def print_summary(self, top_n: int = 10):
        """Print summary of results."""
        results = self.all_results
        title = f"TOP {top_n} PAPERS"
        
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
    """Main function for automated literature sourcing (No HITL)."""
    # Example research topic
    research_topic = "Agent-based modeling in macroeconomics and monetary policy"
    
    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator()
    
    # Run automated search
    print("\n" + "="*70)
    print("AUTOMATED LITERATURE SOURCING (No Human-in-the-Loop)")
    print("="*70)
    
    results = orchestrator.run_automated_search(
        research_topic=research_topic,
        max_results_per_agent=15
    )
    
    # Display results
    orchestrator.print_summary(top_n=15)
    
    # Save results
    orchestrator.save_results("literature_results_automated.csv")
    
    # Final summary
    print(f"\n{'='*70}")
    print("SOURCING COMPLETE")
    print(f"{'='*70}")
    print(f"Total papers found: {len(results)}")
    print(f"Results saved to: literature_results_automated.csv")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
