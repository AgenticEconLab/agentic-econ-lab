# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Data Source Stage - Open Source API Mode

This script discovers and retrieves data from open-source APIs:
- FRED (Federal Reserve Economic Data)
- Yahoo Finance (stock prices, financial data)
- World Bank (development indicators)
- OECD (economic statistics)

Pipeline:
1. DataSourceDiscoveryAgent: Identify appropriate API sources for research question
2. APIConnectionAgent: Establish connections and verify credentials
3. DataSeriesSelectionAgent: Identify specific data series within each API
4. DataRetrievalAgent: Retrieve data for approved series

HITL Checkpoints (1-3):
1. Source Selection Review (after API source discovery)
2. Series Selection Review (after data series identification)
3. Data Quality Review (after data retrieval)

Input: Research question and data requirements
Output: Retrieved data series with metadata
"""

import os
import sys
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

# Add parent directories to path for shared imports
_current_dir = Path(__file__).resolve().parent
_agents_dir = _current_dir.parent.parent.parent  # repository root
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

from shared.llm import LLMClient
from shared.json_repair import repair_json
from shared.observability import MetricsCollector
from shared.tools.register_all import register_all_tools
from shared.tools.tool_registry import ToolRegistry
from shared.auto_input import auto_input, get_default
from shared.tools.redact import redact_secrets
from DataTeam.ael.schemas.stage_outputs import (
    DataRequirement, APISource, DataSeries, RetrievedData, DataSourceOutput,
)

# Try to import API libraries
try:
    import fredapi
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    print("[WARNING] fredapi not installed. FRED API will be simulated.")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("[WARNING] yfinance not installed. Yahoo Finance API will be simulated.")


# ============================================================================
# Environment Configuration
# ============================================================================

def get_env_path():
    """Find and return the path to the .env file in the repository root (the directory holding ael_config.yaml and run_ael_pipeline.py)."""
    current_dir = Path(__file__).resolve().parent
    
    while not ((current_dir / "ael_config.yaml").exists() and (current_dir / "run_ael_pipeline.py").exists()) and current_dir.parent != current_dir:
        current_dir = current_dir.parent
    
    if (current_dir / "ael_config.yaml").exists() and (current_dir / "run_ael_pipeline.py").exists():
        env_path = current_dir / ".env"
        if env_path.exists():
            return str(env_path)
    
    return None


# Load environment variables
env_path = get_env_path()
if env_path:
    load_dotenv(env_path)
else:
    load_dotenv()

# Register all tools in the ToolRegistry
register_all_tools()


# ============================================================================
# Agents
# ============================================================================

class DataSourceDiscoveryAgent:
    """Agent for discovering appropriate API data sources."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "DataSourceDiscoveryAgent"
        self.api_key = openai_api_key
        self.llm = LLMClient(
            temperature=0.3,
            api_key=self.api_key,
            collector=collector,
            agent_name=self.agent_name,
        )
    
    def discover_sources(
        self,
        research_question: str,
        data_requirements: List[DataRequirement],
        available_apis: List[str]
    ) -> List[APISource]:
        """Identify appropriate open-source API data sources for research question."""
        
        print(f"\n[{self.agent_name}] Discovering API sources for research question...")
        print(f"  Research Question: {research_question[:80]}...")
        
        # Prepare requirements context
        reqs_text = "\n".join([
            f"- {r.variable_name}: {r.description} ({r.frequency}, {r.time_period}, {r.geographic_coverage})"
            for r in data_requirements
        ])
        
        try:
            result = self.llm.format_and_invoke(
                system_prompt="You are an expert in free economic and financial data APIs.",
                user_prompt="""Identify appropriate open-source API data sources for this research question.

Research Question: {research_question}

Data Requirements:
{requirements}

Available APIs: {available_apis}

For each relevant source, provide:
- source_id: Short ID (e.g., "API1", "API2")
- source_name: Name — MUST be one of the CONNECTED sources listed below (exact spelling)
- source_type: central_bank, financial, international, or government
- url: API URL
- description: Brief description of what this source offers
- coverage: Geographic and temporal coverage
- data_quality: High, Medium, or Low
- requires_api_key: true/false
- api_key_available: true/false (based on available_apis)
- update_frequency: How often updated
- series_available: Key series/indicators available (5-10)
- research_alignment: A 1-2 sentence explanation of WHY this specific source is appropriate for the research question (e.g., "FRED provides comprehensive US macroeconomic indicators required for fiscal policy analysis, including GDP, unemployment, and interest rates at quarterly frequency")

CONNECTED sources (these have LIVE retrieval — real data will be fetched):
{connected_sources}

IMPORTANT: pick source_name ONLY from the connected list above. A source outside this list has
no connector: its data would have to be SIMULATED, which weakens the study. If you additionally
propose an unconventional/innovative source idea, add it with source_type "innovative" and
data_quality "Low" so reviewers see it is aspirational (not retrievable today).

Return as JSON with "sources" array.
Example: {{{{
  "sources": [
    {{{{
      "source_id": "API1",
      "source_name": "FRED",
      "source_type": "central_bank",
      "url": "https://fred.stlouisfed.org/",
      "description": "Comprehensive economic time series from Federal Reserve Bank of St. Louis",
      "coverage": "US and international, 1950s-present",
      "data_quality": "High",
      "requires_api_key": true,
      "api_key_available": true,
      "update_frequency": "Daily",
      "series_available": ["<series IDs chosen to match the requirements of THIS research question>"],
      "research_alignment": "<why this source fits the specific requirements of THIS research question -- not a generic macro description>"
    }}}}
  ]
}}}}

Respond with ONLY the JSON object, no other text.
""",
                variables={
                    "research_question": research_question,
                    "requirements": reqs_text,
                    "available_apis": ", ".join(available_apis),
                    "connected_sources": _connected_sources_text(),
                },
            )

            data = json.loads(repair_json(result))
            sources = [APISource(**s) for s in data.get("sources", [])]
            
            print(f"[{self.agent_name}] Discovered {len(sources)} API sources")
            for source in sources:
                print(f"  - {source.source_name}: {source.description[:50]}...")
            
            return sources
        
        except Exception as e:
            print(f"    Error discovering sources: {e}")
            print(f"    Using fallback default sources...")
            # Fallback: return default sources based on available APIs
            fallback_sources = []
            for api_name in available_apis:
                if api_name == "FRED":
                    fallback_sources.append(APISource(
                        source_id="API_FRED", source_name="FRED",
                        source_type="central_bank", url="https://fred.stlouisfed.org/",
                        description="Comprehensive economic time series from Federal Reserve Bank of St. Louis",
                        coverage="US and international, 1950s-present", data_quality="High",
                        requires_api_key=True, api_key_available=True, update_frequency="Daily",
                        series_available=["GDPC1", "UNRATE", "CPIAUCSL", "FEDFUNDS", "DGS10"],
                        research_alignment="FRED provides authoritative US macroeconomic indicators from the Federal Reserve, essential for research requiring GDP, employment, and price-level data"
                    ))
                elif api_name == "Yahoo Finance":
                    fallback_sources.append(APISource(
                        source_id="API_YAHOO", source_name="Yahoo Finance",
                        source_type="financial", url="https://finance.yahoo.com/",
                        description="Stock prices, market indices, financial data",
                        coverage="Global, 1970s-present", data_quality="Medium",
                        requires_api_key=False, api_key_available=True, update_frequency="Real-time",
                        series_available=["^GSPC", "^DJI", "^IXIC", "^TNX"],
                        research_alignment="Yahoo Finance provides real-time and historical financial market data for studying asset pricing, market dynamics, and financial-macroeconomic linkages"
                    ))
                elif api_name == "World Bank":
                    fallback_sources.append(APISource(
                        source_id="API_WB", source_name="World Bank",
                        source_type="international", url="https://data.worldbank.org/",
                        description="Development indicators, global economic data",
                        coverage="Global, 1960s-present", data_quality="High",
                        requires_api_key=False, api_key_available=True, update_frequency="Annual",
                        series_available=["NY.GDP.MKTP.CD", "SL.UEM.TOTL.ZS", "FP.CPI.TOTL.ZG"],
                        research_alignment="World Bank provides harmonized cross-country development indicators for comparative and panel analyses of economic growth, poverty, and structural transformation"
                    ))
            return fallback_sources if fallback_sources else []


class APIConnectionAgent:
    """Agent for establishing API connections and verifying credentials."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "APIConnectionAgent"
        self.api_key = openai_api_key
        self.collector = collector
        self.fred_client = None
        
        # Initialize FRED client if available
        fred_api_key = os.getenv("FRED_API_KEY")
        if fred_api_key and FRED_AVAILABLE:
            try:
                self.fred_client = fredapi.Fred(api_key=fred_api_key)
                print(f"[{self.agent_name}] FRED client initialized")
            except Exception as e:
                print(f"[{self.agent_name}] FRED client initialization failed: {e}")
    
    def verify_connections(
        self,
        discovered_sources: List[APISource]
    ) -> List[APISource]:
        """Verify connections to discovered API sources."""
        
        print(f"\n[{self.agent_name}] Verifying API connections...")
        
        verified_sources = []
        
        for source in discovered_sources:
            print(f"  Testing: {source.source_name}...")
            
            if source.source_name == "FRED":
                fred_key = os.getenv("FRED_API_KEY", "")
                if fred_key:
                    try:
                        # Test FRED connection via ToolRegistry
                        _tool_result = ToolRegistry.invoke("fred_get_series", {
                            "series_id": "GDP",
                            "observation_start": "2024-01-01",
                            "observation_end": "2024-06-01",
                        }, collector=self.collector, agent=self.agent_name)
                        if _tool_result.success:
                            source.api_key_available = True
                            print(f"    ✓ FRED: Connected successfully")
                        else:
                            source.api_key_available = False
                            print(f"    ✗ FRED: Connection failed - {_tool_result.error}")
                    except Exception as e:
                        source.api_key_available = False
                        print(f"    ✗ FRED: Connection failed - {e}")
                else:
                    source.api_key_available = False
                    print(f"    ✗ FRED: API key not configured")
            
            elif source.source_name == "Yahoo Finance":
                if YFINANCE_AVAILABLE:
                    try:
                        # Test Yahoo Finance connection via ToolRegistry
                        _tool_result = ToolRegistry.invoke("yfinance_history", {
                            "ticker": "AAPL", "period": "5d"
                        }, collector=self.collector, agent=self.agent_name)
                        source.api_key_available = True
                        print(f"    ✓ Yahoo Finance: Connected successfully")
                    except Exception as e:
                        source.api_key_available = True  # Still available, just test failed
                        print(f"    ~ Yahoo Finance: Available (test warning: {e})")
                else:
                    source.api_key_available = False
                    print(f"    ✗ Yahoo Finance: yfinance not installed")
            
            else:
                # Open keyless connectors (World Bank / DBnomics / OECD / IMF / BIS / ECB /
                # Eurostat / BLS): available whenever a real connector exists; a source with NO
                # connector is marked unavailable so series selection doesn't build on it.
                try:
                    from shared.tools.econ_connectors import connected_sources
                    _connected = {n.lower() for n in connected_sources()}
                except Exception:
                    _connected = {"world bank", "oecd"}
                if str(source.source_name).strip().lower() in _connected:
                    source.api_key_available = True
                    print(f"    ✓ {source.source_name}: Open API connector (no key required)")
                else:
                    source.api_key_available = False
                    print(f"    ✗ {source.source_name}: no live connector — would be simulated")

            verified_sources.append(source)
        
        print(f"[{self.agent_name}] Verified {len(verified_sources)} sources")
        return verified_sources


class DataSeriesSelectionAgent:
    """Agent for identifying specific data series within APIs."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "DataSeriesSelectionAgent"
        self.api_key = openai_api_key
        self.llm = LLMClient(
            temperature=0.3,
            api_key=self.api_key,
            collector=collector,
            agent_name=self.agent_name
        )
    
    def identify_series(
        self,
        research_question: str,
        data_requirements: List[DataRequirement],
        verified_sources: List[APISource]
    ) -> List[DataSeries]:
        """Identify specific data series within each API source."""
        
        print(f"\n[{self.agent_name}] Identifying data series...")
        
        reqs_text = "\n".join([
            f"- {r.variable_name}: {r.description} ({r.frequency})"
            for r in data_requirements
        ])
        
        sources_text = "\n".join([
            f"- {s.source_name}: {', '.join(s.series_available[:5])}"
            for s in verified_sources if s.api_key_available
        ])
        
        system_prompt = "You are an expert in economic data series across multiple APIs. You prioritize data CORRECTNESS by specifying exact identifiers, precise units, and justifying each selection."
        user_prompt = """Identify specific data series to retrieve for this research.

Research Question: {research_question}

Data Requirements:
{requirements}

Available Sources and Series:
{sources}

For each series, provide ALL of the following fields with PRECISE, VERIFIABLE details:
- series_id: Exact series ID (e.g., "GDPC1" for FRED, "^GSPC" for Yahoo) - must be a real, valid identifier
- series_name: Official human-readable name as listed by the source
- source_name: Source API name
- description: EXACT description of what the series measures, including methodology notes (e.g., "Real GDP measured in chained 2012 dollars using Fisher ideal index")
- frequency: daily, weekly, monthly, quarterly, or annual - must match the ACTUAL frequency of the source series
- units: EXACT units of measurement as specified by the source (e.g., "Billions of Chained 2012 Dollars", NOT just "Dollars")
- seasonal_adjustment: "SA" (seasonally adjusted) or "NSA" (not adjusted) - must match the actual series
- start_date: EXACT start date of the series (e.g., "1947-01-01" for GDPC1)
- end_date: "present" or specific date
- last_updated: "recent" or specific date
- variable_role: outcome, predictor, or control

CORRECTNESS REQUIREMENTS (critical):
1. For EACH series, explain in the description WHY this specific series was chosen over alternatives (e.g., why GDPC1 instead of GDP, why CPIAUCSL instead of CPILFESL)
2. Ensure series IDs are REAL and VALID - do not invent series IDs
3. Specify EXACT units as they appear in the source documentation
4. If a series has known limitations or caveats (e.g., revisions, rebasing, seasonal patterns), note them in the description
5. Cross-reference each series against the data requirements to ensure it ACTUALLY satisfies the stated need

Series-ID FORMAT examples (illustrative only -- do NOT default to these; FRED alone hosts 800,000+ series across many domains):
- Macro core: GDPC1 (real GDP), UNRATE (unemployment), CPIAUCSL (CPI), FEDFUNDS (policy rate), PAYEMS, INDPRO
- Financial stress / credit / frictions: NFCI (Chicago Fed Financial Conditions), STLFSI4 (St. Louis Fed Financial Stress), BAMLH0A0HYM2 (high-yield spread), DRTSCILM (bank lending standards), TOTBKCR (bank credit)
- Energy / climate-adjacent: DCOILWTICO (oil), and World Bank / OECD climate, emissions, and green-transition indicators for cross-domain questions
- Yahoo Finance symbols: ^GSPC (S&P 500), ^DJI, ^IXIC, or specific tickers

REQUIRED series_id format PER SOURCE (live retrieval parses these exactly):
{id_formats}

CRITICAL -- SELECT the series that ACTUALLY satisfy the stated data requirements for THIS specific research question, including non-standard, cross-domain, sectoral, or high-frequency series when the question calls for them (e.g., climate/transition, DeFi/crypto, shadow-banking, financial-stress proxies). Do NOT fall back to the standard GDP/unemployment/inflation trio unless the requirements specifically demand those variables. Map each stated requirement to the closest real series, and use World Bank / OECD when FRED lacks the concept.

Return as JSON with "series" array.
Example: {{{{
  "series": [
    {{{{
      "series_id": "GDPC1",
      "series_name": "Real Gross Domestic Product",
      "source_name": "FRED",
      "description": "Real GDP in billions of chained 2012 dollars, seasonally adjusted annual rate. Chosen over nominal GDP (GDP) because real values remove price effects, enabling meaningful cross-period comparison. Uses Bureau of Economic Analysis chain-weighting methodology. Subject to quarterly revisions.",
      "frequency": "quarterly",
      "units": "Billions of Chained 2012 Dollars",
      "seasonal_adjustment": "SA",
      "start_date": "1947-01-01",
      "end_date": "present",
      "last_updated": "recent",
      "variable_role": "outcome"
    }}}}
  ]
}}}}

Respond with ONLY the JSON object, no other text.
"""

        try:
            result = self.llm.format_and_invoke(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                variables={
                    "research_question": research_question,
                    "requirements": reqs_text,
                    "sources": sources_text,
                    "id_formats": _connected_sources_text(),
                }
            )

            data = json.loads(repair_json(result))
            series = [DataSeries(**s) for s in data.get("series", [])]
            
            print(f"[{self.agent_name}] Identified {len(series)} data series")
            for s in series:
                print(f"  - {s.series_id}: {s.series_name} ({s.source_name})")
            
            return series
        
        except Exception as e:
            print(f"    Error identifying series: {e}")
            print(f"    Using fallback default series...")
            # Fallback: return common economic data series
            fallback_series = []
            source_names = {s.source_name for s in verified_sources if s.api_key_available}
            if "FRED" in source_names:
                for sid, sname, desc, freq, units, role in [
                    ("GDPC1", "Real GDP", "Real Gross Domestic Product", "quarterly", "Billions of Chained 2012 Dollars", "outcome"),
                    ("UNRATE", "Unemployment Rate", "Civilian Unemployment Rate", "monthly", "Percent", "predictor"),
                    ("CPIAUCSL", "CPI", "Consumer Price Index for All Urban Consumers", "monthly", "Index 1982-84=100", "predictor"),
                ]:
                    fallback_series.append(DataSeries(
                        series_id=sid, series_name=sname, source_name="FRED",
                        description=desc, frequency=freq, units=units,
                        seasonal_adjustment="SA", start_date="1990-01-01",
                        end_date="present", last_updated="recent", variable_role=role
                    ))
            if "Yahoo Finance" in source_names:
                fallback_series.append(DataSeries(
                    series_id="^GSPC", series_name="S&P 500 Index", source_name="Yahoo Finance",
                    description="S&P 500 stock market index", frequency="daily", units="Index",
                    seasonal_adjustment="NSA", start_date="1990-01-01",
                    end_date="present", last_updated="recent", variable_role="predictor"
                ))
            return fallback_series


class DataRetrievalAgent:
    """Agent for retrieving data from APIs.

    Every real retrieval goes through :meth:`_build_retrieved`, which records: the concept
    check of the source's own title, the frequency inferred from the observation dates
    rather than the selection step's label, a discontinued/stale flag, and projections
    excluded. The FULL observations are kept in ``full_observations`` (written to a sidecar
    file by the orchestrator) so the cleaning stage aligns real data, not 5-row previews."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "DataRetrievalAgent"
        self.api_key = openai_api_key
        self.collector = collector
        self.fred_api_key = os.getenv("FRED_API_KEY", "")
        self.full_observations: Dict[str, List[Dict]] = {}
        self._failure_notes: Dict[str, List[str]] = {}

    def _note_failure(self, series: DataSeries, note: str) -> None:
        if note:
            note = redact_secrets(note)
            self.__dict__.setdefault("_failure_notes", {}).setdefault(
                series.series_id, []).append(note)
            print(f"      {note}")

    def retrieve_data(
        self,
        selected_series: List[DataSeries]
    ) -> List[RetrievedData]:
        """Retrieve data for selected series from each API source."""

        print(f"\n[{self.agent_name}] Retrieving data for {len(selected_series)} series...")

        retrieved_data = []

        for series in selected_series:
            print(f"  Retrieving: {series.series_id} from {series.source_name}...")

            if series.source_name == "FRED":
                # An LLM-proposed FRED id can be a near-miss (CURRALL for CURRCIR) —
                # route the failure through the shared repair ladder before simulating.
                data = (self._retrieve_fred_data(series)
                        or self._retrieve_via_connectors(series)
                        or self._simulate_retrieval(series))
            elif series.source_name == "Yahoo Finance":
                # A Yahoo failure goes through the same repair ladder (FRED carries
                # SP500, VIXCLS, ...) before any simulated fallback
                data = (self._retrieve_yahoo_data(series)
                        or self._retrieve_via_connectors(series)
                        or self._simulate_retrieval(series))
            else:
                # Real open connectors (World Bank / DBnomics / OECD / IMF / BIS / ECB /
                # Eurostat / BLS — all keyless) before falling back to DISCLOSED simulation.
                data = self._retrieve_via_connectors(series) or self._simulate_retrieval(series)

            if data:
                retrieved_data.append(data)
                if getattr(data, "data_simulated", False):
                    # The console log discloses a simulated fallback as the artifact does.
                    print("    ⚠ SIMULATED — all retrieval paths failed "
                          "(disclosed; never certified as real)")
                else:
                    print(f"    ✓ Retrieved {data.num_observations} observations "
                          f"({data.frequency}, title check: {data.title_check})")
            else:
                print(f"    ✗ Retrieval failed")

        print(f"[{self.agent_name}] Retrieved data for {len(retrieved_data)} series")
        return retrieved_data

    def _build_retrieved(
        self, series: DataSeries, obs: List[Dict], source_name: str, src_title: Optional[str],
        notes: str, fetch_id: Optional[str] = None, fetch_source: Optional[str] = None,
        n_projection: int = 0, projection_cutoff: str = "",
    ) -> Optional[RetrievedData]:
        """One RetrievedData from real observations ([{date, value}], chronological)."""
        from shared.tools.econ_connectors import title_concept_check
        from DataTeam.ael.data_checks import discontinued_check, infer_frequency
        if not obs:
            return None
        status, reason = title_concept_check(series.series_name, src_title,
                                             series_id=fetch_id or series.series_id)
        if status == "mismatch":
            notes += (f" PROXY: requested as '{series.series_name}'; the source's own series "
                      f"is '{src_title}' ({reason}).")
        elif status == "unverified":
            notes += (f" UNVERIFIED: {reason}; whether this series is '{series.series_name}' "
                      "could not be checked against the source's title.")
        freq = infer_frequency([o["date"] for o in obs])
        if series.frequency and freq not in ("unknown", "irregular") and \
                freq not in str(series.frequency).lower():
            notes += (f" FREQUENCY: observations are {freq}; the selection step declared "
                      f"'{series.frequency}'.")
        stale, stale_note = discontinued_check(obs[-1]["date"], freq,
                                               getattr(series, "end_date", None))
        if stale:
            notes += f" {stale_note}."
        if n_projection:
            notes += (f" PROJECTIONS EXCLUDED: {n_projection} value(s) dated after "
                      f"{projection_cutoff} (retrieval/vintage date) were dropped.")
        self.__dict__.setdefault("full_observations", {})[series.series_id] = obs
        preview = [{"date": o["date"], series.series_id: o["value"]} for o in obs[-5:]]
        return RetrievedData(
            series_id=series.series_id,
            series_name=series.series_name,
            source_title=src_title,
            proxy_for=(series.series_name if status == "mismatch" else None),
            source_name=source_name,
            retrieval_date=datetime.now().strftime('%Y-%m-%d'),
            num_observations=len(obs),
            start_date=str(obs[0]["date"]),
            end_date=str(obs[-1]["date"]),
            frequency=freq,
            declared_frequency=series.frequency,
            data_preview=preview,
            quality_notes=notes.strip(),
            retrieval_status="Success",
            fetch_id=fetch_id,
            fetch_source=fetch_source,
            title_check=status,
            title_check_reason=reason,
            discontinued=stale,
            projections_excluded=n_projection,
        )

    def _retrieve_fred_data(self, series: DataSeries) -> Optional[RetrievedData]:
        """Retrieve data from FRED API. Returns None on failure so the caller can try the
        shared repair ladder before falling back to disclosed simulation."""

        if not self.fred_api_key:
            return None

        try:
            # Get series data via ToolRegistry
            _tool_result = ToolRegistry.invoke("fred_get_series", {
                "series_id": series.series_id,
            }, collector=self.collector, agent=self.agent_name)
            if not _tool_result.success:
                self._note_failure(series, f"FRED retrieval error: {_tool_result.error}")
                return None
            response = _tool_result.data

            observations = response.get("observations", [])
            valid_obs = []
            for o in observations:
                try:
                    valid_obs.append({"date": o["date"], "value": float(o["value"])})
                except (KeyError, TypeError, ValueError):
                    continue                       # FRED marks missing values "."
            from shared.tools.econ_connectors import drop_projections
            valid_obs, n_proj, cutoff = drop_projections(valid_obs)
            if not valid_obs:
                self._note_failure(series, "FRED returned no numeric observations")
                return None

            # The source's own title, so FRED series are checked against the requested variable.
            src_title = None
            try:
                if FRED_AVAILABLE:
                    if getattr(self, "_fred_info_client", None) is None:
                        self._fred_info_client = fredapi.Fred(api_key=self.fred_api_key)
                    src_title = str(self._fred_info_client.get_series_info(series.series_id)
                                    .get("title") or "") or None
            except Exception:
                src_title = None
            return self._build_retrieved(
                series, valid_obs, "FRED", src_title,
                f"Successfully retrieved {len(valid_obs)} observations from FRED.",
                n_projection=n_proj, projection_cutoff=cutoff)
        except Exception as e:
            self._note_failure(series, f"FRED retrieval error: {e}")

        return None

    def _retrieve_yahoo_data(self, series: DataSeries) -> Optional[RetrievedData]:
        """Retrieve data from Yahoo Finance. Returns None on failure (the error is logged
        and the caller tries the repair ladder before any simulated fallback)."""

        if not YFINANCE_AVAILABLE:
            self._note_failure(series, "Yahoo Finance: yfinance not installed")
            return None

        try:
            from shared.tools.yfinance_tool import ensure_yfinance_cache
            ensure_yfinance_cache()            # per-process cache, never the shared one
            _tool_result = ToolRegistry.invoke("yfinance_history", {
                "ticker": series.series_id, "period": "max"
            }, collector=self.collector, agent=self.agent_name)
            if not _tool_result.success:
                self._note_failure(series, f"Yahoo Finance retrieval error: {_tool_result.error}")
                return None
            data = _tool_result.data
            if data is None or len(data) == 0:
                self._note_failure(series, f"Yahoo Finance returned no rows for "
                                           f"'{series.series_id}'")
                return None
            obs = []
            for idx, row in data.iterrows():
                close = row.get("Close") if hasattr(row, "get") else None
                if close is None or pd.isna(close):
                    continue
                d = str(idx.date()) if hasattr(idx, "date") else str(idx)
                obs.append({"date": d, "value": float(close)})
            from shared.tools.econ_connectors import drop_projections
            obs, n_proj, cutoff = drop_projections(obs)
            # the instrument's own name (e.g. "S&P 500" for ^GSPC), so the title check and the
            # report's number check have a source title instead of the LLM-chosen name
            title = None
            try:
                import yfinance as _yf
                info = _yf.Ticker(series.series_id).get_info() or {}
                title = str(info.get("longName") or info.get("shortName") or "").strip() or None
            except Exception:
                title = None
            return self._build_retrieved(
                series, obs, "Yahoo Finance", title,
                f"Successfully retrieved {len(obs)} daily closes from Yahoo Finance.",
                n_projection=n_proj, projection_cutoff=cutoff)
        except Exception as e:
            self._note_failure(series, f"Yahoo Finance retrieval error: {e}")
        return None

    def validate_retrieved_data(
        self,
        selected_series: List[DataSeries],
        retrieved_data: List[RetrievedData]
    ) -> List[str]:
        """Cross-check retrieved data against series specifications for correctness."""

        print(f"\n[{self.agent_name}] Validating retrieved data against specifications...")
        validation_notes = []

        # Build lookup from series_id to specification
        spec_map = {s.series_id: s for s in selected_series}
        retrieved_map = {d.series_id: d for d in retrieved_data}

        # Check 1: All requested series were retrieved
        missing = set(spec_map.keys()) - set(retrieved_map.keys())
        if missing:
            note = f"MISSING SERIES: {', '.join(missing)} were requested but not retrieved"
            validation_notes.append(note)
            print(f"  WARNING: {note}")

        # Check 2: Validate each retrieved series against its specification
        for series_id, data in retrieved_map.items():
            spec = spec_map.get(series_id)
            if not spec:
                continue

            # the frequency INFERRED from the observation dates vs the declared one
            inferred = str(data.frequency or "").lower()
            if (not data.data_simulated and inferred not in ("", "unknown", "irregular")
                    and inferred not in str(spec.frequency or "").lower()):
                note = (f"{series_id}: Frequency mismatch - declared {spec.frequency}, "
                        f"observations are {inferred}")
                validation_notes.append(note)
                print(f"  WARNING: {note}")

            # Check observation count is reasonable
            if data.num_observations < 10:
                note = f"{series_id}: Very few observations ({data.num_observations}) - may be insufficient for analysis"
                validation_notes.append(note)
                print(f"  WARNING: {note}")

            # Check retrieval status
            if data.retrieval_status not in ["Success"]:
                note = f"{series_id}: Non-success retrieval status: {data.retrieval_status}"
                validation_notes.append(note)
                print(f"  WARNING: {note}")

            if data.title_check == "mismatch":
                note = (f"{series_id}: PROXY — source title '{data.source_title}' is not "
                        f"'{data.series_name}' ({data.title_check_reason})")
                validation_notes.append(note)
                print(f"  WARNING: {note}")
            elif data.title_check == "unverified":
                note = (f"{series_id}: UNVERIFIED — no informative source title to check "
                        f"'{data.series_name}' against")
                validation_notes.append(note)
                print(f"  WARNING: {note}")
            if data.discontinued:
                note = f"{series_id}: DISCONTINUED/STALE — last observation {data.end_date}"
                validation_notes.append(note)
                print(f"  WARNING: {note}")

        if not validation_notes:
            print(f"  All {len(retrieved_data)} series passed validation checks")
        else:
            print(f"  Found {len(validation_notes)} validation issues")

        return validation_notes

    def _retrieve_via_connectors(self, series: DataSeries) -> Optional[RetrievedData]:
        """Real retrieval through the shared open-connector registry (World Bank / DBnomics /
        OECD / IMF / BIS / ECB / Eurostat / BLS — all keyless) plus the repair ladder. Returns
        None when the source has no connector, the fetch yields nothing, or every repair
        candidate failed the concept/geography check (declined, disclosed) — the caller
        then falls back to DISCLOSED simulation, never a silent fake."""
        try:
            from shared.tools.econ_connectors import fetch_series_with_repair
            out, repair_note = fetch_series_with_repair(
                series.source_name, series.series_id, series_name=series.series_name,
                collector=self.collector, agent=self.agent_name)
            if not out:
                self._note_failure(series, repair_note)
                return None
            if repair_note:
                print(f"      {repair_note}")
            notes = f"Successfully retrieved {out['n_obs']} observations from {series.source_name} (open API)."
            if repair_note:
                notes += f" [{repair_note}]"        # disclosed: the id was auto-corrected
            return self._build_retrieved(
                series, out["observations"], out.get("source_name", series.source_name),
                out.get("series_title"), notes,
                fetch_id=out.get("resolved_id"), fetch_source=out.get("resolved_source"),
                n_projection=int(out.get("n_projection_excluded", 0) or 0),
                projection_cutoff=str(out.get("projection_cutoff", "")))
        except Exception as e:
            self._note_failure(series, f"connector retrieval error ({series.source_name}): {e}")
            return None

    def _simulate_retrieval(self, series: DataSeries) -> RetrievedData:
        """Simulate data retrieval for demonstration."""

        # Create simulated preview data
        preview = []
        base_value = 100.0
        for i in range(5):
            preview.append({
                "date": f"2023-{12-i:02d}-01",
                series.series_id: round(base_value + i * 2.5, 2)
            })

        why = "; ".join(getattr(self, "_failure_notes", {}).get(series.series_id, []))
        return RetrievedData(
            series_id=series.series_id,
            series_name=series.series_name,
            source_name=series.source_name,
            retrieval_date=datetime.now().strftime('%Y-%m-%d'),
            num_observations=100,  # Simulated
            start_date="1990-01-01",
            end_date="2023-12-01",
            frequency=series.frequency,
            declared_frequency=series.frequency,
            data_preview=preview,
            quality_notes=("Simulated data for demonstration (no live API retrieval succeeded); "
                           "requirement unmet." + (f" Failures: {why}" if why else "")),
            retrieval_status="Simulated",
            data_simulated=True,
            title_check="unverified",
            title_check_reason="simulated placeholder",
        )


# ============================================================================
# HITL Checkpoints
# ============================================================================

def _connected_sources_text() -> str:
    """Connected-source list + series-id formats (single source of truth: econ_connectors).
    Used for truth-in-prompting (discovery/selection) and the committee's approval context."""
    try:
        from shared.tools.econ_connectors import CONNECTED_SOURCES
        return "\n".join(f"- {name}: {fmt}" for name, fmt in CONNECTED_SOURCES.items())
    except Exception:
        return "- FRED: series id, e.g. CPIAUCSL\n- Yahoo Finance: ticker, e.g. ^GSPC"


def _requirements_context(data_requirements: Optional[List[DataRequirement]]) -> List[Dict]:
    return [{"variable_name": r.variable_name, "description": r.description[:160],
             "frequency": r.frequency, "priority": r.priority}
            for r in (data_requirements or [])][:25]


def checkpoint1_source_selection(discovered_sources: List[APISource],
                                 research_question: str = "",
                                 data_requirements: Optional[List[DataRequirement]] = None):
    """HITL Checkpoint 1: Source Selection Review (the question itself is the prompt)."""
    from DataTeam.ael.hitl import ask
    lines = [f"Research question: {research_question}", "", "Proposed API sources:"]
    for source in discovered_sources:
        status = "connected" if source.api_key_available else "NO live connector (would be simulated)"
        lines.append(f"  - {source.source_name} [{status}]: {source.description[:80]}")
        lines.append(f"      series available: {', '.join(source.series_available[:5])}")
    # The reviewer (human or committee) decides with the CONNECTOR LIST in hand: a source
    # outside it cannot be retrieved live and would be simulated.
    ctx = {
        "research_question": research_question,
        "data_requirements": _requirements_context(data_requirements),
        "proposed_sources": [
            {"source_name": s.source_name, "description": s.description[:120],
             "connected": bool(s.api_key_available),
             "series_available": s.series_available[:5]} for s in discovered_sources
        ],
        "connected_sources_with_live_retrieval": _connected_sources_text(),
        "note": "Sources NOT in the connected list have no live connector — their data would be "
                "SIMULATED (disclosed but weak). Prefer connected sources.",
    }
    return ask("Source Selection Review", lines,
               ["Do these sources cover the data requirements of the research question?",
                "Is any needed source missing, or any proposed source unusable?"],
               default=get_default("query_review") or "approved", context=ctx)


def checkpoint2_series_selection(selected_series: List[DataSeries],
                                 research_question: str = "",
                                 data_requirements: Optional[List[DataRequirement]] = None):
    """HITL Checkpoint 2: Series Selection Review."""
    from DataTeam.ael.hitl import ask
    lines = [f"Research question: {research_question}", "", "Selected data series:"]
    for s in selected_series:
        lines.append(f"  - {s.series_id} ({s.source_name}): {s.series_name} — "
                     f"{s.frequency}, role {s.variable_role}")
    ctx = {
        "research_question": research_question,
        "data_requirements": _requirements_context(data_requirements),
        "selected_series": [
            {"series_id": s.series_id, "series_name": s.series_name,
             "source_name": s.source_name, "frequency": s.frequency, "units": s.units,
             "variable_role": s.variable_role} for s in selected_series
        ],
        "connected_sources_with_live_retrieval": _connected_sources_text(),
    }
    return ask("Series Selection Review", lines,
               ["Does each selected series measure the variable it is meant to measure?",
                "Is any required variable missing, or any series not needed?"],
               default=get_default("query_review") or "approved", context=ctx)


def checkpoint3_data_quality(retrieved_data: List[RetrievedData], research_question: str = ""):
    """HITL Checkpoint 3: Data Quality Review."""
    from DataTeam.ael.hitl import ask
    lines = [f"Research question: {research_question}", "", "Retrieved data:"]
    for d in retrieved_data:
        flags = [f for f, on in (("SIMULATED", d.data_simulated),
                                 ("PROXY", d.title_check == "mismatch"),
                                 ("TITLE UNVERIFIED", d.title_check == "unverified"
                                  and not d.data_simulated),
                                 ("DISCONTINUED", d.discontinued)) if on]
        lines.append(f"  - {d.series_id} ({d.source_name}): {d.num_observations} obs, "
                     f"{d.start_date}..{d.end_date}, {d.frequency}"
                     + (f" [{', '.join(flags)}]" if flags else ""))
    n_sim = sum(1 for d in retrieved_data if getattr(d, "data_simulated", False))
    ctx = {
        "research_question": research_question,
        "retrieved": [
            {"series_id": d.series_id, "requested_as": d.series_name,
             "source_title": d.source_title, "title_check": d.title_check,
             "source_name": d.source_name, "n_obs": d.num_observations,
             "period": f"{d.start_date}..{d.end_date}", "frequency": d.frequency,
             "status": d.retrieval_status, "discontinued": d.discontinued,
             "simulated": bool(getattr(d, "data_simulated", False)),
             "last_values": d.data_preview[-3:]} for d in retrieved_data
        ],
        "n_simulated": n_sim, "n_total": len(retrieved_data),
        "note": "Simulated series are placeholders, not real observations — weigh sufficiency "
                "on the REAL series only.",
    }
    return ask("Data Quality Review", lines,
               ["Is the retrieved data sufficient for the research question?",
                "Are the proxies, gaps and simulated placeholders acceptable?"],
               default=get_default("quality_review") or "approved", context=ctx)


# ============================================================================
# Orchestrator
# ============================================================================

class DataSourceOrchestrator:
    """Orchestrator for the data source stage."""

    def __init__(self, openai_api_key: Optional[str] = None, collector: Optional[MetricsCollector] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        self.discovery_agent = DataSourceDiscoveryAgent(self.api_key, collector=collector)
        self.connection_agent = APIConnectionAgent(self.api_key, collector=collector)
        self.series_agent = DataSeriesSelectionAgent(self.api_key, collector=collector)
        self.retrieval_agent = DataRetrievalAgent(self.api_key, collector=collector)
        self.source_output: Optional[DataSourceOutput] = None
    
    def _fetch_supplemental(self, source: str, series_id: str,
                            variable_name: str) -> Optional[RetrievedData]:
        """Deterministic fetch of one curated (source, id) pair the feasibility scout
        matched outside the retrieved pool. Never raises; None -> disclosed as still unmet."""
        try:
            from shared.tools.econ_connectors import fetch_series_with_repair
            out, repair_note = fetch_series_with_repair(
                source, series_id, series_name=variable_name)
            if not out:
                if repair_note:
                    print(f"      {repair_note}")
                return None
            # series_name stays the requested name (as on the main path); the source's
            # own title and the proxy relation are recorded separately and shown downstream
            # (so e.g. World Bank "Gini index" is not presented as "Gini Coefficient of
            # Consumption").
            spec = DataSeries(
                series_id=series_id, series_name=variable_name, source_name=source,
                description="supplemental fetch scheduled by the feasibility scout",
                frequency="", units="", seasonal_adjustment="", start_date="",
                end_date="present", last_updated="", variable_role="supplemental")
            notes = (f"Supplemental curated fetch: feasibility scout matched requirement "
                     f"'{variable_name}' to {source}/{series_id} outside the retrieval plan."
                     + (f" [{repair_note}]" if repair_note else ""))
            return self.retrieval_agent._build_retrieved(
                spec, out["observations"], out.get("source_name", source),
                out.get("series_title"), notes,
                fetch_id=out.get("resolved_id") or series_id,
                fetch_source=out.get("resolved_source") or source,
                n_projection=int(out.get("n_projection_excluded", 0) or 0),
                projection_cutoff=str(out.get("projection_cutoff", "")))
        except Exception as e:
            print(f"      supplemental fetch error ({source}/{series_id}): {redact_secrets(e)}")
            return None

    def run_source_pipeline(
        self,
        research_question: str,
        data_requirements: List[DataRequirement],
        available_apis: List[str],
        enable_hitl: bool = True,
        supplemental_series: Optional[List[Dict]] = None,
    ) -> DataSourceOutput:
        """Run the complete data source pipeline."""
        
        print(f"\n{'='*70}")
        print(f"OPEN SOURCE API DATA SOURCE PIPELINE")
        print(f"{'='*70}")
        print(f"Research Question: {research_question[:80]}...")
        print(f"Data Requirements: {len(data_requirements)}")
        print(f"Available APIs: {', '.join(available_apis)}")
        print(f"{'='*70}\n")
        
        # Step 1: Discover API sources
        print(f"STEP 1: API SOURCE DISCOVERY")
        print(f"-"*70)
        discovered_sources = self.discovery_agent.discover_sources(
            research_question,
            data_requirements,
            available_apis
        )
        
        # Step 2: Verify connections
        print(f"\nSTEP 2: API CONNECTION VERIFICATION")
        print(f"-"*70)
        verified_sources = self.connection_agent.verify_connections(discovered_sources)
        
        # Checkpoint outcomes are recorded (and acted on where a revision is possible)
        hitl_results = []

        # HITL Checkpoint 1: Source Selection Review
        if enable_hitl:
            cp1 = checkpoint1_source_selection(verified_sources, research_question,
                                               data_requirements)
            hitl_results.append(cp1)
            print(f"[HITL] Source selection: {'approved' if cp1.approved else 'OBJECTION'} "
                  f"('{cp1.response[:120]}')")

        # Step 3: Identify data series
        print(f"\nSTEP 3: DATA SERIES IDENTIFICATION")
        print(f"-"*70)
        selected_series = self.series_agent.identify_series(
            research_question,
            data_requirements,
            verified_sources
        )

        # HITL Checkpoint 2: Series Selection Review — a non-approval triggers ONE bounded
        # re-selection with the objection in hand, then one re-review; a remaining objection
        # is recorded as a limitation (never silently dropped).
        if enable_hitl:
            cp2 = checkpoint2_series_selection(selected_series, research_question,
                                               data_requirements)
            if not cp2.approved:
                print(f"[HITL] Series selection OBJECTION ('{cp2.response[:120]}') — "
                      "one bounded re-selection")
                previous = ", ".join(f"{s.series_id} ({s.series_name})" for s in selected_series)
                revised = self.series_agent.identify_series(
                    f"{research_question}\n\nREVIEWER OBJECTION to the previous selection "
                    f"[{previous}]: {cp2.response}\nRevise the selection to address it.",
                    data_requirements, verified_sources)
                if revised:
                    selected_series = revised
                    cp2.revised = True
                    again = checkpoint2_series_selection(selected_series, research_question,
                                                         data_requirements)
                    cp2.approved, cp2.final_response = again.approved, again.response
            hitl_results.append(cp2)
            print(f"[HITL] Series selection: {'approved' if cp2.approved else 'OBJECTION'}")

        # Step 4: Retrieve data
        print(f"\nSTEP 4: DATA RETRIEVAL")
        print(f"-"*70)
        retrieved_data = self.retrieval_agent.retrieve_data(selected_series)

        # Repairs can COLLAPSE distinct requested ids onto one actual
        # series (SET01+SET02 both -> CPILFESL) — the same data would then enter the
        # dataset twice under two names. Keep the first, drop later collapses, disclosed.
        seen_fetch: Dict[str, str] = {}
        deduped: List[RetrievedData] = []
        for d in retrieved_data:
            actual = d.fetch_id or d.series_id
            if actual in seen_fetch and not d.data_simulated:
                print(f"  [dedupe] dropped '{d.series_id}' — its repair resolved to "
                      f"{actual}, already retrieved as '{seen_fetch[actual]}'")
                continue
            seen_fetch.setdefault(actual, d.series_id)
            deduped.append(d)
        retrieved_data = deduped

        # Step 4a: supplemental curated fetches — series the feasibility scout matched
        # in the connector universe but the retrieval plan missed. Deterministic (exact
        # curated ids), disclosed, deduped against what the plan already fetched.
        if supplemental_series:
            print(f"\nSTEP 4a: SUPPLEMENTAL CURATED FETCHES ({len(supplemental_series)})")
            print(f"-"*70)
            already = {d.series_id for d in retrieved_data} | {
                d.fetch_id for d in retrieved_data if d.fetch_id}
            for supp in supplemental_series:
                sid = (supp.get("series_id") or "").strip()
                src = (supp.get("source") or "").strip()
                vname = supp.get("variable_name") or sid
                if not sid or sid in already:
                    continue
                supp_data = self._fetch_supplemental(src, sid, vname)
                if supp_data is not None:
                    retrieved_data.append(supp_data)
                    already.add(sid)
                    print(f"  ✓ {vname}: {src}/{sid} ({supp_data.num_observations} obs)")
                else:
                    print(f"  ✗ {vname}: {src}/{sid} fetch failed — requirement stays unmet")

        # Step 4b: Validation cross-check
        print(f"\nSTEP 4b: VALIDATION CROSS-CHECK")
        print(f"-"*70)
        validation_notes = self.retrieval_agent.validate_retrieved_data(
            selected_series, retrieved_data
        )

        # HITL Checkpoint 3: Data Quality Review (retrieval already exhausted every real
        # path; an objection is recorded as a limitation that flows to reporting)
        if enable_hitl:
            cp3 = checkpoint3_data_quality(retrieved_data, research_question)
            hitl_results.append(cp3)
            print(f"[HITL] Data quality: {'approved' if cp3.approved else 'OBJECTION'} "
                  f"('{cp3.response[:120]}')")
        
        # Update source series counts
        for source in verified_sources:
            source.series_count = sum(1 for d in retrieved_data if d.source_name == source.source_name)
        
        # Build cross-domain opportunities based on discovered sources
        source_names = [s.source_name for s in verified_sources]
        cross_domain_opps = []
        if len(source_names) >= 2:
            cross_domain_opps.append(
                f"Combine {source_names[0]} macroeconomic data with financial market data from {source_names[1]} to study transmission mechanisms"
            )
        else:
            cross_domain_opps.append(
                "Single-source analysis limits cross-domain integration"
            )
        cross_domain_opps.append(
            "Integrate labor market indicators with monetary policy data to capture employment-inflation dynamics"
        )
        cross_domain_opps.append(
            "Merge international trade data with domestic production data for global value chain analysis"
        )

        total_obs = sum(d.num_observations for d in retrieved_data)

        # Data-integrity flag: if ANY retrieved series is simulated (no live API
        # call succeeded), mark the whole run so downstream cleaning/QA cannot
        # certify the dataset as Gold/real.
        data_simulated = any(getattr(d, "data_simulated", False) for d in retrieved_data)
        num_simulated = sum(1 for d in retrieved_data if getattr(d, "data_simulated", False))
        if data_simulated:
            print(f"\n[WARNING] {num_simulated}/{len(retrieved_data)} series are SIMULATED "
                  f"(no live API). Certification will be capped (never Gold).")
            limitations_simulated = [
                f"{num_simulated} of {len(retrieved_data)} series are simulated/synthetic "
                f"(no live API retrieval) and must not be treated as real data"
            ]
        else:
            limitations_simulated = []

        # Deterministic retrieval facts and reviewer objections, stated
        # once here so they reach the report through the stage output
        limitations_retrieval = []
        for d in retrieved_data:
            if d.data_simulated:
                continue
            if d.title_check == "mismatch":
                limitations_retrieval.append(
                    f"{d.series_id} is a PROXY for '{d.series_name}': the source's own series is "
                    f"'{d.source_title}' ({d.title_check_reason})")
            elif d.title_check == "unverified":
                limitations_retrieval.append(
                    f"{d.series_id} ('{d.series_name}') is UNVERIFIED: {d.title_check_reason}")
            if d.discontinued:
                limitations_retrieval.append(
                    f"{d.series_id} looks DISCONTINUED: last observation {d.end_date}")
            if d.projections_excluded:
                limitations_retrieval.append(
                    f"{d.series_id}: {d.projections_excluded} projected value(s) after the "
                    "retrieval/vintage date were excluded")
        for d in retrieved_data:
            if d.data_simulated and "declined" in (d.quality_notes or ""):
                limitations_retrieval.append(
                    f"{d.series_id} ('{d.series_name}'): repair candidates were DECLINED as a "
                    "different concept or geography — requirement unmet")
        limitations_hitl = [r.limitation() for r in hitl_results if r.limitation()]

        rq_alignment = (
            f"Selected {len(verified_sources)} data sources targeting the research question. "
            f"Each source provides specific economic indicators relevant to the research objectives, "
            f"with a combined total of {total_obs} observations covering the required temporal and geographic scope."
        )

        # Generate assumptions and limitations based on pipeline context
        source_names_str = ', '.join(source_names)
        freq_set = set(s.frequency for s in selected_series)
        geo_set = set(r.geographic_coverage for r in data_requirements)
        assumptions = [
            f"Assumes {source_names_str} APIs provide accurate and up-to-date economic data",
            f"Assumes {', '.join(freq_set)} frequency is sufficient for the research question",
            f"Assumes {', '.join(geo_set)} data coverage is appropriate for the research scope",
            "Assumes API rate limits and data availability remain consistent during retrieval",
            "Assumes historical data revisions are acceptable for the analysis timeframe",
            "Assumes seasonal adjustment methodologies applied by source agencies are appropriate"
        ]
        limitations = [
            "Data coverage limited to publicly available open-source APIs",
            "Historical data may contain revisions not captured at the point of retrieval",
            f"Cross-source frequency alignment ({', '.join(freq_set)}) may require interpolation, introducing measurement noise",
            "Open-source APIs may have delayed updates compared to premium data vendors",
            "Data series identifiers and definitions may change across API versions",
            f"Geographic coverage limited to {', '.join(geo_set)} — results may not generalize to other regions"
        ] + limitations_simulated + limitations_retrieval + limitations_hitl

        # Create output
        self.source_output = DataSourceOutput(
            research_question=research_question,
            data_requirements=data_requirements,
            discovered_sources=verified_sources,
            selected_series=selected_series,
            retrieved_data=retrieved_data,
            cross_domain_opportunities=cross_domain_opps,
            research_question_alignment=rq_alignment,
            assumptions=assumptions,
            limitations=limitations,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "num_requirements": len(data_requirements),
                "num_sources": len(verified_sources),
                "num_series": len(selected_series),
                "num_retrieved": len(retrieved_data),
                "total_observations": total_obs,
                "validation_issues": validation_notes,
                "validation_passed": len(validation_notes) == 0,
                "data_simulated": data_simulated,
                "num_simulated_series": num_simulated,
                "hitl_checkpoints": [r.to_record() for r in hitl_results],
                "hitl_objections": [r.to_record() for r in hitl_results if not r.approved],
                "num_proxy_series": sum(1 for d in retrieved_data if d.title_check == "mismatch"),
                "num_unverified_series": sum(
                    1 for d in retrieved_data
                    if d.title_check == "unverified" and not d.data_simulated),
                "num_discontinued_series": sum(1 for d in retrieved_data if d.discontinued),
            }
        )
        
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Sources Discovered: {len(verified_sources)}")
        print(f"Series Identified: {len(selected_series)}")
        print(f"Data Retrieved: {len(retrieved_data)}")
        print(f"Total Observations: {self.source_output.metadata['total_observations']}")
        print(f"{'='*70}\n")
        
        return self.source_output
    
    def save_source_output(self, filename: str = "api_source_output.json"):
        """Save source output to JSON."""
        if not self.source_output:
            print("No source output to save")
            return

        # The FULL retrieved observations go to a sidecar file (the artifact keeps 5-row
        # previews so downstream prompts/number checks are not flooded); the cleaning stage
        # aligns and merges these real observations.
        kept = {d.series_id for d in self.source_output.retrieved_data if not d.data_simulated}
        full = {sid: obs for sid, obs in self.retrieval_agent.full_observations.items()
                if sid in kept}
        if full:
            base, _ = os.path.splitext(os.path.abspath(filename))
            obs_file = f"{base}_observations.json"
            with open(obs_file, 'w', encoding='utf-8') as f:
                json.dump(full, f)
            self.source_output.metadata["observations_file"] = obs_file
            self.source_output.metadata["observations_stored"] = sorted(full)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.source_output.model_dump(), f, indent=2)
        
        print(f"[Orchestrator] Source output saved to {filename}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main function for data source stage."""
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Example research question
    research_question = "What is the relationship between GDP growth, unemployment, and inflation in the US economy from 1990 to 2023?"
    
    # Create sample data requirements
    data_requirements = [
        DataRequirement(
            requirement_id="R1",
            variable_name="GDP",
            description="Real Gross Domestic Product",
            frequency="quarterly",
            time_period="1990-2023",
            geographic_coverage="United States",
            unit_of_measurement="Billions of chained 2012 dollars",
            priority="High",
            suggested_sources=["FRED"]
        ),
        DataRequirement(
            requirement_id="R2",
            variable_name="Unemployment Rate",
            description="Civilian unemployment rate",
            frequency="monthly",
            time_period="1990-2023",
            geographic_coverage="United States",
            unit_of_measurement="Percent",
            priority="High",
            suggested_sources=["FRED"]
        ),
        DataRequirement(
            requirement_id="R3",
            variable_name="CPI",
            description="Consumer Price Index for All Urban Consumers",
            frequency="monthly",
            time_period="1990-2023",
            geographic_coverage="United States",
            unit_of_measurement="Index 1982-84=100",
            priority="High",
            suggested_sources=["FRED"]
        )
    ]
    
    # Check available APIs
    available_apis = ["Yahoo Finance", "World Bank", "OECD"]
    if os.getenv("FRED_API_KEY"):
        available_apis.insert(0, "FRED")
    
    print(f"Available APIs: {', '.join(available_apis)}\n")
    
    # Run pipeline
    orchestrator = DataSourceOrchestrator()
    source_output = orchestrator.run_source_pipeline(
        research_question=research_question,
        data_requirements=data_requirements,
        available_apis=available_apis,
        enable_hitl=True
    )
    
    # Save outputs
    orchestrator.save_source_output("api_source_output.json")
    
    print("\n" + "="*70)
    print("DATA SOURCE SUMMARY")
    print("="*70)
    print(f"Research Question: {research_question[:60]}...")
    print(f"Requirements: {len(source_output.data_requirements)}")
    print(f"Sources: {len(source_output.discovered_sources)}")
    print(f"Series: {len(source_output.selected_series)}")
    print(f"Retrieved: {len(source_output.retrieved_data)}")
    print("="*70)


if __name__ == "__main__":
    main()
