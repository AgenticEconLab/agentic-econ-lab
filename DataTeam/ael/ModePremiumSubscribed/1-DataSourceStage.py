# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Data Source Stage - Premium Subscribed Data Mode

This script handles:
- Credential validation (Bloomberg, Refinitiv, WRDS)
- Cost estimation for data retrieval

Pipeline:
1. CredentialManagementAgent: Validate access credentials and permissions
2. QueryOptimizationAgent: Estimate costs for data retrieval

HITL Checkpoint (1):
1. Budget Review (after cost estimation)

Input: Research question and budget limit
Output: Credential validation and cost estimation
"""

import os
import sys
import json
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
# Add parent directories to path for shared imports
current_dir = Path(__file__).resolve().parent
agents_dir = current_dir.parent.parent.parent  # repository root
if str(agents_dir) not in sys.path:
    sys.path.insert(0, str(agents_dir))

from shared.auto_input import auto_input, get_default
from shared.llm import LLMClient
from shared.json_repair import repair_json
from shared.observability import MetricsCollector
from DataTeam.ael.schemas.stage_outputs import (
    VendorCredential, CredentialValidation, VendorCostEstimate,
    CostEstimation, PremiumDataSourceOutput,
)


# ============================================================================
# Environment Configuration
# ============================================================================

def get_env_path():
    """Find and return the path to the .env file, checking local ael directory first."""
    current_dir = Path(__file__).resolve().parent

    # First check the ael directory (parent of ModePremiumSubscribed)
    ael_dir = current_dir.parent
    local_env = ael_dir / ".env"
    if local_env.exists():
        return str(local_env)

    # Traverse up to the repository root (the directory holding ael_config.yaml and run_ael_pipeline.py)
    search_dir = current_dir
    while not ((search_dir / "ael_config.yaml").exists() and (search_dir / "run_ael_pipeline.py").exists()) and search_dir.parent != search_dir:
        search_dir = search_dir.parent

    if (search_dir / "ael_config.yaml").exists() and (search_dir / "run_ael_pipeline.py").exists():
        env_path = search_dir / ".env"
        if env_path.exists():
            return str(env_path)

    return None


# Load environment variables
env_path = get_env_path()
if env_path:
    load_dotenv(env_path)
else:
    load_dotenv()


# ============================================================================
# Premium terminal verification (ANTI-FABRICATION GATE)
# ============================================================================
# A bare API-key env var is NOT proof of a usable premium data terminal.
# Bloomberg (BLPAPI -> local Terminal on :8194), Refinitiv Eikon/Workspace
# (Data API -> local desktop proxy on :9000/:9060) and WRDS all require a live
# local terminal/session or institutional connection that an env string alone
# cannot establish. On HPC compute nodes (no terminal, no desktop) these checks
# fail, so the pipeline reports an HONEST N/A instead of fabricating a Grade-A
# run over synthetic data (see README.md, "Teams").

def _probe_tcp(host: str, port: int, timeout: float = 0.75) -> bool:
    """Best-effort: is a TCP listener reachable at host:port right now?"""
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _probe_refinitiv_terminal():
    """Refinitiv Eikon Data API needs the Eikon/Workspace desktop proxy running locally."""
    try:
        import eikon  # noqa: F401
    except ImportError:
        return False, "eikon library not importable"
    for port in (9000, 9060, 9001):
        if _probe_tcp("127.0.0.1", port):
            return True, f"Eikon proxy reachable on 127.0.0.1:{port}"
    return False, "no Eikon/Workspace desktop proxy on localhost (9000/9060)"


def _probe_bloomberg_terminal():
    """BLPAPI connects to the local Bloomberg Terminal session (default :8194)."""
    if _probe_tcp("127.0.0.1", 8194):
        return True, "BLPAPI session reachable on 127.0.0.1:8194"
    return False, "no Bloomberg Terminal session on localhost:8194"


def _probe_wrds_terminal():
    """WRDS auth cannot be auto-verified (remote PG + credentials); require explicit
    operator opt-in rather than ever auto-asserting availability."""
    try:
        import wrds  # noqa: F401
    except ImportError:
        return False, "wrds library not importable"
    return False, ("WRDS auth cannot be auto-verified; set PREMIUM_TERMINAL_VERIFIED=1 "
                   "on a host with valid WRDS credentials")


def _terminal_opt_in(vendor: str) -> bool:
    """Operator escape hatch: assert a genuine, logged-in terminal is attached on THIS
    host. Use ONLY where a real terminal exists; never on HPC/keyless nodes."""
    def _truthy(name: str) -> bool:
        return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")
    return _truthy("PREMIUM_TERMINAL_VERIFIED") or _truthy(f"{vendor.upper()}_TERMINAL_VERIFIED")


_PREMIUM_SPECS = {
    "bloomberg": (("BLOOMBERG_API_KEY",), _probe_bloomberg_terminal),
    "refinitiv": (("REFINITIV_EIKON_API_KEY",), _probe_refinitiv_terminal),
    "wrds": (("WRDS_USERNAME", "WRDS_PASSWORD"), _probe_wrds_terminal),
}


def verify_premium_terminal(vendor: str):
    """Return ``(available: bool, note: str)`` for one premium vendor.

    ``available`` is True ONLY when the credential env var(s) are set AND a real
    terminal is verifiably reachable (live local probe) OR the operator has
    explicitly opted in (PREMIUM_TERMINAL_VERIFIED=1) on a host that genuinely has
    one. The mere presence of an API-key string is NEVER sufficient — conflating
    "key string exists" with "terminal accessible" is exactly the fabrication this
    gate prevents. Probe/import failures fail CLOSED (-> not available)."""
    key = str(vendor).strip().lower()
    spec = _PREMIUM_SPECS.get(key)
    if spec is None:
        return False, f"Unknown premium vendor '{vendor}'"
    env_names, probe = spec
    if not all(os.getenv(n) for n in env_names):
        return False, f"{'/'.join(env_names)} not set"
    if _terminal_opt_in(key):
        return True, (f"{'/'.join(env_names)} set; operator-verified terminal "
                      f"(PREMIUM_TERMINAL_VERIFIED)")
    try:
        ok, note = probe()
    except Exception as e:  # fail CLOSED: any probe error -> treat as not available
        ok, note = False, f"terminal probe error: {e}"
    if ok:
        return True, f"{'/'.join(env_names)} set; live terminal reachable ({note})"
    return False, (
        f"{'/'.join(env_names)} present but NO reachable {vendor} terminal ({note}); "
        f"a credential string alone cannot retrieve data. Set PREMIUM_TERMINAL_VERIFIED=1 "
        f"only on a host with a live terminal."
    )


def detect_premium_credentials() -> Dict:
    """Build the ``credentials_status`` dict using verifiable terminal checks.

    Replaces the old ``bool(os.getenv(KEY))`` test (which fabricated availability
    from a bare key string). Shared by Stage 1's standalone path and the master
    orchestrator so both agree on what is *actually* accessible."""
    fallback_notes = {
        "Bloomberg": "Requires Bloomberg Professional terminal",
        "Refinitiv": "Requires Eikon or DataScope subscription",
        "WRDS": "Requires institutional subscription",
    }
    status = {}
    for vendor in ("Bloomberg", "Refinitiv", "WRDS"):
        available, why = verify_premium_terminal(vendor)
        status[vendor] = {
            "available": available,
            "status": "Configured" if available else "Not configured",
            "note": why or fallback_notes[vendor],
        }
    return status


# ============================================================================
# Agents
# ============================================================================

class CredentialManagementAgent:
    """Agent for validating credentials for premium data sources."""
    
    def __init__(self, openai_api_key: str):
        self.agent_name = "CredentialManagementAgent"
        self.api_key = openai_api_key
    
    def validate_credentials(
        self,
        research_question: str,
        credentials_status: Dict
    ) -> CredentialValidation:
        """Validate access credentials and permissions for premium data sources."""
        
        print(f"\n[{self.agent_name}] Validating credentials for premium data sources...")
        
        vendors = []
        accessible_sources = []
        recommendations = []
        
        # Bloomberg
        bloomberg_status = credentials_status.get("Bloomberg", {})
        bloomberg_available = bloomberg_status.get("available", False)
        vendors.append(VendorCredential(
            vendor="Bloomberg",
            available=bloomberg_available,
            status="Configured" if bloomberg_available else "Not configured",
            subscription_tier="Professional" if bloomberg_available else "N/A",
            data_entitlements=["Equity prices", "Fixed income", "Derivatives", "Economic data"] if bloomberg_available else [],
            notes=bloomberg_status.get("note", "Requires Bloomberg Professional terminal"),
            research_alignment="Bloomberg provides institutional-grade real-time and historical financial data with comprehensive coverage of equity, fixed income, derivatives, and macroeconomic indicators essential for empirical finance research"
        ))
        if bloomberg_available:
            accessible_sources.append("Bloomberg")
        else:
            recommendations.append("Configure BLOOMBERG_API_KEY for Bloomberg access")
        
        print(f"  Bloomberg: {'✓ Configured' if bloomberg_available else '✗ Not configured'}")
        
        # Refinitiv
        refinitiv_status = credentials_status.get("Refinitiv", {})
        refinitiv_available = refinitiv_status.get("available", False)
        vendors.append(VendorCredential(
            vendor="Refinitiv",
            available=refinitiv_available,
            status="Configured" if refinitiv_available else "Not configured",
            subscription_tier="Eikon" if refinitiv_available else "N/A",
            data_entitlements=["Equity prices", "FX rates", "Commodities", "ESG data"] if refinitiv_available else [],
            notes=refinitiv_status.get("note", "Requires Eikon or DataScope subscription"),
            research_alignment="Refinitiv provides deep coverage of FX markets, commodities, and ESG data, enabling cross-asset and sustainability-focused research not fully available through other vendors"
        ))
        if refinitiv_available:
            accessible_sources.append("Refinitiv")
        else:
            recommendations.append("Configure REFINITIV_EIKON_API_KEY for Refinitiv access")
        
        print(f"  Refinitiv: {'✓ Configured' if refinitiv_available else '✗ Not configured'}")
        
        # WRDS
        wrds_status = credentials_status.get("WRDS", {})
        wrds_available = wrds_status.get("available", False)
        vendors.append(VendorCredential(
            vendor="WRDS",
            available=wrds_available,
            status="Configured" if wrds_available else "Not configured",
            subscription_tier="Institutional" if wrds_available else "N/A",
            data_entitlements=["CRSP", "Compustat", "IBES", "TAQ", "OptionMetrics"] if wrds_available else [],
            notes=wrds_status.get("note", "Requires institutional subscription"),
            research_alignment="WRDS provides the gold-standard academic research datasets (CRSP, Compustat, IBES) essential for reproducible empirical finance and accounting research"
        ))
        if wrds_available:
            accessible_sources.append("WRDS")
        else:
            recommendations.append("Configure WRDS_USERNAME and WRDS_PASSWORD for WRDS access")
        
        print(f"  WRDS: {'✓ Configured' if wrds_available else '✗ Not configured'}")
        
        if not accessible_sources:
            recommendations.append("No premium sources accessible - will use simulation mode")
        
        return CredentialValidation(
            validation_timestamp=datetime.now().isoformat(),
            vendors=vendors,
            accessible_sources=accessible_sources,
            recommendations=recommendations
        )


class QueryOptimizationAgent:
    """Agent for estimating costs and optimizing queries."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "QueryOptimizationAgent"
        self.api_key = openai_api_key
        self.llm = LLMClient(
            temperature=0.3,
            api_key=self.api_key,
            collector=collector,
            agent_name=self.agent_name
        )
    
    def estimate_costs(
        self,
        research_question: str,
        credential_validation: CredentialValidation,
        budget_limit: Optional[float] = None
    ) -> CostEstimation:
        """Estimate costs for retrieving required data from premium sources."""
        
        print(f"\n[{self.agent_name}] Estimating costs for data retrieval...")
        
        vendor_estimates = []
        total_cost = 0
        cost_saving_recommendations = []
        
        # Analyze research question to estimate data needs
        data_needs = self._analyze_data_needs(research_question)
        
        # Estimate costs for each accessible source
        for source in credential_validation.accessible_sources:
            estimate = self._estimate_vendor_cost(source, data_needs)
            vendor_estimates.append(estimate)
            total_cost += estimate.estimated_cost
            print(f"  {source}: ${estimate.estimated_cost:,.2f}")
        
        # If no sources accessible, provide simulated estimates
        if not vendor_estimates:
            print("  No accessible sources - providing simulated estimates")
            
            # Simulated Bloomberg estimate
            vendor_estimates.append(VendorCostEstimate(
                vendor="Bloomberg",
                data_type="Equity and economic data",
                estimated_records=10000,
                estimated_api_calls=500,
                cost_per_unit=0.10,
                estimated_cost=50.00,
                cost_breakdown={"api_calls": 50.00, "data_points": 0},
                notes="Simulated estimate - requires Bloomberg terminal"
            ))
            
            # Simulated WRDS estimate
            vendor_estimates.append(VendorCostEstimate(
                vendor="WRDS",
                data_type="CRSP/Compustat data",
                estimated_records=50000,
                estimated_api_calls=10,
                cost_per_unit=0,
                estimated_cost=0,
                cost_breakdown={"subscription": "Included", "queries": 0},
                notes="Simulated estimate - included in institutional subscription"
            ))
            
            total_cost = 50.00
        
        # Generate cost-saving recommendations
        cost_saving_recommendations = self._generate_cost_recommendations(
            vendor_estimates,
            total_cost,
            budget_limit
        )
        
        # Check if within budget
        within_budget = budget_limit is None or total_cost <= budget_limit
        
        print(f"\n  Total Estimated Cost: ${total_cost:,.2f}")
        if budget_limit:
            print(f"  Budget Limit: ${budget_limit:,.2f}")
            print(f"  Within Budget: {'Yes' if within_budget else 'No'}")
        
        return CostEstimation(
            research_question=research_question,
            budget_limit=budget_limit,
            vendor_estimates=vendor_estimates,
            total_estimated_cost=total_cost,
            total_actual_cost=0,
            cost_saving_recommendations=cost_saving_recommendations,
            within_budget=within_budget
        )
    
    def _analyze_data_needs(self, research_question: str) -> Dict:
        """Analyze research question to estimate data needs."""
        
        # Use LLM to analyze data needs
        system_prompt = "You are an expert in financial and economic data requirements. You prioritize CORRECTNESS by specifying exact dataset names, precise coverage requirements, and justifying each recommendation."
        user_prompt = """Analyze this research question and estimate data needs with PRECISE specifications.

Research Question: {research_question}

Provide estimates as JSON with the following fields:
- securities_count: estimated number of securities (explain basis for estimate)
- date_range_years: number of years of data needed (justify the time span)
- frequency: daily/monthly/quarterly/annual (explain why this frequency is appropriate)
- data_types: list of SPECIFIC dataset names needed (use exact vendor dataset names where possible)
- vendors_recommended: list of recommended vendors with justification
- dataset_specifications: list of objects, each with:
  - dataset_name: EXACT dataset name (e.g., "CRSP Monthly Stock File", "Compustat Annual Fundamentals", NOT just "stock prices")
  - vendor: which vendor provides this dataset
  - key_variables: specific variable names needed (e.g., "PRC", "RET", "SHROUT" for CRSP)
  - justification: WHY this dataset was chosen over alternatives
  - known_limitations: any caveats or limitations of this dataset

CORRECTNESS REQUIREMENTS:
1. Use EXACT vendor-specific dataset names, not generic descriptions
2. For each dataset, explain WHY it was chosen and what ALTERNATIVES were considered
3. Specify the EXACT variables needed, not just categories
4. Note any known data quality issues (survivorship bias, backfill bias, look-ahead bias)

Example: {{{{
  "securities_count": 500,
  "date_range_years": 10,
  "frequency": "monthly",
  "data_types": ["CRSP Monthly Stock File", "Compustat Annual Fundamentals", "Thomson Reuters 13F Holdings"],
  "vendors_recommended": ["WRDS", "Bloomberg"],
  "dataset_specifications": [
    {{{{
      "dataset_name": "CRSP Monthly Stock File",
      "vendor": "WRDS",
      "key_variables": ["PRC", "RET", "SHROUT", "VOL", "PERMNO"],
      "justification": "CRSP is the gold standard for US equity returns research, preferred over Bloomberg for academic work due to delisting returns and survivorship-bias-free coverage",
      "known_limitations": "Coverage limited to NYSE/AMEX/NASDAQ-listed securities; delisting returns may understate true losses"
    }}}}
  ]
}}}}

Respond with ONLY the JSON object.
"""

        try:
            result = self.llm.format_and_invoke(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                variables={"research_question": research_question}
            )
            return json.loads(repair_json(result))
        except Exception as e:
            print(f"    Error analyzing data needs: {e}")
            return {
                "securities_count": 100,
                "date_range_years": 5,
                "frequency": "monthly",
                "data_types": ["stock_prices"],
                "vendors_recommended": ["WRDS"]
            }
    
    def _estimate_vendor_cost(self, vendor: str, data_needs: Dict) -> VendorCostEstimate:
        """Estimate cost for a specific vendor."""
        
        securities = data_needs.get("securities_count", 100)
        years = data_needs.get("date_range_years", 5)
        frequency = data_needs.get("frequency", "monthly")
        
        # Calculate periods
        if frequency == "daily":
            periods = years * 252
        elif frequency == "monthly":
            periods = years * 12
        elif frequency == "quarterly":
            periods = years * 4
        else:
            periods = years
        
        records = securities * periods
        
        if vendor == "Bloomberg":
            api_calls = min(records // 100, 1000)
            cost_per_call = 0.10
            estimated_cost = api_calls * cost_per_call
            return VendorCostEstimate(
                vendor="Bloomberg",
                data_type="Financial data",
                estimated_records=records,
                estimated_api_calls=api_calls,
                cost_per_unit=cost_per_call,
                estimated_cost=estimated_cost,
                cost_breakdown={"api_calls": estimated_cost},
                notes="Cost based on API call volume"
            )
        
        elif vendor == "Refinitiv":
            api_calls = min(records // 50, 2000)
            cost_per_call = 0.05
            estimated_cost = api_calls * cost_per_call
            return VendorCostEstimate(
                vendor="Refinitiv",
                data_type="Market data",
                estimated_records=records,
                estimated_api_calls=api_calls,
                cost_per_unit=cost_per_call,
                estimated_cost=estimated_cost,
                cost_breakdown={"api_calls": estimated_cost},
                notes="Cost based on data extraction volume"
            )
        
        else:  # WRDS
            return VendorCostEstimate(
                vendor="WRDS",
                data_type="Academic research data",
                estimated_records=records,
                estimated_api_calls=10,
                cost_per_unit=0,
                estimated_cost=0,
                cost_breakdown={"subscription": "Included"},
                notes="Included in institutional subscription"
            )
    
    def _generate_cost_recommendations(
        self,
        vendor_estimates: List[VendorCostEstimate],
        total_cost: float,
        budget_limit: Optional[float]
    ) -> List[str]:
        """Generate cost-saving recommendations."""
        
        recommendations = []
        
        # General recommendations
        recommendations.append("Request only necessary fields to reduce API calls")
        recommendations.append("Use appropriate date ranges - avoid excessive historical data")
        recommendations.append("Batch requests to reduce API overhead")
        
        # Budget-specific recommendations
        if budget_limit and total_cost > budget_limit:
            recommendations.append(f"BUDGET EXCEEDED: Consider reducing scope or using free alternatives")
            recommendations.append("Prioritize WRDS data (included in subscription)")
            recommendations.append("Consider using open-source alternatives for some data")
        
        # Vendor-specific recommendations
        for estimate in vendor_estimates:
            if estimate.vendor == "Bloomberg" and estimate.estimated_cost > 100:
                recommendations.append("Bloomberg: Consider using bulk data downloads instead of API calls")
            if estimate.vendor == "Refinitiv" and estimate.estimated_cost > 100:
                recommendations.append("Refinitiv: Use DataScope for large extractions")
        
        return recommendations


# ============================================================================
# HITL Checkpoint
# ============================================================================

def checkpoint1_budget_review(cost_estimation: CostEstimation) -> str:
    """HITL Checkpoint 1: Budget Review."""

    print("\n" + "="*70)
    print("🛑 HITL CHECKPOINT 1: Budget Review")
    print("="*70)
    print(f"\nTotal Estimated Cost: ${cost_estimation.total_estimated_cost:,.2f}")

    if cost_estimation.budget_limit:
        print(f"Budget Limit: ${cost_estimation.budget_limit:,.2f}")
        print(f"Within Budget: {'Yes' if cost_estimation.within_budget else 'No'}")

    print("\nCost Breakdown by Vendor:")
    for estimate in cost_estimation.vendor_estimates:
        print(f"  - {estimate.vendor}: ${estimate.estimated_cost:,.2f}")
        print(f"      Records: {estimate.estimated_records:,}")
        print(f"      API Calls: {estimate.estimated_api_calls:,}")

    if cost_estimation.cost_saving_recommendations:
        print("\nCost-Saving Recommendations:")
        for rec in cost_estimation.cost_saving_recommendations[:5]:
            print(f"  - {rec}")

    print("\nDecision needed:")
    print("- Is the estimated cost acceptable?")
    print("- Should we proceed with premium data or use free alternatives?")
    print("- Which vendors should we use?")
    print("- Should we apply cost-saving measures?")

    response = auto_input("\n> ", default=get_default("budget_review")).strip()
    return response if response else "approved"


# ============================================================================
# Orchestrator
# ============================================================================

class DataSourceOrchestrator:
    """Orchestrator for the data source stage."""

    def __init__(self, openai_api_key: Optional[str] = None, collector: Optional[MetricsCollector] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        self.credential_agent = CredentialManagementAgent(self.api_key)
        self.query_agent = QueryOptimizationAgent(self.api_key, collector=collector)
        self.source_output: Optional[PremiumDataSourceOutput] = None

    def run_source_pipeline(
        self,
        research_question: str,
        budget_limit: Optional[float] = None,
        credentials_status: Optional[Dict] = None,
        enable_hitl: bool = True
    ) -> PremiumDataSourceOutput:
        """Run the complete data source pipeline."""
        
        print(f"\n{'='*70}")
        print(f"CREDENTIAL VALIDATION & COST ESTIMATION PIPELINE")
        print(f"{'='*70}")
        print(f"Research Question: {research_question[:60]}..." if len(research_question) > 60 else f"Research Question: {research_question}")
        print(f"{'='*70}\n")
        
        # Use provided credentials status or check fresh. Detection now requires a
        # VERIFIABLE terminal (verify_premium_terminal), not just a key string —
        # otherwise a bare REFINITIV_EIKON_API_KEY falsely reads as "Configured".
        if credentials_status is None:
            credentials_status = detect_premium_credentials()
        
        # Step 1: Credential Validation
        print(f"STEP 1: CREDENTIAL VALIDATION")
        print(f"-"*70)
        credential_validation = self.credential_agent.validate_credentials(
            research_question,
            credentials_status
        )
        
        # Step 2: Cost Estimation
        print(f"\nSTEP 2: COST ESTIMATION")
        print(f"-"*70)
        cost_estimation = self.query_agent.estimate_costs(
            research_question,
            credential_validation,
            budget_limit
        )
        
        # HITL Checkpoint 1: Budget Review
        if enable_hitl:
            feedback = checkpoint1_budget_review(cost_estimation)
            print(f"[HITL] Budget review feedback: {feedback}")
        
        # Build cross-domain opportunities based on premium vendors
        vendor_names = [v.vendor for v in credential_validation.vendors if v.available]
        all_vendor_names = [v.vendor for v in credential_validation.vendors]
        cross_domain_opps = []
        if len(vendor_names) >= 2:
            cross_domain_opps.append(
                f"Combine {vendor_names[0]} financial data with {vendor_names[1]} market data for cross-asset analysis"
            )
        elif len(all_vendor_names) >= 2:
            cross_domain_opps.append(
                f"Combine {all_vendor_names[0]} financial data with {all_vendor_names[1]} market data for cross-asset analysis (pending credential configuration)"
            )
        else:
            cross_domain_opps.append(
                "Single-vendor analysis limits cross-domain integration"
            )
        cross_domain_opps.append(
            "Integrate CRSP equity returns with Compustat fundamentals to link market pricing with firm characteristics"
        )
        cross_domain_opps.append(
            "Merge Bloomberg macroeconomic indicators with WRDS firm-level data for macro-finance transmission analysis"
        )

        accessible_count = len(credential_validation.accessible_sources)
        total_vendors = len(credential_validation.vendors)
        rq_alignment = (
            f"Evaluated {total_vendors} premium data vendors for the research question. "
            f"{accessible_count} vendor(s) are accessible, providing specialized financial and economic datasets "
            f"with institutional-grade coverage appropriate for rigorous empirical analysis."
        )

        # Generate assumptions and limitations
        accessible_vendors_str = ', '.join(credential_validation.accessible_sources) if credential_validation.accessible_sources else 'simulated vendors'
        assumptions = [
            f"Assumes {accessible_vendors_str} provide accurate, timely, and institutional-grade data",
            "Assumes subscription entitlements cover the data fields and time ranges required for the research",
            "Assumes vendor data has been validated and cleaned to professional standards",
            f"Assumes budget limit of ${budget_limit:,.2f} is sufficient for required data extraction" if budget_limit else "Assumes no budget constraint on data acquisition",
            "Assumes vendor API endpoints remain stable during the data retrieval process",
            "Assumes institutional subscription credentials have not expired or been revoked"
        ]
        limitations = [
            "Premium data access is contingent on active subscription credentials",
            f"Cost estimates are approximate and actual costs may vary based on API call volume",
            "Vendor-specific data coverage may have gaps in certain regions, time periods, or asset classes",
            "Proprietary data cannot be redistributed, limiting reproducibility for external researchers",
            "Different vendors may use different methodologies for the same indicator, introducing cross-vendor inconsistency",
            "Data retrieval subject to vendor rate limits and service availability"
        ]

        # Gate: when no premium terminal (Bloomberg/Refinitiv/WRDS) is accessible,
        # emit a CLEAN N/A status instead of crashing. The premium pipeline cannot
        # produce real results without a local terminal, so this is
        # the expected outcome on HPC / keyless environments.
        data_available = bool(credential_validation.accessible_sources)
        if data_available:
            status = "ok"
        else:
            status = "N/A (no premium terminal: Bloomberg/Refinitiv/WRDS not accessible)"
            print(f"\n[N/A] {status} — running in simulation mode; "
                  f"results are placeholders, not real vendor data.")

        # Create output using the dedicated premium schema (NOT the open-source
        # DataSourceStageOutput, which requires data_requirements/discovered_sources/
        # selected_series/retrieved_data and previously crashed every rep here).
        self.source_output = PremiumDataSourceOutput(
            research_question=research_question,
            status=status,
            data_available=data_available,
            budget_limit=budget_limit,
            credential_validation=credential_validation,
            cost_estimation=cost_estimation,
            cross_domain_opportunities=cross_domain_opps,
            research_question_alignment=rq_alignment,
            assumptions=assumptions,
            limitations=limitations,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "pipeline_stage": "data_source",
                "hitl_enabled": enable_hitl,
                "status": status,
                "data_available": data_available
            }
        )
        
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Accessible Sources: {', '.join(credential_validation.accessible_sources) if credential_validation.accessible_sources else 'None'}")
        print(f"Estimated Cost: ${cost_estimation.total_estimated_cost:,.2f}")
        print(f"Within Budget: {'Yes' if cost_estimation.within_budget else 'No'}")
        print(f"{'='*70}\n")
        
        return self.source_output
    
    def save_source_output(self, filename: str = "premium_data_source_output.json"):
        """Save source output to JSON."""
        if not self.source_output:
            print("No source output to save")
            return
        
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
    
    # Get research question
    print("Enter your research question requiring premium data:")
    research_question = auto_input(
        "> ",
        default=get_default("data_research_question"),
    ).strip()

    if not research_question:
        research_question = "Analyze the relationship between institutional ownership and stock returns using CRSP and Compustat data."
        print(f"Using example: {research_question}")

    # Get budget limit
    print("\nEnter budget limit in USD (or press Enter for no limit):")
    budget_input = auto_input("> ", default=get_default("budget_limit")).strip()
    budget_limit = float(budget_input) if budget_input else None
    
    # Run pipeline
    orchestrator = DataSourceOrchestrator()
    source_output = orchestrator.run_source_pipeline(
        research_question=research_question,
        budget_limit=budget_limit,
        enable_hitl=True
    )
    
    # Save outputs
    orchestrator.save_source_output("premium_data_source_output.json")
    
    print("\n" + "="*70)
    print("DATA SOURCE SUMMARY")
    print("="*70)
    print(f"Accessible Sources: {', '.join(source_output.credential_validation.accessible_sources) if source_output.credential_validation.accessible_sources else 'None'}")
    print(f"Estimated Cost: ${source_output.cost_estimation.total_estimated_cost:,.2f}")
    print("="*70)


if __name__ == "__main__":
    main()
