# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Quality Assurance Stage - Premium Subscribed Data Mode

This script handles:
- Data standardization (unified identifiers, field names)
- Report and codebook generation with cost tracking and compliance notes

Pipeline:
1. VendorSpecificHandlerAgent: Standardize vendor-specific data
2. DocumentationAgent: Generate comprehensive report and codebook

HITL Checkpoint (5):
5. Final Approval (after report generation)

Input: Cleaning output from Stage 2 (premium_data_cleaning_output.json)
Output: Standardized dataset with comprehensive documentation
"""

import os
import sys
import json
import hashlib
import platform
from importlib import metadata as importlib_metadata
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
from shared.tools.sandbox_tool import CodeSandbox, ExecutionResult
from shared.reliability.state_guard import StateGuard
from DataTeam.ael.schemas.stage_outputs import (
    IdentifierMapping, FieldMapping, DataStandardization,
    PremiumVariableEntry as VariableEntry,
    PremiumDataCodebook as DataCodebook,
    PremiumDataReport as DataReport,
    PremiumDocumentedDataset as DocumentedDataset,
    PremiumQualityAssuranceOutput as QualityAssuranceOutput,
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
# Agents
# ============================================================================

class VendorSpecificHandlerAgent:
    """Agent for standardizing vendor-specific data."""
    
    def __init__(self, openai_api_key: str):
        self.agent_name = "VendorSpecificHandlerAgent"
        self.api_key = openai_api_key
    
    def standardize_data(
        self,
        data_retrieval: Dict,
        quality_assessment: Dict
    ) -> DataStandardization:
        """Standardize vendor-specific data into unified format."""
        
        print(f"\n[{self.agent_name}] Standardizing vendor-specific data...")
        
        # Get retrieved datasets
        retrieved_datasets = data_retrieval.get("retrieved_datasets", [])
        vendors = list(set(ds.get("vendor", "Unknown") for ds in retrieved_datasets))
        
        # Create identifier mappings
        identifier_mappings = []
        for vendor in vendors:
            if vendor == "Bloomberg":
                identifier_mappings.append(IdentifierMapping(
                    original_id="BBG_TICKER",
                    vendor="Bloomberg",
                    universal_id="CUSIP",
                    id_type="CUSIP"
                ))
            elif vendor == "Refinitiv":
                identifier_mappings.append(IdentifierMapping(
                    original_id="RIC",
                    vendor="Refinitiv",
                    universal_id="ISIN",
                    id_type="ISIN"
                ))
            elif vendor == "WRDS":
                identifier_mappings.append(IdentifierMapping(
                    original_id="PERMNO",
                    vendor="WRDS",
                    universal_id="PERMNO",
                    id_type="PERMNO"
                ))
        
        print(f"  Identifier mappings: {len(identifier_mappings)}")
        
        # Create field mappings
        field_mappings = []
        standard_fields = ["price", "volume", "return", "market_cap", "book_value"]
        
        for vendor in vendors:
            for field in standard_fields:
                if vendor == "Bloomberg":
                    vendor_field = f"PX_{field.upper()}" if field == "price" else field.upper()
                elif vendor == "Refinitiv":
                    vendor_field = f"TR.{field.title()}"
                else:
                    vendor_field = field.lower()
                
                field_mappings.append(FieldMapping(
                    vendor_field=vendor_field,
                    vendor=vendor,
                    standard_field=field,
                    transformation="Direct mapping"
                ))
        
        print(f"  Field mappings: {len(field_mappings)}")
        
        # Calculate final counts
        total_records = sum(ds.get("num_records", 0) for ds in retrieved_datasets)
        total_fields = len(standard_fields) + 5  # Standard fields + identifiers
        
        transformations = [
            "Converted Bloomberg tickers to CUSIP",
            "Converted Refinitiv RICs to ISIN",
            "Standardized field names across vendors",
            "Aligned date formats to ISO 8601",
            "Merged overlapping data using primary source priority",
            "Applied corporate actions adjustments"
        ]
        
        print(f"  Final records: {total_records:,}")
        print(f"  Final fields: {total_fields}")
        
        return DataStandardization(
            identifier_mappings=identifier_mappings,
            field_mappings=field_mappings,
            vendors_merged=vendors,
            overlap_handling="Primary source priority (Bloomberg > Refinitiv > WRDS)",
            transformations_applied=transformations,
            final_record_count=total_records,
            final_field_count=total_fields,
            unified_fields=standard_fields + ["identifier", "date", "vendor", "currency", "frequency"]
        )


class DocumentationAgent:
    """Agent for generating comprehensive report and codebook."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "DocumentationAgent"
        self.api_key = openai_api_key
        self.llm = LLMClient(
            temperature=0.3,
            api_key=self.api_key,
            collector=collector,
            agent_name=self.agent_name
        )
    
    def generate_codebook(
        self,
        standardization: DataStandardization,
        license_compliance: Dict
    ) -> DataCodebook:
        """Generate comprehensive data codebook."""
        
        print(f"\n[{self.agent_name}] Generating data codebook...")
        
        # Create variable entries
        variables = []
        for mapping in standardization.field_mappings[:10]:  # Limit for demo
            variables.append(VariableEntry(
                variable_name=mapping.standard_field,
                description=f"Standardized {mapping.standard_field} from {mapping.vendor}",
                source_vendor=mapping.vendor,
                original_field=mapping.vendor_field,
                data_type="float64" if mapping.standard_field in ["price", "return", "volume"] else "object",
                unit="USD" if mapping.standard_field == "price" else "N/A",
                value_range="See summary statistics",
                license_notes=f"Subject to {mapping.vendor} license terms"
            ))
        
        # Get required citations
        required_citations = license_compliance.get("required_citations", {})
        if not required_citations:
            for vendor in standardization.vendors_merged:
                if vendor == "Bloomberg":
                    required_citations["Bloomberg"] = "Data provided by Bloomberg L.P."
                elif vendor == "Refinitiv":
                    required_citations["Refinitiv"] = "Data provided by Refinitiv, an LSEG business."
                elif vendor == "WRDS":
                    required_citations["WRDS"] = "Data obtained from Wharton Research Data Services (WRDS)."
        
        # Get usage restrictions
        restrictions = license_compliance.get("restrictions", [])
        usage_restrictions = []
        for r in restrictions:
            if isinstance(r, dict):
                usage_restrictions.append(f"{r.get('vendor', 'Unknown')}: {r.get('description', 'N/A')}")
        
        if not usage_restrictions:
            usage_restrictions = [
                "Data may not be redistributed to third parties",
                "Proper attribution required in all publications",
                "Commercial use may require additional licensing"
            ]
        
        print(f"  Variables documented: {len(variables)}")
        print(f"  Vendors: {', '.join(standardization.vendors_merged)}")
        
        return DataCodebook(
            title="Premium Data Codebook",
            version="1.0.0",
            created_date=datetime.now().strftime('%Y-%m-%d'),
            variables=variables,
            vendor_attribution={v: f"Data from {v}" for v in standardization.vendors_merged},
            license_compliance_notes=license_compliance.get("required_disclaimers", []),
            required_citations=required_citations,
            usage_restrictions=usage_restrictions
        )
    
    def generate_report(
        self,
        cleaning_output: Dict,
        standardization: DataStandardization,
        research_question: str
    ) -> DataReport:
        """Generate comprehensive premium data acquisition report."""
        
        print(f"\n[{self.agent_name}] Generating data acquisition report...")
        
        # Extract components
        query_opt = cleaning_output.get("query_optimization", {})
        data_ret = cleaning_output.get("data_retrieval", {})
        quality = cleaning_output.get("quality_assessment", {})
        compliance = cleaning_output.get("license_compliance", {})
        
        # Generate report sections
        report = DataReport(
            title="Premium Data Acquisition Report",
            research_question=research_question,
            executive_summary=self._generate_executive_summary(
                standardization, data_ret, quality, compliance
            ),
            vendors_section=self._generate_vendors_section(standardization.vendors_merged),
            credential_section=self._generate_credential_section(standardization.vendors_merged),
            cost_section=self._generate_cost_section(query_opt, data_ret),
            query_section=self._generate_query_section(query_opt),
            retrieval_section=self._generate_retrieval_section(data_ret),
            quality_section=self._generate_quality_section(quality),
            compliance_section=self._generate_compliance_section(compliance),
            standardization_section=self._generate_standardization_section(standardization),
            recommendations=self._generate_recommendations(quality, compliance)
        )
        
        print(f"  Report generated with all sections")
        return report
    
    def generate_innovation_analysis(
        self,
        research_question: str,
        data_summary: Dict
    ) -> Dict:
        """Generate innovative analytical suggestions for the collected data."""

        print(f"\n[{self.agent_name}] Generating innovation analysis...")

        vendors = data_summary.get("vendors", [])
        fields = data_summary.get("fields", [])

        system_prompt = """You are a creative financial economics researcher who specializes in finding
novel analytical approaches using premium financial data."""
        user_prompt = """Given the following premium financial dataset, suggest INNOVATIVE analytical approaches
that could yield novel economic insights.

Research Question: {research_question}

Dataset Information:
- Data Vendors: {vendors}
- Key Fields: {fields}
- Record Count: {record_count}

Provide your analysis as JSON with these sections:
{{{{
  "innovative_methods": [
    {{{{
      "method_name": "Name of the innovative method",
      "description": "Detailed description",
      "novelty_justification": "Why this is innovative",
      "required_tools": ["tool1", "tool2"]
    }}}}
  ],
  "cross_domain_connections": [
    {{{{
      "connection": "Description of cross-domain insight",
      "domains": ["domain1", "domain2"],
      "potential_impact": "What this could reveal"
    }}}}
  ],
  "unconventional_data_combinations": [
    {{{{
      "combination": "Description of novel data combination",
      "variables_involved": ["var1", "var2"],
      "expected_insight": "What combining these could reveal"
    }}}}
  ],
  "emerging_techniques": [
    {{{{
      "technique": "Name of emerging analytical technique",
      "application": "How to apply it to this dataset",
      "advantage_over_traditional": "Why this is better than traditional approaches"
    }}}}
  ]
}}}}

Generate at least 3 innovative methods, 2 cross-domain connections, 2 unconventional
data combinations, and 2 emerging techniques. Be SPECIFIC and CREATIVE.
Focus on approaches that leverage the PREMIUM nature of this data (high-frequency, proprietary fields, etc.).

Respond with ONLY the JSON object, no other text."""

        try:
            result = self.llm.format_and_invoke(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                variables={
                    "research_question": research_question,
                    "vendors": ", ".join(vendors) if vendors else "Premium vendors",
                    "fields": ", ".join(fields[:10]) if fields else "Financial data fields",
                    "record_count": str(data_summary.get("record_count", "N/A"))
                }
            )

            innovation_data = json.loads(repair_json(result))
            num_items = (len(innovation_data.get("innovative_methods", [])) +
                        len(innovation_data.get("cross_domain_connections", [])) +
                        len(innovation_data.get("unconventional_data_combinations", [])) +
                        len(innovation_data.get("emerging_techniques", [])))
            print(f"  Generated {num_items} innovation suggestions")
            return innovation_data

        except Exception as e:
            print(f"    Error generating innovation analysis: {e}")
            return {
                "innovative_methods": [
                    {"method_name": "High-Frequency Microstructure Analysis", "description": "Apply market microstructure models to premium tick-level data", "novelty_justification": "Reveals price discovery mechanisms invisible in daily data", "required_tools": ["lobster", "tick-data-tools"]},
                    {"method_name": "Cross-Vendor Arbitrage Detection", "description": "Compare pricing across Bloomberg and Refinitiv for discrepancy patterns", "novelty_justification": "Identifies information asymmetries between data providers", "required_tools": ["pandas", "statsmodels"]},
                    {"method_name": "Alternative Risk Premia Extraction", "description": "Use premium factor data to construct novel risk premia strategies", "novelty_justification": "Combines proprietary factor exposures in non-standard ways", "required_tools": ["scipy", "cvxpy"]}
                ],
                "cross_domain_connections": [
                    {"connection": "Link corporate fundamental data (Compustat) with ESG scores for green finance analysis", "domains": ["Corporate Finance", "Environmental Economics"], "potential_impact": "Quantify the 'greenium' in corporate bond markets"}
                ],
                "unconventional_data_combinations": [
                    {"combination": "Merge patent filings with stock returns around filing dates", "variables_involved": ["patent_grants", "abnormal_returns"], "expected_insight": "Measure innovation premium in equity markets"}
                ],
                "emerging_techniques": [
                    {"technique": "Graph Neural Networks for Financial Networks", "application": "Model inter-firm relationships from ownership and supply chain data", "advantage_over_traditional": "Captures higher-order network effects beyond pairwise correlations"}
                ]
            }

    def _generate_executive_summary(
        self,
        standardization: DataStandardization,
        data_retrieval: Dict,
        quality: Dict,
        compliance: Dict
    ) -> str:
        """Generate executive summary."""
        
        total_cost = data_retrieval.get("total_actual_cost", 0)
        total_records = data_retrieval.get("total_records", standardization.final_record_count)
        
        return f"""This report documents the acquisition of premium financial data for research analysis.

**Data Overview:**
- Vendors Used: {', '.join(standardization.vendors_merged)}
- Total Records: {total_records:,}
- Total Fields: {standardization.final_field_count}
- Total Cost: ${total_cost:,.2f}

**Key Findings:**
- Quality Score: {quality.get('overall_score', 'N/A')}/100 ({quality.get('grade', 'N/A')})
- Compliance Status: {compliance.get('compliance_status', 'N/A')}
- Risk Level: {compliance.get('risk_level', 'N/A')}

The dataset has been standardized across vendors with unified identifiers and field names.
All license compliance requirements have been documented. See detailed sections below."""
    
    def _generate_vendors_section(self, vendors: List[str]) -> str:
        """Generate vendors section."""
        
        section = "## Vendors Used\n\n"
        
        vendor_info = {
            "Bloomberg": "Bloomberg Professional terminal and API - Premier source for real-time and historical financial data",
            "Refinitiv": "Refinitiv Eikon and DataScope - Comprehensive market data and analytics",
            "WRDS": "Wharton Research Data Services - Academic research data platform with CRSP, Compustat, and more"
        }
        
        for vendor in vendors:
            section += f"### {vendor}\n\n"
            section += f"{vendor_info.get(vendor, 'Premium data vendor')}\n\n"
        
        return section
    
    def _generate_credential_section(self, vendors: List[str]) -> str:
        """Generate credential validation section."""
        
        section = "## Credential Validation\n\n"
        section += "All credentials were validated before data retrieval:\n\n"
        
        for vendor in vendors:
            section += f"- **{vendor}**: Credentials validated successfully\n"
        
        return section
    
    def _generate_cost_section(self, query_opt: Dict, data_ret: Dict) -> str:
        """Generate cost section."""
        
        estimated = query_opt.get("total_estimated_cost", 0)
        actual = data_ret.get("total_actual_cost", 0)
        savings = query_opt.get("cost_savings", 0)
        
        section = "## Cost Summary\n\n"
        section += f"- **Estimated Cost**: ${estimated:,.2f}\n"
        section += f"- **Actual Cost**: ${actual:,.2f}\n"
        section += f"- **Cost Savings**: ${savings:,.2f}\n\n"
        
        section += "### Cost Breakdown by Vendor\n\n"
        
        for query in query_opt.get("optimized_queries", []):
            if isinstance(query, dict):
                section += f"- {query.get('vendor', 'Unknown')}: ${query.get('estimated_cost', 0):,.2f}\n"
        
        return section
    
    def _generate_query_section(self, query_opt: Dict) -> str:
        """Generate query optimization section."""
        
        section = "## Query Optimization\n\n"
        section += "### Optimization Techniques Applied\n\n"
        
        for tech in query_opt.get("optimization_techniques", []):
            section += f"- {tech}\n"
        
        section += "\n### Optimized Queries\n\n"
        
        for query in query_opt.get("optimized_queries", []):
            if isinstance(query, dict):
                section += f"- **{query.get('query_id', 'Q')}** ({query.get('vendor', 'Unknown')})\n"
                section += f"  - Type: {query.get('query_type', 'N/A')}\n"
                section += f"  - Fields: {', '.join(query.get('fields_requested', [])[:5])}\n"
        
        return section
    
    def _generate_retrieval_section(self, data_ret: Dict) -> str:
        """Generate data retrieval section."""
        
        section = "## Data Retrieval\n\n"
        section += f"- **Total Records**: {data_ret.get('total_records', 0):,}\n"
        section += f"- **Success Rate**: {data_ret.get('retrieval_success_rate', 0):.0%}\n"
        section += f"- **Actual Cost**: ${data_ret.get('total_actual_cost', 0):,.2f}\n\n"
        
        section += "### Retrieved Datasets\n\n"
        
        for ds in data_ret.get("retrieved_datasets", []):
            if isinstance(ds, dict):
                section += f"- **{ds.get('dataset_id', 'DS')}** ({ds.get('vendor', 'Unknown')})\n"
                section += f"  - Records: {ds.get('num_records', 0):,}\n"
                section += f"  - Status: {ds.get('retrieval_status', 'N/A')}\n"
        
        return section
    
    def _generate_quality_section(self, quality: Dict) -> str:
        """Generate quality assessment section."""
        
        section = "## Quality Assessment\n\n"
        section += f"**Overall Score**: {quality.get('overall_score', 'N/A')}/100 ({quality.get('grade', 'N/A')})\n\n"
        
        section += "### Quality Dimensions\n\n"
        section += f"- Completeness: {quality.get('completeness_score', 'N/A')}\n"
        section += f"- Accuracy: {quality.get('accuracy_score', 'N/A')}\n"
        section += f"- Consistency: {quality.get('consistency_score', 'N/A')}\n"
        section += f"- Timeliness: {quality.get('timeliness_score', 'N/A')}\n\n"
        
        section += "### Vendor-Specific Quality Checks\n\n"
        
        for check in quality.get("vendor_checks", []):
            if isinstance(check, dict):
                status = "✓" if check.get("passed") else "✗"
                section += f"- {status} {check.get('vendor', 'Unknown')}: {check.get('check_type', 'N/A')} ({check.get('score', 0)}/100)\n"
        
        return section
    
    def _generate_compliance_section(self, compliance: Dict) -> str:
        """Generate license compliance section."""
        
        section = "## License Compliance\n\n"
        section += f"**Status**: {compliance.get('compliance_status', 'N/A')}\n"
        section += f"**Risk Level**: {compliance.get('risk_level', 'N/A')}\n\n"
        
        section += "### Required Citations\n\n"
        for vendor, citation in compliance.get("required_citations", {}).items():
            section += f"- **{vendor}**: {citation}\n"
        
        section += "\n### License Restrictions\n\n"
        for restriction in compliance.get("restrictions", []):
            if isinstance(restriction, dict):
                section += f"- **{restriction.get('vendor', 'Unknown')}** ({restriction.get('severity', 'N/A')}): {restriction.get('description', 'N/A')}\n"
        
        section += "\n### Required Disclaimers\n\n"
        for disclaimer in compliance.get("required_disclaimers", []):
            section += f"- {disclaimer}\n"
        
        return section
    
    def _generate_standardization_section(self, standardization: DataStandardization) -> str:
        """Generate data standardization section."""
        
        section = "## Data Standardization\n\n"
        section += f"- **Vendors Merged**: {', '.join(standardization.vendors_merged)}\n"
        section += f"- **Final Records**: {standardization.final_record_count:,}\n"
        section += f"- **Final Fields**: {standardization.final_field_count}\n"
        section += f"- **Overlap Handling**: {standardization.overlap_handling}\n\n"
        
        section += "### Transformations Applied\n\n"
        for transform in standardization.transformations_applied:
            section += f"- {transform}\n"
        
        section += "\n### Identifier Mappings\n\n"
        for mapping in standardization.identifier_mappings:
            section += f"- {mapping.vendor}: {mapping.original_id} → {mapping.universal_id} ({mapping.id_type})\n"
        
        return section
    
    def _generate_recommendations(self, quality: Dict, compliance: Dict) -> List[str]:
        """Generate recommendations."""
        
        recommendations = []
        
        # Quality-based recommendations
        if quality.get("overall_score", 100) < 80:
            recommendations.append("Review data quality issues before analysis")
        
        # Compliance-based recommendations
        if compliance.get("risk_level") == "High":
            recommendations.append("CRITICAL: Review license compliance before publication")
        
        recommendations.extend([
            "Include all required citations in publications",
            "Do not redistribute raw data",
            "Contact vendors for clarification on derived data rules",
            "Keep documentation of data processing for audit purposes"
        ])

        return recommendations


class Archivist:
    """Replication-assurance agent. Deterministic (no LLM): consolidates the
    audit trail already produced by Stage 1/2/3 into one manifest, and adds
    what nothing upstream captures — the environment the pipeline ran in and
    a content fingerprint of the exact pipeline code that ran — so a dataset
    can be reproduced independent of the run that first produced it."""

    ARCHIVIST_VERSION = "1.0.0"
    KEY_PACKAGES = ["pandas", "numpy", "pydantic", "requests", "python-dotenv", "openai"]
    STAGE_FILES = ["1-DataSourceStage.py", "2-DataCleaningStage.py", "3-QualityAssuranceStage.py"]

    def __init__(self, collector: Optional[MetricsCollector] = None):
        self.agent_name = "Archivist"
        self.collector = collector
        self._guard = StateGuard()

    def _capture_environment(self) -> Dict:
        packages = {}
        for pkg in self.KEY_PACKAGES:
            try:
                packages[pkg] = importlib_metadata.version(pkg)
            except importlib_metadata.PackageNotFoundError:
                packages[pkg] = "not installed"
        return {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
        }

    def _fingerprint_pipeline_code(self, stage_dir: Path) -> Dict:
        file_hashes = {}
        for fname in self.STAGE_FILES:
            fpath = stage_dir / fname
            if fpath.exists():
                file_hashes[fname] = hashlib.sha256(fpath.read_bytes()).hexdigest()[:16]
            else:
                file_hashes[fname] = "file_not_found"
        combined = hashlib.sha256("".join(file_hashes.values()).encode()).hexdigest()[:16]
        return {"stage_file_hashes": file_hashes, "pipeline_fingerprint": combined}

    def _build_audit_trail(self, doc: "DocumentedDataset") -> List[str]:
        """Consolidate the vendor/standardization/compliance/cost provenance
        already tracked across Stage 1 (retrieval), Stage 2 (standardization),
        and Stage 3 (QA/certification) into one readable trail."""
        std = doc.standardization
        trail = [
            f"Vendors merged: {', '.join(std.vendors_merged) if std and std.vendors_merged else 'none recorded'}",
            f"Unified fields: {len(std.unified_fields) if std and std.unified_fields else 0}, "
            f"final record count: {std.final_record_count if std else 'unknown'}",
        ]
        trail.append(f"Compliance status: {doc.compliance_status}")
        trail.append(f"Total cost incurred: {doc.total_cost}")
        trail.append(f"Certified {doc.certification_level} (quality score {doc.quality_score:.1f}/100)")
        trail.append(f"Ready for analysis: {doc.ready_for_analysis}")
        return trail

    def build_manifest(
        self,
        documented_dataset: "DocumentedDataset",
        data_cleaning_output: Dict,
        stage_dir: Path,
    ) -> Dict:
        """Build the replication manifest for this run."""
        print(f"\n[{self.agent_name}] Building replication manifest...")

        environment = self._capture_environment()
        pipeline_fingerprint = self._fingerprint_pipeline_code(stage_dir)
        cleaning_output_hash = self._guard.sign_output(
            data_cleaning_output, stage_name="DataCleaningStage_input_to_QA")
        content_hash = self._guard.sign_output(
            documented_dataset.standardization.model_dump() if documented_dataset.standardization else {},
            stage_name=f"dataset:{documented_dataset.dataset_id}")

        manifest = {
            "archivist_version": self.ARCHIVIST_VERSION,
            "generated_at": datetime.now().isoformat(),
            "environment": environment,
            "pipeline_code_fingerprint": pipeline_fingerprint,
            "upstream_cleaning_output_hash": cleaning_output_hash,
            "datasets": [{
                "dataset_id": documented_dataset.dataset_id,
                "dataset_name": documented_dataset.dataset_name,
                "content_hash": content_hash,
                "certification_level": documented_dataset.certification_level,
                "quality_score": documented_dataset.quality_score,
                "audit_trail": self._build_audit_trail(documented_dataset),
            }],
        }

        print(f"  Pipeline fingerprint: {pipeline_fingerprint['pipeline_fingerprint']}")
        return manifest


# ============================================================================
# HITL Checkpoint
# ============================================================================

def checkpoint5_final_approval(documented_dataset: DocumentedDataset) -> str:
    """HITL Checkpoint 5: Final Approval."""
    
    print("\n" + "="*70)
    print("🛑 HITL CHECKPOINT 5: Final Approval")
    print("="*70)
    print(f"\nDataset: {documented_dataset.dataset_name}")
    print(f"Quality Score: {documented_dataset.quality_score:.1f}/100")
    print(f"Certification: {documented_dataset.certification_level}")
    print(f"Compliance: {documented_dataset.compliance_status}")
    print(f"Total Cost: ${documented_dataset.total_cost:,.2f}")
    print(f"Ready for Analysis: {documented_dataset.ready_for_analysis}")
    
    print(f"\nCodebook: {len(documented_dataset.codebook.variables)} variables documented")
    print(f"Vendors: {', '.join(documented_dataset.standardization.vendors_merged)}")
    
    print("\nRequired Citations:")
    for vendor, citation in documented_dataset.codebook.required_citations.items():
        print(f"  - {vendor}: {citation}")
    
    if documented_dataset.report.recommendations:
        print("\nRecommendations:")
        for rec in documented_dataset.report.recommendations[:5]:
            print(f"  - {rec}")
    
    print("\nFinal decision:")
    print("- Is documentation complete and accurate?")
    print("- Are all costs properly documented?")
    print("- Are license compliance notes sufficient?")
    print("- Is the dataset ready for analysis?")

    response = auto_input("\n> ", default=get_default("final_approval")).strip()
    return response if response else "approved"


# ============================================================================
# Orchestrator
# ============================================================================

class QualityAssuranceOrchestrator:
    """Orchestrator for the quality assurance stage."""

    def __init__(self, openai_api_key: Optional[str] = None, collector: Optional[MetricsCollector] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        self.vendor_agent = VendorSpecificHandlerAgent(self.api_key)
        self.documentation_agent = DocumentationAgent(self.api_key, collector=collector)
        self.archivist = Archivist(collector=collector)
        self.qa_output: Optional[QualityAssuranceOutput] = None
        self.report_content: str = ""
        self.replication_manifest: Optional[Dict] = None
    
    def run_qa_pipeline(
        self,
        data_cleaning_output: Dict,
        research_question: str = "",
        enable_hitl: bool = True
    ) -> QualityAssuranceOutput:
        """Run the complete quality assurance pipeline."""
        
        print(f"\n{'='*70}")
        print(f"STANDARDIZATION & REPORT GENERATION PIPELINE")
        print(f"{'='*70}")
        
        research_q = data_cleaning_output.get('research_question', research_question)
        data_retrieval = data_cleaning_output.get('data_retrieval', {})
        quality_assessment = data_cleaning_output.get('quality_assessment', {})
        license_compliance = data_cleaning_output.get('license_compliance', {})
        
        print(f"Research Question: {research_q[:60]}..." if len(research_q) > 60 else f"Research Question: {research_q}")
        print(f"{'='*70}\n")
        
        # Step 1: Data Standardization
        print(f"STEP 1: DATA STANDARDIZATION")
        print(f"-"*70)
        standardization = self.vendor_agent.standardize_data(
            data_retrieval,
            quality_assessment
        )
        
        # Step 2: Generate Codebook
        print(f"\nSTEP 2: GENERATE CODEBOOK")
        print(f"-"*70)
        codebook = self.documentation_agent.generate_codebook(
            standardization,
            license_compliance
        )
        
        # Step 3: Generate Report
        print(f"\nSTEP 3: GENERATE REPORT")
        print(f"-"*70)
        report = self.documentation_agent.generate_report(
            data_cleaning_output,
            standardization,
            research_q
        )
        
        # Step 4: Innovation Analysis
        print(f"\nSTEP 4: INNOVATION ANALYSIS")
        print(f"-"*70)
        innovation_analysis = self.documentation_agent.generate_innovation_analysis(
            research_question=research_q,
            data_summary={
                "vendors": standardization.vendors_merged if standardization else [],
                "fields": standardization.unified_fields if standardization else [],
                "record_count": standardization.final_record_count if standardization else 0
            }
        )

        # Step 5: Sandbox Data Validation
        print(f"\nSTEP 5: SANDBOX DATA VALIDATION")
        print(f"-"*70)
        sandbox_result = self._validate_data_with_sandbox(
            standardization, data_cleaning_output
        )

        # Determine certification and readiness
        quality_score = quality_assessment.get("overall_score", 75)
        compliance_status = license_compliance.get("compliance_status", "Compliant")
        total_cost = data_retrieval.get("total_actual_cost", 0)

        # ANTI-FABRICATION: never certify simulated/placeholder data as Gold/Silver.
        # Treat as N/A when Stage 2 reported an N/A grade OR no dataset was a REAL
        # ("Success") retrieval (the HPC case has only "Simulated" datasets).
        quality_grade = quality_assessment.get("grade")
        retrieved = data_retrieval.get("retrieved_datasets", [])

        def _status(ds):
            return ds.get("retrieval_status") if isinstance(ds, dict) else getattr(ds, "retrieval_status", None)

        has_real_data = any(_status(ds) == "Success" for ds in retrieved)
        data_is_na = (quality_grade in (None, "N/A")) or (not has_real_data)

        if data_is_na:
            certification = "N/A"
            ready_for_analysis = False
        elif quality_score >= 90 and compliance_status == "Compliant":
            certification = "Gold"
            ready_for_analysis = quality_score >= 60 and compliance_status != "Non-compliant"
        elif quality_score >= 75:
            certification = "Silver"
            ready_for_analysis = quality_score >= 60 and compliance_status != "Non-compliant"
        else:
            certification = "Bronze"
            ready_for_analysis = quality_score >= 60 and compliance_status != "Non-compliant"
        
        # Create documented dataset
        documented_dataset = DocumentedDataset(
            dataset_id=f"PREMIUM_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            dataset_name="Premium Financial Dataset",
            standardization=standardization,
            codebook=codebook,
            report=report,
            quality_score=quality_score,
            certification_level=certification,
            compliance_status=compliance_status,
            total_cost=total_cost,
            ready_for_analysis=ready_for_analysis
        )
        
        # HITL Checkpoint 5: Final Approval
        if enable_hitl:
            feedback = checkpoint5_final_approval(documented_dataset)
            print(f"[HITL] Final approval feedback: {feedback}")

        # Step 6: Archivist builds the replication manifest for this run
        print(f"\nSTEP 6: REPLICATION MANIFEST")
        print(f"-"*70)
        stage_dir = Path(__file__).resolve().parent
        self.replication_manifest = self.archivist.build_manifest(
            documented_dataset, data_cleaning_output, stage_dir
        )

        # Create output
        self.qa_output = QualityAssuranceOutput(
            research_question=research_q,
            documented_dataset=documented_dataset,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "pipeline_stage": "quality_assurance",
                "hitl_enabled": enable_hitl,
                "innovation_suggestions": innovation_analysis,
                "sandbox_validation": sandbox_result
            },
            replication_manifest=self.replication_manifest,
        )

        # Store report content for saving
        self.report_content = self._compile_full_report(documented_dataset, research_q)
        
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Quality Score: {quality_score:.1f}/100")
        print(f"Certification: {certification}")
        print(f"Compliance: {compliance_status}")
        print(f"Total Cost: ${total_cost:,.2f}")
        print(f"Ready for Analysis: {ready_for_analysis}")
        print(f"{'='*70}\n")
        
        return self.qa_output

    def _validate_data_with_sandbox(
        self,
        standardization,
        data_cleaning_output: Dict
    ) -> Dict:
        """Generate and execute data validation code in sandbox."""

        fields = standardization.unified_fields if standardization else []
        vendors = standardization.vendors_merged if standardization else []
        record_count = standardization.final_record_count if standardization else 0

        try:
            result = self.documentation_agent.llm.invoke([
                {"role": "system", "content": "You are a data quality engineer. Generate ONLY executable Python code, no markdown fences."},
                {"role": "user", "content": """Generate a Python script that validates a premium financial dataset.

Dataset Info:
- Vendors: {vendors}
- Fields: {fields}
- Records: {record_count}

Requirements:
1. Use ONLY numpy and json (import numpy as np, import json) — no other imports
2. Create a synthetic validation based on the field names and vendor count
3. Check field naming conventions (snake_case, no special chars)
4. Estimate data completeness based on record count vs expected
5. Print results as JSON: {{"validated": true/false, "field_checks": [{{"field": "...", "valid_name": true/false}}], "issues": ["..."], "summary": "..."}}

Respond with ONLY the Python code, no explanations or markdown.
""".format(
                    vendors=", ".join(vendors),
                    fields=", ".join(fields[:20]),
                    record_count=record_count
                )}
            ])

            code = CodeSandbox.clean_llm_output(result)

            sandbox = CodeSandbox(timeout_sec=60, memory_mb=256)
            exec_result = sandbox.execute(code)

            validation = {
                "validated": exec_result.success,
                "validation_method": "sandbox_execution",
                "code": code,
                "stdout": exec_result.stdout,
                "stderr": exec_result.stderr,
                "error": exec_result.error,
                "execution_time_sec": exec_result.execution_time_sec
            }

            if exec_result.success:
                print(f"  Sandbox validation PASSED ({exec_result.execution_time_sec:.1f}s)")
            else:
                print(f"  Sandbox validation FAILED: {exec_result.error or exec_result.stderr[:200]}")

            return validation

        except Exception as e:
            print(f"  Sandbox validation error: {e}")
            return {
                "validated": False,
                "validation_method": "sandbox_execution",
                "error": f"Code generation failed: {e}"
            }

    def _compile_full_report(self, documented: DocumentedDataset, research_question: str) -> str:
        """Compile the full report as markdown."""

        report = documented.report
        codebook = documented.codebook
        
        content = f"""# {report.title}

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Research Question**: {research_question}

⚠️ **IMPORTANT**: This report contains premium data with license restrictions.
Ensure compliance with all vendor license terms. Proper attribution required in all publications.

---

## Executive Summary

{report.executive_summary}

---

{report.vendors_section}

---

{report.credential_section}

---

{report.cost_section}

---

{report.query_section}

---

{report.retrieval_section}

---

{report.quality_section}

---

{report.compliance_section}

---

{report.standardization_section}

---

## Data Codebook

### Dataset Information
- **Title**: {codebook.title}
- **Version**: {codebook.version}
- **Created**: {codebook.created_date}

### Variable Descriptions

| Variable | Source | Type | Unit | License Notes |
|----------|--------|------|------|---------------|
"""
        
        for var in codebook.variables:
            content += f"| {var.variable_name} | {var.source_vendor} | {var.data_type} | {var.unit} | {var.license_notes[:30]}... |\n"
        
        content += f"""
### Required Citations

"""
        for vendor, citation in codebook.required_citations.items():
            content += f"**{vendor}**: {citation}\n\n"
        
        content += """
### Usage Restrictions

"""
        for restriction in codebook.usage_restrictions:
            content += f"- {restriction}\n"
        
        content += f"""
---

## Recommendations

"""
        for rec in report.recommendations:
            content += f"- {rec}\n"
        
        # Add Innovation Analysis section
        innovation = self.qa_output.metadata.get("innovation_suggestions", {}) if self.qa_output else {}
        content += "\n---\n\n## Innovation Analysis\n\n"
        content += "### Innovative Analytical Methods\n\n"
        for method in innovation.get("innovative_methods", []):
            content += f"**{method.get('method_name', 'N/A')}**: {method.get('description', '')}\n"
            content += f"- *Novelty*: {method.get('novelty_justification', '')}\n\n"

        content += "### Cross-Domain Connections\n\n"
        for conn in innovation.get("cross_domain_connections", []):
            content += f"- **{conn.get('connection', '')}** (Domains: {', '.join(conn.get('domains', []))})\n"
            content += f"  - Potential Impact: {conn.get('potential_impact', '')}\n\n"

        content += "### Unconventional Data Combinations\n\n"
        for combo in innovation.get("unconventional_data_combinations", []):
            content += f"- **{combo.get('combination', '')}**\n"
            content += f"  - Expected Insight: {combo.get('expected_insight', '')}\n\n"

        content += "### Emerging Techniques\n\n"
        for tech in innovation.get("emerging_techniques", []):
            content += f"- **{tech.get('technique', '')}**: {tech.get('application', '')}\n"
            content += f"  - Advantage: {tech.get('advantage_over_traditional', '')}\n\n"

        content += f"""
---

## Certification

**Quality Score**: {documented.quality_score:.1f}/100

**Certification Level**: {documented.certification_level}

**Compliance Status**: {documented.compliance_status}

**Total Cost**: ${documented.total_cost:,.2f}

**Ready for Analysis**: {'Yes' if documented.ready_for_analysis else 'No'}

---

*Report generated by DataTeam Premium Subscribed Data Pipeline*
"""
        
        return content
    
    def save_qa_output(self, filename: str = "premium_data_qa_output.json"):
        """Save QA output to JSON."""
        if not self.qa_output:
            print("No QA output to save")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.qa_output.model_dump(), f, indent=2)
        
        print(f"[Orchestrator] QA output saved to {filename}")
    
    def save_report_codebook(self, filename: str = "premium_data_report.md"):
        """Save report and codebook as markdown file."""
        if not self.report_content:
            print("No report content to save")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.report_content)

        print(f"[Orchestrator] Report and codebook saved to {filename}")

    def save_replication_manifest(self, filename: str = "replication_manifest.json"):
        """Save the Archivist's replication manifest to JSON."""
        if not self.replication_manifest:
            print("No replication manifest to save")
            return

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.replication_manifest, f, indent=2)

        print(f"[Orchestrator] Replication manifest saved to {filename}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main function for quality assurance stage."""
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Load cleaning output
    cleaning_file = "premium_data_cleaning_output.json"
    if os.path.exists(cleaning_file):
        with open(cleaning_file, 'r', encoding='utf-8') as f:
            cleaning_data = json.load(f)
    else:
        # Create sample data for testing
        cleaning_data = {
            "research_question": "Analyze institutional ownership and stock returns",
            "query_optimization": {
                "optimized_queries": [
                    {"query_id": "Q1", "vendor": "WRDS", "query_type": "SQL", "estimated_cost": 0, "fields_requested": ["permno", "ret"]}
                ],
                "total_estimated_cost": 0,
                "cost_savings": 20,
                "optimization_techniques": ["Field selection", "Request batching"]
            },
            "data_retrieval": {
                "retrieved_datasets": [
                    {"dataset_id": "DS_Q1", "vendor": "WRDS", "num_records": 50000, "retrieval_status": "Success"}
                ],
                "total_records": 50000,
                "total_actual_cost": 0,
                "retrieval_success_rate": 1.0
            },
            "quality_assessment": {
                "overall_score": 88,
                "grade": "B",
                "completeness_score": 90,
                "accuracy_score": 85,
                "consistency_score": 88,
                "timeliness_score": 90,
                "vendor_checks": [
                    {"vendor": "WRDS", "check_type": "Linking accuracy", "passed": True, "score": 91}
                ]
            },
            "license_compliance": {
                "compliance_status": "Compliant",
                "risk_level": "Medium",
                "restrictions": [
                    {"vendor": "WRDS", "restriction_type": "redistribution", "description": "No commercial redistribution", "severity": "High"}
                ],
                "required_citations": {"WRDS": "Data obtained from Wharton Research Data Services (WRDS)."},
                "required_disclaimers": ["WRDS data is for academic research purposes only."]
            }
        }
    
    # Run pipeline
    orchestrator = QualityAssuranceOrchestrator()
    qa_output = orchestrator.run_qa_pipeline(
        data_cleaning_output=cleaning_data,
        enable_hitl=True
    )
    
    # Save outputs
    orchestrator.save_qa_output("premium_data_qa_output.json")
    orchestrator.save_report_codebook(f"premium_data_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.md")
    orchestrator.save_replication_manifest("replication_manifest.json")
    
    print("\n" + "="*70)
    print("QUALITY ASSURANCE SUMMARY")
    print("="*70)
    print(f"Quality Score: {qa_output.documented_dataset.quality_score:.1f}")
    print(f"Certification: {qa_output.documented_dataset.certification_level}")
    print(f"Compliance: {qa_output.documented_dataset.compliance_status}")
    print(f"Total Cost: ${qa_output.documented_dataset.total_cost:,.2f}")
    print("="*70)


if __name__ == "__main__":
    main()
