# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Gap Detection Stage - Automated Mode (No Firecrawl, No HITL)
This script analyzes literature from Stage 1 to detect research gaps and construct knowledge graphs.

Pipeline:
1. PaperDecomposer: Structural analysis of papers (methods, findings, limitations)
2. GapFinder: Research opportunity detection (methodological, theoretical, empirical gaps)
3. KnowledgeWeaver: Graph construction (relationships between papers, concepts, gaps)

Input: Literature batch from Stage 1 (literature_batch.json)
Output: Analyzed literature with research gaps and knowledge graph
"""
from __future__ import annotations

import os
import json
import re
import tempfile
import sys
from pathlib import Path as _Path
from typing import List, Dict, Optional, Set, Tuple, Literal
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
from enum import Enum

# Add agents dir to path for shared imports
_agents_dir = _Path(__file__).resolve().parent.parent.parent.parent
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))
from shared.llm import LLMClient
from shared.observability import MetricsCollector
from shared.tools.tool_registry import ToolRegistry
from shared.auto_input import auto_input, get_default
from LiteratureTeam.ael.schemas.stage_outputs import (
    PaperStructure, ResearchGap, KnowledgeNode, KnowledgeEdge,
    KnowledgeGraph, GapAnalysisResult,
)

# Load environment variables
load_dotenv()

# Try to import PDF parsing libraries
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

PDF_PARSING_AVAILABLE = PYMUPDF_AVAILABLE or PDFPLUMBER_AVAILABLE

class DecompositionMode(str, Enum):
    """Mode for paper decomposition."""
    METADATA = "metadata"  # Default: analyze title, authors, abstract only
    DEEP = "deep"  # On-demand: download and parse full PDF content

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
    import re
    
    # First, try to fix common LaTeX/math escape issues
    # Replace \ast, \mathbb, etc. with escaped versions
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

def safe_create_research_gap(gap_data: dict, gap_type: str, index: int) -> Optional[ResearchGap]:
    """Safely create ResearchGap with defaults for missing fields."""
    # Coerce LLM list-fields that sometimes arrive as a string -> list, so we keep the
    # real grounding instead of falling back to the ["Multiple papers"] placeholder.
    for _f in ("evidence_papers", "related_trends", "suggested_approaches"):
        _v = gap_data.get(_f)
        if isinstance(_v, str):
            _s = _v.strip()
            gap_data[_f] = [x.strip() for x in re.split(r"[;\n|]+", _s) if x.strip()] if _s else []
    try:
        return ResearchGap(**gap_data)
    except Exception:
        gap_id_prefix = {"methodological": "METH", "theoretical": "THEO", "empirical": "EMP", "data": "DATA"}.get(gap_type, "GAP")
        defaults = {
            "gap_id": gap_data.get("gap_id", f"{gap_id_prefix}_GAP_{index:03d}"),
            "gap_type": gap_data.get("gap_type", gap_type),
            "gap_title": gap_data.get("gap_title", gap_data.get("title", "Untitled Gap")),
            "gap_description": gap_data.get("gap_description", gap_data.get("description", "No description provided")),
            "evidence_papers": gap_data.get("evidence_papers", ["Multiple papers"]),
            "severity": gap_data.get("severity", "Medium"),
            "addressability": gap_data.get("addressability", "Moderate"),
            "related_trends": gap_data.get("related_trends", []),
            "potential_impact": gap_data.get("potential_impact", "Impact not specified"),
            "suggested_approaches": gap_data.get("suggested_approaches", ["Further research needed"]),
        }
        try:
            return ResearchGap(**defaults)
        except Exception as e:
            print(f"    Warning: Could not create gap even with defaults: {e}")
            return None

def create_paper_structure_from_dict(data: dict) -> Optional[PaperStructure]:
    """
    Create a PaperStructure from a dictionary with type conversion.
    Handles cases where LLM returns strings instead of lists.
    Supports both metadata and deep analysis fields.

    Args:
        data: Dictionary with paper structure information

    Returns:
        PaperStructure object or None if creation fails
    """
    def ensure_list(value) -> List[str]:
        """Convert value to list if it's a string."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            # Split by common separators if it looks like a list in string form
            if ',' in value:
                return [item.strip() for item in value.split(',') if item.strip()]
            elif ';' in value:
                return [item.strip() for item in value.split(';') if item.strip()]
            else:
                return [value] if value.strip() else []
        return []

    def ensure_list_of_dicts(value) -> List[Dict]:
        """Convert value to list of dicts."""
        if value is None:
            return []
        if isinstance(value, list):
            return [v for v in value if isinstance(v, dict)]
        if isinstance(value, dict):
            return [value]
        return []

    try:
        # Convert all list fields (basic fields)
        cleaned_data = {
            "paper_title": data.get("paper_title", "Unknown"),
            "research_objectives": ensure_list(data.get("research_objectives")),
            "methodologies": ensure_list(data.get("methodologies")),
            "key_findings": ensure_list(data.get("key_findings")),
            "theoretical_contributions": ensure_list(data.get("theoretical_contributions")),
            "limitations": ensure_list(data.get("limitations")),
            "future_work_suggestions": ensure_list(data.get("future_work_suggestions")),
            "data_sources": ensure_list(data.get("data_sources")),
            "assumptions": ensure_list(data.get("assumptions")),
        }

        # Deep analysis fields (optional)
        cleaned_data["decomposition_mode"] = data.get("decomposition_mode", "metadata")
        cleaned_data["full_text_available"] = data.get("full_text_available", False)

        # Dict fields
        if data.get("methodology_details"):
            cleaned_data["methodology_details"] = data["methodology_details"] if isinstance(
                data["methodology_details"], dict) else None
        if data.get("data_description"):
            cleaned_data["data_description"] = data["data_description"] if isinstance(
                data["data_description"], dict) else None

        # String fields
        if data.get("empirical_strategy"):
            cleaned_data["empirical_strategy"] = str(data["empirical_strategy"])
        if data.get("sample_description"):
            cleaned_data["sample_description"] = str(data["sample_description"])

        # List fields
        cleaned_data["model_equations"] = ensure_list(data.get("model_equations"))
        cleaned_data["robustness_checks"] = ensure_list(data.get("robustness_checks"))
        cleaned_data["sections_extracted"] = ensure_list(data.get("sections_extracted"))
        cleaned_data["variable_definitions"] = ensure_list_of_dicts(data.get("variable_definitions"))
        cleaned_data["domain_tags"] = ensure_list(data.get("domain_tags"))

        return PaperStructure(**cleaned_data)
    except Exception as e:
        print(f"    Warning: Failed to create PaperStructure: {e}")
        return None

def create_research_gap_from_dict(gap_data: dict, gap_type: str = "methodological", gap_counter: int = 0) -> Optional[ResearchGap]:
    """
    Create a ResearchGap from a dictionary with fallback values for missing fields.
    
    Args:
        gap_data: Dictionary with gap information (may be incomplete)
        gap_type: Default gap type if not provided
        gap_counter: Counter for generating gap_id if missing
        
    Returns:
        ResearchGap object or None if creation fails
    """
    try:
        # Extract or generate required fields
        gap_id = gap_data.get('gap_id', f"{gap_type.upper()}_GAP_{gap_counter:03d}")
        gap_type_val = gap_data.get('gap_type', gap_type)
        
        # Handle description field (might be 'description' or 'gap_description')
        description = gap_data.get('gap_description') or gap_data.get('description', '')
        if not description:
            # Try to create description from other fields
            description = gap_data.get('gap_title', '') or gap_data.get('title', '')
        
        # Extract title (might be 'title' or 'gap_title')
        title = gap_data.get('gap_title') or gap_data.get('title', 'Unnamed Gap')
        if title == 'Unnamed Gap' and description:
            # Use first part of description as title
            title = description[:60].rstrip('.') + ('...' if len(description) > 60 else '')
        
        # Create ResearchGap with all fields
        gap = ResearchGap(
            gap_id=gap_id,
            gap_type=gap_type_val,
            gap_title=title,
            gap_description=description,
            evidence_papers=gap_data.get('evidence_papers', gap_data.get('papers', [])),
            severity=gap_data.get('severity', 'Medium'),
            addressability=gap_data.get('addressability', 'Moderate'),
            related_trends=gap_data.get('related_trends', []),
            potential_impact=gap_data.get('potential_impact', 'Moderate impact expected'),
            suggested_approaches=gap_data.get('suggested_approaches', gap_data.get('approaches', []))
        )
        return gap
    except Exception as e:
        print(f"    Warning: Failed to create ResearchGap from data: {e}")
        return None

# ========== AGENTS ==========

class PaperDecomposer:
    """
    Agent for structural analysis of papers with dual-mode capability.

    Modes:
    - METADATA (default): Analyze title, authors, year, and abstract using LLM
    - DEEP (on-demand): Download and parse full PDF for detailed methodological analysis

    Deep mode is useful when:
    - Comparing similar economic studies
    - Analyzing detailed methodological approaches
    - Extracting equations, variable definitions, and empirical strategies
    """

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "PaperDecomposer"
        self.api_key = openai_api_key
        self.collector = collector
        self.llm = LLMClient(


            temperature=0.3,

            api_key=self.api_key,

            collector=collector,

            agent_name=self.agent_name

        )
        self.llm_deep = LLMClient(


            temperature=0.2,

            api_key=self.api_key,

            collector=collector,

            agent_name=self.agent_name

        )
        self._pdf_cache: Dict[str, str] = {}  # Cache downloaded PDFs

    def decompose_papers(
        self,
        literature_batch: Dict,
        mode: DecompositionMode = DecompositionMode.METADATA,
        deep_analysis_paper_ids: Optional[List[str]] = None
    ) -> List[PaperStructure]:
        """
        Decompose papers into structural components.

        Args:
            literature_batch: Literature data from Stage 1
            mode: Default decomposition mode for all papers
            deep_analysis_paper_ids: List of paper IDs/titles for deep analysis
                                    (overrides mode for specific papers)

        Returns:
            List of PaperStructure objects
        """
        literature_items = literature_batch.get('literature_items', [])
        deep_paper_set = set(deep_analysis_paper_ids or [])

        print(f"\n[{self.agent_name}] Decomposing {len(literature_items)} papers...")
        print(f"  Default mode: {mode.value}")
        if deep_paper_set:
            print(f"  Papers for deep analysis: {len(deep_paper_set)}")
        if mode == DecompositionMode.DEEP or deep_paper_set:
            print(f"  PDF parsing available: {PDF_PARSING_AVAILABLE}")
            if PYMUPDF_AVAILABLE:
                print(f"    - PyMuPDF: Available")
            if PDFPLUMBER_AVAILABLE:
                print(f"    - pdfplumber: Available")

        # Decompose all papers CONCURRENTLY — each _decompose_single_paper is an independent
        # LLM call; vLLM batches concurrent requests (~7.5x at concurrency 8) for ~the same
        # wall-time, so this replaces the sequential per-paper loop.
        def _decompose_one(paper):
            paper_title = paper.get('title', '')
            paper_id = paper.get('paperId', paper.get('id', ''))
            paper_mode = mode
            if paper_title in deep_paper_set or paper_id in deep_paper_set:
                paper_mode = DecompositionMode.DEEP
            return self._decompose_single_paper(paper, paper_mode)

        print(f"  Decomposing {len(literature_items)} papers (concurrent)...")
        from shared.parallel import parallel_map
        _results = parallel_map(_decompose_one, literature_items)
        paper_structures = [s for s in _results if s and not isinstance(s, Exception)]

        # Summary
        metadata_count = sum(1 for s in paper_structures if s.decomposition_mode == "metadata")
        deep_count = sum(1 for s in paper_structures if s.decomposition_mode == "deep")

        print(f"[{self.agent_name}] Decomposed {len(paper_structures)} papers")
        print(f"  - Metadata mode: {metadata_count}")
        print(f"  - Deep mode: {deep_count}")

        return paper_structures

    def _decompose_single_paper(
        self,
        paper: Dict,
        mode: DecompositionMode = DecompositionMode.METADATA
    ) -> Optional[PaperStructure]:
        """Decompose a single paper using specified mode."""

        if mode == DecompositionMode.DEEP:
            return self._decompose_deep(paper)
        else:
            return self._decompose_metadata(paper)

    def _decompose_metadata(self, paper: Dict) -> Optional[PaperStructure]:
        """Decompose a single paper using metadata only (default mode)."""
        try:

            result = self.llm.format_and_invoke(

                system_prompt="""You are an expert economics research analyst specializing in structural decomposition of academic papers.
    You excel at identifying the precise methodological choices, theoretical foundations, and empirical strategies used in economics research.""",

                user_prompt="""Analyze the following economics paper and extract its structural components with precision.

                Title: {title}
                Authors: {authors}
                Year: {year}
                Abstract: {abstract}

                Extract the following (be SPECIFIC and DETAILED, not generic):
                - research_objectives: Main research objectives (2-4 points, state the SPECIFIC economic question)
                - methodologies: SPECIFIC methodologies used (e.g., "panel data regression with firm fixed effects", "Bayesian DSGE estimation", NOT just "empirical analysis")
                - key_findings: Key findings WITH quantitative results if mentioned (3-5 points)
                - theoretical_contributions: Theoretical contributions to economics literature (1-3 points)
                - limitations: Stated or implied limitations (2-4 points, be specific about what restricts generalizability)
                - future_work_suggestions: Suggested future work (1-3 points)
                - data_sources: SPECIFIC data sources (e.g., "FRED quarterly GDP data 2000-2020", NOT just "macroeconomic data")
                - assumptions: Key economic assumptions (e.g., "rational expectations", "Cobb-Douglas production function")
                - domain_tags: 2-4 research domain labels for this paper (e.g., ["macroeconomics", "monetary policy", "DSGE modeling"]). Use specific economics subfields and methodological domains. Examples: macroeconomics, microeconomics, labor economics, trade economics, behavioral economics, financial economics, development economics, public economics, health economics, environmental economics, econometrics, machine learning, agent-based modeling, game theory, network economics, computational economics.

                Return as a JSON object with these fields.
                Example: {{
              "paper_title": "...",
              "research_objectives": ["obj1", "obj2"],
              "methodologies": ["method1", "method2"],
              ...
                }}

                Respond with ONLY the JSON object, no other text.
                """,

                variables={
                "title": paper.get('title', 'Unknown'),
                "authors": ', '.join(paper.get('authors', [])[:3]),
                "year": paper.get('year', 'N/A'),
                "abstract": paper.get('abstract', 'No abstract available')[:500]
                }

            )

            if not result:
                print(f"    Warning: Empty response for paper '{paper.get('title', 'Unknown')[:50]}...'")
                return None

            # Use safe JSON parsing with escape sequence handling
            structure_data = safe_parse_json(result, "paper structure")
            if structure_data is None:
                return None

            # Add mode indicator
            structure_data['decomposition_mode'] = 'metadata'
            structure_data['full_text_available'] = False

            # Use helper function to create PaperStructure with type conversion
            structure = create_paper_structure_from_dict(structure_data)

            return structure

        except Exception as e:
            print(f"    Error decomposing paper '{paper.get('title', 'Unknown')[:50]}...': {e}")
            return None

    def _decompose_deep(self, paper: Dict) -> Optional[PaperStructure]:
        """
        Deep decomposition: Download PDF and extract detailed content.

        This mode provides:
        - Full methodology breakdown
        - Variable definitions and data descriptions
        - Model equations and empirical strategies
        - Robustness checks performed
        """
        paper_title = paper.get('title', 'Unknown')
        print(f"    [Deep Mode] Analyzing: {paper_title[:60]}...")

        # Try to get PDF content
        pdf_text = self._get_pdf_content(paper)

        if pdf_text:
            return self._analyze_full_text(paper, pdf_text)
        else:
            # Fall back to enhanced metadata analysis
            print(f"    [Deep Mode] PDF not available, using enhanced metadata analysis")
            return self._analyze_enhanced_metadata(paper)

    def _get_pdf_content(self, paper: Dict) -> Optional[str]:
        """Attempt to download and extract text from paper PDF."""

        if not PDF_PARSING_AVAILABLE:
            print(f"      PDF parsing libraries not available (install PyMuPDF or pdfplumber)")
            return None

        # Try different PDF sources
        pdf_url = None

        # 1. Check for openAccessPdf from Semantic Scholar
        if 'openAccessPdf' in paper and paper['openAccessPdf']:
            pdf_url = paper['openAccessPdf'].get('url')

        # 2. Check for arXiv link
        if not pdf_url and 'url' in paper:
            url = paper['url']
            if 'arxiv.org/abs/' in url:
                arxiv_id = url.split('arxiv.org/abs/')[-1]
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            elif 'arxiv.org/pdf/' in url:
                pdf_url = url

        # 3. Check for direct PDF link
        if not pdf_url and 'pdf_url' in paper:
            pdf_url = paper['pdf_url']

        if not pdf_url:
            print(f"      No PDF URL found for paper")
            return None

        # Check cache
        if pdf_url in self._pdf_cache:
            return self._pdf_cache[pdf_url]

        # Download PDF
        try:
            print(f"      Downloading PDF from: {pdf_url[:80]}...")
            _tool_result = ToolRegistry.invoke("web_fetch", {
                "url": pdf_url,
                "params": {},
                "headers": {"User-Agent": "Mozilla/5.0 (Research Bot)"},
                "retries": 0,
            }, collector=self.collector, agent=self.agent_name)

            if not _tool_result.success:
                print(f"      Failed to download PDF: {_tool_result.error}")
                return None

            status_code = _tool_result.data.get("status_code", 0)
            if status_code != 200:
                print(f"      Failed to download PDF: HTTP {status_code}")
                return None

            # web_fetch returns text/JSON; for PDF binary we need raw bytes
            pdf_data = _tool_result.data.get("data", "")
            if not pdf_data or not isinstance(pdf_data, str) or len(pdf_data) < 100:
                print(f"      PDF content too small or unavailable via web_fetch")
                return None

            # Save to temp file and extract text
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False, mode='w', encoding='utf-8') as tmp_file:
                tmp_file.write(pdf_data)
                tmp_path = tmp_file.name

            try:
                pdf_text = self._extract_pdf_text(tmp_path)
                if pdf_text:
                    self._pdf_cache[pdf_url] = pdf_text
                return pdf_text
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        except Exception as e:
            print(f"      Error downloading PDF: {e}")
            return None

    def _extract_pdf_text(self, pdf_path: str) -> Optional[str]:
        """Extract text from PDF file."""

        try:
            if PYMUPDF_AVAILABLE:
                # Use PyMuPDF (faster and more accurate)
                doc = fitz.open(pdf_path)
                text_parts = []
                for page in doc:
                    text_parts.append(page.get_text())
                doc.close()
                text = "\n".join(text_parts)

            elif PDFPLUMBER_AVAILABLE:
                # Use pdfplumber as fallback
                with pdfplumber.open(pdf_path) as pdf:
                    text_parts = []
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    text = "\n".join(text_parts)
            else:
                return None

            # Clean up text
            text = text.strip()
            if len(text) < 500:
                print(f"      PDF text too short ({len(text)} chars), may be image-based")
                return None

            print(f"      Extracted {len(text):,} characters from PDF")
            return text

        except Exception as e:
            print(f"      Error extracting PDF text: {e}")
            return None

    def _analyze_full_text(self, paper: Dict, pdf_text: str) -> Optional[PaperStructure]:
        """Analyze paper using full PDF text for deep decomposition."""

        # Truncate text for LLM (keep most relevant sections)
        max_chars = 25000
        if len(pdf_text) > max_chars:
            # Try to keep introduction, methodology, and conclusion
            pdf_text = self._extract_key_sections(pdf_text, max_chars)
        paper_title = paper.get('title', 'Unknown')

        try:

            result = self.llm.format_and_invoke(

                system_prompt="""You are an expert economics researcher analyzing academic papers in detail.
    Extract comprehensive structural information for methodological comparison and replication.
    Return ONLY a flat JSON object (not nested) with no markdown formatting.""",

                user_prompt="""Analyze this economics paper and extract ALL fields into a FLAT JSON object.

    Title: {title}
    Authors: {authors}
    Year: {year}

    FULL TEXT:
    {full_text}

    Return a FLAT JSON object with these exact field names at the top level:

    {{
      "paper_title": "exact title string",
      "research_objectives": ["objective 1", "objective 2"],
      "methodologies": ["specific method 1", "specific method 2"],
      "key_findings": ["finding 1 with numbers if available", "finding 2"],
      "theoretical_contributions": ["contribution 1"],
      "limitations": ["limitation 1", "limitation 2"],
      "future_work_suggestions": ["suggestion 1"],
      "data_sources": ["specific source like CRSP 1990-2020"],
      "assumptions": ["key assumption 1"],
      "methodology_details": {{
        "estimation_method": "e.g., OLS, GMM, Maximum Likelihood",
        "identification_strategy": "how causality is established",
        "control_variables": ["control 1", "control 2"],
        "fixed_effects": "e.g., firm and year fixed effects",
        "clustering": "e.g., clustered at firm level"
      }},
      "data_description": {{
        "sample_period": "e.g., 1990-2020",
        "sample_size": "number of observations",
        "unit_of_analysis": "e.g., firm-year",
        "key_variables": ["dependent var", "independent vars"]
      }},
      "model_equations": ["equation 1 in LaTeX or text"],
      "empirical_strategy": "detailed description of empirical approach",
      "robustness_checks": ["check 1", "check 2"],
      "variable_definitions": [{{"name": "var1", "definition": "def1"}}],
      "sample_description": "description of sample selection",
      "sections_extracted": ["Introduction", "Data", "Methodology", "Results"],
      "domain_tags": ["economics subfield 1", "methodology domain", "application area"]
    }}

    IMPORTANT: Return ONLY the JSON object. No markdown code blocks. No explanatory text.
    Do NOT use nested structures like "BASIC STRUCTURE" or "DEEP ANALYSIS".
    All fields must be at the top level of the JSON object.""",

                variables={
                "title": paper_title,
                "authors": ', '.join(paper.get('authors', [])[:5]),
                "year": paper.get('year', 'N/A'),
                "full_text": pdf_text
                }

            )

            if not result:
                print(f"      Warning: Empty response from deep analysis")
                return self._decompose_metadata(paper)

            structure_data = safe_parse_json(result, "deep paper structure")

            # Handle nested structure if LLM still returns it
            if structure_data and isinstance(structure_data, dict):
                # Check for nested "BASIC STRUCTURE" or similar keys
                for key in ['BASIC STRUCTURE', 'BASIC_STRUCTURE', 'basic_structure']:
                    if key in structure_data:
                        basic = structure_data[key]
                        # Merge basic structure into top level
                        if isinstance(basic, dict):
                            for k, v in basic.items():
                                if k not in structure_data:
                                    structure_data[k] = v
                        del structure_data[key]

                for key in ['DEEP ANALYSIS', 'DEEP_ANALYSIS', 'deep_analysis']:
                    if key in structure_data:
                        deep = structure_data[key]
                        # Merge deep analysis into top level
                        if isinstance(deep, dict):
                            for k, v in deep.items():
                                if k not in structure_data:
                                    structure_data[k] = v
                        del structure_data[key]

            if structure_data is None:
                return self._decompose_metadata(paper)

            # Ensure paper_title is set from the original paper
            if not structure_data.get('paper_title') or structure_data.get('paper_title') == 'Unknown':
                structure_data['paper_title'] = paper_title

            # Add mode indicators
            structure_data['decomposition_mode'] = 'deep'
            structure_data['full_text_available'] = True

            structure = create_paper_structure_from_dict(structure_data)
            return structure

        except Exception as e:
            print(f"      Error in deep analysis: {e}")
            return self._decompose_metadata(paper)

    def _analyze_enhanced_metadata(self, paper: Dict) -> Optional[PaperStructure]:
        """Enhanced metadata analysis when PDF is not available."""
        try:

            result = self.llm.format_and_invoke(

                system_prompt="""You are an expert economics researcher. Analyze this paper based on available metadata
                and infer as much methodological detail as possible from the abstract.""",

                user_prompt="""Analyze the following economics paper and extract detailed structural information.
                Based on the abstract, infer methodological details commonly used in similar studies.

                Title: {title}
                Authors: {authors}
                Year: {year}
                Abstract: {abstract}
                Venue: {venue}

                Extract:
                1. Basic fields: paper_title, research_objectives, methodologies, key_findings,
               theoretical_contributions, limitations, future_work_suggestions, data_sources, assumptions

                2. Inferred methodology details (best guess based on paper topic and abstract):
               - methodology_details: JSON with estimation_method, identification_strategy, etc.
               - empirical_strategy: Inferred empirical approach
               - model_equations: Any equations mentioned or implied

                3. domain_tags: 2-4 research domain labels for this paper (e.g., ["macroeconomics", "machine learning", "behavioral economics"]). Use specific economics subfields and methodological domains.

                Return as a JSON object. Mark inferred fields clearly.
                Respond with ONLY the JSON object, no other text.
                """,

                variables={
                "title": paper.get('title', 'Unknown'),
                "authors": ', '.join(paper.get('authors', [])[:3]),
                "year": paper.get('year', 'N/A'),
                "abstract": paper.get('abstract', 'No abstract available'),
                "venue": paper.get('venue', paper.get('journal', 'N/A'))
                }

            )

            if not result:
                return self._decompose_metadata(paper)

            structure_data = safe_parse_json(result, "enhanced paper structure")
            if structure_data is None:
                return self._decompose_metadata(paper)

            structure_data['decomposition_mode'] = 'deep'
            structure_data['full_text_available'] = False

            structure = create_paper_structure_from_dict(structure_data)
            return structure

        except Exception as e:
            print(f"      Error in enhanced metadata analysis: {e}")
            return self._decompose_metadata(paper)

    def _extract_key_sections(self, text: str, max_chars: int) -> str:
        """Extract key sections from PDF text to fit within character limit."""

        # Common section headers in economics papers
        section_patterns = [
            r'(?i)(abstract|introduction|literature\s+review|data|methodology|'
            r'empirical\s+strategy|model|results|discussion|conclusion|references)'
        ]

        # Try to identify section boundaries
        sections = {}
        lines = text.split('\n')
        current_section = "preamble"
        current_content = []

        for line in lines:
            line_lower = line.lower().strip()
            # Check if this line is a section header
            for pattern in section_patterns:
                if re.match(pattern, line_lower):
                    # Save previous section
                    if current_content:
                        sections[current_section] = '\n'.join(current_content)
                    current_section = line_lower.split()[0] if line_lower else "unknown"
                    current_content = []
                    break
            current_content.append(line)

        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)

        # Prioritize sections for economics papers
        priority_sections = [
            'introduction', 'abstract', 'methodology', 'empirical',
            'data', 'model', 'results', 'conclusion'
        ]

        result_parts = []
        remaining_chars = max_chars

        for priority in priority_sections:
            for section_name, content in sections.items():
                if priority in section_name.lower() and remaining_chars > 0:
                    # Take what we can fit
                    take_chars = min(len(content), remaining_chars // 2)
                    result_parts.append(f"=== {section_name.upper()} ===\n{content[:take_chars]}")
                    remaining_chars -= take_chars

        # If we have room, add more content
        if remaining_chars > 1000:
            for section_name, content in sections.items():
                if section_name not in [p for p in priority_sections]:
                    take_chars = min(len(content), remaining_chars)
                    result_parts.append(f"=== {section_name.upper()} ===\n{content[:take_chars]}")
                    remaining_chars -= take_chars
                    if remaining_chars < 500:
                        break

        return '\n\n'.join(result_parts) if result_parts else text[:max_chars]

    def decompose_for_comparison(
        self,
        literature_batch: Dict,
        comparison_topic: str = "methodology"
    ) -> List[PaperStructure]:
        """
        Decompose papers with deep analysis for methodological comparison.

        Use this when researchers need to compare similar economic studies.

        Args:
            literature_batch: Literature data from Stage 1
            comparison_topic: Focus area - "methodology", "data", "results"

        Returns:
            List of PaperStructure with deep analysis
        """
        print(f"\n[{self.agent_name}] Deep decomposition for {comparison_topic} comparison...")

        # Use deep mode for all papers
        return self.decompose_papers(
            literature_batch,
            mode=DecompositionMode.DEEP
        )

class GapFinder:
    """Agent for research opportunity detection."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "GapFinder"
        self.api_key = openai_api_key
        self.collector = collector
        self.llm = LLMClient(


            temperature=0.4,

            api_key=self.api_key,

            collector=collector,

            agent_name=self.agent_name

        )
    
    def find_gaps(
        self,
        paper_structures: List[PaperStructure],
        literature_batch: Dict
    ) -> List[ResearchGap]:
        """Identify research gaps from paper structures."""
        
        print(f"\n[{self.agent_name}] Identifying research gaps from {len(paper_structures)} papers...")
        
        # Prepare analysis context
        trends = literature_batch.get('trend_analyses', [])
        insights = literature_batch.get('insights', [])
        
        # Identify different types of gaps
        methodological_gaps = self._find_methodological_gaps(paper_structures)
        theoretical_gaps = self._find_theoretical_gaps(paper_structures, insights)
        empirical_gaps = self._find_empirical_gaps(paper_structures, trends)
        data_gaps = self._find_data_gaps(paper_structures)
        
        all_gaps = methodological_gaps + theoretical_gaps + empirical_gaps + data_gaps
        
        # Deduplicate and rank gaps
        unique_gaps = self._deduplicate_gaps(all_gaps)
        ranked_gaps = self._rank_gaps(unique_gaps)
        
        print(f"[{self.agent_name}] Identified {len(ranked_gaps)} unique research gaps")
        print(f"  - Methodological: {len(methodological_gaps)}")
        print(f"  - Theoretical: {len(theoretical_gaps)}")
        print(f"  - Empirical: {len(empirical_gaps)}")
        print(f"  - Data: {len(data_gaps)}")
        
        return ranked_gaps
    
    def _find_methodological_gaps(self, paper_structures: List[PaperStructure]) -> List[ResearchGap]:
        """Find methodological gaps."""
        
        # Collect all methodologies and limitations
        all_methods = []
        all_limitations = []
        
        for paper in paper_structures:
            all_methods.extend(paper.methodologies)
            all_limitations.extend(paper.limitations)
        
        # Check if we have data to analyze
        if not all_methods and not all_limitations:
            return []
        
        # Prepare summary
        methods_summary = "\n".join([f"- {m}" for m in list(set(all_methods))[:20]]) if all_methods else "No methodologies found"
        limitations_summary = "\n".join([f"- {l}" for l in all_limitations[:30]]) if all_limitations else "No limitations found"
        try:

        
            result = self.llm.format_and_invoke(

        
                system_prompt="""You are a senior economics researcher specializing in methodological evaluation.
    You identify gaps by reasoning about what methods SHOULD exist but DON'T, based on stated limitations and the range of approaches used.""",

        
                user_prompt="""Identify 3-5 methodological gaps based on the methods used and limitations stated.

    Methodologies Used:
    {methods}

    Stated Limitations:
    {limitations}

    REASONING PROCESS (follow these steps):
    1. First, identify which standard economics methodologies are MISSING from the set used (e.g., if only OLS is used, note absence of IV/GMM for endogeneity)
    2. Second, examine limitations to find RECURRING themes (e.g., multiple papers acknowledge "small sample" → gap in large-scale empirical validation)
    3. Third, identify methodological COMBINATIONS that no paper attempts (e.g., combining ABM with empirical calibration)
    4. Fourth, assess whether existing methods adequately address the economic IDENTIFICATION problem

    For each gap, provide:
    - gap_id: Unique ID (e.g., "METH_GAP_001")
    - gap_type: "methodological"
    - gap_title: Concise, specific title (NOT generic like "need for more research")
    - gap_description: 3-4 sentences explaining: (a) WHAT is missing, (b) WHY it matters for the economics question, (c) which papers' findings would change if this gap were addressed
    - evidence_papers: List of specific paper titles that reveal this gap (MUST cite exact titles from the input above; each gap MUST link to specific paper limitations that create it)
    - severity: "Critical" if it undermines core findings, "High" if it limits generalizability, "Medium" if alternative approaches exist, "Low" if minor
    - addressability: "Easy" if standard methods exist, "Moderate" if requires new combination, "Difficult" if requires new methodology
    - related_trends: Empty list for now
    - potential_impact: SPECIFIC expected impact (e.g., "Would resolve endogeneity concern in 60% of reviewed studies")
    - suggested_approaches: List of 2-3 CONCRETE approaches (e.g., "Apply difference-in-differences with staggered treatment timing", NOT "use better methods")

    Return as a JSON array of gap objects.

    Respond with ONLY the JSON array, no other text.
    """,

        
                variables={
                "methods": methods_summary,
                "limitations": limitations_summary
                }

        
            )
            
            if not result:
                print(f"    Warning: Empty response from LLM for methodological gaps")
                return []
            
            gaps_data = safe_parse_json(result, "methodological gaps")
            if gaps_data is None:
                return []
            
            # Handle both list and dict formats
            if isinstance(gaps_data, dict):
                if 'gaps' in gaps_data:
                    gaps_data = gaps_data['gaps']
                elif 'data' in gaps_data:
                    gaps_data = gaps_data['data']
                else:
                    gaps_data = [gaps_data]
            elif not isinstance(gaps_data, list):
                print(f"    Warning: Unexpected gaps data format")
                return []
            
            gaps = []
            for idx, gap in enumerate(gaps_data):
                gap_obj = create_research_gap_from_dict(gap, gap_type="methodological", gap_counter=idx)
                if gap_obj:
                    gaps.append(gap_obj)
            
            return gaps
        
        except Exception as e:
            print(f"    Error finding methodological gaps: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _find_theoretical_gaps(
        self,
        paper_structures: List[PaperStructure],
        insights: List[Dict]
    ) -> List[ResearchGap]:
        """Find theoretical gaps."""
        
        # Collect theoretical contributions
        all_contributions = []
        for paper in paper_structures:
            all_contributions.extend(paper.theoretical_contributions)
        
        # Check if we have data to analyze
        if not all_contributions and not insights:
            return []
        
        # Prepare insights summary
        insights_summary = "\n".join([
            f"- {ins.get('insight_title', 'N/A')}: {ins.get('insight_description', 'N/A')[:100]}..."
            for ins in insights[:10]
        ]) if insights else "No insights available"
        
        contributions_summary = "\n".join([f"- {c}" for c in all_contributions[:20]]) if all_contributions else "No theoretical contributions found"
        try:

        
            result = self.llm.format_and_invoke(

        
                system_prompt="""You are a senior economics theorist who identifies gaps in theoretical frameworks.
    You reason about which economic mechanisms are under-theorized, which assumptions are too restrictive, and where competing theories remain unreconciled.""",

        
                user_prompt="""Identify 2-4 theoretical gaps based on the contributions and insights below.

    Theoretical Contributions:
    {contributions}

    Research Insights:
    {insights}

    REASONING PROCESS:
    1. Identify theoretical TENSIONS: where do papers make contradictory assumptions or reach conflicting conclusions?
    2. Identify UNDER-THEORIZED mechanisms: what economic channels are acknowledged but not formally modeled?
    3. Identify OVERLY RESTRICTIVE assumptions: which simplifying assumptions limit real-world applicability?
    4. Identify MISSING INTEGRATION: which theoretical traditions should be combined but haven't been?

    For each gap, provide:
    - gap_id: Unique ID (e.g., "THEO_GAP_001")
    - gap_type: "theoretical"
    - gap_title: Specific title naming the theoretical tension or missing mechanism
    - gap_description: 3-4 sentences explaining: (a) the theoretical deficiency, (b) WHY current theory is insufficient, (c) what a resolution would look like
    - evidence_papers: Papers whose theoretical frameworks reveal this gap
    - severity: "Critical"/"High"/"Medium"/"Low" with justification in description
    - addressability: "Easy"/"Moderate"/"Difficult"
    - related_trends: Empty list for now
    - potential_impact: How filling this gap would advance economic theory
    - suggested_approaches: 2-3 concrete theoretical directions (e.g., "Extend Bewley-Aiyagari framework to include bounded rationality")

    Return as a JSON array of gap objects.

    Respond with ONLY the JSON array, no other text.
    """,

        
                variables={
                "contributions": contributions_summary,
                "insights": insights_summary
                }

        
            )
            
            if not result:
                print(f"    Warning: Empty response from LLM for theoretical gaps")
                return []
            
            gaps_data = safe_parse_json(result, "theoretical gaps")
            if gaps_data is None:
                return []
            
            # Handle both list and dict formats
            if isinstance(gaps_data, dict):
                if 'gaps' in gaps_data:
                    gaps_data = gaps_data['gaps']
                elif 'data' in gaps_data:
                    gaps_data = gaps_data['data']
                else:
                    gaps_data = [gaps_data]
            elif not isinstance(gaps_data, list):
                print(f"    Warning: Unexpected gaps data format")
                return []
            
            gaps = []
            for idx, gap in enumerate(gaps_data):
                gap_obj = create_research_gap_from_dict(gap, gap_type="theoretical", gap_counter=idx)
                if gap_obj:
                    gaps.append(gap_obj)
            
            return gaps
        
        except Exception as e:
            print(f"    Error finding theoretical gaps: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _find_empirical_gaps(
        self,
        paper_structures: List[PaperStructure],
        trends: List[Dict]
    ) -> List[ResearchGap]:
        """Find empirical gaps."""
        
        # Collect findings and future work
        all_findings = []
        all_future_work = []
        
        for paper in paper_structures:
            all_findings.extend(paper.key_findings)
            all_future_work.extend(paper.future_work_suggestions)
        
        # Check if we have data to analyze
        if not all_future_work and not trends:
            return []
        
        trends_summary = "\n".join([
            f"- {t.get('trend_name', 'N/A')}: {t.get('description', 'N/A')[:100]}..."
            for t in trends[:7]
        ]) if trends else "No trends available"
        
        future_work_summary = "\n".join([f"- {fw}" for fw in all_future_work[:20]]) if all_future_work else "No future work suggestions found"
        try:

        
            result = self.llm.format_and_invoke(

        
                system_prompt="""You are a senior empirical economist who identifies gaps in empirical evidence.
    You reason about which empirical tests are missing, which data sources are underutilized, and where results lack external validity.""",

        
                user_prompt="""Identify 2-4 empirical gaps based on the future work suggestions and research trends below.

    Future Work Suggestions (from reviewed papers):
    {future_work}

    Research Trends:
    {trends}

    REASONING PROCESS:
    1. Which empirical TESTS have been proposed but not yet conducted? (e.g., "out-of-sample validation of ABM predictions")
    2. Which GEOGRAPHIC or TEMPORAL contexts are under-studied? (e.g., "most studies use US data; emerging economies lack evidence")
    3. Which DATA SOURCES could provide new evidence but haven't been exploited? (e.g., "admin data", "natural experiments")
    4. Where do EXISTING FINDINGS need replication or robustness checks?

    For each gap, provide:
    - gap_id: Unique ID (e.g., "EMP_GAP_001")
    - gap_type: "empirical"
    - gap_title: Specific title (e.g., "No out-of-sample validation for ABM policy counterfactuals")
    - gap_description: 3-4 sentences: (a) what evidence is missing, (b) why it matters, (c) what data/design could fill it
    - evidence_papers: Papers whose suggestions or limitations reveal this gap
    - severity: "Critical"/"High"/"Medium"/"Low"
    - addressability: "Easy"/"Moderate"/"Difficult"
    - related_trends: Empty list for now
    - potential_impact: Concrete empirical contribution expected
    - suggested_approaches: 2-3 concrete approaches (e.g., "Use FRED real-time data vintages to test forecast accuracy")

    Return as a JSON array of gap objects.

    Respond with ONLY the JSON array, no other text.
    """,

        
                variables={
                "future_work": future_work_summary,
                "trends": trends_summary
                }

        
            )
            
            if not result:
                print(f"    Warning: Empty response from LLM for empirical gaps")
                return []
            
            gaps_data = safe_parse_json(result, "empirical gaps")
            if gaps_data is None:
                return []
            
            # Handle both list and dict formats
            if isinstance(gaps_data, dict):
                if 'gaps' in gaps_data:
                    gaps_data = gaps_data['gaps']
                elif 'data' in gaps_data:
                    gaps_data = gaps_data['data']
                else:
                    gaps_data = [gaps_data]
            elif not isinstance(gaps_data, list):
                print(f"    Warning: Unexpected gaps data format")
                return []
            
            gaps = []
            for idx, gap in enumerate(gaps_data):
                gap_obj = create_research_gap_from_dict(gap, gap_type="empirical", gap_counter=idx)
                if gap_obj:
                    gaps.append(gap_obj)
            
            return gaps
        
        except Exception as e:
            print(f"    Error finding empirical gaps: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _find_data_gaps(self, paper_structures: List[PaperStructure]) -> List[ResearchGap]:
        """Find data-related gaps."""
        
        # Collect data sources
        all_data_sources = []
        for paper in paper_structures:
            all_data_sources.extend(paper.data_sources)
        
        if not all_data_sources:
            return []
        
        data_summary = "\n".join([f"- {ds}" for ds in list(set(all_data_sources))[:15]])
        try:

        
            result = self.llm.format_and_invoke(

        
                system_prompt="You are an expert at identifying data gaps in research.",

        
                user_prompt="""Based on data sources used, identify 1-3 data gaps.
            
                Data Sources Used:
                {data_sources}
            
                For each gap, provide the same structure with gap_type: "data".
            
                Return as a JSON array of gap objects.
            
                Respond with ONLY the JSON array, no other text.
                """,

        
                variables={
                "data_sources": data_summary
                }

        
            )
            
            if not result:
                print(f"    Warning: Empty response from LLM for data gaps")
                return []
            
            gaps_data = safe_parse_json(result, "data gaps")
            if gaps_data is None:
                return []
            
            # Handle both list and dict formats
            if isinstance(gaps_data, dict):
                if 'gaps' in gaps_data:
                    gaps_data = gaps_data['gaps']
                elif 'data' in gaps_data:
                    gaps_data = gaps_data['data']
                else:
                    gaps_data = [gaps_data]
            elif not isinstance(gaps_data, list):
                print(f"    Warning: Unexpected gaps data format")
                return []
            
            gaps = []
            for idx, gap in enumerate(gaps_data):
                gap_obj = create_research_gap_from_dict(gap, gap_type="data", gap_counter=idx)
                if gap_obj:
                    gaps.append(gap_obj)
            
            return gaps
        
        except Exception as e:
            print(f"    Error finding data gaps: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _deduplicate_gaps(self, gaps: List[ResearchGap]) -> List[ResearchGap]:
        """Remove duplicate gaps based on similarity."""
        unique_gaps = []
        seen_titles = set()
        
        for gap in gaps:
            normalized_title = gap.gap_title.lower().strip()
            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_gaps.append(gap)
        
        return unique_gaps
    
    def _rank_gaps(self, gaps: List[ResearchGap]) -> List[ResearchGap]:
        """Rank gaps by severity and addressability."""
        
        severity_order = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
        addressability_order = {"Easy": 3, "Moderate": 2, "Difficult": 1}
        
        def gap_score(gap: ResearchGap) -> Tuple[int, int]:
            sev = severity_order.get(gap.severity, 0)
            addr = addressability_order.get(gap.addressability, 0)
            return (sev, addr)
        
        gaps.sort(key=gap_score, reverse=True)
        return gaps

class KnowledgeWeaver:
    """Agent for knowledge graph construction."""

    def __init__(self, openai_api_key: str, collector: Optional[MetricsCollector] = None):
        self.agent_name = "KnowledgeWeaver"
        self.api_key = openai_api_key
        self.collector = collector
        self.llm = LLMClient(


            temperature=0.4,

            api_key=self.api_key,

            collector=collector,

            agent_name=self.agent_name

        )
    
    def construct_graph(
        self,
        literature_batch: Dict,
        paper_structures: List[PaperStructure],
        research_gaps: List[ResearchGap]
    ) -> KnowledgeGraph:
        """Construct knowledge graph from literature and gaps."""
        
        print(f"\n[{self.agent_name}] Constructing knowledge graph...")
        
        nodes = []
        edges = []
        
        # Create nodes for papers
        paper_nodes = self._create_paper_nodes(literature_batch.get('literature_items', []))
        nodes.extend(paper_nodes)
        
        # Create nodes for concepts/methods
        concept_nodes = self._create_concept_nodes(paper_structures)
        nodes.extend(concept_nodes)
        
        # Create nodes for trends
        trend_nodes = self._create_trend_nodes(literature_batch.get('trend_analyses', []))
        nodes.extend(trend_nodes)
        
        # Create nodes for gaps
        gap_nodes = self._create_gap_nodes(research_gaps)
        nodes.extend(gap_nodes)
        
        # Create edges between papers and methods
        method_edges = self._create_method_edges(paper_structures, paper_nodes, concept_nodes)
        edges.extend(method_edges)
        
        # Create edges between papers and gaps
        gap_edges = self._create_gap_edges(research_gaps, paper_nodes, gap_nodes)
        edges.extend(gap_edges)
        
        # Create edges between papers and trends
        trend_edges = self._create_trend_edges(
            literature_batch.get('citations', []),
            paper_nodes,
            trend_nodes
        )
        edges.extend(trend_edges)
        
        # Create citation edges (if available)
        citation_edges = self._create_citation_edges(paper_nodes)
        edges.extend(citation_edges)
        
        graph = KnowledgeGraph(
            nodes=nodes,
            edges=edges,
            metadata={
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "node_types": self._count_node_types(nodes),
                "edge_types": self._count_edge_types(edges),
                "created_at": datetime.now().isoformat()
            }
        )
        
        print(f"[{self.agent_name}] Graph constructed:")
        print(f"  - Nodes: {len(nodes)}")
        print(f"  - Edges: {len(edges)}")
        print(f"  - Node types: {graph.metadata['node_types']}")
        
        return graph
    
    def _create_paper_nodes(self, literature_items: List[Dict]) -> List[KnowledgeNode]:
        """Create nodes for papers."""
        nodes = []
        
        for idx, paper in enumerate(literature_items):
            node = KnowledgeNode(
                node_id=f"PAPER_{idx:03d}",
                node_type="paper",
                label=paper.get('title', 'Unknown'),
                properties={
                    "authors": paper.get('authors', []),
                    "year": paper.get('year'),
                    "citation_count": paper.get('citation_count'),
                    "source": paper.get('source'),
                    "url": paper.get('url')
                }
            )
            nodes.append(node)
        
        return nodes
    
    def _create_concept_nodes(self, paper_structures: List[PaperStructure]) -> List[KnowledgeNode]:
        """Create nodes for concepts and methods."""
        nodes = []
        seen_concepts = set()
        
        # Extract unique methods
        for paper in paper_structures:
            for method in paper.methodologies:
                if method and method not in seen_concepts:
                    seen_concepts.add(method)
                    node = KnowledgeNode(
                        node_id=f"METHOD_{len(nodes):03d}",
                        node_type="method",
                        label=method,
                        properties={}
                    )
                    nodes.append(node)
        
        return nodes
    
    def _create_trend_nodes(self, trends: List[Dict]) -> List[KnowledgeNode]:
        """Create nodes for trends."""
        nodes = []
        
        for idx, trend in enumerate(trends):
            node = KnowledgeNode(
                node_id=f"TREND_{idx:03d}",
                node_type="trend",
                label=trend.get('trend_name', 'Unknown'),
                properties={
                    "description": trend.get('description'),
                    "emergence_year": trend.get('emergence_year'),
                    "growth_trajectory": trend.get('growth_trajectory')
                }
            )
            nodes.append(node)
        
        return nodes
    
    def _create_gap_nodes(self, research_gaps: List[ResearchGap]) -> List[KnowledgeNode]:
        """Create nodes for research gaps."""
        nodes = []
        
        for gap in research_gaps:
            node = KnowledgeNode(
                node_id=gap.gap_id,
                node_type="gap",
                label=gap.gap_title,
                properties={
                    "gap_type": gap.gap_type,
                    "severity": gap.severity,
                    "addressability": gap.addressability,
                    "description": gap.gap_description
                }
            )
            nodes.append(node)
        
        return nodes
    
    def _create_method_edges(
        self,
        paper_structures: List[PaperStructure],
        paper_nodes: List[KnowledgeNode],
        method_nodes: List[KnowledgeNode]
    ) -> List[KnowledgeEdge]:
        """Create edges between papers and methods."""
        edges = []
        
        # Create mapping of paper titles to node IDs
        paper_map = {node.label: node.node_id for node in paper_nodes}
        method_map = {node.label: node.node_id for node in method_nodes}
        
        for paper_struct in paper_structures:
            paper_id = paper_map.get(paper_struct.paper_title)
            if not paper_id:
                continue
            
            for method in paper_struct.methodologies:
                method_id = method_map.get(method)
                if method_id:
                    edge = KnowledgeEdge(
                        source_id=paper_id,
                        target_id=method_id,
                        relationship="uses_method",
                        weight=1.0
                    )
                    edges.append(edge)
        
        return edges
    
    def _create_gap_edges(
        self,
        research_gaps: List[ResearchGap],
        paper_nodes: List[KnowledgeNode],
        gap_nodes: List[KnowledgeNode]
    ) -> List[KnowledgeEdge]:
        """Create edges between papers and gaps."""
        edges = []
        
        paper_map = {node.label: node.node_id for node in paper_nodes}
        
        for gap in research_gaps:
            gap_id = gap.gap_id
            
            for paper_title in gap.evidence_papers:
                paper_id = paper_map.get(paper_title)
                if paper_id:
                    edge = KnowledgeEdge(
                        source_id=paper_id,
                        target_id=gap_id,
                        relationship="reveals_gap",
                        weight=1.0
                    )
                    edges.append(edge)
        
        return edges
    
    def _create_trend_edges(
        self,
        citations: List[Dict],
        paper_nodes: List[KnowledgeNode],
        trend_nodes: List[KnowledgeNode]
    ) -> List[KnowledgeEdge]:
        """Create edges between papers and trends."""
        edges = []
        
        paper_map = {node.label: node.node_id for node in paper_nodes}
        trend_map = {node.label: node.node_id for node in trend_nodes}
        
        for citation in citations:
            paper_title = citation.get('paper_title')
            paper_id = paper_map.get(paper_title)
            
            if not paper_id:
                continue
            
            for trend_name in citation.get('related_trends', []):
                trend_id = trend_map.get(trend_name)
                if trend_id:
                    edge = KnowledgeEdge(
                        source_id=paper_id,
                        target_id=trend_id,
                        relationship="relates_to_trend",
                        weight=1.0
                    )
                    edges.append(edge)
        
        return edges
    
    def _create_citation_edges(self, paper_nodes: List[KnowledgeNode]) -> List[KnowledgeEdge]:
        """Create citation edges between papers (placeholder - would need citation data)."""
        # This is a placeholder - in a real implementation, you'd extract citation relationships
        return []
    
    def _count_node_types(self, nodes: List[KnowledgeNode]) -> Dict[str, int]:
        """Count nodes by type."""
        counts = {}
        for node in nodes:
            counts[node.node_type] = counts.get(node.node_type, 0) + 1
        return counts
    
    def _count_edge_types(self, edges: List[KnowledgeEdge]) -> Dict[str, int]:
        """Count edges by relationship type."""
        counts = {}
        for edge in edges:
            counts[edge.relationship] = counts.get(edge.relationship, 0) + 1
        return counts

# ========== ORCHESTRATOR ==========

class GapDetectionOrchestrator:
    """Orchestrates the gap detection pipeline."""

    def __init__(self, openai_api_key: Optional[str] = None, collector: Optional[MetricsCollector] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.collector = collector

        # Initialize agents
        self.paper_decomposer = PaperDecomposer(self.api_key, collector=collector)
        self.gap_finder = GapFinder(self.api_key, collector=collector)
        self.knowledge_weaver = KnowledgeWeaver(self.api_key, collector=collector)
        
        self.gap_analysis_result: Optional[GapAnalysisResult] = None
    
    def run_gap_detection_pipeline(
        self,
        literature_batch: Dict
    ) -> GapAnalysisResult:
        """Run the complete gap detection pipeline."""
        
        print(f"\n{'='*70}")
        print(f"GAP DETECTION PIPELINE")
        print(f"{'='*70}")
        print(f"Input: {literature_batch.get('metadata', {}).get('total_papers', 0)} papers")
        print(f"{'='*70}\n")
        
        # Step 1: PaperDecomposer - Structural analysis
        paper_structures = self.paper_decomposer.decompose_papers(literature_batch)
        
        # Step 2: GapFinder - Research opportunity detection
        research_gaps = self.gap_finder.find_gaps(paper_structures, literature_batch)
        
        # Step 3: KnowledgeWeaver - Graph construction
        knowledge_graph = self.knowledge_weaver.construct_graph(
            literature_batch,
            paper_structures,
            research_gaps
        )
        
        # Create gap summary
        gap_summary = self._create_gap_summary(research_gaps)
        
        # Build reasoning trace for transparency
        reasoning_trace = {
            "decomposition_rationale": f"Decomposed {len(paper_structures)} papers using {'deep PDF analysis' if any(p.full_text_available for p in paper_structures) else 'metadata-based analysis'}. Papers were decomposed by extracting research objectives, methodologies, findings, and limitations to enable systematic gap identification.",
            "gap_identification_method": "Gaps identified through 4-stage process: (1) methodological gap analysis comparing used vs. expected methods, (2) theoretical gap analysis checking assumption tensions, (3) empirical gap analysis identifying untested hypotheses, (4) data gap analysis assessing source coverage.",
            "gap_evidence_linking": {
                gap.gap_id: {
                    "source_papers": gap.evidence_papers,
                    "triggering_limitations": [l for p in paper_structures for l in p.limitations if any(kw in l.lower() for kw in gap.gap_title.lower().split()[:3])][:3]
                }
                for gap in research_gaps[:15]
            }
        }

        assumptions = [
            "Paper abstracts and metadata accurately reflect the full paper's methodology and findings",
            "The literature batch from Stage 1 is representative of the research area",
            "Gap severity ratings are based on the reviewed subset, not the entire literature",
            "Methodological gaps assume standard economics research practices as the benchmark",
            "Papers not available in full text were decomposed using metadata only, which may miss nuances"
        ]

        limitations = [
            "Gap analysis is limited to the papers retrieved in Stage 1; important works may be missing",
            "Deep decomposition depends on PDF availability; metadata-only analysis is shallower",
            "Gap severity and addressability are estimated heuristically, not empirically validated",
            "Cross-disciplinary gaps may be underrepresented if the literature batch is domain-narrow",
            "LLM-based gap identification may reflect training data biases rather than true research gaps"
        ]

        # Create result
        self.gap_analysis_result = GapAnalysisResult(
            paper_structures=paper_structures,
            research_gaps=research_gaps,
            knowledge_graph=knowledge_graph,
            gap_summary=gap_summary,
            metadata={
                "created_at": datetime.now().isoformat(),
                "total_papers_analyzed": len(paper_structures),
                "total_gaps_identified": len(research_gaps),
                "graph_nodes": len(knowledge_graph.nodes),
                "graph_edges": len(knowledge_graph.edges)
            },
            reasoning_trace=reasoning_trace,
            assumptions=assumptions,
            limitations=limitations
        )
        
        print(f"\n{'='*70}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Papers Analyzed: {len(paper_structures)}")
        print(f"Gaps Identified: {len(research_gaps)}")
        print(f"Knowledge Graph: {len(knowledge_graph.nodes)} nodes, {len(knowledge_graph.edges)} edges")
        print(f"{'='*70}\n")
        
        return self.gap_analysis_result
    
    def _create_gap_summary(self, research_gaps: List[ResearchGap]) -> Dict:
        """Create summary statistics for gaps."""
        
        gap_types = {}
        severities = {}
        addressabilities = {}
        
        for gap in research_gaps:
            gap_types[gap.gap_type] = gap_types.get(gap.gap_type, 0) + 1
            severities[gap.severity] = severities.get(gap.severity, 0) + 1
            addressabilities[gap.addressability] = addressabilities.get(gap.addressability, 0) + 1
        
        return {
            "total_gaps": len(research_gaps),
            "by_type": gap_types,
            "by_severity": severities,
            "by_addressability": addressabilities
        }
    
    def load_literature_batch(self, filepath: str) -> Dict:
        """Load literature batch from Stage 1."""
        print(f"[Orchestrator] Loading literature batch from: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"[Orchestrator] Loaded batch with {data.get('metadata', {}).get('total_papers', 0)} papers")
        return data
    
    def save_gap_analysis(self, filepath: str):
        """Save the complete gap analysis to JSON."""
        if not self.gap_analysis_result:
            print("No gap analysis to save.")
            return
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.gap_analysis_result.model_dump(), f, indent=2, ensure_ascii=False)
        
        print(f"[Orchestrator] Gap analysis saved to: {filepath}")
    
    def save_gaps_csv(self, filepath: str):
        """Save research gaps to CSV for easy viewing."""
        if not self.gap_analysis_result:
            print("No gap analysis to save.")
            return
        
        data = [gap.model_dump() for gap in self.gap_analysis_result.research_gaps]
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"[Orchestrator] Gaps CSV saved to: {filepath}")
    
    def save_graph_json(self, filepath: str):
        """Save knowledge graph to JSON."""
        if not self.gap_analysis_result:
            print("No gap analysis to save.")
            return
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.gap_analysis_result.knowledge_graph.model_dump(), f, indent=2, ensure_ascii=False)
        
        print(f"[Orchestrator] Knowledge graph saved to: {filepath}")
    
    def print_summary(self):
        """Print a summary of the gap analysis."""
        if not self.gap_analysis_result:
            print("No gap analysis available.")
            return
        
        result = self.gap_analysis_result
        
        print(f"\n{'='*70}")
        print(f"GAP ANALYSIS SUMMARY")
        print(f"{'='*70}\n")
        
        # Gap summary
        print(f"Research Gaps ({result.gap_summary['total_gaps']}):")
        print(f"  By Type: {result.gap_summary['by_type']}")
        print(f"  By Severity: {result.gap_summary['by_severity']}")
        print(f"  By Addressability: {result.gap_summary['by_addressability']}")
        
        # Top gaps
        print(f"\nTop 10 Research Gaps (by severity):")
        for i, gap in enumerate(result.research_gaps[:10], 1):
            print(f"  {i}. [{gap.gap_type.upper()}] {gap.gap_title}")
            print(f"     Severity: {gap.severity} | Addressability: {gap.addressability}")
            print(f"     {gap.gap_description[:100]}...")
        
        # Knowledge graph
        print(f"\nKnowledge Graph:")
        print(f"  Nodes: {len(result.knowledge_graph.nodes)}")
        print(f"  Edges: {len(result.knowledge_graph.edges)}")
        print(f"  Node Types: {result.knowledge_graph.metadata.get('node_types', {})}")
        print(f"  Edge Types: {result.knowledge_graph.metadata.get('edge_types', {})}")
        
        print(f"\n{'='*70}\n")

# ========== MAIN ==========

def main():
    """Main function to run the gap detection stage."""
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Initialize orchestrator
    orchestrator = GapDetectionOrchestrator()
    
    # Load literature batch from Stage 1
    batch_file = auto_input(
        "Enter path to literature batch JSON (or press Enter for default): ",
        default=get_default("batch_file_path"),
    ).strip()
    if not batch_file:
        batch_file = "literature_batch.json"
    
    try:
        literature_batch = orchestrator.load_literature_batch(batch_file)
    except Exception as e:
        print(f"Error loading literature batch: {e}")
        print("Please run Stage 1 first to generate the literature batch.")
        return
    
    # Run pipeline
    gap_analysis = orchestrator.run_gap_detection_pipeline(literature_batch)
    
    # Save results
    orchestrator.save_gap_analysis("gap_analysis_results.json")
    orchestrator.save_gaps_csv("research_gaps.csv")
    orchestrator.save_graph_json("knowledge_graph.json")
    
    # Print summary
    orchestrator.print_summary()
    
    print("\n" + "="*70)
    print("Gap detection complete!")
    print("Output files:")
    print("  - gap_analysis_results.json (complete analysis)")
    print("  - research_gaps.csv (gaps only)")
    print("  - knowledge_graph.json (graph structure)")
    print("="*70)

if __name__ == "__main__":
    main()
