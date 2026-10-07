# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Unified Console UI for AEL Workflows

This module provides consistent, clean console output for all 15 workflows.
Design principles:
1. Clear visual hierarchy - distinguish stages from steps
2. No duplicate output - each message appears once
3. Progress visibility - show current position in workflow
4. Minimal but informative - key info without clutter
5. Error visibility - highlight errors without overwhelming
6. Consistent symbols - unified iconography across all workflows

Usage:
    from shared.console_ui import ConsoleUI

    ui = ConsoleUI(team="IdeationTeam", mode="ModeNoWcNoHITL")
    ui.workflow_header(topic="Agent-based modeling...")
    ui.show_api_keys()  # Show all loaded API keys
    ui.stage_start(1, 3, "Literature Sourcing")
    ui.agent_result("TrendSurfer", items=15)
    ui.stage_complete(1, items=19, duration=301.45)
    ui.workflow_complete(total_duration=384.73, stages_completed=3)
"""

import os
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any


# Standard API keys used across AEL workflows
# Format: (env_var_name, display_name, required_for_teams_or_modes)
STANDARD_API_KEYS = [
    ("OPENAI_API_KEY", "OpenAI", True),  # required only for OpenAI models (see below)
    ("SEMANTIC_SCHOLAR_API_KEY", "Semantic Scholar", False),
    ("FIRECRAWL_API_KEY", "Firecrawl", False),
    ("GOOGLE_API_KEY", "Google Search", False),
    ("GOOGLE_SEARCH_ENGINE_ID", "Google CSE ID", False),
    ("BRAVE_API_KEY", "Brave Search", False),
    ("SERPER_API_KEY", "Serper", False),
    ("FRED_API_KEY", "FRED", False),
    ("WORLD_BANK_API_KEY", "World Bank", False),
    ("BLOOMBERG_API_KEY", "Bloomberg", False),
    ("REFINITIV_EIKON_API_KEY", "Refinitiv Eikon", False),
    ("WRDS_USERNAME", "WRDS Username", False),
]


# Module-level registry of the most recently rendered HITL preview items.
# Stage-level feedback collectors can translate a reviewer's numeric tokens
# ("1, 3") back into titles/questions without threading a ConsoleUI reference
# down through every stage method.
_LAST_PREVIEW_ITEMS: List[Dict[str, Any]] = []


def _set_last_preview_items(items: List[Dict[str, Any]]) -> None:
    """Replace the module-level preview cache (called by ConsoleUI internally)."""
    global _LAST_PREVIEW_ITEMS
    _LAST_PREVIEW_ITEMS = list(items)


def get_last_preview_items() -> List[Dict[str, Any]]:
    """Read the most recent preview items rendered at a HITL checkpoint."""
    return list(_LAST_PREVIEW_ITEMS)


def resolve_feedback_entries(
    raw: str,
    keys: Optional[List[str]] = None,
    items: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """
    Translate comma-separated feedback input into resolved strings.

    Numeric tokens ("1", "3") are looked up against ``items`` (defaults to the
    last preview rendered at a HITL checkpoint), using the first key in
    ``keys`` that exists on the row. Non-numeric tokens and out-of-range
    indices pass through unchanged. Empty input returns ``[]``.

    ``keys`` default is ``["title", "question"]`` — works for paper titles,
    concept titles, and research questions without caller-side config.
    """
    if not raw:
        return []
    pool = items if items is not None else _LAST_PREVIEW_ITEMS
    if keys is None:
        keys = ["title", "question"]

    resolved: List[str] = []
    for tok in (t.strip() for t in raw.split(",")):
        if not tok:
            continue
        if tok.isdigit() and pool:
            idx = int(tok) - 1
            if 0 <= idx < len(pool):
                row = pool[idx]
                for k in keys:
                    if k in row and row[k]:
                        resolved.append(str(row[k]))
                        break
                else:
                    resolved.append(tok)
                continue
        resolved.append(tok)
    return resolved


class ConsoleUI:
    """Unified console UI for all AEL workflows."""

    # Box drawing characters (ASCII-safe for Windows compatibility)
    BOX_TL = "+"  # Top-left
    BOX_TR = "+"  # Top-right
    BOX_BL = "+"  # Bottom-left
    BOX_BR = "+"  # Bottom-right
    BOX_H = "-"   # Horizontal
    BOX_V = "|"   # Vertical

    # Status symbols
    SYM_SUCCESS = "[OK]"
    SYM_ERROR = "[!]"
    SYM_WARNING = "[~]"
    SYM_INFO = "[i]"
    SYM_RUNNING = "[>]"
    SYM_PENDING = "[ ]"
    SYM_HITL = "[?]"

    # Progress bar characters
    PROG_FULL = "#"
    PROG_EMPTY = "-"

    # Width settings
    WIDTH = 72

    def __init__(
        self,
        team: str,
        mode: str,
        total_stages: int = 3,
        verbose: bool = False
    ):
        """
        Initialize ConsoleUI.

        Args:
            team: Team name (IdeationTeam, LiteratureTeam, etc.)
            mode: Mode name (ModeNoWcNoHITL, etc.)
            total_stages: Total number of stages in workflow
            verbose: If True, show detailed output
        """
        self.team = team
        self.mode = mode
        self.total_stages = total_stages
        # AEL_VERBOSE env var enables verbose output globally
        self.verbose = verbose or os.environ.get("AEL_VERBOSE", "").lower() in ("true", "1")
        self.current_stage = 0
        self.errors: List[str] = []
        self.start_time: Optional[datetime] = None
        self._last_agent: Optional[str] = None

    def _line(self, char: str = "-") -> str:
        """Create a horizontal line."""
        return char * self.WIDTH

    def _box_line(self, text: str, fill: str = " ") -> str:
        """Create a boxed line with text."""
        inner_width = self.WIDTH - 4  # Account for "| " and " |"
        return f"{self.BOX_V} {text:{fill}<{inner_width}} {self.BOX_V}"

    def _center(self, text: str, width: Optional[int] = None) -> str:
        """Center text within width."""
        w = width or self.WIDTH
        return text.center(w)

    def _truncate(self, text: str, max_len: int = 50) -> str:
        """Truncate text with ellipsis if too long."""
        if len(text) <= max_len:
            return text
        return text[:max_len-3] + "..."

    # =========================================================================
    # Workflow Level
    # =========================================================================

    def workflow_header(
        self,
        topic: Optional[str] = None,
        execution_id: Optional[str] = None,
        **metadata
    ):
        """
        Print workflow header.

        Args:
            topic: Research topic or main input
            execution_id: Unique execution ID
            **metadata: Additional metadata to display
        """
        self.start_time = datetime.now()

        print()
        print(self._line("="))
        print(f"  WORKFLOW: {self.team} / {self.mode}")
        if execution_id:
            print(f"  Execution ID: {execution_id}")
        print(f"  Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if topic:
            print(f"  Topic: {self._truncate(topic, 55)}")
        for key, value in metadata.items():
            if value:
                display_key = key.replace("_", " ").title()
                print(f"  {display_key}: {self._truncate(str(value), 55)}")
        print(self._line("="))
        print()

    def show_api_keys(
        self,
        custom_keys: Optional[List[str]] = None,
        show_all: bool = False
    ):
        """
        Display loaded API keys status.

        Args:
            custom_keys: Optional list of specific env var names to check
            show_all: If True, show all standard keys; if False, show only loaded ones
        """
        print("  API Keys:")

        # Determine which keys to check
        if custom_keys:
            keys_to_check = [(k, k.replace("_", " ").replace("API KEY", "").strip().title(), False)
                           for k in custom_keys]
        else:
            keys_to_check = STANDARD_API_KEYS

        loaded_count = 0
        missing_required = []

        # The OpenAI key is required only when the effective model (AEL_MODEL, else
        # ael_config.yaml default_model) is an OpenAI model
        try:
            from shared.model_config import default_model
            from shared.provider_map import detect_provider
            openai_optional = detect_provider(os.environ.get("AEL_MODEL") or default_model()) != "openai"
        except Exception:
            openai_optional = True

        for env_var, display_name, required in keys_to_check:
            if env_var == "OPENAI_API_KEY" and openai_optional:
                required = False
            value = os.getenv(env_var)

            if value:
                # Mask the key value (show first 8 chars + ...)
                masked = f"{value[:8]}..." if len(value) > 8 else value[:4] + "..."
                print(f"    {self.SYM_SUCCESS} {display_name:<20} {masked}")
                loaded_count += 1
            elif show_all or required:
                status = "REQUIRED" if required else "not set"
                symbol = self.SYM_ERROR if required else self.SYM_PENDING
                print(f"    {symbol} {display_name:<20} {status}")
                if required:
                    missing_required.append(display_name)

        # Summary line
        if missing_required:
            print(f"\n  {self.SYM_WARNING} Missing required: {', '.join(missing_required)}")
        elif loaded_count == 0:
            print(f"    {self.SYM_WARNING} No API keys found in environment")

        print()

    def workflow_complete(
        self,
        success: bool = True,
        total_duration: Optional[float] = None,
        stages_completed: Optional[int] = None,
        total_errors: Optional[int] = None,
        output_files: Optional[List[str]] = None,
        summary: Optional[Dict[str, Any]] = None
    ):
        """
        Print workflow completion summary.

        Args:
            success: Whether workflow completed successfully
            total_duration: Total duration in seconds
            stages_completed: Number of stages completed
            total_errors: Total number of errors
            output_files: List of output files
            summary: Additional summary data
        """
        if total_duration is None and self.start_time:
            total_duration = (datetime.now() - self.start_time).total_seconds()

        stages = stages_completed if stages_completed is not None else self.current_stage
        errors = total_errors if total_errors is not None else len(self.errors)

        status_symbol = self.SYM_SUCCESS if success else self.SYM_ERROR
        status_text = "COMPLETE" if success else "FAILED"

        print()
        print(self._line("="))
        print(f"  {status_symbol} WORKFLOW {status_text}")
        print(self._line("-"))

        # Stats line
        duration_str = f"{total_duration:.1f}s" if total_duration else "N/A"
        print(f"  Duration: {duration_str} | Stages: {stages}/{self.total_stages} | Errors: {errors}")

        # Output files (show absolute paths so user can find them)
        if output_files:
            abs_files = [os.path.abspath(f) for f in output_files]
            print(f"  Output: {abs_files[0]}")
            for f in abs_files[1:]:
                print(f"          {f}")

        # Summary data
        if summary:
            print(self._line("-"))
            for key, value in summary.items():
                display_key = key.replace("_", " ").title()
                print(f"  {display_key}: {value}")

        print(self._line("="))

        # Next steps guidance
        if success and output_files:
            print()
            print("  Next steps:")
            txt_files = [f for f in output_files if f.endswith('.txt')]
            json_files = [f for f in output_files if f.endswith('.json')]
            if txt_files:
                abs_txt = os.path.abspath(txt_files[0])
                print(f"    View results:  python -c \"print(open(r'{abs_txt}').read())\"")
            elif json_files:
                abs_json = os.path.abspath(json_files[0])
                print(f"    View results:  python -c \"import json; print(json.dumps(json.load(open(r'{abs_json}')), indent=2))\"")
            print(f"    Evaluate:      python -m evaluation.run_ael_evaluation --tier2   # scores all team/mode outputs")
        print()

    def observability_summary(self, metrics=None):
        """
        Print observability metrics summary (LLM usage, tool usage, per-agent/stage breakdown).

        Args:
            metrics: A MetricsCollector object (calls .get_summary()), a summary dict,
                     or None (shows "no data" message).
        """
        print()
        print(self._line("="))
        print("  OBSERVABILITY METRICS")
        print(self._line("-"))

        # Resolve metrics to a summary dict
        summary = None
        if metrics is not None:
            if hasattr(metrics, "get_summary"):
                summary = metrics.get_summary()
            elif isinstance(metrics, dict):
                summary = metrics

        if summary is None or (
            summary.get("llm", {}).get("total_calls", 0) == 0
            and summary.get("tools", {}).get("total_calls", 0) == 0
        ):
            print(f"  {self.SYM_INFO} No observability data collected for this run.")
            print(self._line("="))
            print()
            return

        # -- LLM Usage --
        llm = summary.get("llm", {})
        llm_calls = llm.get("total_calls", 0)
        llm_errors = llm.get("error_count", 0)
        total_tokens = llm.get("total_tokens", 0)
        prompt_tokens = llm.get("total_prompt_tokens", 0)
        completion_tokens = llm.get("total_completion_tokens", 0)
        llm_cost = llm.get("total_cost_usd", 0.0)
        llm_latency = llm.get("avg_latency_seconds", 0.0)

        print("  LLM Usage")
        print(f"    Calls:     {llm_calls:<20} Errors:    {llm_errors}")
        print(f"    Tokens:    {total_tokens:,} total ({prompt_tokens:,} in / {completion_tokens:,} out)")
        print(f"    Cost:      ${llm_cost:.4f} USD")
        print(f"    Latency:   {llm_latency:.2f}s avg per call")
        print(self._line("-"))

        # -- Tool Usage --
        tools = summary.get("tools", {})
        tool_calls = tools.get("total_calls", 0)
        tool_errors = tools.get("error_count", 0)
        success_rate = tools.get("success_rate", 0.0) * 100
        tool_latency = tools.get("avg_latency_seconds", 0.0)

        print("  Tool Usage")
        print(f"    Calls:     {tool_calls:<20} Errors:    {tool_errors}")
        print(f"    Success:   {success_rate:.1f}%")
        print(f"    Latency:   {tool_latency:.2f}s avg per call")
        print(self._line("-"))

        # -- Per-Agent Breakdown --
        by_agent = summary.get("by_agent", {})
        if by_agent:
            print("  Per-Agent Breakdown")
            hdr = f"  {'Agent':<18} {'LLM Calls':>9}  {'Tokens':>8}  {'Cost':>7}  {'Tool Calls':>10}  {'Latency':>8}"
            print(hdr)
            print(f"  {'-'*18} {'-'*9}  {'-'*8}  {'-'*7}  {'-'*10}  {'-'*8}")
            for agent_name in sorted(by_agent.keys()):
                ag = by_agent[agent_name]
                ag_llm = ag.get("llm", {})
                ag_tools = ag.get("tools", {})
                a_llm_calls = ag_llm.get("total_calls", 0)
                a_tokens = ag_llm.get("total_tokens", 0)
                a_cost = ag_llm.get("total_cost_usd", 0.0)
                a_tool_calls = ag_tools.get("total_calls", 0)
                a_llm_lat = ag_llm.get("avg_latency_seconds", 0.0)
                a_tool_lat = ag_tools.get("avg_latency_seconds", 0.0)
                # Use LLM latency if available, else tool latency
                a_lat = a_llm_lat if a_llm_calls > 0 else a_tool_lat
                print(f"  {agent_name:<18} {a_llm_calls:>9}  {a_tokens:>8,}  ${a_cost:>6.4f}  {a_tool_calls:>10}  {a_lat:>7.1f}s")
            print(self._line("-"))

        # -- Per-Stage Breakdown --
        by_stage = summary.get("by_stage", {})
        if by_stage:
            print("  Per-Stage Breakdown")
            hdr = f"  {'Stage':<18} {'LLM Calls':>9}  {'Tokens':>8}  {'Cost':>7}  {'Tool Calls':>10}  {'Latency':>8}"
            print(hdr)
            print(f"  {'-'*18} {'-'*9}  {'-'*8}  {'-'*7}  {'-'*10}  {'-'*8}")
            for stage_name in by_stage:
                sg = by_stage[stage_name]
                sg_llm = sg.get("llm", {})
                sg_tools = sg.get("tools", {})
                s_llm_calls = sg_llm.get("total_calls", 0)
                s_tokens = sg_llm.get("total_tokens", 0)
                s_cost = sg_llm.get("total_cost_usd", 0.0)
                s_tool_calls = sg_tools.get("total_calls", 0)
                s_llm_lat = sg_llm.get("avg_latency_seconds", 0.0)
                s_tool_lat = sg_tools.get("avg_latency_seconds", 0.0)
                s_lat = s_llm_lat if s_llm_calls > 0 else s_tool_lat
                print(f"  {stage_name:<18} {s_llm_calls:>9}  {s_tokens:>8,}  ${s_cost:>6.4f}  {s_tool_calls:>10}  {s_lat:>7.1f}s")
            print(self._line("-"))

        # -- Footer --
        total_cost = llm_cost
        total_calls_str = f"{llm_calls} LLM + {tool_calls} tool calls"
        total_errs = llm_errors + tool_errors
        print(f"  Total Cost: ${total_cost:.4f} USD | {total_calls_str} | {total_errs} errors")
        print(self._line("="))
        print()

    # =========================================================================
    # Stage Level
    # =========================================================================

    def stage_start(
        self,
        stage_num: int,
        stage_name: str,
        description: Optional[str] = None
    ):
        """
        Print stage start header.

        Args:
            stage_num: Current stage number (1-indexed)
            stage_name: Name of the stage
            description: Optional description
        """
        self.current_stage = stage_num
        self._last_agent = None

        # Calculate proper padding for stage header
        stage_text = f"STAGE {stage_num}/{self.total_stages}: {stage_name}"
        inner_width = self.WIDTH - 4  # Account for "| " and " |"

        print()
        print(f"+{'-' * (self.WIDTH - 2)}+")
        print(f"| {stage_text:<{inner_width}} |")
        if description:
            print(f"| {description:<{inner_width}} |")
        print(f"+{'-' * (self.WIDTH - 2)}+")
        print()

    def stage_complete(
        self,
        stage_num: int,
        status: str = "success",
        items: int = 0,
        duration: Optional[float] = None,
        output_file: Optional[str] = None,
        error_message: Optional[str] = None
    ):
        """
        Print stage completion summary.

        Args:
            stage_num: Stage number that completed
            status: Status (success/error/skipped)
            items: Number of items processed
            duration: Duration in seconds
            output_file: Primary output file
            error_message: Error message if failed
        """
        if status == "success":
            symbol = self.SYM_SUCCESS
        elif status == "error":
            symbol = self.SYM_ERROR
        else:
            symbol = self.SYM_WARNING

        duration_str = f"{duration:.1f}s" if duration else ""

        print()
        print(f"  {self._line('-')[:60]}")

        if status == "success":
            items_str = f"{items} items" if items != 1 else "1 item"
            print(f"  {symbol} Stage {stage_num} complete: {items_str} | {duration_str}")
        else:
            print(f"  {symbol} Stage {stage_num} {status}: {error_message or 'Unknown error'}")

        if output_file and status == "success":
            print(f"      Output: {output_file}")

        print()

    # =========================================================================
    # Agent Level
    # =========================================================================

    def agent_start(self, agent_name: str, task: Optional[str] = None):
        """
        Print agent start (only if verbose or first mention).

        Args:
            agent_name: Name of the agent
            task: Current task description
        """
        if self._last_agent != agent_name:
            self._last_agent = agent_name
            if self.verbose and task:
                print(f"  [{agent_name}] {task}")

    def agent_result(
        self,
        agent_name: str,
        items: int = 0,
        message: Optional[str] = None,
        errors: int = 0
    ):
        """
        Print agent result on a single line.

        Args:
            agent_name: Name of the agent
            items: Number of items found/processed
            message: Optional result message
            errors: Number of errors encountered
        """
        # Progress bar (max 20 chars, scales with items)
        bar_width = 20
        filled = min(bar_width, items) if items > 0 else 0
        bar = self.PROG_FULL * filled + self.PROG_EMPTY * (bar_width - filled)

        # Format result (consistent width: 3 digits + space + 5 chars = 9 chars)
        items_str = f"{items:>3} items"

        # Error suffix
        error_str = f" ({errors} errors)" if errors > 0 else ""

        # Message or default (2 spaces between bar and items for readability)
        if message:
            print(f"  [{agent_name:<15}] {bar}  {items_str}{error_str}")
            print(f"                      {self._truncate(message, 48)}")
        else:
            print(f"  [{agent_name:<15}] {bar}  {items_str}{error_str}")

    def agent_progress(
        self,
        agent_name: str,
        current: int,
        total: int,
        item_name: str = "items"
    ):
        """
        Print agent progress (for long-running operations).

        Args:
            agent_name: Name of the agent
            current: Current progress
            total: Total items
            item_name: Name of items being processed
        """
        bar_width = 20
        pct = current / total if total > 0 else 0
        filled = int(bar_width * pct)
        bar = self.PROG_FULL * filled + self.PROG_EMPTY * (bar_width - filled)

        # Carriage return to update same line
        print(f"\r  [{agent_name:<15}] {bar} {current}/{total} {item_name}", end="", flush=True)

        # Newline when complete
        if current >= total:
            print()

    # =========================================================================
    # HITL Checkpoints
    # =========================================================================

    def hitl_checkpoint(
        self,
        checkpoint_num: int,
        checkpoint_name: str,
        summary: Optional[Dict[str, Any]] = None,
        options: Optional[List[str]] = None,
        preview: Optional[Dict[str, Any]] = None,
    ):
        """
        Print HITL checkpoint header.

        Args:
            checkpoint_num: Checkpoint number
            checkpoint_name: Name of checkpoint
            summary: Summary data to display (count-style key/value rows)
            options: Available options for user
            preview: Optional content preview to give reviewers context before
                the feedback prompts. Shape:
                    {
                        "title": "Top 10 of 53 papers",     # optional header line
                        # Single-section form:
                        "items": [ {col: val, ...}, ... ],
                        "columns": [(key, header, width), ...],
                        # OR multi-section form:
                        "sections": [
                            {"subtitle": "...", "items": [...], "columns": [...]},
                            ...
                        ],
                        "limit": 10,                         # max rows per section
                        "full_contents_path": "/abs/path.csv",  # file reminder
                        "full_contents_label": "Full ranked list",
                    }
        """
        print()
        print(f"  {self._line('*')[:60]}")
        print(f"  {self.SYM_HITL} CHECKPOINT {checkpoint_num}: {checkpoint_name}")
        print(f"  {self._line('*')[:60]}")

        if summary:
            for key, value in summary.items():
                display_key = key.replace("_", " ").title()
                print(f"    {display_key}: {value}")

        if preview:
            self._render_preview(preview)

        if options:
            print()
            print("  Options:")
            for opt in options:
                print(f"    - {opt}")

        print()

    def _render_preview(self, preview: Dict[str, Any]) -> None:
        """Render a HITL content preview (table + file-path reminder)."""
        title = preview.get("title")
        if title:
            print()
            print(f"  {title}")

        # Stash items so `resolve_feedback_entries` can translate numeric
        # tokens the reviewer types ("1, 3") back to titles/questions for the
        # upcoming stage-level prompt. Flattened across sections — resolvers
        # pick the matching key per prompt (e.g. "title", "question").
        collected: List[Dict[str, Any]] = []
        sections = preview.get("sections")
        if sections:
            for section in sections:
                collected.extend(section.get("items") or [])
                self._render_preview_section(section, default_limit=preview.get("limit"))
        else:
            collected.extend(preview.get("items") or [])
            self._render_preview_section(preview, default_limit=preview.get("limit"))
        _set_last_preview_items(collected)

        full_path = preview.get("full_contents_path")
        if full_path:
            label = preview.get("full_contents_label", "Full contents")
            print()
            print(f"  {label}: {full_path}")

    def _render_preview_section(
        self,
        section: Dict[str, Any],
        default_limit: Optional[int] = None,
    ) -> None:
        """Render one table section inside a preview."""
        items = section.get("items") or []
        if not items:
            return

        subtitle = section.get("subtitle")
        columns = section.get("columns")
        if not columns:
            # Fallback: stringify every key in the first item at width 40.
            columns = [(k, k.replace("_", " ").title(), 40) for k in items[0].keys()]

        limit = section.get("limit", default_limit)
        shown = items if limit is None else items[:limit]

        if subtitle:
            print()
            print(f"  {subtitle}:")

        header_row = "  ".join(f"{header:<{width}}" for _, header, width in columns)
        sep_row = "  ".join("-" * width for _, _, width in columns)
        print(f"   {header_row}")
        print(f"   {sep_row}")
        for item in shown:
            cells = []
            for key, _, width in columns:
                val = item.get(key, "")
                s = "" if val is None else str(val)
                if len(s) > width:
                    s = s[: max(1, width - 3)] + "..."
                cells.append(f"{s:<{width}}")
            print(f"   {'  '.join(cells)}")

        remaining = len(items) - len(shown)
        if remaining > 0:
            print(f"   (+{remaining} more)")

    def hitl_response(self, response: str, auto: bool = False):
        """
        Print HITL response.

        Args:
            response: User response
            auto: Whether response was auto-generated
        """
        if auto:
            print(f"  {self.SYM_INFO} Auto-response: {response}")
        else:
            print(f"  {self.SYM_INFO} Response: {response}")
        print()

    # =========================================================================
    # Messages
    # =========================================================================

    def info(self, message: str):
        """Print info message."""
        print(f"  {self.SYM_INFO} {message}")

    def success(self, message: str):
        """Print success message."""
        print(f"  {self.SYM_SUCCESS} {message}")

    def warning(self, message: str):
        """Print warning message."""
        print(f"  {self.SYM_WARNING} {message}")

    def error(self, message: str, store: bool = True):
        """
        Print error message.

        Args:
            message: Error message
            store: Whether to store for summary
        """
        print(f"  {self.SYM_ERROR} {message}")
        if store:
            self.errors.append(message)

    def debug(self, message: str):
        """Print debug message (only in verbose mode)."""
        if self.verbose:
            print(f"      {message}")

    # =========================================================================
    # Utility
    # =========================================================================

    def divider(self, char: str = "-"):
        """Print a divider line."""
        print(f"  {char * 60}")

    def blank(self):
        """Print blank line."""
        print()

    def section(self, title: str):
        """Print a section header within a stage."""
        print(f"\n  --- {title} ---\n")

    def key_value(self, key: str, value: Any, indent: int = 2):
        """Print a key-value pair."""
        spaces = " " * indent
        print(f"{spaces}{key}: {value}")

    def list_items(self, items: List[str], numbered: bool = False, indent: int = 4):
        """Print a list of items."""
        spaces = " " * indent
        for i, item in enumerate(items, 1):
            if numbered:
                print(f"{spaces}{i}. {item}")
            else:
                print(f"{spaces}- {item}")

    def table_row(self, columns: List[str], widths: Optional[List[int]] = None):
        """Print a table row."""
        if widths is None:
            widths = [20] * len(columns)

        row = "  "
        for col, width in zip(columns, widths):
            row += f"{str(col):<{width}}"
        print(row)

    # =========================================================================
    # Final Summary (for research outputs)
    # =========================================================================

    def research_summary(
        self,
        title: str,
        items: List[Dict[str, Any]],
        item_key: str = "question",
        score_key: Optional[str] = "priority_score"
    ):
        """
        Print research output summary (e.g., final questions).

        Args:
            title: Summary title
            items: List of result items
            item_key: Key for main item text
            score_key: Key for score (optional)
        """
        print()
        print(self._line("="))
        print(f"  {title}")
        print(self._line("="))
        print()

        for i, item in enumerate(items, 1):
            text = str(item.get(item_key, str(item)))

            # Wrap long text to fit console width (preserve full content)
            max_line_width = self.WIDTH - 6  # Account for "  N. " prefix
            if len(text) <= max_line_width:
                print(f"  {i}. {text}")
            else:
                # Print first line with number
                print(f"  {i}. {text[:max_line_width]}")
                # Print continuation lines with proper indentation
                remaining = text[max_line_width:]
                indent = "     "  # 5 spaces to align with text after "  N. "
                while remaining:
                    chunk = remaining[:max_line_width]
                    remaining = remaining[max_line_width:]
                    print(f"{indent}{chunk}")

            if score_key and score_key in item:
                score = item[score_key]
                print(f"     Score: {score}")
            print()

        print(self._line("="))
        print()


# ============================================================================
# Convenience Functions
# ============================================================================

def create_ui(team: str, mode: str, **kwargs) -> ConsoleUI:
    """
    Factory function to create ConsoleUI.

    Args:
        team: Team name
        mode: Mode name
        **kwargs: Additional ConsoleUI arguments

    Returns:
        ConsoleUI instance
    """
    return ConsoleUI(team=team, mode=mode, **kwargs)


# Example usage when run directly
if __name__ == "__main__":
    # Demo the UI
    ui = ConsoleUI(team="IdeationTeam", mode="ModeNoWcNoHITL", total_stages=3)

    ui.workflow_header(
        topic="Agent-based modeling in macroeconomics and monetary policy",
        execution_id="demo123"
    )

    # Show API keys status
    ui.show_api_keys()

    ui.stage_start(1, "Literature Sourcing")
    ui.agent_result("TrendSurfer", items=15)
    ui.agent_result("TopicCrawler", items=1, errors=1)
    ui.agent_result("ScholarSearcher", items=1, errors=1)
    ui.agent_result("GreyScout", items=0, errors=2)
    ui.stage_complete(1, status="success", items=19, duration=301.45, output_file="literature_results.csv")

    ui.stage_start(2, "Research Question Refinement")
    ui.agent_result("Ideator", items=9, message="Generated 9 concepts")
    ui.agent_result("Refiner", items=8, message="Formulated 8 questions")
    ui.stage_complete(2, status="success", items=8, duration=40.20, output_file="refinement_results.json")

    ui.stage_start(3, "Question Integration")
    ui.agent_result("Contextualizer", items=8)
    ui.agent_result("Finalizer", items=5)
    ui.stage_complete(3, status="success", items=5, duration=43.08, output_file="finalized_questions.json")

    ui.workflow_complete(
        success=True,
        stages_completed=3,
        total_errors=4,
        output_files=["finalized_research_questions_automated.json"],
        summary={
            "Literature Papers": 19,
            "Refined Questions": 8,
            "Final Questions": 5
        }
    )

    # Demo observability summary with mock data
    mock_observability = {
        "schema_version": "1.0.0",
        "llm": {
            "total_calls": 23, "total_prompt_tokens": 14449,
            "total_completion_tokens": 7025, "total_tokens": 21474,
            "total_cost_usd": 0.0064, "avg_latency_seconds": 6.45, "error_count": 0
        },
        "tools": {
            "total_calls": 8, "success_count": 8, "error_count": 0,
            "success_rate": 1.0, "avg_latency_seconds": 2.14
        },
        "by_agent": {
            "Contextualizer": {"llm": {"total_calls": 1, "total_tokens": 2002, "total_cost_usd": 0.0008, "avg_latency_seconds": 23.0, "error_count": 0}, "tools": {"total_calls": 0, "avg_latency_seconds": 0.0, "error_count": 0}},
            "Finalizer": {"llm": {"total_calls": 1, "total_tokens": 2321, "total_cost_usd": 0.0008, "avg_latency_seconds": 17.8, "error_count": 0}, "tools": {"total_calls": 0, "avg_latency_seconds": 0.0, "error_count": 0}},
            "GreyScout": {"llm": {"total_calls": 1, "total_tokens": 383, "total_cost_usd": 0.0001, "avg_latency_seconds": 3.0, "error_count": 0}, "tools": {"total_calls": 2, "avg_latency_seconds": 1.5, "error_count": 0}},
            "Ideator": {"llm": {"total_calls": 1, "total_tokens": 4438, "total_cost_usd": 0.0014, "avg_latency_seconds": 32.4, "error_count": 0}, "tools": {"total_calls": 0, "avg_latency_seconds": 0.0, "error_count": 0}},
            "Orchestrator": {"llm": {"total_calls": 15, "total_tokens": 6572, "total_cost_usd": 0.0011, "avg_latency_seconds": 1.0, "error_count": 0}, "tools": {"total_calls": 0, "avg_latency_seconds": 0.0, "error_count": 0}},
            "Refiner": {"llm": {"total_calls": 1, "total_tokens": 4589, "total_cost_usd": 0.0018, "avg_latency_seconds": 45.0, "error_count": 0}, "tools": {"total_calls": 0, "avg_latency_seconds": 0.0, "error_count": 0}},
            "ScholarSearcher": {"llm": {"total_calls": 1, "total_tokens": 378, "total_cost_usd": 0.0001, "avg_latency_seconds": 5.5, "error_count": 0}, "tools": {"total_calls": 2, "avg_latency_seconds": 2.0, "error_count": 0}},
            "TopicCrawler": {"llm": {"total_calls": 1, "total_tokens": 398, "total_cost_usd": 0.0001, "avg_latency_seconds": 2.6, "error_count": 0}, "tools": {"total_calls": 2, "avg_latency_seconds": 1.0, "error_count": 0}},
            "TrendSurfer": {"llm": {"total_calls": 1, "total_tokens": 393, "total_cost_usd": 0.0001, "avg_latency_seconds": 4.2, "error_count": 0}, "tools": {"total_calls": 2, "avg_latency_seconds": 1.5, "error_count": 0}},
        },
        "by_stage": {
            "Sourcing": {"llm": {"total_calls": 19, "total_tokens": 8124, "total_cost_usd": 0.0017, "avg_latency_seconds": 1.6, "error_count": 0}, "tools": {"total_calls": 8, "avg_latency_seconds": 1.5, "error_count": 0}},
            "Refinement": {"llm": {"total_calls": 2, "total_tokens": 9027, "total_cost_usd": 0.0031, "avg_latency_seconds": 38.7, "error_count": 0}, "tools": {"total_calls": 0, "avg_latency_seconds": 0.0, "error_count": 0}},
            "Integration": {"llm": {"total_calls": 2, "total_tokens": 4323, "total_cost_usd": 0.0016, "avg_latency_seconds": 20.4, "error_count": 0}, "tools": {"total_calls": 0, "avg_latency_seconds": 0.0, "error_count": 0}},
        }
    }
    ui.observability_summary(mock_observability)
