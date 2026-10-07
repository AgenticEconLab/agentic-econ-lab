# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Auto-input utility for HITL workflow testing.

This module provides functions that enable Human-in-the-Loop (HITL) workflows to run
automatically for testing purposes by using default values when no human input is
provided within a configurable timeout period.

Environment Variables:
    AUTO_HITL_MODE: Set to 'true', '1', or 'yes' to enable fully automatic mode
    AUTO_HITL_TIMEOUT: Timeout in seconds before using default (default: 120)

Usage:
    from shared.auto_input import auto_input, get_default

    # Instead of: response = input("Enter topic: ").strip()
    response = auto_input("Enter topic: ", default=get_default("research_topic")).strip()
"""

import os
import sys
import threading
import queue
from typing import Optional

# Environment variable names
AUTO_HITL_MODE_VAR = "AUTO_HITL_MODE"
AUTO_HITL_TIMEOUT_VAR = "AUTO_HITL_TIMEOUT"
DEFAULT_TIMEOUT = 120

# Default values registry for different input types across all teams
DEFAULTS = {
    # ========== IdeationTeam Defaults ==========
    # Orchestrator
    "research_topic": "Agent-based and heterogeneous-agent modeling in macroeconomics and monetary policy",  # dev-stage default: evaluator expertise; release -> field-neutral

    # 1-SourcingStage: collect_human_feedback()
    "relevant_papers": "",
    "irrelevant_papers": "",
    "missing_topics": "",
    "additional_keywords": "",
    "general_comments": "",

    # 2-RefinementStage: collect_human_feedback()
    "promising_items": "",
    "weak_items": "",
    "missing_angles": "",
    "focus_directions": "",

    # 3-IntegrationStage: collect_integration_feedback()
    "questions_to_merge": "",
    "questions_to_discard": "",
    "questions_to_prioritize": "",
    "additional_guidance": "",

    # ========== LiteratureTeam Defaults ==========
    "questions_file_path": "",  # Empty to use sample questions
    "approve_stage": "yes",     # Approve and proceed
    "refine_choice": "4",       # Re-run with same parameters
    "retry_choice": "no",       # Don't retry on failure
    "go_back_choice": "2",      # Exit pipeline (safest default)
    "new_max_papers": "15",     # Default papers per question
    "additional_question": "",  # No additional question
    "add_search_terms": "no",   # Don't add terms

    # ========== ModelTeam Defaults ==========
    "hitl_decision": "A",       # Approve
    "skip_confirm": "y",        # Confirm skip if chosen
    "feedback_text": "",        # No feedback
    "model_research_questions_path": "finalized_research_questions_automated.json",
    # Stage-entry file-path defaults (used by __main__ blocks when stages are
    # invoked directly). Empty means the script's own file-existence check
    # will abort fast instead of hanging on input() when AUTO_HITL_TIMEOUT
    # fires with no operator present.
    "literature_file_path": "literature_batch.json",
    "theory_file_path": "theory_output.json",
    "design_file_path": "model_design_output.json",

    # ========== LiteratureTeam Stage-Entry Defaults ==========
    # questions_file_path already defined above; these two are new.
    "batch_file_path": "literature_batch.json",
    "gap_file_path": "gap_analysis_results.json",

    # ========== DataTeam Defaults ==========
    "budget_review": "approved",
    "query_review": "approved",
    "quality_review": "approved",
    "compliance_review": "approved",
    "final_approval": "approved",
    "data_research_question": "Analyze the relationship between institutional ownership and stock returns using premium data sources.",
    "budget_limit": "",
    "user_uploaded_file_path": "",   # empty triggers example_data.csv creation

    # ========== General Defaults ==========
    "yes_no": "yes",
    "confirm": "y",
    "empty": "",
}


def is_auto_mode() -> bool:
    """Check if AUTO_HITL_MODE is enabled."""
    return os.getenv(AUTO_HITL_MODE_VAR, "").lower() in ("true", "1", "yes")


# HITL resolution modes (who answers a WithHITL checkpoint). Two independent axes:
# MODE (NoHITL/WithHITL = does the graph have checkpoints) is structural; RESOLUTION
# (below) only applies to WithHITL workflows. See README.md, "Running".
HITL_MODE_VAR = "AEL_HITL_MODE"
HITL_MODES = ("interactive", "auto", "llm_economist")


def get_hitl_mode() -> str:
    """Resolve the HITL resolution mode: 'interactive' | 'auto' | 'llm_economist'.

    - ``interactive`` (default): a real human answers (deployment / a real study).
    - ``auto``: return the canned default — an INFRASTRUCTURE smoke-test with no
      economic meaning. Off by default (``AUTO_HITL_MODE=false``).
    - ``llm_economist``: a 3-model open-weight committee answers with a reasoned 2/3
      vote — used for the paper's worked examples / meaningful WithHITL test runs.

    Selected by ``AEL_HITL_MODE``. Back-compat: ``AUTO_HITL_MODE=true`` maps to
    ``auto`` when ``AEL_HITL_MODE`` is unset.
    """
    mode = os.getenv(HITL_MODE_VAR, "").strip().lower()
    if mode in HITL_MODES:
        return mode
    if is_auto_mode():
        return "auto"
    return "interactive"


def get_timeout() -> int:
    """Get the configured timeout in seconds."""
    try:
        return int(os.getenv(AUTO_HITL_TIMEOUT_VAR, str(DEFAULT_TIMEOUT)))
    except ValueError:
        return DEFAULT_TIMEOUT


def get_default(key: str, fallback: str = "") -> str:
    """
    Get a default value from the registry.

    Args:
        key: The key for the default value
        fallback: Fallback value if key not found

    Returns:
        The default value or fallback
    """
    return DEFAULTS.get(key, fallback)


def timed_input(prompt: str, timeout: int, default: str) -> str:
    """
    Input with timeout that waits indefinitely once the user starts typing.

    Two-phase behavior:
      Phase 1 (idle): Wait up to ``timeout`` seconds for the first keypress.
                      If nothing is pressed, return ``default``.
      Phase 2 (typing): Once a keypress is detected, wait with no time limit
                        until the user presses Enter.

    Uses ``input()`` in a daemon thread for line reading (handles echo, line
    editing, encoding) and platform-specific keypress detection (msvcrt on
    Windows, select on Unix) only to distinguish idle vs. typing.

    Args:
        prompt: The prompt to display
        timeout: Seconds to wait for the *first* keypress before using default
        default: Default value to use if no keypress within timeout

    Returns:
        User input or default value
    """
    import time as _time

    default_display = default if len(default) <= 60 else default[:57] + "..."
    print(f"[Timeout: {timeout}s | Default: '{default_display}']")

    result_queue: queue.Queue = queue.Queue()

    def _reader():
        try:
            result_queue.put(input(prompt))
        except (EOFError, Exception):
            result_queue.put(default)

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()

    # --- Obtain a platform-specific "has the user pressed a key?" function ---
    _has_keypress = None
    try:
        import msvcrt
        _has_keypress = msvcrt.kbhit          # Windows
    except ImportError:
        try:
            import select
            if hasattr(sys.stdin, "fileno"):
                def _check_stdin():
                    try:
                        return bool(select.select([sys.stdin], [], [], 0)[0])
                    except Exception:
                        return False
                _has_keypress = _check_stdin   # Unix / Mac
        except ImportError:
            pass

    interval = 0.1
    elapsed = 0.0
    user_typing = False

    while True:
        # Check whether input() has returned (user pressed Enter)
        try:
            return result_queue.get_nowait()
        except queue.Empty:
            pass

        if not user_typing:
            # Detect first keypress
            if _has_keypress is not None:
                try:
                    if _has_keypress():
                        user_typing = True
                        continue          # skip timeout, enter Phase 2
                except Exception:
                    pass

            # Timeout only applies while idle
            if elapsed >= timeout:
                print(
                    f"\n[AUTO_HITL] Timeout ({timeout}s) reached. "
                    f"Using default: '{default_display}'"
                )
                return default

            elapsed += interval

        _time.sleep(interval)


def auto_input(
    prompt: str,
    default: str = "",
    input_type: str = "general",
    timeout: Optional[int] = None,
    context: Optional[dict] = None,
) -> str:
    """
    Smart input function that supports auto-mode for HITL testing.

    In AUTO_HITL_MODE (fully automatic):
        - Immediately returns the default value
        - Prints the prompt and default for logging

    In normal mode:
        - Waits for user input with timeout
        - Returns default if timeout is reached

    Args:
        prompt: The prompt to display to the user
        default: Default value to use in auto mode or on timeout
        input_type: Type of input for logging (not used for lookup)
        timeout: Optional custom timeout in seconds
        context: Optional structured material under review (e.g. the papers/questions
            being judged). Passed to the LLM-Economist committee so its decision/feedback
            is grounded in the actual content; ignored in interactive/auto modes.

    Returns:
        User input or default value
    """
    mode = get_hitl_mode()

    # LLM-Economist committee: a reasoned human stand-in (testing / worked examples).
    if mode == "llm_economist":
        print(f"{prompt}")
        try:
            from shared.llm_economist import resolve_checkpoint
            answer = resolve_checkpoint(prompt, default=default, input_type=input_type, context=context)
            print(f"[LLM_ECONOMIST] committee decision: '{answer}'")
            return answer
        except Exception as e:  # never break a run on committee failure
            print(f"[LLM_ECONOMIST] committee unavailable ({e}); using default: '{default}'")
            return default

    # Infrastructure smoke-test: return the canned default (no economic meaning).
    if mode == "auto":
        print(f"{prompt}")
        print(f"[AUTO_HITL_MODE] Using default: '{default}'")
        return default

    # Interactive: a real human answers, with a timeout fallback to default.
    actual_timeout = timeout if timeout is not None else get_timeout()
    return timed_input(prompt, actual_timeout, default)


def auto_multiline_input(
    prompt: str,
    end_marker: str = "",
    default: str = "",
    timeout: Optional[int] = None,
    context: Optional[dict] = None,
) -> str:
    """
    Multi-line input that supports auto-mode.

    For inputs that expect multiple lines (like feedback text),
    this function handles the collection appropriately.

    Args:
        prompt: The prompt to display
        end_marker: What signifies end of input (empty line by default)
        default: Default value in auto mode
        timeout: Custom timeout in seconds
        context: Optional structured material under review, passed to the committee so
            its generated feedback is grounded in the actual content.

    Returns:
        Collected multi-line input or default
    """
    mode = get_hitl_mode()

    # LLM-Economist committee. Multi-line prompts are review/feedback fields (empty
    # default) — the committee generates reasoned, context-grounded economist feedback.
    if mode == "llm_economist":
        print(f"{prompt}")
        try:
            from shared.llm_economist import resolve_checkpoint
            answer = resolve_checkpoint(prompt, default=default, input_type="general", context=context)
            print(f"[LLM_ECONOMIST] committee decision (multi-line): '{answer}'")
            return answer
        except Exception as e:
            print(f"[LLM_ECONOMIST] committee unavailable ({e}); using default: '{default}'")
            return default

    if mode == "auto":
        print(f"{prompt}")
        print(f"[AUTO_HITL_MODE] Using default for multi-line input: '{default}'")
        return default

    # For multi-line input in normal mode, we still need timeout support
    # But multi-line input is complex with timeouts, so we use a simpler approach
    actual_timeout = timeout if timeout is not None else get_timeout()

    # Use single-line timed input for the first line
    first_line = timed_input(prompt, actual_timeout, default)

    # If we got the default (timeout), return it
    if first_line == default and default != "":
        return default

    # If empty line, return what we have
    if first_line == "":
        return default

    return first_line


# Convenience functions for common input patterns
def auto_yes_no(prompt: str, default: str = "yes") -> str:
    """Yes/no input with auto-mode support."""
    return auto_input(prompt, default=default, input_type="yes_no")


def auto_choice(prompt: str, default: str, choices: list = None) -> str:
    """Multiple choice input with auto-mode support."""
    return auto_input(prompt, default=default, input_type="choice")


def auto_path(prompt: str, default: str = "") -> str:
    """File path input with auto-mode support."""
    return auto_input(prompt, default=default, input_type="path")
