# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Argument Correctness Scorer — Score tool argument quality from observability traces.

Analyzes tool call arguments to check if they are well-formed,
non-empty, and match expected patterns for each tool type.

No additional LLM calls needed — computed purely from observability data.
"""

from typing import Any, Dict, List, Optional


# Expected argument patterns per tool
EXPECTED_ARGS: Dict[str, Dict[str, str]] = {
    "arxiv_search": {
        "required": ["query"],
        "type_checks": {"query": "str", "max_results": "int"},
    },
    "web_search": {
        "required": ["query"],
        "type_checks": {"query": "str"},
    },
    "fred_fetch": {
        "required": ["series_id"],
        "type_checks": {"series_id": "str"},
    },
    "web_fetch": {
        "required": ["url"],
        "type_checks": {"url": "str"},
    },
    "yfinance_fetch": {
        "required": ["ticker"],
        "type_checks": {"ticker": "str"},
    },
    "semantic_scholar_search": {
        "required": ["query"],
        "type_checks": {"query": "str"},
    },
}


class ArgumentCorrectnessScorer:
    """
    Score tool argument quality from observability traces.

    Metrics:
    - Required args present: Were all required arguments provided?
    - Non-empty values: Were argument values non-empty/non-null?
    - Type correctness: Were argument types as expected?
    """

    def __init__(
        self,
        expected_args: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        self.expected_args = expected_args or EXPECTED_ARGS

    def score(
        self,
        execution_log: Dict[str, Any],
        team: str = "",
    ) -> Dict[str, Any]:
        """
        Score argument correctness from an execution log.

        Args:
            execution_log: Parsed execution_log.json with observability data.
            team: Team name (unused, for API consistency).

        Returns:
            Dict with argument_correctness score (0-1) and details.
        """
        obs = execution_log.get("observability", {})
        tool_calls = obs.get("tool_calls", [])

        if not tool_calls:
            return {
                "argument_correctness": 1.0,
                "details": "No tool calls to evaluate",
                "tool_calls_total": 0,
                "args_correct": 0,
                "args_missing": 0,
                "args_empty": 0,
            }

        total_checks = 0
        passed_checks = 0
        missing_args = 0
        empty_args = 0

        for call in tool_calls:
            tool_name = call.get("tool", call.get("name", ""))
            args = call.get("arguments", call.get("args", {}))
            if not isinstance(args, dict):
                args = {}

            spec = self.expected_args.get(tool_name)
            if spec is None:
                # Unknown tool — count all args as correct if non-empty
                if args:
                    total_checks += 1
                    non_empty = sum(1 for v in args.values() if v is not None and v != "")
                    if non_empty == len(args):
                        passed_checks += 1
                    else:
                        empty_args += len(args) - non_empty
                continue

            # Check required args
            required = spec.get("required", [])
            for req in required:
                total_checks += 1
                if req in args:
                    val = args[req]
                    if val is not None and val != "":
                        passed_checks += 1
                    else:
                        empty_args += 1
                else:
                    missing_args += 1

            # Check type correctness for present args
            type_checks = spec.get("type_checks", {})
            for arg_name, expected_type in type_checks.items():
                if arg_name in args and args[arg_name] is not None:
                    total_checks += 1
                    if self._check_type(args[arg_name], expected_type):
                        passed_checks += 1

        score = passed_checks / total_checks if total_checks > 0 else 1.0

        return {
            "argument_correctness": round(score, 4),
            "tool_calls_total": len(tool_calls),
            "total_checks": total_checks,
            "args_correct": passed_checks,
            "args_missing": missing_args,
            "args_empty": empty_args,
        }

    @staticmethod
    def _check_type(value: Any, expected: str) -> bool:
        """Check if a value matches the expected type string."""
        type_map = {
            "str": str,
            "int": int,
            "float": (int, float),
            "list": list,
            "dict": dict,
            "bool": bool,
        }
        expected_type = type_map.get(expected)
        if expected_type is None:
            return True
        return isinstance(value, expected_type)
