# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Validation Stage — executed check battery against the derived modules (§4.5 TestSuite +
Debugger).

Checks per module: compiles, parameters complete, residuals finite, steady state solves
(max residual < 1e-6), parameter sensitivity, system complete. A failing derived module is a
refusal to certify — there is no LLM code to debug. Only a module derived
from the whole equation system is `validated`; a solving fragment is `fragment_solves`.

Input:  generation_output.json
Output: validation_output.json

This file is byte-identical across CodeTeam mode directories (mode-copy invariant).
"""

import json
import os
import sys
from pathlib import Path as _Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

_agents_dir = _Path(__file__).resolve().parent.parent.parent.parent
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

from shared.auto_input import auto_input
from shared.llm import LLMClient
from shared.observability import MetricsCollector

from CodeTeam.ael.code_harness import GenerationResult, ValidationResult, cleanup_workdir, validate_module

load_dotenv()


# ========== AGENT ==========

class Debugger:
    """Narrates check failures; certifies nothing itself."""

    def __init__(self, openai_api_key: str = "", collector: Optional[MetricsCollector] = None):
        self.agent_name = "Debugger"
        self.llm = LLMClient(
            temperature=0.3,
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
            collector=collector,
            agent_name=self.agent_name,
        )

    def narrate(self, checks_dump: str) -> str:
        raw = self.llm.invoke([
            {"role": "system", "content":
                "You summarize executed validation checks of a derived economic-model "
                "module. Only restate the given results; suggest what a failing check "
                "implies about the model specification."},
            {"role": "user", "content":
                f"Executed checks:\n{checks_dump[:2400]}\n\nIn <=120 words: the verdict "
                "and what any failures imply."},
        ], max_tokens=350)
        return str(raw).strip()


# ========== ORCHESTRATOR ==========

class CodeValidationOrchestrator:
    """Runs the executed check battery for every generated module."""

    def __init__(self, openai_api_key: str = "", collector: Optional[MetricsCollector] = None,
                 quiet: bool = False):
        self.collector = collector
        self.quiet = quiet
        self.debugger = Debugger(openai_api_key, collector=collector)
        self.results: List[ValidationResult] = []
        self.narrative: str = ""

    def _print(self, msg: str):
        if not self.quiet:
            print(msg)

    def run_validation_pipeline(self, generation_output: Dict,
                                workdir: str = "generated_models") -> List[ValidationResult]:
        gens = [GenerationResult(**g) for g in (generation_output or {}).get("results") or []]
        self._print(f"[ValidationStage] Executing check battery for {len(gens)} module(s)...")
        self.results = []
        for i, gen in enumerate(gens):
            val = validate_module(gen, workdir)
            n_pass = sum(1 for c in val.checks if c.passed)
            self._print(f"  [{i + 1}] {str(gen.model_title)[:50]}: {val.verdict} "
                        f"({n_pass}/{len(val.checks)} checks passed)")
            self.results.append(val)
        cleanup_workdir(workdir)   # drop execution scratch + __pycache__, keep model_N.py

        try:
            dump = json.dumps([v.model_dump() for v in self.results], default=str)
            self.narrative = self.debugger.narrate(dump)
        except Exception as e:
            self._print(f"[ValidationStage] Narration skipped (non-critical): {e}")
        return self.results

    def save_validation_output(self, filename: str = "validation_output.json") -> str:
        if not self.results:
            raise RuntimeError("run_validation_pipeline first")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"results": [v.model_dump() for v in self.results],
                       "narrative": self.narrative}, f, indent=2, default=str)
        self._print(f"[ValidationStage] Output saved to {filename}")
        return filename


if __name__ == "__main__":
    gen_path = auto_input("Path to generation_output.json: ",
                          default="generation_output.json").strip()
    with open(gen_path, "r", encoding="utf-8") as f:
        generation_output = json.load(f)
    orch = CodeValidationOrchestrator()
    orch.run_validation_pipeline(generation_output)
    orch.save_validation_output()
