# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Code Generation Stage — modules DERIVED from the parsed equation system (§4.5 Coder).

The derivation is fully deterministic (calib_harness parser -> sympy -> pycode template):
no LLM text ever enters the generated module. The Coder agent narrates the generated
interface and the refusals; models whose equations don't parse get `ungenerable` with the
parser's reasons — never a hallucinated implementation. (LLM-freeform codegen remains
deferred behind sandbox hardening.)

Input:  model_design_output.json (+ calibration_output.json for parameter values)
Output: generation_output.json (+ generated_models/*.py)

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

from CodeTeam.ael.code_harness import GenerationResult, generate_module

load_dotenv()


# ========== AGENT ==========

class Coder:
    """Narrates the derived module's interface; writes no code."""

    def __init__(self, openai_api_key: str = "", collector: Optional[MetricsCollector] = None):
        self.agent_name = "Coder"
        self.llm = LLMClient(
            temperature=0.3,
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
            collector=collector,
            agent_name=self.agent_name,
        )

    def narrate(self, summary: str) -> str:
        raw = self.llm.invoke([
            {"role": "system", "content":
                "You document a deterministically derived economic-model module. Describe "
                "only what the summary states; never invent capabilities."},
            {"role": "user", "content":
                f"Generation summary:\n{summary[:2400]}\n\n"
                "In <=120 words: what was generated, what was refused and why."},
        ], max_tokens=350)
        return str(raw).strip()


# ========== ORCHESTRATOR ==========

class CodeGenerationOrchestrator:
    """Derives one module per formal model; honest per-model verdicts."""

    def __init__(self, openai_api_key: str = "", collector: Optional[MetricsCollector] = None,
                 quiet: bool = False):
        self.collector = collector
        self.quiet = quiet
        self.coder = Coder(openai_api_key, collector=collector)
        self.results: List[GenerationResult] = []
        self.narrative: str = ""

    def _print(self, msg: str):
        if not self.quiet:
            print(msg)

    def run_generation_pipeline(self, model_design_output: Dict,
                                calibration_output: Optional[Dict] = None,
                                modules_dir: str = "generated_models") -> List[GenerationResult]:
        models = [m for m in (model_design_output or {}).get("formal_models") or []
                  if isinstance(m, dict)]
        self._print(f"[CodeGenerationStage] Deriving modules for {len(models)} model(s)...")
        os.makedirs(modules_dir, exist_ok=True)
        self.results = []
        for i, model in enumerate(models):
            gen = generate_module(model, calibration_output)
            if gen.module_text:
                path = os.path.join(modules_dir, f"model_{i + 1}.py")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(gen.module_text)
                gen.notes.append(f"module written to {path}")
            self._print(f"  [{i + 1}] {str(gen.model_title)[:50]}: {gen.verdict}"
                        + (f" ({gen.reason})" if gen.reason else ""))
            self.results.append(gen)

        # A manifest so the published folder reads as a code deliverable, not a dump
        lines = ["# Generated model modules\n",
                 "Deterministically derived from each formal model's parsed equation system "
                 "(sympy -> numpy/scipy); no free-hand LLM code. Run any module directly "
                 "(`python model_N.py`) to solve and print its steady state.\n"]
        for i, r in enumerate(self.results):
            lines.append(f"- `model_{i + 1}.py` — {r.model_title}: verdict `{r.verdict}`"
                         f" ({r.n_parseable}/{r.n_equations} equations parsed"
                         + (f"; {r.reason}" if r.reason else "") + ")")
        try:
            with open(os.path.join(modules_dir, "README.md"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except OSError as e:
            self._print(f"[CodeGenerationStage] README skipped: {e}")

        try:
            summary = json.dumps([{k: v for k, v in r.model_dump().items()
                                   if k != "module_text"} for r in self.results], default=str)
            self.narrative = self.coder.narrate(summary)
        except Exception as e:
            self._print(f"[CodeGenerationStage] Narration skipped (non-critical): {e}")
        return self.results

    def save_generation_output(self, filename: str = "generation_output.json") -> str:
        if not self.results:
            raise RuntimeError("run_generation_pipeline first")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"results": [r.model_dump() for r in self.results],
                       "narrative": self.narrative}, f, indent=2, default=str)
        self._print(f"[CodeGenerationStage] Output saved to {filename}")
        return filename


if __name__ == "__main__":
    design_path = auto_input("Path to model_design_output.json: ",
                             default="../../ModelTeam/ael/ModeNoWcNoHITL/model_design_output.json").strip()
    calib_path = auto_input("Path to calibration_output.json (optional): ",
                            default="../../ModelTeam/ael/ModeNoWcNoHITL/calibration_output.json").strip()
    with open(design_path, "r", encoding="utf-8") as f:
        design = json.load(f)
    calib = None
    if calib_path and os.path.exists(calib_path):
        with open(calib_path, "r", encoding="utf-8") as f:
            calib = json.load(f)
    orch = CodeGenerationOrchestrator()
    orch.run_generation_pipeline(design, calib)
    orch.save_generation_output()
