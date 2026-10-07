# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Experimentation Stage — comparative statics around the calibrated point (§4.5 BatchRunner,
VersionManager, Optimizer, DocuAgent).

For each validated module: sweep every calibrated parameter ±10%, re-solve the steady state,
and report the percent response of every steady-state variable. Deterministic numerics;
non-convergent variants are reported, never interpolated.

VersionManager (deterministic, no LLM) content-hashes each derived module's source and its
calibrated parameters into a model_version_id, so BatchRunner's results can be traced to
exactly the code + parameters that produced them. Optimizer (deterministic, no LLM) times
each parameter sweep as it runs and reports the real wall-clock profile — it diagnoses where
time goes; it does not rewrite code, since free-hand code edits would break the "no LLM code"
guarantee the whole team is built on. DocuAgent narrates the combined generation + validation
+ experimentation + version record into one technical-documentation summary, under the same
"describe only what's given" guardrail as Coder/Debugger/BatchRunner.

Input:  generation_output.json + validation_output.json
Output: experimentation_output.json, code_version_manifest.json

This file is byte-identical across CodeTeam mode directories (mode-copy invariant).
"""

import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path as _Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

_agents_dir = _Path(__file__).resolve().parent.parent.parent.parent
if str(_agents_dir) not in sys.path:
    sys.path.insert(0, str(_agents_dir))

from shared.auto_input import auto_input
from shared.llm import LLMClient
from shared.observability import MetricsCollector
from shared.reliability.state_guard import StateGuard

from CodeTeam.ael.code_harness import (
    ExperimentResult,
    GenerationResult,
    ValidationResult,
    comparative_statics,
)

load_dotenv()


# ========== AGENTS ==========

class VersionManager:
    """Content-hash versioning of derived modules. Deterministic, no LLM: a model
    version is defined by its exact generated source plus its exact calibrated
    parameters, not by a mutable label."""

    STAGE_FILES = ["1-CodeGenerationStage.py", "2-ValidationStage.py", "3-ExperimentationStage.py"]

    def __init__(self, collector: Optional[MetricsCollector] = None):
        self.agent_name = "VersionManager"
        self.collector = collector
        self._guard = StateGuard()

    def _fingerprint_pipeline_code(self, stage_dir: _Path) -> Dict:
        file_hashes = {}
        for fname in self.STAGE_FILES:
            fpath = stage_dir / fname
            if fpath.exists():
                file_hashes[fname] = hashlib.sha256(fpath.read_bytes()).hexdigest()[:16]
            else:
                file_hashes[fname] = "file_not_found"
        combined = hashlib.sha256("".join(file_hashes.values()).encode()).hexdigest()[:16]
        return {"stage_file_hashes": file_hashes, "pipeline_fingerprint": combined}

    def version_model(self, gen: GenerationResult) -> Dict:
        """One model_version_id per derived module: hash(module source) x hash(calibrated
        parameters). Identical code with different calibrated values gets a different id,
        and vice versa — either change is a different, traceable version."""
        module_hash = self._guard.sign_output(gen.module_text, stage_name=f"module:{gen.model_title}")
        params_hash = self._guard.sign_output(gen.parameters, stage_name=f"params:{gen.model_title}")
        model_version_id = hashlib.sha256(f"{module_hash}{params_hash}".encode()).hexdigest()[:16]
        return {
            "model_title": gen.model_title,
            "module_hash": module_hash,
            "parameters_hash": params_hash,
            "model_version_id": model_version_id,
            "n_parameters": len(gen.parameters),
        }

    def build_manifest(self, gens: List[GenerationResult], stage_dir: _Path) -> Dict:
        print(f"\n[{self.agent_name}] Versioning {len(gens)} derived module(s)...")
        versions = [self.version_model(g) for g in gens]
        manifest = {
            "generated_at": datetime.now().isoformat(),
            "pipeline_code_fingerprint": self._fingerprint_pipeline_code(stage_dir),
            "model_versions": versions,
        }
        for v in versions:
            print(f"  {v['model_title'][:50]}: version {v['model_version_id']}")
        return manifest


class Optimizer:
    """Performance profiler over the comparative-statics sweep. Deterministic, no
    LLM, and no code rewriting: it reports real wall-clock timing and convergence
    diagnostics from the sweep BatchRunner already runs, so a researcher knows
    where optimization effort would matter. It does not modify the derived module
    (free-hand edits to LLM-adjacent code would break the "no LLM code" guarantee
    the team is built on)."""

    def __init__(self, collector: Optional[MetricsCollector] = None):
        self.agent_name = "Optimizer"
        self.collector = collector

    def profile(self, timed_sweeps: List[Dict], exps: List[ExperimentResult]) -> Dict:
        print(f"\n[{self.agent_name}] Profiling {len(timed_sweeps)} sweep(s)...")
        if not timed_sweeps:
            return {"models_profiled": 0, "notes": ["no completed sweeps to profile"]}

        durations = [t["duration_sec"] for t in timed_sweeps]
        total = sum(durations)
        slowest = max(timed_sweeps, key=lambda t: t["duration_sec"])

        n_statics = sum(len(e.statics) for e in exps)
        n_nonconvergent = sum(1 for e in exps for s in e.statics if not s.converged)

        profile = {
            "models_profiled": len(timed_sweeps),
            "total_sweep_time_sec": round(total, 4),
            "mean_sweep_time_sec": round(total / len(durations), 4),
            "max_sweep_time_sec": round(max(durations), 4),
            "slowest_model": slowest["model_title"],
            "per_model_timing_sec": {t["model_title"]: round(t["duration_sec"], 4) for t in timed_sweeps},
            "total_parameter_perturbations": n_statics,
            "nonconvergent_perturbations": n_nonconvergent,
            "nonconvergent_rate": round(n_nonconvergent / n_statics, 4) if n_statics else None,
        }
        print(f"  Total sweep time: {profile['total_sweep_time_sec']:.2f}s "
              f"(slowest: {profile['slowest_model'][:40]})")
        if n_statics:
            print(f"  Non-convergent perturbations: {n_nonconvergent}/{n_statics} "
                  f"({profile['nonconvergent_rate']:.1%})")
        return profile


class DocuAgent:
    """Narrates the combined generation + validation + experimentation + version
    record into one technical-documentation summary; writes no code and invents
    nothing beyond what the summary states."""

    def __init__(self, openai_api_key: str = "", collector: Optional[MetricsCollector] = None):
        self.agent_name = "DocuAgent"
        self.llm = LLMClient(
            temperature=0.3,
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
            collector=collector,
            agent_name=self.agent_name,
        )

    def narrate(self, summary: str) -> str:
        raw = self.llm.invoke([
            {"role": "system", "content":
                "You write technical documentation for a deterministically derived "
                "economic-model implementation. Describe only what the summary states "
                "-- what was generated, validated, swept, and versioned; never invent "
                "capabilities, algorithms, or results not present in the summary."},
            {"role": "user", "content":
                f"Pipeline summary:\n{summary[:2800]}\n\n"
                "In <=200 words, write technical documentation covering: what module(s) "
                "were derived and from how many equations; what the validation checks "
                "found; what the comparative-statics sweep showed and how long it took; "
                "and each module's version identifier for provenance."},
        ], max_tokens=550)
        return str(raw).strip()


# ========== AGENT ==========

class BatchRunner:
    """Narrates the computed comparative statics; runs nothing itself."""

    def __init__(self, openai_api_key: str = "", collector: Optional[MetricsCollector] = None):
        self.agent_name = "BatchRunner"
        self.llm = LLMClient(
            temperature=0.3,
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
            collector=collector,
            agent_name=self.agent_name,
        )

    def narrate(self, statics_dump: str) -> str:
        raw = self.llm.invoke([
            {"role": "system", "content":
                "You summarize computed comparative statics of an economic model. Repeat "
                "the given numbers verbatim; interpret directions economically."},
            {"role": "user", "content":
                f"Comparative statics:\n{statics_dump[:2400]}\n\nIn <=150 words: which "
                "parameters move which steady-state variables, and in which direction?"},
        ], max_tokens=400)
        return str(raw).strip()


# ========== ORCHESTRATOR ==========

class ExperimentationOrchestrator:
    """Runs comparative statics for every validated module."""

    def __init__(self, openai_api_key: str = "", collector: Optional[MetricsCollector] = None,
                 quiet: bool = False):
        self.collector = collector
        self.quiet = quiet
        self.batch_runner = BatchRunner(openai_api_key, collector=collector)
        self.version_manager = VersionManager(collector=collector)
        self.optimizer = Optimizer(collector=collector)
        self.docu_agent = DocuAgent(openai_api_key, collector=collector)
        self.results: List[ExperimentResult] = []
        self.narrative: str = ""
        self.version_manifest: Dict = {}
        self.performance_profile: Dict = {}
        self.technical_documentation: str = ""

    def _print(self, msg: str):
        if not self.quiet:
            print(msg)

    def run_experimentation_pipeline(self, generation_output: Dict, validation_output: Dict,
                                     workdir: str = "generated_models") -> List[ExperimentResult]:
        gens = [GenerationResult(**g) for g in (generation_output or {}).get("results") or []]
        vals = [ValidationResult(**v) for v in (validation_output or {}).get("results") or []]
        self._print(f"[ExperimentationStage] Comparative statics for {len(gens)} model(s)...")
        self.results = []
        timed_sweeps: List[Dict] = []
        for i, (gen, val) in enumerate(zip(gens, vals)):
            t0 = time.perf_counter()
            exp = comparative_statics(gen, val, workdir)
            duration = time.perf_counter() - t0
            timed_sweeps.append({"model_title": gen.model_title, "duration_sec": duration})
            self._print(f"  [{i + 1}] {str(gen.model_title)[:50]}: {exp.verdict} "
                        f"({len(exp.statics)} sweeps, {duration:.3f}s)")
            self.results.append(exp)

        try:
            dump = json.dumps([e.model_dump() for e in self.results], default=str)
            self.narrative = self.batch_runner.narrate(dump)
        except Exception as e:
            self._print(f"[ExperimentationStage] Narration skipped (non-critical): {e}")

        # VersionManager: content-hash version each derived module (deterministic).
        stage_dir = _Path(__file__).resolve().parent
        self.version_manifest = self.version_manager.build_manifest(gens, stage_dir)

        # Optimizer: profile the real timing already measured above (deterministic).
        self.performance_profile = self.optimizer.profile(timed_sweeps, self.results)

        # DocuAgent: narrate the combined record.
        try:
            doc_summary = json.dumps({
                "n_models": len(gens),
                "generation": [{"title": g.model_title, "verdict": g.verdict,
                                "n_equations": g.n_equations, "n_parseable": g.n_parseable}
                               for g in gens],
                "validation": [{"title": g.model_title, "verdict": v.verdict,
                                "n_checks": len(v.checks),
                                "n_passed": sum(1 for c in v.checks if c.passed)}
                               for g, v in zip(gens, vals)],
                "experimentation": [{"verdict": e.verdict, "n_sweeps": len(e.statics)}
                                     for e in self.results],
                "performance_profile": self.performance_profile,
                "model_versions": self.version_manifest.get("model_versions", []),
            }, default=str)
            self.technical_documentation = self.docu_agent.narrate(doc_summary)
        except Exception as e:
            self._print(f"[ExperimentationStage] Documentation skipped (non-critical): {e}")

        return self.results

    def save_experimentation_output(self, filename: str = "experimentation_output.json") -> str:
        if not self.results:
            raise RuntimeError("run_experimentation_pipeline first")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"results": [e.model_dump() for e in self.results],
                       "narrative": self.narrative,
                       "version_manifest": self.version_manifest,
                       "performance_profile": self.performance_profile,
                       "technical_documentation": self.technical_documentation},
                      f, indent=2, default=str)
        self._print(f"[ExperimentationStage] Output saved to {filename}")
        return filename

    def save_version_manifest(self, filename: str = "code_version_manifest.json") -> str:
        if not self.version_manifest:
            self._print("No version manifest to save")
            return filename
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.version_manifest, f, indent=2, default=str)
        self._print(f"[ExperimentationStage] Version manifest saved to {filename}")
        return filename


if __name__ == "__main__":
    gen_path = auto_input("Path to generation_output.json: ",
                          default="generation_output.json").strip()
    val_path = auto_input("Path to validation_output.json: ",
                          default="validation_output.json").strip()
    with open(gen_path, "r", encoding="utf-8") as f:
        generation_output = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        validation_output = json.load(f)
    orch = ExperimentationOrchestrator()
    orch.run_experimentation_pipeline(generation_output, validation_output)
    orch.save_experimentation_output()
    orch.save_version_manifest()
