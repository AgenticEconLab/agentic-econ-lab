# Changelog

All notable changes to the Agentic Econ Lab (AEL) are documented here.

## [0.7.3] - 2026-10 — first public release

The code of the paper *Agentic Workflows for Economic Research: Design, Implementation, and
Evaluation*. The paper's 26 chained runs were made with version 0.7.2, which was not released;
this release differs from it as listed under "Changed after the runs". The report
number-consistency verdicts in the paper come from the checker in this release, applied offline
to the stored reports with `pipeline/recheck_consistency.py`.

### Pipeline
- Seven teams (Ideation, Literature, Model, Data, Code, Estimation, Reporting) chained by a
  cross-team orchestrator (`run_ael_pipeline.py`, `pipeline/`) with an artifact store whose
  hand-offs are HMAC-signed, per-team and overall budgets, and checkpoint/resume.
- Model–Data feasibility loop (`DataTeam/ael/feasibility/`): data requirements, a graded
  availability report, model revision requests, at most three cycles, and a re-grade of the
  final design.
- Human-in-the-loop checkpoints answered interactively or by an LLM-Economist committee of
  three open-weight models from distinct families (gpt-oss-20b, Phi-4-mini, Olmo-3.1-32B), each
  reviewing through a different lens; a decision passes with two thirds of the votes cast, and
  ballots are logged on a best-effort basis.

### Deterministic components
- Calibration harness (`ModelTeam/ael/calib_harness/`): equations parsed with sympy,
  closed-form structural moments, bounded least squares against cited targets, registered
  canonical models for simulated method of moments, and scoped verdicts (calibrated,
  point-calibrated, partially calibrated, simulation-calibrated, archetype-calibrated,
  uncalibratable with a recorded reason).
- Code harness (`CodeTeam/ael/code_harness/`): executable steady-state modules generated from the
  parsed equations, validated by execution; generation verdicts generated, partial, or ungenerable.
- Estimation harness (`EstimationTeam/ael/estim_harness/`): the language model proposes a
  specification, which the committee reviews before estimation; the harness estimates it
  (OLS with HAC standard errors by default), runs a diagnostic battery and robustness sweeps,
  tests the hypotheses, and returns estimated, fragile, or inestimable.
- Report harness (`ReportingTeam/ael/report_harness/`): deterministic assembly, limitations
  collected from every stage's recorded caveats, and a number-consistency check of the report
  body against the upstream artifacts.

### Data and literature
- Open data connectors: FRED, World Bank, DBnomics (OECD, IMF, BIS), ECB, Eurostat, BLS, Yahoo
  Finance; a series that cannot be retrieved is carried only as a disclosed simulation, which
  the estimation harness excludes.
- Open literature sources: OpenAlex, arXiv, Semantic Scholar, Crossref; optional abstract cache
  and Elsevier abstract fill.
- Open web stack: SearXNG and DDGS for discovery search, trafilatura and crawl4ai for crawling;
  commercial providers when configured.

### Serving and evaluation
- LLM provider map with three layers: vLLM (open-weight, local or cluster), Ollama (laptop),
  commercial APIs.
- Two-tier evaluation (`evaluation/`): structural metrics from execution logs and a two-judge
  open-weight review (Mistral-Small-3.2-24B, Gemma-3-27B) on six content dimensions with
  team-specific rubrics; scripts for human-expert validation of the judge scores.
- SLURM and vLLM scripts for a four-GPU node (`hpc/`; see the README).

### Changed after the runs
- The number-consistency checker recognizes Unicode minus signs, rounding to the printed number
  of digits, and stated ranges, which 0.7.2 missed. The paper reports this checker's verdicts
  (14 of 26 reports consistent); the verdicts stored by the runs, from 0.7.2, are 10 of 26.
- Series retrieved from Yahoo Finance carry the source's official name when Yahoo provides one.
- Each provider receives only its own API key, and a model whose provider has no configured
  endpoint is rejected instead of falling back to OpenAI. The runs used local vLLM models with a
  placeholder key, so this does not affect them.
- The evaluation runners exit with code 1 when they fail or evaluate nothing, report partial
  Tier-2 scores separately, and aggregate all repetitions. The paper's evaluation numbers are
  computed by the scripts archived with the run outputs, not by these runners.
- The repository root is located by `ael_config.yaml` and `run_ael_pipeline.py` rather than by a
  `.env` file.
- The default model is `vllm/qwen3.6-27b-fp8` on a local vLLM server (`ael_config.yaml` `default_model`,
  read by every agent through `shared/model_config.default_model`); agents no longer name a
  model of their own. The agents no longer require `OPENAI_API_KEY` at start; the client checks
  the key of the provider it calls, and a local model needs none. The runs set `AEL_MODEL`
  to a local vLLM model and a placeholder key, so neither change affects them.

### Tests
- 2,972 unit and integration tests, run without network access or LLM calls.

### Known limitations
- The paper's runs used local vLLM models only. The commercial provider routes (OpenAI,
  Anthropic, Google, DeepSeek, Mistral, xAI, OpenRouter, Perplexity, Groq) are covered by mocked
  tests; Perplexity and Groq model names are sent with their catalog prefix and have not been
  checked against the live APIs. A model whose provider has no configured endpoint is rejected.
- `ModePremiumSubscribed` needs a local Refinitiv Eikon, Bloomberg, or WRDS session.
- Only the default mode of each team is maintained (`ModeWithWcWithHITL` for Ideation, Literature
  and Model; `ModeNoWcWithHITL` for Code, Estimation and Reporting); it is the configuration of the
  paper's runs. The other modes of these teams date from earlier development, were not updated with
  it, and will be removed in the next release.
- Evaluation (`python -m evaluation.run_ael_evaluation`):
  - Without `--multirun-dir` it evaluates the four original teams (15 team/mode configurations).
    Chained seven-team runs are evaluated with `--multirun-dir` on the view that
    `pipeline/build_eval_view.py` builds.
  - The multi-run aggregation aggregates 12 of the 14 dimensions (Simulation Fidelity and Causal
    Validity are left out), and its console summary mislabels the last columns.
  - The 2x2 factorial-effects analysis recognizes only the former mode names (`WithFc`, `NoFc`);
    with the current `Wc` names it reports no effects.
  - `--tier2-ensemble` has no effect together with `--multirun-dir`; there, Tier 2 runs one judge
    per call (`--tier2 --judge-model ...`).
  - `pipeline/build_eval_view.py` assumes the default modes.
  - Local Tier-2 evaluation does not read `.env`; export the endpoint variables in the shell.
  The paper's numbers do not depend on these paths: they are computed from the run files by the
  scripts archived with the run outputs.
- Standalone CodeTeam, EstimationTeam and ReportingTeam runs: the built-in fallback input paths
  do not resolve; pass explicit paths.
- In the pipeline, CodeTeam's review checkpoint is not invoked; it runs only in the standalone
  orchestrator.
- `AEL_LLM_TIMEOUT` is read when the code is imported, before `.env` is loaded; export it in the
  shell.
- Committee ballots are written to several files unless an absolute `AEL_HITL_LOG` is exported.
- In the pipeline, only `llm_economist` mode decides by vote whether a stage is revised. With
  `interactive` or `auto`, Literature and Model run a fixed number of rounds, and Estimation and
  Reporting accept the first proposal, so feedback entered there does not trigger a revision.

## Earlier versions

Versions 0.1 to 0.7.2 were internal development versions and were not released. They added,
in order: the four original teams and their operational modes (web crawl, human-in-the-loop),
the evaluation framework, the shared infrastructure (instrumentation, provider map,
verification, security), the open-weight serving stack, the deterministic calibration harness,
the Code, Estimation, and Reporting teams, the cross-team pipeline, the feasibility loop, and
the LLM-Economist committee.
