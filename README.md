# Agentic Econ Lab (AEL)

![license](https://img.shields.io/badge/license-MIT-blue) ![python](https://img.shields.io/badge/python-3.10%2B-blue)

AEL is a multi-agent framework for economics research. Seven agent teams cover research
questions, literature review, model building and calibration, data retrieval, model code,
estimation, and report writing. They run one at a time or as a chained pipeline under a
cross-team orchestrator. At each step a language model proposes and a deterministic component
decides: equations are parsed and calibrated against cited targets, regressions are estimated and
diagnosed by an econometric harness, and each non-exempt number in the final report is looked up
among the values of the upstream artifacts.

This repository holds the code of the paper *Agentic Workflows for Economic Research: Design,
Implementation, and Evaluation* (Dawid, Harting, Wang, Wang, Yi;
[arXiv:2504.09736](https://arxiv.org/abs/2504.09736)). The paper's 26 chained runs were made with
version 0.7.2, served entirely on open-weight models (vLLM); [CHANGELOG.md](CHANGELOG.md) lists
what changed since, and the known limitations of this release. The run outputs, the scripts that
compute the paper's run-level numbers, and the expert ratings are archived at
[doi:10.5281/zenodo.23193486](https://doi.org/10.5281/zenodo.23193486).

## Quick start

```bash
git clone https://github.com/AgenticEconLab/agentic-econ-lab.git ael && cd ael
python3 -m venv .venv && source .venv/bin/activate     # Python >= 3.10
pip install -r requirements-ael.txt
playwright install chromium                            # only for the crawl4ai crawler
cp .env.example .env                                   # then edit .env
export PYTHONPATH="$PWD"                               # PowerShell: $env:PYTHONPATH = "$PWD"
export AEL_MODEL=vllm/qwen3.6-27b-fp8                  # or ollama/NAME, or a hosted model id
python -m pytest tests -q                              # unit and integration tests
python run_ael_pipeline.py --topic "Fiscal multipliers in a currency union" --hitl-mode auto
```

`AEL_MODEL` sets the model of the agents. `--hitl-mode auto` answers the checkpoints with their
default answers, without economic judgement; use it for a first test. The default
`--hitl-mode llm_economist` additionally needs the three committee models served (see
Checkpoints below).

## Teams

Each team is a directory `<Team>/ael/` with one sub-directory per mode. A mode directory holds
`0-MasterOrchestrator.py`, which runs all stages, and one numbered file per stage; most stage files
can also be run alone (ReportingTeam's drafting stage runs only through its orchestrator or the
pipeline). The 2x2 modes are `Mode{No,With}Wc{No,With}HITL`: **Wc** means the stages use web search
and page crawling; **HITL** means they stop at human-in-the-loop checkpoints.
The paper's runs used one configuration, which is also the pipeline default: `ModeWithWcWithHITL`
for Ideation, Literature and Model, `ModeOpenSourceAPI` for Data, and `ModeNoWcWithHITL` for Code,
Estimation and Reporting. For these six teams only the default mode is maintained; their other
modes date from earlier development, were not updated with it, and will be removed in the next
release. DataTeam's three workflows are alternative data sources.

| Team | Stages | Modes | Deterministic component and its verdicts |
|---|---|---|---|
| IdeationTeam | sourcing, refinement, integration | 2x2 | seed-topic check, citation audit |
| LiteratureTeam | gathering, gap detection, synthesis | 2x2 | citation verification against OpenAlex, Crossref, arXiv, Semantic Scholar |
| ModelTeam | theory, design, calibration (+ optional simulation) | 2x2 | `calib_harness/`: `calibrated`, `point_calibrated`, `partially_calibrated`, `simulation_calibrated`, `archetype_calibrated`, `uncalibratable` |
| DataTeam | source, cleaning, quality assurance | `ModeOpenSourceAPI`, `ModePremiumSubscribed`, `ModeUserUploaded` | open data connectors; quality checks; Model–Data feasibility loop (`feasibility/`) |
| CodeTeam | generation, validation, experimentation | `ModeNoWc{No,With}HITL` | `code_harness/`: modules generated from the parsed equations, not written by the LLM; generation `generated`, `partial`, `ungenerable`; validation `validated`, `fragment_solves`, `partial`, `failed`, `not_applicable`; experimentation `completed`, `partial`, `not_applicable` |
| EstimationTeam | estimation, diagnostics, inference and robustness | `ModeNoWc{No,With}HITL` | `estim_harness/`: `estimated`, `fragile`, `inestimable` (statsmodels/linearmodels, HAC errors by default) |
| ReportingTeam | interpretation, drafting, quality | `ModeNoWc{No,With}HITL` | `report_harness/`: each non-exempt number of the draft is looked up among the values of the upstream artifacts (exempt: small integers used as counts or labels, confidence levels, digits in URLs and DOIs); `consistent`, `has_unverified`, `empty`. A number found upstream is not thereby shown to be used correctly |

`ModePremiumSubscribed` needs a local Refinitiv Eikon, Bloomberg or WRDS session; without one it
reports N/A. In pipeline runs with `--data-mode user_uploaded`, `AEL_USER_DATA_PATH` gives the
data file (default: the mode's `example_data.csv`); standalone `ModeUserUploaded` runs ask for
the path and use the example when the answer is empty. Without `FRED_API_KEY`, FRED series are fetched through keyless connectors
where possible (for example DBnomics); a series that cannot be retrieved is simulated and flagged
`data_simulated`, and the estimation harness excludes it.

## Configuration

`.env` in the repository root holds the common settings; `.env.example` lists them. Some variables
are read when the code is imported, before `.env` is loaded, and must be exported in the shell
instead (noted below).

**Model.** A model name is routed by prefix: `vllm/<served-name>` goes to `VLLM_BASE_URL`
(default `http://localhost:11434/v1/chat/completions`) or to the server given for that name in
`AEL_VLLM_ENDPOINTS`; `ollama/<name>` goes to a local Ollama server; `gpt-*`, `claude-*`,
`gemini-*`, `deepseek-*`, `mistral-*`, `grok-*` go to the provider's API with the provider key from
`.env`. The part after `vllm/` must equal the server's `--served-model-name`. The model used is, in
order: `AEL_MODEL` (or `--model`), the per-team and per-stage entry in `ael_config.yaml`, then
`default_model` there (`vllm/qwen3.6-27b-fp8`). Standalone ModelTeam runs load the
`ael_config.yaml` in their mode directory (feature flags only) unless `AEL_MODEL_CONFIG` is set, so
for them set `AEL_MODEL`.

| Variable | Default | Effect |
|---|---|---|
| `AEL_VLLM_ENDPOINTS` | unset | `name=url,...`: different served names on different servers (used by the committee) |
| `AEL_LLM_TIMEOUT` | `600` | per-call timeout in seconds; export it in the shell (read at import). Literature synthesis raises it for the rest of the process to at least `AEL_SYNTHESIS_LLM_TIMEOUT` (default 900) |
| `AEL_DISABLE_THINKING` | unset | `1` turns off reasoning traces for Qwen models served through vLLM or Ollama |
| `VLLM_MAX_MODEL_LEN` | `16384` | context length the client assumes when capping a completion; set it to the server's `--max-model-len`. The prompt length is estimated (characters / 4) and prompts are not truncated |
| `AEL_LLM_CONCURRENCY` | `8` | parallel per-item LLM calls |
| `FRED_API_KEY` | unset | native FRED retrieval |
| `OPENALEX_API_KEY` | unset | OpenAlex searches (keyless use is limited to about 100 a day) |
| `SEARXNG_BASE_URL` | `http://localhost:8899` | open discovery search (`hpc/setup_searxng.sh` installs SearXNG) |

**Web crawl and search.** Crawling tries `WEBCRAWL_PROVIDER` (default `trafilatura`), then the
providers in `WEBCRAWL_FALLBACK_CHAIN` (empty by default; for example `crawl4ai,tavily,firecrawl`).
Discovery search (IdeationTeam `--auto`, `--topic auto`) tries SearXNG, DDGS, then Tavily and Brave
if their keys are set. The agents' `web_search` tool uses Tavily, Brave or Serper, depending on which
key is set. Tavily, Firecrawl, Jina, Brave and Serper are commercial services.

**Literature abstracts.** Gathering keeps the abstracts of the retrieved records (Semantic Scholar
or Crossref, arXiv, OpenAlex). Missing or short abstracts of records with a DOI are filled from
`ABSTRACT_CACHE` (a local DOI-to-abstract CSV), then from the Elsevier API (`ELSEVIER_API_KEY`,
optional `ELSEVIER_INSTTOKEN`, `ELSEVIER_PROXY`), which may supply abstracts that the open sources
lack. Records still without an abstract longer than 50 characters are dropped. Each run prints its
abstract coverage.

## Running

**One team**, from its mode directory:

```bash
cd IdeationTeam/ael/ModeNoWcNoHITL && python 0-MasterOrchestrator.py --topic "Fiscal multipliers in a currency union"
```

Ideation, Literature, Model and Data take `--topic`; IdeationTeam `ModeWithWc*` also takes
`--auto` (trending topics from the web). CodeTeam takes `--design` and `--calibration`,
EstimationTeam `--model-spec`, `--data` and `--question`, ReportingTeam `--estimation`,
`--artifacts-dir` and `--question`. Give these three teams explicit absolute paths: their built-in
fallback paths do not resolve in this release, and ReportingTeam loads further upstream artifacts
only from `--artifacts-dir` (or `AEL_REPORT_ARTIFACTS`). Outputs are written into the mode directory.

**The full pipeline**, from the repository root:

| Flag | Default | Effect |
|---|---|---|
| `--topic` | "Agent-based and heterogeneous-agent modeling in macroeconomics and monetary policy" | research topic; `auto` proposes directions across JEL fields and the committee picks one |
| `--mode` | per `pipeline_modes:` in `ael_config.yaml` | mode of Ideation, Literature and Model |
| `--data-mode` | `open_source_api` | `premium_subscribed` or `user_uploaded` |
| `--hitl-mode` | `llm_economist` | who answers checkpoints (below) |
| `--model` | `AEL_MODEL` | model for all teams |
| `--output-dir` | `pipeline_output` | run directory |
| `--resume` | off | continue from the checkpoints of a previous run |

The pipeline runs Ideation → Literature → Model → Data, then the Model–Data feasibility loop (the
model's data requirements are graded against the available data, and unmet requirements return to
the model as a revision request or are recorded as limitations; at most three cycles), then Code
(off unless `AEL_CODE_TEAM_ENABLED=1`) → Estimation → Reporting. The feasibility loop's decisions
are not checkpointed; set `AEL_FEASIBILITY_LOOP=0` on a resumed run whose loop already ran.
Optional capabilities are switched under `feature_flags:` in `ael_config.yaml` or with
`AEL_<FLAG>=0|1`. The built-in budget is USD 5.00 per run, USD 2.00 per team and 2,000,000 tokens.
Concurrent runs need separate copies of the code, because teams write working files into their
directories.

**Checkpoints.** `--hitl-mode` (or `AEL_HITL_MODE`) chooses who answers: `interactive` (a person at
the console; the default for standalone runs), `llm_economist` (a committee of three open-weight
models, `vllm/gpt-oss-20b`, `vllm/phi-4-mini` and `vllm/olmo-3.1-32b-instruct`, changed with
`AEL_HITL_COMMITTEE`), or `auto` (canned answers, for smoke tests only). In the pipeline, only
`llm_economist` decides by vote whether a stage is revised. With `interactive` or `auto`,
Literature and Model run their standard number of rounds, and Estimation and Reporting accept
the first proposal: feedback entered there is recorded but does not trigger a revision. The default members come
from model families other than the default generating model (Qwen) and the evaluation judges
(Mistral, Gemma); if you change models, check this separation yourself, since only member names
containing `qwen`, `mistral` or `gemma` trigger a warning. A decision passes with two thirds of the
votes cast; a member whose call fails abstains; if every call fails, an ordinary checkpoint takes its
default answer and a proposition vote fails. Ballots are logged on a best-effort basis and, without
further setting, end up in several files; export an absolute `AEL_HITL_LOG` before launch to collect
them in one. In the pipeline, CodeTeam's review checkpoint is not invoked; it runs only in the
standalone orchestrator.

**Outputs** of a pipeline run: `<output-dir>/<Team>/` (team outputs), `artifacts/` (the signed
artifacts passed between teams), `checkpoints/`, `feasibility_report.json`,
`pipeline_manifest.json`, and the report `ReportingTeam/research_report.md`. Post-run tools in
`pipeline/`: `link_outputs.py` (a browsable copy of the deliverables), `run_audit_report.py`
(machine verdicts and a review checklist), `recheck_consistency.py` (reruns the report's number
check), `build_eval_view.py` (the layout the evaluator reads; it assumes the default modes).

## Evaluation

`evaluation/run_ael_evaluation.py` scores outputs on 14 dimensions. Tier 1 computes structural
metrics from output files and execution logs; Tier 2 is an LLM judge that scores six content
dimensions. Local Tier-2 evaluation does not read `.env`: export `VLLM_BASE_URL` or
`AEL_VLLM_ENDPOINTS` (and `AEL_JUDGE_MODEL`, if used) in the shell. To score a pipeline run with two
judges from model families other than the generating model's:

```bash
RUN_DIR=pipeline_output                                 # the run's --output-dir
python pipeline/build_eval_view.py "$RUN_DIR"
python evaluation/run_ael_evaluation.py --multirun-dir "$RUN_DIR/_eval_view" --tier2 --judge-model vllm/mistral-small-3.2-24b-fp8
python evaluation/run_ael_evaluation.py --multirun-dir "$RUN_DIR/_eval_view" --tier2 --judge-model vllm/gemma-3-27b-it-fp8
```

## Running on a GPU cluster

`hpc/` holds SLURM and vLLM scripts. The paper's runs used one node with four A100 40GB GPUs per
pipeline run and vLLM 0.22.1:

| Role | Model (served name) | GPU, port |
|---|---|---|
| Generating model | `Qwen/Qwen3.6-27B-FP8` (`qwen3.6-27b-fp8`) | GPU0, 11434 |
| Committee | `microsoft/Phi-4-mini-instruct` (`phi-4-mini`), `openai/gpt-oss-20b` (`gpt-oss-20b`) | GPU1, 11435 and 11436 |
| Committee | `allenai/Olmo-3.1-32B-Instruct` (`olmo-3.1-32b-instruct`) | GPU2+3, 11437 |
| Tier-2 judges | `RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8`, `RedHatAI/gemma-3-27b-it-FP8-dynamic` | one GPU each, separate jobs |

Set `AEL_SCRATCH` (a writable directory for virtual environments, weights, caches and outputs) and
check the `#SBATCH` lines of each script: `--account` and `--partition` are commented out, and
`--gres=gpu:a100:N` may need your cluster's GPU naming. One-time setup: a vLLM environment at
`$AEL_SCRATCH/tools/venv-vllm` (`pip install vllm==0.22.1`), an AEL environment at
`$AEL_SCRATCH/tools/venv-ael` (`pip install -r requirements-ael.txt`), or other paths exported as
`AEL_VLLM_VENV` and `AEL_VENV`; `bash hpc/setup_searxng.sh`; and the model weights downloaded into
`${HF_HOME:-$AEL_SCRATCH/hf-cache}` (the jobs run offline). Then, from the repository root:

```bash
RUN_LABEL=rep01 sbatch hpc/full_pipeline.sbatch            # one seven-team run, up to 7 h
RUN_DIR="$AEL_SCRATCH/agentic-runs/rep01_JOBID"           # JOBID: the job id printed by sbatch
python pipeline/build_eval_view.py "$RUN_DIR"
RUN_TREE="$RUN_DIR/_eval_view" sbatch hpc/eval.sbatch tier2          # Mistral judge
```

The Gemma judge runs with `JUDGE_REPO=RedHatAI/gemma-3-27b-it-FP8-dynamic SERVED=gemma-3-27b-it-fp8
JUDGE_SERVE='--limit-mm-per-prompt {"image":0}'`. `hpc/smoke.sbatch` and
`hpc/committee_smoke.sbatch` check the serving setup first; each script's header lists its
options.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Connection refused on port 11434 | No vLLM server is running for the default model; start one or set `AEL_MODEL` |
| HTTP 404 "model does not exist" | The name after `vllm/` differs from `--served-model-name`, or two servers share a port |
| HTTP 400 from a Mistral or Gemma model | `AEL_DISABLE_THINKING=1` with a served name containing `qwen`; serve non-Qwen models under other names |
| Qwen runs are slow or time out | Reasoning traces; set `AEL_DISABLE_THINKING=1`, export a larger `AEL_LLM_TIMEOUT` |
| vLLM: `torch.cuda.is_available()` is `False` | PyTorch built for a CUDA version above the driver's; install vLLM for a supported CUDA build |
| vLLM: out of memory during CUDA-graph capture | Serve with `--enforce-eager` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| vLLM: `[Errno 28] ENOSPC` on a shared file system | Put `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `VLLM_CACHE_ROOT` on node-local disk |
| Many papers dropped for missing abstracts | Publisher-restricted abstracts; see Literature abstracts above |
| A headless run waits at every checkpoint | `AEL_HITL_MODE=interactive` is set; use `--hitl-mode llm_economist` |

## Repository layout

```
├── IdeationTeam/ LiteratureTeam/ ModelTeam/ DataTeam/ CodeTeam/ EstimationTeam/ ReportingTeam/
├── pipeline/          cross-team orchestrator, artifact store, team runners, post-run tools
├── evaluation/        two-tier evaluation, human-validation scripts
├── shared/            LLM client and provider map, HITL committee, tools, instrumentation, verification
├── hpc/               SLURM and vLLM scripts
├── tests/             unit and integration tests
├── run_ael_pipeline.py   full-pipeline entry point
├── ael_config.yaml       per-stage models, pipeline modes, feature flags
├── requirements-ael.txt  dependencies
└── .env.example          environment variables
```

## Contributing, license, citation

See [CONTRIBUTING.md](CONTRIBUTING.md). MIT License, see [LICENSE](LICENSE).

```bibtex
@article{dawid2026agentic,
  title   = {Agentic Workflows for Economic Research: Design, Implementation, and Evaluation},
  author  = {Dawid, Herbert and Harting, Philipp and Wang, Hankui and Wang, Zhongli and Yi, Jiachen},
  journal = {arXiv preprint arXiv:2504.09736},
  year    = {2026}
}
```
