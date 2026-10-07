# Contributing to Agentic Econ Lab (AEL)

This guide covers the development setup, tests, code conventions, and how to submit changes.

## Development setup

Requirements: Python 3.10 or later, Git, and access to at least one LLM endpoint (a local
vLLM or Ollama server, or a commercial API key).

```bash
git clone https://github.com/AgenticEconLab/agentic-econ-lab.git ael && cd ael
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-ael.txt
cp .env.example .env                                  # then edit .env
export PYTHONPATH="$PWD"                              # Windows PowerShell: $env:PYTHONPATH = "$PWD"
python -c "from shared.instrumentation import WorkflowLogger; print('OK')"
```

## Running tests

```bash
python -m pytest tests/unit -q              # unit tests (no network, no LLM calls)
python -m pytest tests/integration -q       # integration tests
python -m pytest tests -q                   # everything
```

The calibration harness has its own script tests, run from the repository root (not from
inside `calib_harness/`, whose `types.py` would shadow the standard library):

```bash
PYTHONPATH="$PWD:$PWD/ModelTeam/ael" python ModelTeam/ael/calib_harness/tests/test_targets.py
```

## Code conventions

- PEP 8, type hints on public functions, docstrings on public classes and functions;
  Pydantic `BaseModel` for data schemas (per-team schemas live in `<Team>/ael/schemas/`).
- Each stage file `N-<Stage>Stage.py` holds its agent classes (each wraps
  `shared.llm.LLMClient` with an `agent_name`) and a `<Stage>Orchestrator` that runs them and
  writes the stage's `*_output.json`; `0-MasterOrchestrator.py` runs the stages in order.
  Most stage files can also be run on their own (ReportingTeam's drafting stage cannot).
- Shared components accept an optional `collector=None` (the `MetricsCollector` used for
  observability) and must work without it.
- Language models propose; deterministic code decides. Calibration, estimation, code
  generation, and report numbers go through the harnesses (`calib_harness`, `estim_harness`,
  `code_harness`, `report_harness`), not through free-form model output. New features that
  produce a verdict should follow the same pattern and report what they refused and why.
- Fixes and features should work for any field of economics, not only for the case that
  exposed the problem.
- New capabilities that change pipeline behaviour go behind a flag in `ael_config.yaml`
  (default off).
- Every `.py` file starts with the license header, after any shebang or encoding line:

  ```python
  # SPDX-License-Identifier: MIT
  # Copyright (c) 2026 AgenticEconLab
  ```

## Tests

- Unit tests go in `tests/unit/test_<module>.py`, integration tests in `tests/integration/`.
- Use the fixtures in `tests/conftest.py` and `tests/fixtures/`.
- Unit tests must not call a real LLM or external API; mock them.
- New code comes with tests.

## Adding a data source

1. Add a client in `shared/data/` (e.g. `my_source.py`).
2. Add its tracked wrapper in `shared/observability.py`.
3. Register it in the relevant DataTeam stage files.
4. Add tests in `tests/unit/test_my_source.py`.
5. Add any new key to `.env.example`.

## Adding an LLM provider

1. Add the provider to `shared/provider_map.py` (`PROVIDER_ROUTES`,
   `OPENAI_COMPATIBLE_PROVIDERS`, `PROVIDER_API_KEY_ENV`).
2. Add model entries to `shared/llm_router.py` (`MODEL_CATALOG`, `PROVIDER_ENV_KEYS`,
   `PROVIDER_BASE_URLS`).
3. Add tests in `tests/unit/test_multi_provider_llm.py`.
4. Add the key variable to `.env.example`.

## Submitting changes

1. Create a branch from `main`.
2. Make the change with tests; run `python -m pytest tests -q`.
3. Open a pull request describing what changed and why.

Commit messages: a short prefix and a plain description, e.g.
`fix: correct budget calculation in multi-run evaluation`.

## Reporting issues

Open an issue with a description of the problem, steps to reproduce, expected and actual
behaviour, the model and provider used, and your Python version and OS.
