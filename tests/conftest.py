# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Shared test configuration for AEL test suite.

Adds the repository root to sys.path so that all test files can import from
shared/, evaluation/, pipeline/, etc. without per-file sys.path hacks.
"""

import sys
from pathlib import Path

# repository root — parent of tests/
_agents_dir = str(Path(__file__).resolve().parent.parent)
if _agents_dir not in sys.path:
    sys.path.insert(0, _agents_dir)

# Network-off default for the unit suite: provider-index search rungs (FRED/WDI/DBnomics
# lookups in econ_connectors) must never fire live HTTP from tests — tests that exercise
# them mock the HTTP layer explicitly; live verification is a manual/sbatch concern.
import os
os.environ.setdefault("AEL_PROVIDER_SEARCH", "0")

# Tests that build OpenAI-provider clients need a key value even though every LLM call is
# mocked. A placeholder lets the suite run on a fresh clone without a .env file; no request is sent.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder-not-a-real-key")
