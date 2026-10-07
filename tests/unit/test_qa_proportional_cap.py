# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""The simulated-data integrity cap was formerly all-or-nothing —
one disclosed simulated series out of 15 branded the dataset with the full-simulation
floor of 50. The cap is now proportional to the simulated SHARE, and both integrity
invariants hold: never Gold with any simulation; >=half simulated keeps the 50 floor."""

import importlib.util
from pathlib import Path

import pytest

_STAGE = (Path(__file__).resolve().parent.parent.parent
          / "DataTeam" / "ael" / "ModeOpenSourceAPI" / "3-QualityAssuranceStage.py")


@pytest.fixture(scope="module")
def agent(request):
    spec = importlib.util.spec_from_file_location("qa_stage_n9", _STAGE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in dir(mod):
        cls = getattr(mod, name)
        if isinstance(cls, type) and hasattr(cls, "validate_dataset"):
            a = object.__new__(cls)
            a.agent_name = "TestValidator"
            return a
    raise AssertionError("no class with validate_dataset found")


class _QA:
    def __init__(self, score):
        self.overall_score = score


def test_one_of_fifteen_simulated_is_capped_proportionally(agent):
    score, cert = agent.validate_dataset(None, [_QA(92.0)] * 3,
                                         data_simulated=True, sim_share=1 / 15)
    assert score == pytest.approx(90.0 - 80.0 / 15, abs=0.1)   # ~84.7, not 50
    assert cert == "Silver"                                     # never Gold


def test_majority_simulated_keeps_the_hard_floor(agent):
    score, cert = agent.validate_dataset(None, [_QA(92.0)] * 3,
                                         data_simulated=True, sim_share=0.6)
    assert score == 50.0 and cert == "Bronze"


def test_fully_simulated_unchanged(agent):
    score, cert = agent.validate_dataset(None, [_QA(92.0)], data_simulated=True)
    assert score == 50.0 and cert == "Bronze"    # sim_share default -> 1.0 fallback


def test_all_real_uncapped(agent):
    score, cert = agent.validate_dataset(None, [_QA(92.0)], data_simulated=False)
    assert score == 92.0 and cert == "Gold"


def test_never_gold_even_at_tiny_share(agent):
    score, cert = agent.validate_dataset(None, [_QA(99.0)] * 5,
                                         data_simulated=True, sim_share=0.01)
    assert cert != "Gold" and score <= 90.0
