# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""DRS requirement classification.

In a live run the feasibility loop demanded derived statistics (Output Volatility, Evasion Gini,
elasticities) and latent behavioral parameters (Perceived Detection Probability, Trust Decay)
as retrievable SERIES — always `unavailable`, so MRRs asked models to delete their own
calibration targets. Derived statistics are pipeline-computable and behavioral latents are
calibration parameters; both become non-essential (Medium) with a disclosed tag, and
behavioral-latent VARIABLES never become data requirements at all (like shocks).
"""

from DataTeam.ael.feasibility.adapters import (
    _is_behavioral_latent,
    _is_derived_moment,
    drs_from_model_artifacts,
)


def _drs(variables=None, targets=None):
    model_design = {"formal_models": [{
        "model_title": "Test Model",
        "variables": [
            {"variable_name": n, "variable_symbol": n, "variable_type": "observable",
             "description": d} for n, d in (variables or [])
        ],
    }]}
    calibration = {"calibrated_models": [{
        "empirical_targets": [
            {"target_id": f"T{i+1}", "target_name": n, "description": d, "importance": "high"}
            for i, (n, d) in enumerate(targets or [])
        ],
        "model_moments": [],
    }]}
    return drs_from_model_artifacts("q?", model_design, calibration)


def _req(drs, name):
    for r in drs.requirements:
        if r.variable_name == name:
            return r
    return None


class TestDetectors:
    def test_derived_moment_words(self):
        for name in ("Output Volatility", "Evasion Gini Coefficient",
                     "Autocorrelation of Returns (1-min)", "Frisch Elasticity of Labor Supply",
                     "Mean Reversion Half-Life", "Consumption Volatility relative to Output"):
            assert _is_derived_moment(name), name

    def test_plain_rates_ratios_indices_are_not_derived(self):
        for name in ("Effective Tax Rate", "Tax Revenue to GDP Ratio",
                     "Consumer Price Index", "Tax Compliance Rate", "Bid-Ask Spread"):
            assert not _is_derived_moment(name), name

    def test_behavioral_latents(self):
        for name in ("Perceived Detection Probability", "Aggregate Trust Level",
                     "Trust Decay Rate", "Policy Entropy Level", "Risk Aversion Premium",
                     "Learning Convergence Speed", "Social Contagion Coefficient",
                     "Perceived Fairness Index"):
            assert _is_behavioral_latent(name), name

    def test_retrievable_survey_series_are_not_behavioral_latents(self):
        for name in ("Consumer Sentiment Index", "Unemployment Rate",
                     "Federal Funds Effective Rate", "True Income"):
            assert not _is_behavioral_latent(name), name


class TestDrsClassification:
    def test_derived_moment_variable_downgraded_and_tagged(self):
        drs = _drs(variables=[("Output Volatility", "std of output growth")])
        req = _req(drs, "Output Volatility")
        assert req is not None and req.priority == "Medium"
        assert "derived statistic" in req.description
        assert req.suggested_sources == ["pipeline-computed"]

    def test_behavioral_latent_variable_never_becomes_a_requirement(self):
        drs = _drs(variables=[("Perceived Detection Probability", "agents' subjective belief"),
                              ("Unemployment Rate", "headline U-3 rate")])
        assert _req(drs, "Perceived Detection Probability") is None
        assert _req(drs, "Unemployment Rate").priority == "High"

    def test_derived_and_latent_targets_downgraded_with_tags(self):
        drs = _drs(targets=[
            ("Evasion Elasticity to Penalty", "response of evasion to penalties"),
            ("Aggregate Trust Level", "population trust in institutions"),
            ("Tax Compliance Rate", "share of taxes actually paid"),
        ])
        derived = _req(drs, "Evasion Elasticity to Penalty")
        assert derived.priority == "Medium" and "derived statistic" in derived.description
        latent = _req(drs, "Aggregate Trust Level")
        assert latent.priority == "Medium" and "latent behavioral parameter" in latent.description
        # a plain rate stays a genuine (possibly unmet) essential requirement — honest
        assert _req(drs, "Tax Compliance Rate").priority == "High"

    def test_book_satisfied_tag_wins_over_derived_tag(self):
        model_design = {"formal_models": [{"model_title": "m", "variables": []}]}
        calibration = {"calibrated_models": [{
            "empirical_targets": [{"target_id": "T1", "target_name": "Inflation Volatility",
                                   "description": "", "importance": "high"}],
            "model_moments": [{"moment_name": "inflation_volatility", "empirical_value": 0.012}],
        }]}
        drs = drs_from_model_artifacts("q?", model_design, calibration)
        req = _req(drs, "Inflation Volatility")
        assert req.priority == "Medium"
        assert "calibration target book" in req.description
        assert "derived statistic" not in req.description
