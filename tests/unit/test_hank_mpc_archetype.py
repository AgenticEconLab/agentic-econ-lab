# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""The HANK/TANK hand-to-mouth archetype activates the dormant cited `mpc` target.

Aggregate quarterly MPC = λ·1 + (1−λ)·mpc_u (mpc_u = 0.05, Kaplan-Violante) is monotone in the
hand-to-mouth share λ, so the JPS/KV quarterly MPC target (0.25±0.08) just-identifies λ. Covers:
the closed form, the CONFIRM role gate (keywords + [0,1] range; savings propensity ≠ HtM share),
moment detection, and an end-to-end point calibration through harness.run."""

import pytest

pytest.importorskip("scipy")
pytest.importorskip("sympy")

from ModelTeam.ael.calib_harness import archetypes
from ModelTeam.ael.calib_harness import run as harness_run


def test_aggregate_mpc_closed_form():
    assert archetypes.aggregate_mpc(0.0) == pytest.approx(0.05)     # no HtM -> PIH mpc
    assert archetypes.aggregate_mpc(1.0) == pytest.approx(1.0)      # all HtM -> spend everything
    assert archetypes.aggregate_mpc(0.30) == pytest.approx(0.335)   # 0.3 + 0.7*0.05
    # monotone in the share
    assert archetypes.aggregate_mpc(0.5) > archetypes.aggregate_mpc(0.2)


def test_role_confirms_htm_share_and_rejects_lookalikes():
    roles = archetypes.map_param_roles([
        {"parameter_symbol": "lambda", "parameter_name": "hand-to-mouth share",
         "description": "share of hand-to-mouth households", "typical_range": "[0.2, 0.5]"},
        {"parameter_symbol": "chi", "parameter_name": "rule-of-thumb fraction",
         "description": "fraction of rule-of-thumb consumers", "typical_range": "[0.1, 0.4]"},
        # NOT HtM: a savings propensity (the mesa_wealth parameter) must not be captured
        {"parameter_symbol": "lam", "parameter_name": "savings propensity",
         "description": "fraction of wealth retained each period", "typical_range": "[0.1, 0.9]"},
        # HtM wording but an implausible range -> unresolved (CONFIRM discipline)
        {"parameter_symbol": "mu", "parameter_name": "hand-to-mouth share",
         "description": "share of constrained households", "typical_range": "[0.5, 2.0]"},
    ])
    by_sym = {r.symbol: r.role for r in roles}
    assert by_sym["lambda"] == "hand_to_mouth_share"
    assert by_sym["chi"] == "hand_to_mouth_share"
    assert by_sym["lam"] is None
    assert by_sym["mu"] is None


def test_detect_moments_proposes_mpc_pin():
    roles = archetypes.map_param_roles([
        {"parameter_symbol": "lambda", "parameter_name": "hand-to-mouth share",
         "description": "share of hand-to-mouth households", "typical_range": "[0.2, 0.5]"}])
    moments = archetypes.detect_moments([], roles)
    mpc = [m for m in moments if m.moment_key == "mpc"]
    assert len(mpc) == 1
    m = mpc[0]
    assert m.pins == "lambda" and m.frequency == "quarterly" and m.archetype == "hank_htm"
    assert m.value_fn({"lambda": 0.30}) == pytest.approx(0.335)


def test_harness_point_calibrates_htm_share_against_cited_mpc_target():
    """End-to-end: a TANK model with an HtM share gets ESTIMATED against mpc=0.25 (JPS/KV).
    Solving 0.25 = lam + (1-lam)*0.05 gives lam ~= 0.2105 — inside the literature range."""
    model = {
        "model_title": "Two-agent NK model with hand-to-mouth households",
        "parameters": [
            {"parameter_symbol": "lambda", "parameter_name": "hand-to-mouth share",
             "description": "share of hand-to-mouth households", "typical_range": "[0.2, 0.5]"},
        ],
        "equations": [
            {"equation_plain": "C = lambda*C_h + (1-lambda)*C_s", "equation_latex": ""},
        ],
    }
    proposed = [{"parameter_symbol": "lambda", "calibrated_value": 0.40}]
    out = harness_run(model, proposed)
    assert out.calibration_status == "point_calibrated"
    assert out.estimated
    lam = out.estimated_params.get("lambda")
    assert lam is not None and abs(lam - 0.2105) < 0.02
    assert out.moment_match_score is not None and out.moment_match_score > 0.9
