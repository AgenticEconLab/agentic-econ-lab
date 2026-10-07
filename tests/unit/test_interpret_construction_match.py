# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Interpreter construction matching: (a) for b1 log(X) + b2 X*Z the interpreter reported
b1 + b2 mean(Z) as the marginal effect of X, matching the main effect on the series alone
(the derivative is b1/X + b2 Z); (b) a differenced regressor got an 'elasticity at means'
from mean(delta X). Main effects now match on the full construction, and the elasticity at
means is computed only for true level-on-level."""

import numpy as np
import pytest

from ReportingTeam.ael.report_harness import interpret_estimation


def _result(regressors, coefs, dependent=None):
    rng = np.random.default_rng(0)
    cols = [r["name"] for r in regressors]
    data = [{"date": f"{1970 + i}-01-01", "y": float(rng.normal(10, 1)),
             **{c: float(rng.normal(3, 1)) for c in cols}} for i in range(40)]
    spec = {"dependent": dependent or {"name": "y", "series_ref": "Y", "transform": "level"},
            "regressors": regressors}
    out = interpret_estimation({"outcome": {
        "verdict": "estimated", "analysis_data": data, "spec": spec, "dependent_name": "y",
        "coefficients": [{"name": k, "estimate": v, "p_value": 0.01} for k, v in coefs]}})
    return {e.param: e for e in out.effects}, data


def _xz(**base):
    return [{"name": "X main", "series_ref": "X", "transform": "level", **base},
            {"name": "XZ", "series_ref": "X", "transform": "level", "interact_with": "Z"},
            {"name": "Z", "series_ref": "Z", "transform": "level"}]


COEFS = (("const", 1.0), ("X main", 2.0), ("XZ", 0.5), ("Z", 0.1))


@pytest.mark.parametrize("change", [{"transform": "log"}, {"lag": 1},
                                    {"subtract_ref": "W"}])
def test_main_effect_with_other_construction_is_not_combined(change):
    effects, _ = _result(_xz(**change), COEFS)
    e = effects["XZ"]
    assert e.conditional_marginal_effect is None
    assert "different construction" in e.note


def test_subtract_transform_must_match_too():
    regs = [{"name": "X main", "series_ref": "X", "transform": "level", "subtract_ref": "W",
             "subtract_transform": "yoy_pct_change"},
            {"name": "XZ", "series_ref": "X", "transform": "level", "subtract_ref": "W",
             "interact_with": "Z"},
            {"name": "Z", "series_ref": "Z", "transform": "level"}]
    effects, _ = _result(regs, COEFS)
    assert effects["XZ"].conditional_marginal_effect is None


def test_identical_construction_still_combined():
    effects, data = _result(_xz(), COEFS)
    mean_z = float(np.mean([d["Z"] for d in data]))
    assert effects["XZ"].conditional_marginal_effect == pytest.approx(2.0 + 0.5 * mean_z)


def test_no_elasticity_for_differenced_regressor():
    regs = [{"name": "dX", "series_ref": "X", "transform": "diff"},
            {"name": "L", "series_ref": "L", "transform": "level"}]
    effects, _ = _result(regs, (("const", 1.0), ("dX", 3.0), ("L", 1.0)))
    assert effects["dX"].elasticity_at_means is None
    assert effects["L"].elasticity_at_means is not None


def test_no_elasticity_when_either_side_is_a_subtraction():
    regs = [{"name": "R", "series_ref": "I", "transform": "level", "subtract_ref": "P"}]
    effects, _ = _result(regs, (("const", 1.0), ("R", 3.0)))
    assert effects["R"].elasticity_at_means is None
    regs = [{"name": "L", "series_ref": "L", "transform": "level"}]
    dep = {"name": "y", "series_ref": "Y", "transform": "level", "subtract_ref": "Q"}
    effects, _ = _result(regs, (("const", 1.0), ("L", 3.0)), dependent=dep)
    assert effects["L"].elasticity_at_means is None
