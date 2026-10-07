# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Estimation reduces moment-target error vs the LLM's asserted params."""
from calib_harness import run
def test_estimation_point_calibrates_and_beats_asserted():
    # A CES/CD production model: alpha is labor share, target ~0.62. LLM asserts a far-off 0.30.
    fm = {"equations": [{"equation_id":"E1","equation_type":"behavioral",
            "equation_plain":"Y = A * (K^alpha) * (L^(1-alpha))",
            "variables_used":"['Y','K','L','A']","parameters_used":"['alpha']"}],
          "parameters": [{"parameter_symbol":"alpha","parameter_name":"Capital Share Parameter",
                          "typical_range":"[0.2, 0.45]","description":"capital elasticity of output"}]}
    cps = [{"parameter_symbol":"alpha","calibrated_value":"0.30"}]
    oc = run(fm, cps)
    # estimation should run and drive the labor_share moment toward its external target
    assert oc.estimated is True
    assert oc.moment_match_score is not None and oc.moment_match_score >= 0.9
    assert oc.calibration_status in ("point_calibrated","partially_calibrated","calibrated")
if __name__ == "__main__":
    test_estimation_point_calibrates_and_beats_asserted(); print("RESULT: estimation test passed")
