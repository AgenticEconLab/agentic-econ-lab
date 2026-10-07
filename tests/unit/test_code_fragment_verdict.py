# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""In two v0.7.1 runs a module derived from 1 of 18 equations was reported
'validated 5/5' and swept by comparative statics. Only a complete module validates; a solving
fragment gets 'fragment_solves' and no comparative statics."""

from CodeTeam.ael.code_harness import comparative_statics, generate_module, validate_module


def _eq(eid, plain, variables, parameters):
    return {"equation_id": eid, "equation_plain": plain,
            "variables_used": variables, "parameters_used": parameters}


class TestFragmentVerdict:
    def _fragment(self):
        # one parseable equation solves on its own; the rest of the model is refused
        return {"model_title": "fragment", "equations": [
            _eq("EQ1", "k = s * k**alpha / delta", ["k"], ["s", "alpha", "delta"]),
            _eq("EQ2", "c_t = E_t[c_{t+1}] * beta", ["c_t"], ["beta"]),
        ], "parameters": [{"parameter_symbol": s} for s in ("s", "alpha", "delta", "beta")]}

    def _calib(self):
        return {"calibrated_models": [{"model_title": "fragment", "calibrated_parameters": [
            {"parameter_symbol": "s", "calibrated_value": 0.2},
            {"parameter_symbol": "alpha", "calibrated_value": 0.33},
            {"parameter_symbol": "delta", "calibrated_value": 0.1},
            {"parameter_symbol": "beta", "calibrated_value": 0.99}]}]}

    def test_partial_module_that_solves_is_not_validated(self, tmp_path):
        gen = generate_module(self._fragment(), self._calib())
        assert gen.verdict == "partial" and gen.n_parseable == 1 and gen.square
        val = validate_module(gen, str(tmp_path))
        assert val.verdict == "fragment_solves"
        complete = next(c for c in val.checks if c.name == "system_complete")
        assert not complete.passed and "1/2" in complete.detail
        assert val.steady_state                      # the fragment's solution is still reported

    def test_no_comparative_statics_on_a_fragment(self, tmp_path):
        gen = generate_module(self._fragment(), self._calib())
        val = validate_module(gen, str(tmp_path))
        exp = comparative_statics(gen, val, str(tmp_path))
        assert exp.verdict == "not_applicable" and not exp.statics
        assert "fragment" in exp.notes[0]
