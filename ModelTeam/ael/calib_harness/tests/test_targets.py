# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Unit tests for calib_harness.targets + macro_targets.yaml.

Run (no pytest required — plain assertions + a tiny harness so it works under the lightweight venv):

    PYTHONPATH=ModelTeam/ael \\
        python ModelTeam/ael/calib_harness/tests/test_targets.py
"""

from __future__ import annotations

import math
import os
import sys

from calib_harness.targets import _DEFAULT_YAML, get_target, load_targets
from calib_harness.types import TargetRow

_RESULTS = []


def _check(name, fn):
    try:
        fn()
    except AssertionError as e:
        _RESULTS.append((name, "FAIL", str(e)))
    except Exception as e:  # noqa: BLE001 - report unexpected errors as failures, don't crash the run
        _RESULTS.append((name, "ERROR", f"{type(e).__name__}: {e}"))
    else:
        _RESULTS.append((name, "PASS", ""))


# --------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------
def test_loads_at_least_20_rows():
    targets = load_targets()
    assert isinstance(targets, dict), "load_targets must return a dict"
    assert len(targets) >= 20, f"expected >=20 target rows, got {len(targets)}"
    print(f"    [rows loaded] {len(targets)}")


def test_every_row_is_a_valid_targetrow():
    targets = load_targets()
    for key, row in targets.items():
        assert isinstance(row, TargetRow), f"{key}: not a TargetRow ({type(row)})"
        assert row.moment_key == key, f"{key}: moment_key mismatch ({row.moment_key})"
        # numeric, finite value
        assert isinstance(row.value, float) and math.isfinite(row.value), f"{key}: bad value {row.value!r}"
        # std_error strictly positive
        assert isinstance(row.std_error, float) and row.std_error > 0.0, f"{key}: std_error must be >0, got {row.std_error!r}"
        # non-empty source citation
        assert isinstance(row.source_citation, str) and row.source_citation.strip(), f"{key}: empty source_citation"
        # frequency must be one of the supported kinds (drives annualization downstream)
        assert row.frequency in ("annual", "quarterly"), f"{key}: unexpected frequency {row.frequency!r}"
        assert row.geography, f"{key}: empty geography"


def test_no_duplicate_keys_and_keys_nonempty():
    targets = load_targets()
    for key in targets:
        assert key and key.strip(), "empty moment_key encountered"
    # load_targets raises on dup keys; reaching here means none.


def test_specific_values_match_cited_literature():
    """Spot-check that a few load-bearing values are exactly the curated, cited magnitudes."""
    expected = {
        "capital_output_ratio": 3.0,     # PWT/NIPA, annual
        "labor_share": 0.60,             # K&N / BLS
        "investment_output_ratio": 0.20, # BEA GPDI/GDP
        "depreciation_rate": 0.06,       # BEA, annual
        "substitution_elasticity_KL": 0.50,  # Chirinko (2008)
        "calvo_price_stickiness": 0.66,  # Nakamura-Steinsson (2008)
        "taylor_phi_pi": 1.5,            # Taylor (1993)
        "taylor_phi_y": 0.5,             # Taylor (1993)
        "trend_inflation": 0.02,         # FOMC 2% target
    }
    for key, want in expected.items():
        row = get_target(key)
        assert row is not None, f"missing expected target '{key}'"
        assert abs(row.value - want) < 1e-12, f"{key}: value {row.value} != expected {want}"
        print(f"    [value] {key:<28} = {row.value:<6}  ({row.source_citation[:48]}...)")


def test_task_required_moment_keys_present():
    """Every moment_key the build brief named must exist."""
    required = [
        "capital_output_ratio", "labor_share", "investment_output_ratio", "real_interest_rate",
        "depreciation_rate", "substitution_elasticity_KL", "calvo_price_stickiness",
        "price_duration_quarters", "price_markup", "taylor_phi_pi", "taylor_phi_y",
        "trend_inflation", "frisch_elasticity", "intertemporal_elasticity_substitution",
    ]
    targets = load_targets()
    missing = [k for k in required if k not in targets]
    assert not missing, f"missing required moment_keys: {missing}"
    print(f"    [coverage] all {len(required)} brief-required keys present")


def test_real_interest_rate_in_plausible_band():
    row = get_target("real_interest_rate")
    assert row is not None
    assert 0.005 <= row.value <= 0.03, f"real_interest_rate {row.value} outside plausible 0.5-3% band"


def test_calvo_and_price_duration_internally_consistent():
    """Expected price duration ~ 1/(1-theta) for Calvo theta."""
    theta = get_target("calvo_price_stickiness").value
    dur = get_target("price_duration_quarters").value
    implied = 1.0 / (1.0 - theta)
    assert abs(dur - implied) < 0.7, f"price_duration {dur} inconsistent with 1/(1-theta)={implied:.2f}"
    print(f"    [consistency] Calvo theta={theta} -> implied duration {implied:.2f}q (target {dur}q)")


def test_fred_series_id_dropped_from_targetrow():
    """The YAML carries fred_series_id on some rows; the locked TargetRow must NOT gain that field."""
    assert not hasattr(get_target("trend_inflation"), "fred_series_id"), \
        "TargetRow unexpectedly has a fred_series_id attribute (contract changed!)"
    # but the YAML row itself does carry it (documentation for optional FRED recompute)
    import yaml
    with open(_DEFAULT_YAML, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    has_any_fred = any("fred_series_id" in r for r in doc["targets"])
    assert has_any_fred, "expected at least one YAML row to document a fred_series_id"


def test_get_target_missing_returns_none():
    assert get_target("this_moment_does_not_exist_xyz") is None


def test_default_yaml_path_resolves():
    assert os.path.exists(_DEFAULT_YAML), f"default yaml not found at {_DEFAULT_YAML}"


# --------------------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------------------
def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} tests from test_targets.py\n")
    for fn in tests:
        print(f"- {fn.__name__}")
        _check(fn.__name__, fn)

    print("\n" + "=" * 70)
    npass = sum(1 for _, s, _ in _RESULTS if s == "PASS")
    nfail = sum(1 for _, s, _ in _RESULTS if s != "PASS")
    for name, status, msg in _RESULTS:
        line = f"{status:5} {name}"
        if msg:
            line += f"  -> {msg}"
        print(line)
    print("=" * 70)
    print(f"RESULT: {npass} passed, {nfail} failed, out of {len(_RESULTS)}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
