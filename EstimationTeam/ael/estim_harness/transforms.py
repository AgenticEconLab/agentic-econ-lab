# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Deterministic series transforms and frequency alignment for the estimation harness.

All functions are pure pandas/numpy — no LLM, no network. Frequency inference is by median
observation gap so it works on any connector's output without trusting metadata.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ordered coarse -> fine; alignment always resamples to the COARSEST native frequency present.
_FREQ_ORDER = ["annual", "quarterly", "monthly", "weekly", "daily"]
_FREQ_TO_PANDAS = {"annual": "YS", "quarterly": "QS", "monthly": "MS", "weekly": "W", "daily": "D"}
# Periods-per-year used by yoy transforms.
_FREQ_PERIODS = {"annual": 1, "quarterly": 4, "monthly": 12, "weekly": 52, "daily": 252}


def infer_frequency(index: pd.DatetimeIndex) -> str:
    """Median day-gap -> frequency bucket. Defaults to 'annual' for sparse/odd spacing."""
    if len(index) < 3:
        return "annual"
    gaps = np.diff(index.values).astype("timedelta64[D]").astype(int)
    med = float(np.median(gaps))
    if med <= 3:
        return "daily"
    if med <= 10:
        return "weekly"
    if med <= 45:
        return "monthly"
    if med <= 135:
        return "quarterly"
    return "annual"


def _parse_date(d) -> Optional[pd.Timestamp]:
    """Timestamp for ISO dates and period labels ('2020-Q1', '2020Q1', '2020-03', '2020')."""
    txt = str(d).strip()
    if not txt:
        return None
    for cand in (txt, txt.replace("-Q", "Q")):
        try:
            return pd.Timestamp(cand)
        except (TypeError, ValueError):
            pass
        try:
            return pd.Period(cand).to_timestamp()
        except (TypeError, ValueError):
            pass
    return None


def to_series(observations: List[dict], value_key: Optional[str] = None) -> Optional[pd.Series]:
    """Normalize a connector/preview observation list into a sorted, deduplicated pd.Series."""
    if not observations:
        return None
    dates, values = [], []
    for o in observations:
        d = o.get("date")
        if value_key is not None:
            v = o.get(value_key)
        else:
            v = o.get("value")
            if v is None:  # DataTeam previews key the value by series_id
                others = [x for k, x in o.items() if k != "date"]
                v = others[0] if others else None
        # parse both before keeping either: appending the value first left values and dates
        # of different lengths when a date failed to parse
        try:
            val = float(v)
        except (TypeError, ValueError):
            continue
        ts = _parse_date(d)
        if ts is None:
            continue
        values.append(val)
        dates.append(ts)
    if len(dates) < 2:
        return None
    s = pd.Series(values, index=pd.DatetimeIndex(dates)).sort_index()
    return s[~s.index.duplicated(keep="last")]


def align_panel(series_map: Dict[str, pd.Series]) -> Tuple[pd.DataFrame, str]:
    """Inner-join all series at the coarsest native frequency (downsampling by period mean)."""
    freqs = {name: infer_frequency(s.index) for name, s in series_map.items()}
    coarsest = min(freqs.values(), key=_FREQ_ORDER.index) if freqs else "annual"
    rule = _FREQ_TO_PANDAS[coarsest]
    resampled = {name: s.resample(rule).mean() for name, s in series_map.items()}
    panel = pd.DataFrame(resampled).dropna(how="any")
    return panel, coarsest


def _pct_change(s: pd.Series, periods: int, notes: Optional[List[str]]) -> pd.Series:
    """Percent change, refused per observation where it has no meaning:
    when the base is zero or the series changes sign between base and current period, the
    ratio's sign and size say nothing about growth (a move from -1 to +1 is not -200%).
    Those observations become NaN and the count is disclosed in ``notes``."""
    base = s.shift(periods)
    out = s.pct_change(periods=periods, fill_method=None) * 100.0
    bad = base.notna() & s.notna() & ((base == 0) | (np.sign(base) != np.sign(s)))
    if bad.any():
        out = out.mask(bad)
        if notes is not None:
            notes.append(f"percent change undefined for {int(bad.sum())} observation(s) where "
                         "the series is zero or changes sign between periods; those "
                         "observations are dropped")
    return out


def apply_transform(s: pd.Series, transform: str, frequency: str,
                    notes: Optional[List[str]] = None) -> pd.Series:
    """Apply one named transform. Log transforms refuse non-positive series (returns all-NaN
    would silently shrink the sample, so we raise and let the harness issue an honest note).
    Percent changes are set to NaN (with a note) where the base is zero or the sign flips."""
    if transform == "level":
        return s
    if transform == "log":
        if (s <= 0).any():
            raise ValueError("log transform on non-positive values")
        return np.log(s)
    if transform == "diff":
        return s.diff()
    if transform == "log_diff":
        if (s <= 0).any():
            raise ValueError("log_diff transform on non-positive values")
        return np.log(s).diff()
    if transform == "pct_change":
        return _pct_change(s, 1, notes)
    if transform == "yoy_pct_change":
        return _pct_change(s, _FREQ_PERIODS.get(frequency, 1), notes)
    raise ValueError(f"unknown transform: {transform}")
