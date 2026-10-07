# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Deterministic data checks for the DataTeam.

Series are not zipped by position from 5-row previews, quality is not scored by an LLM from
metadata alone, frequency is not taken from an LLM label, and a dataset is not "validated"
merely because a generated script ran. Everything here is computed from the observations
themselves:

  * :func:`infer_frequency` — frequency from observation dates, not from an LLM label;
  * :func:`discontinued_check` — last observation too old for the requested sample;
  * :func:`series_checks` — constant / all-NaN / duplicate dates / too few observations /
    simulated, which DETERMINE a series' verdict;
  * :func:`align_and_merge` — date-aligned merge at the coarsest common frequency
    (period-mean aggregation, documented), from the actual observations.
"""

from __future__ import annotations

import os
import re
from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd

FREQ_ORDER = ["daily", "weekly", "monthly", "quarterly", "semiannual", "annual"]
_PANDAS_RULE = {"daily": "D", "weekly": "W", "monthly": "MS", "quarterly": "QS",
                "semiannual": "6MS", "annual": "YS"}
# how stale the last observation may be (days) before a series counts as discontinued
_STALE_DAYS = {"daily": 120, "weekly": 120, "monthly": 180, "quarterly": 365,
               "semiannual": 730, "annual": 3 * 365}

AGGREGATION_NOTE = ("aligned at the coarsest common frequency; higher-frequency series "
                    "aggregated by PERIOD MEAN (appropriate for rates, prices and stocks; a "
                    "flow variable would need a period SUM — check units); partial periods "
                    "dropped")


def min_observations() -> int:
    """Observation-count floor for a usable series (env AEL_DATA_MIN_OBS, default 20)."""
    try:
        return max(1, int(os.environ.get("AEL_DATA_MIN_OBS", "20")))
    except ValueError:
        return 20


def parse_period(label) -> Optional[pd.Timestamp]:
    """Period label -> period-START timestamp: YYYY, YYYY-Qn/YYYYQn, YYYY-Sn, YYYY-MM,
    YYYY-Mmm, YYYY-Www, YYYY-MM-DD (time part ignored). None if unparseable."""
    s = str(label or "").strip()
    try:
        m = re.fullmatch(r"(\d{4})", s)
        if m:
            return pd.Timestamp(int(m.group(1)), 1, 1)
        m = re.fullmatch(r"(\d{4})-?Q([1-4])", s, re.IGNORECASE)
        if m:
            return pd.Timestamp(int(m.group(1)), 3 * int(m.group(2)) - 2, 1)
        m = re.fullmatch(r"(\d{4})-?S([12])", s, re.IGNORECASE)
        if m:
            return pd.Timestamp(int(m.group(1)), 1 if m.group(2) == "1" else 7, 1)
        m = re.fullmatch(r"(\d{4})-?M?(\d{2})", s, re.IGNORECASE)
        if m and 1 <= int(m.group(2)) <= 12:
            return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
        m = re.fullmatch(r"(\d{4})-?W(\d{2})", s, re.IGNORECASE)
        if m:
            return pd.Timestamp(date.fromisocalendar(int(m.group(1)), int(m.group(2)), 1))
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
        if m:
            return pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except (ValueError, OverflowError):
        return None
    return None


def to_series(observations: List[Dict], value_key: str = "value") -> pd.Series:
    """[{date, value}, ...] -> float Series on a DatetimeIndex (unparseable dates dropped;
    duplicate dates KEPT so the duplicate check can see them; sorted)."""
    idx, vals = [], []
    for o in observations or []:
        ts = parse_period(o.get("date"))
        if ts is None:
            continue
        v = o.get(value_key)
        if v is None and value_key == "value":
            v = next((x for k, x in o.items() if k != "date"), None)
        try:
            v = float(v) if v is not None else float("nan")
        except (TypeError, ValueError):
            v = float("nan")
        idx.append(ts)
        vals.append(v)
    s = pd.Series(vals, index=pd.DatetimeIndex(idx), dtype=float)
    return s.sort_index(kind="stable")


def infer_frequency(dates) -> str:
    """Frequency from observation dates (median spacing). 'unknown' with < 3 dates,
    'irregular' when the spacing matches no calendar frequency."""
    ts = sorted({d for d in (parse_period(x) if not isinstance(x, pd.Timestamp) else x
                             for x in dates) if d is not None})
    if len(ts) < 3:
        return "unknown"
    gaps = pd.Series(ts).diff().dropna().dt.days
    med = float(gaps.median())
    if med <= 4:
        return "daily"
    if 5 <= med <= 10:
        return "weekly"
    if 25 <= med <= 35:
        return "monthly"
    if 80 <= med <= 100:
        return "quarterly"
    if 170 <= med <= 200:
        return "semiannual"
    if 350 <= med <= 380:
        return "annual"
    return "irregular"


def _requested_end(requested_end) -> date:
    s = str(requested_end or "").strip().lower()
    if not s or s in ("present", "current", "latest", "now", "recent", "ongoing"):
        return date.today()
    m = re.search(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", s)
    if not m:
        return date.today()
    years = re.findall(r"\d{4}", s)
    y = int(years[-1])                       # "1990-2023" -> the END year
    if len(years) > 1 or not m.group(2):
        return min(date(y, 12, 31), date.today())
    return min(date(y, int(m.group(2)), int(m.group(3) or 1)), date.today())


def discontinued_check(last_date, frequency: str, requested_end=None) -> Tuple[bool, str]:
    """(True, note) when the last observation predates the requested sample end by more than
    the frequency's publication-lag allowance (TEDRATE ended 2022, DED1 in 2016)."""
    last = parse_period(last_date) if not isinstance(last_date, pd.Timestamp) else last_date
    if last is None:
        return False, ""
    end = _requested_end(requested_end)
    allowance = _STALE_DAYS.get(frequency, 3 * 365)
    lag = (pd.Timestamp(end) - last).days
    if lag > allowance:
        return True, (f"DISCONTINUED/STALE: last observation {last.date()} is {lag} days before "
                      f"the requested sample end {end} (allowance for {frequency or 'unknown'} "
                      f"data: {allowance} days)")
    return False, ""


def series_checks(observations: List[Dict], simulated: bool = False,
                  min_obs: Optional[int] = None, preview_only: bool = False) -> Dict:
    """Deterministic checks on one series' actual observations. ``verdict`` is 'fail' when
    any blocking check fails; ``issues`` lists every failed check in words."""
    min_obs = min_obs or min_observations()
    s = to_series(observations)
    n_total = int(len(s))
    vals = s.dropna()
    n_valid = int(len(vals))
    dup = int(s.index.duplicated().sum())
    constant = n_valid >= 2 and float(vals.max() - vals.min()) == 0.0
    checks = {
        "simulated": bool(simulated),
        "all_nan": n_valid == 0,
        "constant": bool(constant),
        "duplicate_dates": dup,
        "n_observations": n_valid,
        "below_min_observations": n_valid < min_obs,
        "min_observations": min_obs,
        "preview_only": bool(preview_only),
        "frequency": infer_frequency(list(vals.index)),
        "missing_share": round(1 - n_valid / n_total, 4) if n_total else 1.0,
    }
    issues = []
    if checks["simulated"]:
        issues.append("simulated placeholder, not retrieved data")
    if checks["all_nan"]:
        issues.append("no numeric observations")
    if checks["constant"]:
        issues.append(f"constant series (every value = {float(vals.iloc[0]):g})")
    if dup:
        issues.append(f"{dup} duplicate date(s)")
    if checks["below_min_observations"]:
        issues.append(f"only {n_valid} observation(s) (< {min_obs})"
                      + (" — only the stored preview was available" if preview_only else ""))
    # non-blocking: a frozen tail (the latest values all identical) — e.g. a tracker that has
    # reported 0 for months; the series is real but its recent values carry no variation
    warnings = []
    tail = vals.iloc[-6:]
    if not constant and len(vals) >= 12 and len(tail) == 6 and float(tail.max() - tail.min()) == 0.0:
        run = 6
        while run < len(vals) and float(vals.iloc[-run - 1]) == float(tail.iloc[-1]):
            run += 1
        warnings.append(f"last {run} observations identical (= {float(tail.iloc[-1]):g})")
    checks["frozen_tail"] = bool(warnings)
    checks["warnings"] = warnings
    checks["issues"] = issues
    checks["verdict"] = "fail" if issues else "pass"
    return checks


def coarsest(frequencies: List[str]) -> str:
    known = [f for f in frequencies if f in FREQ_ORDER]
    return max(known, key=FREQ_ORDER.index) if known else "annual"


def align_and_merge(series_map: Dict[str, pd.Series],
                    target_frequency: Optional[str] = None) -> Tuple[pd.DataFrame, str, Dict]:
    """Date-aligned merge. Each series is de-duplicated (last value per date), resampled to the
    target frequency (default: the coarsest native one) by period mean, and outer-joined on
    the period index. Returns (frame, target_frequency, per-series info)."""
    freqs = {k: infer_frequency(list(v.dropna().index)) for k, v in series_map.items()}
    target = target_frequency or coarsest(list(freqs.values()))
    rule = _PANDAS_RULE.get(target, "YS")
    cols, info = {}, {}
    for k, s in series_map.items():
        s = s[~s.index.duplicated(keep="last")].dropna()
        if s.empty:
            info[k] = {"native_frequency": freqs[k], "aligned_observations": 0,
                       "method": "empty"}
            continue
        native = freqs[k]
        if native == target:
            method = "no_conversion (snapped to period start)"
        elif native in FREQ_ORDER and FREQ_ORDER.index(native) > FREQ_ORDER.index(target):
            method = (f"NOT upsampled: native {native} is coarser than target {target}; values "
                      "placed at their period start, other periods left missing")
        else:
            method = f"period mean ({native} -> {target})"
        r = s.resample(rule).mean().dropna()
        if native != target and len(r) > 1:
            # a period mean over a partial period (the current year's Jan-Aug) is not the
            # period's value: keep periods holding >= 75% of the typical native count
            counts = s.resample(rule).count().reindex(r.index)
            r = r[counts >= 0.75 * float(counts.median())]
            method += "; partial periods (< 75% of the typical native count) dropped"
        cols[k] = r
        info[k] = {"native_frequency": native, "aligned_observations": int(len(r)),
                   "method": method,
                   "start": str(r.index[0].date()) if len(r) else "",
                   "end": str(r.index[-1].date()) if len(r) else ""}
    frame = pd.DataFrame(cols).sort_index() if cols else pd.DataFrame()
    frame.index.name = "date"
    return frame, target, info
