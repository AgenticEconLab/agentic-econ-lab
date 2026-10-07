# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Load the DataTeam's artifact into a full-length aligned panel.

The DataTeam artifact records provenance (source_name + series_id) but only a 5-point
``data_preview`` — real estimation needs the complete series, so this loader RE-FETCHES
each series through the same open connectors DataTeam used:

    FRED           -> fredapi native client (FRED_API_KEY)
    Yahoo Finance  -> yfinance native client
    everything else-> shared.tools.econ_connectors.fetch_series (keyless open REST)

A series that cannot be re-fetched falls back to its preview ONLY as a disclosed note — with
5 points it will fail the harness's min-observation gate and yield an honest ``inestimable``
rather than a fit on fabricated coverage. Simulated series (``data_simulated: True``) are
excluded up front: estimating on disclosed-simulated data would launder it into "empirics".

Archived snapshot first (pre-rerun review #5): when the DataTeam source stage recorded its
full observations (``metadata.observations_file``), each series is loaded from that snapshot —
the same vintage the DataTeam checked — and the DataTeam's deterministic checks
(:func:`DataTeam.ael.data_checks.series_checks`: constant, all-NaN, duplicate dates,
simulated) are applied to it; a failing series is excluded with a note, never re-fetched.
Only a series ABSENT from the snapshot is fetched again, and the note records that a new
fetch (a new vintage) was used. Standalone artifacts without a snapshot keep the re-fetch
path.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .transforms import to_series

# Older DataTeam artifacts record the repaired canonical fetch id only inside the
# quality_notes text: "[id repaired via DBnomics search: 'ORIG' (SRC) -> CANONICAL]".
# When the first-class field is absent, recover it here —
# re-fetching with the ORIGINAL id is exactly the near-miss the repair already fixed once.
_REPAIR_RE = re.compile(r"->\s*([^\s\]]+)\]")


def repaired_fetch_id(item: dict) -> Optional[str]:
    m = _REPAIR_RE.search(str(item.get("quality_notes", "")))
    return m.group(1) if m else None


def find_retrieved_data(obj: Any) -> List[dict]:
    """Locate the ``retrieved_data`` list anywhere inside a DataTeam output / pipeline
    artifact envelope ({name, data, ...}) — schema-agnostic on purpose."""
    if isinstance(obj, dict):
        rd = obj.get("retrieved_data")
        if isinstance(rd, list) and rd:
            return rd
        for v in obj.values():
            found = find_retrieved_data(v)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_retrieved_data(v)
            if found:
                return found
    return []


def _find_key(obj: Any, key: str) -> Any:
    """First value stored under ``key`` anywhere inside a (nested) artifact envelope."""
    if isinstance(obj, dict):
        if obj.get(key):
            return obj[key]
        for v in obj.values():
            found = _find_key(v, key)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_key(v, key)
            if found:
                return found
    return None


def load_observation_snapshot(data_artifact: Any) -> Tuple[Dict[str, List[dict]], str]:
    """The DataTeam's archived full observations ({series_id: [{date, value}, ...]}) named in
    ``metadata.observations_file``, and that path. ({}, "") when the artifact records none or
    the file is unreadable (standalone runs)."""
    path = _find_key(data_artifact, "observations_file")
    if not path or not os.path.exists(str(path)):
        return {}, str(path or "")
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): v for k, v in (data or {}).items() if isinstance(v, list)}, str(path)
    except Exception:
        return {}, str(path)


# the DataTeam's deterministic checks that make a series unusable whatever the sample size
# (the observation-count floor is applied separately by ``min_series_obs``)
_BLOCKING_CHECKS = ("simulated", "all_nan", "constant", "duplicate_dates")


def snapshot_check_failures(observations: List[dict], simulated: bool = False) -> List[str]:
    """Failed blocking DataTeam checks on one archived series ([] = usable)."""
    try:
        from DataTeam.ael.data_checks import series_checks
    except Exception:                                  # standalone EstimationTeam install
        return []
    checks = series_checks(observations, simulated=simulated, min_obs=1)
    if not any(checks.get(k) for k in _BLOCKING_CHECKS):
        return []
    return [i for i in checks.get("issues") or [] if not i.startswith("only ")]


def _fetch_fred(series_id: str) -> Optional[pd.Series]:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        return None
    try:
        from fredapi import Fred
        data = Fred(api_key=api_key).get_series(series_id)
        return data.dropna() if data is not None and len(data) else None
    except Exception:
        return None


def _fetch_yahoo(series_id: str) -> Optional[pd.Series]:
    try:
        import yfinance as yf
        hist = yf.Ticker(series_id).history(period="max", auto_adjust=True)
        if hist is None or hist.empty:
            return None
        close = hist["Close"].dropna()
        close.index = pd.DatetimeIndex(close.index.tz_localize(None))
        return close
    except Exception:
        return None


def _fetch_connector(source_name: str, series_id: str, collector=None) -> Optional[pd.Series]:
    try:
        from shared.tools.econ_connectors import fetch_series
        result = fetch_series(source_name, series_id, collector=collector, agent="EstimationLoader")
        if result and result.get("observations"):
            return to_series(result["observations"])
    except Exception:
        pass
    return None


def load_panel(
    data_artifact: Any,
    collector=None,
    min_series_obs: int = 8,
) -> Tuple[Optional[Dict[str, pd.Series]], Dict[str, str], List[str]]:
    """Load full-length series from a DataTeam artifact — UNJOINED.

    Returns (series_map, alias_map, notes): ``series_map`` maps series_id -> full pd.Series;
    ``alias_map`` maps lower-cased series_id AND series_name to the series_id so specs may
    reference either; ``notes`` disclose every fallback and exclusion. series_map is None
    when nothing usable loaded.

    Alignment is deliberately NOT done here: joining all series at once lets one sparse
    series (say, an intermittent Gini) empty the intersection for everyone. The harness
    aligns only the variables a specification actually uses (per-spec maximal sample).
    """
    retrieved = find_retrieved_data(data_artifact)
    notes: List[str] = []
    series_map: Dict[str, pd.Series] = {}
    alias_map: Dict[str, str] = {}
    snapshot, snapshot_path = load_observation_snapshot(data_artifact)
    if snapshot:
        notes.append(f"loaded the DataTeam's archived observations ({len(snapshot)} series, "
                     f"{snapshot_path}); only series absent from it are fetched again")
    elif snapshot_path:
        notes.append(f"the DataTeam's observations file {snapshot_path} is missing or "
                     "unreadable — series are fetched again (new vintage)")

    for item in retrieved:
        sid = str(item.get("series_id", "")).strip()
        sname = str(item.get("series_name", "")).strip()
        source = str(item.get("source_name", "")).strip()
        if not sid:
            continue
        simulated = item.get("data_simulated")
        if simulated is True or str(simulated).strip().lower() in ("true", "1", "yes"):
            notes.append(f"excluded {sid} ({source}): data_simulated=True — simulated series "
                         "are never estimated as empirics")
            continue

        s: Optional[pd.Series] = None
        if sid in snapshot:
            failed = snapshot_check_failures(snapshot[sid])
            if failed:
                notes.append(f"excluded {sid} ({source}): failed the DataTeam's deterministic "
                             f"checks on the archived observations — {'; '.join(failed)}")
                continue
            try:
                s = to_series(snapshot[sid])
            except Exception as e:  # never abort estimation on one archived series
                notes.append(f"excluded {sid} ({source}): archived observations could not be "
                             f"converted ({type(e).__name__})")
                continue
            if s is None:
                notes.append(f"excluded {sid} ({source}): fewer than 2 usable archived "
                             "observations")
                continue
            notes.append(f"{sid} ({source}): {len(s)} archived observations from the "
                         f"DataTeam (vintage retrieved "
                         f"{item.get('retrieval_date') or 'at the DataTeam stage'})")
        # Prefer the first-class canonical fetch coordinates recorded by DataTeam
        fetch_id = item.get("fetch_id")
        fetch_source = item.get("fetch_source")
        archived = s is not None
        if fetch_id and s is None:
            fsrc = str(fetch_source or source)
            if fsrc.upper() == "FRED":
                s = _fetch_fred(str(fetch_id))
            else:
                s = _fetch_connector(fsrc, str(fetch_id), collector=collector)
            if s is not None:
                notes.append(f"{sid} ({source}): re-fetched via canonical fetch_id "
                             f"{fsrc}/{fetch_id}")
        if s is None and source.upper() == "FRED":
            s = _fetch_fred(sid)
        elif s is None and source.lower().startswith("yahoo"):
            s = _fetch_yahoo(sid)
        elif s is None:
            fix = repaired_fetch_id(item)   # legacy artifacts: regex from quality_notes
            if fix:
                s = _fetch_connector("DBnomics", fix, collector=collector)
                if s is not None:
                    notes.append(f"{sid} ({source}): re-fetched via the repaired canonical id "
                                 f"{fix} recorded in quality_notes")
            if s is None:
                s = _fetch_connector(source, sid, collector=collector)

        fetched_new = s is not None and not archived
        if fetched_new:
            notes.append(f"{sid} ({source}): "
                         + ("absent from the DataTeam's archived observations — "
                            if snapshot else "")
                         + f"NEW fetch at estimation time (new vintage, retrieved "
                         f"{date.today().isoformat()}; the DataTeam retrieved "
                         f"{item.get('retrieval_date') or 'an earlier vintage'})")
        if s is None:
            s = to_series(item.get("data_preview") or [])
            if s is not None:
                notes.append(f"{sid} ({source}): full re-fetch failed — falling back to the "
                             f"{len(s)}-point artifact preview (disclosed)")
        if s is None or len(s) < 2:
            notes.append(f"excluded {sid} ({source}): no usable observations")
            continue
        if len(s) < min_series_obs:
            # A handful of points can never be estimated, but inner-joining it would destroy
            # the overlap of every OTHER series — exclude it and say so.
            notes.append(f"excluded {sid} ({source}): only {len(s)} observations "
                         f"(< {min_series_obs}) — would poison the panel overlap")
            continue

        series_map[sid] = s
        alias_map[sid.lower()] = sid
        if sname:
            alias_map[sname.lower()] = sid
        # The source's own title is an alias too, so the Estimator sees what the series
        # IS; a proxy relation travels into the estimation artifact (and the report) as a note.
        title = str(item.get("source_title") or "").strip()
        if title:
            alias_map[title.lower()] = sid
        if item.get("proxy_for"):
            notes.append(f"proxy disclosure: {sid} ({source}) is the source's '{title or sid}', "
                         f"used for the requested variable '{item.get('proxy_for')}'")

    if not series_map:
        notes.append("no real series could be loaded from the data artifact")
        return None, alias_map, notes

    notes.append(f"loaded {len(series_map)} full series "
                 f"({', '.join(f'{k}:{len(v)}' for k, v in series_map.items())})")
    return series_map, alias_map, notes
