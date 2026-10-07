# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Open, keyless connectors for real economic data — the DataTeam's real-retrieval registry.

Why: the retrieval dispatch had exactly TWO real connectors (FRED via fredapi, Yahoo Finance via
yfinance); every other "source" silently fell to simulation, so FRED was structurally the only real
data in every macro run. This module adds real connectors for the major OPEN
economic-data APIs — all keyless, matching the project's open-stack default:

  * **World Bank**   api.worldbank.org           id: ``NY.GDP.MKTP.CD`` or ``DE:SL.UEM.TOTL.ZS``
  * **DBnomics**     api.db.nomics.world         id: ``PROVIDER/DATASET/SERIES`` (aggregates 80+
                     providers: OECD, IMF, Eurostat, ECB, BIS, BLS, INSEE, Bundesbank, ...)
  * **OECD / IMF / BIS**  via DBnomics           id: ``DATASET/SERIES`` (provider prepended)
  * **ECB**          data-api.ecb.europa.eu      id: ``FLOW/KEY`` e.g. ``EXR/D.USD.EUR.SP00.A``
  * **Eurostat**     ec.europa.eu/eurostat       id: ``dataset?dim=..`` e.g.
                     ``une_rt_m?geo=EA20&s_adj=SA&sex=T&age=TOTAL&unit=PC_ACT``
  * **BLS**          api.bls.gov v2              id: ``CUUR0000SA0`` (unregistered tier)

All HTTP goes through ``shared.observability.tracked_get`` (per-call metrics + the AEL_HTTP_TIMEOUT
bound). Every fetcher returns a NORMALIZED dict or None — never raises, never fabricates:

    {"observations": [{"date": "...", "value": float}, ...],   # chronological
     "n_obs": int, "start_date": str, "end_date": str, "source_name": str}

``fetch_series(source_name, series_id)`` dispatches by canonical source name;
``connected_sources()`` lists what is genuinely implemented (for truth-in-prompting + the
committee's source-approval context). A fetch that yields no observations returns None so the
caller can fall back to DISCLOSED simulation."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------------------------
# normalization helpers
# ---------------------------------------------------------------------------------------------

def _today():
    """Retrieval date (a function so tests can pin it)."""
    from datetime import date
    return date.today()


def period_end(date_str: str):
    """Last calendar day covered by an observation's period label, or None if unparseable.

    Handles ``YYYY``, ``YYYY-Qn``/``YYYYQn``, ``YYYY-Sn``, ``YYYY-MM``, ``YYYY-Mmm``,
    ``YYYY-Www`` and ``YYYY-MM-DD`` (a dated observation covers that day)."""
    import calendar
    from datetime import date
    s = str(date_str or "").strip()
    try:
        m = re.fullmatch(r"(\d{4})", s)
        if m:
            return date(int(m.group(1)), 12, 31)
        m = re.fullmatch(r"(\d{4})-?Q([1-4])", s, re.IGNORECASE)
        if m:
            y, q = int(m.group(1)), int(m.group(2))
            return date(y, 3 * q, calendar.monthrange(y, 3 * q)[1])
        m = re.fullmatch(r"(\d{4})-?S([12])", s, re.IGNORECASE)
        if m:
            y = int(m.group(1))
            return date(y, 6, 30) if m.group(2) == "1" else date(y, 12, 31)
        m = re.fullmatch(r"(\d{4})-?M?(\d{2})", s, re.IGNORECASE)
        if m and 1 <= int(m.group(2)) <= 12:
            y, mo = int(m.group(1)), int(m.group(2))
            return date(y, mo, calendar.monthrange(y, mo)[1])
        m = re.fullmatch(r"(\d{4})-?W(\d{2})", s, re.IGNORECASE)
        if m:
            return date.fromisocalendar(int(m.group(1)), int(m.group(2)), 7)
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
        if m:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None
    return None


def drop_projections(obs: List[Dict[str, Any]], release_date: Optional[str] = None):
    """Split off observations whose period ends after the retrieval date or after the
    series' release/vintage date — those values are PROJECTIONS (IMF WEO vintages run to
    2030; CBO potential output to 2035), not observed data.

    Returns ``(kept, n_dropped, cutoff_iso)``. Unparseable dates are kept."""
    from datetime import date
    cutoff = _today()
    if release_date:
        rel = period_end(str(release_date)[:10])
        if rel is not None and rel < cutoff:
            cutoff = rel
    kept = []
    dropped = 0
    for o in obs:
        end = period_end(o.get("date", ""))
        if end is not None and end > cutoff:
            dropped += 1
        else:
            kept.append(o)
    return kept, dropped, cutoff.isoformat()


def _norm(obs: List[Dict[str, Any]], source_name: str,
          series_title: Optional[str] = None,
          release_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Chronologically-sorted, value-parsed normalization; None when nothing usable.

    ``series_title`` is the SOURCE's own name for the series (e.g. World Bank "Gini index"),
    kept so a series matched to a differently named requirement is disclosed as a proxy.
    Observations after the retrieval date / ``release_date`` (vintage) are projections and are
    EXCLUDED here, so every caller — DataTeam and the estimation loader — sees only
    observed data; the count and cutoff are reported in ``n_projection_excluded`` /
    ``projection_cutoff``."""
    clean = []
    for o in obs:
        try:
            v = float(o["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if v != v:                                   # NaN
            continue
        d = str(o.get("date", "")).strip()
        if d:
            clean.append({"date": d, "value": v})
    clean, n_proj, cutoff = drop_projections(clean, release_date)
    if not clean:
        return None
    clean.sort(key=lambda o: o["date"])
    out = {"observations": clean, "n_obs": len(clean),
           "start_date": clean[0]["date"], "end_date": clean[-1]["date"],
           "source_name": source_name, "series_title": series_title}
    if n_proj:
        out["n_projection_excluded"] = n_proj
        out["projection_cutoff"] = cutoff
    return out


def _get_json(url: str, params: Optional[dict] = None, collector=None, agent: str = ""):
    from shared.observability import tracked_get
    resp = tracked_get(url, params=params, collector=collector, agent=agent, retries=2)
    return resp.json()


# ---------------------------------------------------------------------------------------------
# connectors (each: series_id -> normalized dict or None; never raises)
# ---------------------------------------------------------------------------------------------

def fetch_world_bank(series_id: str, collector=None, agent: str = "") -> Optional[Dict[str, Any]]:
    """World Bank Open Data. id = ``INDICATOR`` (country defaults to US) or ``COUNTRY:INDICATOR``."""
    try:
        country, _, indicator = series_id.strip().rpartition(":")
        country = country or "US"
        data = _get_json(
            f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}",
            params={"format": "json", "per_page": "500"}, collector=collector, agent=agent)
        rows = data[1] if isinstance(data, list) and len(data) > 1 and data[1] else []
        title = None
        if rows and isinstance(rows[0].get("indicator"), dict):
            title = rows[0]["indicator"].get("value")
        return _norm([{"date": r.get("date"), "value": r.get("value")} for r in rows], "World Bank",
                     series_title=title)
    except Exception:
        return None


def fetch_dbnomics(series_id: str, collector=None, agent: str = "") -> Optional[Dict[str, Any]]:
    """DBnomics aggregator. id = ``PROVIDER/DATASET/SERIES`` (e.g. ``OECD/KEI/PRINTO01.USA.GY.M``)."""
    try:
        sid = series_id.strip().strip("/")
        if sid.count("/") != 2:
            return None
        data = _get_json(f"https://api.db.nomics.world/v22/series/{sid}",
                         params={"observations": "1"}, collector=collector, agent=agent)
        docs = (data.get("series") or {}).get("docs") or []
        if not docs:
            return None
        periods = docs[0].get("period") or []
        values = docs[0].get("value") or []
        # indexed_at = when this vintage entered DBnomics: values for periods ending after it
        # (WEO projections) are not observations
        return _norm([{"date": p, "value": v} for p, v in zip(periods, values)], "DBnomics",
                     series_title=docs[0].get("series_name"),
                     release_date=docs[0].get("indexed_at"))
    except Exception:
        return None


def _via_dbnomics(provider: str, source_name: str):
    """OECD/IMF/BIS ride DBnomics: id = ``DATASET/SERIES`` with the provider prepended."""
    def fetch(series_id: str, collector=None, agent: str = "") -> Optional[Dict[str, Any]]:
        out = fetch_dbnomics(f"{provider}/{series_id.strip().strip('/')}", collector, agent)
        if out:
            out["source_name"] = source_name
        return out
    return fetch


def fetch_ecb(series_id: str, collector=None, agent: str = "") -> Optional[Dict[str, Any]]:
    """ECB Data Portal (SDMX). id = ``FLOW/KEY`` (e.g. ``EXR/D.USD.EUR.SP00.A``)."""
    try:
        from shared.observability import tracked_get
        flow, _, key = series_id.strip().partition("/")
        if not key:
            return None
        resp = tracked_get(f"https://data-api.ecb.europa.eu/service/data/{flow}/{key}",
                           params={"format": "csvdata"}, collector=collector, agent=agent, retries=2)
        import csv as _csv
        import io as _io
        rows = list(_csv.reader(_io.StringIO(resp.text)))   # quote-aware: TITLE fields contain commas
        if len(rows) < 2:
            return None
        header = [h.strip() for h in rows[0]]
        t_i, v_i = header.index("TIME_PERIOD"), header.index("OBS_VALUE")
        # the series' own title (for the concept check); TITLE_COMPL when TITLE is absent
        ti_i = next((header.index(h) for h in ("TITLE", "TITLE_COMPL") if h in header), None)
        obs = []
        title = None
        for cells in rows[1:]:
            if len(cells) > max(t_i, v_i):
                obs.append({"date": cells[t_i], "value": cells[v_i]})
            if title is None and ti_i is not None and len(cells) > ti_i and cells[ti_i].strip():
                title = cells[ti_i].strip()
        return _norm(obs, "ECB", series_title=title)
    except Exception:
        return None


def fetch_eurostat(series_id: str, collector=None, agent: str = "") -> Optional[Dict[str, Any]]:
    """Eurostat (JSON-stat). id = ``dataset?dim=val&...`` — filters must pin all non-time dims,
    e.g. ``une_rt_m?geo=DE&s_adj=SA&sex=T&age=TOTAL&unit=PC_ACT``."""
    try:
        dataset, _, query = series_id.strip().partition("?")
        params = {"format": "JSON"}
        for pair in query.split("&"):
            k, _, v = pair.partition("=")
            if k and v:
                params[k] = v
        data = _get_json(
            f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}",
            params=params, collector=collector, agent=agent)
        values = data.get("value") or {}
        dims = data.get("dimension") or {}
        time_dim = dims.get("time") or {}
        idx = (time_dim.get("category") or {}).get("index") or {}
        # JSON-stat flattens ALL dims; the flat index aligns with the time index ONLY when
        # every non-time dimension is pinned to a single category. If the caller's filters
        # left a dimension open, the mapping would silently return ONE unidentified category —
        # decline honestly instead (the caller falls back to disclosed simulation).
        for name, d in dims.items():
            if name == "time" or not isinstance(d, dict):
                continue
            cats = (d.get("category") or {}).get("index") or {}
            if len(cats) > 1:
                return None
        by_pos = {int(pos): label for label, pos in idx.items()}
        obs = [{"date": by_pos.get(int(i), ""), "value": v} for i, v in values.items()]
        return _norm(obs, "Eurostat")
    except Exception:
        return None


def fetch_bls(series_id: str, collector=None, agent: str = "") -> Optional[Dict[str, Any]]:
    """BLS Public API v2 (unregistered tier). id = a BLS series id (e.g. ``CUUR0000SA0``)."""
    try:
        from datetime import datetime
        end = datetime.now().year
        data = _get_json(f"https://api.bls.gov/publicAPI/v2/timeseries/data/{series_id.strip()}",
                         params={"startyear": str(end - 9), "endyear": str(end)},  # 10y = tier max
                         collector=collector, agent=agent)
        if data.get("status") != "REQUEST_SUCCEEDED":
            return None
        series = (data.get("Results") or {}).get("series") or []
        obs = []
        for row in (series[0].get("data") or []) if series else []:
            period = row.get("period", "")           # M01..M12 / Q01.. / A01
            year = row.get("year", "")
            if period == "M13":
                continue                              # annual-average pseudo-month — not monthly data
            if period.startswith("M"):
                date = f"{year}-{period[1:]}"
            elif period.startswith("Q"):
                date = f"{year}-Q{period[2:]}"
            else:
                date = year
            obs.append({"date": date, "value": row.get("value")})
        return _norm(obs, "BLS")
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------------------------

_CONNECTORS = {
    "world bank": fetch_world_bank,
    "dbnomics": fetch_dbnomics,
    "oecd": _via_dbnomics("OECD", "OECD"),
    "imf": _via_dbnomics("IMF", "IMF"),
    "bis": _via_dbnomics("BIS", "BIS"),
    "ecb": fetch_ecb,
    "eurostat": fetch_eurostat,
    "bls": fetch_bls,
}

# Canonical display names + id formats — single source of truth for the discovery prompt
# (truth-in-prompting) and the committee's source-approval context.
CONNECTED_SOURCES: Dict[str, str] = {
    "FRED": "series id, e.g. CPIAUCSL, FEDFUNDS",
    "Yahoo Finance": "ticker, e.g. ^GSPC, AAPL",
    "World Bank": "indicator (US default) or COUNTRY:INDICATOR, e.g. NY.GDP.MKTP.CD or DE:SL.UEM.TOTL.ZS",
    "DBnomics": "PROVIDER/DATASET/SERIES, e.g. OECD/KEI/PRINTO01.USA.GY.M",
    "OECD": "DATASET/SERIES (via DBnomics), e.g. KEI/PRINTO01.USA.GY.M",
    "IMF": "DATASET/SERIES (via DBnomics), e.g. WEO:2025-04/USA.NGDP_RPCH",
    "BIS": "DATASET/SERIES (via DBnomics)",
    "ECB": "FLOW/KEY, e.g. EXR/D.USD.EUR.SP00.A",
    "Eurostat": "dataset?dims (pin all non-time dims), e.g. une_rt_m?geo=DE&s_adj=SA&sex=T&age=TOTAL&unit=PC_ACT",
    "BLS": "series id, e.g. CUUR0000SA0 (CPI-U), CES0500000003 (avg hourly earnings)",
}


def connected_sources() -> List[str]:
    """Names of sources with a REAL connector (FRED/Yahoo handled by the DataTeam stage itself)."""
    return list(CONNECTED_SOURCES)


def fetch_series(source_name: str, series_id: str, collector=None, agent: str = "") -> Optional[Dict[str, Any]]:
    """Fetch real observations for ``series_id`` from ``source_name`` (case-insensitive).

    Returns the normalized dict, or None when the source has no connector here or the fetch
    yielded nothing — the caller then falls back to DISCLOSED simulation, never a silent fake."""
    fn = _CONNECTORS.get(str(source_name or "").strip().lower())
    if fn is None:
        return None
    return fn(str(series_id or ""), collector=collector, agent=agent)


# ---------------------------------------------------------------------------------------------
# id repair — simulated series in full runs typically trace to an LLM
# series-id near-miss (GER for DEU; invalid OECD/IMF dataset keys). One bounded repair pass:
# country-code normalization, then a lookup through DBnomics' open free-text search.
# ---------------------------------------------------------------------------------------------

# Common LLM country-code mistakes -> valid World Bank codes (WB accepts ISO2/ISO3).
_COUNTRY_ALIASES = {
    "GER": "DEU", "UK": "GBR", "ENG": "GBR", "SPA": "ESP", "NED": "NLD",
    "HOL": "NLD", "SUI": "CHE", "SWI": "CHE", "POR": "PRT", "GRE": "GRC",
    "DEN": "DNK", "IRE": "IRL", "SOUTH KOREA": "KOR",
    # (no "SK": that is the ISO 3166-1 code of Slovakia, not an alias of Korea)
}


def _world_bank_iso3(series_id: str) -> Tuple[Optional[str], str]:
    """(ISO3 or None, raw country code) of a World Bank ``COUNTRY:INDICATOR`` id. An id with
    no country prefix is the connector's documented US default. Common LLM aliases (GER,
    UK, ...) are mapped first; any other code must be a real ISO 3166-1 / World Bank
    aggregate code — an unknown one yields None (refused), never a default country."""
    from shared.tools.iso_countries import to_iso3
    sid = str(series_id or "")
    raw = (sid.partition(":")[0] if ":" in sid else "US").strip().upper()
    return to_iso3(_COUNTRY_ALIASES.get(raw, raw)), raw


# Geography tokens accepted per hint: a repaired series must MENTION the requested
# geography in its code or name — the first search hit for "CO2 emissions" was an ARUBA
# series that then sailed through QA at 84/100. Wrong-country real data is WORSE than
# disclosed simulation, so a candidate that fails the check is rejected.
_GEO_TOKENS = {
    "US": ("USA", "UNITED STATES", "-US-", ".US."),
    "USA": ("USA", "UNITED STATES"),
}


def _matches_geo(text: str, geo_hint: str) -> bool:
    tokens = _GEO_TOKENS.get((geo_hint or "US").strip().upper(),
                             ((geo_hint or "").strip().upper(),))
    t = (text or "").upper()
    return any(tok and tok in t for tok in tokens)


def search_dbnomics_series(query: str, collector=None, agent: str = "",
                           geo_hint: str = "US") -> Optional[str]:
    """Resolve a human series description to a full DBnomics id (``PROVIDER/DATASET/SERIES``)
    via the open, keyless DBnomics search: dataset-level free-text search first, then a
    series-level search inside each of the top datasets (scanning only the single top
    dataset missed valid repairs whose home dataset ranked 2nd/3rd). Candidates whose
    code/name do not mention ``geo_hint`` are REJECTED (no silent wrong-country
    substitutions). Returns the id or None. Never raises."""
    q = (query or "").strip()
    if not q:
        return None
    try:
        data = _get_json("https://api.db.nomics.world/v22/search",
                         params={"q": q, "limit": "3"}, collector=collector, agent=agent)
        docs = (data.get("results") or {}).get("docs") or []
        for doc in docs:
            provider, dataset = doc.get("provider_code"), doc.get("code")
            if not provider or not dataset:
                continue
            hit = _geo_series_in_dataset(provider, dataset, q, geo_hint,
                                         collector=collector, agent=agent)
            if hit:
                return hit
        return None                                    # no geography-consistent candidate
    except Exception:
        return None


def _geo_series_in_dataset(provider: str, dataset: str, query: str, geo_hint: str,
                           collector=None, agent: str = "") -> Optional[str]:
    """First geography-consistent series id inside one DBnomics dataset, or None."""
    try:
        data = _get_json(f"https://api.db.nomics.world/v22/series/{provider}/{dataset}",
                         params={"q": f"{query} {geo_hint}".strip(), "limit": "5",
                                 "observations": "0"},
                         collector=collector, agent=agent)
        for sd in (data.get("series") or {}).get("docs") or []:
            code = sd.get("series_code") or ""
            name = sd.get("series_name") or ""
            if code and _matches_geo(f"{code} {name}", geo_hint):
                return f"{provider}/{dataset}/{code}"
    except Exception:
        pass
    return None


def search_wdi_series(query: str, collector=None, agent: str = "",
                      geo_hint: str = "US") -> Optional[str]:
    """World-Bank-pinned repair: search INSIDE DBnomics' WB/WDI dataset, where the
    generic dataset-level search often never lands. WDI series codes embed the indicator id
    and the country (e.g. ``A-GC.TAX.TOTL.GD.ZS-USA``), so the geography guard applies
    naturally. Returns a full DBnomics id or None. Never raises."""
    q = (query or "").strip()
    if not q:
        return None
    return _geo_series_in_dataset("WB", "WDI", q, geo_hint, collector=collector, agent=agent)


# Tokens too generic to certify a search hit on their own (they appear in most titles).
_SEARCH_STOP = {"rate", "ratio", "index", "total", "level", "of", "the", "for", "and",
                "all", "percent", "united", "states", "us", "usa"}


def _core_tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
            if t not in _SEARCH_STOP and len(t) > 2}


# Example: "CPI: Food" once matched core CPI ("All Items LESS FOOD and Energy") —
# token coverage is blind to negation. Strip "less/excluding X"-style clauses (up to the
# next delimiter) from BOTH query and title before comparing, so a component named only
# inside an exclusion can never certify a hit — and a genuine "CPI less food and energy"
# query still matches the core-CPI title (both sides lose the clause symmetrically).
_NEGATION_CLAUSE = re.compile(
    r"\b(?:less|excluding|except|ex)\b[^,:;(]*", re.IGNORECASE)


def _strip_negation_clauses(text: str) -> str:
    return _NEGATION_CLAUSE.sub(" ", text or "")


def search_fred_series(query: str, collector=None, agent: str = "") -> Optional[str]:
    """FRED full-text index search (mechanism-first, 2026-07-12): resolve a human series
    name to a FRED id at runtime instead of growing the curated table per run.

    Relevance-guarded: EVERY core token of the query must appear in the hit title, else
    decline (FRED's index is loose; 'infant mortality' returns
    life-expectancy series). Strict precision over recall: paraphrase staples the guard
    rejects belong in the small curated layer; the long tail falls through to the WDI /
    DBnomics rungs. Needs FRED_API_KEY; returns the id or None. Never raises."""
    q = (query or "").strip()
    api_key = os.environ.get("FRED_API_KEY")
    if not q or not api_key:
        return None
    want = _core_tokens(_strip_negation_clauses(q))
    if not want:
        return None
    try:
        data = _get_json("https://api.stlouisfed.org/fred/series/search",
                         params={"search_text": q, "api_key": api_key, "file_type": "json",
                                 "limit": "5", "order_by": "popularity",
                                 "sort_order": "desc"},
                         collector=collector, agent=agent)
        for hit in data.get("seriess") or []:
            title = hit.get("title") or ""
            if hit.get("id") and want <= _core_tokens(_strip_negation_clauses(title)):
                # geography guard: FRED mirrors international series ("... for Lao PDR").
                # A "for <place>" clause that isn't the United States is a wrong-geo hit.
                m = re.search(r"\bfor (.+)$", title, flags=re.IGNORECASE)
                if m and not re.search(r"united states|u\.s\b", m.group(1), re.IGNORECASE):
                    continue
                return str(hit["id"])
        return None
    except Exception:
        return None


# In-process cache: the feasibility loop re-grades every requirement each cycle, and the
# scout consults the resolver for each pool miss — one lookup per distinct name is enough.
_RESOLVE_CACHE: Dict[str, Optional[Tuple[str, str]]] = {}


class FetchableHit(tuple):
    """``(source, series_id)`` plus the concept-check result against the requested name.

    A plain 2-tuple to existing callers (``source, sid = hit``; ``hit == ("FRED", "GCE")``);
    the feasibility scout reads ``concept_status`` (``match`` / ``mismatch`` /
    ``unverified``), ``title`` and ``reason`` to grade a non-matching hit PROXY_ONLY."""

    def __new__(cls, source: str, series_id: str, title: Optional[str] = None,
                concept_status: str = "unverified", reason: str = ""):
        obj = super().__new__(cls, (source, series_id))
        obj.title = title
        obj.concept_status = concept_status
        obj.reason = reason
        return obj

    def __reduce__(self):
        return (FetchableHit, (self[0], self[1], self.title, self.concept_status, self.reason))


def fetch_series_title(source_name: str, series_id: str, collector=None,
                       agent: str = "") -> Optional[str]:
    """The source's own title for a series id (metadata only, no observations). FRED needs
    FRED_API_KEY; World Bank uses the indicator endpoint; DBnomics the series document.
    Returns None when unavailable. Never raises."""
    src = str(source_name or "").strip().lower()
    sid = str(series_id or "").strip()
    if not sid:
        return None
    try:
        if src == "fred":
            key = os.environ.get("FRED_API_KEY")
            if not key:
                return None
            data = _get_json("https://api.stlouisfed.org/fred/series",
                             params={"series_id": sid, "api_key": key, "file_type": "json"},
                             collector=collector, agent=agent)
            rows = data.get("seriess") or []
            return (rows[0].get("title") or None) if rows else None
        if src == "world bank":
            indicator = sid.rpartition(":")[2]
            data = _get_json(f"https://api.worldbank.org/v2/indicator/{indicator}",
                             params={"format": "json"}, collector=collector, agent=agent)
            rows = data[1] if isinstance(data, list) and len(data) > 1 and data[1] else []
            return (rows[0].get("name") or None) if rows else None
        if src == "dbnomics" and sid.count("/") == 2:
            data = _get_json(f"https://api.db.nomics.world/v22/series/{sid}",
                             params={"observations": "0"}, collector=collector, agent=agent)
            docs = (data.get("series") or {}).get("docs") or []
            return (docs[0].get("series_name") or None) if docs else None
    except Exception:
        return None
    return None


def resolve_fetchable_series(series_name: str, collector=None,
                             agent: str = "") -> Optional[Tuple[str, str]]:
    """General fetchable-universe resolver for the feasibility scout (infrastructure
    over per-run curation): curated precision layer first,
    then the FRED index, then WDI. Set AEL_PROVIDER_SEARCH=0 to disable the network rungs
    (unit tests / offline).

    Every candidate's OFFICIAL title is checked against the
    requested name (:func:`title_concept_check`); without this check the scout graded 'Tax-to-GDP' as feasible
    via GDP levels and 'Wealth Gini' via the income Gini. The first candidate whose title
    matches is returned; otherwise the best non-matching candidate is returned with its
    ``concept_status`` so the scout records it as a proxy, never a direct match. Offline, a
    curated hit cannot be verified and carries ``unverified``. Returns a
    :class:`FetchableHit` (a ``(source, series_id)`` tuple) or None. Never raises."""
    key = (series_name or "").strip().lower()
    if not key:
        return None
    if key in _RESOLVE_CACHE:
        return _RESOLVE_CACHE[key]
    online = os.environ.get("AEL_PROVIDER_SEARCH", "1") != "0"
    fallback: Optional[FetchableHit] = None

    def _graded(source: str, sid: str, title: Optional[str]) -> FetchableHit:
        status, reason = title_concept_check(series_name, title, series_id=sid)
        return FetchableHit(source, sid, title, status, reason)

    hit: Optional[FetchableHit] = None
    curated = lookup_known_series(series_name)
    if curated:
        title = fetch_series_title(*curated, collector=collector, agent=agent) if online else None
        cand = _graded(curated[0], curated[1], title)
        if cand.concept_status == TITLE_MATCH:
            hit = cand
        else:
            fallback = cand
    if hit is None and online:
        fred_id = search_fred_series(series_name, collector=collector, agent=agent)
        if fred_id:
            cand = _graded("FRED", fred_id, fetch_series_title("FRED", fred_id, collector, agent))
            if cand.concept_status == TITLE_MATCH:
                hit = cand
            else:
                fallback = fallback or cand
        if hit is None:
            wdi_id = search_wdi_series(series_name, collector=collector, agent=agent)
            if wdi_id:
                cand = _graded("DBnomics", wdi_id,
                               fetch_series_title("DBnomics", wdi_id, collector, agent))
                if cand.concept_status == TITLE_MATCH:
                    hit = cand
                else:
                    fallback = fallback or cand
    hit = hit or fallback
    _RESOLVE_CACHE[key] = hit
    return hit


# ---------------------------------------------------------------------------------------------
# Curated exact ids. The discovery step's "Series available" lists are LLM-proposed and
# near-miss real ids (GC.TAX.TOTL.ZS for GC.TAX.TOTL.GD.ZS; CURRALL for CURRCIR). For the most
# common indicators we KNOW the exact id — a name-keyed lookup repairs deterministically before
# any search API is consulted. Phrases are matched on word boundaries, first entry wins, so
# keep more specific phrases earlier. US-scoped by construction (FRED ids / WB US default).
#
# FROZEN BY POLICY ("we are building an infrastructure, not tuning the results" — the same
# rule that stops archetype accretion in the calibration harness). Do NOT add entries because
# a run missed a series: the GENERAL path is runtime provider-index search
# (search_fred_series / search_wdi_series / search_dbnomics_series via
# resolve_fetchable_series). This table is only a small precision-override layer for
# paraphrase staples the strict search guard cannot certify (e.g. "government spending" →
# GCE, whose FRED title says "Consumption Expenditures", not "spending").
# ---------------------------------------------------------------------------------------------

KNOWN_SERIES_IDS = [
    # (source, series_id, phrases)
    ("FRED", "CURRCIR",   ("currency in circulation",)),
    ("FRED", "CUSR0000SAH1", ("cpi shelter", "shelter")),
    ("FRED", "CPIAUCSL",  ("consumer price index", "cpi", "inflation")),
    ("FRED", "FEDFUNDS",  ("federal funds", "policy rate")),
    ("FRED", "UNRATE",    ("unemployment rate",)),
    ("FRED", "GDPC1",     ("real gross domestic product", "real gdp")),
    ("FRED", "M2SL",      ("money supply", "m2")),
    ("FRED", "M1SL",      ("m1",)),
    ("FRED", "DGS10",     ("10-year treasury", "10 year treasury")),
    ("FRED", "PCEPI",     ("pce price index",)),
    ("FRED", "PCE",       ("personal consumption expenditure",)),
    ("FRED", "PAYEMS",    ("nonfarm payroll", "total nonfarm")),
    ("FRED", "CES0500000003", ("average hourly earnings",)),
    ("FRED", "INDPRO",    ("industrial production",)),
    ("FRED", "HOUST",     ("housing starts",)),
    ("FRED", "LABSHPUSA156NRUG", ("labor share",)),
    ("World Bank", "GC.TAX.TOTL.GD.ZS", ("tax revenue",)),
    ("World Bank", "FX.OWN.TOTL.ZS", ("account ownership", "accounts at a financial institution",
                                      "mobile-money-service", "mobile money")),
    ("World Bank", "IT.NET.USER.ZS", ("internet users", "individuals using the internet")),
    ("World Bank", "SI.POV.GINI", ("gini",)),
    ("World Bank", "NY.GDP.MKTP.KD.ZG", ("gdp growth",)),
    ("World Bank", "NY.GDP.MKTP.CD", ("gross domestic product", "gdp")),
    ("World Bank", "SP.POP.TOTL", ("total population", "population")),
    # live-verified: poverty, income/wealth distribution, risk-free + real rates — near-miss
    # classes observed in full runs (SI.POV.GAP for SI.POV.GAPS; SI.DST.05TH for SI.DST.05TH.20).
    ("World Bank", "SI.POV.GAPS", ("poverty gap",)),
    ("World Bank", "SI.POV.DDAY", ("poverty headcount",)),
    ("World Bank", "SI.DST.10TH.10", ("income share held by highest 10", "top 10% income")),
    ("World Bank", "SI.DST.05TH.20", ("income share held by highest 20", "top 20% income",
                                      "highest quintile")),
    ("World Bank", "SI.DST.FRST.20", ("income share held by lowest 20", "bottom 20% income",
                                      "lowest quintile")),
    ("FRED", "TB3MS", ("3-month treasury", "treasury bill", "risk-free rate")),
    ("FRED", "REAINTRATREARAT10Y", ("real interest rate",)),
    ("FRED", "WFRBST01134", ("top 1% wealth share", "wealth share held by the top 1")),
    # live-verified: a full run once force-mapped "Household Debt Service Ratio" to HOUST (housing starts) and needed Government Spending for var:G_t.
    ("FRED", "TDSP", ("debt service ratio", "debt service payments", "household debt service")),
    ("FRED", "GCE", ("government consumption", "government spending", "government expenditure",
                     "government purchases")),
    # live-verified: LLM near-miss id PCEPCC96 once ended as disclosed simulation.
    ("FRED", "PCECC96", ("real personal consumption", "real pce", "real consumption")),
]

# Sub-aggregate qualifiers: a query naming a COMPONENT ("CPI-U Shelter") must not match
# a curated all-items id (CPIAUCSL) just because the aggregate phrase ("cpi") appears. A match
# is declined when the query contains one of these tokens and the entry's phrases don't.
_COMPONENT_QUALIFIERS = ("shelter", "food", "energy", "core", "rent", "medical",
                         "apparel", "gasoline", "durables", "nondurables")


def lookup_known_series(series_name: str):
    """Curated exact-id lookup by human series name. Returns (source, series_id) or None."""
    name = (series_name or "").lower()
    if not name:
        return None

    def _has(text: str, token: str) -> bool:
        return re.search(r"(?<![a-z0-9])" + re.escape(token) + r"(?![a-z0-9])", text) is not None

    qualifiers = [q for q in _COMPONENT_QUALIFIERS if _has(name, q)]
    for source, sid, phrases in KNOWN_SERIES_IDS:
        phrase_text = " ".join(phrases)
        if any(not _has(phrase_text, q) for q in qualifiers):
            continue  # entry doesn't cover the component the query names
        for phrase in phrases:
            if _has(name, phrase):
                return source, sid
    return None


def _fetch_fred_native(series_id: str) -> Optional[Dict[str, Any]]:
    """Minimal FRED fetch (fredapi + FRED_API_KEY) so repair candidates that land on FRED
    ids are retrievable here — the DataTeam's own FRED path lives in its stage. Never raises."""
    import os
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        return None
    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        data = fred.get_series(series_id)
        if data is None or not len(data):
            return None
        obs = [{"date": str(idx.date()), "value": val}
               for idx, val in data.dropna().items()]
        title = None
        try:
            title = str(fred.get_series_info(series_id).get("title") or "") or None
        except Exception:
            pass
        return _norm(obs, "FRED", series_title=title)
    except Exception:
        return None


def fetch_series_with_repair(
    source_name: str, series_id: str, series_name: str = "", collector=None, agent: str = "",
):
    """``fetch_series`` plus a bounded repair ladder when the id fails.

    Repairs, in order:
      (a) World Bank country-code aliases (``GER:`` -> ``DEU:``);
      (b) curated exact-id lookup by the human ``series_name`` (``KNOWN_SERIES_IDS``) —
          deterministic, covers the most common LLM id near-misses (CURRALL -> CURRCIR);
      (a2) the DBnomics WDI mirror of an exact World Bank indicator id (provider redundancy);
      (b2) FRED index search by name;
      (c) World-Bank-pinned WDI search (World Bank sources whose curated lookup missed);
      (d) generic DBnomics free-text search (top-3 datasets, geography-guarded).

    A candidate that is a DIFFERENT series than the requested
    one (rungs b, b2, c, d) is accepted only when its official title passes
    :func:`title_concept_check` against ``series_name`` AND its geography is compatible with
    the requested source/id (:func:`repair_acceptable`). The curated phrase table matched
    'gdp' inside '(% of GDP)' and stored US nominal GDP under a credit-to-GDP id; such
    candidates are now DECLINED and the requirement stays unmet (disclosed), never a different
    concept under the requested id. Rungs (a) and (a2) fetch the same indicator and need no
    concept check.

    Returns ``(normalized_dict_or_None, note)``: on success the note discloses the repair
    ("" when none was needed); on failure it lists declined candidates ("" when there were
    none)."""
    src = str(source_name or "").strip().lower()
    out = (_fetch_fred_native(series_id) if src == "fred"
           else fetch_series(source_name, series_id, collector=collector, agent=agent))
    if out:
        return out, ""

    declined: List[str] = []

    def _accept(candidate, rep_source: str, rep_id: str) -> bool:
        ok, why = repair_acceptable(series_name, source_name, series_id, candidate,
                                    rep_source, rep_id)
        if not ok and candidate:
            declined.append(why)
        return ok

    def _accept_mirror(candidate, mirror_id: str) -> bool:
        # same indicator on another host: with a requested name, the full repair acceptance
        # (title concept + geography); without one, the geography check alone
        if (series_name or "").strip():
            return _accept(candidate, "DBnomics", mirror_id)
        title = candidate.get("series_title") or ""
        g_req = implied_geography(source_name, series_id, "")
        g_rep = implied_geography("DBnomics", mirror_id, title)
        if _geo_compatible(g_req, g_rep):
            return True
        declined.append(f"candidate DBnomics/{mirror_id} ('{title}') declined — geography "
                        f"{', '.join(sorted(g_rep))} does not match requested "
                        f"{', '.join(sorted(g_req))}")
        return False

    def _declined_note() -> str:
        return ("repair declined (requirement unmet): " + " | ".join(declined)) if declined else ""

    # (a) country-code normalization (World Bank COUNTRY:INDICATOR ids)
    if src == "world bank" and ":" in (series_id or ""):
        country, _, indicator = series_id.partition(":")
        fixed = _COUNTRY_ALIASES.get(country.strip().upper())
        if fixed:
            out = fetch_series(source_name, f"{fixed}:{indicator}", collector=collector, agent=agent)
            if out:
                out["resolved_id"], out["resolved_source"] = f"{fixed}:{indicator}", source_name
                return out, f"id repaired: country code {country.strip()} -> {fixed}"

    # (b) curated exact ids by human name — concept- and geography-checked against the
    # fetched series' own title (the phrase table alone matched 'gdp' inside '% of GDP')
    known = lookup_known_series(series_name)
    if known:
        k_source, k_id = known
        if (k_source, k_id) != (source_name, series_id):
            out = (_fetch_fred_native(k_id) if k_source == "FRED"
                   else fetch_series(k_source, k_id, collector=collector, agent=agent))
            if out and _accept(out, k_source, k_id):
                out["resolved_id"], out["resolved_source"] = k_id, k_source
                out["title_check"] = TITLE_MATCH
                return out, (f"id repaired via curated table: '{series_id}' ({source_name}) "
                             f"-> {k_source}/{k_id}")

    # (a2) a TRANSIENT World Bank outage can turn known-good ids into disclosed
    # simulations. DBnomics mirrors the entire WDI catalogue — for an exact
    # World-Bank-shaped indicator id, try the mirror deterministically before any search.
    # Provider redundancy, not id repair: same indicator, same geography, different host.
    # Audit #3: the country code is normalized through the complete ISO 3166-1 table (an
    # unknown code is refused, never defaulted to USA), and the mirror's response passes the
    # same acceptance check as every other repair (title concept + geography).
    if src == "world bank":
        indicator = series_id.partition(":")[2] if ":" in (series_id or "") else series_id
        iso3, raw_country = _world_bank_iso3(series_id)
        if indicator and re.fullmatch(r"[A-Z]{2}\.[A-Z0-9.]+", indicator or ""):
            if not iso3:
                declined.append(f"DBnomics WDI mirror not tried — unknown country code "
                                f"'{raw_country}'")
            else:
                mirror_id = f"WB/WDI/A-{indicator}-{iso3}"
                out = fetch_dbnomics(mirror_id, collector=collector, agent=agent)
                if out and _accept_mirror(out, mirror_id):
                    out["resolved_id"] = mirror_id
                    out["resolved_source"] = "DBnomics"
                    if (series_name or "").strip():
                        out["title_check"] = TITLE_MATCH
                    return out, (f"provider redundancy: World Bank unavailable for "
                                 f"'{series_id}' — served from the DBnomics WDI mirror "
                                 f"({mirror_id})")

    # (b2) FRED index search by human name — the general mechanism (curated is only the
    # precision override). Relevance-guarded; disabled offline via AEL_PROVIDER_SEARCH=0.
    if os.environ.get("AEL_PROVIDER_SEARCH", "1") != "0":
        fred_hit = search_fred_series(series_name or series_id, collector=collector, agent=agent)
        if fred_hit and (str(fred_hit) != str(series_id) or src != "fred"):
            out = _fetch_fred_native(fred_hit)
            if out and _accept(out, "FRED", fred_hit):
                out["resolved_id"], out["resolved_source"] = fred_hit, "FRED"
                out["title_check"] = TITLE_MATCH
                return out, (f"id repaired via FRED index search: '{series_id}' "
                             f"({source_name}) -> FRED/{fred_hit}")

    # search geography: the requested World Bank country (ISO3), else the US default
    geo_hint = (_world_bank_iso3(series_id)[0] or "US") if src == "world bank" else "US"

    # (c) World-Bank-pinned WDI search — the generic dataset search often never lands on WDI
    if src == "world bank":
        repaired_id = search_wdi_series(series_name or series_id, collector=collector,
                                        agent=agent, geo_hint=geo_hint)
        if repaired_id:
            out = fetch_dbnomics(repaired_id, collector=collector, agent=agent)
            if out and _accept(out, "DBnomics", repaired_id):
                out["resolved_id"], out["resolved_source"] = repaired_id, "DBnomics"
                out["title_check"] = TITLE_MATCH
                return out, (f"id repaired via WDI search: '{series_id}' ({source_name}) "
                             f"-> {repaired_id}")

    # (d) generic DBnomics free-text lookup by the series' human name (works for any source);
    # geography-guarded — a wrong-country hit is rejected, not silently substituted.
    if src in _CONNECTORS or src in ("world bank", "fred"):
        repaired_id = search_dbnomics_series(series_name or series_id, collector=collector,
                                             agent=agent, geo_hint=geo_hint)
        if repaired_id:
            out = fetch_dbnomics(repaired_id, collector=collector, agent=agent)
            if out and _accept(out, "DBnomics", repaired_id):
                out["resolved_id"], out["resolved_source"] = repaired_id, "DBnomics"
                out["title_check"] = TITLE_MATCH
                return out, (f"id repaired via DBnomics search: '{series_id}' ({source_name}) "
                             f"-> {repaired_id}")
    return None, _declined_note()


# ---------------------------------------------------------------------------------------------
# Concept check: requested variable name vs the SOURCE's own series title.
#
# The earlier rule (token Jaccard >= 0.5, acronym expansion inflating the overlap, an empty
# title read as "no mismatch") passed 'Consumption-to-GDP Ratio' as 'GDP (current US$)',
# FDI 'net inflows' as 'net outflows', 'US Corporate HY OAS' as an Emerging-Markets index and
# 'Unemployment Rate Volatility' as 'Unemployment Rate'. The rule below is field-agnostic:
#   * every content word of the requested name (outside parenthetical glosses) must appear
#     in the title (after acronym expansion, British/US spelling and light stemming);
#   * construct words (volatility, uncertainty, growth/change, ratio/to-GDP/share, per capita,
#     spread, dispersion, expectation/forecast, tracker, real/nominal) must agree;
#   * direction words (inflow/outflow, import/export, asset/liability, ...) must not flip;
#   * named geographies must not conflict;
#   * a missing or uninformative title is UNVERIFIED — a distinct, disclosed status, never
#     a silent match.
# ---------------------------------------------------------------------------------------------

# common acronyms in requested names vs spelled-out source titles (and vice versa)
_TITLE_ACRONYMS = {
    "gdp": ["gross", "domestic", "product"], "gnp": ["gross", "national", "product"],
    "gni": ["gross", "national", "income"], "cpi": ["consumer", "price"],
    "ppi": ["producer", "price"], "pce": ["personal", "consumption", "expenditure"],
    "fdi": ["foreign", "direct", "investment"], "tfp": ["total", "factor", "productivity"],
    "fed": ["federal"], "govt": ["government"], "gov": ["government"],
    "hy": ["high", "yield"], "oas": ["option", "adjusted", "spread"], "vix": ["volatility"],
    "ted": ["ted", "spread"],
    # monetary aggregates: titles are often the bare code ("M2")
    "m1": ["m1", "money", "stock"], "m2": ["m2", "money", "stock"], "m3": ["m3", "money", "stock"],
}

# spelling variants and plain synonyms -> one canonical token (applied before stemming)
_TITLE_SYNONYMS = {
    "labour": "labor", "harmonised": "harmonized", "programme": "program", "centre": "center",
    "spending": "expenditure", "outlay": "expenditure", "outlays": "expenditure",
    "purchases": "expenditure", "purchase": "expenditure",
    "jobless": "unemployment", "payroll": "employee", "payrolls": "employee",
    "employees": "employee", "excluding": "less", "excl": "less",
}

# non-content words: function words, units, frequencies, and construct words that are
# compared by the construct rules below rather than as content tokens
_TITLE_STOP = {
    "the", "of", "and", "for", "in", "on", "at", "by", "to", "with", "from", "as", "or",
    "against", "versus", "vs", "between", "over", "per", "via",
    "rate", "index", "indice", "total", "level", "sery", "data", "value", "measure",
    "indicator", "annual", "annualized", "quarterly", "monthly", "weekly", "daily",
    "frequency", "percent", "percentage", "pct", "sa", "nsa", "saar", "seasonally", "adjust",
    "unit", "billion", "million", "thousand", "dollar", "usd", "lcu", "average", "mean",
    "all", "item", "proxy", "coefficient", "close", "closing", "sector",
    "current", "constant", "real", "nominal", "ratio", "share",
    "growth", "end", "period", "us", "usa", "state", "united",
}

# geography phrases -> canonical key (compared as sets; removed before content tokenizing)
_GEO_PATTERNS = [
    ("us", r"\bunited states\b|\bu\.s\.(?:a\.)?|\busa\b|\bus\b(?!\$)|\bamerican?\b"),
    ("euro_area", r"\beuro[- ]?area\b|\beuro[- ]?zone\b|\bea ?(?:1[0-9]|20)\b|\bemu\b"),
    ("eu", r"\beuropean union\b|\beu(?: ?2[78])?\b"),
    ("uk", r"\bunited kingdom\b|\buk\b|\bbritain\b|\bbritish\b|\bengland\b"),
    ("germany", r"\bgermany\b|\bgerman\b"), ("france", r"\bfrance\b|\bfrench\b"),
    ("japan", r"\bjapan(?:ese)?\b"), ("china", r"\bchina\b|\bchinese\b"),
    ("india", r"\bindia\b|\bindian\b"), ("canada", r"\bcanada\b|\bcanadian\b"),
    ("italy", r"\bitaly\b|\bitalian\b"), ("spain", r"\bspain\b|\bspanish\b"),
    ("brazil", r"\bbrazil(?:ian)?\b"), ("mexico", r"\bmexic(?:o|an)\b"),
    ("korea", r"\bkorea(?:n)?\b"), ("russia", r"\brussia(?:n)?\b"),
    ("australia", r"\baustralia(?:n)?\b"), ("switzerland", r"\bswitzerland\b|\bswiss\b"),
    ("sweden", r"\bswed(?:en|ish)\b"), ("norway", r"\bnorw(?:ay|egian)\b"),
    ("netherlands", r"\bnetherlands\b|\bdutch\b"), ("turkey", r"\bturkey\b|\bt[uü]rkiye\b"),
    ("argentina", r"\bargentin(?:a|e|ian)\b"), ("south_africa", r"\bsouth africa(?:n)?\b"),
    ("indonesia", r"\bindonesia(?:n)?\b"),
    ("emerging", r"\bemerging (?:markets?|economies|economy)\b"),
    ("advanced", r"\badvanced economies\b"),
    ("world", r"\bworld\b(?! bank)"),
]
_GEO_RES = [(k, re.compile(rx, re.IGNORECASE)) for k, rx in _GEO_PATTERNS]
# euro area and EU are nested European aggregates — treated as compatible with each other
_GEO_FAMILY = {"euro_area": "europe", "eu": "europe"}

# ISO codes (World Bank / IMF / Eurostat ids) -> canonical geography key
_ISO_GEO = {
    "US": "us", "USA": "us", "DE": "germany", "DEU": "germany", "FR": "france", "FRA": "france",
    "GB": "uk", "GBR": "uk", "UK": "uk", "JP": "japan", "JPN": "japan", "CN": "china",
    "CHN": "china", "IN": "india", "IND": "india", "CA": "canada", "CAN": "canada",
    "IT": "italy", "ITA": "italy", "ES": "spain", "ESP": "spain", "BR": "brazil",
    "BRA": "brazil", "MX": "mexico", "MEX": "mexico", "KR": "korea", "KOR": "korea",
    "RU": "russia", "RUS": "russia", "AU": "australia", "AUS": "australia",
    "CH": "switzerland", "CHE": "switzerland", "SE": "sweden", "SWE": "sweden",
    "NO": "norway", "NOR": "norway", "NL": "netherlands", "NLD": "netherlands",
    "TR": "turkey", "TUR": "turkey", "AR": "argentina", "ARG": "argentina",
    "ZA": "south_africa", "ZAF": "south_africa", "ID": "indonesia", "IDN": "indonesia",
    "EA": "euro_area", "EMU": "euro_area", "EA19": "euro_area", "EA20": "euro_area",
    "XC": "euro_area", "EU": "eu", "EU27": "eu", "EU27_2020": "eu", "EU28": "eu",
    "WLD": "world", "1W": "world",
}

# construct classes: (name, regex, two_sided). one-sided = only a REQUESTED construct the
# title lacks is a mismatch; two-sided = a title construct the request lacks is one too.
_CONSTRUCTS = [
    ("volatility", r"\bvolatilit(?:y|ies)\b|\bvariances?\b|\bstandard deviations?\b", True),
    ("uncertainty", r"\buncertainty\b", True),
    ("tracker", r"\btrackers?\b", True),
    ("growth/change", r"\bgrowth\b|(?:\bpercent|%|\bpct\.?) change\b|\bchange (?:in|from)\b|"
                      r"\binflation\b|\byoy\b|\byear[- ]over[- ]year\b|\bannual %", True),
    ("expectation/forecast", r"\bexpectations?\b|\bexpected\b|\bforecasts?\b|\bprojections?\b",
     True),
    ("spread", r"\bspreads?\b|\boption[- ]adjusted\b|\boas\b", True),
    ("dispersion", r"\bdispersion\b|\bdisagreement\b", True),
    ("per capita", r"\bper capita\b|\bper person\b|\bper head\b", True),
    ("ratio/share", r"\bratios?\b|%\s*of\b|\bpercent(?:age)? of\b|\bper ?cent of\b|\bshares?\b|"
                    r"\w-to-\w|\bto gdp\b|\bas a (?:percent|share)\b", False),
]
_CONSTRUCT_RES = [(n, re.compile(rx, re.IGNORECASE), two) for n, rx, two in _CONSTRUCTS]
# a title expressing ANY normalization satisfies a requested ratio construct
_RATIO_SATISFIED = re.compile(r"\brate\b|%|\bpercent\b|\bper\b", re.IGNORECASE)

_REAL_RE = re.compile(r"\breal\b|\bconstant (?:prices?|dollars?|\d{4}|us\$|lcu|national)|"
                      r"\bchained\b|\binflation[- ]adjusted\b|\bdeflated\b", re.IGNORECASE)
_NOMINAL_RE = re.compile(r"\bnominal\b|\bcurrent (?:prices?|dollars?|us\$|lcu|national)",
                         re.IGNORECASE)

# direction pairs: the request naming one side while the title names only the other flips
# the concept (FDI net inflows vs net outflows)
_DIRECTIONS = [
    (r"\binflows?\b", r"\boutflows?\b"),
    (r"\bimports?\b", r"\bexports?\b"),
    (r"\bassets?\b", r"\bliabilit(?:y|ies)\b"),
    (r"\blending\b", r"\bborrowing\b"),
    (r"\blong[- ]term\b", r"\bshort[- ]term\b"),
    (r"\bmale\b", r"\bfemale\b"),
]
_DIRECTION_RES = [(re.compile(a, re.IGNORECASE), re.compile(b, re.IGNORECASE))
                  for a, b in _DIRECTIONS]

_PAREN_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]")

# Identifier codes inside a requested name's parenthetical gloss ("Money stock (M3)",
# "GDP per capita, PPP", "FDI (BoP, current US$)") name WHICH measure is meant and are
# compared against the title; units, adjustment flags and publisher names are not.
_PAREN_NON_IDENTIFIERS = {
    "us", "usa", "usd", "lcu", "sa", "nsa", "saar", "yoy", "qoq", "mom", "ilo", "oecd",
    "imf", "wb", "wdi", "bis", "ecb", "fred", "bls", "bea", "cbo", "eia", "un", "eu",
}


def _paren_identifiers(text: str) -> set:
    """Lower-cased identifier codes named inside parentheses/brackets of ``text``: 2-6
    character alphanumeric tokens with two or more capitals (PPP, BoP) or a capital plus a
    digit (M3). Pure numbers, ordinary words and the non-identifiers above are skipped."""
    ids = set()
    for group in _PAREN_RE.findall(text or ""):
        for tok in re.findall(r"(?<![A-Za-z0-9$])([A-Za-z][A-Za-z0-9]{1,5})(?![A-Za-z0-9$])",
                              group):
            caps = sum(c.isupper() for c in tok)
            if caps >= 2 or (tok[0].isupper() and any(c.isdigit() for c in tok)):
                if tok.lower() not in _PAREN_NON_IDENTIFIERS and tok.upper() not in _ISO_GEO:
                    ids.add(tok.lower())
    return ids


def _identifier_in_title(ident: str, title: str) -> bool:
    words = set(re.findall(r"[a-z0-9]+", (title or "").lower()))
    if ident in words:
        return True
    expansion = _TITLE_ACRONYMS.get(ident)
    if expansion and ident not in ("m1", "m2", "m3"):    # an aggregate code must appear as is
        return all(w in words or _stem(w) in {_stem(x) for x in words} for w in expansion)
    return False


# Measurement type: a RATE (percent, rate) and a LEVEL (level, number, count, index) of the
# same variable are different series (unemployment rate vs unemployment level). Compared in
# both directions, only when each side states its type unambiguously.
_RATE_TYPE_RE = re.compile(r"\brates?\b|%|\bpercent(?:age)?\b|\bper ?cent\b", re.IGNORECASE)
_LEVEL_TYPE_RE = re.compile(r"\blevels?\b|\bnumber\b|\bcounts?\b|\bindex\b|\bindices\b|"
                            r"\bindexes\b", re.IGNORECASE)


def _measurement_type(text: str) -> str:
    r, lv = bool(_RATE_TYPE_RE.search(text or "")), bool(_LEVEL_TYPE_RE.search(text or ""))
    if r and not lv:
        return "rate"
    if lv and not r:
        return "level"
    return ""


def _stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    if len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 5 and w.endswith("ed"):
        w = w[:-2]
    return w


def geographies(text: str) -> set:
    """Canonical geography keys named in ``text`` ('US$' is a currency, not a place)."""
    return {k for k, rx in _GEO_RES if rx.search(text or "")}


def _strip_geo(text: str) -> str:
    out = text or ""
    for _, rx in _GEO_RES:
        out = rx.sub(" ", out)
    return out


def _content_tokens(text: str) -> set:
    out = set()
    for w in re.findall(r"[a-z0-9]+", _strip_geo(text).lower()):
        for part in _TITLE_ACRONYMS.get(w, [w]):
            part = _TITLE_SYNONYMS.get(part, part)
            part = _TITLE_SYNONYMS.get(_stem(part), _stem(part))
            if part in _TITLE_STOP:
                continue
            if len(part) >= 3 or any(c.isdigit() for c in part):
                out.add(part)
    return out


_ACRONYM_RE = re.compile(r"\b(" + "|".join(_TITLE_ACRONYMS) + r")\b", re.IGNORECASE)


def _expand_acronyms(text: str) -> str:
    return _ACRONYM_RE.sub(lambda m: " ".join(_TITLE_ACRONYMS[m.group(1).lower()]), text or "")


def _geo_compatible(a: set, b: set) -> bool:
    if not a or not b:
        return True
    fa = a | {_GEO_FAMILY[g] for g in a if g in _GEO_FAMILY}
    fb = b | {_GEO_FAMILY[g] for g in b if g in _GEO_FAMILY}
    return bool(fa & fb)


TITLE_MATCH, TITLE_MISMATCH, TITLE_UNVERIFIED = "match", "mismatch", "unverified"


_SATISFIED_BY = {"yield": {"yield", "maturity", "rate"}, "rate": {"rate", "yield", "spread"}}
# an "Average Loan-to-Value Ratio" request matched "...Average Interest Rate at Origination by LTV"
_INTEREST_RATE_TITLE = re.compile(r"\binterest rates?\b|\brates? at origination\b", re.IGNORECASE)
_RATE_WORDS = re.compile(r"\brates?\b|\byields?\b|\binterest\b|\bspreads?\b|\bcoupon\b",
                         re.IGNORECASE)


def title_concept_check(requested: str, source_title: Optional[str],
                        series_id: str = "") -> Tuple[str, str]:
    """Compare a requested variable name with the source's own series title.

    Returns ``(status, reason)`` with status one of ``"match"``, ``"mismatch"`` (the title
    names a different concept: present it only as a disclosed proxy) or ``"unverified"``
    (no informative title to compare against — disclosed, never a silent match)."""
    req = (requested or "").strip()
    title = (source_title or "").strip()
    if not req:
        return TITLE_UNVERIFIED, "no requested name to compare"
    if not title or (series_id and title.lower() == str(series_id).strip().lower()):
        return TITLE_UNVERIFIED, "the source supplied no informative series title"
    have = _content_tokens(title)
    if not have:
        return TITLE_UNVERIFIED, f"the source title '{title}' carries no content words"
    want = _content_tokens(_PAREN_RE.sub(" ", req)) or _content_tokens(req)
    idents = _paren_identifiers(req)
    m_req, m_title = _measurement_type(req), _measurement_type(title)
    raw_title = title
    # construct/direction rules read acronym-expanded text ('HY OAS' names a spread)
    req, title = _expand_acronyms(req), _expand_acronyms(title)

    reasons: List[str] = []
    # interest-rate measures share one vocabulary: a Treasury 'constant maturity' series is a
    # yield, and a 'TED rate' is the TED spread (otherwise T10Y2Y and TEDRATE are misflagged)
    missing = sorted(m for m in want - have if not (_SATISFIED_BY.get(m, set()) & have))
    if missing:
        reasons.append(f"title lacks {', '.join(missing)}")
    # identifier codes in the request's parenthetical ('Money stock (M3)' vs 'M2 Money Stock')
    lacking_ids = sorted(i for i in idents if not _identifier_in_title(i, raw_title))
    if lacking_ids:
        reasons.append(f"title lacks the identifier {', '.join(x.upper() for x in lacking_ids)}")
    if m_req and m_title and m_req != m_title:
        reasons.append(f"measurement type differs (requested a {m_req}, the title is a "
                       f"{m_title})")
    for name, rx, two_sided in _CONSTRUCT_RES:
        r_has, t_has = bool(rx.search(req)), bool(rx.search(title))
        if name == "ratio/share":
            # two-sided: a normalized title (ratio, %, per, rate) satisfies a requested ratio,
            # and a ratio title needs a request that names some normalization
            if r_has and not t_has and not _RATIO_SATISFIED.search(title):
                reasons.append("requested a ratio/share; the title is not one")
            elif t_has and not r_has and not _RATIO_SATISFIED.search(req):
                reasons.append("the title is a ratio/share; the request is not one")
            continue
        if r_has and not t_has:
            reasons.append(f"requested construct '{name}' is absent from the title")
        elif two_sided and t_has and not r_has:
            reasons.append(f"the title is a '{name}' construct the request does not name")
    if _INTEREST_RATE_TITLE.search(title) and not _RATE_WORDS.search(req):
        reasons.append("the title is an interest rate; the request is not a rate")
    if _REAL_RE.search(req) and not _REAL_RE.search(title):
        reasons.append("requested real (deflated) values; the title is not real")
    elif _NOMINAL_RE.search(req) and _REAL_RE.search(title) and not _NOMINAL_RE.search(title):
        reasons.append("requested nominal values; the title is real")
    for a, b in _DIRECTION_RES:
        for x, y in ((a, b), (b, a)):
            if x.search(req) and not y.search(req) and y.search(title) and not x.search(title):
                reasons.append("direction flipped (the title names the opposite flow/side)")
    g_req, g_title = geographies(req), geographies(title)
    if not _geo_compatible(g_req, g_title):
        reasons.append(f"geography differs ({', '.join(sorted(g_req))} requested, title "
                       f"{', '.join(sorted(g_title))})")
    if reasons:
        return TITLE_MISMATCH, "; ".join(reasons)
    return TITLE_MATCH, ""


def title_proxy_mismatch(requested: str, source_title: Optional[str]) -> bool:
    """True when the source's own series title names a different concept than the requested
    variable (e.g. World Bank SI.POV.GINI, titled "Gini index", was presented as "Gini
    Coefficient of Consumption"). An absent title is NOT a mismatch here — it is the distinct
    ``unverified`` status of :func:`title_concept_check`, which callers must disclose."""
    return title_concept_check(requested, source_title)[0] == TITLE_MISMATCH


# ---------------------------------------------------------------------------------------------
# requested-vs-repaired geography (a euro-area M3 request must never be "repaired" to a
# US aggregate). A name's own geography wins; else the provider/id implies one; else unknown.
# ---------------------------------------------------------------------------------------------

_US_PROVIDERS = {"fred", "bls"}


def implied_geography(source_name: str, series_id: str = "", name_or_title: str = "") -> set:
    """Geography keys implied by a series' name/title, else by its provider and id."""
    g = geographies(name_or_title)
    if g:
        return g
    src = str(source_name or "").strip().lower()
    sid = str(series_id or "").strip()
    if src in _US_PROVIDERS:
        return {"us"}
    if src == "ecb":
        return {"euro_area"}
    if src == "eurostat":
        m = re.search(r"\bgeo=([A-Za-z0-9_]+)", sid)
        code = (m.group(1).upper() if m else "")
        return {_ISO_GEO.get(code, code.lower())} if code else {"eu"}
    if src == "world bank":
        iso3, raw = _world_bank_iso3(sid)
        if not iso3:
            return {raw.lower()} if raw else set()
        return {_ISO_GEO.get(iso3, iso3.lower())}
    if src == "dbnomics":
        # WDI mirror / WDI search ids end in the ISO3 country: WB/WDI/A-<indicator>-<ISO3>
        m = re.fullmatch(r"WB/WDI/[A-Z]-[A-Z0-9.]+-([A-Z0-9]{3})", sid)
        if m:
            code = m.group(1)
            return {_ISO_GEO.get(code, code.lower())}
    if src in ("dbnomics", "oecd", "imf", "bis"):
        for code in re.findall(r"(?<![A-Za-z0-9])([A-Z]{2,3}|EA\d\d|EU\d\d)(?![A-Za-z0-9])", sid):
            if code in _ISO_GEO:
                return {_ISO_GEO[code]}
    return set()


def repair_acceptable(requested_name: str, requested_source: str, requested_id: str,
                      out: Optional[Dict[str, Any]], repaired_source: str,
                      repaired_id: str) -> Tuple[bool, str]:
    """A repaired series may be stored under the requested id ONLY when its official
    title passes the concept check against the requested name AND its geography is
    compatible with the requested one. Returns (ok, reason-if-declined)."""
    if not out:
        return False, "no observations"
    requested = (requested_name or "").strip() or str(requested_id or "")
    title = out.get("series_title")
    status, reason = title_concept_check(requested, title, series_id=repaired_id)
    if status != TITLE_MATCH:
        return False, (f"candidate {repaired_source}/{repaired_id} ('{title or 'no title'}') "
                       f"declined — concept check {status}: {reason or status}")
    g_req = implied_geography(requested_source, requested_id, requested_name)
    g_rep = implied_geography(repaired_source, repaired_id, title or "")
    if not _geo_compatible(g_req, g_rep):
        return False, (f"candidate {repaired_source}/{repaired_id} ('{title}') declined — "
                       f"geography {', '.join(sorted(g_rep))} does not match requested "
                       f"{', '.join(sorted(g_req))}")
    return True, ""
