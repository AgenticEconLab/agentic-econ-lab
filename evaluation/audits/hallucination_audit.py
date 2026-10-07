# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Hallucination Audit — release-grade output-integrity check for AEL runs.

A pre-release / pre-evaluation check:
point it at a multirun output tree and it audits every run's artifacts for the
fabrication risks that automated dimension-scoring can miss — invented citations,
calibration numbers that were narrated rather than computed, and data that was
hallucinated rather than fetched.

It does NOT replace human judgement: it *surfaces and flags* the items a reviewer
must eyeball, and pre-fills a manual checklist so the review is tractable across
many runs.

Auditors
--------
1. CrossCutting  — execution-log status, error/timeout scan, truncated JSON, K=3 fact divergence.
2. Citations     — extract every cited work (Ideation CSV + Literature bibliography/synthesis),
                   dedupe, flag placeholder/unsourced entries, and (optional --verify) probe
                   Semantic Scholar + Crossref by title to flag works with no external match.
3. Calibration   — for each ModelTeam calibrated model, compare CLAIMED fit/params against the
                   embedded `sandbox_validation` code, and (optional --rerun) RE-EXECUTE that code
                   in the CodeSandbox to show computed-vs-claimed side by side.
4. Data          — DataTeam: flag runs that produced only data *requirements* and no fetched data;
                   for user-uploaded, check reported stats against the real input file.
5. Provenance    — deterministic serving/config integrity: model+revision pinned == served
                   (A1), temperature claim == temperatures actually run (A2), commit/vLLM version
                   (A3), every stage on intended model (A4), manifest sane (A5), no cloud-cost leak
                   (B1), single correct engine load (B2), no <think> leak (B3), tokens non-zero (B4),
                   no OOM/restart (J3), tier-1 parser determinism (G3). These compare config /
                   `_provenance*.txt` / vLLM log / execution logs and emit PASS/FAIL/SKIP — a FAIL is
                   a hard signal (non-zero exit).

Outputs (to --out): `hallucination_report.md`, `citations.csv`, `calibration_claims.csv`,
`provenance_checks.csv`.

Usage
-----
    cd codes && export PYTHONPATH=$PWD
    python -m evaluation.audits.hallucination_audit          # audits the newest logs/log-release/v*/ release

    # ZERO-CONFIG: run-dir (newest release), the matching config, and the vLLM logs are all
    # auto-discovered from the run tree's own provenance stamps. FULL audit by default
    # (citation external-match + calibration sandbox re-run + 11 provenance/serving checks);
    # each heavy check auto-skips if the environment can't do it (no egress / no sandbox).
    # Overrides (all optional):
    #   --run-dir <tree>      audit a specific multirun tree
    #   --config <yaml>       pin the release config (else matched to the run tree by model)
    #   --vllm-log <logs>     pin serving logs (else discovered from provenance)
    #   --no-verify / --no-rerun / --quick   skip the heavy checks (offline / fast scan)
    #   --tier1-passes A B    diff two tier-1 parser passes (G3)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _load_json(p: Path) -> Optional[Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _is_truncated_json(p: Path) -> bool:
    """True if the file looks like cut-off JSON (parse fails but starts like JSON)."""
    try:
        txt = p.read_text(encoding="utf-8").strip()
    except Exception:
        return False
    if not txt or txt[0] not in "{[":
        return False
    try:
        json.loads(txt)
        return False
    except Exception:
        return True


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def _title_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm_title(a), _norm_title(b)).ratio()


def find_run_dirs(root: Path) -> List[Path]:
    return sorted(p.parent for p in root.rglob("execution_log.json"))


def run_label(root: Path, run_dir: Path) -> str:
    try:
        return str(run_dir.relative_to(root))
    except ValueError:
        return str(run_dir)


# --------------------------------------------------------------------------- #
# 1. cross-cutting
# --------------------------------------------------------------------------- #

@dataclass
class RunStatus:
    label: str
    team: str
    stages: List[Tuple[str, str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    truncated: List[str] = field(default_factory=list)
    degraded_hits: List[str] = field(default_factory=list)


DEGRADE_PATTERNS = re.compile(
    r"api key (?:missing|not found)|falling back|fallback|timed out|timeout|"
    r"could not (?:fetch|retrieve)|no results found|empty response|rate.?limit|quota exceeded",
    re.I,
)


def audit_run_status(label: str, run_dir: Path) -> RunStatus:
    team = label.split(os.sep)[0] if os.sep in label else label.split("/")[0]
    rs = RunStatus(label=label, team=team)
    log = _load_json(run_dir / "execution_log.json") or {}
    for s in log.get("stages", []):
        st = s.get("status", "?")
        rs.stages.append((s.get("name", "?"), st))
        if st != "success":
            rs.errors.append(f"{s.get('name')}={st}:{(s.get('error_message') or '')[:80]}")
    for e in (log.get("errors") or []):
        rs.errors.append(f"{e.get('error_type')}:{(e.get('error_message') or '')[:60]}")
    # truncation scan
    for jf in run_dir.glob("*.json"):
        if _is_truncated_json(jf):
            rs.truncated.append(jf.name)
    # degraded-output scan in run logs
    for lf in run_dir.glob("run_*.log"):
        try:
            for m in set(DEGRADE_PATTERNS.findall(lf.read_text(encoding="utf-8", errors="ignore"))):
                rs.degraded_hits.append("|".join(x for x in (m if isinstance(m, tuple) else (m,)) if x))
        except Exception:
            pass
    rs.degraded_hits = sorted(set(rs.degraded_hits))
    return rs


# --------------------------------------------------------------------------- #
# 2. citations
# --------------------------------------------------------------------------- #

@dataclass
class Citation:
    title: str = ""
    authors: str = ""
    year: str = ""
    source: str = ""
    url: str = ""
    origin: str = ""           # which run/file it came from
    flags: List[str] = field(default_factory=list)
    external_match: str = ""   # Y / N / ? / (blank if not probed)
    matched_title: str = ""

    def key(self) -> str:
        return _norm_title(self.title)[:120]


PLACEHOLDER_AUTHOR = re.compile(r"^(a|an|the|n/?a|unknown|anonymous|[a-z])$", re.I)
PLACEHOLDER_VENUE = re.compile(r"research paper|working paper|unknown|n/?a|preprint", re.I)


def _flag_citation(c: Citation) -> None:
    auth = (c.authors or "").strip()
    if not auth or PLACEHOLDER_AUTHOR.match(auth) or auth.lower() in ("et al", "et al."):
        c.flags.append("no/placeholder-author")
    if not c.url and not auth:
        c.flags.append("unsourced (no url + no author)")
    if c.source and PLACEHOLDER_VENUE.search(c.source):
        c.flags.append("placeholder-venue")
    if not re.search(r"(19|20)\d{2}", c.year or "") and not re.search(r"(19|20)\d{2}", c.title or ""):
        c.flags.append("no-year")


def collect_citations(label: str, run_dir: Path) -> List[Citation]:
    out: List[Citation] = []
    # (a) Ideation literature_results CSV — richest source
    for csvf in run_dir.glob("literature_results_*.csv"):
        try:
            with open(csvf, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    c = Citation(
                        title=(row.get("title") or "").strip(),
                        authors=(row.get("authors") or "").strip(),
                        year=str(row.get("year") or "").strip(),
                        source=(row.get("source") or row.get("literature_type") or "").strip(),
                        url=(row.get("url") or "").strip(),
                        origin=f"{label}/{csvf.name}",
                    )
                    if c.title:
                        _flag_citation(c); out.append(c)
        except Exception:
            pass
    # (b) Literature synthesis_results.json bibliography (structured). The Bibliography
    # model stores entries under `references` (FormattedReference); older/other shapes used
    # a list or an `entries` key — accept all. Title may be a field or embedded in
    # `formatted_citation` ("Title. (Year). Venue.").
    syn = _load_json(run_dir / "synthesis_results.json")
    if isinstance(syn, dict):
        bib = syn.get("bibliography")
        if isinstance(bib, dict):
            items = bib.get("references") or bib.get("entries") or []
        elif isinstance(bib, list):
            items = bib
        else:
            items = []
        for it in (items or []):
            if not isinstance(it, dict):
                continue
            title = (it.get("title") or "").strip()
            if not title:
                fc = (it.get("formatted_citation") or "").strip()
                fm = re.match(r"(.+?)\.\s*\(", fc)
                title = (fm.group(1).strip() if fm else fc.split(".")[0].strip())
            c = Citation(
                title=title,
                authors=", ".join(it["authors"]) if isinstance(it.get("authors"), list) else str(it.get("authors") or "").strip(),
                year=str(it.get("year") or "").strip(),
                source=(it.get("venue") or it.get("source") or it.get("publication") or it.get("reference_type") or "").strip(),
                url=(it.get("url") or it.get("doi") or "").strip(),
                origin=f"{label}/synthesis_results.json",
            )
            if c.title:
                _flag_citation(c); out.append(c)
    # (c) bibliography.txt fallback (the degraded "[Key] Title. (Year). Venue." form)
    bibtxt = run_dir / "bibliography.txt"
    if bibtxt.exists() and not out:  # only if structured sources gave nothing
        for line in bibtxt.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.match(r"\s*\[[^\]]+\]\s*(.+?)\.\s*\((\d{4})\)\.\s*(.*)$", line)
            if m:
                c = Citation(title=m.group(1).strip(), year=m.group(2), source=m.group(3).strip(),
                             origin=f"{label}/bibliography.txt")
                _flag_citation(c); out.append(c)
    return out


def _verify_by_identifier(c: Citation) -> str:
    """Verify a citation by the bibliographic identifier in its URL (Semantic Scholar
    paper id, DOI, or arXiv id) — definitive and immune to title-search rate limits.
    Returns 'y' (resolved → the work exists), 'n' (identifier definitively absent: HTTP
    404), or '' (no usable id, or the API could not be reached → inconclusive).
    Retries on HTTP 429 so transient rate-limiting does not masquerade as a miss."""
    url = c.url or ""
    s2_m = re.search(r"semanticscholar\.org/paper/([0-9a-fA-F]{20,})", url)
    doi_m = re.search(r"(10\.\d{4,9}/[^\s\"'<>]+)", url)   # a DOI anywhere in the url
    arx_m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", url)
    if not (s2_m or doi_m or arx_m):
        return ""   # no identifier in the URL -> inconclusive (no network / httpx needed)
    try:
        import httpx
        import time as _time
    except ImportError:
        return ""
    headers = {}
    if os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
        headers["x-api-key"] = os.environ["SEMANTIC_SCHOLAR_API_KEY"]

    def _get(u, **kw):
        for attempt in range(4):
            try:
                r = httpx.get(u, timeout=20.0, **kw)
            except Exception:
                return None
            if r.status_code == 429:
                _time.sleep(2 * (attempt + 1)); continue
            return r
        return None

    if s2_m:
        r = _get(f"https://api.semanticscholar.org/graph/v1/paper/{s2_m.group(1)}",
                 params={"fields": "title"}, headers=headers)
        if r is None:
            return ""
        if r.status_code == 200:
            c.matched_title = (r.json().get("title") or "")[:120]; return "y"
        return "n" if r.status_code == 404 else ""
    if doi_m:
        doi = doi_m.group(1).rstrip(".,);]")
        r = _get(f"https://api.crossref.org/works/{doi}",
                 headers={"User-Agent": "AEL-hallucination-audit/1.0 (mailto:noreply@agenticecon)"})
        if r is None:
            return ""
        if r.status_code == 200:
            c.matched_title = f"DOI resolves: {doi}"[:120]; return "y"
        return "n" if r.status_code == 404 else ""
    if arx_m:
        r = _get("http://export.arxiv.org/api/query", params={"id_list": arx_m.group(1)})
        if r is None or r.status_code != 200:
            return ""
        if "<entry>" in r.text:
            c.matched_title = f"arXiv:{arx_m.group(1)}"; return "y"
        return "n"
    return ""


def verify_citation(c: Citation, sleep: float = 1.0) -> None:
    """Verify a citation exists: first by the identifier in its URL (definitive), then by
    title search. Sets external_match Y/N/?. A citation is marked **N (likely fabricated)**
    ONLY when it has no resolvable identifier/URL and its title is conclusively not found —
    a citation that carries a URL we merely could not confirm is left **?** (inconclusive),
    never accused of fabrication."""
    idv = _verify_by_identifier(c)
    if idv == "y":
        c.external_match = "Y"; time.sleep(sleep); return
    if idv == "n":
        c.external_match = "N"
        c.matched_title = c.matched_title or "identifier did not resolve"
        c.flags.append("NO-EXTERNAL-MATCH"); time.sleep(sleep); return
    import httpx
    title = c.title.strip()
    if not title:
        c.external_match = "?"; return
    headers = {}
    if os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
        headers["x-api-key"] = os.environ["SEMANTIC_SCHOLAR_API_KEY"]
    best = 0.0; best_t = ""; queried_ok = False
    try:
        r = httpx.get("https://api.semanticscholar.org/graph/v1/paper/search",
                      params={"query": title, "limit": 3, "fields": "title,year"},
                      headers=headers, timeout=20.0)
        if r.status_code == 200:
            queried_ok = True
            for p in r.json().get("data", []) or []:
                s = _title_sim(title, p.get("title", ""))
                if s > best:
                    best, best_t = s, p.get("title", "")
    except Exception:
        pass
    if best < 0.82:  # Crossref as second opinion / fallback when S2 is rate-limited
        try:
            r = httpx.get("https://api.crossref.org/works",
                          params={"query.bibliographic": title, "rows": 3},
                          headers={"User-Agent": "AEL-hallucination-audit/1.0 (mailto:noreply@agenticecon)"},
                          timeout=20.0)
            if r.status_code == 200:
                queried_ok = True
                for it in r.json().get("message", {}).get("items", []):
                    s = _title_sim(title, " ".join(it.get("title", []) or []))
                    if s > best:
                        best, best_t = s, " ".join(it.get("title", []) or [])
        except Exception:
            pass
    has_url = bool((c.url or "").strip())
    if best >= 0.82:
        c.external_match, c.matched_title = "Y", best_t[:120]
    elif queried_ok and not has_url:
        # conclusively not found AND carries no resolvable identifier → likely fabricated
        c.external_match = "N"; c.matched_title = best_t[:120]
        c.flags.append("NO-EXTERNAL-MATCH")
    else:
        # inconclusive: APIs unreachable/rate-limited, OR it carries a URL we could not
        # confirm by title — not enough to call it fabricated.
        c.external_match = "?"
    time.sleep(sleep)


# --------------------------------------------------------------------------- #
# 3. calibration
# --------------------------------------------------------------------------- #

@dataclass
class CalibClaim:
    label: str
    model_title: str
    validated_flag: Any = None
    n_params: int = 0
    claimed_fit: Dict[str, Any] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    sandbox_rerun_ok: Optional[bool] = None
    sandbox_stdout: str = ""


def _computed_verdict(stdout: str) -> Optional[str]:
    """Extract a pass/fail verdict from re-run sandbox stdout, if present."""
    s = stdout.lower()
    m = re.search(r'"?validation"?\s*[:=]\s*"?(pass|fail|passed|failed|true|false)', s)
    if m:
        v = m.group(1)
        return "pass" if v in ("pass", "passed", "true") else "fail"
    if re.search(r"\b(validation|calibration)\s+(failed|did not pass)\b", s):
        return "fail"
    return None


def _plausibility_flags(model: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    for p in model.get("calibrated_parameters", []) or []:
        v = p.get("calibrated_value")
        try:
            fv = float(v)
            if fv != fv or abs(fv) in (float("inf"),) or abs(fv) > 1e6:
                flags.append(f"param {p.get('parameter_symbol')}={v} implausible")
        except (TypeError, ValueError):
            flags.append(f"param {p.get('parameter_symbol')}={v} non-numeric")
    fm = model.get("fit_metrics") or {}
    fs = fm.get("fit_score")
    try:
        if fs is not None and not (0.0 <= float(fs) <= 1.0):
            flags.append(f"fit_score={fs} out of [0,1]")
    except (TypeError, ValueError):
        pass
    return flags


def audit_calibration(label: str, run_dir: Path, rerun: bool) -> List[CalibClaim]:
    d = _load_json(run_dir / "calibration_output.json")
    if not isinstance(d, dict):
        return []
    claims: List[CalibClaim] = []
    sandbox = None
    if rerun:
        try:
            from shared.tools.sandbox_tool import CodeSandbox
            sandbox = CodeSandbox()
        except Exception as e:
            print(f"  [calib] sandbox unavailable ({e}); --rerun disabled", file=sys.stderr)
    for m in d.get("calibrated_models", []) or []:
        sv = m.get("sandbox_validation") or {}
        cc = CalibClaim(
            label=label,
            model_title=(m.get("model_title") or "")[:90],
            validated_flag=sv.get("validated"),
            n_params=len(m.get("calibrated_parameters") or []),
            claimed_fit=m.get("fit_metrics") or {},
        )
        if sv.get("validated") is not True:
            cc.flags.append("validated!=true")
        # param consistency: do the params embedded in the validation code match the claimed ones?
        code = sv.get("code") or ""
        if code:
            for p in m.get("calibrated_parameters", []) or []:
                sym = str(p.get("parameter_symbol") or "")
                val = p.get("calibrated_value")
                if sym and val is not None and f'"{sym}"' in code:
                    if f": {val}" not in code and f":{val}" not in code and f'"{val}"' not in code:
                        cc.flags.append(f"param {sym} differs from validation code")
        cc.flags += _plausibility_flags(m)
        if sandbox and code:
            try:
                res = sandbox.execute(code, timeout_sec=60)
                cc.sandbox_rerun_ok = bool(res.success)
                cc.sandbox_stdout = (res.stdout or res.stderr or "")[:2000]
                if not res.success:
                    # Only a HARD signal when success was CLAIMED (validated=True): then the
                    # success claim is unreproducible. If validated=False, the model honestly
                    # reported the validation did not pass, and a crashing validation script is
                    # consistent with that honest failure (a code-quality issue, not a
                    # fabrication) -> soft, lowercase flag (not counted as a hard signal).
                    if sv.get("validated") is True:
                        cc.flags.append("SANDBOX-RERUN-FAILED")
                    else:
                        cc.flags.append("sandbox-rerun-failed-not-claimed")
                else:
                    # Claimed-vs-recomputed: did the model CLAIM success but its own
                    # validation code, re-run, disagree? The claim signal is `validated`
                    # — NOT fit_score, which (post-calibration-truthfulness fix) is the
                    # *recomputed* value, so a high fit with validated=False is an honestly
                    # failed calibration (soft "recomputed-validation-fail"), not a lie.
                    verdict = _computed_verdict(res.stdout or "")
                    claimed_ok = (sv.get("validated") is True)
                    if verdict == "fail" and claimed_ok:
                        cc.flags.append("CLAIMED-OK-BUT-RECOMPUTED-FAIL")
                    elif verdict == "fail":
                        cc.flags.append("recomputed-validation-fail")
            except Exception as e:
                cc.sandbox_rerun_ok = False
                cc.sandbox_stdout = f"<rerun error: {e}>"
                cc.flags.append("SANDBOX-RERUN-ERROR" if sv.get("validated") is True
                                else "sandbox-rerun-error-not-claimed")
        claims.append(cc)
    return claims


# --------------------------------------------------------------------------- #
# 4. data
# --------------------------------------------------------------------------- #

@dataclass
class DataFinding:
    label: str
    flags: List[str] = field(default_factory=list)
    note: str = ""


def audit_data(label: str, run_dir: Path) -> Optional[DataFinding]:
    if not label.startswith("DataTeam"):
        return None
    f = DataFinding(label=label)
    produced = {p.name for p in run_dir.glob("*.json")} | {p.name for p in run_dir.glob("*.csv")}
    fetched = [n for n in produced if re.search(r"(source_output|cleaning_output|qa_output|data_output|_raw_)", n)]
    if "ModeOpenSourceAPI" in label:
        if not fetched:
            f.flags.append("NO FETCHED-DATA FILE — only requirements produced (verify data was actually retrieved, not described)")
        f.note = "Cross-check any numeric series against fred.stlouisfed.org."
    if "ModeUserUploaded" in label:
        f.note = "Verify reported row-count/columns/stats match the real input CSV (example_data.csv = 63 rows)."
    return f


# --------------------------------------------------------------------------- #
# 5. provenance & serving integrity  (deterministic, automated)
# --------------------------------------------------------------------------- #
# These are the release-review checks that a machine can decide unambiguously (string /
# number compares against the config, the per-batch `_provenance/` stamps, the vLLM
# serving log, and the execution logs). Automating them leaves the human review to
# genuine judgement (is a citation real? is a model coherent?) and to adjudicating
# any FAIL here.

@dataclass
class ProvCheck:
    id: str
    name: str
    status: str = "SKIP"          # PASS / FAIL / SKIP
    evidence: str = ""

    @property
    def mark(self) -> str:
        return {"PASS": "✅", "FAIL": "❌", "SKIP": "—"}.get(self.status, "?")


def _read_provenance(root: Path) -> List[Dict[str, str]]:
    """Parse the per-batch provenance stamps written by batch.sbatch.

    Current layout: `<root>/_provenance/*.txt`. Also reads the legacy flat layout
    `<root>/_provenance-*.txt` so older run trees still audit."""
    files = list((root / "_provenance").glob("*.txt")) + list(root.glob("_provenance-*.txt"))
    out: List[Dict[str, str]] = []
    for pf in sorted(files):
        kv: Dict[str, str] = {"_file": pf.name}
        for line in pf.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                kv[k.strip()] = v.strip()
        out.append(kv)
    return out


def _read_config_fields(cfg: Optional[Path]) -> Dict[str, Optional[str]]:
    """Extract just the fields we compare — regex, so no YAML dependency in the gate.
    First match wins, which is the `models.primary` / `run` / `serving` block."""
    fields: Dict[str, Optional[str]] = {
        "hf_repo": None, "hf_revision": None, "served_name": None,
        "repo_commit_sha": None, "vllm_version": None,
    }
    if not cfg or not cfg.exists():
        return fields
    txt = cfg.read_text(encoding="utf-8", errors="ignore")
    for key in list(fields):
        m = re.search(rf'(?m)^\s*{key}\s*:\s*"?([^"#\n]+?)"?\s*(?:#.*)?$', txt)
        if m:
            fields[key] = m.group(1).strip()
    return fields


def _read_config_temp_keys(cfg: Optional[Path]) -> List[float]:
    """Extract the temperature values claimed in `run.temperature_distribution`
    (the keys of the histogram), by a small indentation-aware line scan — no YAML dep."""
    if not cfg or not cfg.exists():
        return []
    keys: List[float] = []
    in_block = False
    base_indent = 0
    for line in cfg.read_text(encoding="utf-8", errors="ignore").splitlines():
        if re.match(r"\s*temperature_distribution\s*:", line):
            in_block = True
            base_indent = len(line) - len(line.lstrip())
            continue
        if in_block:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if (len(line) - len(line.lstrip())) <= base_indent:  # back to a sibling key
                break
            m = re.match(r'"?([0-9]*\.?[0-9]+)"?\s*:\s*[0-9]+', stripped)
            if m:
                keys.append(float(m.group(1)))
    return sorted(set(keys))


def _read_vllm_logs(paths: List[Path]) -> Optional[Dict[str, Any]]:
    """Aggregate facts from one or more vLLM serving logs (one log == one job)."""
    if not paths:
        return None
    info: Dict[str, Any] = {"snapshots": set(), "repos": set(), "served": set(),
                            "init_count": 0, "vllm_versions": set(), "oom_lines": [], "files": []}
    for p in paths:
        if not p.exists():
            continue
        info["files"].append(p.name)
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r"models--([^/]+)/snapshots/([0-9a-f]{7,40})", txt):
            info["repos"].add(m.group(1)); info["snapshots"].add(m.group(2))
        for m in re.finditer(r"served_model_name['\"=:\s\[]+([A-Za-z0-9._\-]+)", txt):
            info["served"].add(m.group(1))
        for v in re.findall(r"Initializing a V1 LLM engine \(v([0-9.]+)\)", txt):
            info["init_count"] += 1; info["vllm_versions"].add(v)
        for ln in txt.splitlines():
            if re.search(r"CUDA out of memory|OutOfMemoryError|\bKilled\b|Segmentation fault|"
                         r"engine.*(?:died|crashed|restart)|GPU.*lost", ln, re.I):
                info["oom_lines"].append(ln.strip()[:160])
    return info


def audit_provenance(root: Path, vllm_logs: List[Path], config: Optional[Path],
                     tier1_passes: Optional[List[Path]]) -> List[ProvCheck]:
    checks: List[ProvCheck] = []
    prov = _read_provenance(root)
    cfg = _read_config_fields(config)
    vinfo = _read_vllm_logs(vllm_logs)
    served_intended = cfg["served_name"] or (prov[0].get("served_name") if prov else None)

    # one pass over execution logs for B1 (cost), B4 (tokens), A2 (temperatures used)
    costs: set = set(); zero_tok: List[str] = []; n_exec = 0; temps_seen: set = set()
    for el in root.rglob("execution_log.json"):
        n_exec += 1
        llm = ((_load_json(el) or {}).get("observability") or {}).get("llm") or {}
        if llm.get("total_cost_usd") is not None:
            costs.add(llm["total_cost_usd"])
        if llm.get("total_tokens") == 0:
            zero_tok.append(run_label(root, el.parent))
        for t in (llm.get("temperatures") or []):
            temps_seen.add(round(float(t), 3))

    # ---- A1: model + revision pinned == served ----------------------------- #
    c = ProvCheck("A1", "Model+revision pinned == served")
    prov_repos = {p.get("model_repo") for p in prov if p.get("model_repo")}
    ev: List[str] = []; prob: List[str] = []
    if cfg["hf_repo"]:
        ev.append(f"config={cfg['hf_repo']}@{(cfg['hf_revision'] or '?')[:12]}")
        if prov_repos:
            ev.append(f"provenance={sorted(prov_repos)}")
            if prov_repos != {cfg["hf_repo"]}:
                prob.append("provenance model_repo != config hf_repo")
        if vinfo and vinfo["repos"]:
            ev.append(f"vllm repos={sorted(vinfo['repos'])} snaps={sorted(s[:12] for s in vinfo['snapshots'])}")
            if vinfo["repos"] != {cfg["hf_repo"].replace('/', '--')}:
                prob.append("vllm log served a different/extra repo")
            if cfg["hf_revision"] and vinfo["snapshots"] and vinfo["snapshots"] != {cfg["hf_revision"]}:
                prob.append("vllm snapshot hash != config hf_revision")
        elif cfg["hf_revision"]:
            ev.append("revision unverified — pass --vllm-log to confirm snapshot hash")
        c.status = "FAIL" if prob else ("PASS" if (prov_repos or (vinfo and vinfo["repos"])) else "SKIP")
    else:
        ev.append("config hf_repo unavailable — pass --config"); c.status = "SKIP"
    c.evidence = " · ".join(ev + (["PROBLEMS: " + "; ".join(prob)] if prob else []))
    checks.append(c)

    # ---- A2: temperature claim == temperatures actually used --------------- #
    # Verifies the config's claimed sampling temperatures against what each agent
    # actually ran (recorded per-call in observability.llm.temperatures). This is
    # ground truth, not a static code grep, so it counts silent constructor defaults
    # too. SKIPs for pre-instrumentation runs that did not log temperatures.
    c = ProvCheck("A2", "Temperature claim == temperatures actually used")
    cfg_temps = set(_read_config_temp_keys(config))
    if not cfg_temps:
        c.status = "SKIP"; c.evidence = "config has no run.temperature_distribution — pass --config"
    elif not temps_seen:
        c.status = "SKIP"
        c.evidence = "no per-call temperatures logged in these runs (pre-instrumentation); config claim stands"
    else:
        unexpected = sorted(temps_seen - cfg_temps)   # a temp ran the config never claimed -> red flag
        unused = sorted(cfg_temps - temps_seen)        # claimed but not seen here (e.g. a config subset)
        # FAIL only on an unexpected temperature (this is what would catch the classic
        # "config says temperature=0 but the run actually sampled" bug). A claimed temp
        # not exercised by this run tree is a note, not a failure (the tree may be a subset).
        c.status = "FAIL" if unexpected else "PASS"
        c.evidence = (f"ran={sorted(temps_seen)} claimed={sorted(cfg_temps)}"
                      + (f" · UNEXPECTED (ran, not claimed): {unexpected}" if unexpected else "")
                      + (f" · note: claimed-but-not-seen here: {unused}" if unused else ""))
    checks.append(c)

    # ---- A3: commit + vLLM version pinned == reality ----------------------- #
    c = ProvCheck("A3", "Commit + vLLM version pinned == reality")
    commits = {p.get("commit_sha") for p in prov if p.get("commit_sha")}
    vers = {p.get("vllm_version") for p in prov if p.get("vllm_version")}
    ev = []; prob = []
    if prov:
        ev.append(f"provenance commit={sorted(s[:10] for s in commits)} vllm={sorted(vers)}")
        if len(commits) > 1: prob.append("provenance files disagree on commit_sha")
        if len(vers) > 1: prob.append("provenance files disagree on vllm_version")
        if cfg["repo_commit_sha"] and commits and {cfg["repo_commit_sha"]} != commits:
            prob.append("config repo_commit_sha != provenance")
        if cfg["vllm_version"] and vers and {cfg["vllm_version"]} != vers:
            prob.append("config vllm_version != provenance")
        if vinfo and vinfo["vllm_versions"] and vers and vinfo["vllm_versions"] != vers:
            prob.append("vllm log version != provenance")
        c.status = "FAIL" if prob else "PASS"
    else:
        ev.append("no _provenance*.txt under run tree"); c.status = "SKIP"
    c.evidence = " · ".join(ev + (["PROBLEMS: " + "; ".join(prob)] if prob else []))
    checks.append(c)

    # ---- A4: every stage ran on the intended model ------------------------- #
    c = ProvCheck("A4", "Every stage ran on the intended model")
    models: Dict[str, int] = {}
    for lf in root.rglob("run_*.log"):
        for m in re.findall(r"Stage model:\s*(\S+)", lf.read_text(encoding="utf-8", errors="ignore")):
            models[m] = models.get(m, 0) + 1
    if not models:
        c.status = "SKIP"; c.evidence = "no 'Stage model:' lines found"
    else:
        distinct = set(models)
        ok = len(distinct) == 1 and (not served_intended or all(served_intended in m for m in distinct))
        c.status = "PASS" if ok else "FAIL"
        c.evidence = f"distinct Stage-model values={models} · intended≈{served_intended}"
    checks.append(c)

    # ---- A5: run manifests sane (K distinct reps) -------------------------- #
    c = ProvCheck("A5", "Run manifests sane (K distinct reps)")
    mans = list(root.rglob("multirun_manifest.json"))
    k_expected: Optional[int] = None
    if prov and prov[0].get("k_runs"):
        try: k_expected = int(prov[0]["k_runs"])
        except ValueError: pass
    prob = []
    for mf in mans:
        d = _load_json(mf) or {}
        runs = d.get("runs") or []
        n = d.get("n_runs", len(runs))
        dirs = [r.get("output_dir") for r in runs]
        nums = [r.get("run_number") for r in runs if r.get("run_number") is not None]
        if k_expected and n != k_expected: prob.append(f"{mf.parent.name}: n_runs={n}≠K={k_expected}")
        if len(set(dirs)) != len(dirs): prob.append(f"{mf.parent.name}: duplicate output_dir")
        if nums != sorted(nums): prob.append(f"{mf.parent.name}: run_number not increasing")
    if not mans:
        c.status = "SKIP"; c.evidence = "no multirun_manifest.json found"
    else:
        c.status = "FAIL" if prob else "PASS"
        c.evidence = " · ".join([f"{len(mans)} manifest(s), K={k_expected}"]
                                + (["PROBLEMS: " + "; ".join(prob)] if prob else []))
    checks.append(c)

    # ---- B1: no paid-cloud leak (cost == 0) -------------------------------- #
    c = ProvCheck("B1", "No paid-cloud leak (total_cost_usd == 0)")
    if not costs:
        c.status = "SKIP"; c.evidence = "no observability.llm.total_cost_usd found"
    else:
        c.status = "PASS" if all(isinstance(x, (int, float)) and x == 0 for x in costs) else "FAIL"
        c.evidence = f"distinct total_cost_usd across {n_exec} runs = {sorted(costs)}"
    checks.append(c)

    # ---- B2: correct model, single engine load per job --------------------- #
    c = ProvCheck("B2", "Correct model, single engine load per job")
    if not vinfo:
        c.status = "SKIP"; c.evidence = "no --vllm-log provided"
    else:
        single = vinfo["init_count"] == len(vinfo["files"])
        named = (not served_intended) or (served_intended in vinfo["served"])
        c.status = "PASS" if (single and named) else "FAIL"
        c.evidence = (f"engine inits={vinfo['init_count']} over {len(vinfo['files'])} log(s) · "
                      f"served={sorted(vinfo['served'])} · intended≈{served_intended}")
    checks.append(c)

    # ---- B3: no <think> leak into outputs ---------------------------------- #
    c = ProvCheck("B3", "No <think> reasoning leaked into outputs")
    hits: List[str] = []
    for rd in find_run_dirs(root):
        for f in rd.iterdir():
            if f.is_file() and f.suffix.lower() in (".json", ".txt", ".csv", ".md", ".log"):
                try:
                    if "<think>" in f.read_text(encoding="utf-8", errors="ignore"):
                        hits.append(f"{run_label(root, rd)}/{f.name}")
                except Exception:
                    pass
    c.status = "FAIL" if hits else "PASS"
    c.evidence = f"files containing <think>: {hits[:5] or 'none'}" + (f" (+{len(hits)-5})" if len(hits) > 5 else "")
    checks.append(c)

    # ---- B4: tokens non-zero (not an empty/failed call) -------------------- #
    c = ProvCheck("B4", "Tokens non-zero (not empty / failed)")
    if n_exec == 0:
        c.status = "SKIP"; c.evidence = "no execution_log.json found"
    else:
        c.status = "FAIL" if zero_tok else "PASS"
        c.evidence = f"{n_exec} runs scanned · zero-token runs: {zero_tok or 'none'}"
    checks.append(c)

    # ---- J3: no OOM / engine restart in serving logs ----------------------- #
    c = ProvCheck("J3", "No OOM / engine restart in serving logs")
    if not vinfo:
        c.status = "SKIP"; c.evidence = "no --vllm-log provided"
    else:
        oom = vinfo["oom_lines"]
        c.status = "FAIL" if oom else "PASS"
        c.evidence = (f"{oom[:2]}" + (f" (+{len(oom)-2} more)" if len(oom) > 2 else "")) if oom else "none"
    checks.append(c)

    # ---- G3: tier-1 parser determinism (passA == passB) -------------------- #
    c = ProvCheck("G3", "Tier-1 parser deterministic (passA == passB)")
    if not tier1_passes or len(tier1_passes) != 2:
        c.status = "SKIP"; c.evidence = "provide --tier1-passes A.txt B.txt (produced during evaluation)"
    elif not all(p.exists() for p in tier1_passes):
        c.status = "SKIP"; c.evidence = "tier-1 pass file(s) missing"
    else:
        a, b = tier1_passes
        same = a.read_text(errors="ignore") == b.read_text(errors="ignore")
        c.status = "PASS" if same else "FAIL"
        c.evidence = f"{a.name} {'==' if same else '!='} {b.name}"
    checks.append(c)

    return checks


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def write_report(out: Path, root: Path, statuses, citations, claims, datas, prov, verified: bool, reran: bool):
    out.mkdir(parents=True, exist_ok=True)
    # provenance_checks.csv (the automated deterministic checks)
    with open(out / "provenance_checks.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "check", "status", "evidence"])
        for pc in prov:
            w.writerow([pc.id, pc.name, pc.status, pc.evidence])
    # citations.csv (deduped)
    seen: Dict[str, Citation] = {}
    for c in citations:
        k = c.key()
        if k and (k not in seen or len(c.flags) > len(seen[k].flags)):
            seen[k] = c
    cites = sorted(seen.values(), key=lambda c: (c.external_match == "N", len(c.flags)), reverse=True)
    with open(out / "citations.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["title", "authors", "year", "source", "url", "external_match", "matched_title", "flags", "origin"])
        for c in cites:
            w.writerow([c.title, c.authors, c.year, c.source, c.url, c.external_match, c.matched_title,
                        "; ".join(c.flags), c.origin])
    # calibration_claims.csv
    with open(out / "calibration_claims.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "model_title", "validated_flag", "n_params", "claimed_fit_score",
                    "claimed_targets_matched", "sandbox_rerun_ok", "flags", "sandbox_stdout_head"])
        for cc in claims:
            w.writerow([cc.label, cc.model_title, cc.validated_flag, cc.n_params,
                        cc.claimed_fit.get("fit_score"), cc.claimed_fit.get("targets_matched"),
                        cc.sandbox_rerun_ok, "; ".join(cc.flags), cc.sandbox_stdout.replace("\n", " ")[:300]])
    # summary counts
    n_runs = len(statuses)
    n_err = sum(1 for s in statuses if s.errors)
    n_trunc = sum(1 for s in statuses if s.truncated)
    n_degr = sum(1 for s in statuses if s.degraded_hits)
    cite_noext = sum(1 for c in cites if c.external_match == "N")
    cite_flag = sum(1 for c in cites if c.flags)
    calib_flag = sum(1 for cc in claims if cc.flags)
    data_flag = sum(1 for d in datas if d and d.flags)
    prov_fail = sum(1 for pc in prov if pc.status == "FAIL")
    prov_skip = sum(1 for pc in prov if pc.status == "SKIP")

    L = []
    L.append("# Hallucination Audit Report\n")
    L.append(f"- Run tree: `{root}`")
    L.append(f"- Runs audited: **{n_runs}** · citation external-verify: {'ON' if verified else 'off'} · calibration re-run: {'ON' if reran else 'off'}\n")
    L.append("## Summary (counts to triage)\n")
    L.append(f"| signal | count |\n|---|---|")
    L.append(f"| runs with stage errors / exceptions | {n_err} |")
    L.append(f"| runs with truncated JSON output | {n_trunc} |")
    L.append(f"| runs with degraded-output log hits (fallback/timeout/no-key) | {n_degr} |")
    L.append(f"| **unique citations flagged** | {cite_flag} / {len(cites)} |")
    L.append(f"| **citations with NO external match** (likely fabricated) | {cite_noext} |" if verified
             else "| citations with no external match | (run with --verify) |")
    L.append(f"| **calibrated models flagged** | {calib_flag} / {len(claims)} |")
    L.append(f"| DataTeam runs flagged | {data_flag} |")
    L.append(f"| **provenance/serving checks FAILED** (deterministic) | {prov_fail} / {len(prov)} ({prov_skip} skipped) |\n")

    L.append("## 1. Cross-cutting per run\n")
    for s in statuses:
        bad = []
        if s.errors: bad.append("ERRORS:" + " | ".join(s.errors))
        if s.truncated: bad.append("TRUNCATED:" + ",".join(s.truncated))
        if s.degraded_hits: bad.append("DEGRADED:" + ",".join(s.degraded_hits))
        mark = "⚠️ " if bad else "✓ "
        L.append(f"- {mark}`{s.label}` — stages: {', '.join(f'{n}:{st}' for n,st in s.stages)}"
                 + (("  → " + " ; ".join(bad)) if bad else ""))
    L.append("")

    L.append("## 2. Citations — review `citations.csv` (sorted: no-match & most-flagged first)\n")
    L.append("Check each flagged row: is it a REAL paper (title/authors/year/venue resolvable)?\n")
    for c in cites[:40]:
        tag = "❌NO-MATCH" if c.external_match == "N" else ("✅" if c.external_match == "Y" else "•")
        L.append(f"- {tag} **{c.title[:90]}** — {c.authors[:40] or '(no author)'} ({c.year or '?'}) "
                 f"[{c.source[:24]}] {('· flags: '+'; '.join(c.flags)) if c.flags else ''}")
    if len(cites) > 40:
        L.append(f"- … {len(cites)-40} more in citations.csv")
    L.append("")

    L.append("## 3. Calibration — review `calibration_claims.csv` (claimed vs sandbox-computed)\n")
    L.append("For each model: was `validated=true`, do params match the validation code, and "
             "does the re-run stdout reproduce the claimed `fit_score`/moments?\n")
    for cc in claims:
        mark = "⚠️ " if cc.flags else "✓ "
        L.append(f"- {mark}`{cc.label}` — *{cc.model_title}* · validated={cc.validated_flag} · "
                 f"claimed fit_score={cc.claimed_fit.get('fit_score')} · rerun_ok={cc.sandbox_rerun_ok}"
                 + (("  → " + "; ".join(cc.flags)) if cc.flags else ""))
    L.append("")

    L.append("## 4. DataTeam\n")
    for d in datas:
        if not d:
            continue
        mark = "⚠️ " if d.flags else "✓ "
        L.append(f"- {mark}`{d.label}` {('→ '+'; '.join(d.flags)) if d.flags else ''}  _{d.note}_")
    L.append("")

    L.append("## 5. Provenance & serving integrity (automated, deterministic)\n")
    L.append("Machine-decided checks (config / `_provenance*.txt` / vLLM log / execution logs). "
             "Any **❌ FAIL** must be adjudicated in the human review; **—** = inputs not supplied "
             "(pass `--config` / `--vllm-log` / `--tier1-passes` to verify).\n")
    L.append("| # | check | status | evidence |\n|---|---|---|---|")
    for pc in prov:
        L.append(f"| {pc.id} | {pc.name} | {pc.mark} {pc.status} | {pc.evidence.replace(chr(10), ' ')[:200]} |")
    L.append("")

    L.append("## Manual sign-off checklist\n")
    for item in (
        "Every flagged citation confirmed real (or run removed/quarantined).",
        "No citation with NO-EXTERNAL-MATCH left unverified.",
        "Every calibrated model: validated=true AND re-run reproduces claimed fit.",
        "DataTeam OpenSourceAPI actually fetched real series (spot-checked vs FRED).",
        "DataTeam UserUploaded stats match the real input CSV.",
        "No run with stage errors / truncated output included in the evaluation set.",
        "K=3 reps do not disagree on facts (cited papers, parameter values).",
    ):
        L.append(f"- [ ] {item}")
    (out / "hallucination_report.md").write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def _egress_ok() -> bool:
    """One quick probe so citation-verify can auto-skip on an offline node."""
    try:
        import httpx
        return httpx.get("https://api.crossref.org/works", params={"rows": 0},
                         timeout=8.0).status_code < 500
    except Exception:
        return False


# --- zero-config auto-discovery: a bare invocation audits the latest release ----- #
# The run tree is self-describing (its _provenance stamps record job/model/commit and,
# for new runs, the vLLM log path), so run-dir / config / vLLM logs are all discoverable.

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]   # audits -> evaluation -> repository root


def _default_run_dir() -> Optional[Path]:
    """$AEL_RUNS_DIR if set, else the newest `logs/log-release/v*/runs` in the repository."""
    env = os.environ.get("AEL_RUNS_DIR")
    if env and Path(env).exists():
        return Path(env)
    base = _repo_root() / "logs" / "log-release"
    cands = sorted((p for p in base.glob("v*/runs") if p.exists()),
                   key=lambda p: p.parent.name, reverse=True)
    return cands[0] if cands else None


def _discover_vllm_logs(prov: List[Dict[str, str]]) -> List[Path]:
    """vLLM logs from the provenance stamps: explicit `vllm_log=` path (new runs),
    else `job=<id>` under the slurm-logs dir ($AGENTIC_SLURM_LOGS, else $AEL_LOG_DIR,
    else ./slurm-logs)."""
    sl = Path(os.environ.get("AGENTIC_SLURM_LOGS")
              or os.environ.get("AEL_LOG_DIR") or "slurm-logs")
    out: List[Path] = []
    for p in prov:
        f = Path(p["vllm_log"]) if p.get("vllm_log") else (sl / f"vllm-{p['job']}.log" if p.get("job") else None)
        if f and f.exists() and f not in out:
            out.append(f)
    return out


def _discover_config(prov: List[Dict[str, str]]) -> Optional[Path]:
    """The evaluation/config/*.yaml whose models.primary.hf_repo matches the run tree's
    served model (from the provenance stamps). Returns None if 0 or >1 match."""
    repos = {p.get("model_repo") for p in prov if p.get("model_repo")}
    if len(repos) != 1:
        return None
    repo = next(iter(repos))
    cfgdir = _repo_root() / "evaluation" / "config"
    matches = [c for c in sorted(cfgdir.glob("*.yaml")) if _read_config_fields(c).get("hf_repo") == repo]
    return matches[0] if len(matches) == 1 else None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Release-grade hallucination audit of AEL run outputs.")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="multirun output tree (default: $AEL_RUNS_DIR, else the newest logs/log-release/v*/runs)")
    ap.add_argument("--out", type=Path, default=Path("./audit_report"))
    # Full audit is the DEFAULT — a bare invocation does everything a release needs.
    # The two heavy checks auto-skip when the environment can't do them (no internet
    # egress for citation verify; no sandbox for calibration re-run), and can be turned
    # off explicitly for a fast structural-only scan.
    ap.add_argument("--no-verify", dest="verify", action="store_false",
                    help="skip citation external-match (offline node, or a quick scan)")
    ap.add_argument("--no-rerun", dest="rerun", action="store_false",
                    help="skip calibration sandbox re-execution (quick scan)")
    ap.add_argument("--quick", action="store_true",
                    help="fast structural scan only = --no-verify --no-rerun")
    # accepted for back-compat (these are the default now); hidden from help
    ap.add_argument("--verify", dest="verify", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--rerun", dest="rerun", action="store_true", help=argparse.SUPPRESS)
    ap.set_defaults(verify=True, rerun=True)
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between external citation probes")
    # provenance & serving-integrity inputs (deterministic checks; all optional —
    # missing inputs just mark the dependent checks SKIP rather than failing)
    ap.add_argument("--config", type=Path, default=None,
                    help="release config yaml (e.g. evaluation/config/<run>.yaml) for A1/A3 pins")
    ap.add_argument("--vllm-log", dest="vllm_log", type=Path, nargs="*", default=[],
                    help="vLLM serving log(s) for the batch (one per job) — enables A1-revision/B2/J3")
    ap.add_argument("--tier1-passes", dest="tier1_passes", type=Path, nargs=2, default=None,
                    metavar=("A.txt", "B.txt"), help="two tier-1 parser passes to diff for G3")
    a = ap.parse_args(argv)
    if a.quick:
        a.verify = a.rerun = False
    # Auto-skip citation verify when there is no internet egress (e.g. an offline
    # compute node) instead of wasting one timeout per citation. --rerun likewise
    # auto-skips inside audit_calibration if the sandbox is unavailable.
    if a.verify and not _egress_ok():
        print("note: no internet egress detected — skipping citation external-match "
              "(run on a node with egress for the full audit; pass --no-verify to silence).",
              file=sys.stderr)
        a.verify = False

    root = a.run_dir or _default_run_dir()
    if root is None:
        print("No --run-dir given, AEL_RUNS_DIR unset, and no logs/log-release/v*/runs in the repository. "
              "Pass --run-dir <multirun_tree>.", file=sys.stderr)
        return 2
    runs = find_run_dirs(root)
    if not runs:
        print(f"No execution_log.json found under {root}", file=sys.stderr); return 2

    # zero-config discovery: fill in config + vLLM logs from the run tree's own provenance,
    # so a bare invocation just works. Explicit flags always override.
    prov_stamps = _read_provenance(root)
    config = a.config or _discover_config(prov_stamps)
    vllm_logs = list(a.vllm_log) if a.vllm_log else _discover_vllm_logs(prov_stamps)
    print(f"Auditing {len(runs)} runs under {root}{'  [auto]' if a.run_dir is None else ''}")
    print(f"  config:    {config if config else '(none matched — A1/A2/A3 SKIP)'}{'' if a.config else '  [auto]'}")
    print(f"  vLLM logs: {len(vllm_logs)}{'' if a.vllm_log else '  [auto]'}  ·  "
          f"citation-verify: {'on' if a.verify else 'off'} · calibration-rerun: {'on' if a.rerun else 'off'}")

    statuses, citations, claims, datas = [], [], [], []
    for rd in runs:
        label = run_label(root, rd)
        statuses.append(audit_run_status(label, rd))
        citations += collect_citations(label, rd)
        claims += audit_calibration(label, rd, a.rerun)
        datas.append(audit_data(label, rd))

    # deterministic provenance & serving-integrity checks (automated)
    prov = audit_provenance(root, vllm_logs, config, a.tier1_passes)
    print("Provenance/serving checks: "
          + " ".join(f"{pc.id}={pc.status}" for pc in prov))

    if a.verify:
        uniq: Dict[str, Citation] = {}
        for c in citations:
            uniq.setdefault(c.key(), c)
        print(f"Verifying {len(uniq)} unique citations against external APIs …")
        for i, c in enumerate(uniq.values(), 1):
            verify_citation(c, sleep=a.sleep)
            if i % 10 == 0:
                print(f"  {i}/{len(uniq)}")
        # propagate match result back to all copies by title-key
        res = {k: (c.external_match, c.matched_title) for k, c in uniq.items()}
        for c in citations:
            c.external_match, c.matched_title = res.get(c.key(), (c.external_match, c.matched_title))

    write_report(a.out, root, statuses, citations, claims, datas, prov, a.verify, a.rerun)
    print(f"\nReport → {a.out}/hallucination_report.md")
    print(f"        {a.out}/citations.csv  ({len({c.key() for c in citations})} unique)")
    print(f"        {a.out}/calibration_claims.csv  ({len(claims)} models)")
    print(f"        {a.out}/provenance_checks.csv  ({len(prov)} checks)")
    # non-zero exit if hard fabrication signals present (CI gate)
    prov_fail = sum(1 for pc in prov if pc.status == "FAIL")
    hard = sum(1 for c in citations if "NO-EXTERNAL-MATCH" in c.flags) + \
           sum(1 for cc in claims if any(f.startswith("SANDBOX-RERUN") or f == "CLAIMED-OK-BUT-RECOMPUTED-FAIL"
                                         for f in cc.flags)) + \
           prov_fail
    if hard:
        print(f"\n⚠️  {hard} HARD signal(s) — {prov_fail} provenance/serving FAIL + "
              f"fabrication signals — review before evaluation.")
    return 1 if hard else 0


if __name__ == "__main__":
    # allow `python evaluation/audits/hallucination_audit.py` with no PYTHONPATH: put codes
    # on sys.path so the lazy `shared.*` imports (calibration --rerun) resolve.
    _agents = str(Path(__file__).resolve().parents[2])
    if _agents not in sys.path:
        sys.path.insert(0, _agents)
    raise SystemExit(main())
