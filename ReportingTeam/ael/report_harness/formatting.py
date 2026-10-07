# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Deterministic journal-style re-render (§4.7 Formatter's numeric core).

Re-renders the ALREADY VERIFIED report (Stage 3's number-consistency check has already run)
into a specific journal-style reference format and appends a Data & Code Availability
statement built from real pipeline facts -- never touches the estimation numbers or the
verified original file; writes a separate file.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _iter(x) -> List[dict]:
    return [d for d in x if isinstance(d, dict)] if isinstance(x, list) else []


def _aea_reference(item: Dict) -> str:
    """One AEA-style reference line from a literature-batch item. Only artifact fields --
    volume/issue/page numbers are omitted (not fabricated) when the artifact lacks them."""
    authors = item.get("authors") or []
    names = (", ".join(str(a).strip() for a in authors if str(a).strip())
             if isinstance(authors, list) else str(authors).strip()) or "Unknown"
    year = item.get("year") or "n.d."
    title = str(item.get("title", "")).strip()
    # Never print the search provider ('source': Crossref, OpenAlex, arXiv) as the venue.
    from shared.tools.scholarly_search import clean_venue
    venue = clean_venue(item.get("venue")) or ""
    parts = [f"{names}.", f"{year}.", f'"{title}."']
    if venue:
        parts.append(f"*{venue}*.")
    return " ".join(parts)


def format_for_journal(report_markdown: str, literature_batch: Dict,
                       style: str = "AEA") -> str:
    """Re-render the References section in the given style. The rest of the verified
    report body is carried through UNCHANGED (re-rendering estimation numbers would risk
    diverging from the number-consistency check that already passed on the original).

    Section headings in ``assemble_report`` are always preceded by a blank line (its own
    ``\\n`` plus the ``"\\n".join`` separator) -- the marker here matches that double-newline
    convention exactly, so re-splicing doesn't collapse or duplicate the blank lines around
    the References section."""
    items = [i for i in _iter((literature_batch or {}).get("literature_items")) if i.get("title")]
    if not items:
        return report_markdown

    heading = next((h for h in ("## 8. References", "## 9. References")
                   if f"\n\n{h}" in report_markdown), None)
    if heading is None:
        return report_markdown
    marker = f"\n\n{heading}"

    body_lines = [f"*Formatted in {style} style.*\n"]
    items = sorted(items, key=lambda i: (str((i.get("authors") or ["ZZ"])[0]
                                             if isinstance(i.get("authors"), list)
                                             else i.get("authors", "ZZ")).lower(),
                                         str(i.get("year") or "")))
    for item in items:
        body_lines.append(f"- {_aea_reference(item)}")
    new_section = marker + "\n" + "\n".join(body_lines)

    head, _, tail = report_markdown.partition(marker)
    # tail starts at `marker` itself; find the NEXT heading (Provenance) after it.
    next_pos = tail.find("\n\n## ", len(marker))
    rest = tail[next_pos:] if next_pos != -1 else ""
    return head + new_section + rest


def build_availability_statement(data_source: Dict, report_markdown: str) -> str:
    """Data & Code Availability statement from real pipeline facts -- counts, not claims."""
    retrieved = _iter((data_source or {}).get("retrieved_data"))
    n_real = sum(1 for r in retrieved if not r.get("data_simulated"))
    n_simulated = len(retrieved) - n_real
    has_code = "## 5b. Code Implementation" in (report_markdown or "")

    lines = ["**Data Availability:**"]
    if retrieved:
        lines.append(f"{n_real} of {len(retrieved)} series retrieved from live open-data "
                     f"sources; {n_simulated} disclosed as simulated (see Limitations).")
    else:
        lines.append("No retrieved-data artifact was available for this run.")
    lines.append("\n**Code Availability:**")
    lines.append("Executable model modules are published alongside this report "
                 "(see Section 5b)." if has_code else
                 "No derived code modules were generated for this run.")
    return "\n".join(lines)
