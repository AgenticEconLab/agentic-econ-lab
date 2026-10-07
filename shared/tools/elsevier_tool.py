# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""Elsevier (Scopus Abstract Retrieval) tool — institutional abstract fill.

Fills the one gap keyless sources can't: abstracts for Elsevier-published economics
journal articles (J. Econometrics, J. Monetary Econ, European Econ Review, …), which
OpenAlex/Crossref/Semantic Scholar all lack (Elsevier withholds them).

Auth (two headers):
  * X-ELS-APIKey   = ELSEVIER_API_KEY        (always required)
  * X-ELS-Insttoken= ELSEVIER_INSTTOKEN      (institutional token — makes the key work
    OFF the institution's network, i.e. from the HPC with NO VPN. Without it the API
    key is IP-restricted to the institution's network.)

Endpoint:  GET https://api.elsevier.com/content/abstract/doi/{doi}  -> JSON
Open-first policy: GATED on ELSEVIER_API_KEY and OPTIONAL — used only as the last-resort
abstract fill for Elsevier DOIs; returns "" (logged) when unconfigured.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from shared.tools.tool_registry import ToolRegistry

ABSTRACT_DOI_URL = "https://api.elsevier.com/content/abstract/doi/{doi}"


def elsevier_available() -> bool:
    """True iff an Elsevier API key is configured (optional, open-first)."""
    return bool(os.getenv("ELSEVIER_API_KEY"))


def _extract_abstract(data: Dict[str, Any]) -> str:
    """Pull the abstract out of a Scopus Abstract-Retrieval JSON response."""
    try:
        core = data["abstracts-retrieval-response"]["coredata"]
        desc = core.get("dc:description")
        if isinstance(desc, str):
            return desc.strip()
        # sometimes nested under abstract/para
        if isinstance(desc, dict):
            return str(desc.get("abstract", {}).get("ce:para", "")).strip()
    except Exception:
        pass
    return ""


class ElsevierAbstractInput(BaseModel):
    doi: str = Field(description="DOI of the (Elsevier) article to fetch the abstract for")


def elsevier_abstract_handler(doi: str, collector: object = None, agent: str = "") -> Dict[str, Any]:
    """Return {'abstract': str, 'doi': str, 'error': Optional[str]} for an Elsevier DOI."""
    key = os.getenv("ELSEVIER_API_KEY")
    if not key:
        return {"abstract": "", "doi": doi, "error": "no ELSEVIER_API_KEY"}
    headers = {"X-ELS-APIKey": key, "Accept": "application/json"}
    insttoken = os.getenv("ELSEVIER_INSTTOKEN")
    if insttoken:
        headers["X-ELS-Insttoken"] = insttoken
    # Optional relay (if only the IP-restricted key is available, no insttoken): route via
    # a proxy inside the institution's network. Prefer the insttoken — then no proxy/VPN is needed.
    kwargs: Dict[str, Any] = {"headers": headers, "collector": collector, "agent": agent, "retries": 2}
    proxy = os.getenv("ELSEVIER_PROXY")
    if proxy:
        kwargs["proxies"] = {"http": proxy, "https": proxy}
    from shared.observability import tracked_get
    try:
        resp = tracked_get(ABSTRACT_DOI_URL.format(doi=doi.replace("https://doi.org/", "")), **kwargs)
        data = resp.json() if hasattr(resp, "json") else {}
    except Exception as e:
        msg = str(e)
        hint = " (401/403 → add ELSEVIER_INSTTOKEN, or you're off your institution's network)" if ("401" in msg or "403" in msg) else ""
        return {"abstract": "", "doi": doi, "error": f"{type(e).__name__}: {msg[:80]}{hint}"}
    return {"abstract": _extract_abstract(data), "doi": doi, "error": None}


ToolRegistry.register(
    name="elsevier_abstract",
    description="Elsevier Scopus Abstract Retrieval (institutional, optional): abstract by DOI",
    input_schema=ElsevierAbstractInput,
    handler=elsevier_abstract_handler,
    category="search",
)
