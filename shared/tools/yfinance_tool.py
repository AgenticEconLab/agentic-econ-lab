# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
Yahoo Finance Tool — MCP-aligned wrapper around tracked_yfinance_history.

Registers the yfinance_history tool in the ToolRegistry with a validated
Pydantic input schema.
"""

import os
import tempfile
from typing import Any, Optional

from pydantic import BaseModel, Field

from shared.tools.tool_registry import ToolRegistry

_YF_CACHE_DIR: Optional[str] = None


def ensure_yfinance_cache() -> Optional[str]:
    """Point yfinance's sqlite caches (timezone + cookie DBs) at a per-process directory.

    A shared ``~/.cache/py-yfinance/`` DB was observed corrupt ("database disk
    image is malformed"), sending every Yahoo series in most runs to simulation. Concurrent
    jobs sharing one sqlite file is the corruption path, so each process gets its own cache:
    ``AEL_YF_CACHE_DIR`` if set, else a fresh temp dir (node-local under SLURM). Idempotent;
    returns the directory, or None when yfinance is unavailable. Never raises."""
    global _YF_CACHE_DIR
    if _YF_CACHE_DIR:
        return _YF_CACHE_DIR
    try:
        import yfinance as yf
        d = os.environ.get("AEL_YF_CACHE_DIR") or tempfile.mkdtemp(
            prefix=f"ael-yfinance-{os.getpid()}-")
        os.makedirs(d, exist_ok=True)
        yf.set_tz_cache_location(d)
        setter = getattr(yf, "set_cache_location", None) or getattr(
            getattr(yf, "cache", None), "set_cache_location", None)
        if setter:
            setter(d)
        _YF_CACHE_DIR = d
        return d
    except Exception:
        return None


class YFinanceHistoryInput(BaseModel):
    """Input schema for Yahoo Finance history retrieval."""
    ticker: str = Field(description="Ticker symbol (e.g., 'AAPL', '^GSPC')")
    period: str = Field(default="1y", description="Period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max")


def yfinance_history_handler(
    ticker: str,
    period: str = "1y",
    collector: object = None,
    agent: str = "",
) -> Any:
    """
    Retrieve historical price data from Yahoo Finance.

    Delegates to shared.observability.tracked_yfinance_history for the actual
    API call and observability tracking.
    """
    from shared.observability import tracked_yfinance_history

    ensure_yfinance_cache()
    return tracked_yfinance_history(
        ticker=ticker,
        period=period,
        collector=collector,
        agent=agent,
    )


# Register in the global tool registry
ToolRegistry.register(
    name="yfinance_history",
    description="Retrieve historical price data from Yahoo Finance",
    input_schema=YFinanceHistoryInput,
    handler=yfinance_history_handler,
    category="data",
)
