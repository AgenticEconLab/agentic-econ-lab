# SPDX-License-Identifier: MIT
# Copyright (c) 2026 AgenticEconLab
"""
FRED Data Tool — MCP-aligned wrapper around tracked_fred_get_series.

Registers the fred_get_series tool in the ToolRegistry with a validated
Pydantic input schema.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from shared.tools.tool_registry import ToolRegistry


class FredGetSeriesInput(BaseModel):
    """Input schema for FRED data retrieval."""
    series_id: str = Field(description="FRED series ID (e.g., 'GDP', 'CPIAUCSL')")
    observation_start: Optional[str] = Field(default=None, description="Start date (YYYY-MM-DD)")
    observation_end: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD)")
    frequency: Optional[str] = Field(default=None, description="Frequency: d, w, bw, m, q, sa, a")
    units: Optional[str] = Field(default=None, description="Units: lin, chg, ch1, pch, pc1, pca, cch, cca, log")


def fred_get_series_handler(
    series_id: str,
    observation_start: Optional[str] = None,
    observation_end: Optional[str] = None,
    frequency: Optional[str] = None,
    units: Optional[str] = None,
    collector: object = None,
    agent: str = "",
) -> Any:
    """
    Retrieve a data series from FRED.

    Delegates to shared.observability.tracked_fred_get_series for the actual
    API call and observability tracking.
    """
    from shared.observability import tracked_fred_get_series

    kwargs = {}
    if observation_start:
        kwargs["observation_start"] = observation_start
    if observation_end:
        kwargs["observation_end"] = observation_end
    if frequency:
        kwargs["frequency"] = frequency
    if units:
        kwargs["units"] = units

    return tracked_fred_get_series(
        series_id=series_id,
        collector=collector,
        agent=agent,
        **kwargs,
    )


# Register in the global tool registry
ToolRegistry.register(
    name="fred_get_series",
    description="Retrieve economic data series from the Federal Reserve (FRED) API",
    input_schema=FredGetSeriesInput,
    handler=fred_get_series_handler,
    category="data",
)
