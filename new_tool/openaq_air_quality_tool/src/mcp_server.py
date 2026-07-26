"""Streamable HTTP MCP wrapper for the restricted OpenAQ lookup."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

try:
    from .air_quality_tool import get_air_quality
except ImportError:  # Direct ``python src/mcp_server.py`` execution.
    from air_quality_tool import get_air_quality


MCP_HOST = os.getenv("LIVE_AIR_QUALITY_MCP_HOST", "127.0.0.1").strip() or "127.0.0.1"
MCP_PORT = int(os.getenv("LIVE_AIR_QUALITY_MCP_PORT", "8001"))

mcp = FastMCP(
    "restricted-live-air-quality-fr-de-it",
    host=MCP_HOST,
    port=MCP_PORT,
    stateless_http=True,
    json_response=True,
)


def _success(data: Any) -> str:
    return json.dumps(
        {"status": "ok", "data": data},
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )


def _controlled_error(result: dict[str, Any]) -> str:
    return json.dumps(
        {
            "status": "error",
            "tool": "get_current_air_quality",
            "error_type": result.get("status", "upstream_error"),
            "message": result.get(
                "error",
                "The live air-quality lookup could not complete.",
            ),
        },
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )


@mcp.tool()
def get_current_air_quality(location: str) -> str:
    """Return latest available PM2.5 and NO2 near an FR/DE/IT place.

    Use when:
    - The user asks for a recent or current PM2.5 or NO2 observation near a
      named city or settlement in France, Germany, or Italy.

    Do NOT use:
    - For 2024 annual country statistics, station extremes, historical trends,
      policy or methodology questions, or documentary evidence.
    - To determine compliance with an annual WHO guideline or legal limit.
      A recent station observation is not an annual mean.

    Returns:
    - A standard ``{"status": "ok", "data": ...}`` MCP envelope. ``data``
      reports ``ok``, ``partial``, ``no_data``, ``ambiguous_location``, or
      ``rejected`` and preserves each observation's timestamp, unit, station,
      provider, source attribution, and distance. PM2.5 and NO2 may come from
      different stations and times. Values are latest available within the
      freshness window, not guaranteed instantaneous measurements.

    Prefer:
    - The archival structured measurement tools for 2024 country comparisons
      and station extremes, and document search for guidance or methodology.
    - If ``data.status`` is ``ambiguous_location``, ask the user to select one
      returned candidate; never choose a candidate yourself.

    Example:
    - ``{"location": "Berlin, Germany"}``

    Args:
        location: City or settlement name, optionally including region and
            country, with a maximum length of 120 characters.
    """
    result = get_air_quality(location)
    if result.get("status") in {
        "ok",
        "partial",
        "no_data",
        "ambiguous_location",
        "rejected",
    }:
        return _success(result)
    return _controlled_error(result)


if __name__ == "__main__":
    print(
        "Starting restricted live-air-quality MCP server at "
        f"http://{MCP_HOST}:{MCP_PORT}/mcp",
        file=sys.stderr,
        flush=True,
    )
    mcp.run(transport="streamable-http")
