"""Persistent MCP HTTP server for air-quality evidence and measurements.

Run this service in its own terminal for a persistent endpoint::

    python src/mcp_server.py

The default MCP endpoint is ``http://127.0.0.1:8000/mcp``.  ``MCP_HOST`` and
``MCP_PORT`` can override the bind address.  If no loopback server is already
available, ``src/agent.py`` may launch this module as a managed subprocess.
The agent still communicates exclusively through Streamable HTTP and never
imports or invokes the tools in-process.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from measurements import (
    InvalidMeasurementQuery,
    MeasurementDataError,
    MeasurementStore,
)


ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    # The server can still run when configuration is supplied by the shell.
    pass

MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1").strip() or "127.0.0.1"
MCP_PORT = int(os.getenv("MCP_PORT", "8000"))

# Stateless Streamable HTTP with JSON responses is the recommended FastMCP
# configuration for a standalone tool service.  Heavy models/data are still
# lazy-loaded once per Python process by the helpers below.
mcp = FastMCP(
    "air-quality-evidence-tools",
    host=MCP_HOST,
    port=MCP_PORT,
    stateless_http=True,
    json_response=True,
)

_measurement_store: MeasurementStore | None = None
_retriever: Any | None = None
_measurement_lock = threading.Lock()
_retriever_lock = threading.Lock()


def _log(message: str) -> None:
    print(f"[mcp] {message}", file=sys.stderr, flush=True)


def _success(data: Any) -> str:
    return json.dumps(
        {"status": "ok", "data": data},
        ensure_ascii=False,
        indent=2,
    )


def _error(tool: str, exc: Exception) -> str:
    if isinstance(exc, InvalidMeasurementQuery):
        error_type = "invalid_arguments"
        message = str(exc)
    elif isinstance(exc, (MeasurementDataError, FileNotFoundError)):
        error_type = "data_unavailable"
        message = str(exc)
    elif isinstance(exc, ValueError):
        error_type = "invalid_arguments"
        message = str(exc)
    else:
        error_type = "internal_error"
        message = "The tool could not complete the request. Check server logs."
        print(f"[{tool}] {type(exc).__name__}: {exc}", file=sys.stderr)
    return json.dumps(
        {
            "status": "error",
            "tool": tool,
            "error_type": error_type,
            "message": message,
        },
        ensure_ascii=False,
        indent=2,
    )


def _get_measurement_store() -> MeasurementStore:
    global _measurement_store
    if _measurement_store is None:
        with _measurement_lock:
            if _measurement_store is None:
                _log("Loading processed measurement store")
                _measurement_store = MeasurementStore(root=ROOT)
                _log("Measurement store ready")
    return _measurement_store


def _get_retriever() -> Any:
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                # Lazy import lets the three local measurement tools operate even
                # if neural-retrieval dependencies are unavailable.
                from retrieval import AdvancedRetriever

                # Keep model/index loading diagnostics in the server terminal
                # instead of mixing them into the tool's returned evidence.
                with redirect_stdout(sys.stderr):
                    _log("Loading retrieval models and dense index")
                    _retriever = AdvancedRetriever(root=ROOT)
                    _log("Retriever ready")
    return _retriever


@mcp.tool()
def search_air_quality_evidence(
    query: str,
    top_k: int = 4,
    use_hyde: bool = True,
) -> str:
    """Retrieve authoritative explanatory evidence from the document corpus.

    Use when: the question asks about WHO guidelines, EU legal limits, EEA
    methodology, definitions, evidence status, or narrative interpretation.
    Do NOT use for: calculating country statistics or ranking sampling points;
    use the measurement tools for those exact numeric operations.
    Returns: JSON containing ranked child matches expanded to their parent
    passages, document metadata, page ranges, source URLs, and retrieval timing.
    Prefer: combine this tool with a measurement tool for questions that ask
    both what a threshold means and how 2024 observations compare with it.
    Example: query="What is the WHO 2021 annual PM2.5 guideline?", top_k=4
    """

    tool_name = "search_air_quality_evidence"
    try:
        started = time.perf_counter()
        _log(f"{tool_name} started (HyDE={bool(use_hyde)}, top_k={top_k})")
        cleaned_query = " ".join(str(query).split())
        if not cleaned_query:
            raise ValueError("query cannot be empty")
        if len(cleaned_query) > 2_000:
            raise ValueError("query cannot exceed 2,000 characters")
        selected_k = int(top_k)
        if selected_k < 1 or selected_k > 8:
            raise ValueError("top_k must be between 1 and 8")

        retriever = _get_retriever()
        with redirect_stdout(sys.stderr):
            output = retriever.retrieve(
                cleaned_query,
                final_k=selected_k,
                use_hyde=bool(use_hyde),
            )
        hyde = output["hyde"]
        if not use_hyde:
            hyde_status = "disabled"
            _log("HyDE disabled for this retrieval call")
        elif hyde["used"]:
            hyde_status = "generated"
            usage = hyde.get("usage") or {}
            _log(
                "HyDE generated successfully "
                f"({hyde.get('generated_characters', 0)} characters, "
                f"{int(usage.get('total_tokens') or 0)} tokens)"
            )
        else:
            hyde_status = "fallback"
            _log(f"HyDE fell back to the original query: {hyde.get('error')}")
        _log(
            f"{tool_name} retrieval completed in "
            f"{time.perf_counter() - started:.2f}s"
        )

        results = []
        for row in output["results"]:
            results.append(
                {
                    "rank": row["rank"],
                    "parent_id": row["parent_id"],
                    "matched_child_id": row["matched_child_id"],
                    "doc_id": row["doc_id"],
                    "title": row["title"],
                    "publisher": row["publisher"],
                    "document_type": row["document_type"],
                    "evidence_status": row["evidence_status"],
                    "publication_year": row["publication_year"],
                    "data_year": row["data_year"],
                    "page_start": row["page_start"],
                    "page_end": row["page_end"],
                    "source_url": row["source_url"],
                    "matched_child_text": row["matched_child_text"],
                    "parent_text": row["parent_text"],
                    "reranker_score": row["reranker_score"],
                    "rrf_score": row["rrf_score"],
                }
            )

        return _success(
            {
                "query": cleaned_query,
                # HyDE text is intentionally omitted: it is a retrieval aid,
                # never evidence for the final answer.
                "hyde_requested": bool(use_hyde),
                "hyde_status": hyde_status,
                "hyde_used": bool(hyde["used"]),
                "hyde_model": hyde.get("model"),
                "hyde_generated_characters": int(
                    hyde.get("generated_characters") or 0
                ),
                "hyde_error": hyde.get("error"),
                "hyde_usage": hyde.get("usage", {}),
                "timings_ms": output["timings_ms"],
                "results": results,
            }
        )
    except Exception as exc:
        return _error(tool_name, exc)


@mcp.tool()
def get_country_air_quality(
    country: str,
    pollutant: str,
    year: int = 2024,
) -> str:
    """Calculate annual station statistics for one country and pollutant.

    Use when: the user asks for a 2024 PM2.5 or NO2 country summary, including
    median, quartiles, range, coverage, or percentages above WHO/EU benchmarks.
    Do NOT use for: explaining whether a benchmark is legally binding; retrieve
    documentary evidence with search_air_quality_evidence for that question.
    Returns: JSON with exact retained sampling-point statistics, thresholds,
    exceedance counts, excluded low-coverage counts, and provenance.
    Prefer: use compare_countries when the question explicitly compares two or
    more countries.
    Example: country="France", pollutant="PM2.5", year=2024
    """

    tool_name = "get_country_air_quality"
    try:
        result = _get_measurement_store().get_country_air_quality(
            country=country,
            pollutant=pollutant,
            year=year,
        )
        return _success(result)
    except Exception as exc:
        return _error(tool_name, exc)


@mcp.tool()
def compare_countries(
    pollutant: str,
    countries: str = "FR,DE,IT",
    year: int = 2024,
    benchmark: str = "who_2021",
    rank_by: str = "median",
) -> str:
    """Compare France, Germany, and/or Italy using exact annual statistics.

    Use when: the question compares countries for PM2.5 or NO2, asks which has
    the highest median, or compares shares above WHO/EU thresholds.
    Do NOT use for: a one-country summary (use get_country_air_quality) or
    interpreting health or legal meaning (use search_air_quality_evidence
    alongside the appropriate measurement tool for that context).
    Returns: JSON ranking with station counts, excluded counts, annual means,
    benchmark thresholds, exceedance counts, percentages, and limitations.
    Prefer: rank_by="median" for concentration comparisons or
    rank_by="pct_above" for benchmark-exceedance comparisons.
    Example: pollutant="PM2.5", countries="FR,DE,IT",
    benchmark="who_2021", rank_by="pct_above"
    """

    tool_name = "compare_countries"
    try:
        result = _get_measurement_store().compare_countries(
            pollutant=pollutant,
            countries=countries,
            year=year,
            benchmark=benchmark,
            rank_by=rank_by,
        )
        return _success(result)
    except Exception as exc:
        return _error(tool_name, exc)


@mcp.tool()
def find_station_extremes(
    country: str,
    pollutant: str,
    year: int = 2024,
    direction: str = "highest",
    limit: int = 5,
) -> str:
    """Find sampling points with the highest or lowest annual concentrations.

    Use when: the user asks for high/low annual sampling-point values or wants
    the records furthest from WHO/EU benchmarks within one supported country.
    Do NOT use for: naming cities or inferring population exposure; the current
    dataset contains sampling-point identifiers but no location metadata.
    Returns: JSON with sampling-point IDs, annual means, coverage, benchmark
    flags, threshold distances, source members, and a location warning.
    Prefer: keep limit small; use get_country_air_quality for aggregate results.
    Example: country="IT", pollutant="NO2", direction="highest", limit=5
    """

    tool_name = "find_station_extremes"
    try:
        result = _get_measurement_store().find_station_extremes(
            country=country,
            pollutant=pollutant,
            year=year,
            direction=direction,
            limit=limit,
        )
        return _success(result)
    except Exception as exc:
        return _error(tool_name, exc)


if __name__ == "__main__":
    print(
        f"Starting air-quality MCP server at http://{MCP_HOST}:{MCP_PORT}/mcp",
        file=sys.stderr,
        flush=True,
    )
    mcp.run(transport="streamable-http")
