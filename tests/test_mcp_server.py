"""Integration tests for the air-quality measurement store and MCP server.

Run from the repository root:
    python -m pytest tests/test_mcp_server.py -v

The normal suite does not call the neural retriever or a paid API.  Set
RUN_RETRIEVAL_INTEGRATION=1 to include one end-to-end retrieval tool call.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SERVER = SRC / "mcp_server.py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from measurements import InvalidMeasurementQuery, MeasurementStore


def _store() -> MeasurementStore:
    """Create the real store; environment variables may override data paths."""

    return MeasurementStore(root=ROOT)


def _content_text(result) -> str:
    return "\n".join(
        block.text for block in result.content if getattr(block, "text", None)
    )


def _available_local_port() -> int:
    """Reserve an ephemeral loopback port long enough to discover its number."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_until_listening(
    process: subprocess.Popen[str],
    host: str,
    port: int,
    timeout_seconds: float = 15.0,
) -> None:
    """Wait for the child server with a hard deadline and early-exit check."""

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"MCP server exited during startup with status {return_code}"
            )
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(
        f"MCP server did not listen on {host}:{port} within {timeout_seconds:.0f}s"
    )


@contextmanager
def _running_mcp_server() -> Iterator[str]:
    """Run the real Streamable HTTP server on an isolated loopback port."""

    host = "127.0.0.1"
    port = _available_local_port()
    server_env = dict(os.environ)
    server_env["MCP_HOST"] = host
    server_env["MCP_PORT"] = str(port)

    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as server_log:
        process = subprocess.Popen(
            [sys.executable, str(SERVER)],
            cwd=ROOT,
            env=server_env,
            stdin=subprocess.DEVNULL,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            try:
                _wait_until_listening(process, host, port)
            except (RuntimeError, TimeoutError) as exc:
                server_log.flush()
                server_log.seek(0)
                details = server_log.read()[-4_000:].strip()
                suffix = f"\nMCP server output:\n{details}" if details else ""
                raise RuntimeError(f"{exc}{suffix}") from exc
            yield f"http://{host}:{port}/mcp"
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def test_measurement_store_country_summary() -> None:
    result = _store().get_country_air_quality("France", "PM2.5", 2024)
    assert result["country_code"] == "FR"
    assert result["sampling_points"] == 239
    assert result["annual_mean_ug_m3"]["median"] == pytest.approx(7.85, abs=0.01)
    assert result["benchmarks"]["who_2021"]["pct_above"] == pytest.approx(
        95.40, abs=0.01
    )
    assert result["data_quality"]["sampling_points_excluded"] == 21


def test_measurement_store_comparison_and_validation() -> None:
    result = _store().compare_countries(
        pollutant="PM2.5",
        countries="FR,DE,IT",
        benchmark="who_2021",
        rank_by="median",
    )
    assert [row["country_code"] for row in result["countries"]] == ["IT", "DE", "FR"]
    assert result["countries"][0]["median_ug_m3"] == pytest.approx(11.33, abs=0.01)

    with pytest.raises(InvalidMeasurementQuery):
        _store().get_country_air_quality("Spain", "PM2.5", 2024)


async def _exercise_mcp(include_retrieval: bool) -> None:
    pytest.importorskip("mcp")
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    with _running_mcp_server() as server_url:
        async with streamable_http_client(server_url) as (
            read_stream,
            write_stream,
            _get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                listed = await session.list_tools()
                names = {tool.name for tool in listed.tools}
                assert names == {
                    "search_air_quality_evidence",
                    "get_country_air_quality",
                    "compare_countries",
                    "find_station_extremes",
                }

                country = await session.call_tool(
                    "get_country_air_quality",
                    {"country": "FR", "pollutant": "PM2.5", "year": 2024},
                )
                country_payload = json.loads(_content_text(country))
                assert country_payload["status"] == "ok"
                assert country_payload["data"]["sampling_points"] == 239

                comparison = await session.call_tool(
                    "compare_countries",
                    {
                        "pollutant": "PM2.5",
                        "countries": "FR,DE,IT",
                        "benchmark": "who_2021",
                        "rank_by": "pct_above",
                    },
                )
                comparison_payload = json.loads(_content_text(comparison))
                assert comparison_payload["status"] == "ok"
                assert comparison_payload["data"]["highest"] == "IT"

                extremes = await session.call_tool(
                    "find_station_extremes",
                    {
                        "country": "IT",
                        "pollutant": "NO2",
                        "direction": "highest",
                        "limit": 3,
                    },
                )
                extremes_payload = json.loads(_content_text(extremes))
                assert extremes_payload["status"] == "ok"
                assert len(extremes_payload["data"]["results"]) == 3

                invalid = await session.call_tool(
                    "get_country_air_quality",
                    {"country": "Spain", "pollutant": "PM2.5", "year": 2024},
                )
                invalid_payload = json.loads(_content_text(invalid))
                assert invalid_payload["status"] == "error"
                assert invalid_payload["error_type"] == "invalid_arguments"

                if include_retrieval:
                    retrieval = await session.call_tool(
                        "search_air_quality_evidence",
                        {
                            "query": "What is the WHO annual PM2.5 guideline?",
                            "top_k": 2,
                            "use_hyde": True,
                        },
                    )
                    retrieval_payload = json.loads(_content_text(retrieval))
                    assert retrieval_payload["status"] == "ok"
                    assert retrieval_payload["data"]["results"]


def test_mcp_server_measurement_tools() -> None:
    asyncio.run(
        asyncio.wait_for(_exercise_mcp(include_retrieval=False), timeout=45.0)
    )


def test_mcp_server_retrieval_tool_optional() -> None:
    if os.getenv("RUN_RETRIEVAL_INTEGRATION") != "1":
        pytest.skip("Set RUN_RETRIEVAL_INTEGRATION=1 for the model/API integration test")
    asyncio.run(
        asyncio.wait_for(_exercise_mcp(include_retrieval=True), timeout=300.0)
    )
