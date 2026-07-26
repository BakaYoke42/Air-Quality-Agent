"""Focused tests for the agent's managed Streamable HTTP MCP startup."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import agent as agent_module


class _FakeProcess:
    pid = 4321

    def __init__(self) -> None:
        self.running = True
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return None if self.running else 0

    def terminate(self) -> None:
        self.terminated = True
        self.running = False

    def kill(self) -> None:
        self.killed = True
        self.running = False

    def wait(self) -> int:
        self.running = False
        return 0


@pytest.mark.parametrize(
    "url, expected",
    [
        ("http://127.0.0.1:8000/mcp", ("127.0.0.1", 8000)),
        ("http://localhost:8123/mcp/", ("127.0.0.1", 8123)),
        ("http://[::1]:9000/mcp", ("::1", 9000)),
        ("https://127.0.0.1:8000/mcp", None),
        ("http://example.org:8000/mcp", None),
        ("http://127.0.0.1:8000/custom", None),
    ],
)
def test_local_mcp_spawn_target_is_restricted_to_served_loopback_urls(
    url: str,
    expected: tuple[str, int] | None,
) -> None:
    assert agent_module._local_mcp_spawn_target(url) == expected


def test_existing_local_server_is_reused_without_spawning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def listening(host: str, port: int, **_: object) -> bool:
        assert (host, port) == ("127.0.0.1", 8000)
        return True

    def unexpected_spawn(*args: object, **kwargs: object) -> None:
        raise AssertionError("an already-running MCP server must not be replaced")

    monkeypatch.setattr(agent_module, "_endpoint_is_listening", listening)
    monkeypatch.setattr(agent_module.subprocess, "Popen", unexpected_spawn)

    managed = asyncio.run(
        agent_module._maybe_start_local_mcp_server(
            "http://127.0.0.1:8000/mcp"
        )
    )
    assert managed is None


def test_external_server_remains_externally_managed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unexpected_probe(*args: object, **kwargs: object) -> bool:
        raise AssertionError("remote endpoints must be left to the HTTP client")

    monkeypatch.setattr(agent_module, "_endpoint_is_listening", unexpected_probe)
    assert (
        asyncio.run(
            agent_module._maybe_start_local_mcp_server(
                "https://mcp.example.org/mcp"
            )
        )
        is None
    )


def test_unavailable_local_server_is_started_with_matching_bind_and_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe_results = iter((False, True))
    popen_call: dict[str, object] = {}
    fake_process = _FakeProcess()

    async def listening(host: str, port: int, **_: object) -> bool:
        assert (host, port) == ("127.0.0.1", 8765)
        return next(probe_results)

    def fake_popen(command: list[str], **kwargs: object) -> _FakeProcess:
        popen_call["command"] = command
        popen_call.update(kwargs)
        return fake_process

    monkeypatch.setattr(agent_module, "_endpoint_is_listening", listening)
    monkeypatch.setattr(agent_module.subprocess, "Popen", fake_popen)

    managed = asyncio.run(
        agent_module._maybe_start_local_mcp_server(
            "http://localhost:8765/mcp"
        )
    )

    assert managed is not None
    assert popen_call["command"] == [
        sys.executable,
        str(ROOT / "src" / "mcp_server.py"),
    ]
    assert popen_call["cwd"] == str(ROOT)
    child_env = popen_call["env"]
    assert isinstance(child_env, dict)
    assert child_env["MCP_HOST"] == "127.0.0.1"
    assert child_env["MCP_PORT"] == "8765"

    asyncio.run(managed.stop())
    assert fake_process.terminated is True
    assert fake_process.killed is False
