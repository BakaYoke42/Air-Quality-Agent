"""Small tests for the frontend's fail-closed error boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pyarrow as pa

import streamlit_app
from agent import AgentError
from streamlit.testing.v1 import AppTest


def test_safe_error_message_redacts_credential_like_values() -> None:
    message = streamlit_app._safe_error_message(
        AgentError("provider failed: api_key=sk-this-must-not-appear")
    )

    assert "sk-this-must-not-appear" not in message
    assert "[REDACTED]" in message


def test_unexpected_error_is_not_echoed_to_browser() -> None:
    message = streamlit_app._safe_error_message(
        RuntimeError("internal path and sensitive implementation details")
    )

    assert "internal path" not in message
    assert "unexpectedly" in message


def test_frontend_initial_view_renders_without_running_the_agent() -> None:
    app = AppTest.from_file(str(streamlit_app.ROOT / "streamlit_app.py")).run(
        timeout=20
    )

    assert not app.exception
    assert app.title[0].value == "European Air-Quality Evidence Agent"


def test_budget_rows_have_arrow_compatible_column_types() -> None:
    budget = SimpleNamespace(
        llm_calls=6,
        max_llm_calls=12,
        tool_calls=2,
        max_tool_calls=8,
        input_tokens=1_234,
        max_input_tokens=20_000,
        output_tokens=567,
        max_output_tokens=10_000,
        cost_usd=0.0,
        max_cost_usd=None,
    )

    rows = streamlit_app._budget_usage_rows(budget)

    assert all(isinstance(value, str) for row in rows for value in row.values())
    pa.Table.from_pandas(pd.DataFrame(rows))


def test_frontend_does_not_use_removed_container_width_argument() -> None:
    source = (streamlit_app.ROOT / "streamlit_app.py").read_text(encoding="utf-8")

    assert "use_container_width" not in source
