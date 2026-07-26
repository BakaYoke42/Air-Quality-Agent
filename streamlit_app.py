"""Streamlit interface for the guarded European air-quality agent.

The UI is deliberately a thin client over :func:`run_with_mcp_server`.  It
does not select tools, retrieve evidence, or synthesize answers itself, so the
CLI and browser interface share exactly the same guardrails and HTTP MCP
pipeline.
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent import (  # noqa: E402 - src is intentionally added to sys.path above
    DEFAULT_QUESTION,
    AgentError,
    AgentRunResult,
    run_with_mcp_server,
)


EXAMPLE_QUESTIONS = (
    DEFAULT_QUESTION,
    "What was the median 2024 NO2 sampling-point annual mean in Germany?",
    "Which retained Italian sampling points had the highest 2024 PM2.5 annual means?",
    "What annual PM2.5 value does the WHO recommend, and how does it differ from the EU legal limit?",
)

_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|secret(?:[_-]?key)?|authorization|bearer|token)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)
_KEY_LIKE_VALUE = re.compile(r"(?i)\b(?:sk|pk|sec|key)-[A-Za-z0-9_-]{8,}\b")


def _safe_error_message(error: Exception) -> str:
    """Return a short UI-safe error without leaking credential-like values."""

    if not isinstance(error, (AgentError, ValueError)):
        return (
            "The run stopped unexpectedly. Check the terminal log for details; "
            "no partial answer was returned."
        )
    message = str(error).strip() or "The guarded agent rejected this run."
    message = _SECRET_ASSIGNMENT.sub(r"\1\2[REDACTED]", message)
    message = _KEY_LIKE_VALUE.sub("[REDACTED]", message)
    return message[:800]


def _run_agent(question: str, *, use_hyde: bool) -> AgentRunResult:
    """Load private configuration silently and execute the shared async runner."""

    load_dotenv(ROOT / ".env", override=False)
    return asyncio.run(run_with_mcp_server(question, use_hyde=use_hyde))


def _metric(label: str, value: Any, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def _render_answer(result: AgentRunResult) -> None:
    decision = result.reasoning.critic
    verdict_colour = "green" if decision.verdict == "PASS" else "orange"
    st.markdown(
        f":{verdict_colour}[**Critic verdict: {decision.verdict}**]  \n"
        f"Agreement: **{decision.agreement}**"
    )
    st.markdown(decision.final_answer)
    st.info(
        "AI-generated evidence synthesis. Verify important decisions against "
        "the cited D#/M# evidence; this is not a legal-compliance or health diagnosis."
    )


def _render_evidence(result: AgentRunResult) -> None:
    st.caption(
        "Each D# identifies one retrieved parent passage; each M# identifies one "
        "structured tool-result payload. MCP tool-call numbers are shown "
        "separately."
    )
    if not result.evidence:
        st.warning("The run returned no citable evidence.")
        return

    for item in result.evidence:
        label = f"[{item.evidence_id}] {item.summary}"
        with st.expander(label):
            left, right = st.columns(2)
            left.markdown(f"**Evidence type:** `{item.evidence_type}`")
            right.markdown(f"**Source tool (provenance):** `{item.source_tool}`")
            safe_text = item.safe_text
            st.text(safe_text[:8_000])
            if len(safe_text) > 8_000:
                st.caption("Evidence preview truncated in the interface.")


def _render_tool_calls(result: AgentRunResult) -> None:
    st.caption(
        "These are MCP actions selected by the LLM and approved by L4. They are "
        "not citation IDs."
    )
    if not result.tool_calls:
        st.warning("No MCP tool calls were executed.")
        return

    rows = [
        {
            "Call": call.sequence,
            "MCP tool": call.tool_name,
            "Evidence produced": ", ".join(call.evidence_ids) or "none",
            "Latency (ms)": call.latency_ms,
            "Audit": "required" if call.audit_required else "standard",
        }
        for call in result.tool_calls
    ]
    st.dataframe(rows, hide_index=True, width="stretch")

    for call in result.tool_calls:
        with st.expander(f"Call {call.sequence}: {call.tool_name}"):
            st.markdown(f"**Purpose:** {call.purpose}")
            st.markdown(
                "**Produced evidence:** "
                + (", ".join(f"[{item}]" for item in call.evidence_ids) or "none")
            )
            st.markdown("**Validated arguments**")
            st.json(call.arguments, expanded=False)


def _budget_usage_rows(budget: Any) -> list[dict[str, str]]:
    """Return Arrow-safe budget rows with one consistent display type."""

    max_cost = (
        f"${budget.max_cost_usd:.6f}"
        if budget.max_cost_usd is not None
        else "disabled"
    )
    return [
        {
            "Resource": "LLM calls",
            "Used": f"{budget.llm_calls:,}",
            "Limit": f"{budget.max_llm_calls:,}",
        },
        {
            "Resource": "MCP tool calls",
            "Used": f"{budget.tool_calls:,}",
            "Limit": f"{budget.max_tool_calls:,}",
        },
        {
            "Resource": "Input tokens",
            "Used": f"{budget.input_tokens:,}",
            "Limit": f"{budget.max_input_tokens:,}",
        },
        {
            "Resource": "Output tokens",
            "Used": f"{budget.output_tokens:,}",
            "Limit": f"{budget.max_output_tokens:,}",
        },
        {
            "Resource": "Estimated USD",
            "Used": f"${budget.cost_usd:.6f}",
            "Limit": max_cost,
        },
    ]


def _render_diagnostics(result: AgentRunResult) -> None:
    budget = result.budget
    st.subheader("Run summary")
    metric_columns = st.columns(5)
    with metric_columns[0]:
        _metric("Latency", f"{result.total_latency_ms / 1_000:.2f} s")
    with metric_columns[1]:
        _metric("LLM calls", budget.llm_calls)
    with metric_columns[2]:
        _metric("MCP calls", len(result.tool_calls))
    with metric_columns[3]:
        _metric("Tokens", f"{budget.total_tokens:,}")
    with metric_columns[4]:
        _metric("Est. cost", f"${budget.cost_usd:.6f}")

    st.subheader("Critic")
    critic = result.reasoning.critic
    critic_columns = st.columns(3)
    critic_columns[0].metric("Verdict", critic.verdict)
    critic_columns[1].metric("Draft agreement", critic.agreement)
    selected_label = (
        "Critic rewrite"
        if critic.selected_draft == "NONE"
        else f"Draft {critic.selected_draft}"
    )
    critic_columns[2].metric("Final source", selected_label)
    if critic.issues:
        st.markdown("**Issues checked or corrected**")
        for issue in critic.issues:
            st.markdown(f"- {issue}")
    else:
        st.caption("The critic reported no remaining issues.")

    st.subheader("HyDE")
    hyde_calls = [call for call in result.tool_calls if call.hyde_status is not None]
    if not hyde_calls:
        st.caption(
            "HyDE was not invoked. It only runs when the model selects the "
            "documentary retrieval tool."
        )
    else:
        hyde_rows = [
            {
                "MCP call": call.sequence,
                "Status": call.hyde_status,
                "Used": call.hyde_used,
                "Model": call.hyde_model or "not reported",
                "Generated chars": call.hyde_generated_characters,
                "Tokens": call.hyde_total_tokens,
                "Safe error": call.hyde_error or "",
            }
            for call in hyde_calls
        ]
        st.dataframe(hyde_rows, hide_index=True, width="stretch")

    with st.expander("TokenBudget limits and usage"):
        st.dataframe(
            _budget_usage_rows(budget),
            hide_index=True,
            width="stretch",
        )

    with st.expander("LLM call timings and token usage"):
        records = [
            {
                "Call": call.call_name,
                "Model": call.model,
                "Input tokens": call.input_tokens,
                "Output tokens": call.output_tokens,
                "Latency (ms)": call.latency_ms,
            }
            for call in result.model_calls
        ]
        if records:
            st.dataframe(records, hide_index=True, width="stretch")
        else:
            st.caption("No provider usage records were reported.")

    with st.expander(f"Three independent synthesis drafts (k={result.reasoning.k})"):
        for draft in result.reasoning.drafts:
            citation_state = "valid" if draft.citations_valid else "rejected"
            st.markdown(
                f"**Draft {draft.index} — citations {citation_state}; "
                f"confidence {draft.confidence or 'not parsed'}**"
            )
            if draft.unsupported_citations:
                st.warning(
                    "Unsupported citations: " + ", ".join(draft.unsupported_citations)
                )
            st.markdown(draft.content)
            st.divider()

    with st.expander("Raw critic record"):
        st.text(critic.raw_response)


def _render_sidebar() -> bool:
    with st.sidebar:
        st.markdown("### Run controls")
        use_hyde = st.toggle(
            "Enable HyDE for documentary search",
            value=True,
            help=(
                "HyDE is only called if the model selects documentary retrieval. "
                "Measurement-only questions normally do not invoke it."
            ),
        )
        st.markdown("### Example questions")
        for index, example in enumerate(EXAMPLE_QUESTIONS, start=1):
            if st.button(
                f"Example {index}",
                key=f"example_{index}",
                help=example,
                width="stretch",
            ):
                st.session_state["question_input"] = example

        st.markdown("### Supported scope")
        st.markdown(
            "**Countries:** France, Germany, Italy  \n"
            "**Pollutants:** PM2.5, NO2  \n"
            "**Measurements:** 2024"
        )
        st.caption(
            "The interface does not display environment variables or credentials. "
            "The agent communicates with its tools over MCP Streamable HTTP."
        )
    return use_hyde


def main() -> None:
    st.set_page_config(
        page_title="European Air-Quality Evidence Agent",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
          .block-container {max-width: 1180px; padding-top: 2.2rem;}
          [data-testid="stMetric"] {
            background: color-mix(in srgb, var(--primary-color) 7%, transparent);
            border: 1px solid color-mix(in srgb, var(--primary-color) 18%, transparent);
            border-radius: 0.8rem;
            padding: 0.75rem 0.9rem;
          }
          [data-testid="stForm"] {border-radius: 1rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    use_hyde = _render_sidebar()
    st.title("European Air-Quality Evidence Agent")
    st.markdown(
        "Ask a focused question about **2024 PM2.5 or NO2** in France, Germany, "
        "or Italy. The LLM selects from live MCP tools; deterministic guardrails "
        "approve each call and check final citation IDs against the run allowlist. "
        "ID validation does not by itself prove claim-to-evidence entailment."
    )

    if "question_input" not in st.session_state:
        st.session_state["question_input"] = DEFAULT_QUESTION

    with st.form("agent_question_form"):
        st.text_area(
            "Research question",
            key="question_input",
            height=115,
            max_chars=2_000,
            help="Questions outside the documented scope are rejected before HTTP work starts.",
        )
        submitted = st.form_submit_button(
            "Run guarded analysis", type="primary", width="stretch"
        )

    if submitted:
        question = st.session_state["question_input"].strip()
        if not question:
            st.warning("Enter a question before running the agent.")
        else:
            st.session_state.pop("agent_result", None)
            st.session_state.pop("agent_error", None)
            with st.status(
                "Running guardrails, MCP tools, three syntheses, and critic…",
                expanded=True,
            ) as status:
                st.write("The full agent pipeline is running over Streamable HTTP.")
                try:
                    result = _run_agent(question, use_hyde=use_hyde)
                except Exception as error:  # UI boundary: always fail closed
                    st.session_state["agent_error"] = _safe_error_message(error)
                    status.update(label="Run stopped safely", state="error")
                else:
                    st.session_state["agent_result"] = result
                    status.update(label="Analysis complete", state="complete", expanded=False)

    error_message = st.session_state.get("agent_error")
    if error_message:
        st.error(error_message)
        st.caption(
            "Check that the API configuration is present and review the terminal "
            "log if the safe message above is not enough to diagnose the problem."
        )

    result = st.session_state.get("agent_result")
    if result is not None:
        answer_tab, evidence_tab, tools_tab, diagnostics_tab = st.tabs(
            ["Answer", "Evidence", "MCP tool calls", "Diagnostics"]
        )
        with answer_tab:
            _render_answer(result)
        with evidence_tab:
            _render_evidence(result)
        with tools_tab:
            _render_tool_calls(result)
        with diagnostics_tab:
            _render_diagnostics(result)

    st.divider()
    st.caption(
        "AI disclosure: responses are model-generated and critic-checked. "
        "Sampling-point summaries are not population-weighted exposure estimates."
    )


if __name__ == "__main__":
    main()
