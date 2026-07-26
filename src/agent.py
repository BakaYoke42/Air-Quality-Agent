"""Runnable guarded air-quality MCP client and answer agent.

The agent connects to a Streamable HTTP MCP endpoint, gives its live tool
schemas and docstrings to the configured model for tool selection, validates
every proposed action with L1/L4 guardrails, and delegates final answer
synthesis to ``reasoning.py`` (three parallel voices plus one critic).  When
the configured endpoint is local and not already listening, the CLI starts
``src/mcp_server.py`` as a managed subprocess and stops it after the run.

Examples from the repository root:

    python src/agent.py
    python src/agent.py "Compare 2024 PM2.5 in France, Germany and Italy."
    python src/agent.py --no-hyde "What is the WHO annual PM2.5 guideline?"
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import ipaddress
import json
import os
import re
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from guardrails import (
    ACTION_RISK_MATRIX,
    BudgetExceeded,
    BudgetSnapshot,
    TokenBudget,
    l1_filter,
    l4_gate,
    sanitise_tool_result,
)
from reasoning import (
    ModelResponse,
    ReasoningEngine,
    ReasoningResult,
    format_reasoning_result,
)
from observability import Observability


ROOT = Path(__file__).resolve().parents[1]
AGENT_VERSION = "0.6.0"
_PROCESS_STARTED = time.perf_counter()
DEFAULT_QUESTION = (
    "Compare the 2024 PM2.5 sampling-point results for France, Germany and "
    "Italy against the WHO 2021 annual guideline, and explain what the "
    "comparison means."
)


class AgentError(RuntimeError):
    """Base class for controlled, user-facing agent failures."""


class ConfigurationError(AgentError):
    """Raised when a dependency or required environment setting is absent."""


class UnsupportedQuestion(AgentError):
    """Raised when a question falls outside this agent's documented scope."""


class ToolExecutionError(AgentError):
    """Raised when an MCP tool returns a controlled error payload."""


@dataclass
class _ManagedMCPServer:
    """A local HTTP MCP subprocess owned by one agent invocation."""

    process: subprocess.Popen[Any]

    async def stop(self) -> None:
        """Stop only the child that this invocation started."""

        if self.process.poll() is not None:
            return
        with suppress(OSError):
            self.process.terminate()
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self.process.wait),
                timeout=5.0,
            )
        except (asyncio.TimeoutError, subprocess.TimeoutExpired):
            with suppress(OSError):
                self.process.kill()
            await asyncio.to_thread(self.process.wait)


def _progress(message: str) -> None:
    """Write an immediate stage marker so a slow call is never a blank screen."""

    enabled = os.getenv("AGENT_PROGRESS", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return
    elapsed = time.perf_counter() - _PROCESS_STARTED
    print(f"[agent +{elapsed:6.1f}s] {message}", file=sys.stderr, flush=True)


def _configure_console_encoding() -> None:
    """Allow scientific Unicode such as PM₂.₅/NO₂ on Windows consoles."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError):
                pass


@dataclass(frozen=True)
class PlannedToolCall:
    name: str
    arguments: dict[str, Any]
    purpose: str


@dataclass(frozen=True)
class ToolCallResult:
    """One executed MCP action, kept separate from citable evidence IDs."""

    sequence: int
    tool_name: str
    arguments: dict[str, Any]
    purpose: str
    latency_ms: float
    audit_required: bool
    evidence_ids: tuple[str, ...]
    hyde_used: bool | None = None
    hyde_status: str | None = None
    hyde_model: str | None = None
    hyde_generated_characters: int = 0
    hyde_total_tokens: int = 0
    hyde_error: str | None = None


@dataclass(frozen=True)
class EvidenceItem:
    """One independently citable documentary or measurement result."""

    evidence_id: str
    evidence_type: str
    source_tool: str
    summary: str
    safe_text: str


@dataclass(frozen=True)
class ModelCallRecord:
    call_name: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


@dataclass(frozen=True)
class AgentRunResult:
    agent_version: str
    question: str
    planned_tools: tuple[PlannedToolCall, ...]
    tool_calls: tuple[ToolCallResult, ...]
    evidence: tuple[EvidenceItem, ...]
    reasoning: ReasoningResult
    model_calls: tuple[ModelCallRecord, ...]
    budget: BudgetSnapshot
    total_latency_ms: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _model_call_sort_key(record: ModelCallRecord) -> tuple[int, int | str]:
    """Keep concurrent model-call metrics in logical voice/critic order."""

    if record.call_name == "tool_selection":
        return (0, 0)
    match = re.fullmatch(r"synthesis_voice_(\d+)", record.call_name)
    if match:
        return (1, int(match.group(1)))
    if record.call_name == "critic":
        return (2, 0)
    return (3, record.call_name)


def _load_environment() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise ConfigurationError(
            "Missing dependency 'python-dotenv'. Install project requirements first."
        ) from exc
    load_dotenv(ROOT / ".env")


def _environment_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if value < 0:
        raise ConfigurationError(f"{name} cannot be negative")
    return value


def _environment_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    if value < 0:
        raise ConfigurationError(f"{name} cannot be negative")
    return value


def _token_budget_from_environment() -> TokenBudget:
    """Build one per-run safety budget; zero price/cost cap supports free tiers."""

    max_cost = _environment_float("AGENT_MAX_ESTIMATED_USD", 0.0)
    return TokenBudget(
        max_llm_calls=_environment_int("AGENT_MAX_LLM_CALLS", 6),
        max_tool_calls=_environment_int("AGENT_MAX_TOOL_CALLS", 8),
        max_input_tokens=_environment_int("AGENT_MAX_INPUT_TOKENS", 120_000),
        max_output_tokens=_environment_int("AGENT_MAX_OUTPUT_TOKENS", 8_000),
        max_cost_usd=None if max_cost == 0 else max_cost,
        input_cost_per_million=_environment_float(
            "AGENT_INPUT_USD_PER_MILLION", 0.0
        ),
        output_cost_per_million=_environment_float(
            "AGENT_OUTPUT_USD_PER_MILLION", 0.0
        ),
    )


def _contains(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _mentions_who(question: str) -> bool:
    """Recognise the organisation without treating the pronoun 'who' as a topic."""

    return bool(
        re.search(r"\bWHO\b", question)
        or re.search(r"\bworld health organi[sz]ation\b", question, re.IGNORECASE)
        or re.search(
            r"\bwho\s+(?:2021\s+)?(?:air[- ]quality\s+)?guidelines?\b",
            question,
            re.IGNORECASE,
        )
    )


def _explicit_measurement_years(question: str) -> list[int]:
    """Extract explicit structured-data years without blocking source dates.

    L4 remains authoritative for every model-proposed ``year`` argument.  This
    preflight is intentionally conservative: it rejects early only when the
    question clearly combines the supported country scope with measurement
    analysis. Documentary questions may legitimately mention publication,
    monitoring, guideline, directive, report, or target years.
    """

    country_scope = re.search(
        r"\b(?:france|french|germany|german|italy|italian|"
        r"all\s+three\s+countries)\b",
        question,
        flags=re.IGNORECASE,
    ) or re.search(r"\b(?:FR|DE|IT)\b", question)
    measurement_intent = re.search(
        r"\b(?:pm\s*2[.,]?5|pm25|no\s*2|no2|nitrogen dioxide|"
        r"measurements?|measured|sampling[- ]points?|stations?|"
        r"concentrations?|annual means?|medians?|quartiles?|coverage|"
        r"distributions?|compare|comparison|rank|highest|lowest)\b",
        question,
        flags=re.IGNORECASE,
    )
    if not country_scope or not measurement_intent:
        return []

    documentary_terms = {
        "who",
        "eea",
        "etc",
        "he",
        "eu",
        "directive",
        "regulation",
        "guideline",
        "guidelines",
        "law",
        "legal",
        "limit",
        "limits",
        "standard",
        "standards",
        "target",
        "targets",
        "report",
        "reports",
        "status",
        "methodology",
        "publication",
        "published",
    }
    years: list[int] = []
    for match in re.finditer(r"\b(?:19|20)\d{2}\b", question):
        year = int(match.group())
        # Keep the window narrow so an earlier documentary reference (for
        # example "EEA 2026 report") cannot accidentally relabel a later
        # unsupported country-measurement year such as 2023.
        before = re.findall(r"[A-Za-z]+", question[: match.start()].lower())[-2:]
        after = re.findall(r"[A-Za-z]+", question[match.end() :].lower())[:2]
        is_documentary_year = bool(
            documentary_terms.intersection(before)
            or documentary_terms.intersection(after)
        )
        if not is_documentary_year and year not in years:
            years.append(year)
    return years


def _is_air_quality_question(question: str) -> bool:
    return _mentions_who(question) or _contains(
        r"\b(air quality|air pollution|pollut(?:ant|ion)|pm\s*2|pm25|no\s*2|"
        r"nitrogen dioxide|particulate|eea|air-quality|directive)\b",
        question,
    )


def _validate_question_scope(question: str) -> None:
    """Apply domain/year safety limits without choosing any MCP tool."""

    if not _is_air_quality_question(question):
        raise UnsupportedQuestion(
            "This agent only supports European air-quality questions about "
            "PM2.5/NO2 documents and 2024 FR/DE/IT measurements."
        )
    unsupported_years = [
        year for year in _explicit_measurement_years(question) if year != 2024
    ]
    if unsupported_years:
        raise UnsupportedQuestion(
            "Structured measurements are available only for 2024; "
            f"requested data year(s): {unsupported_years}."
        )


def _preflight_question(question: str) -> str:
    """Apply local input controls before any model, MCP, or network activity."""

    _progress("Applying input guardrail")
    filtered = l1_filter(question, strict=True, max_characters=2_000)
    if not filtered.allowed:
        raise AgentError(f"L1 blocked the question: {filtered.reason}")
    _validate_question_scope(filtered.text)
    return filtered.text


TOOL_SELECTION_SYSTEM_PROMPT = """You are the tool-selection controller for a
European air-quality evidence agent. Read every supplied MCP tool description,
especially its 'Use when', 'Do NOT use', 'Returns', 'Prefer', and 'Example'
guidance. Select every tool needed to answer the user's question with grounded
evidence, including both documentary and measurement tools for mixed questions.

Rules:
- Return tool calls only; do not answer the user.
- Use only supplied tools and their JSON schemas.
- Do not invent missing countries, pollutants, years, or unsupported locations.
- Structured measurements cover only 2024, France/Germany/Italy, and PM2.5/NO2.
- Prefer one comparison call over separate country calls when comparing countries.
- Avoid duplicate or unnecessary calls.
"""


def _mcp_value(tool: Any, name: str, default: Any = None) -> Any:
    if isinstance(tool, Mapping):
        return tool.get(name, default)
    return getattr(tool, name, default)


def _openai_tool_specs(mcp_tools: Sequence[Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Convert allowlisted MCP definitions to OpenAI-compatible function tools."""

    specs: list[dict[str, Any]] = []
    descriptions: dict[str, str] = {}
    for tool in mcp_tools:
        name = str(_mcp_value(tool, "name", "")).strip()
        if name not in ACTION_RISK_MATRIX:
            continue
        description = str(_mcp_value(tool, "description", "") or "").strip()
        schema = _mcp_value(tool, "inputSchema", None)
        if schema is None:
            schema = _mcp_value(tool, "input_schema", None)
        if not isinstance(schema, Mapping):
            raise AgentError(f"MCP tool '{name}' has no valid input schema")
        if not description:
            raise AgentError(f"MCP tool '{name}' has no description/docstring")
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": dict(schema),
                },
            }
        )
        descriptions[name] = description
    if not specs:
        raise AgentError("MCP server exposed no allowlisted, documented tools")
    return specs, descriptions


class OpenAICompatibleModel:
    """Asynchronous adapter for Mistral and other OpenAI-compatible APIs."""

    def __init__(
        self,
        observability: Observability | None = None,
        budget: TokenBudget | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ConfigurationError(
                "Missing dependency 'openai'. Install project requirements first."
            ) from exc

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = (
            os.getenv("LLM_MODEL", "").strip()
            or os.getenv("HYDE_MODEL", "").strip()
        )
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY is missing from .env")
        if not self.model:
            raise ConfigurationError("LLM_MODEL (or HYDE_MODEL) is missing from .env")

        options: dict[str, Any] = {
            "api_key": api_key,
            "timeout": float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "90")),
            "max_retries": int(os.getenv("LLM_MAX_RETRIES", "1")),
        }
        base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        if base_url:
            options["base_url"] = base_url
        self.client = AsyncOpenAI(**options)
        self.observability = observability or Observability(None, "not configured")
        self.budget = budget or TokenBudget()
        self.records: list[ModelCallRecord] = []

    @staticmethod
    def _estimate_input_tokens(messages: Sequence[Mapping[str, str]]) -> int:
        # Provider-neutral fallback for APIs that omit token usage.
        characters = sum(len(str(message.get("content", ""))) for message in messages)
        return max(1, characters // 4 + 16 * len(messages))

    async def __call__(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        call_name: str,
    ) -> ModelResponse:
        estimated_input = self._estimate_input_tokens(messages)
        try:
            reservation = self.budget.reserve_llm_call(
                input_tokens=estimated_input,
                output_tokens=int(max_tokens),
            )
        except BudgetExceeded as exc:
            raise AgentError(str(exc)) from exc
        _progress(f"LLM {call_name}: started ({self.model})")
        with reservation:
            with self.observability.observation(
                as_type="generation",
                name=call_name,
                model=self.model,
                input=messages,
                metadata={
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                },
            ) as generation:
                started = time.perf_counter()
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                )
                latency_ms = round((time.perf_counter() - started) * 1_000, 2)
                content = (response.choices[0].message.content or "").strip()
                if not content:
                    raise AgentError(f"LLM call '{call_name}' returned an empty response")

                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "prompt_tokens", None)
                output_tokens = getattr(usage, "completion_tokens", None)
                if input_tokens is None:
                    input_tokens = estimated_input
                if output_tokens is None:
                    output_tokens = max(1, len(content) // 4)
                try:
                    reservation.commit(
                        input_tokens=int(input_tokens),
                        output_tokens=int(output_tokens),
                    )
                except BudgetExceeded as exc:
                    raise AgentError(str(exc)) from exc
                self.observability.update(
                    generation,
                    output=content,
                    usage_details={
                        "prompt_tokens": int(input_tokens),
                        "completion_tokens": int(output_tokens),
                        "total_tokens": int(input_tokens) + int(output_tokens),
                    },
                    metadata={"latency_ms": latency_ms},
                )
                record = ModelCallRecord(
                    call_name=call_name,
                    model=self.model,
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    latency_ms=latency_ms,
                )
                self.records.append(record)
                _progress(f"LLM {call_name}: completed in {latency_ms / 1_000:.2f}s")
                return ModelResponse(
                    content=content,
                    model=self.model,
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    latency_ms=latency_ms,
                )

    async def select_tools(
        self,
        *,
        question: str,
        mcp_tools: Sequence[Any],
        use_hyde: bool,
    ) -> list[PlannedToolCall]:
        """Ask the model to choose calls from live MCP schemas and docstrings."""

        tools, descriptions = _openai_tool_specs(mcp_tools)
        messages = [
            {"role": "system", "content": TOOL_SELECTION_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        estimated_input = self._estimate_input_tokens(messages) + max(
            1,
            len(json.dumps(tools, ensure_ascii=False)) // 4,
        )
        try:
            reservation = self.budget.reserve_llm_call(
                input_tokens=estimated_input,
                output_tokens=800,
            )
        except BudgetExceeded as exc:
            raise AgentError(str(exc)) from exc
        _progress(f"LLM tool_selection: started ({self.model})")
        with (
            reservation,
            self.observability.observation(
                as_type="generation",
                name="tool_selection",
                model=self.model,
                input={"messages": messages, "tools": tools},
                metadata={
                    "tool_choice": "any",
                    "parallel_tool_calls": True,
                    "available_tools": sorted(descriptions),
                },
            ) as generation,
        ):
            started = time.perf_counter()
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="any",  # Mistral: force one or more model-selected tools.
                parallel_tool_calls=True,
                temperature=0.0,
                max_tokens=800,
            )
            latency_ms = round((time.perf_counter() - started) * 1_000, 2)
            if not response.choices:
                raise AgentError("LLM tool selection returned no choices")
            message = response.choices[0].message
            raw_calls = list(getattr(message, "tool_calls", None) or [])
            if not raw_calls:
                raise UnsupportedQuestion(
                    "The model did not select an evidence tool for this question."
                )
            if len(raw_calls) > 8:
                raise AgentError("LLM proposed more than the maximum eight tool calls")

            plan: list[PlannedToolCall] = []
            seen: set[tuple[str, str]] = set()
            traced_calls: list[dict[str, Any]] = []
            for raw_call in raw_calls:
                function = getattr(raw_call, "function", None)
                name = str(getattr(function, "name", "")).strip()
                if name not in descriptions:
                    raise AgentError(f"LLM selected unknown or blocked tool '{name}'")
                raw_arguments = getattr(function, "arguments", "{}")
                try:
                    arguments = (
                        json.loads(raw_arguments)
                        if isinstance(raw_arguments, str)
                        else raw_arguments
                    )
                except json.JSONDecodeError as exc:
                    raise AgentError(
                        f"LLM returned invalid JSON arguments for '{name}'"
                    ) from exc
                if not isinstance(arguments, dict):
                    raise AgentError(f"LLM arguments for '{name}' must be an object")
                if name == "search_air_quality_evidence":
                    # The CLI switch is an execution policy; the model still
                    # decides whether retrieval itself is needed.
                    arguments["use_hyde"] = bool(use_hyde)

                fingerprint = (
                    name,
                    json.dumps(arguments, ensure_ascii=False, sort_keys=True),
                )
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                first_line = next(
                    (line.strip() for line in descriptions[name].splitlines() if line.strip()),
                    descriptions[name],
                )
                plan.append(PlannedToolCall(name, arguments, first_line))
                traced_calls.append(
                    {
                        "id": str(getattr(raw_call, "id", "")),
                        "name": name,
                        "arguments": arguments,
                    }
                )

            if not plan:
                raise AgentError("LLM tool selection produced only duplicate calls")

            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)
            if input_tokens is None:
                input_tokens = estimated_input
            if output_tokens is None:
                output_tokens = max(
                    1,
                    len(json.dumps(traced_calls, ensure_ascii=False)) // 4,
                )
            try:
                reservation.commit(
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                )
            except BudgetExceeded as exc:
                raise AgentError(str(exc)) from exc
            self.observability.update(
                generation,
                output={"tool_calls": traced_calls},
                usage_details={
                    "prompt_tokens": int(input_tokens),
                    "completion_tokens": int(output_tokens),
                    "total_tokens": int(input_tokens) + int(output_tokens),
                },
                metadata={"latency_ms": latency_ms},
            )
            self.records.append(
                ModelCallRecord(
                    call_name="tool_selection",
                    model=self.model,
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    latency_ms=latency_ms,
                )
            )
            _progress(
                "LLM tool_selection: selected "
                + ", ".join(call.name for call in plan)
                + f" in {latency_ms / 1_000:.2f}s"
            )
            return plan


def _tool_content_text(result: Any) -> str:
    parts = [
        block.text
        for block in getattr(result, "content", [])
        if isinstance(getattr(block, "text", None), str)
    ]
    return "\n".join(parts).strip()


def _trace_tool_output(tool_name: str, data: Any) -> Any:
    """Keep traces useful without duplicating every long RAG passage."""

    if tool_name != "search_air_quality_evidence" or not isinstance(data, dict):
        return data
    return {
        "query": data.get("query"),
        "hyde_requested": data.get("hyde_requested"),
        "hyde_status": data.get("hyde_status"),
        "hyde_used": data.get("hyde_used"),
        "hyde_model": data.get("hyde_model"),
        "hyde_generated_characters": data.get("hyde_generated_characters"),
        "hyde_error": data.get("hyde_error"),
        "timings_ms": data.get("timings_ms"),
        "results": [
            {
                "rank": row.get("rank"),
                "parent_id": row.get("parent_id"),
                "title": row.get("title"),
                "page_start": row.get("page_start"),
                "page_end": row.get("page_end"),
            }
            for row in data.get("results", [])
            if isinstance(row, dict)
        ],
    }


def _expand_citable_evidence(
    call: PlannedToolCall,
    data: Any,
    *,
    next_document_number: int,
    next_measurement_number: int,
) -> tuple[list[EvidenceItem], int, int]:
    """Turn one tool payload into D*/M* items without citing the MCP action."""

    items: list[EvidenceItem] = []
    if call.name == "search_air_quality_evidence":
        if not isinstance(data, Mapping) or not isinstance(data.get("results"), list):
            raise ToolExecutionError(
                "search_air_quality_evidence returned no document result list"
            )
        for row in data["results"]:
            if not isinstance(row, Mapping):
                raise ToolExecutionError(
                    "search_air_quality_evidence returned a malformed document result"
                )
            evidence_id = f"D{next_document_number}"
            next_document_number += 1
            title = str(row.get("title") or row.get("doc_id") or "Retrieved document")
            page_start = row.get("page_start")
            page_end = row.get("page_end")
            page_note = (
                f", pages {page_start}-{page_end}"
                if page_start is not None
                else ""
            )
            items.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    evidence_type="retrieved_document",
                    source_tool=call.name,
                    summary=f"{title}{page_note}",
                    safe_text=sanitise_tool_result(
                        json.dumps(dict(row), ensure_ascii=False, indent=2)
                    ),
                )
            )
        if not items:
            raise ToolExecutionError("Document retrieval returned zero citable results")
        return items, next_document_number, next_measurement_number

    evidence_id = f"M{next_measurement_number}"
    next_measurement_number += 1
    items.append(
        EvidenceItem(
            evidence_id=evidence_id,
            evidence_type="structured_measurement",
            source_tool=call.name,
            summary=call.purpose,
            safe_text=sanitise_tool_result(
                json.dumps(data, ensure_ascii=False, indent=2)
            ),
        )
    )
    return items, next_document_number, next_measurement_number


async def _collect_evidence(
    session: Any,
    plan: Sequence[PlannedToolCall],
    observability: Observability | None = None,
    budget: TokenBudget | None = None,
) -> tuple[list[ToolCallResult], list[EvidenceItem]]:
    observer = observability or Observability(None, "not configured")
    run_budget = budget or TokenBudget()
    tool_calls: list[ToolCallResult] = []
    evidence: list[EvidenceItem] = []
    next_document_number = 1
    next_measurement_number = 1
    tool_timeout = float(os.getenv("MCP_TOOL_TIMEOUT_SECONDS", "120"))
    for index, call in enumerate(plan, start=1):
        decision = l4_gate(call.name, call.arguments)
        if not decision.allowed:
            raise AgentError(f"L4 blocked {call.name}: {decision.reason}")

        try:
            # Consume immediately before dispatch: failed/time-out calls still
            # count as attempted external actions.
            run_budget.reserve_tool_call().commit()
        except BudgetExceeded as exc:
            raise AgentError(str(exc)) from exc

        _progress(f"MCP tool {index}/{len(plan)} {call.name}: started")
        hyde_used: bool | None = None
        hyde_status: str | None = None
        hyde_model: str | None = None
        hyde_generated_characters = 0
        hyde_total_tokens = 0
        hyde_error: str | None = None
        with observer.observation(
            as_type="tool",
            name=f"mcp.{call.name}",
            input=dict(call.arguments),
            metadata={"audit_required": decision.audit_required},
        ) as tool_span:
            started = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    session.call_tool(call.name, call.arguments),
                    timeout=tool_timeout,
                )
            except asyncio.TimeoutError as exc:
                raise ToolExecutionError(
                    f"MCP tool '{call.name}' exceeded {tool_timeout:.0f}s. "
                    "Check the MCP server terminal for the active retrieval stage."
                ) from exc
            latency_ms = round((time.perf_counter() - started) * 1_000, 2)
            raw = _tool_content_text(result)
            if not raw:
                raise ToolExecutionError(f"MCP tool '{call.name}' returned no text")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ToolExecutionError(
                    f"MCP tool '{call.name}' returned invalid JSON"
                ) from exc
            if payload.get("status") != "ok":
                raise ToolExecutionError(
                    f"{call.name}: {payload.get('message', 'unknown controlled error')}"
                )

            data = payload.get("data", payload)
            (
                new_evidence,
                next_document_number,
                next_measurement_number,
            ) = _expand_citable_evidence(
                call,
                data,
                next_document_number=next_document_number,
                next_measurement_number=next_measurement_number,
            )
            evidence.extend(new_evidence)
            evidence_ids = tuple(item.evidence_id for item in new_evidence)
            observer.update(
                tool_span,
                output=_trace_tool_output(call.name, data),
                metadata={
                    "audit_required": decision.audit_required,
                    "latency_ms": latency_ms,
                    "evidence_ids": evidence_ids,
                },
            )

            # HyDE executes inside the retrieval tool.  The MCP response reports
            # its real usage/duration, which is attached as a nested generation.
            if call.name == "search_air_quality_evidence" and isinstance(data, dict):
                hyde_used = bool(data.get("hyde_used"))
                hyde_status = str(data.get("hyde_status") or "") or None
                hyde_model = str(data.get("hyde_model") or "") or None
                hyde_generated_characters = int(
                    data.get("hyde_generated_characters") or 0
                )
                hyde_error = data.get("hyde_error")
                hyde_usage = data.get("hyde_usage") or {}
                usage_details = {
                    "prompt_tokens": int(hyde_usage.get("input_tokens") or 0),
                    "completion_tokens": int(hyde_usage.get("output_tokens") or 0),
                    "total_tokens": int(hyde_usage.get("total_tokens") or 0),
                }
                hyde_total_tokens = usage_details["total_tokens"]
                with observer.observation(
                    as_type="generation",
                    name="hyde",
                    model=(
                        str(data.get("hyde_model") or "").strip()
                        or os.getenv("HYDE_MODEL", "").strip()
                        or os.getenv("LLM_MODEL", "").strip()
                    ),
                    input={"query": call.arguments.get("query")},
                    metadata={
                        "reported_by": "MCP retrieval server",
                        "status": data.get("hyde_status"),
                        "generated_characters": data.get(
                            "hyde_generated_characters"
                        ),
                        "reported_latency_ms": (data.get("timings_ms") or {}).get("hyde_ms"),
                    },
                ) as hyde_span:
                    update: dict[str, Any] = {
                        "output": {
                            "used": hyde_used,
                            "error": hyde_error,
                        }
                    }
                    if usage_details["total_tokens"]:
                        update["usage_details"] = usage_details
                    observer.update(hyde_span, **update)
                if hyde_used:
                    try:
                        run_budget.record_llm_call(
                            input_tokens=usage_details["prompt_tokens"],
                            output_tokens=usage_details["completion_tokens"],
                        )
                    except BudgetExceeded as exc:
                        raise AgentError(str(exc)) from exc

        tool_calls.append(
            ToolCallResult(
                sequence=index,
                tool_name=call.name,
                arguments=dict(call.arguments),
                purpose=call.purpose,
                latency_ms=latency_ms,
                audit_required=decision.audit_required,
                evidence_ids=evidence_ids,
                hyde_used=hyde_used,
                hyde_status=hyde_status,
                hyde_model=hyde_model,
                hyde_generated_characters=hyde_generated_characters,
                hyde_total_tokens=hyde_total_tokens,
                hyde_error=hyde_error,
            )
        )
        _progress(f"MCP tool {call.name}: completed in {latency_ms / 1_000:.2f}s")
    return tool_calls, evidence


def _assemble_evidence_context(evidence: Sequence[EvidenceItem]) -> str:
    blocks = []
    for item in evidence:
        blocks.append(
            f"[{item.evidence_id}] EVIDENCE TYPE: {item.evidence_type}\n"
            f"SUMMARY: {item.summary}\n"
            f"SOURCE TOOL (PROVENANCE ONLY; DO NOT CITE): {item.source_tool}\n"
            "BEGIN UNTRUSTED EVIDENCE\n"
            f"{item.safe_text}\n"
            "END UNTRUSTED EVIDENCE"
        )
    return "\n\n".join(blocks)


class AirQualityAgent:
    """Guarded orchestrator operating over an already-initialised MCP session."""

    def __init__(
        self,
        model: OpenAICompatibleModel,
        observability: Observability | None = None,
        budget: TokenBudget | None = None,
    ) -> None:
        self.model = model
        self.observability = observability or Observability(None, "not configured")
        self.budget = budget or getattr(model, "budget", None) or TokenBudget()
        if hasattr(model, "budget"):
            model.budget = self.budget
        self.reasoning = ReasoningEngine(model, k=3)

    async def run(
        self,
        question: str,
        session: Any,
        *,
        use_hyde: bool = True,
    ) -> AgentRunResult:
        started = time.perf_counter()
        filtered_text = _preflight_question(question)
        _progress("Discovering MCP tools")
        try:
            listed = await asyncio.wait_for(
                session.list_tools(),
                timeout=float(os.getenv("MCP_CONNECT_TIMEOUT_SECONDS", "15")),
            )
        except asyncio.TimeoutError as exc:
            raise AgentError("MCP tools/list timed out; verify that the server is healthy") from exc
        available = {tool.name for tool in listed.tools}
        _progress(f"MCP discovery complete: {len(available)} tools available")

        _progress("Asking the LLM to select tools from MCP docstrings")
        plan = await self.model.select_tools(
            question=filtered_text,
            mcp_tools=listed.tools,
            use_hyde=use_hyde,
        )
        missing = {call.name for call in plan} - available
        if missing:
            raise AgentError(f"MCP server is missing required tools: {sorted(missing)}")
        _progress("Model plan: " + ", ".join(call.name for call in plan))

        tool_calls, evidence = await _collect_evidence(
            session,
            plan,
            self.observability,
            self.budget,
        )
        context = _assemble_evidence_context(evidence)
        allowed_evidence_ids = tuple(item.evidence_id for item in evidence)
        _progress("Starting self-consistency reasoning (3 parallel syntheses + critic)")
        reasoning = await self.reasoning.synthesize(
            filtered_text,
            context,
            allowed_evidence_ids=allowed_evidence_ids,
        )
        _progress("Critic completed; final answer ready")
        total_latency_ms = round((time.perf_counter() - started) * 1_000, 2)
        return AgentRunResult(
            agent_version=AGENT_VERSION,
            question=filtered_text,
            planned_tools=tuple(plan),
            tool_calls=tuple(tool_calls),
            evidence=tuple(evidence),
            reasoning=reasoning,
            model_calls=tuple(sorted(self.model.records, key=_model_call_sort_key)),
            budget=self.budget.snapshot(),
            total_latency_ms=total_latency_ms,
        )


def _local_mcp_spawn_target(mcp_url: str) -> tuple[str, int] | None:
    """Return a safe local bind target when this project can serve ``mcp_url``.

    Auto-start is deliberately limited to plain HTTP loopback URLs using the
    FastMCP server's fixed ``/mcp`` route.  Remote and HTTPS endpoints remain
    externally managed and are never replaced by a local process.
    """

    try:
        parsed = urlsplit(mcp_url)
        hostname = (parsed.hostname or "").rstrip(".").lower()
        port = parsed.port
    except ValueError as exc:
        raise ConfigurationError(f"Invalid MCP_SERVER_URL: {exc}") from exc

    if parsed.scheme.lower() != "http" or not hostname:
        return None
    if parsed.path.rstrip("/") != "/mcp" or parsed.query or parsed.fragment:
        return None

    is_loopback = hostname == "localhost"
    if not is_loopback:
        with suppress(ValueError):
            is_loopback = ipaddress.ip_address(hostname).is_loopback
    if not is_loopback:
        return None

    bind_host = "127.0.0.1" if hostname == "localhost" else hostname
    return bind_host, port or 80


async def _endpoint_is_listening(
    host: str,
    port: int,
    *,
    timeout_seconds: float = 0.35,
) -> bool:
    """Probe a loopback TCP endpoint without sending an invalid MCP request."""

    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout_seconds,
        )
    except (asyncio.TimeoutError, OSError):
        return False

    writer.close()
    with suppress(OSError):
        await writer.wait_closed()
    return True


async def _maybe_start_local_mcp_server(mcp_url: str) -> _ManagedMCPServer | None:
    """Start the project MCP server only for an unavailable loopback endpoint."""

    target = _local_mcp_spawn_target(mcp_url)
    if target is None:
        return None
    host, port = target
    if await _endpoint_is_listening(host, port):
        _progress("Using the MCP server already listening at the configured endpoint")
        return None

    server_path = ROOT / "src" / "mcp_server.py"
    if not server_path.is_file():
        raise ConfigurationError(f"Local MCP server entry point is missing: {server_path}")

    server_env = dict(os.environ)
    server_env["MCP_HOST"] = host
    server_env["MCP_PORT"] = str(port)
    _progress(
        "Local MCP endpoint is unavailable; starting the Streamable HTTP server"
    )
    try:
        process = subprocess.Popen(
            [sys.executable, str(server_path)],
            cwd=str(ROOT),
            env=server_env,
            stdin=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise AgentError(f"Could not start the local MCP server: {exc}") from exc

    managed = _ManagedMCPServer(process)
    try:
        timeout_seconds = _environment_float("MCP_AUTO_START_TIMEOUT_SECONDS", 20.0)
        if timeout_seconds == 0:
            raise ConfigurationError(
                "MCP_AUTO_START_TIMEOUT_SECONDS must be greater than zero"
            )
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            # Check the endpoint first so a concurrent agent that won the bind
            # race can still supply the shared server.
            if await _endpoint_is_listening(host, port):
                _progress(f"Auto-started MCP server is ready (PID {process.pid})")
                return managed
            return_code = process.poll()
            if return_code is not None:
                raise AgentError(
                    "The local MCP server exited during startup with status "
                    f"{return_code}. Run 'python src/mcp_server.py' to inspect its logs."
                )
            await asyncio.sleep(0.05)
        raise AgentError(
            f"The local MCP server did not become ready within {timeout_seconds:g}s"
        )
    except BaseException:
        await managed.stop()
        raise


async def run_with_mcp_server(
    question: str,
    *,
    use_hyde: bool = True,
) -> AgentRunResult:
    """Connect over HTTP, auto-starting this project's local server if needed."""

    # Reject locally before opening Streamable HTTP. Besides saving work, this
    # prevents cancellation of MCP's initialized notification from surfacing as
    # a noisy Starlette ClientDisconnect on immediately rejected questions.
    clean_question = _preflight_question(question)

    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as exc:
        raise ConfigurationError(
            "Missing Streamable HTTP support from 'mcp'. Use Python 3.11+ and "
            "install a stable mcp 1.x release."
        ) from exc

    mcp_url = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp").strip()
    if not mcp_url.startswith(("http://", "https://")):
        raise ConfigurationError("MCP_SERVER_URL must be an http:// or https:// URL")
    observer = Observability.from_environment()
    _progress(f"Langfuse {observer.status()}")
    budget = _token_budget_from_environment()
    model = OpenAICompatibleModel(observer, budget)
    agent = AirQualityAgent(model, observer, budget)

    managed_server: _ManagedMCPServer | None = None
    try:
        managed_server = await _maybe_start_local_mcp_server(mcp_url)
        _progress(f"Connecting to MCP server: {mcp_url}")
        with observer.observation(
            as_type="agent",
            name="air-quality-agent.run",
            input={"question": question},
            metadata={
                "agent_version": AGENT_VERSION,
                "mcp_server_url": mcp_url,
                "hyde_enabled": bool(use_hyde),
            },
        ) as agent_span:
            async with streamable_http_client(mcp_url) as (
                read_stream,
                write_stream,
                _,
            ):
                _progress("HTTP transport connected; initializing MCP session")
                session_options: dict[str, Any] = {}
                if "read_timeout_seconds" in inspect.signature(ClientSession).parameters:
                    session_options["read_timeout_seconds"] = timedelta(
                        seconds=float(os.getenv("MCP_READ_TIMEOUT_SECONDS", "120"))
                    )
                async with ClientSession(
                    read_stream,
                    write_stream,
                    **session_options,
                ) as session:
                    try:
                        await asyncio.wait_for(
                            session.initialize(),
                            timeout=float(os.getenv("MCP_CONNECT_TIMEOUT_SECONDS", "15")),
                        )
                    except asyncio.TimeoutError as exc:
                        raise AgentError(
                            f"MCP initialization timed out at {mcp_url}. "
                            "Verify that the configured endpoint is a healthy MCP server."
                        ) from exc
                    _progress("MCP session initialized")
                    result = await agent.run(clean_question, session, use_hyde=use_hyde)
                    observer.update(
                        agent_span,
                        output={
                            "verdict": result.reasoning.critic.verdict,
                            "agreement": result.reasoning.critic.agreement,
                            "tools": [call.name for call in result.planned_tools],
                            "allowed_evidence_ids": [
                                item.evidence_id for item in result.evidence
                            ],
                        },
                        metadata={
                            "agent_version": AGENT_VERSION,
                            "mcp_server_url": mcp_url,
                            "hyde_enabled": bool(use_hyde),
                            "total_latency_ms": result.total_latency_ms,
                            "budget": result.budget.as_dict(),
                        },
                    )
                    return result
    except (AgentError, ValueError):
        raise
    except Exception as exc:
        raise AgentError(
            f"Agent run failed while using {mcp_url}: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if managed_server is not None:
            _progress("Stopping the auto-started local MCP server")
            await managed_server.stop()
        _progress("Flushing Langfuse traces")
        observer.flush()


async def run_with_local_mcp(question: str, *, use_hyde: bool = True) -> AgentRunResult:
    """Backward-compatible name for the managed Streamable HTTP runner."""

    return await run_with_mcp_server(question, use_hyde=use_hyde)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the guarded European air-quality evidence agent."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Air-quality question. A complete demonstration question is used if omitted.",
    )
    parser.add_argument(
        "--no-hyde",
        action="store_true",
        help="Disable the HyDE retrieval call for a faster/offline retrieval smoke test.",
    )
    parser.add_argument(
        "--mcp-url",
        help="Override MCP_SERVER_URL for this run (default: http://127.0.0.1:8000/mcp).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete machine-readable run, including all three drafts.",
    )
    parser.add_argument(
        "--show-drafts",
        action="store_true",
        help="Print all three candidate syntheses after the critic-checked answer.",
    )
    return parser.parse_args()


def _print_human_result(result: AgentRunResult, *, show_drafts: bool) -> None:
    print(f"Question: {result.question}\n")
    print("MCP tool calls (not citation IDs):")
    for tool_call in result.tool_calls:
        produced = ", ".join(tool_call.evidence_ids)
        print(
            f"- call {tool_call.sequence}: {tool_call.tool_name} "
            f"-> evidence {produced}"
        )
        if tool_call.hyde_used is not None:
            if tool_call.hyde_used:
                model_note = f" via {tool_call.hyde_model}" if tool_call.hyde_model else ""
                print(
                    f"  HyDE: generated successfully{model_note}; "
                    f"{tool_call.hyde_generated_characters} characters, "
                    f"{tool_call.hyde_total_tokens} tokens"
                )
            elif tool_call.arguments.get("use_hyde") is False:
                print("  HyDE: disabled by --no-hyde")
            else:
                note = tool_call.hyde_error or "provider returned no expansion"
                print(f"  HyDE: fallback to original query ({note})")
    print("\nAllowed evidence citations:")
    for item in result.evidence:
        print(f"- [{item.evidence_id}] {item.summary}")
    print()
    print(format_reasoning_result(result.reasoning))
    if show_drafts:
        print("\nINDEPENDENT SYNTHESIS DRAFTS")
        for draft in result.reasoning.drafts:
            print(f"\n--- DRAFT {draft.index} ---\n{draft.content}")
    print("\nRUN METRICS")
    print(f"- Agent version: {result.agent_version}")
    print(f"- Total latency: {result.total_latency_ms / 1_000:.2f} s")
    hyde_calls = sum(call.hyde_used is True for call in result.tool_calls)
    print(f"- LLM calls: {result.budget.llm_calls} total")
    print("- Synthesis calls: exactly 3 (run in parallel)")
    print(
        "- Other LLM calls: 1 tool selection + 1 critic"
        + (" + 1 HyDE" if hyde_calls else "")
    )
    print(f"- MCP tool calls: {len(result.tool_calls)}")
    print(
        f"- Token budget: {result.budget.total_tokens} tokens, "
        f"${result.budget.cost_usd:.6f} estimated"
    )
    print("\nDisclosure: This is an AI-generated evidence synthesis; verify important decisions against the cited sources.")


def main() -> int:
    _configure_console_encoding()
    args = _parse_args()
    question = " ".join(args.question).strip()
    if not question:
        question = DEFAULT_QUESTION
        _progress(
            "No question supplied; running the documented demonstration question"
        )
    try:
        _load_environment()
        if args.mcp_url:
            os.environ["MCP_SERVER_URL"] = args.mcp_url.strip()
        _progress("Configuration loaded")
        result = asyncio.run(
            run_with_mcp_server(question, use_hyde=not args.no_hyde)
        )
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (AgentError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    else:
        _print_human_result(result, show_drafts=args.show_drafts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AGENT_VERSION",
    "DEFAULT_QUESTION",
    "AgentRunResult",
    "AirQualityAgent",
    "ConfigurationError",
    "EvidenceItem",
    "ModelCallRecord",
    "OpenAICompatibleModel",
    "PlannedToolCall",
    "ToolCallResult",
    "run_with_local_mcp",
    "run_with_mcp_server",
]
