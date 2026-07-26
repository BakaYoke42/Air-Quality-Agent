"""Checkpointed ten-run end-to-end operational evaluation.

Unlike ``evaluate_retrieval.py``, every case goes through the production agent:
live MCP discovery, LLM tool selection, L1/L4 controls, tool execution,
self-consistency k=3, citation validation, and the critic. Expected tools are
labels used only after the run; they are never supplied to the planner.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent import (  # noqa: E402
    AgentRunResult,
    _maybe_start_local_mcp_server,
    run_with_mcp_server,
)
from reasoning import (  # noqa: E402
    extract_evidence_citations,
    validate_evidence_citations,
)


AGENT_TOOL_NAMES = frozenset(
    {
        "search_air_quality_evidence",
        "get_country_air_quality",
        "compare_countries",
        "find_station_extremes",
    }
)


class OperationalEvaluationError(RuntimeError):
    """Raised for invalid cases or an unusable evaluation configuration."""


def _load_environment() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise OperationalEvaluationError(
            "Install requirements before running the operational benchmark."
        ) from exc
    load_dotenv(ROOT / ".env")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise OperationalEvaluationError(f"Missing JSONL input: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OperationalEvaluationError(
                    f"Invalid JSON in {path}, line {line_number}."
                ) from exc
            if not isinstance(value, dict):
                raise OperationalEvaluationError(
                    f"Expected an object in {path}, line {line_number}."
                )
            records.append(value)
    return records


def load_cases(path: Path) -> list[dict[str, Any]]:
    records = _load_jsonl(path)
    if len(records) != 10:
        raise OperationalEvaluationError(
            f"The rubric benchmark requires exactly 10 cases; found {len(records)}."
        )
    required = {"id", "question", "expected_capability", "expected_tools"}
    identifiers: list[str] = []
    for record in records:
        missing = required - set(record)
        if missing:
            raise OperationalEvaluationError(
                f"Operational case is missing fields: {sorted(missing)}."
            )
        identifier = str(record["id"]).strip()
        question = str(record["question"]).strip()
        expected_capability = str(record["expected_capability"]).strip()
        expected_tools = record["expected_tools"]
        if not identifier or not question or not expected_capability:
            raise OperationalEvaluationError(
                "Case ID, question, and expected capability cannot be empty."
            )
        if (
            not isinstance(expected_tools, list)
            or not expected_tools
            or not all(
                isinstance(tool, str) and tool.strip() for tool in expected_tools
            )
        ):
            raise OperationalEvaluationError(
                f"{identifier} expected_tools must be a non-empty string list."
            )
        if len(expected_tools) != len(set(expected_tools)):
            raise OperationalEvaluationError(
                f"{identifier} expected_tools must not contain duplicates."
            )
        unknown_tools = set(expected_tools) - AGENT_TOOL_NAMES
        if unknown_tools:
            raise OperationalEvaluationError(
                f"{identifier} contains unknown expected tools: "
                f"{sorted(unknown_tools)}."
            )
        identifiers.append(identifier)
    if len(identifiers) != len(set(identifiers)):
        raise OperationalEvaluationError("Operational case IDs must be unique.")
    covered_tools = {
        tool for record in records for tool in record["expected_tools"]
    }
    missing_tool_coverage = AGENT_TOOL_NAMES - covered_tools
    if missing_tool_coverage:
        raise OperationalEvaluationError(
            "Operational cases must cover all production tools; missing "
            f"{sorted(missing_tool_coverage)}."
        )
    if not any(len(record["expected_tools"]) > 1 for record in records):
        raise OperationalEvaluationError(
            "Operational cases must include at least one mixed-tool question."
        )
    return records


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _existing_records(
    path: Path,
    cases_by_id: dict[str, dict[str, Any]],
    *,
    retry_errors: bool,
    use_hyde: bool,
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = _load_jsonl(path)
    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        identifier = str(record.get("id", ""))
        if identifier not in cases_by_id:
            raise OperationalEvaluationError(
                f"Checkpoint contains unknown case ID: {identifier!r}."
            )
        if identifier in seen:
            raise OperationalEvaluationError(
                f"Checkpoint contains duplicate case ID: {identifier!r}."
            )
        if record.get("question") != cases_by_id[identifier]["question"]:
            raise OperationalEvaluationError(
                f"Checkpoint question changed for {identifier}; use a new output path."
            )
        if record.get("expected_capability") != cases_by_id[identifier][
            "expected_capability"
        ]:
            raise OperationalEvaluationError(
                f"Checkpoint capability changed for {identifier}; "
                "use a new output path."
            )
        if record.get("expected_tools") != cases_by_id[identifier]["expected_tools"]:
            raise OperationalEvaluationError(
                f"Checkpoint expected tools changed for {identifier}; "
                "use a new output path."
            )
        if record.get("status") not in {"ok", "error"}:
            raise OperationalEvaluationError(
                f"Checkpoint has invalid status for {identifier!r}."
            )
        if record.get("hyde_requested") is not bool(use_hyde):
            raise OperationalEvaluationError(
                f"Checkpoint HyDE policy differs for {identifier}; "
                "use a new output path."
            )
        seen.add(identifier)
        if retry_errors and record.get("status") != "ok":
            continue
        accepted.append(record)
    return accepted


def _tool_record(tool_call: Any) -> dict[str, Any]:
    return {
        "sequence": tool_call.sequence,
        "tool_name": tool_call.tool_name,
        "arguments": tool_call.arguments,
        "purpose": tool_call.purpose,
        "latency_ms": tool_call.latency_ms,
        "audit_required": tool_call.audit_required,
        "evidence_ids": list(tool_call.evidence_ids),
        "hyde_used": tool_call.hyde_used,
        "hyde_status": tool_call.hyde_status,
        "hyde_model": tool_call.hyde_model,
        "hyde_generated_characters": tool_call.hyde_generated_characters,
        "hyde_total_tokens": tool_call.hyde_total_tokens,
        "hyde_error": tool_call.hyde_error,
    }


def _successful_record(
    case: dict[str, Any],
    result: AgentRunResult,
    *,
    use_hyde: bool,
) -> dict[str, Any]:
    expected_tools = list(case["expected_tools"])
    planned_tools = [call.name for call in result.planned_tools]
    executed_tools = [call.tool_name for call in result.tool_calls]
    allowed_ids = [item.evidence_id for item in result.evidence]
    cited_ids = extract_evidence_citations(result.reasoning.critic.final_answer)
    citation_allowlist_valid, unsupported = validate_evidence_citations(
        result.reasoning.critic.final_answer,
        allowed_ids,
    )
    citations_present = bool(cited_ids)
    budget = result.budget.as_dict()
    budget["total_tokens"] = result.budget.total_tokens
    return {
        "id": case["id"],
        "question": case["question"],
        "expected_capability": case["expected_capability"],
        "expected_tools": expected_tools,
        "status": "ok",
        "hyde_requested": bool(use_hyde),
        "agent_version": result.agent_version,
        "planned_tools": [
            {
                "name": call.name,
                "arguments": call.arguments,
                "purpose": call.purpose,
            }
            for call in result.planned_tools
        ],
        "executed_tools": [_tool_record(call) for call in result.tool_calls],
        "actual_tool_names": executed_tools,
        "planned_tool_set_exact_match": set(planned_tools) == set(expected_tools),
        "executed_tool_set_exact_match": set(executed_tools)
        == set(expected_tools),
        # Backward-compatible alias for the autonomous planner decision.
        "tool_set_exact_match": set(planned_tools) == set(expected_tools),
        "critic": {
            "verdict": result.reasoning.critic.verdict,
            "agreement": result.reasoning.critic.agreement,
            "selected_draft": result.reasoning.critic.selected_draft,
            "issues": list(result.reasoning.critic.issues),
        },
        "allowed_evidence_ids": allowed_ids,
        "cited_evidence_ids": list(cited_ids),
        "citation_ids_present": citations_present,
        "citation_allowlist_valid": citation_allowlist_valid,
        "citation_ids_valid": citations_present and citation_allowlist_valid,
        "unsupported_citation_ids": list(unsupported),
        "final_answer": result.reasoning.critic.final_answer,
        "latency_seconds": round(result.total_latency_ms / 1_000, 3),
        "budget": budget,
        "model_calls": [asdict(record) for record in result.model_calls],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _error_record(
    case: dict[str, Any],
    exc: Exception,
    *,
    use_hyde: bool,
) -> dict[str, Any]:
    return {
        "id": case["id"],
        "question": case["question"],
        "expected_capability": case["expected_capability"],
        "expected_tools": list(case["expected_tools"]),
        "status": "error",
        "hyde_requested": bool(use_hyde),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_summary(
    records: Sequence[dict[str, Any]],
    *,
    use_hyde: bool,
) -> dict[str, Any]:
    successful = [record for record in records if record.get("status") == "ok"]
    failed = [record for record in records if record.get("status") == "error"]
    tool_distribution: Counter[str] = Counter()
    planner_distribution: Counter[str] = Counter()
    critic_distribution: Counter[str] = Counter()
    error_distribution: Counter[str] = Counter()
    model_call_distribution: Counter[str] = Counter()
    hyde_status_distribution: Counter[str] = Counter()
    hyde_model_distribution: Counter[str] = Counter()
    for record in successful:
        tool_distribution.update(record["actual_tool_names"])
        planner_distribution.update(
            call["name"] for call in record.get("planned_tools", [])
        )
        critic_distribution.update([record["critic"]["verdict"]])
        model_call_distribution.update(
            call["call_name"] for call in record.get("model_calls", [])
        )
        hyde_status_distribution.update(
            str(call["hyde_status"])
            for call in record.get("executed_tools", [])
            if call.get("hyde_status")
        )
        hyde_model_distribution.update(
            str(call["hyde_model"])
            for call in record.get("executed_tools", [])
            if call.get("hyde_model")
        )
    error_distribution.update(
        str(record.get("error_type") or "UnknownError") for record in failed
    )

    def mean(path: tuple[str, ...]) -> float:
        values: list[float] = []
        for record in successful:
            value: Any = record
            for key in path:
                value = value[key]
            values.append(float(value))
        return round(statistics.fmean(values), 6) if values else 0.0

    total_tokens = sum(
        int(record["budget"]["total_tokens"]) for record in successful
    )
    total_input_tokens = sum(
        int(record["budget"]["input_tokens"]) for record in successful
    )
    total_output_tokens = sum(
        int(record["budget"]["output_tokens"]) for record in successful
    )
    planned_exact_count = sum(
        bool(record["planned_tool_set_exact_match"]) for record in successful
    )
    executed_exact_count = sum(
        bool(record["executed_tool_set_exact_match"]) for record in successful
    )
    citation_present_count = sum(
        bool(record["citation_ids_present"]) for record in successful
    )
    citation_allowlist_valid_count = sum(
        bool(record["citation_allowlist_valid"]) for record in successful
    )
    valid_citation_count = sum(
        bool(record["citation_ids_valid"]) for record in successful
    )

    def success_rate(count: int) -> float:
        return round(count / len(successful), 6) if successful else 0.0

    return {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "end_to_end_agent_selected_tools",
        "autonomous_tool_selection": True,
        "expected_tools_are_post_run_labels": True,
        "hyde_enabled": bool(use_hyde),
        "expected_questions": 10,
        "recorded_questions": len(records),
        "successful_questions": len(successful),
        "failed_questions": len(failed),
        "success_rate": round(len(successful) / 10, 6),
        "status": (
            "complete"
            if len(records) == 10 and len(successful) == 10
            else "incomplete"
        ),
        "average_latency_seconds": mean(("latency_seconds",)),
        "average_cost_usd": mean(("budget", "cost_usd")),
        "total_cost_usd": round(
            sum(float(record["budget"]["cost_usd"]) for record in successful),
            6,
        ),
        "average_input_tokens": round(
            total_input_tokens / len(successful), 2
        )
        if successful
        else 0.0,
        "average_output_tokens": round(
            total_output_tokens / len(successful), 2
        )
        if successful
        else 0.0,
        "average_total_tokens": round(total_tokens / len(successful), 2)
        if successful
        else 0.0,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "average_llm_calls": mean(("budget", "llm_calls")),
        "average_mcp_calls": mean(("budget", "tool_calls")),
        "total_executed_tool_calls": sum(tool_distribution.values()),
        "tool_selection_distribution": dict(sorted(planner_distribution.items())),
        "tool_call_distribution": dict(sorted(tool_distribution.items())),
        "model_call_distribution": dict(sorted(model_call_distribution.items())),
        "hyde_status_distribution": dict(sorted(hyde_status_distribution.items())),
        "hyde_model_distribution": dict(sorted(hyde_model_distribution.items())),
        "hyde_total_tokens": sum(
            int(call.get("hyde_total_tokens") or 0)
            for record in successful
            for call in record.get("executed_tools", [])
        ),
        "hyde_generated_characters": sum(
            int(call.get("hyde_generated_characters") or 0)
            for record in successful
            for call in record.get("executed_tools", [])
        ),
        "hyde_error_count": sum(
            bool(call.get("hyde_error"))
            for record in successful
            for call in record.get("executed_tools", [])
        ),
        "critic_verdict_distribution": dict(sorted(critic_distribution.items())),
        "error_distribution": dict(sorted(error_distribution.items())),
        "planned_tool_set_exact_match_count": planned_exact_count,
        "planned_tool_set_exact_match_rate": success_rate(planned_exact_count),
        "executed_tool_set_exact_match_count": executed_exact_count,
        "executed_tool_set_exact_match_rate": success_rate(executed_exact_count),
        # Backward-compatible aliases used by the initial report template.
        "tool_set_exact_match_count": planned_exact_count,
        "tool_set_exact_match_rate": success_rate(planned_exact_count),
        "citation_present_count": citation_present_count,
        "citation_present_rate": success_rate(citation_present_count),
        "citation_allowlist_valid_count": citation_allowlist_valid_count,
        "citation_allowlist_valid_rate": success_rate(
            citation_allowlist_valid_count
        ),
        "valid_citation_count": valid_citation_count,
        "valid_citation_rate": success_rate(valid_citation_count),
        "cost_note": (
            "USD cost is the configured TokenBudget estimate. With zero "
            "per-million prices it records the configured free-tier cost, "
            "not a universal provider price. Aggregates include successful "
            "runs only because failed runs do not return a final budget snapshot."
        ),
    }


async def evaluate(
    cases: Sequence[dict[str, Any]],
    *,
    details_path: Path,
    summary_path: Path,
    retry_errors: bool,
    use_hyde: bool,
) -> dict[str, Any]:
    cases_by_id = {str(case["id"]): case for case in cases}
    records = _existing_records(
        details_path,
        cases_by_id,
        retry_errors=retry_errors,
        use_hyde=use_hyde,
    )
    completed = {str(record["id"]) for record in records}
    if len(completed) == len(cases):
        summary = build_summary(records, use_hyde=use_hyde)
        _write_json(summary_path, summary)
        return summary

    mcp_url = os.getenv(
        "MCP_SERVER_URL",
        "http://127.0.0.1:8000/mcp",
    ).strip()
    managed_server = await _maybe_start_local_mcp_server(mcp_url)
    try:
        for position, case in enumerate(cases, start=1):
            identifier = str(case["id"])
            if identifier in completed:
                print(f"[{position:02d}/10] {identifier}: checkpoint already present")
                continue
            print(f"[{position:02d}/10] {identifier}: running full agent", flush=True)
            try:
                result = await run_with_mcp_server(
                    str(case["question"]),
                    use_hyde=use_hyde,
                )
                record = _successful_record(
                    case,
                    result,
                    use_hyde=use_hyde,
                )
                print(
                    f"[{position:02d}/10] {identifier}: "
                    f"ok tools={record['actual_tool_names']} "
                    f"latency={record['latency_seconds']:.3f}s",
                    flush=True,
                )
            except Exception as exc:
                record = _error_record(
                    case,
                    exc,
                    use_hyde=use_hyde,
                )
                print(
                    f"[{position:02d}/10] {identifier}: "
                    f"error {type(exc).__name__}: {exc}",
                    flush=True,
                )
            records.append(record)
            completed.add(identifier)
            records.sort(
                key=lambda item: list(cases_by_id).index(str(item["id"]))
            )
            _write_jsonl(details_path, records)
            _write_json(
                summary_path,
                build_summary(records, use_hyde=use_hyde),
            )
    finally:
        if managed_server is not None:
            await managed_server.stop()

    summary = build_summary(records, use_hyde=use_hyde)
    _write_json(summary_path, summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the checkpointed ten-case production-agent benchmark."
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data" / "evaluation" / "operational_questions.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "evaluation" / "results",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry checkpoint rows whose status is not ok.",
    )
    parser.add_argument(
        "--no-hyde",
        action="store_true",
        help="Use one consistent HyDE-disabled policy for all ten cases.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _load_environment()
    cases = load_cases(args.questions.resolve())
    output_dir = args.output_dir.resolve()
    summary = asyncio.run(
        evaluate(
            cases,
            details_path=output_dir / "operational_details.jsonl",
            summary_path=output_dir / "operational_summary.json",
            retry_errors=args.retry_errors,
            use_hyde=not args.no_hyde,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
