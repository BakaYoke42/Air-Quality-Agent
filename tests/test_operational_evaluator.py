"""Offline tests for the checkpointed end-to-end benchmark runner."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import evaluate_agent as evaluator


@dataclass(frozen=True)
class _ModelCall:
    call_name: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class _Budget:
    input_tokens = 100
    output_tokens = 50

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "max_llm_calls": 8,
            "max_tool_calls": 4,
            "max_input_tokens": 50_000,
            "max_output_tokens": 10_000,
            "max_cost_usd": None,
            "llm_calls": 5,
            "tool_calls": 1,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": 0.2,
            "reserved_llm_calls": 0,
            "reserved_tool_calls": 0,
            "reserved_input_tokens": 0,
            "reserved_output_tokens": 0,
            "reserved_cost_usd": 0.0,
        }


def _case(identifier: str = "op_test") -> dict[str, Any]:
    return {
        "id": identifier,
        "question": f"Question for {identifier}?",
        "expected_capability": "documentary search",
        "expected_tools": ["search_air_quality_evidence"],
    }


def _result() -> SimpleNamespace:
    planned = SimpleNamespace(
        name="search_air_quality_evidence",
        arguments={"query": "air quality"},
        purpose="Retrieve documentary evidence.",
    )
    executed = SimpleNamespace(
        sequence=1,
        tool_name="search_air_quality_evidence",
        arguments={"query": "air quality", "use_hyde": True},
        purpose="Retrieve documentary evidence.",
        latency_ms=12.5,
        audit_required=False,
        evidence_ids=("D1",),
        hyde_used=True,
        hyde_status="generated",
        hyde_model="test-hyde-model",
        hyde_generated_characters=120,
        hyde_total_tokens=25,
        hyde_error=None,
    )
    critic = SimpleNamespace(
        verdict="PASS",
        agreement="3/3",
        selected_draft="1",
        issues=(),
        final_answer=(
            "EVIDENCE:\n- Source fact. [D1]\n\n"
            "ANALYSIS:\nThe fact answers the question.\n\n"
            "CONCLUSION:\nSupported conclusion. [D1]\n\n"
            "CONFIDENCE: HIGH"
        ),
    )
    return SimpleNamespace(
        agent_version="test-version",
        planned_tools=(planned,),
        tool_calls=(executed,),
        evidence=(SimpleNamespace(evidence_id="D1"),),
        reasoning=SimpleNamespace(critic=critic),
        budget=_Budget(),
        model_calls=(
            _ModelCall("tool_selection", "test-model", 20, 10, 1.0),
            _ModelCall("critic", "test-model", 20, 10, 1.0),
        ),
        total_latency_ms=2_000,
    )


def _successful(
    identifier: str = "op_test",
    *,
    use_hyde: bool = True,
) -> dict[str, Any]:
    return evaluator._successful_record(
        _case(identifier),
        _result(),
        use_hyde=use_hyde,
    )


def test_operational_dataset_has_ten_cases_all_tools_and_mixed_plans() -> None:
    cases = evaluator.load_cases(
        evaluator.ROOT / "data" / "evaluation" / "operational_questions.jsonl"
    )

    assert len(cases) == 10
    assert {
        tool for case in cases for tool in case["expected_tools"]
    } == evaluator.AGENT_TOOL_NAMES
    assert sum(len(case["expected_tools"]) > 1 for case in cases) >= 1


def test_success_record_requires_present_allowlisted_citation() -> None:
    record = _successful()

    assert record["hyde_requested"] is True
    assert record["planned_tool_set_exact_match"] is True
    assert record["executed_tool_set_exact_match"] is True
    assert record["executed_tools"][0]["hyde_generated_characters"] == 120
    assert record["executed_tools"][0]["hyde_total_tokens"] == 25
    assert record["cited_evidence_ids"] == ["D1"]
    assert record["citation_ids_present"] is True
    assert record["citation_allowlist_valid"] is True
    assert record["citation_ids_valid"] is True

    no_citation_result = _result()
    no_citation_result.reasoning.critic.final_answer = (
        no_citation_result.reasoning.critic.final_answer.replace(" [D1]", "")
    )
    no_citation = evaluator._successful_record(
        _case(),
        no_citation_result,
        use_hyde=True,
    )
    assert no_citation["citation_ids_present"] is False
    assert no_citation["citation_allowlist_valid"] is True
    assert no_citation["citation_ids_valid"] is False

    invented_result = _result()
    invented_result.reasoning.critic.final_answer = (
        invented_result.reasoning.critic.final_answer.replace("[D1]", "[D99]")
    )
    invented = evaluator._successful_record(
        _case(),
        invented_result,
        use_hyde=True,
    )
    assert invented["citation_ids_present"] is True
    assert invented["citation_allowlist_valid"] is False
    assert invented["citation_ids_valid"] is False
    assert invented["unsupported_citation_ids"] == ["D99"]


def test_summary_aggregates_operational_metrics() -> None:
    records: list[dict[str, Any]] = []
    tools = (
        ["search_air_quality_evidence"],
        ["search_air_quality_evidence"],
        ["search_air_quality_evidence"],
        ["search_air_quality_evidence"],
        ["get_country_air_quality"],
        ["get_country_air_quality"],
        ["compare_countries"],
        ["compare_countries"],
        ["find_station_extremes"],
        ["find_station_extremes"],
    )
    for index, actual_tools in enumerate(tools, start=1):
        record = copy.deepcopy(_successful(f"op_{index:02d}"))
        record["actual_tool_names"] = actual_tools
        record["planned_tools"] = [
            {"name": tool, "arguments": {}, "purpose": "test"}
            for tool in actual_tools
        ]
        record["executed_tools"] = [
            {
                "tool_name": tool,
                "hyde_status": (
                    "generated"
                    if tool == "search_air_quality_evidence"
                    else None
                ),
            }
            for tool in actual_tools
        ]
        record["planned_tool_set_exact_match"] = index <= 9
        record["executed_tool_set_exact_match"] = index <= 8
        record["tool_set_exact_match"] = index <= 9
        record["citation_allowlist_valid"] = index <= 9
        record["citation_ids_valid"] = index <= 9
        records.append(record)

    summary = evaluator.build_summary(records, use_hyde=True)

    assert summary["status"] == "complete"
    assert summary["successful_questions"] == 10
    assert summary["failed_questions"] == 0
    assert summary["success_rate"] == 1.0
    assert summary["average_latency_seconds"] == 2.0
    assert summary["average_cost_usd"] == 0.2
    assert summary["total_cost_usd"] == 2.0
    assert summary["average_input_tokens"] == 100.0
    assert summary["average_output_tokens"] == 50.0
    assert summary["average_total_tokens"] == 150.0
    assert summary["total_tokens"] == 1_500
    assert summary["tool_call_distribution"] == {
        "compare_countries": 2,
        "find_station_extremes": 2,
        "get_country_air_quality": 2,
        "search_air_quality_evidence": 4,
    }
    assert summary["hyde_status_distribution"] == {"generated": 4}
    assert summary["model_call_distribution"] == {
        "critic": 10,
        "tool_selection": 10,
    }
    assert summary["planned_tool_set_exact_match_rate"] == 0.9
    assert summary["executed_tool_set_exact_match_rate"] == 0.8
    assert summary["citation_present_rate"] == 1.0
    assert summary["citation_allowlist_valid_rate"] == 0.9
    assert summary["valid_citation_rate"] == 0.9


def test_checkpoint_rejects_changed_policy_and_retries_only_errors(
    tmp_path: Path,
) -> None:
    success_case = _case("op_success")
    error_case = _case("op_error")
    cases_by_id = {
        success_case["id"]: success_case,
        error_case["id"]: error_case,
    }
    success = _successful("op_success", use_hyde=True)
    error = evaluator._error_record(
        error_case,
        ValueError("synthetic failure"),
        use_hyde=True,
    )
    details = tmp_path / "details.jsonl"
    evaluator._write_jsonl(details, [success, error])

    resumed = evaluator._existing_records(
        details,
        cases_by_id,
        retry_errors=True,
        use_hyde=True,
    )
    assert [record["id"] for record in resumed] == ["op_success"]

    with pytest.raises(
        evaluator.OperationalEvaluationError,
        match="HyDE policy differs",
    ):
        evaluator._existing_records(
            details,
            cases_by_id,
            retry_errors=False,
            use_hyde=False,
        )


def test_complete_checkpoint_does_not_start_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    details = tmp_path / "details.jsonl"
    summary_path = tmp_path / "summary.json"
    evaluator._write_jsonl(details, [_successful(use_hyde=False)])

    async def unexpected_start(_: str) -> None:
        raise AssertionError("A complete checkpoint must not start MCP.")

    monkeypatch.setattr(evaluator, "_maybe_start_local_mcp_server", unexpected_start)
    summary = asyncio.run(
        evaluator.evaluate(
            [case],
            details_path=details,
            summary_path=summary_path,
            retry_errors=False,
            use_hyde=False,
        )
    )

    assert summary["recorded_questions"] == 1
    assert summary["hyde_enabled"] is False
    assert summary_path.is_file()


def test_agent_receives_only_question_and_hyde_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    calls: list[tuple[str, bool]] = []

    class _ManagedServer:
        stopped = False

        async def stop(self) -> None:
            self.stopped = True

    managed = _ManagedServer()

    async def fake_start(_: str) -> _ManagedServer:
        return managed

    async def fake_run(question: str, *, use_hyde: bool) -> SimpleNamespace:
        calls.append((question, use_hyde))
        return _result()

    monkeypatch.setattr(evaluator, "_maybe_start_local_mcp_server", fake_start)
    monkeypatch.setattr(evaluator, "run_with_mcp_server", fake_run)
    summary = asyncio.run(
        evaluator.evaluate(
            [case],
            details_path=tmp_path / "details.jsonl",
            summary_path=tmp_path / "summary.json",
            retry_errors=False,
            use_hyde=False,
        )
    )

    assert calls == [(case["question"], False)]
    assert managed.stopped is True
    assert summary["autonomous_tool_selection"] is True
    assert summary["expected_tools_are_post_run_labels"] is True
    assert summary["hyde_enabled"] is False


def test_keyboard_interrupt_is_not_saved_as_a_case_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()

    class _ManagedServer:
        stopped = False

        async def stop(self) -> None:
            self.stopped = True

    managed = _ManagedServer()

    async def fake_start(_: str) -> _ManagedServer:
        return managed

    async def interrupted_run(question: str, *, use_hyde: bool) -> None:
        del question, use_hyde
        raise KeyboardInterrupt

    monkeypatch.setattr(evaluator, "_maybe_start_local_mcp_server", fake_start)
    monkeypatch.setattr(evaluator, "run_with_mcp_server", interrupted_run)
    details = tmp_path / "details.jsonl"

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(
            evaluator.evaluate(
                [case],
                details_path=details,
                summary_path=tmp_path / "summary.json",
                retry_errors=False,
                use_hyde=True,
            )
        )

    assert managed.stopped is True
    assert not details.exists()
