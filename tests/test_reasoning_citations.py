"""Offline tests for deterministic evidence-citation enforcement."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reasoning import (
    ModelResponse,
    ReasoningEngine,
    validate_evidence_citations,
    validate_synthesis_format,
)


VALID_ANSWER = """EVIDENCE:
- The documentary source supplies an annual benchmark. [D1]
- The structured result supplies a sampling-point median. [M1]

ANALYSIS:
1. The two values can be compared because their pollutant, period, and unit match. [D1] [M1]

CONCLUSION: The comparison is supported but remains a sampling-point summary. [M1]

CONFIDENCE: HIGH - both necessary evidence types are supplied. [D1] [M1]
"""


def _critic(
    final_answer: str,
    *,
    verdict: str = "PASS",
    selected_draft: str = "1",
) -> str:
    return f"""VERDICT: {verdict}
AGREEMENT: 3/3
SELECTED_DRAFT: {selected_draft}
ISSUES:
- None
FINAL_ANSWER:
{final_answer}
"""


class CitationModel:
    def __init__(
        self,
        critic_answer: str,
        *,
        draft_answer: str = VALID_ANSWER,
    ) -> None:
        self.critic_answer = critic_answer
        self.draft_answer = draft_answer
        self.calls: list[dict] = []

    async def __call__(self, **kwargs) -> ModelResponse:
        self.calls.append(kwargs)
        content = (
            self.critic_answer
            if kwargs["call_name"] == "critic"
            else self.draft_answer
        )
        return ModelResponse(content=content, model="citation-test-model")


def test_allowed_citations_are_passed_to_every_call_and_accepted() -> None:
    model = CitationModel(_critic(VALID_ANSWER))

    result = asyncio.run(
        ReasoningEngine(model).synthesize(
            "Compare the supplied annual values.",
            "[D1] documentary benchmark\n[M1] structured measurement",
            allowed_evidence_ids=("D1", "M1"),
        )
    )

    assert len(model.calls) == 4
    for call in model.calls:
        user_prompt = call["messages"][1]["content"]
        assert "ALLOWED EVIDENCE IDS (complete allowlist):" in user_prompt
        assert "[D1], [M1]" in user_prompt
    assert all(draft.citations_valid for draft in result.drafts)
    assert result.critic.verdict == "PASS"
    assert validate_evidence_citations(
        result.critic.final_answer,
        ("D1", "M1"),
    ) == (True, ())


def test_pass_with_none_selection_is_normalized_to_revision() -> None:
    model = CitationModel(
        _critic(VALID_ANSWER, verdict="PASS", selected_draft="NONE")
    )

    result = asyncio.run(
        ReasoningEngine(model).synthesize(
            "Compare the supplied annual values.",
            "[D1] documentary benchmark\n[M1] structured measurement",
            allowed_evidence_ids=("D1", "M1"),
        )
    )

    assert result.critic.verdict == "REVISED"
    assert result.critic.selected_draft == "NONE"
    assert any("PASS must select" in issue for issue in result.critic.issues)


def test_pass_with_edited_final_answer_is_normalized_to_revision() -> None:
    edited = VALID_ANSWER.replace(
        "The comparison is supported",
        "The grounded comparison is supported",
    )
    model = CitationModel(_critic(edited))

    result = asyncio.run(
        ReasoningEngine(model).synthesize(
            "Compare the supplied annual values.",
            "[D1] documentary benchmark\n[M1] structured measurement",
            allowed_evidence_ids=("D1", "M1"),
        )
    )

    assert result.critic.verdict == "REVISED"
    assert result.critic.selected_draft == "NONE"
    assert result.critic.final_answer == edited.strip()
    assert any("unchanged" in issue for issue in result.critic.issues)


def test_citation_free_generated_answer_is_rejected() -> None:
    citation_free = VALID_ANSWER.replace(" [D1]", "").replace(" [M1]", "")
    model = CitationModel(_critic(citation_free), draft_answer=citation_free)

    result = asyncio.run(
        ReasoningEngine(model).synthesize(
            "Compare the supplied annual values.",
            "[D1] documentary benchmark\n[M1] structured measurement",
            allowed_evidence_ids=("D1", "M1"),
        )
    )

    assert not any(draft.citations_valid for draft in result.drafts)
    assert result.critic.verdict == "REVISED"
    assert result.critic.selected_draft == "NONE"
    assert "No grounded answer can be returned" in result.critic.final_answer
    assert any("at least one allowed" in issue for issue in result.critic.issues)


def test_malformed_critic_uses_revision_not_fallback() -> None:
    model = CitationModel("The critic returned malformed output.")

    result = asyncio.run(
        ReasoningEngine(model).synthesize(
            "Compare the supplied annual values.",
            "[D1] documentary benchmark\n[M1] structured measurement",
            allowed_evidence_ids=("D1", "M1"),
        )
    )

    assert result.critic.verdict == "REVISED"
    assert result.critic.selected_draft == "NONE"
    assert result.critic.final_answer == VALID_ANSWER.strip()
    assert any("format was invalid" in issue for issue in result.critic.issues)


def test_empty_required_section_is_invalid() -> None:
    empty_analysis = VALID_ANSWER.replace(
        "ANALYSIS:\n"
        "1. The two values can be compared because their pollutant, period, "
        "and unit match. [D1] [M1]\n\n",
        "ANALYSIS:\n\n",
    )

    assert validate_synthesis_format(empty_analysis) == (False, ("ANALYSIS",))


def test_invented_critic_citation_forces_revision_and_is_not_returned() -> None:
    invented = VALID_ANSWER.replace("[D1]", "[D99]", 1)
    model = CitationModel(_critic(invented, verdict="PASS"))

    result = asyncio.run(
        ReasoningEngine(model).synthesize(
            "Compare the supplied annual values.",
            "[D1] documentary benchmark\n[M1] structured measurement",
            allowed_evidence_ids=("D1", "M1"),
        )
    )

    assert validate_evidence_citations(invented, ("D1", "M1")) == (
        False,
        ("D99",),
    )
    assert result.critic.verdict == "REVISED"
    assert "D99" not in result.critic.final_answer
    assert any("D99" in issue for issue in result.critic.issues)
    assert validate_evidence_citations(
        result.critic.final_answer,
        ("D1", "M1"),
    ) == (True, ())
