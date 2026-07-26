from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval import AdvancedRetriever


class _FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _fake_response(text: str = "An official hypothetical evidence passage."):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(
            prompt_tokens=21,
            completion_tokens=9,
            total_tokens=30,
        ),
    )


def test_hyde_calls_configured_model_and_reports_generation(monkeypatch) -> None:
    created: dict[str, object] = {}
    completions = _FakeCompletions(_fake_response())

    def fake_openai(**kwargs):
        created.update(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=completions))

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=fake_openai))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "fallback-model")
    monkeypatch.setenv("HYDE_MODEL", "hyde-model")
    monkeypatch.setenv("HYDE_REQUEST_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("HYDE_MAX_RETRIES", "0")

    retriever = AdvancedRetriever.__new__(AdvancedRetriever)
    result = retriever.generate_hypothetical_document("WHO PM2.5 guideline")

    assert result["used"] is True
    assert result["model"] == "hyde-model"
    assert result["generated_characters"] == len(result["text"])
    assert result["usage"] == {
        "input_tokens": 21,
        "output_tokens": 9,
        "total_tokens": 30,
    }
    assert created["timeout"] == 12.0
    assert created["max_retries"] == 0
    assert completions.calls[0]["model"] == "hyde-model"


def test_hyde_failure_falls_back_to_original_query(monkeypatch) -> None:
    class FailingCompletions:
        @staticmethod
        def create(**_kwargs):
            raise TimeoutError("provider timed out")

    def fake_openai(**_kwargs):
        return SimpleNamespace(
            chat=SimpleNamespace(completions=FailingCompletions())
        )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=fake_openai))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "fallback-model")
    monkeypatch.delenv("HYDE_MODEL", raising=False)

    query = "Compare annual NO2 guidance"
    retriever = AdvancedRetriever.__new__(AdvancedRetriever)
    result = retriever.generate_hypothetical_document(query)

    assert result["used"] is False
    assert result["text"] == query
    assert result["model"] == "fallback-model"
    assert result["generated_characters"] == 0
    assert result["error"] == "TimeoutError: provider timed out"


def test_mcp_retrieval_reports_hyde_status_without_exposing_passage(
    monkeypatch,
) -> None:
    import mcp_server

    class FakeRetriever:
        @staticmethod
        def retrieve(*_args, **_kwargs):
            return {
                "hyde": {
                    "used": True,
                    "text": "This generated passage must stay private to retrieval.",
                    "error": None,
                    "usage": {
                        "input_tokens": 21,
                        "output_tokens": 9,
                        "total_tokens": 30,
                    },
                    "model": "hyde-model",
                    "generated_characters": 54,
                },
                "timings_ms": {"hyde_ms": 10.0, "total_ms": 20.0},
                "results": [
                    {
                        "rank": 1,
                        "parent_id": "parent-1",
                        "matched_child_id": "child-1",
                        "doc_id": "doc-1",
                        "title": "WHO guidance",
                        "publisher": "WHO",
                        "document_type": "guideline",
                        "evidence_status": "authoritative",
                        "publication_year": 2021,
                        "data_year": None,
                        "page_start": 1,
                        "page_end": 1,
                        "source_url": "https://example.invalid/source",
                        "matched_child_text": "Matched evidence",
                        "parent_text": "Expanded parent evidence",
                        "reranker_score": 1.0,
                        "rrf_score": 0.03,
                    }
                ],
            }

    monkeypatch.setattr(mcp_server, "_retriever", FakeRetriever())
    payload = json.loads(
        mcp_server.search_air_quality_evidence(
            "WHO PM2.5 guideline", top_k=1, use_hyde=True
        )
    )

    assert payload["status"] == "ok"
    data = payload["data"]
    assert data["hyde_requested"] is True
    assert data["hyde_status"] == "generated"
    assert data["hyde_used"] is True
    assert data["hyde_model"] == "hyde-model"
    assert data["hyde_usage"]["total_tokens"] == 30
    assert "This generated passage" not in json.dumps(payload)
