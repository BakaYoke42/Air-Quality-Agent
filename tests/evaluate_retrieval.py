"""One baseline-versus-final RAG evaluation for the homework report.

The comparison deliberately holds answer generation and RAGAS judges constant:

* baseline: dense cosine retrieval over the flat 500-word baseline chunks;
* final: BM25 + dense + RRF + child reranking + unique parent expansion.

Run ``--retrieval-only`` for an offline retrieval audit.  A full run adds the
four metrics required by the assignment: context_recall, context_precision,
faithfulness, and answer_relevancy.  It never prints configuration secrets.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    for candidate in (script_dir, script_dir.parent):
        if (candidate / "data").is_dir() and (candidate / "src").is_dir():
            return candidate
    return Path.cwd()


ROOT = project_root()
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retrieval import AdvancedRetriever, load_jsonl, top_indices  # noqa: E402


REQUIRED_GOLD_FIELDS = {
    "id",
    "question",
    "reference_answer",
    "relevant_doc_ids",
    "relevant_parent_ids",
    "category",
    "difficulty",
}
REQUIRED_RAGAS_METRICS = (
    "context_recall",
    "context_precision",
    "faithfulness",
    "answer_relevancy",
)


class EvaluationError(RuntimeError):
    """Controlled configuration or evaluation failure."""


def load_golden(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(
                    f"Invalid JSON in {path}, line {line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise EvaluationError(
                    f"Expected an object in {path}, line {line_number}"
                )
            missing = REQUIRED_GOLD_FIELDS - set(record)
            if missing:
                raise EvaluationError(
                    f"{path}, line {line_number} is missing: {sorted(missing)}"
                )
            records.append(record)
    return records


def validate_golden(
    golden: Sequence[dict[str, Any]],
    retriever: AdvancedRetriever,
) -> None:
    if len(golden) < 10:
        raise EvaluationError(
            "The rubric requires at least 10 golden questions"
        )
    question_ids = [str(row["id"]) for row in golden]
    if len(question_ids) != len(set(question_ids)):
        raise EvaluationError("Golden question IDs must be unique")

    known_parents = set(retriever.parents)
    known_docs = {str(record["doc_id"]) for record in retriever.parents.values()}
    for row in golden:
        relevant_parents = {str(value) for value in row["relevant_parent_ids"]}
        relevant_docs = {str(value) for value in row["relevant_doc_ids"]}
        if not relevant_parents or not relevant_docs:
            raise EvaluationError(
                f"{row['id']} must have relevant parent and document IDs"
            )
        missing_parents = relevant_parents - known_parents
        missing_docs = relevant_docs - known_docs
        if missing_parents:
            raise EvaluationError(
                f"{row['id']} has unknown parents: {sorted(missing_parents)}"
            )
        if missing_docs:
            raise EvaluationError(
                f"{row['id']} has unknown documents: {sorted(missing_docs)}"
            )


def ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def reciprocal_rank(predicted: Sequence[str], relevant: set[str]) -> float:
    for rank, item_id in enumerate(predicted, start=1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(predicted: Sequence[str], relevant: set[str], k: int) -> float:
    gains = [1.0 if item_id in relevant else 0.0 for item_id in predicted[:k]]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
    ideal_count = min(len(relevant), k)
    if ideal_count == 0:
        return 0.0
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal


def deterministic_rank_metrics(
    predicted: Sequence[str],
    relevant: set[str],
    *,
    top_k: int,
) -> dict[str, float]:
    ranked = list(predicted[:top_k])
    matches = relevant.intersection(ranked)
    return {
        "hit_at_k": float(bool(matches)),
        "precision_at_k": len(matches) / max(1, len(ranked)),
        "recall_at_k": len(matches) / max(1, len(relevant)),
        "mrr_at_k": reciprocal_rank(ranked, relevant),
        "ndcg_at_k": ndcg_at_k(ranked, relevant, top_k),
    }


class BaselineDenseRetriever:
    """Basic flat-chunk dense retrieval using the final pipeline's embedder."""

    def __init__(self, root: Path, advanced: AdvancedRetriever) -> None:
        path = root / "data" / "corpus_processed" / "baseline_chunks.jsonl"
        if not path.is_file():
            raise EvaluationError(f"Baseline chunks are missing: {path}")
        self.records = load_jsonl(path)
        if not self.records:
            raise EvaluationError("The baseline corpus is empty")
        self.embedder = advanced.embedder
        texts = [str(row["search_text"]) for row in self.records]
        self.embeddings = np.asarray(
            self.embedder.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )

    def retrieve(self, question: str, *, top_k: int) -> dict[str, Any]:
        started = time.perf_counter()
        query = np.asarray(
            self.embedder.encode(
                [question],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )[0]
        indices = top_indices(self.embeddings @ query, top_k)
        results = []
        for rank, index in enumerate(indices, start=1):
            row = self.records[index]
            results.append(
                {
                    "rank": rank,
                    "context_id": str(row["chunk_id"]),
                    "parent_id": None,
                    "doc_id": str(row["doc_id"]),
                    "title": str(row["title"]),
                    "text": str(row["text"]),
                }
            )
        return {
            "pipeline": "dense cosine over flat 500-word baseline chunks",
            "latency_ms": round((time.perf_counter() - started) * 1_000, 2),
            "results": results,
        }


def _pipeline_detail(
    results: Sequence[dict[str, Any]],
    *,
    latency_ms: float,
    pipeline: str,
    relevant_docs: set[str],
    relevant_parents: set[str],
    top_k: int,
) -> dict[str, Any]:
    doc_ids = ordered_unique(str(row["doc_id"]) for row in results)
    parent_ids = ordered_unique(
        str(row["parent_id"])
        for row in results
        if row.get("parent_id") is not None
    )
    detail: dict[str, Any] = {
        "pipeline": pipeline,
        "latency_ms": float(latency_ms),
        "retrieved_context_ids": [str(row["context_id"]) for row in results],
        "retrieved_doc_ids": doc_ids,
        "retrieved_parent_ids": parent_ids,
        "retrieved_contexts": [str(row["text"]) for row in results],
        "context_metadata": [
            {
                "rank": int(row["rank"]),
                "context_id": str(row["context_id"]),
                "parent_id": row.get("parent_id"),
                "doc_id": str(row["doc_id"]),
                "title": str(row["title"]),
            }
            for row in results
        ],
        "document_metrics": deterministic_rank_metrics(
            doc_ids, relevant_docs, top_k=top_k
        ),
    }
    detail["parent_metrics"] = (
        deterministic_rank_metrics(parent_ids, relevant_parents, top_k=top_k)
        if parent_ids
        else None
    )
    return detail


def collect_retrieval(
    advanced: AdvancedRetriever,
    baseline: BaselineDenseRetriever,
    golden: Sequence[dict[str, Any]],
    *,
    top_k: int,
    use_hyde: bool,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for number, gold in enumerate(golden, start=1):
        question = str(gold["question"])
        relevant_docs = {str(value) for value in gold["relevant_doc_ids"]}
        relevant_parents = {str(value) for value in gold["relevant_parent_ids"]}

        baseline_output = baseline.retrieve(question, top_k=top_k)
        final_output = advanced.retrieve(
            question,
            final_k=top_k,
            use_hyde=use_hyde,
        )
        final_results = [
            {
                "rank": row["rank"],
                "context_id": row["matched_child_id"],
                "parent_id": row["parent_id"],
                "doc_id": row["doc_id"],
                "title": row["title"],
                "text": row["parent_text"],
            }
            for row in final_output["results"]
        ]

        detail = {
            "id": str(gold["id"]),
            "question": question,
            "reference_answer": str(gold["reference_answer"]),
            "category": str(gold["category"]),
            "difficulty": str(gold["difficulty"]),
            "relevant_doc_ids": sorted(relevant_docs),
            "relevant_parent_ids": sorted(relevant_parents),
            "baseline": _pipeline_detail(
                baseline_output["results"],
                latency_ms=baseline_output["latency_ms"],
                pipeline=baseline_output["pipeline"],
                relevant_docs=relevant_docs,
                relevant_parents=relevant_parents,
                top_k=top_k,
            ),
            "final": _pipeline_detail(
                final_results,
                latency_ms=final_output["timings_ms"]["total_ms"],
                pipeline=(
                    "HyDE + BM25/dense/RRF + child reranking + parent expansion"
                    if use_hyde
                    else "BM25/dense/RRF + child reranking + parent expansion"
                ),
                relevant_docs=relevant_docs,
                relevant_parents=relevant_parents,
                top_k=top_k,
            ),
        }
        details.append(detail)
        base_hit = int(detail["baseline"]["document_metrics"]["hit_at_k"])
        final_hit = int(detail["final"]["document_metrics"]["hit_at_k"])
        print(
            f"[{number:02d}/{len(golden):02d}] {gold['id']} "
            f"document hit baseline={base_hit} final={final_hit}"
        )
    return details


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    return round(statistics.fmean(materialized), 4) if materialized else 0.0


def aggregate_pipeline(
    details: Sequence[dict[str, Any]],
    pipeline_name: str,
) -> dict[str, Any]:
    rows = [row[pipeline_name] for row in details]
    document_metric_names = (
        "hit_at_k",
        "precision_at_k",
        "recall_at_k",
        "mrr_at_k",
        "ndcg_at_k",
    )
    output: dict[str, Any] = {
        "questions": len(rows),
        "average_retrieval_latency_ms": round(
            statistics.fmean(float(row["latency_ms"]) for row in rows), 2
        ),
        "deterministic_document_metrics": {
            name: _mean(row["document_metrics"][name] for row in rows)
            for name in document_metric_names
        },
    }
    parent_rows = [row["parent_metrics"] for row in rows if row["parent_metrics"]]
    if parent_rows:
        output["deterministic_parent_metrics"] = {
            name: _mean(row[name] for row in parent_rows)
            for name in document_metric_names
        }
    ragas_rows = [row.get("ragas") for row in rows if row.get("ragas")]
    if ragas_rows:
        output["ragas"] = {
            metric: _mean(row[metric] for row in ragas_rows)
            for metric in REQUIRED_RAGAS_METRICS
        }
    return output


def _load_environment() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise EvaluationError("Install requirements before evaluation") from exc
    load_dotenv(ROOT / ".env")


def _openai_configuration() -> tuple[str, str, str]:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
    model = (
        (os.getenv("RAGAS_ANSWER_MODEL") or "").strip()
        or (os.getenv("LLM_MODEL") or "").strip()
    )
    if not api_key or not model:
        raise EvaluationError(
            "A full run requires OPENAI_API_KEY and RAGAS_ANSWER_MODEL (or LLM_MODEL)"
        )
    return api_key, base_url, model


def _score_value(result: Any) -> float:
    value = getattr(result, "value", result)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"RAGAS returned a non-numeric score: {value!r}") from exc
    if not math.isfinite(numeric):
        raise EvaluationError(f"RAGAS returned a non-finite score: {numeric}")
    return numeric


async def add_answers_and_ragas(details: list[dict[str, Any]]) -> None:
    """Generate matched answers and attach the four actual RAGAS metrics.

    A single pipeline requires one answer call plus ten judge calls when
    ``top_k=4`` (recall=1, precision=4, faithfulness=2, relevancy=3). Metrics
    are independent, so they run concurrently. Small semaphores bound both the
    pipelines in flight and active LLM call groups to avoid provider bursts.
    """

    try:
        from openai import AsyncOpenAI
        from ragas.embeddings import HuggingFaceEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:
        raise EvaluationError(
            "Full scoring requires ragas==0.4.3; run pip install -r requirements.txt"
        ) from exc

    api_key, base_url, answer_model = _openai_configuration()
    client_options: dict[str, Any] = {
        "api_key": api_key,
        "timeout": float(os.getenv("RAGAS_REQUEST_TIMEOUT_SECONDS", "120")),
        "max_retries": int(os.getenv("RAGAS_MAX_RETRIES", "1")),
    }
    if base_url:
        client_options["base_url"] = base_url
    client = AsyncOpenAI(**client_options)
    judge_model = (
        (os.getenv("RAGAS_JUDGE_MODEL") or "").strip() or answer_model
    )
    judge = llm_factory(judge_model, client=client)
    embedding_model = (
        (os.getenv("RAGAS_EMBEDDING_MODEL") or "").strip()
        or (os.getenv("EMBEDDING_MODEL") or "").strip()
        or "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    )
    embedding_options: dict[str, Any] = {"model": embedding_model}
    retrieval_device = (os.getenv("RETRIEVAL_DEVICE") or "").strip()
    if retrieval_device:
        embedding_options["device"] = retrieval_device
    judge_embeddings = HuggingFaceEmbeddings(**embedding_options)

    metrics = {
        "context_recall": ContextRecall(llm=judge),
        "context_precision": ContextPrecision(llm=judge),
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(
            llm=judge,
            embeddings=judge_embeddings,
        ),
    }
    answer_system = (
        "Answer only from the supplied contexts. Be concise, preserve dates and "
        "units, and say when the contexts do not support an answer. Do not cite "
        "sources because citation style is evaluated separately."
    )
    max_tokens = int(os.getenv("RAGAS_ANSWER_MAX_TOKENS", "350"))
    max_concurrent_pipelines = int(
        os.getenv("RAGAS_MAX_CONCURRENT_PIPELINES", "2")
    )
    if max_concurrent_pipelines < 1:
        raise EvaluationError("RAGAS_MAX_CONCURRENT_PIPELINES must be at least 1")
    max_concurrent_llm_groups = int(
        os.getenv("RAGAS_MAX_CONCURRENT_LLM_GROUPS", "3")
    )
    if max_concurrent_llm_groups < 1:
        raise EvaluationError("RAGAS_MAX_CONCURRENT_LLM_GROUPS must be at least 1")
    pipeline_semaphore = asyncio.Semaphore(max_concurrent_pipelines)
    llm_group_semaphore = asyncio.Semaphore(max_concurrent_llm_groups)
    total_pipelines = len(details) * 2
    completed_pipelines = 0

    async def evaluate_pipeline(
        job_number: int,
        detail: dict[str, Any],
        pipeline_name: str,
    ) -> None:
        nonlocal completed_pipelines
        async with pipeline_semaphore:
            pipeline = detail[pipeline_name]
            contexts = pipeline["retrieved_contexts"]
            context_block = "\n\n".join(
                f"CONTEXT {index}:\n{text}"
                for index, text in enumerate(contexts, start=1)
            )
            label = f"{detail['id']} {pipeline_name}"
            print(
                f"[RAGAS job {job_number:02d}/{total_pipelines:02d}] "
                f"{label}: generating answer",
                flush=True,
            )
            async with llm_group_semaphore:
                response = await client.chat.completions.create(
                    model=answer_model,
                    temperature=0,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": answer_system},
                        {
                            "role": "user",
                            "content": (
                                f"QUESTION:\n{detail['question']}\n\n{context_block}"
                            ),
                        },
                    ],
                )
            answer = (response.choices[0].message.content or "").strip()
            if not answer:
                raise EvaluationError(
                    f"Answer generation returned empty text for {detail['id']} {pipeline_name}"
                )
            pipeline["answer"] = answer
            print(f"  {label}: answer complete; scoring four metrics", flush=True)

            async def score_metric(
                metric_name: str,
                score_factory: Any,
            ) -> tuple[str, float]:
                async with llm_group_semaphore:
                    value = _score_value(await score_factory())
                print(f"  {label}: {metric_name} complete", flush=True)
                return metric_name, value

            metric_results = await asyncio.gather(
                score_metric(
                    "context_recall",
                    lambda: metrics["context_recall"].ascore(
                        user_input=detail["question"],
                        retrieved_contexts=contexts,
                        reference=detail["reference_answer"],
                    ),
                ),
                score_metric(
                    "context_precision",
                    lambda: metrics["context_precision"].ascore(
                        user_input=detail["question"],
                        retrieved_contexts=contexts,
                        reference=detail["reference_answer"],
                    ),
                ),
                score_metric(
                    "faithfulness",
                    lambda: metrics["faithfulness"].ascore(
                        user_input=detail["question"],
                        response=answer,
                        retrieved_contexts=contexts,
                    ),
                ),
                score_metric(
                    "answer_relevancy",
                    lambda: metrics["answer_relevancy"].ascore(
                        user_input=detail["question"],
                        response=answer,
                    ),
                ),
            )
            scores = dict(metric_results)
            pipeline["ragas"] = {
                key: round(value, 6) for key, value in scores.items()
            }
            completed_pipelines += 1
            print(
                f"[RAGAS {completed_pipelines:02d}/{total_pipelines:02d}] "
                f"{label}: pipeline complete",
                flush=True,
            )

    jobs = [
        (detail, pipeline_name)
        for detail in details
        for pipeline_name in ("baseline", "final")
    ]
    tasks = [
        evaluate_pipeline(job_number, detail, pipeline_name)
        for job_number, (detail, pipeline_name) in enumerate(jobs, start=1)
    ]
    await asyncio.gather(*tasks)


def write_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    details: Sequence[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "evaluation_details.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for row in details:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_summary(
    details: Sequence[dict[str, Any]],
    *,
    top_k: int,
    use_hyde: bool,
    retrieval_only: bool,
) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "questions": len(details),
        "top_k": top_k,
        "hyde_enabled_for_final": use_hyde,
        "mode": "retrieval_only" if retrieval_only else "full_ragas",
        "required_ragas_metrics": list(REQUIRED_RAGAS_METRICS),
        "comparison_control": (
            "The same answer model, prompt, RAGAS judge, and embedding model are "
            "used for baseline and final; only retrieval changes."
        ),
        "baseline": aggregate_pipeline(details, "baseline"),
        "final": aggregate_pipeline(details, "final"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one baseline-versus-final retrieval and RAGAS evaluation."
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=ROOT / "golden_dataset.jsonl",
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--hyde",
        action="store_true",
        help="Include optional HyDE in the final pipeline (off by default for repeatability).",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip answer generation and RAGAS judge calls.",
    )
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "evaluation" / "results",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k < 1 or args.top_k > 10:
        print("ERROR: --top-k must be between 1 and 10", file=sys.stderr)
        return 2
    try:
        _load_environment()
        golden = load_golden(args.golden)
        advanced = AdvancedRetriever(root=ROOT, rebuild_index=args.rebuild_index)
        validate_golden(golden, advanced)
        baseline = BaselineDenseRetriever(ROOT, advanced)
        details = collect_retrieval(
            advanced,
            baseline,
            golden,
            top_k=args.top_k,
            use_hyde=args.hyde,
        )

        # Save the retrieval audit before any optional judge call, so a provider
        # failure never destroys the useful deterministic comparison.
        summary = build_summary(
            details,
            top_k=args.top_k,
            use_hyde=args.hyde,
            retrieval_only=True,
        )
        write_outputs(args.output_dir, summary, details)

        if not args.retrieval_only:
            asyncio.run(add_answers_and_ragas(details))
            summary = build_summary(
                details,
                top_k=args.top_k,
                use_hyde=args.hyde,
                retrieval_only=False,
            )
            write_outputs(args.output_dir, summary, details)
    except (EvaluationError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nBaseline/final evaluation complete")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results written to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
