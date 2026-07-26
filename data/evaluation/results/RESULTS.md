# Accepted full RAGAS result

The accepted controlled comparison finished at
`2026-07-23T09:37:03.403492+00:00`. It contains 14 unique golden questions,
four contexts per pipeline, non-empty baseline/final answers, and all four
required RAGAS scores. Every aggregate in `evaluation_summary.json` was
recomputed from the 14 `evaluation_details.jsonl` rows and matched exactly.

HyDE was disabled for both branches. The answer model, answer prompt, RAGAS
judge, evaluator embedding, question set, and `top_k=4` were held constant; the
retrieval pipeline was the controlled variable.

## Required RAGAS comparison

| Metric | Baseline | Final | Change |
| --- | ---: | ---: | ---: |
| Context recall | 0.7143 | 1.0000 | +0.2857 |
| Context precision | 0.5952 | 0.7480 | +0.1528 |
| Faithfulness | 0.9107 | 0.9286 | +0.0179 |
| Answer relevancy | 0.7110 | 0.9663 | +0.2553 |

## Supplementary retrieval diagnostics

| Metric | Baseline | Final | Change |
| --- | ---: | ---: | ---: |
| Document hit@4 | 0.9286 | 1.0000 | +0.0714 |
| Unique-document precision@4 | 0.3869 | 0.5060 | +0.1191 |
| Document recall@4 | 0.9286 | 1.0000 | +0.0714 |
| MRR@4 | 0.7024 | 0.9167 | +0.2143 |
| nDCG@4 | 0.7610 | 0.9379 | +0.1769 |
| Mean retrieval latency | 12.72 ms | 1302.56 ms | +1289.84 ms |

The final retriever is approximately 102.4 times slower because it performs
BM25 and dense child retrieval, reciprocal-rank fusion, cross-encoder
reranking, and parent expansion. The latency is retrieval-only, not complete
agent latency.

Final-only exact-parent metrics are: hit/recall `0.9286`, precision `0.2321`,
MRR `0.6786`, and nDCG `0.7401`. A flat baseline has no parent IDs, so its
parent metrics are undefined rather than zero.

## Important limitations visible in the run

- `gold_05` retrieved the correct 2008 Directive but missed the exact relevant
  parent passage. Its document hit is 1, parent hit is 0, and final
  faithfulness is 0. This demonstrates why document-level hit alone is
  insufficient.
- `gold_10` answered `93%` while the reference was `92.5%`. Higher-ranked
  context contained the rounded value while the intended source passage ranked
  fourth.
- `gold_11` answered `67%` instead of the requested annual value `60.7%`,
  confusing an adjacent daily statistic with the annual statistic. Its four
  RAGAS scores remained high because those metrics do not directly measure
  reference-answer correctness.
- This is one judge run over 14 documentary questions, without repeated-judge
  confidence intervals. It does not exercise MCP tool selection, structured
  measurement tools, guardrail rejection, cost, or end-to-end latency. Those
  belong to the separate operational benchmark and deterministic tests.

Future evaluation should add answer-correctness or deterministic
numeric-reference checks, while keeping this accepted result unchanged.

## Preservation

A byte-identical snapshot is stored under
`archive/2026-07-23_full_ragas/`.

| Artifact | SHA-256 |
| --- | --- |
| `evaluation_summary.json` | `014C1F8E3681B66DE7A5882A54C3F5E60D097F88C562EE31956AD11904FC9084` |
| `evaluation_details.jsonl` | `DF9453A599F0A4AD6A72C0B964A3038BADE5DEEEAC9DA4C35BCA3BD2F81BAF5C` |

The saved artifacts do not record the resolved model IDs, provider endpoint,
package versions, total evaluator duration, generation/judge latency, tokens,
or USD cost. These quantities are unavailable from the saved artifacts.
