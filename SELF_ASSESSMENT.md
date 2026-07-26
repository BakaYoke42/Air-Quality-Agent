# Rubric Self-Assessment

**Authors:** Ahmed Aziz Ben Aissa, Thibault Goutorbe, Baptiste LANGLOIS, and
Tong Li

**Assessment date:** 23 July 2026

## Repository gate

| Gate requirement | Verified evidence | Status |
| --- | --- | :---: |
| Public and accessible repository | An unauthenticated HTTPS request to `github.com/zaizou1003/Air-Quality-Agent` returned HTTP 200 on 23 July 2026 | **PASS** |
| `pip install -r requirements.txt` | Fully pinned requirements; clean-clone `--dry-run --no-index` and `pip check` succeeded | **PASS** |
| `python src/agent.py` produces output | The exact no-argument command completed an end-to-end run and intentionally used the documented demonstration question | **PASS** |
| Security test imports and runs | `tests/test_security.py`: 16/16 passed, including all five required attacks | **PASS** |

The repository gate is satisfied.

## Score by criterion

| Criterion | Self-score | Evidence and justification |
| --- | ---: | --- |
| **A. Retrieval pipeline** | **15/15** | `src/retrieval.py` implements BM25+dense hybrid retrieval, RRF, cross-encoder reranking, parent-child expansion, and unique-parent selection. The controlled 14-question comparison in `data/evaluation/results/RESULTS.md` shows improvement on all four RAGAS metrics. |
| **B. MCP server** | **10/10** | `src/mcp_server.py` exposes four Streamable HTTP tools with complete “Use when / Do NOT use / Returns / Example” descriptions and controlled error envelopes. `MCP_INSPECTOR_RESULTS.md` records successful discovery and calls for all four tools. |
| **C. Security stack** | **10/10** | `src/guardrails.py` contains Unicode-aware L1 filtering, the L4 action-risk matrix, and integrated TokenBudget controls. `SECURITY_RESULTS.md` records 16/16 passing tests, including the five required attacks and the deliberate budget rejection before dispatch. |
| **D. Reasoning strategy** | **10/10** | `src/reasoning.py` contains a worked few-shot example, the required EVIDENCE/ANALYSIS/CONCLUSION/CONFIDENCE contract, exactly three concurrent synthesis voices, and an independent critic. Citation allowlists are also validated deterministically. |
| **E. Observability** | **5/5** | `src/observability.py` instruments the agent, planner, LLM calls, MCP calls, critic, and HyDE while recording agent version `0.6.0`. Langfuse delivery was verified in the web dashboard, and the report defines a p95 latency alert. A sanitized dashboard screenshot should be retained with the submission evidence. |
| **F. RAGAS baseline and improvement** | **12/12** | Four required metrics were run on 14 questions with the answer model, judge, embeddings, prompt, `top_k`, and HyDE policy held constant. Baseline/final artifacts and per-question results are preserved, with technique-linked explanations and disclosed anomalies. |
| **G. Cost and latency reporting** | **7/8** | The real LLM-selected-tool agent completed ten runs. `OPERATIONAL_RESULTS.md` reports 19.8555 s mean latency, mean tokens, all 14 tool calls by tool, and the deliberate TokenBudget trigger. One point is withheld because `$0.000000` is a configured TokenBudget estimate for the free-plan settings, not independent provider billing evidence. |
| **H. Problem statement and architecture** | **8/8** | `REPORT.md` names a municipal/regional policy analyst, a concrete cross-country briefing, and an estimated 2–4-hour manual workflow. Its diagram matches the running HTTP/LLM/MCP/retrieval/reasoning path and explains the trade-off between LLM tool selection and deterministic execution authority. |
| **I. EU AI Act assessment** | **6/6** | The report gives a limited/transparency-risk assessment using Article 6/Annex III criteria, derives the Article 50 disclosure obligation, and points to the first-interaction and persistent disclosures implemented in Streamlit. |
| **J. Limitations and next steps** | **6/6** | The report describes specific failure conditions for citation entailment, numeric correctness, latency/tool over-selection, geographic/year scope, and unauthenticated deployment, each paired with a concrete technical next step. |
| **K. AI disclosure and ownership** | **9/10** | The report explicitly distinguishes AI-assisted from AI-generated work and explains the human direction, testing, modification, and review. One point is withheld because full code ownership can only be established if every group member can explain the relevant functions under follow-up questioning. |
| **Total after repository gate passes** | **98/100** | Technical 50/50 + Evaluation 19/20 + Report 20/20 + Transparency 9/10. |

## Known evidence limitations

- The RAGAS study is one 14-question judge run without component ablations or
  repeated-judge confidence intervals.
- RAGAS does not directly guarantee reference-answer correctness; one saved
  case selected an adjacent daily percentage instead of the requested annual
  value.
- All ten operational critic outcomes were `REVISED`. They produced validated
  repaired answers, but none of those ten drafts was accepted unchanged.
- Strict expected-tool-set agreement was 8/10 because the LLM selected
  defensible or redundant additional tools in two cases.
- Deterministic citation validation proves that IDs belong to the run's
  allowlist; it does not itself prove claim-level entailment.
- The OpenAQ code under `new_tool/` is an isolated future prototype and is not
  counted as a fifth production tool or as part of the reported evaluations.
