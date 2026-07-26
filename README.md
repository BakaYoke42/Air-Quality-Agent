# European Air-Quality Evidence Agent

**Authors:** Ahmed Aziz Ben Aissa, Thibault Goutorbe, Baptiste LANGLOIS, and
Tong Li

**Repository:** **Repository:** [https://github.com/BakaYoke42/Air-Quality-Agent] (https://github.com/BakaYoke42/Air-Quality-Agent)

This project is a guarded AI research agent for comparing 2024 PM2.5 and NO2
sampling-point measurements in France, Germany, and Italy with authoritative
WHO guidance, EU legal limits, and EEA methodology.

The intended user is a European environmental-policy analyst preparing a
cross-country briefing. For example, the analyst can ask how 2024 PM2.5
sampling-point results in France, Germany, and Italy compare with WHO health
guidance and EU law. The agent combines reproducible calculations with
retrieved source passages and a critic-checked cited answer, replacing manual
source lookup and spreadsheet aggregation that would otherwise take hours.

It combines:

- hybrid BM25 and dense retrieval with reciprocal-rank fusion;
- parent-child chunks and cross-encoder reranking;
- optional HyDE query expansion;
- four tools exposed by a standalone MCP Streamable HTTP server;
- LLM-selected tools using live MCP schemas and docstrings;
- L1/L4 guardrails and a concurrency-safe TokenBudget;
- three parallel synthesis voices followed by an independent critic;
- deterministic evidence-ID allowlist validation;
- failure-isolated Langfuse tracing.

See [REPORT.md](REPORT.md) for the project report,
[SELF_ASSESSMENT.md](SELF_ASSESSMENT.md) for the rubric self-assessment,
[docs/architecture.md](docs/architecture.md) for the detailed control flow,
and [data/README.md](data/README.md) for provenance and regeneration.

## Supported scope

- Countries: France (`FR`), Germany (`DE`), and Italy (`IT`)
- Pollutants: PM2.5 and NO2
- Structured-measurement year: 2024
- Documentary evidence: WHO, EU, EEA, and ETC HE sources

Results are unweighted summaries across retained reporting sampling points.
They are not population-weighted exposure estimates, city-level estimates, or
formal legal-compliance determinations.

### Measurement snapshot

- Temporal coverage: `2024-01-01 00:00` to `2024-12-31 00:00`
- Known missing period: the final 24 hours of 2024
- Minimum retained coverage: 75%
- Accepted validity codes: 1, 2, and 3
- Accepted verification code: 1
- Unit: µg/m³
- Retained sampling-point/pollutant series: 2,019
- Excluded low-coverage series: 86

## Architecture summary

```text
User question
  -> local L1 injection/scope preflight
  -> reuse or auto-start the local MCP HTTP service
  -> MCP tools/list over Streamable HTTP
  -> Mistral selects tools from live descriptions and JSON schemas
  -> TokenBudget + allowlist + L4 argument validation
  -> MCP tool execution
  -> retrieved parent passages D1, D2, ...
     structured tool-result payloads M1, M2, ...
  -> explicit ALLOWED EVIDENCE IDS
  -> three parallel synthesis calls
  -> deterministic draft citation validation
  -> critic
  -> deterministic critic citation validation
  -> final answer
```

MCP execution records and evidence IDs use separate namespaces. Each retrieved
parent passage receives its own run-local `D#`, even when several passages come
from the same source document. Each structured measurement tool-result payload
receives one `M#`; a payload may contain several countries or stations.
Deterministic validation rejects bracketed IDs outside the explicit allowlist,
but ID validity alone does not prove that a claim is entailed by the cited
evidence. An unsupported citation in the critic response forces `REVISED`,
never `PASS`.

## Requirements

- Python 3.11
- Internet access for Mistral and initial model downloads
- A Mistral API key, or another OpenAI-compatible tool-capable provider
- Node.js only for optional MCP Inspector use

`mistral-medium-latest` supports the function-calling workflow used here.

## Installation

Minimum clean-clone sequence:

```bash
git clone https://github.com/zaizou1003/Air-Quality-Agent.git air-quality-agent
cd air-quality-agent
cp .env.example .env
# Fill in OPENAI_API_KEY in .env.
pip install -r requirements.txt
python src/agent.py
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install --upgrade pip
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and add `OPENAI_API_KEY`. Keep `.env` private.

### Linux or macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

## Running the agent

After installation, the cross-platform entry point is:

```bash
python src/agent.py
```

When no question is provided, this command intentionally runs the built-in
France/Germany/Italy PM2.5 demonstration question so that a clean-clone check
immediately produces a complete agent answer. To ask a different question,
pass it as the positional argument, as shown below.

The equivalent explicit Windows virtual-environment invocation is:

```powershell
.\.venv311\Scripts\python.exe src\agent.py
```

For the default loopback `/mcp` URL, the agent reuses an existing server or
starts `src/mcp_server.py`, waits until it is ready, communicates exclusively
over Streamable HTTP, and stops only the child process it created. Remote and
HTTPS MCP endpoints remain externally managed.

Starting a persistent server is optional and avoids reloading retrieval models
between repeated CLI or frontend runs:

```powershell
.\.venv311\Scripts\python.exe src\mcp_server.py
```

Example custom question:

```powershell
.\.venv311\Scripts\python.exe src\agent.py --show-drafts "Compare 2024 PM2.5 in France, Germany and Italy against the WHO annual guideline."
```

Useful flags:

- `--no-hyde`: disable HyDE and use the original retrieval query only.
- `--show-drafts`: print all three independent synthesis drafts.
- `--json`: print the complete machine-readable run record.
- `--mcp-url URL`: override `MCP_SERVER_URL` for one run.

## Frontend

Run the Streamlit interface after installation:

```powershell
.\.venv311\Scripts\python.exe -m streamlit run streamlit_app.py
```

The browser UI uses the same guarded agent and Streamable HTTP MCP path as the
CLI. It presents the critic-checked answer, D#/M# evidence, MCP calls as separate
provenance records, HyDE status, TokenBudget usage, model-call timing, and the
three synthesis drafts. It also displays an explicit AI-use disclosure. For the
fastest repeated queries, keep the optional persistent MCP server running.

## LLM-selected MCP tools

The model receives each live MCP tool name, input schema, and docstring. The
docstrings contain `Use when`, `Do NOT use`, `Returns`, `Prefer`, and `Example`
guidance.

| Tool | Purpose |
| --- | --- |
| `search_air_quality_evidence` | WHO/EU/EEA documents, definitions, law, methodology, and interpretation |
| `get_country_air_quality` | Exact 2024 summary for one country and pollutant |
| `compare_countries` | Exact cross-country comparison and ranking |
| `find_station_extremes` | Highest or lowest retained sampling-point annual means |

The LLM proposes calls, but deterministic code remains authoritative: unknown
tools and invalid countries, pollutants, years, benchmarks, limits, or injected
queries are blocked before execution.

## HyDE

HyDE runs only when the model selects `search_air_quality_evidence`. A
structured-only country question normally uses a measurement tool and therefore
does not make a HyDE call.

For documentary retrieval, the server reports whether HyDE was `generated`,
`disabled`, or used `fallback`, plus the model, duration, generated character
count, token usage, and safe error text. The hypothetical passage is never
returned as evidence. It defaults to a 45-second timeout and zero retries, then
falls back safely to the original question.

Because HyDE executes inside the MCP server, its provider-reported usage is
returned in the MCP payload and added to the run budget after the external HyDE
call. HyDE still has its own timeout and safe fallback.

## Reasoning and call count

Self-consistency is exactly `k=3`. The three synthesis calls run concurrently;
the critic runs afterward.

A normal structured query therefore makes five LLM calls:

```text
1 tool-selection call + 3 synthesis calls + 1 critic = 5
```

A successful documentary HyDE expansion adds one more LLM call. The CLI prints
this breakdown explicitly.

## TokenBudget

The per-run budget limits LLM calls, MCP calls, input/output tokens, and an
optional estimated USD amount. Default limits are documented in `.env.example`.
Prices can remain zero for a free provider while call and token limits stay
active.

The TokenBudget boundary-condition test is
`tests/test_security.py::test_token_budget_deliberately_triggers_before_excess_output_is_reserved`.
With a 100-output-token ceiling and 80 tokens already reserved, a second
21-token reservation is rejected before dispatch because it would attempt 101
tokens. Releasing the first reservation restores reserved usage to zero.

## Langfuse

Set these values in `.env`:

```text
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=...
```

Check authentication:

```powershell
.\.venv311\Scripts\python.exe src\observability.py --check
```

Traces include the top-level agent, tool selection, MCP tools, optional HyDE,
three synthesis generations, the critic, token usage, and latency.

Langfuse authentication and web-dashboard ingestion were verified in the
configured development environment. A traced run shows the
top-level agent observation with `agent_version`, tool selection, executed MCP
calls, three overlapping synthesis generations, the critic, and HyDE when
enabled. Credentials remain only in the local `.env`.

Proposed monitoring alert: alert when p95 `total_latency_ms` exceeds 90,000 ms
across the latest 20 successful runs. Sustained breaches indicate provider delay,
cold retrieval models, or excessive tool planning. This is the alert policy for
the report; it is not presented as an already configured dashboard alert.

## Tests

Run the complete default suite:

```powershell
.\.venv311\Scripts\python.exe -B -m pytest -p no:cacheprovider -q
```

Security only:

```powershell
.\.venv311\Scripts\python.exe -B -m pytest -p no:cacheprovider tests\test_security.py -v
```

Citation validation only:

```powershell
.\.venv311\Scripts\python.exe -B -m pytest -p no:cacheprovider tests\test_reasoning_citations.py -v
```

The live neural-retrieval test is opt-in:

```powershell
$env:RUN_RETRIEVAL_INTEGRATION = "1"
.\.venv311\Scripts\python.exe -B -m pytest -p no:cacheprovider tests\test_mcp_server.py::test_mcp_server_retrieval_tool_optional -v
Remove-Item Env:RUN_RETRIEVAL_INTEGRATION
```

The root suite completed with **75 passed and 1 skipped**. The isolated OpenAQ
prototype completed its own **18 passed** suite, and `python -m pip check`
reported no broken requirements.

## MCP Inspector

With the server running:

```powershell
npx @modelcontextprotocol/inspector --cli http://127.0.0.1:8000/mcp --transport http --method tools/list
```

The Inspector CLI `0.17.2` check succeeded over Streamable HTTP and
returned exactly the four tools listed above, with their JSON schemas and full
selection descriptions. Each tool was then called successfully, and an invalid
`ES` country produced a controlled `invalid_arguments` response rather than an
uncaught exception. See
[MCP_INSPECTOR_RESULTS.md](data/evaluation/results/MCP_INSPECTOR_RESULTS.md).

The verification host used Node `20.19.2`, while that Inspector release
declares Node `>=22.7.5`. npm emitted an engine warning, but every Inspector
command exited with status 0. Upgrade Node for warning-free future Inspector
use; the Python application itself does not depend on Node.

## Evaluation

There is one baseline-versus-final evaluator for the 14-question root golden
dataset:

```powershell
# Offline retrieval audit: no answer or judge calls
.\.venv311\Scripts\python.exe tests\evaluate_retrieval.py --retrieval-only

# Full controlled comparison with all four required RAGAS metrics
.\.venv311\Scripts\python.exe tests\evaluate_retrieval.py
```

The baseline is dense cosine retrieval over flat 500-word chunks. The final
pipeline is BM25 + dense + RRF + child reranking + unique-parent expansion;
HyDE is off by default so the comparison is repeatable (`--hyde` enables it).
The full run holds the answer prompt/model and RAGAS judges constant across both
pipelines, then reports `context_recall`, `context_precision`, `faithfulness`,
and `answer_relevancy`. It also retains deterministic document hit/precision/
recall, MRR, nDCG, parent metrics, and retrieval latency.

The evaluator prints answer and per-metric progress and uses bounded
concurrency.
`RAGAS_MAX_CONCURRENT_PIPELINES` defaults to `2` and
`RAGAS_MAX_CONCURRENT_LLM_GROUPS` to `3`; set both to `1` if a provider imposes
a particularly strict rate limit.

Outputs are saved to `data/evaluation/results/evaluation_summary.json` and
`evaluation_details.jsonl`. The saved full run finished on 2026-07-23 and
contains all 14 unique questions, non-empty baseline/final answers, and all four
required scores:

| Metric | Baseline | Final | Change |
| --- | ---: | ---: | ---: |
| Context recall | 0.7143 | 1.0000 | +0.2857 |
| Context precision | 0.5952 | 0.7480 | +0.1528 |
| Faithfulness | 0.9107 | 0.9286 | +0.0179 |
| Answer relevancy | 0.7110 | 0.9663 | +0.2553 |

The final pipeline's hybrid retrieval and parent expansion improved coverage;
RRF plus cross-encoder reranking improved the concentration and ordering of
useful context. Mean retrieval-only latency increased from `12.72 ms` to
`1302.56 ms` (`102.4x`), which is the explicit quality/latency trade-off.
Supplementary document recall@4 improved from `0.9286` to `1.0000`, MRR from
`0.7024` to `0.9167`, and nDCG from `0.7610` to `0.9379`.

The result is preserved under
`data/evaluation/results/archive/2026-07-23_full_ragas/`; the full audit and
artifact hashes are in
[data/evaluation/results/RESULTS.md](data/evaluation/results/RESULTS.md).
Important caveats remain: `gold_05` missed its exact relevant parent,
`gold_10` returned a rounded `93%` instead of `92.5%`, and `gold_11` selected
the adjacent daily `67%` statistic instead of the requested annual `60.7%`.
The four required RAGAS metrics do not directly test reference-answer
correctness, and this single 14-question judge run has no repeated-judge
confidence intervals.

The separate end-to-end benchmark also completed with agent version `0.6.0`.
All ten questions passed through the real LLM planner using the live MCP names,
schemas, and docstrings; expected-tool labels were used only after each run for
scoring.

| Operational measure | Result |
| --- | ---: |
| Completed questions | 10/10 |
| Mean end-to-end latency | 19.8555 s |
| TokenBudget-estimated mean USD cost | $0.000000 |
| Mean input / output / total tokens | 22,948.6 / 1,254.8 / 24,203.4 |
| Mean counted LLM calls | 5.5 |
| Mean MCP calls | 1.4 |
| Strict expected-tool-set match | 8/10 (80%) |
| Answers with present, allowlisted citations | 10/10 |

The model selected all four tools: five documentary searches, two country
summaries, five country comparisons, and two station-extremes calls (14 total).
The zero-dollar figure is the TokenBudget estimate under the configured
zero-price free-plan settings, not a general Mistral pricing claim. Full run
notes, including the two strict tool-set mismatches and the `REVISED` critic
outcomes, are in
[OPERATIONAL_RESULTS.md](data/evaluation/results/OPERATIONAL_RESULTS.md).

The final security run passed all 16 security tests, including the five
required attack cases: L1 blocked four input attacks and L4 blocked the
unknown `delete_measurements` action. The deliberate TokenBudget boundary test
also passed by rejecting an attempted 101 tokens against a 100-token ceiling
before provider dispatch. See
[SECURITY_RESULTS.md](data/evaluation/results/SECURITY_RESULTS.md).

The 14 questions meet the `>=10` requirement for the documentary RAGAS run, but
not for testing the complete agent. They do not cover structured country
measurements, country comparisons, station extremes, mixed document/measurement
tool plans, rejected injection/out-of-scope/year requests, invented citations,
or a budget failure. Those behaviors belong in the existing deterministic test
suite plus a separate mixed ten-run operational set; adding rejection prompts
to the RAGAS golden set would incorrectly score a safe refusal as a bad answer.

## Limitations and next sprint

- Country comparisons are unweighted across retained monitoring points.
  Unequal station placement can distort comparisons. A next version should add
  population- or grid-weighted exposure estimates while preserving the current
  sampling-point view.
- Structured data covers only France, Germany, Italy, PM2.5, NO2, and 2024.
  The year preflight now distinguishes documentary publication, methodology,
  guideline, directive, and target years from explicit country-measurement
  requests; L4 still enforces `year=2024` authoritatively. Because the preflight
  is a conservative language heuristic, further multilingual and adversarial
  date-routing tests remain useful.
- Citation validation proves run-scoped ID membership, not claim-level
  entailment; a real but irrelevant `D#` could still be attached to a claim.
  Sentence-level citation-entailment verification is a concrete next step.
- The first documentary request can be slow because the embedding and
  cross-encoder models load lazily. Warm persistent serving, retrieval caching,
  and a measured latency service-level objective would address cold starts.
- The MCP endpoint has no authentication or TLS and is safe only on loopback.
  Remote deployment requires an authenticated reverse proxy, TLS, and rate
  limiting.

## Repository layout

```text
src/                    agent, MCP server, retrieval, reasoning, guardrails
scripts/                corpus and measurement preparation
tests/                  offline, security, citation, and HTTP tests
data/                   processed corpus, index, measurements, eval outputs
docs/architecture.md    detailed system architecture
new_tool/               isolated experimental live OpenAQ MCP prototype
streamlit_app.py        browser frontend over the shared agent pipeline
golden_dataset.jsonl    evaluation questions and references
requirements.txt        pinned dependencies
.env.example            safe configuration template
```

`new_tool/openaq_air_quality_tool/` remains a standalone experimental
prototype. It is not registered with the main agent, is not one of the four
graded MCP tools, and was not part of the reported evaluations.

## Troubleshooting

- Local MCP startup timeout: run `src/mcp_server.py` directly to inspect its
  startup log, or increase `MCP_AUTO_START_TIMEOUT_SECONDS`.
- Remote MCP connection refused: verify `MCP_SERVER_URL`; remote/HTTPS services
  are never auto-started by the local agent.
- First documentary query is slow: embedding and reranker models are loading.
- HyDE fallback: inspect the MCP server's explicit HyDE status/error line.
- `Terminating session: None`: normal stateless Streamable HTTP cleanup.
- Locally rejected questions never open HTTP, preventing the former immediate
  `ClientDisconnect` traceback.
- Never commit `.env`, virtual environments, or the raw EEA ZIP archives.
