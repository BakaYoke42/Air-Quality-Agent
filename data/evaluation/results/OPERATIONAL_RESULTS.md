# End-to-end operational benchmark

The accepted operational benchmark contains ten completed agent runs using
agent version `0.6.0`. Every question went through L1, live MCP discovery,
Mistral tool selection from the four live names/schemas/docstrings, L4,
Streamable HTTP execution, three concurrent syntheses, deterministic citation
validation, and the critic. The `expected_tools` fields were used only for
post-run scoring and were never sent to the planner.

## Accepted measurements

| Measure | Result |
| --- | ---: |
| Completed questions | 10/10 |
| Mean end-to-end latency | 19.8555 s |
| TokenBudget-estimated mean USD cost | $0.000000 |
| Mean input tokens | 22,948.6 |
| Mean output tokens | 1,254.8 |
| Mean total tokens | 24,203.4 |
| Mean counted LLM calls | 5.5 |
| Mean MCP calls | 1.4 |
| Strict expected-tool-set match | 8/10 (80%) |
| Final answers with present, allowlisted citations | 10/10 |

The USD figure is the TokenBudget estimate under the configured zero
per-million prices for the user's free Mistral plan. It is not a general price
claim about Mistral.

## Tool-call distribution

| MCP tool | Calls |
| --- | ---: |
| `search_air_quality_evidence` | 5 |
| `get_country_air_quality` | 2 |
| `compare_countries` | 5 |
| `find_station_extremes` | 2 |
| **Total** | **14** |

All four tools were selected by the model. Five documentary calls generated
HyDE successfully (1,379 tokens in total, no fallback). The ten planner calls,
30 synthesis calls, and ten critic calls are recorded separately; the five
successful HyDE calls account for the difference between 50 direct agent model
records and 55 TokenBudget-counted LLM calls.

## Manual review

- `op_06` made a defensible extra documentary search for the EU 2030 threshold
  in addition to the expected country-comparison tool.
- `op_10` used two single-country `compare_countries` calls plus documentary
  search instead of `get_country_air_quality` plus search. The plan was
  redundant but returned the requested evidence. These two cases remain strict
  mismatches; the labels were not changed after seeing the output.
- All critic verdicts were `REVISED`. This is a successful repair state, not a
  failed run: the strict critic mainly removed repeated raw values, and the
  deterministic parser forbids `PASS` if the critic changes a selected draft.
- Manual review of an earlier version exposed an incomplete distribution
  summary and wording that could imply 2024 non-compliance with a future 2030
  standard. Version `0.6.0` now requires requested distribution/coverage fields
  and calls future thresholds benchmarks until they apply. The accepted
  `op_03` and `op_10` conclusions reflect those corrections.

The checkpointed final measurement was resumed, so it includes two cold
retriever/model starts. One initial `op_10` attempt ended with a transient MCP
task-group error before a result snapshot existed; the runner preserved the
other nine records and retried only that case successfully. Accepted latency,
token, and cost aggregates use the ten completed result records and exclude
that failed attempt.

## Preservation

The accepted files are copied under
`archive/2026-07-23_operational_agent_v060/`.

| Artifact | SHA-256 |
| --- | --- |
| `operational_summary.json` | `645B375649EAD77FB8B29CBCD7198D924DDA765765A825DFA6CC024052DB4E39` |
| `operational_details.jsonl` | `82D12C199D195D4D730BF283EC3A8A01BD3C22F8070670D0CE9061BD9FD5B49D` |
