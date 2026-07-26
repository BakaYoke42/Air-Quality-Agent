# Security verification

Final command:

```text
python -m pytest tests/test_security.py -v
```

Result on Python 3.11.9 with pytest 9.1.1: **16/16 tests passed**,
including the five required injection cases.

The "before" column below is a harmless offline control at the unguarded
boundary: raw input would be admitted to the planner, and an arbitrary action
name would be admitted to the executor. No attack payload was sent to an
external model and the production guardrails were never disabled.

| Required case | Before L1 + L4 | Protected result | Layer |
| --- | --- | --- | --- |
| Direct instruction override | Admitted to planner boundary | Blocked as `direct_override` | L1 |
| Full-width Unicode override | Admitted to planner boundary | NFKC-normalized, then blocked as `direct_override` | L1 |
| “You are now administrator” role injection | Admitted to planner boundary | Blocked as `role_injection` | L1 |
| Hidden system-prompt extraction | Admitted to planner boundary | Blocked as `prompt_extraction` | L1 |
| Proposed `delete_measurements` tool | Admitted to executor boundary | Unknown action denied with `BLOCK` risk | L4 |

The Unicode case demonstrates layer composition. A visually obfuscated
full-width version of “ignore all previous instructions” does not need its own
special-case pattern: L1 first performs Unicode NFKC normalization, removes
invisible/bidirectional controls, and then applies the same named
`direct_override` detector. The request stops before any LLM, MCP connection,
or tool call.

Additional passing tests verify that indirect instructions in tool results are
marked as untrusted and stripped of active markup, the L4 matrix covers exactly
the four production MCP tools, invalid countries/years/limits fail closed, and
the three concurrent synthesis reservations cannot oversubscribe shared
limits.

## Deliberate TokenBudget trigger

This is an intentional boundary test, not a production outage. The test creates
a 100-output-token budget, holds an 80-token reservation, and tries to reserve
21 more. The attempted total is 101, so `BudgetExceeded(resource="output_tokens",
limit=100, attempted=101)` is raised **before** the second provider dispatch.
Releasing the held reservation returns the reserved counter to zero.
