# 📊 Grading Rubric — Homework Project

> **Available now — read before you start.**
> Total: 100 points. No peer coins. No live presentation. The grade comes entirely from the submitted repository and report.

---

## Score breakdown

| Component | Points |
|-----------|--------|
| Technical implementation | 50 |
| Evaluation & measurements | 20 |
| Report quality | 20 |
| AI use transparency | 10 |
| **Total** | **100** |

---

## Technical implementation — 50 pts

### A. Retrieval pipeline `15 pts`

| Score | Description |
|-------|-------------|
| 13–15 | Hybrid search (BM25 + dense + RRF) implemented and working. Cross-encoder reranking applied. Parent-child chunking used. RAGAS shows measurable improvement vs basic retrieval. |
| 9–12 | Hybrid search implemented. Reranking present. Minor issues — e.g. RRF fusion works but parent-child not implemented. |
| 5–8 | One of the three techniques (hybrid / reranking / parent-child) implemented. Basic retrieval for the others. |
| 1–4 | Basic top-k cosine similarity only. No advanced retrieval techniques from Block 1. |
| 0 | No retrieval pipeline. Agent answers from model knowledge only. |

### B. MCP server `10 pts`

| Score | Description |
|-------|-------------|
| 9–10 | MCP server with 3+ tools. All tools have complete docstrings (Use when / Do NOT use / Returns / Example). All tools handle errors gracefully (no uncaught exceptions). MCP inspector tests pass. |
| 6–8 | MCP server with 2+ tools. Docstrings present but incomplete. Error handling on most tools. |
| 3–5 | MCP server exists but tools have minimal descriptions or unreliable error handling. |
| 1–2 | MCP server skeleton only — tools do not actually work. |
| 0 | No MCP server. Tools defined as plain Python functions only. |

### C. Security stack `10 pts`

| Score | Description |
|-------|-------------|
| 9–10 | L1 input filter with injection patterns and unicode normalisation. L4 action gate with ACTION_RISK_MATRIX for all tools. TokenBudget integrated. 5/5 injection tests pass. Tests included in `tests/test_security.py`. |
| 6–8 | L1 and L4 present. 4/5 injection tests pass. TokenBudget present. |
| 3–5 | One of L1 or L4 implemented. Fewer than 4 injection tests pass. |
| 1–2 | Security mentioned in report but minimal code implementation. |
| 0 | No security implementation. Injection tests not run. |

### D. Reasoning strategy `10 pts`

| Score | Description |
|-------|-------------|
| 9–10 | Few-shot CoT with EVIDENCE/ANALYSIS/CONCLUSION/CONFIDENCE format in synthesis system prompt. Self-Consistency k≥3 on the final synthesis step. Confidence tagging used throughout. |
| 6–8 | Zero-shot CoT used. Confidence tagging present. Self-Consistency implemented but k=1 or not connected to synthesis step. |
| 3–5 | "Think step by step" added. No confidence tagging. No Self-Consistency. |
| 1–2 | No CoT. Direct prompting only. |

### E. Observability `5 pts`

| Score | Description |
|-------|-------------|
| 5 | Langfuse traces visible for all runs. Agent, LLM calls, and tool calls each have their own span. Agent version logged. At least one monitoring alert described. |
| 3–4 | Langfuse traces visible. Most spans present. Agent version not logged. |
| 1–2 | Langfuse connected but minimal instrumentation (only top-level trace, no tool spans). |
| 0 | No observability. |

---

## Evaluation & measurements — 20 pts

### F. RAGAS baseline and improvement `12 pts`

| Score | Description |
|-------|-------------|
| 11–12 | RAGAS run on ≥10 questions. All 4 metrics reported (context_recall, context_precision, faithfulness, answer_relevancy). Baseline documented before Block 1 improvements. Final scores show measurable improvement. Each improvement linked to the technique that caused it. |
| 8–10 | RAGAS run on ≥5 questions. 3–4 metrics reported. Baseline present. Improvements documented but not all explained. |
| 5–7 | RAGAS run but fewer than 5 questions. Baseline missing or incomplete. Some metrics reported. |
| 2–4 | RAGAS mentioned but only run once. No baseline comparison. |
| 0–1 | RAGAS not run. No quantitative evaluation of retrieval quality. |

### G. Cost and latency reporting `8 pts`

| Score | Description |
|-------|-------------|
| 7–8 | Average cost per run (USD) reported over ≥10 runs. Average latency (seconds) reported. Tool call distribution reported (how many times each tool was called). TokenBudget triggered at least once during testing (documented). |
| 4–6 | Cost and latency reported. Fewer than 10 runs. Tool distribution not reported. |
| 1–3 | One of cost or latency reported. Testing was minimal. |
| 0 | No cost or latency measurement. |

---

## Report quality — 20 pts

### H. Problem statement and architecture `8 pts`

| Score | Description |
|-------|-------------|
| 7–8 | Problem statement names a specific user and a specific scenario. Architecture diagram matches the running code exactly. One design decision is explained with its trade-off. |
| 5–6 | Problem statement is specific. Diagram mostly matches code. Design decisions described but not justified. |
| 3–4 | Vague problem statement ("helps people learn about X"). Diagram is incomplete or outdated. |
| 1–2 | No real problem statement. No architecture diagram. |

### I. EU AI Act assessment `6 pts`

| Score | Description |
|-------|-------------|
| 6 | Risk tier identified with specific justification referencing the EU AI Act criteria. Obligation derived from the tier and implementation described (e.g. user disclosure in the UI). |
| 4–5 | Risk tier identified. Justification present but generic. Obligation mentioned but not implemented. |
| 2–3 | Risk tier identified without justification. No obligation described. |
| 0–1 | EU AI Act not addressed or clearly wrong (e.g. claims "no regulation applies"). |

### J. Limitations and what's next `6 pts`

| Score | Description |
|-------|-------------|
| 5–6 | Two or more specific limitations identified with the conditions under which they would manifest. "What's next" section is technically concrete (names a specific technique, not "improve the agent"). |
| 3–4 | Limitations listed but generic ("the agent could be more accurate"). |
| 1–2 | One sentence on limitations. No specificity. |
| 0 | No limitations section. |

---

## AI use transparency — 10 pts

### K. Disclosure table + code ownership `10 pts`

| Score | Description |
|-------|-------------|
| 9–10 | AI Usage table filled in honestly and specifically. Report explains what was written vs what was generated vs what was modified. Every function in the codebase can be explained by the group (verified by follow-up questions if the instructor has doubts). |
| 6–8 | Table filled in. Some specificity. One or two functions may be hard to explain under questioning. |
| 3–5 | Table present but vague ("we used AI for some parts"). Several functions not fully understood. |
| 1–2 | Minimal or no disclosure. AI use appears extensive but not acknowledged. |
| 0 | No disclosure. Or: follow-up questions reveal the group did not understand any of the code. |

---

## Repository quality (pass/fail gate)

Before the rubric is applied, the repository must pass this gate:

- [ ] Repository is public and accessible
- [ ] `pip install -r requirements.txt` completes without errors
- [ ] `python src/agent.py` runs and produces output following the README
- [ ] `python -m pytest tests/test_security.py` runs without import errors

**If the repository fails the gate, the maximum score for Technical Implementation is 10/50 (the non-runnable portion).** The report can still be graded normally.

---

## Final score formula

```
technical_implementation  = A + B + C + D + E    (max 50)
evaluation_measurements   = F + G                (max 20)
report_quality            = H + I + J            (max 20)
ai_transparency           = K                    (max 10)

final_score = technical + evaluation + report + transparency
```

Maximum: 100 points.

---

## Grade interpretation

| Score | Interpretation |
|-------|---------------|
| 85–100 | Exceptional — production-grade system, all techniques integrated, honest and specific report |
| 70–84 | Strong — working system, good evaluation, minor gaps |
| 55–69 | Satisfactory — system runs, basic techniques present, limited evaluation depth |
| 40–54 | Weak — partial implementation, missing evaluation, superficial report |
| Below 40 | Insufficient — system does not run, techniques not implemented, or disclosure issues |
