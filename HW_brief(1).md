# 📝 Homework Brief — Production AI Agent

> **Groups of 2–3 · GitHub repo + report · Due thursday, july 23, at 23:59**
> **Late submissions receive 0. No exceptions.**

---

## The task

Build a **complete production AI agent** that addresses a real-world problem. Your agent must integrate everything covered in this course — not as a demonstration, but as a working system.

The difference from the first deliverable: you now have production-grade retrieval (hybrid search + reranking), a working MCP server, an injection-tested guardrail stack, and a reasoning strategy. Your homework agent uses all of these.

---

## What you are building

A research or analysis agent that:
- Uses **hybrid search** (BM25 + dense + RRF) for retrieval
- Has **cross-encoder reranking** applied before context assembly
- Exposes its tools via a **custom MCP server** (at least 3 tools)
- Has **L1 input filtering** and **L4 action gating** implemented
- Uses **few-shot CoT** with EVIDENCE / ANALYSIS / CONCLUSION / CONFIDENCE format
- Applies **Self-Consistency k=3** on the final synthesis step
- Is **instrumented with Langfuse** — every LLM call and tool call has a span
- Has a **second agent role** — at minimum a critic that checks the output before returning it

---

## Topic list

Choose one. First registered, first served. If your topic is taken, choose another — every topic has enough depth.

| # | Topic | Example angle |
|---|-------|--------------|
| 1 | 🌊 Climate displacement | Risk prediction by region · relocation corridor analysis · aid allocation research |
| 2 | 🦠 Pandemic preparedness | Early signal detection from news + preprints · situation report generation |
| 3 | 🍎 Food security | Crop failure early warning · supply chain disruption analysis |
| 4 | 🏙️ Urban migration | Push/pull factor analysis · receiving city capacity assessment |
| 5 | ♻️ Environmental monitoring | Deforestation tracking · pollution event synthesis |
| 6 | 💊 Drug resistance | AMR trend monitoring · treatment protocol research |
| 7 | 📰 Disinformation detection | Narrative origin tracing · cross-source claim verification |
| 8 | ⚖️ AI governance | EU AI Act compliance research · regulation comparison across jurisdictions |

---

## Required repository structure

```
your-repo/
├── README.md              # setup instructions + architecture description
├── REPORT.md              # the written report (see structure below)
├── requirements.txt       # pinned dependencies, runs clean from scratch
├── .env.example           # all required keys listed, no values
├── src/
│   ├── agent.py           # main agent loop
│   ├── mcp_server.py      # your MCP server (3+ tools)
│   ├── retrieval.py       # hybrid search + reranking
│   ├── guardrails.py      # L1 filter + L4 gate + TokenBudget
│   └── reasoning.py       # few-shot CoT + self-consistency
├── tests/
│   └── test_security.py   # 5 injection tests — must all pass
├── docs/
│   └── architecture.md    # architecture diagram + component descriptions
└── data/
    └── README.md          # describes what to put here + how to populate
```

**The instructor will clone your repository and run it.** If it does not run from a fresh clone following your README, it is graded as non-functional.

```bash
# Your README must enable this exact sequence:
git clone your-repo
cd your-repo
cp .env.example .env    # student fills in their own keys
pip install -r requirements.txt
python src/agent.py     # agent runs and produces output
```

---

## REPORT.md structure

Maximum 4 pages. No filler. Every section must be specific to your project.

### 1. Problem statement (½ page)
Who is the user? What does your agent do that a chatbot or search engine cannot? One concrete scenario where your agent produces a useful output that would otherwise take hours.

### 2. Architecture (½ page + diagram)
Describe each component and its role. Your diagram must match the running code. Explain one design decision that was not obvious — why did you make that choice?

### 3. Evaluation (1 page)
Present your RAGAS table: baseline (before Block 1 improvements) vs final (after all improvements). For each metric that improved, explain which technique caused the improvement. For each metric that did not improve, explain why.

| Metric | Baseline | Final | Technique that caused the change |
|--------|---------|-------|----------------------------------|
| context_recall | | | |
| context_precision | | | |
| faithfulness | | | |
| answer_relevancy | | | |

Also report: average run cost (USD), average latency (seconds), and tool call distribution (how many times each tool was called across 10 test runs).

### 4. Security (½ page)
List your 5 injection test results (before and after L1+L4). Describe one real injection attempt your system blocked and explain exactly which layer caught it.

### 5. EU AI Act assessment (½ page)
What risk tier is your agent (prohibited / high / limited / minimal)? Justify with reference to the specific criteria. What obligation does this tier create, and how did you implement it?

### 6. Limitations & what's next (½ page)
What would break first in production? What would you add in the next sprint? Be specific — "improve the agent" is not a limitation.

### 7. AI use disclosure
| Component | Written by human | AI-assisted | AI-generated |
|-----------|-----------------|-------------|-------------|
| Problem statement | | | |
| Architecture | | | |
| Core agent loop (agent.py) | | | |
| MCP server (mcp_server.py) | | | |
| Guardrails (guardrails.py) | | | |
| Retrieval pipeline | | | |
| Report text | | | |

Be honest. You will be asked to explain any function in your codebase.

---

## Submission

1. Push everything to `main` before the deadline
2. Email the repository URL to the instructor with subject: `[PGE5 HW] Group N — Topic name`
3. The instructor clones and runs the repository after the deadline

**Deadline: exactly 4 days from today at 23:59.** Commits after the deadline are not counted — the instructor clones at 23:59.

---

## What a strong submission looks like

A strong submission has:
- A specific problem with a real user (not "it answers questions about X")
- RAGAS scores that improved from baseline, with the technique identified
- All 5 injection tests passing
- A Langfuse trace with at least 5 spans visible (agent + 2 LLM calls + 2 tool calls)
- A critic agent that produces a visible verdict on the output
- A report where every claim is backed by a number or a code reference

A weak submission has:
- A vague problem statement ("help people learn about climate change")
- Missing RAGAS baseline (no comparison possible)
- Security tests not run or not all passing
- A report that describes the agent in general terms without specifics
- Code that does not run from a clean clone
