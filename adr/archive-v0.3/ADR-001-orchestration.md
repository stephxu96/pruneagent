# ADR-001: Orchestration — LangGraph Agent Graph with Parallel Probes and Experiment-Choice Branches

- **Status:** Proposed (v0.3; supersedes v0.2 prune / quantize branches)
- **Date:** 2026-10-30
- **Deciders:** Steph Xu
- **Related:** ADR-002, ADR-004, ADR-005; ARCHITECTURE.md §4

## Context

PruneAgent automates the researcher's sensitivity-analysis workflow. The value of the agentic architecture is that the agent chooses the next experiment (probe, apply, start from OWL, roll back, stop) instead of a fixed script or a person. Layer-local sensitivity probes are independent per layer and can run in parallel. Code must still enforce size, quality and budget limits. A run lasts 20–40 rounds and must survive a Colab disconnect. Course guidance: name who chooses the next step; code enforces gates.

## Options

| Option | Who chooses the next step | Fit | Cost |
|---|---|---|---|
| A. Fixed script (sweep, then choose) | Code | The manual workflow automated as-is; no adaptive experiment choice | ~1 day |
| **B. LangGraph state graph** | LLM agent chooses the experiment; code gates | `Send` fan-out + reducer for parallel probes; explicit branches per agent action; invalid / rollback edges; SQLite checkpointer for resume | ~1.5 days incl. learning |
| C. ReAct agent with tools | LLM | The LLM could skip evaluation or rollback; limits become requests, not rules | ~1 day; hard to trust |
| D. Reinforcement learning (AMC, FastForward) | Learned policy | GPU-hours per model | Exceeds the compute budget |

## Decision

**B.** `ORCHESTRATION: hybrid. The LLM agent chooses the next experiment and per-layer sparsity. Code owns probing, validation, pruning, the per-step and cumulative perplexity gate, and accept / rollback.` LLM calls go through the project's own client (cache, cost meter, rationale log).

## Consequences

- Positive: parallel probes cut wall-clock time on one GPU; every agent choice is an explicit, logged branch; limits cannot be skipped; a disconnect loses at most one round.
- Negative: a dependency to pin and learn; GPU-bound full-model evaluations stay sequential.

## Evidence That Would Reverse This Decision

- Probes show no speed-up when run concurrently and the agent uses only `apply` → option A with a hand-built loop.
- LangGraph checkpoints prove unreliable on Google Drive → option A with plan JSON saved each round.
