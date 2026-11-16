# ADR-001: Orchestration — LangGraph Agent Graph with Parallel Probes, Parallel Candidate Evaluation and Experiment-Choice Branches

- **Status:** Proposed (v0.6; code-owned stop and warnings for the device target)
- **Date:** 2026-11-16
- **Deciders:** Steph Xu
- **Related:** ADR-002, ADR-004, ADR-005, ADR-006, ADR-007; ARCHITECTURE.md §4

## Context

PruneAgent searches for a per-layer pruning plan that stays within Wanda-50% quality while saving the most measured memory (ADR-004, ADR-006). The agent chooses the next experiment (probe, apply, start from FLAP, roll back, stop) and, per layer, which strategy to use (ADR-007). Two kinds of work are independent and can run in parallel: layer-local sensitivity probes, and the evaluation of several candidate plans in one round (memory, estimated speed on the target device and perplexity for each). Code must still enforce the quality bar and budget, and stop when the user's device target is met. A run lasts 20–40 rounds and must survive a Colab disconnect. Course guidance: name who chooses the next step; code enforces gates.

## Options

| Option | Who chooses the next step | Fit | Cost |
|---|---|---|---|
| A. Fixed script (sweep, then choose) | Code | No adaptive experiment choice | ~1 day |
| **B. LangGraph state graph** | LLM agent chooses the experiment; code gates | `Send` fan-out + reducers for parallel probes and parallel candidate evaluation; explicit branches per agent action; invalid / rollback edges; SQLite checkpointer for resume | ~1.5 days incl. learning |
| C. ReAct agent with tools | LLM | The LLM could skip evaluation or rollback; limits become requests, not rules | ~1 day; hard to trust |
| D. Reinforcement learning (AMC, FastForward) | Learned policy | GPU-hours per model | Exceeds the compute budget |

## Decision

**B.** `ORCHESTRATION: hybrid. The LLM agent chooses the next experiment and, per layer, the strategy and its setting. Code owns probing, validation, pruning, parallel evaluation of up to three candidates, the quality-bar gate, accept / rollback, the device-target feasibility check and warnings, and stopping when the device target is met.` LLM calls go through the project's own client (cache, cost meter, rationale log).

## Consequences

- Positive: parallel probes and parallel candidate evaluation cut wall-clock time; memory and estimated speed are computed from the plan without the GPU, so only perplexity uses it; every agent choice is an explicit, logged branch; limits cannot be skipped; a disconnect loses at most one round.
- Negative: a dependency to pin and learn; three candidate models at once may not fit a T4 for Qwen3-1.7B (evaluate in sequence there; production run on A100).

## Evidence That Would Reverse This Decision

- Parallel probes and candidates show no speed-up and the agent uses only `apply` with one candidate → option A with a hand-built loop.
- LangGraph checkpoints prove unreliable on Google Drive → option A with plan JSON saved each round.
