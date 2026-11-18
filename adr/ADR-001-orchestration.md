# ADR-001: Orchestration — LangGraph Agent Graph with Parallel Probes, Parallel Candidate Evaluation and Experiment-Choice Branches

> **Revised 2026-11-17 (v0.8):** quality is compared with Baseline: FLAP (FLAP cut to the same memory limit) instead of Baseline: Wanda-50%. Code cuts FLAP to the memory limit once before the search and shows its perplexity to the agent. The gate is unchanged: among candidates within the per-step limit, keep the one with the lowest perplexity.

> **Revised 2026-11-17 (v0.7):** the search fits a required memory limit and keeps the most quality. Code records each candidate against Baseline: Wanda-50% (no allowance for now) instead of gating on a quality bar with a margin.

- **Status:** Proposed (v0.8; memory limit required; compared with Baseline: FLAP)
- **Date:** 2026-11-16
- **Deciders:** Steph Xu
- **Related:** ADR-002, ADR-004, ADR-005, ADR-006, ADR-007; ARCHITECTURE.md §4

## Context

PruneAgent searches for a per-layer pruning plan that fits a memory limit the user sets and keeps the most quality, compared with Baseline: FLAP cut to the same memory limit (ADR-004, ADR-006). The agent chooses the next experiment (probe, apply, start from FLAP, roll back, stop) and, per layer, which strategy to use (ADR-007). Two kinds of work are independent and can run in parallel: layer-local sensitivity probes, and the evaluation of several candidate plans in one round (memory, estimated speed on the target device and perplexity (how well the model predicts real text; lower is better) for each). Code must still enforce the per-step quality limit and budget, and stop when the memory limit is met. A run lasts 20–40 rounds and must survive a Colab disconnect. Course guidance: name who chooses the next step; code enforces gates.

## Options

| Option | Who chooses the next step | Fit | Cost |
|---|---|---|---|
| A. Fixed script (sweep, then choose) | Code | No adaptive experiment choice | ~1 day |
| **B. LangGraph state graph** | LLM agent chooses the experiment; code gates | `Send` fan-out + reducers for parallel probes and parallel candidate evaluation; explicit branches per agent action; invalid / rollback edges; SQLite checkpointer for resume | ~1.5 days incl. learning |
| C. ReAct agent with tools | LLM | The LLM could skip evaluation or rollback; limits become requests, not rules | ~1 day; hard to trust |
| D. Reinforcement learning (AMC, FastForward) | Learned policy | GPU-hours per model | Exceeds the compute budget |

## Decision

**B.** `ORCHESTRATION: hybrid. The LLM agent chooses the next experiment and which strategy to apply to which layers; it never sets amounts. Code owns step sizes, the choice of weights to remove, probing, validation, pruning, parallel evaluation of up to three candidates, the gate (per-step quality limit; lowest perplexity wins), accept / rollback, the device-target feasibility check and warnings, cutting Baseline: FLAP to the memory limit once before the search, and stopping when the memory limit is met.` Agent calls go through the project's own client, which sends the same prompt and JSON schema to either Codex CLI (read-only, no tools) or an LLM API (cache, usage meter, rationale log; ADR-005).

## Consequences

- Positive: parallel probes and parallel candidate evaluation cut wall-clock time; memory and estimated speed are computed from the plan without the GPU, so only perplexity uses it; every agent choice is an explicit, logged branch; limits cannot be skipped; a disconnect loses at most one round.
- Negative: a dependency to pin and learn; three candidate models at once may not fit a T4 for Qwen3-1.7B (evaluate in sequence there; production run on A100).

## Evidence That Would Reverse This Decision

- Parallel probes and candidates show no speed-up and the agent uses only `apply` with one candidate → option A with a hand-built loop.
- LangGraph checkpoints prove unreliable on Google Drive → option A with plan JSON saved each round.
