# ADR-002: State Is a Per-Layer Sparsity Plan; Pruning Rebuilt from Dense Weights; Rollback = Earlier Accepted Plan

- **Status:** Proposed (v0.3)
- **Date:** 2026-10-30
- **Deciders:** Steph Xu
- **Related:** ADR-001, ADR-003; ARCHITECTURE.md §4.3

## Context

Rollback must be exact and cheap, and the agent may roll back further than one step. Model weights (1–3 GB) cannot be copied into graph state every round. v0.3 is pruning only.

## Options

| Option | Rollback | Cost per round | Risk |
|---|---|---|---|
| A. Prune in place; save a weight copy each round | Restore copy | GBs written per round | Slow on Drive; memory |
| B. Keep masks; restore on rollback | Restore masks | Masks as large as the weights (bit-packed: 1/16) | More state |
| **C. Keep only the per-layer plan {layer: sparsity}; rebuild from a dense copy each round** | Revert to any earlier accepted plan | Wanda scores reuse cached calibration statistics | Dense copy in CPU memory |

## Decision

**C.** The plan is the only thing pruning depends on. State keeps the list of accepted plans, so the agent's `rollback` action can return to any of them. Deterministic: the same plan always gives the same model and perplexity.

## Consequences

- Positive: exact multi-step rollback; small checkpoints; any round reproducible from the log.
- Negative: dense copy in CPU RAM (~3.4 GB for 1.7B in fp16).

## Evidence That Would Reverse This Decision

- CPU RAM cannot hold the dense copy → option B with bit-packed masks on Drive or local drive.
