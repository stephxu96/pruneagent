# ADR-002: State Is a Per-Layer Plan; Pruning Rebuilt from Dense Weights; Rollback = Earlier Accepted Plan

- **Status:** Proposed (v0.5; plan adds per-layer Wanda sparsity, ADR-007)
- **Date:** 2026-10-31
- **Deciders:** Steph Xu
- **Related:** ADR-001, ADR-003, ADR-007; ARCHITECTURE.md §4.3

## Context

Rollback must be exact and cheap, and the agent may roll back further than one step. Each round evaluates up to three candidate plans. Model weights (1–3 GB) cannot be copied into graph state every round. Per layer, the plan sets MLP neurons kept, attention groups kept, whether the layer is dropped, and unstructured Wanda sparsity on the remaining weights.

## Options

| Option | Rollback | Cost per round | Risk |
|---|---|---|---|
| A. Prune in place; save a weight copy each round | Restore copy | GBs written per round | Slow on Drive; memory |
| B. Keep neuron, group and mask index lists; restore on rollback | Restore indices | Small | Indices depend on scores; same as C in practice |
| **C. Keep only the per-layer plan; rebuild from a dense copy for each candidate** | Revert to any earlier accepted plan | Seconds: slice and mask weights by cached scores | Dense copy in CPU memory |

## Decision

**C.** Scores are computed once on the dense model and cached: FLAP fluctuation metric and bias compensation values for neurons and groups, Wanda weight scores for unstructured sparsity, Block Influence for layers. Given a plan, pruning keeps the top-scoring neurons and groups in each layer, adds FLAP's compensation bias, drops the planned layers, then zeroes the lowest-Wanda-score weights to the layer's sparsity. The plan is the only thing pruning depends on. State keeps the list of accepted plans, so `rollback` can return to any of them. Deterministic: the same plan always gives the same model and perplexity (how well the model predicts real text; lower is better).

## Consequences

- Positive: exact multi-step rollback; small checkpoints; any round reproducible from the log; a rebuild takes seconds, not a calibration pass (rerunning the example text through the model).
- Negative: dense copy in CPU RAM (~3.4 GB for 1.7B in fp16).
- Negative: scores come from the dense model, so they do not update after earlier layers are pruned or dropped. The full-model gate catches the effect; phase 1 compares against FLAP's and Wanda's own one-pass procedures.

## Evidence That Would Reverse This Decision

- CPU RAM cannot hold the dense copy → reload dense weights from local disk each round.
- Cached dense-model scores give clearly worse perplexity than recomputed scores at the same plan → recompute scores after each accepted step (one calibration pass per round).
