# ADR-006: Equal Size Means Measured Memory — Sparse-Format File Size and Peak GPU Memory

- **Status:** Proposed (v0.3)
- **Date:** 2026-10-30
- **Deciders:** Steph Xu
- **Related:** ADR-003, ADR-004; ARCHITECTURE.md §1.1, §6

## Context

Instructor feedback: define equal size by the artifact and memory measurement actually reported, including how pruned weights are stored and loaded, to connect evaluation to the memory problem motivating the project. Unstructured sparsity stored as dense tensors with zeros saves no memory.

## Options

| Option | What is compared | Problem |
|---|---|---|
| A. Nominal sparsity (% weights zero) | A ratio | Says nothing about memory actually used |
| B. Dense file with zeros | File size | Same size as the original model |
| **C. Sparse-format file size + peak GPU memory** | Bytes on disk and on GPU | Depends on the format; must be stated |

## Decision

**C.**

- Store pruned linear layers in a stated sparse format (compressed sparse row or bitmask + values; chosen in week 1 by measured size on Qwen3-0.6B).
- Report for every method: file size on disk (MB) and peak GPU memory when loading and running a fixed evaluation batch (MB).
- "Equal size" = equal measured file size within 1%; baselines are pruned until they match the agent's measured size, or vice versa.
- Nominal sparsity is reported alongside, never as the size criterion.

## Consequences

- Positive: the size claim is measured and reproducible; directly tied to the single-GPU memory problem.
- Negative: sparse formats only save memory above a sparsity threshold; loading for evaluation may require densifying, which is reported as peak GPU memory.

## Evidence That Would Reverse This Decision

- No sparse format reduces file size at the target sparsity → report nominal sparsity with an explicit limitation, and state the memory claim as out of reach for unstructured pruning.
