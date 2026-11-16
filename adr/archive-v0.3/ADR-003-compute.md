# ADR-003: Compute — Mac and Colab Pro T4 for Development; A100 for Final Runs

- **Status:** Proposed (v0.3)
- **Date:** 2026-10-30
- **Deciders:** Steph Xu
- **Related:** ADR-002, ADR-006; ARCHITECTURE.md §1.6, §10

## Context

Available hardware: a Mac with Apple silicon, and Colab Pro with 270.34 compute units (units expire after 90 days). Available accelerators: T4, L4, A100, G4 GPUs and v5e / v6e TPUs; H100 not available on this plan. The A100 costs more compute units per hour than the T4. The T4 has no bf16 support; the A100 does. Sessions can end, so state must be checkpointed. v0.3 uses Qwen3-0.6B (development) and Qwen3-1.7B (final) only.

## Options

| Option | Fits | Compute units | Risk |
|---|---|---|---|
| A. T4 for everything | Qwen3 ≤ 1.7B in fp16 | Lowest | No bf16; slower final evaluation |
| **B. Mac + T4 for development; A100 for final runs** | bf16 for final runs | Moderate; capped at 200 | Units run out if development drifts to A100 |
| C. A100 for everything | Larger models | Highest | Exhausts units before final runs |

## Decision

**B.**

- Development on the Mac (Qwen3-0.6B) and Colab T4; fp16 with fp32 fallback on T4.
- Final run (Qwen3-1.7B) on A100 in bf16.
- Hard cap of 200 compute units; 70 in reserve.
- Checkpoint every step to Google Drive or local drive; `resume` restarts from the last checkpoint.
- Success criterion S3 stays on the T4 (≤ 2 h per run): it matches the single-GPU target user.
- Memory is measured, not assumed (ADR-006). Speed-up out of scope.

## Consequences

- Positive: bf16 removes the T4 numerical risk for final results; budget is explicit.
- Negative: compute units are a finite resource to track.

## Evidence That Would Reverse This Decision

- More than 50 units used by the end of week 2 → T4 only, including the final run.
- A run exceeds 2 h or $5 → Qwen3-0.6B only.
