# ADR-003: Compute — M4 Mac and Colab Pro T4 for Development and Speed Measurement; A100 for the Production Run

> **Revised 2026-11-17 (v0.8):** Baseline: FLAP, cut to each memory limit used, replaces Baseline: Wanda-50% as the comparison; it runs on the same machines as the agent for the same model. The Wanda-50% numbers already recorded are kept as an optional backup. The compute plan is otherwise unchanged.

> **Revised 2026-11-17 (v0.7):** the Wanda-50% "quality bar" is now Baseline: Wanda-50%, the quality reference with no allowance for now. The compute plan is unchanged.

- **Status:** Proposed (v0.8; Baseline: FLAP on the same machines as the agent)
- **Date:** 2026-11-16
- **Deciders:** Steph Xu
- **Related:** ADR-002, ADR-006, ADR-007; ARCHITECTURE.md §1.6, §10

## Context

Available hardware: an M4 Mac, and Colab Pro with 270.34 compute units (units expire after 90 days). Available accelerators: T4, L4, A100, G4 GPUs and v5e / v6e TPUs; H100 not available on this plan. The A100 costs more compute units per hour than the T4. The T4 has no bf16 support; the A100 does. Sessions can end, so state must be checkpointed. Qwen3-0.6B is always the development model and Qwen3-1.7B the production-run model. The device target is user-set (ADR-006), so speed is estimated for any device and checked against machines that can be measured: a laptop (the M4) and a GPU (the T4).

## Options

| Option | Fits | Compute units | Risk |
|---|---|---|---|
| A. T4 for everything | Qwen3 ≤ 1.7B in fp16 | Lowest | No bf16; no laptop measurement |
| **B. M4 + T4 for development and speed measurement; A100 for the production run** | bf16 and room for parallel candidates in the production run; speed measured on a laptop and a GPU | Moderate; capped at 200 | Units run out if development drifts to A100 |
| C. A100 for everything | Larger models | Highest | Exhausts units early; not representative of a laptop or a small GPU |

## Decision

**B.**

- Development on the M4 (Qwen3-0.6B, PyTorch fp16 on the Mac GPU) and Colab T4; fp16 with fp32 fallback on T4.
- Production run (Qwen3-1.7B) on A100 in bf16.
- Baseline: FLAP is cut to each memory limit once per model, on the machine that runs the agent for that model (0.6B on M4 or T4, 1.7B on A100). The Wanda-50% reference (optional backup) is already recorded for both models (tasks 1.1 and 1.10).
- Memory and tokens per second for every compared method measured on the M4 and the T4 with a fixed prompt length, batch size and generation length (ADR-006). M4 measurements cost no compute units.
- Hard cap of 200 compute units; 70 in reserve.
- Checkpoint every step to Google Drive or local drive; `resume` restarts from the last checkpoint.

## Consequences

- Positive: bf16 removes the T4 numerical risk for production results; speed is measured on a laptop and a GPU, which checks the device estimate twice; budget is explicit.
- Negative: compute units are a finite resource to track; T4 speed runs add a short session per method; Mac GPU speed for non-uniform widths is untested until phase 1.

## Evidence That Would Reverse This Decision

- More than 50 units used by the end of phase 2 → T4 only, including the production run, with candidates evaluated in sequence.
- A run exceeds 2 h → Qwen3-0.6B only.
- Mac GPU speed is unstable across repeat runs (> 10% variation) → report M4 as memory only; T4 stays the measured speed reference.
