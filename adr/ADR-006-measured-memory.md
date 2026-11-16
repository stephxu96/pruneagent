# ADR-006: Optional Device Target with Early and Final Warnings; Compare at Equal Quality; Speed Estimated for the Target Device, Measured on M4 and T4

- **Status:** Proposed (v0.6; adds the device target and warnings; speed no longer tied to the T4)
- **Date:** 2026-11-16
- **Deciders:** Steph Xu
- **Related:** ADR-003, ADR-004, ADR-007; ARCHITECTURE.md §1.1, §4.4, §6

## Context

Small models like Qwen3-0.6B and 1.7B are deployed on laptops and phones, not on a T4. Users have different limits (memory available, how fast text must appear), so the pruning target should be theirs to set. Pruning alone may not reach a given target at Wanda-50% quality: at these model sizes, a laptop already fits the unpruned model, and a phone may need more than pruning can remove. The user needs to know that, early if possible.

Savings must be real: unstructured zeros stored densely save nothing. Every candidate needs memory and speed numbers each round, and measuring speed on a device for every candidate is slow; phones cannot be measured at all in this project (phone runtimes generally assume uniform layer widths).

## Options

| Option | Target | Problem |
|---|---|---|
| A. Fixed reference GPU (T4) as the objective (v0.5) | None | Not a device anyone deploys these models on |
| B. Named device presets only (laptop, phone) | Preset | Users' limits vary; presets hide the numbers |
| **C. Optional device target as explicit fields; speed estimated from the device's numbers; warnings when out of reach** | User-set | Estimates can be off; checked against two measured machines |

## Decision

**C.**

- **Device target (optional):** memory limit (GB), minimum decode speed (tokens/s, optional), memory bandwidth (GB/s), compute (TFLOPs), CPU or GPU. Example values for an M4 laptop, the T4 and a phone are documented, not built in as the only choices.
- **Search objective:** with a target, prune until the target is met within the quality bar, then stop: further pruning only costs quality. Without a target, save as much weight memory as the bar allows.
- **Estimated speed (search):** decode tokens/s ≈ memory bandwidth ÷ bytes of weights and key/value cache read per generated token at a 512-token context; prompt time ≈ prompt FLOPs ÷ compute. Computed from the plan, no hardware needed.
- **Early warning (before the search):** the target memory is below the smallest model the structure rules allow (every layer at its minimum, 6 layers dropped, embeddings unchanged), or the unpruned model already meets the target.
- **Final warning (end of search):** the quality bar ran out before the target was met; the report gives the closest plan and the gap (GB over the limit, tokens/s short).
- **Equal quality for baselines:** each baseline (FLAP, uniform Wanda-sp, ShortGPT) is pruned to the largest ratio whose validation perplexity stays within the bar; methods are compared on savings.
- **Weight memory:** bytes in dense fp16 storage, embeddings included, zeroed weights counted at full cost. Confirmed by file size at the end.
- **Measured (final, every compared method):** parameter count, file size, peak memory, prefill and decode tokens/s, on the M4 laptop (PyTorch on the Mac GPU) and the Colab T4, at a fixed prompt length, batch size and generation length.
- **Phase-1 check:** estimated vs. measured decode speed on ~10 plans, on both machines. Phone estimates are reported only if the estimate is within ~25% of measured speed on both.
- MLP widths in multiples of 64 neurons, so matrix shapes stay efficient on GPU kernels.

## Consequences

- Positive: the target is the user's; "what can pruning alone achieve" is answered by every run, including when the answer is "not enough"; the same search works for any device because bytes saved is device-independent.
- Positive: two measured machines check the estimate, so a phone estimate has stated evidence behind it.
- Negative: phone results are estimates only.
- Negative: embeddings (~18% of Qwen3-1.7B, ~25% of Qwen3-0.6B) are not pruned, so total memory falls less than the share of block weights removed; this also sets the floor for the early warning.
- Negative: two methods with equal memory can differ in speed (dropped layers speed up decoding more than narrower layers); reported, not equalized.

## Evidence That Would Reverse This Decision

- Estimated speed ranks plans differently from measured speed on either machine → measure tokens/s on the accepted plan every 5 rounds and use measured speed for targets on that machine; drop phone speed estimates.
- Pruned models show no tokens-per-second gain over the unpruned model on either machine → report memory only; speed targets produce an early warning.
