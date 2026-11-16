# ADR-004: Evaluation — Validation-Split Gate with Per-Step and Cumulative Limits; Test-Split Final Evaluation; Effort Measures

- **Status:** Proposed (v0.3)
- **Date:** 2026-10-30
- **Deciders:** Steph Xu
- **Related:** ADR-001, ADR-006; ARCHITECTURE.md §1.3, §6

## Context

Instructor feedback: (1) a 15% limit against only the previous step lets small acceptable steps add up to a large loss; (2) the measurements used to choose actions must be separate from the final quality check; (3) a well-supported negative result is acceptable. The value claim is workflow automation, so effort must be measured alongside quality.

## Decision

1. **Gate (every applied step):** WikiText-2 **validation** split, fixed windows. ROLLBACK if perplexity rises > 15% over the previous accepted step **or** exceeds the cumulative limit vs. the dense model (set per target in week 1 from uniform Wanda at that size).
2. **Final evaluation (end of run):** WikiText-2 **test** split; Wanda's seven zero-shot tasks (BoolQ, RTE, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA) via lm-evaluation-harness; MMLU.
3. **Baselines at equal measured memory (ADR-006):** uniform Wanda, OWL, manual sensitivity sweep. Same calibration data (C4, 128 × 2,048 tokens), same tests.
4. **Effort measures:** full-model evaluations, probe count, GPU time and manual decisions per method, from the decision log.
5. **Success criteria:** S1–S4 in ARCHITECTURE.md §1.3, fixed before the final run; a negative quality result with full logs is a valid outcome.
6. **Tuning discipline:** instructions tuned on Qwen3-0.6B only; frozen before Qwen3-1.7B.
7. **Reference checks first:** uniform Wanda through the system matches the reference code; community Qwen3-1.7B Wanda checkpoints as a sanity check only.

## Consequences

- Positive: no leakage between gate and reported results; cumulative drift bounded; quality and effort both reported.
- Negative: two perplexity splits to maintain; the manual sweep costs GPU time in week 1.

## Evidence That Would Reverse This Decision

- Validation and test perplexity disagree in ranking across methods → add a held-out C4 perplexity check.
