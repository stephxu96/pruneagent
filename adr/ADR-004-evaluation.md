# ADR-004: Evaluation — Wanda-50% Quality Bar with Fixed Margin; Validation-Split Gate; Test-Split Final Evaluation; Baselines Pruned to the Same Bar

- **Status:** Proposed (v0.6; adds device-target runs and M4 measurement)
- **Date:** 2026-11-16
- **Deciders:** Steph Xu
- **Related:** ADR-001, ADR-006, ADR-007; ARCHITECTURE.md §1.3, §6

## Context

The goal is a model within a fixed margin of Wanda's quality that, unlike Wanda, is actually smaller and faster. Wanda's unstructured sparsity keeps quality well but saves no memory or time on standard hardware, so it sets the quality bar, not the size target. Instructor feedback still applies: (1) per-step limits alone let small steps add up to a large loss; (2) the measurements used to choose actions must be separate from the final quality check; (3) a well-supported negative result is acceptable.

## Decision

1. **Quality bar:** Wanda at 50% unstructured sparsity on the same model, same calibration data (example text run through the model to measure which inputs are large; it decides which weights are removed, and is kept separate from the test text). Margin fixed before the production run: perplexity (how well the model predicts real text; lower is better) ≤ 10% above Wanda-50%; average multiple-choice accuracy (share of questions answered correctly) ≤ 2 points below.
2. **Gate (every candidate):** a fixed subset of 32 of the 128 WikiText-2 **validation** windows (checked in phase 1 to rank plans like the full split), skipping the unchanged layers before the candidate's first change (their output is saved from the accepted model; every later layer is recomputed, so the result is exact). Reject if perplexity > 1.10 × Wanda-50% perplexity on the same subset, or rises > 15% over the last accepted plan. Accept the passing candidate that meets the device target with the lowest perplexity, or, if none meets it (or no target is set), the one with the largest weight-memory saving.
3. **Final evaluation (end of run):** WikiText-2 **test** split; Wanda's 7 multiple-choice question sets (BoolQ, RTE, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA) via lm-evaluation-harness; MMLU. Memory and tokens per second on the M4 laptop and the T4 (ADR-006).
4. **Runs:** the production run (Qwen3-1.7B) has no device target, so S1 and S2 measure the most pruning alone can save at the bar. Two device-target runs show the warnings: one target reachable within the bar, one not (FR13, FR14).
5. **Compared methods:** dense model; Wanda-50% (the bar); FLAP and uniform Wanda-sp, each pruned to the largest ratio within the bar (found by bisection on validation perplexity); ShortGPT pruned the same way, reported alongside. Same calibration data (C4, 128 × 2,048 tokens), same tests. LLM-Pruner is cited, not run: it needs a recovery fine-tune.
6. **Success criteria:** S1–S4 in ARCHITECTURE.md §1.3, fixed before the production run; a negative result with full logs is a valid outcome.
7. **Search measures:** full-model evaluations, probe count, GPU time, agent calls and tokens, and how often each strategy is proposed and accepted (prediction P1), from the decision log.
8. **Tuning discipline:** instructions tuned on Qwen3-0.6B only; frozen before the Qwen3-1.7B production run.
9. **Reference checks first:** Wanda-50%, FLAP and uniform Wanda-sp through the system match the adapted reference code.

## Consequences

- Positive: success is defined by a published method's quality; no leakage between gate and reported results; cumulative drift bounded by the bar; every method is compared on savings at the same quality.
- Negative: two perplexity splits to maintain; pruning each baseline to the bar costs ~6–8 extra evaluations per baseline.

## Evidence That Would Reverse This Decision

- Validation and test perplexity disagree in ranking across methods → add a held-out C4 perplexity check.
- Wanda-50% on Qwen3-0.6B is so degraded that the bar is trivially met → use Wanda at 40% as the bar, fixed before the production run.
