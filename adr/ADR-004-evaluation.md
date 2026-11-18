# ADR-004: Evaluation — Baseline: FLAP at the Same Memory Limit (No Allowance); Validation-Split Gate; Test-Split Final Evaluation; Wanda-50% and Uniform Wanda-sp as Optional Backups

> **Revised 2026-11-17 (v0.8):** the primary comparison is now FLAP cut to the same memory limit (Baseline: FLAP). Wanda-50% and uniform Wanda-sp become optional backups, reported only if useful; the ShortGPT comparison is dropped. No allowance or quality threshold. Reason: FLAP actually shrinks the model and, at moderate cuts, matches Wanda-50% quality (Qwen3-0.6B with 11.1% of weight memory removed: perplexity 16.63 vs. 23.68, mean accuracy 43.95% vs. 43.80%), so one comparison answers the question.

> **Revised 2026-11-17 (v0.7):** the fixed margin (perplexity ≤ 10% above Wanda-50%, accuracy ≤ 2 points below) is removed. Baseline: Wanda-50% is the quality reference with no allowance for now; any allowance is decided after seeing the results. Compared methods are cut to the same memory limit and compared on quality, instead of pruned to the largest ratio within the margin.

- **Status:** Proposed (v0.8; Baseline: FLAP at the same memory limit, no allowance)
- **Date:** 2026-11-16
- **Deciders:** Steph Xu
- **Related:** ADR-001, ADR-006, ADR-007; ARCHITECTURE.md §1.3, §6

## Context

The goal is a model that fits a memory limit the user sets and keeps as much quality as possible, judged against an existing method at the same memory limit. Wanda's unstructured sparsity keeps quality well but saves no memory or time on standard hardware, so it cannot be compared at the same memory limit. FLAP is structured: it actually shrinks the model, and at moderate cuts it keeps Wanda-50%'s quality (Qwen3-0.6B, 11.1% of weight memory removed: perplexity 16.63 vs. 23.68, mean accuracy 43.95% vs. 43.80%). Instructor feedback still applies: (1) per-step limits alone let small steps add up to a large loss; (2) the measurements used to choose actions must be separate from the final quality check; (3) a well-supported negative result is acceptable.

## Decision

1. **Baseline: FLAP (primary comparison):** FLAP cut to the same memory limit as the agent (the smallest FLAP pruning ratio whose weights fit), on the same model, with FLAP's published settings, including its calibration data (example text run through the model to measure which inputs are large; it decides which weights are removed, and is kept separate from the test text). The project question: does the agent keep more quality, in perplexity (how well the model predicts real text; lower is better) and average multiple-choice accuracy (share of questions answered correctly), than Baseline: FLAP at the same memory limit? No allowance or quality threshold: results are reported against Baseline: FLAP as measured. The unpruned model is the no-pruning reference point, not a baseline.
2. **Gate (every candidate):** a fixed subset of 32 of the 128 WikiText-2 **validation** windows (checked in phase 1 to rank plans like the full split), skipping the unchanged layers before the candidate's first change (their output is saved from the accepted model; every later layer is recomputed, so the result is exact). Baseline: FLAP's perplexity on the same subset is shown to the agent; no candidate is rejected on it. Reject if perplexity rises > 15% over the last accepted plan. Accept the passing candidate with the lowest perplexity, ties broken by estimated speed.
3. **Final evaluation (end of run):** WikiText-2 **test** split; Wanda's 7 multiple-choice question sets (BoolQ, RTE, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA) via lm-evaluation-harness; MMLU. Memory and tokens per second on the M4 laptop and the T4 (ADR-006).
4. **Runs:** the production run (Qwen3-1.7B) is given a memory limit; S1 and S2 measure the quality kept at that limit. A second run at a limit pruning cannot reach shows the final warning (FR13, FR14).
5. **Compared methods:** Baseline: FLAP cut to the same memory limit; the unpruned model as the no-pruning reference point. Optional backups, reported only if useful: uniform Wanda-sp cut to the same memory limit (a weaker structured method) and Wanda-50% (saves no memory, so at most a quality reference). Same tests for all. ShortGPT and LLM-Pruner are cited, not run.
6. **Success criteria:** S1–S4 in ARCHITECTURE.md §1.3, fixed before the production run; a negative result with full logs is a valid outcome.
7. **Search measures:** full-model evaluations, probe count, GPU time, agent calls and tokens, and how often each strategy is proposed and accepted (prediction P1), from the decision log.
8. **Tuning discipline:** instructions tuned on Qwen3-0.6B only; frozen before the Qwen3-1.7B production run.
9. **Reference checks first:** FLAP, Wanda-50% and uniform Wanda-sp through the system match the adapted reference code.

## Consequences

- Positive: one comparison answers the question: a published method that really shrinks the model, at the same memory limit; no leakage between gate and reported results.
- Negative: two perplexity splits to maintain; nothing rejects a candidate for falling below Baseline: FLAP, so loss across many steps is held back only by the per-step limit and by stopping at the memory limit, and the gap is reported.
- Negative: at large cuts FLAP itself loses much quality, so beating it says less; Wanda-50% can then be reported alongside as a backup quality reference.

## Evidence That Would Reverse This Decision

- Validation and test perplexity disagree in ranking across methods → add a held-out C4 perplexity check.
- At the memory limits used, FLAP falls far below Wanda-50% quality → report Wanda-50% alongside as a backup quality reference.
