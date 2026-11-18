# ADR-004: Evaluation — Baseline: Wanda-50% as Quality Reference (No Allowance for Now); Validation-Split Gate; Test-Split Final Evaluation; Compared Methods Cut to the Same Memory Limit

> **Revised 2026-11-17 (v0.7):** the fixed margin (perplexity ≤ 10% above Wanda-50%, accuracy ≤ 2 points below) is removed. Baseline: Wanda-50% is the quality reference with no allowance for now; any allowance is decided after seeing the results. Compared methods are cut to the same memory limit and compared on quality, instead of pruned to the largest ratio within the margin.

- **Status:** Proposed (v0.7; Baseline: Wanda-50% with no allowance; comparison at the same memory limit)
- **Date:** 2026-11-16
- **Deciders:** Steph Xu
- **Related:** ADR-001, ADR-006, ADR-007; ARCHITECTURE.md §1.3, §6

## Context

The goal is a model that fits a memory limit the user sets and keeps as much quality as possible, judged against Wanda's; unlike Wanda, it is actually smaller and faster. Wanda's unstructured sparsity keeps quality well but saves no memory or time on standard hardware, so it sets the quality reference, not the size target. Instructor feedback still applies: (1) per-step limits alone let small steps add up to a large loss; (2) the measurements used to choose actions must be separate from the final quality check; (3) a well-supported negative result is acceptable.

## Decision

1. **Baseline: Wanda-50% (quality reference):** Wanda at 50% unstructured sparsity on the same model, same calibration data (example text run through the model to measure which inputs are large; it decides which weights are removed, and is kept separate from the test text). Every result is reported against its perplexity (how well the model predicts real text; lower is better) and average multiple-choice accuracy (share of questions answered correctly). No allowance for now: whether any shortfall from Wanda-50% is acceptable is decided after seeing the results, and fixed before the production run.
2. **Gate (every candidate):** a fixed subset of 32 of the 128 WikiText-2 **validation** windows (checked in phase 1 to rank plans like the full split), skipping the unchanged layers before the candidate's first change (their output is saved from the accepted model; every later layer is recomputed, so the result is exact). Each candidate's perplexity is recorded against Wanda-50% on the same subset; no candidate is rejected on it for now. Reject if perplexity rises > 15% over the last accepted plan. Accept the passing candidate with the lowest perplexity, ties broken by estimated speed.
3. **Final evaluation (end of run):** WikiText-2 **test** split; Wanda's 7 multiple-choice question sets (BoolQ, RTE, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA) via lm-evaluation-harness; MMLU. Memory and tokens per second on the M4 laptop and the T4 (ADR-006).
4. **Runs:** the production run (Qwen3-1.7B) is given a memory limit; S1 and S2 measure the quality kept at that limit. A second run at a limit pruning cannot reach shows the final warning (FR13, FR14).
5. **Compared methods:** dense model; Baseline: Wanda-50% (the quality reference); FLAP and uniform Wanda-sp, each cut to the same memory limit as the agent and compared on quality (perplexity and mean accuracy); ShortGPT cut the same way, reported alongside. Same calibration data (C4, 128 × 2,048 tokens), same tests. LLM-Pruner is cited, not run: it needs a recovery fine-tune.
6. **Success criteria:** S1–S4 in ARCHITECTURE.md §1.3, fixed before the production run; a negative result with full logs is a valid outcome.
7. **Search measures:** full-model evaluations, probe count, GPU time, agent calls and tokens, and how often each strategy is proposed and accepted (prediction P1), from the decision log.
8. **Tuning discipline:** instructions tuned on Qwen3-0.6B only; frozen before the Qwen3-1.7B production run.
9. **Reference checks first:** Wanda-50%, FLAP and uniform Wanda-sp through the system match the adapted reference code.

## Consequences

- Positive: quality is judged against a published method; no leakage between gate and reported results; every method is compared on quality at the same memory limit.
- Negative: two perplexity splits to maintain; with no allowance, nothing rejects a candidate for falling below Wanda-50%, so loss across many steps is held back only by the per-step limit and by stopping at the memory limit, and the gap is reported.

## Evidence That Would Reverse This Decision

- Validation and test perplexity disagree in ranking across methods → add a held-out C4 perplexity check.
- Wanda-50% on Qwen3-0.6B is so degraded that matching it says little about quality → use Wanda at 40% as the reference, fixed before the production run.
