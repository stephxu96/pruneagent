# Midpoint run: the agent beats Baseline: FLAP at the same memory limit

## Headline

On the small model (Qwen3-0.6B-Base), cut to 10% less weight memory, the agent's model scores **15.55 perplexity on held-out test text, against 17.90 for Baseline: FLAP**: 13% lower (better) at the same memory limit. It got there in 13 rounds, with every round's step kept. This is one run on one model, and accuracy on the multiple-choice tests has not been measured yet.

## The question

Can the per-layer agent keep more quality than FLAP, the existing method, when both are cut to the same memory limit, with the same calibration data and the same test text?

## What we did

- **Model and limit:** Qwen3-0.6B-Base, weights limited to 1,072.9 MB (10% below the unpruned 1,192.1 MB).
- **Agent:** Codex (gpt-5.6-terra, high reasoning effort) with the v1 instructions. Each round it proposed up to 3 candidates; code applied them, measured perplexity on 32 validation passages, and kept the one with the smallest perplexity rise per MB freed (rejecting any that raised perplexity more than 15% in one round).
- **Evidence given to the agent:** the quick tests and the measured per-layer damage table, both made with the same C4 calibration as FLAP.
- **Baseline: FLAP:** FLAP cut to the same limit with the same C4 calibration.
- **Held-out check:** because the agent chose every step on those 32 passages, we also measured both models on text the search never saw: the 96 other validation passages and the full WikiText-2 test split.

## Results

Perplexity, lower is better.

| Model | Weight memory | 32 passages used by the search | Held-out validation (96 passages) | Test split |
|---|---|---|---|---|
| Unpruned | 1,192.1 MB | 13.15 | 13.30 | 12.66 |
| **Agent** | **1,071.4 MB** | **16.11** | **16.46** | **15.55** |
| Baseline: FLAP | 1,072.4 MB | 18.84 | 18.46 | 17.90 |

The agent's lead holds on held-out text (11% on unseen validation, 13% on test), so it is not an artefact of choosing steps on the scoring passages.

## How the agent got there

| Round | Kept step | Perplexity after (32 passages) |
|---|---|---|
| 1 | Drop layer 14 | 14.09 |
| 2 | Remove an attention group in layer 5 | 14.06 |
| 3 | Remove an attention group in layer 1 | 14.02 |
| 4–11 | Narrow the feed-forward block in layers 5–12 and 21, a few at a time | 14.71 |
| 12 | Drop layer 13 | 15.57 |
| 13 | Narrow 10 more feed-forward blocks and remove an attention group in layer 7, to land just under the limit | 16.11 |

It followed the measured damage table rather than the misleading quick tests: it dropped layers 14 and 13, which the table shows are the safest to drop, not layers 3–7, which the quick tests suggested. Its reasons cite the evidence it used, for example "Measured single-step perplexity is 14.09 (+7.1%) for a 31.5 MB reduction, the best measured whole-layer tradeoff."

## Cost

13 Codex calls (about 114,000 tokens; about 8 minutes of agent time in total) and about 24 minutes of Mac GPU time for the search, plus about 15 minutes for the held-out check. No Colab.

## Limits

- One run, one model, one memory limit. Codex's answers vary from run to run; repeats come in tuning (3.1).
- Multiple-choice accuracy is not measured yet; FLAP is the stronger test there (its accuracy was the binding constraint in phase 1).
- The small model only; the production comparison on 1.7B is phase 4.

Files: `decisions.jsonl` (every round, candidates and reasons), `plans/` (every kept plan), `heldout.json` (held-out numbers). Reproduce the held-out check with `scripts/heldout_check.py runs/<run-id>`.
