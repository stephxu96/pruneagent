# Midpoint run: the agent beats Baseline: FLAP at the same memory limit

## Headline

On the small model (Qwen3-0.6B-Base), cut to 10% less weight memory, the agent's model scores **15.55 perplexity on held-out test text, against 17.90 for Baseline: FLAP**: 13% lower (better) at the same memory limit. It got there in 13 rounds, with every round's step kept. On seven multiple-choice question sets it is clearly more accurate on 2 (by 6.7 and 2.4 points), and FLAP is clearly ahead on none; the other 5 are too close to call. This is one run on one model.

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

## Accuracy on multiple-choice questions

Perplexity measures how well a model predicts text; accuracy measures whether it picks the right answer to a question. We scored both pruned models on seven standard question sets used in earlier pruning research (percent correct, higher is better), on the Mac GPU.

The sets differ a lot in size, from 277 to 10,042 questions, so we compare them one at a time rather than averaging them. A gap counts as clear only when it is about twice the random variation expected for a set of that size.

| Question set | What it tests | Questions | Unpruned (Colab GPU) | **Agent** | Baseline: FLAP | Agent minus FLAP | Clear difference? |
|---|---|---:|---:|---:|---:|---:|---|
| arc_easy | school science questions, easier | 2,376 | 65.2 | **61.7** | 55.0 | +6.7 | **yes** |
| boolq | yes/no questions about a passage | 3,270 | 69.9 | **64.7** | 62.3 | +2.4 | **yes** |
| arc_challenge | school science questions, harder | 1,172 | 33.2 | 27.6 | 25.4 | +2.2 | no |
| rte | does one sentence follow from another | 277 | 69.7 | 56.3 | 52.7 | +3.6 | no |
| hellaswag | the most sensible ending for a sentence | 10,042 | 41.0 | 36.5 | 36.3 | +0.2 | no |
| winogrande | who or what a pronoun refers to | 1,267 | 58.5 | 56.0 | 57.1 | −1.2 | no |
| openbookqa | science questions needing a known fact | 500 | 24.6 | 21.2 | 23.4 | −2.2 | no |

The agent is clearly ahead on 2 sets and FLAP is clearly ahead on none; the other 5 are too close to call. The unpruned scores were measured earlier on a Colab GPU, so compare them loosely; the two pruned models were measured the same way on the same machine.

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

13 Codex calls (about 114,000 tokens; about 8 minutes of agent time in total) and about 24 minutes of Mac GPU time for the search, plus about 15 minutes for the held-out check and 74 minutes for the accuracy check. No Colab.

## Limits

- One run, one model, one memory limit. Codex's answers vary from run to run; repeats come in tuning (3.1).
- The accuracy lead is smaller than the perplexity lead: clear on 2 of 7 question sets, too close to call on the rest.
- The small model only; the production comparison on 1.7B is phase 4.

Files: `decisions.jsonl` (every round, candidates and reasons), `plans/` (every kept plan), `heldout.json` (held-out numbers), `accuracy.json` (accuracy per question set). Reproduce with `scripts/heldout_check.py runs/<run-id>` and `scripts/accuracy_check.py runs/<run-id>`.
