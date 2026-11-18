# Agent round 1, first attempt (task 1.11): step sizes rounded to 0.1 MB

Model Qwen/Qwen3-0.6B-Base; agent: Codex CLI (gpt-5.6-terra). Prompt: 4,070 characters, 626 words, about 1,960 tokens. Codex reported 17891 tokens in total for the call.

Memory limit: 1,072.9 MB of weights, 119.2 MB (10%) less than the unpruned 1,192.1 MB.

Perplexity on the same 32 validation windows (lower is better):
- unpruned model: 13.15
- Baseline: FLAP cut to the same limit (ratio 0.135437, 119.2 MB removed): 15.40
- Wanda-50% (secondary reference; frees no memory): 23.36

| # | Valid | Choices | Perplexity | vs Baseline: FLAP | vs Wanda-50% | Memory removed | Within limit |
|---|---|---|---|---|---|---|---|
| 1 | yes | 1 x drop_layer, 33 x narrow_mlp, 14 x remove_attention_group | 20.58 | +34% | -12% | 118.3 MB | no |
| 2 | yes | 1 x drop_layer, 33 x narrow_mlp, 14 x remove_attention_group | 20.45 | +33% | -12% | 118.3 MB | no |

**Candidate 1.** Frees 119.9 MB. Drops layer 3, the lowest-error removable block, then concentrates MLP and attention pruning in low-sensitivity early/middle layers.

Choices (layers; a layer listed twice takes two steps): drop_layer on [3]; narrow_mlp on [1, 1, 2, 2, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 20, 21, 22, 23, 27]; remove_attention_group on [2, 2, 4, 4, 5, 5, 6, 7, 8, 8, 9, 11, 13, 14]

**Candidate 2.** Frees 119.9 MB. A near-equivalent fallback that drops layer 4 instead, retaining layer 3 and exploiting its exceptionally low MLP and attention pruning errors.

Choices (layers; a layer listed twice takes two steps): drop_layer on [4]; narrow_mlp on [1, 1, 2, 2, 3, 3, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 20, 21, 22, 23, 27]; remove_attention_group on [2, 2, 3, 3, 5, 5, 6, 7, 8, 8, 9, 11, 13, 14]

