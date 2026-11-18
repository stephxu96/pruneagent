# Agent round 1 (task 1.11)

Model Qwen/Qwen3-0.6B-Base; agent: Codex CLI (gpt-5.6-terra). Prompt: 4,084 characters, 626 words, about 1,960 tokens. Codex reported 24646 tokens in total for the call.

Memory limit: 1,072.9 MB of weights, 119.2 MB (10%) less than the unpruned 1,192.1 MB.

Perplexity on the same 32 validation windows (lower is better):
- unpruned model: 13.15
- Baseline: FLAP cut to the same limit (ratio 0.135437, 119.2 MB removed): 15.40
- Wanda-50% (secondary reference; frees no memory): 23.36

| # | Valid | Choices | Perplexity | vs Baseline: FLAP | vs Wanda-50% | Memory removed | Within limit |
|---|---|---|---|---|---|---|---|
| 1 | yes | 1 x drop_layer, 45 x narrow_mlp | 18.61 | +21% | -20% | 119.9 MB | yes |
| 2 | yes | 54 x narrow_mlp, 9 x remove_attention_group | 15.72 | +2% | -33% | 120.3 MB | yes |
| 3 | yes | 1 x drop_layer, 45 x narrow_mlp | 19.22 | +25% | -18% | 119.9 MB | yes |

**Candidate 1.** Frees 119.842 MB. Drops layer 3, the uniquely safest full-layer removal, then distributes MLP narrowing broadly while avoiding layer 0.

Choices (layers; a layer listed twice takes two steps): drop_layer on [3]; narrow_mlp on [1, 1, 2, 2, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 22, 23, 24, 25, 26, 27]

**Candidate 2.** Frees 120.195 MB without dropping a layer. It avoids the highly sensitive first layer and uses attention-group removal only where the one-step quick test is especially low.

Choices (layers; a layer listed twice takes two steps): narrow_mlp on [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27]; remove_attention_group on [2, 3, 4, 5, 6, 7, 8, 9, 14]

**Candidate 3.** Frees 119.842 MB. A close fallback to dropping layer 3, useful if held-out perplexity prefers preserving layer 3 despite its slightly lower estimated drop error.

Choices (layers; a layer listed twice takes two steps): drop_layer on [4]; narrow_mlp on [1, 1, 2, 2, 3, 3, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 22, 23, 24, 25, 26, 27]

