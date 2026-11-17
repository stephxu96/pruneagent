# ADR-007: Per-Layer Strategy Menu (MLP Neurons, Attention Groups, Whole Layers, Unstructured Wanda), Described Neutrally; Quantization Deferred

- **Status:** Proposed (v0.5; adds unstructured Wanda to the menu; quantization deferred)
- **Date:** 2026-10-31
- **Deciders:** Steph Xu
- **Related:** ADR-001, ADR-002, ADR-004, ADR-006; ARCHITECTURE.md §1.4, §4

## Context

The project promises a model at Wanda's quality that is actually smaller and faster on a single GPU. Unstructured pruning (Wanda) zeroes weights but keeps every matrix the same shape, so on a T4 or Mac the model uses the same memory and runs at the same speed. Structured pruning removes whole rows, columns and layers, so the matrices shrink, but it costs more quality per parameter removed, which makes per-layer choices matter.

Which strategies to offer is itself a choice. Leaving out a strategy because it is expected to be useless decides the question for the agent and removes the chance to see whether the agent works it out.

Qwen3 constraints: grouped-query attention, with 16 query heads sharing 8 key/value heads (2 query heads per group). Attention can only be pruned one key/value group at a time: 8 coarse steps per layer. MLP neurons are fine-grained (3,072 per layer in Qwen3-0.6B, 6,144 in Qwen3-1.7B). Reference code for FLAP, Wanda-sp and Wanda targets LLaMA (no grouped-query attention).

## Options

| Option | Saves memory / time on T4 and Mac | Build cost | Problem |
|---|---|---|---|
| A. Structured strategies only | Yes | Moderate | Decides for the agent that unstructured sparsity is useless |
| **B. Structured strategies plus unstructured Wanda per layer, described neutrally** | Structured: yes; Wanda: no, measured honestly | Moderate; Wanda already built for the bar | The agent can waste rounds on Wanda; the log shows it |
| C. B plus hidden-dimension pruning across all layers (SliceGPT-style) | Yes | High: changes every layer and the embeddings | Exceeds four weeks for one builder |
| D. B plus 2:4 sparsity | Only on A100 sparse kernels | Moderate | Speed is measured on the T4, which has no such kernels |

## Decision

**B.**

- **Per layer, the agent chooses a strategy; code does the rest.** Strategies: narrow the MLP, remove an attention group (each takes 2 query heads with it), drop the layer, or add unstructured Wanda sparsity. Code applies one fixed step per choice (10% of the original MLP width rounded to 64 neurons; one group; the whole layer; +10 points of sparsity up to 70%), picks what is removed by score, and performs the pruning. The agent never sets a number.
- **Neutral instructions:** each strategy is described by what it does to the weights, at the same level of detail, with the objective stated (stay within the quality bar while reducing measured memory and latency). No ranking or hints.
- **Scoring:** FLAP's fluctuation metric and bias compensation for neurons and groups; Wanda's weight score for unstructured sparsity; ShortGPT's Block Influence for layers. All computed once on the dense model and cached (ADR-002).
- **Prediction P1:** the agent will rarely choose Wanda, since it saves no memory. Measured by strategy counts in the log, not enforced.
- **Quantization:** deferred. If added later, it is a one-time weight-only step (8-bit or 4-bit) after the search, followed by the same memory, speed and perplexity checks. Wanda is not quantized.

## Consequences

- Positive: memory and tokens per second are measured directly on dense, smaller matrices (ADR-006); four strategies per layer give the agent a real choice; the strategy counts show whether the agent understood the objective.
- Negative: non-uniform widths do not load in standard Hugging Face model classes; a loader rebuilds the model from the plan JSON (~100 lines).
- Negative: a few hours in phase 1 to adapt FLAP and Wanda scoring to grouped-query attention and Qwen3 layer names.

## Evidence That Would Reverse This Decision

- At the Wanda-50% bar, FLAP saves < 5% of weight memory on Qwen3-0.6B → report the limit; the result then rests on how the agent mixes strategies.
- Non-uniform widths give no tokens-per-second gain over the dense model on the T4 → round widths to multiples of 128, or restrict to uniform widths within layer groups.
