# ADR-009: Search Strategies: How Much One Step May Change

- **Status:** Accepted (v0.10, 2026-11-20)
- **Date:** 2026-11-20
- **Deciders:** Steph Xu
- **Related:** ADR-001, ADR-002; ARCHITECTURE.md §4.2, §4.4, §4.5

## Context

The proposal describes an agent that prunes "layer by layer": each step, it decides which kind of cut suits a layer. Since the first committed architecture (v0.6), a candidate has been a set of (layer, method) choices applied together, with no limit on how many layers it touches. Every run so far used that. The 1.7B run changed 6 layers in every kept step; the 0.6B 20% run changed 1, 1, 3, 3, then mostly 6, and 16 in its last step.

Steph's intent, stated on 2026-11-17 ("we are iteratively doing the decision 'what kind of pruning method to use' and not deciding for all layers at once"), was never turned into a rule. The difference showed when the run replay began showing exactly which weights each step removed.

Both ways are worth measuring, on the same evidence, the same evaluation and the same gate, so the results compare.

## Options

| Option | What changes | Problem |
|---|---|---|
| A. Replace several-layers steps with one-layer steps | Rules and one instruction sentence | Throws away the comparison; earlier runs no longer match the system |
| B. A second copy of the search (graph, prompt, validator) | Everything | The copies drift; evaluation could differ and runs would no longer compare |
| **C. A search strategy setting that changes only what one candidate may contain and the layer order** | `search.py`; one rule; three placeholders in the instructions; the sweep's layer order and end | None found |

## Decision

**C.** `src/pruneagent/search.py` defines each search strategy as data:
- the `Rules` fields it sets (unless a run sets them itself), which the validator enforces, and
- the sentences of the agent's instructions that differ: what a candidate is (`{candidate_rule}`), whether earlier plans can be returned to (`{return_rule}`, `{rollback_line}`);
- the order of layers (`order`: free, or a sweep).

| Strategy | Rules | Instructions |
|---|---|---|
| `multi_layer` (default: every run so far) | none: any number of layers per candidate, ≤ 2 steps of a method on a layer | The v1 sentences, word for word: the prompt is byte-identical to before (tested) |
| `layer_by_layer` | ≤ 1 layer per candidate (`max_layers_per_candidate`); ≤ 7 steps of a method on it (the bank); sweep order | Goes through the layers in order; each candidate is a cut plan for this round's layer; no candidates = leave it; no going back |

`layer_by_layer` is a **sweep**: one pass through the model, first layer to last, so a run has exactly one round per layer (28 for Qwen3-0.6B and 1.7B). Round k decides layer k only. The prompt adds a "This round" section: the layer, the memory still to free and the layers left (an even share per layer, to pace the cuts), and the **bank** for that layer: slim 1 to 7 steps, remove 1 to 7 attention units, remove the whole layer, or no candidates to leave it as it is. Every reply gives the round's reason, also when the layer is left (required since 2026-11-20, after the first run left layer 0 with no reason in the step log; a reply without one is asked again). The step size stays fixed, so the agent picks and combines bank entries but never sets an amount. It proposes up to 3 cut plans for the layer; code measures each and the shared gate keeps the best within the per-step limit, or none (the layer stays). A reply that breaks a rule is refused and the same layer is asked again (up to 3 times in a row). The run ends when the memory limit is met or after the last layer; the rounds-without-a-kept-step and round limits don't apply, and rolling back is not offered.

Steph's design (2026-11-20): a proposal is a cut plan; three proposals test three cut plans; the cut plans come from a bank the AI picks from and combines. The first version of this strategy (any layer each round, revisits allowed) was replaced the same day before any run used it.

Shared by every strategy and not to be made strategy-specific: the scores and which neurons a step removes, step sizes, candidate evaluation, the gate and its per-step limit, the memory-limit stop, the checks after the run (unseen text, quiz questions), the saved model and the reports.

The run's `RunConfig.search` records the strategy. It is a `pruneagent run --search` option, a field of the app's start request and of the pruning and feasibility workflows, and the app shows it on every run (“Strategy”), in the start dialog (a choice, default several layers) and on Compare (one line per strategy).

## Consequences

- Positive: the two ways of pruning are measured with identical evaluation; earlier runs stay valid and reproducible (same prompt, same rules); a third strategy is one more entry.
- Negative: a sweep may end short of the memory limit if the agent under-cuts early layers; it then reports the gap.
- Result (2026-11-20, [docs/strategy/finding.md](../docs/strategy/finding.md)): on Qwen3-0.6B at 20% smaller, one layer per step got 40.2% of quiz questions right and several layers per step 40.6% (FLAP 34.6%). Both stay; several layers per step stays the default, and the start popup recommends the one with fewer AI calls (one per layer, fixed, vs the check's estimate).
- Neutral: the agent-answer cache never mixes strategies, since their prompts differ.
