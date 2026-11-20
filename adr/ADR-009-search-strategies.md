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
| **C. A search strategy setting that changes only what one candidate may contain** | `search.py`; one rule; one placeholder in the instructions | None found |

## Decision

**C.** `src/pruneagent/search.py` defines each search strategy as data:
- the `Rules` fields it sets (unless a run sets them itself), which the validator enforces, and
- the one sentence of the agent's instructions that says what a candidate is (`{candidate_rule}`).

| Strategy | Rules | Instructions |
|---|---|---|
| `multi_layer` (default: every run so far) | none: any number of layers per candidate, ≤ 2 steps of a method on a layer | The v1 sentence, word for word: the prompt is byte-identical to before (tested) |
| `layer_by_layer` | ≤ 1 layer per candidate (`max_layers_per_candidate`); ≤ 4 steps of a method on that layer | "Each candidate changes one layer only…" |

Up to 4 steps on the one layer, instead of 2: one step on one layer frees little (Qwen3-0.6B: about 2 MB slimming, 1.6 MB per attention unit, about 30 MB removing the layer), and the step size stays fixed so the agent still never sets amounts. Each round still offers up to 3 candidates, each on its own layer, and the gate keeps one, so every kept step changes exactly one layer.

Shared by every strategy and not to be made strategy-specific: the scores and which neurons a step removes, step sizes, candidate evaluation, the gate, stop rules, the checks after the run (unseen text, quiz questions), the saved model and the reports.

The run's `RunConfig.search` records the strategy. It is a `pruneagent run --search` option, a field of the app's start request and of the pruning and feasibility workflows, and the app shows it on every run (“How it cuts”), in the start dialog (a choice, default several layers) and on Compare (one line per strategy).

## Consequences

- Positive: the two ways of pruning are measured with identical evaluation; earlier runs stay valid and reproducible (same prompt, same rules); a third strategy is one more entry.
- Negative: a one-layer run needs more rounds for the same memory cut; at 1.7B, 20% smaller is estimated at about 23 rounds of the 40 allowed.
- Neutral: the agent-answer cache never mixes strategies, since their prompts differ.
