"""Search strategies: how much one candidate may change (ADR-009).

A search strategy is the one part of a run that decides the shape of the agent's steps. It sets two things:

- rule changes (`Rules` fields), which the validator enforces, and
- one sentence of the agent's instructions saying what a candidate is.

Everything else is shared by every strategy and must stay so, so that runs compare: the scores and which neurons a
step removes, the step sizes, the evaluation of each candidate, the gate (keep or undo), the stop rules, the
checks after the run and the reports. A new strategy is one more entry in `STRATEGIES`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SearchName = Literal["multi_layer", "layer_by_layer"]
DEFAULT: SearchName = "multi_layer"


@dataclass(frozen=True)
class SearchStrategy:
    name: SearchName
    label: str  # what people see
    summary: str  # one plain sentence for the app and reports
    candidate_rule: str  # the sentence in the agent's instructions (v1 and later: {candidate_rule})
    rules: dict = field(default_factory=dict)  # `Rules` fields this strategy sets, unless a run sets them itself


STRATEGIES: dict[str, SearchStrategy] = {
    "multi_layer": SearchStrategy(
        name="multi_layer",
        label="Several layers per step",
        summary="Each step may cut any number of layers at once.",
        # the v1 instructions' original sentence, word for word: runs with this strategy get the same prompt as before
        candidate_rule=("Each candidate is a list of (layer, method) choices, applied together to the current "
                        "model, one fixed-size step per choice."),
    ),
    "layer_by_layer": SearchStrategy(
        name="layer_by_layer",
        label="One layer per step",
        summary="Each step cuts one layer; the agent decides which layer and which kind of cut, step after step.",
        candidate_rule=("Each candidate changes one layer only: a list of (layer, method) choices that all name "
                        "the same layer, one fixed-size step per choice; repeat a method to go further on that "
                        "layer. Different candidates may try different layers."),
        # one step on one layer frees little (0.6B: ~2 MB slimming, ~30 MB removing the layer), so a candidate may
        # take up to 4 steps of a method on its layer instead of 2, to reach a memory limit within the round limit
        rules={"max_layers_per_candidate": 1, "max_same_steps": 4},
    ),
}


def get(name: str) -> SearchStrategy:
    try:
        return STRATEGIES[name]
    except KeyError:
        raise ValueError(f"no search strategy {name!r}; choose one of {', '.join(STRATEGIES)}") from None
