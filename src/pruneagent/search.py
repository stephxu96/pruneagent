"""Search strategies: how much one candidate may change (ADR-009).

A search strategy is the one part of a run that decides the shape of the agent's steps. It sets two things:

- rule changes (`Rules` fields), which the validator enforces, and
- one sentence of the agent's instructions saying what a candidate is.

Everything else is shared by every strategy and must stay so, so that runs compare: the scores and which neurons a
step removes, the step sizes, the evaluation of each candidate, the gate (keep or undo), the stop rules, the
checks after the run and the reports. A new strategy is one more entry in `STRATEGIES`.

`layer_by_layer` is a sweep: round k decides layer k (first to last) from a bank of cuts for that layer (slim 1-7
steps, remove 1-7 attention units, remove the layer, or leave it), up to 3 cut plans measured per round, the best
kept by the shared gate. The run ends after the last layer (or earlier, when the memory limit is met).
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
    # the other strategy-specific sentences of the instructions, by placeholder (v1: {return_rule}, {rollback_line})
    texts: dict = field(default_factory=dict)
    # "free": each round the agent may change any layers. "sweep": one pass through the model, first layer to last;
    # round k decides layer k only (or leaves it as it is), so a run has exactly one decision per layer
    order: Literal["free", "sweep"] = "free"


STRATEGIES: dict[str, SearchStrategy] = {
    "multi_layer": SearchStrategy(
        name="multi_layer",
        label="Several layers per step",
        summary="Each step may cut any number of layers at once.",
        # the v1 instructions' original sentence, word for word: runs with this strategy get the same prompt as before
        candidate_rule=("Each candidate is a list of (layer, method) choices, applied together to the current "
                        "model, one fixed-size step per choice."),
        texts={  # word for word as before
            "return_rule": "You can return to an earlier kept plan, but a later step cannot undo the damage of an "
                           "earlier one.",
            "rollback_line": 'To go back, answer {"rollback_to": <index of an earlier kept plan>} instead of\n'
                             'candidates.',
        },
    ),
    "layer_by_layer": SearchStrategy(
        name="layer_by_layer",
        label="One layer per step",
        summary=("One pass through the model, first layer to last: each round decides one layer. The AI proposes "
                 "up to 3 cut plans for it from a fixed bank of cuts; code measures them and keeps the best."),
        candidate_rule=("This run goes through the layers in order, one layer per round (see \"This round\" "
                        "below). Each candidate is a cut plan for this round's layer only: a list of (layer, method) "
                        "choices that all name that layer, one fixed-size step per choice; repeat a method to cut "
                        "deeper. Propose no candidates to leave the layer as it is. A layer is decided once: later "
                        "rounds cannot change it, and there is no returning to an earlier plan."),
        # the bank: every cut code can make on one layer (up to 7 steps of slimming or of attention units, or the
        # whole layer); the per-layer minimums still apply
        rules={"max_layers_per_candidate": 1, "max_same_steps": 7},
        texts={
            "return_rule": "A later step cannot undo the damage of an earlier one, so pace the cuts: the memory "
                           "still to free is shared by the layers left.",
            "rollback_line": 'To leave the layer as it is, answer {"candidates": []}.',
        },
        order="sweep",
    ),
}


def get(name: str) -> SearchStrategy:
    try:
        return STRATEGIES[name]
    except KeyError:
        raise ValueError(f"no search strategy {name!r}; choose one of {', '.join(STRATEGIES)}") from None
