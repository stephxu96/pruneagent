"""The per-layer plan (ARCHITECTURE.md §4.3): the only thing pruning depends on.

`step` turns one agent choice (layer, strategy) into a plan change with the fixed step sizes of §4.2;
`validate` rejects plans that cannot be built. The search rules of §4.4 (at least 25% of MLP width, at most
6 dropped layers, step limits) belong to the Validator, not here.
"""

from __future__ import annotations

import copy
from typing import TypedDict

MLP_ROUND = 64  # MLP widths are kept in multiples of 64 neurons
MLP_STEP_FRACTION = 0.10  # narrow_mlp: 10% of the original width per step
WANDA_STEP = 0.10  # wanda_sparsity: +10 points per step
WANDA_MAX = 0.70
STRATEGIES = ("narrow_mlp", "remove_attention_group", "drop_layer", "wanda_sparsity")


class LayerPlan(TypedDict):
    mlp_keep: int  # MLP neurons kept, multiple of 64 (or the full width)
    groups_keep: int  # key/value groups kept, 1..num_key_value_heads (each with its query heads)
    dropped: bool  # whole layer removed; the other fields are then ignored
    wanda_sparsity: float  # unstructured sparsity on the layer's remaining weights, 0.0-0.7


Plan = dict[int, LayerPlan]


class PlanError(ValueError):
    """The plan cannot be built on this model."""


def _cfg(config) -> dict:
    return config if isinstance(config, dict) else config.to_dict()


def shape(config) -> tuple[int, int, int]:
    """(layers, MLP width, key/value groups) of the unpruned model."""
    c = _cfg(config)
    return c["num_hidden_layers"], c["intermediate_size"], c["num_key_value_heads"]


def dense_plan(config) -> Plan:
    n_layers, width, groups = shape(config)
    return {i: LayerPlan(mlp_keep=width, groups_keep=groups, dropped=False, wanda_sparsity=0.0)
            for i in range(n_layers)}


def mlp_step(width: int) -> int:
    """Neurons removed by one narrow_mlp step: 10% of the original width rounded to 64 (at least 64).
    Qwen3-0.6B: 3,072 -> 320."""
    return max(MLP_ROUND, round(width * MLP_STEP_FRACTION / MLP_ROUND) * MLP_ROUND)


def normalize(plan: dict, config) -> Plan:
    """Fill layers missing from `plan` with the unpruned setting; accept string layer keys (from JSON)."""
    full = dense_plan(config)
    for key, layer in plan.items():
        i = int(key)
        if i not in full:
            raise PlanError(f"layer {i} does not exist (model has {len(full)} layers)")
        unknown = set(layer) - set(LayerPlan.__annotations__)
        if unknown:
            raise PlanError(f"layer {i}: unknown fields {sorted(unknown)}")
        full[i] = LayerPlan(**{**full[i], **layer})
    return full


def problems(plan: Plan, config) -> list[str]:
    """Reasons the plan cannot be built; empty when it can."""
    n_layers, width, groups = shape(config)
    out = []
    if sorted(plan) != list(range(n_layers)):
        out.append(f"plan must cover layers 0..{n_layers - 1}")
    for i, p in sorted(plan.items()):
        if not isinstance(p["dropped"], bool):
            out.append(f"layer {i}: dropped must be true or false")
        if p["dropped"]:
            continue
        g, m, s = p["groups_keep"], p["mlp_keep"], p["wanda_sparsity"]
        if not isinstance(g, int) or not 1 <= g <= groups:
            out.append(f"layer {i}: groups_keep {g} outside 1..{groups}"
                       + (" (to remove every group, drop the layer)" if g == 0 else ""))
        if not isinstance(m, int) or not 1 <= m <= width:
            out.append(f"layer {i}: mlp_keep {m} outside 1..{width}"
                       + (" (to remove the whole MLP, drop the layer)" if m == 0 else ""))
        elif m != width and m % MLP_ROUND:
            out.append(f"layer {i}: mlp_keep {m} is not a multiple of {MLP_ROUND}")
        if not isinstance(s, (int, float)) or not 0.0 <= s <= WANDA_MAX + 1e-9:
            out.append(f"layer {i}: wanda_sparsity {s} outside 0..{WANDA_MAX}")
    if plan and all(p["dropped"] for p in plan.values()):
        out.append("every layer is dropped")
    return out


def validate(plan: dict, config) -> Plan:
    """Normalized plan, or PlanError listing every problem."""
    full = normalize(plan, config)
    errs = problems(full, config)
    if errs:
        raise PlanError("; ".join(errs))
    return full


def step(plan: Plan, layer: int, strategy: str, config) -> Plan:
    """New plan with one step of `strategy` applied to `layer` (§4.2). Raises PlanError if the step is
    impossible (layer already dropped, last group, MLP or sparsity at its limit)."""
    if strategy not in STRATEGIES:
        raise PlanError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    _, width, _ = shape(config)
    new = copy.deepcopy(normalize(plan, config))
    if layer not in new:
        raise PlanError(f"layer {layer} does not exist")
    p = new[layer]
    if p["dropped"]:
        raise PlanError(f"layer {layer} is already dropped")
    if strategy == "narrow_mlp":
        # the first step from a width that is not a multiple of 64 lands on one
        p["mlp_keep"] = (p["mlp_keep"] // MLP_ROUND) * MLP_ROUND if p["mlp_keep"] % MLP_ROUND else p["mlp_keep"]
        p["mlp_keep"] -= mlp_step(width)
    elif strategy == "remove_attention_group":
        p["groups_keep"] -= 1
    elif strategy == "drop_layer":
        p["dropped"] = True
    else:
        p["wanda_sparsity"] = round(p["wanda_sparsity"] + WANDA_STEP, 4)
    errs = problems(new, config)
    if errs:
        raise PlanError(f"{strategy} on layer {layer}: " + "; ".join(errs))
    return new


def apply_steps(plan: Plan, steps: list[dict], config) -> Plan:
    """Apply [{"layer": i, "strategy": s}, ...] in order."""
    for s in steps:
        plan = step(plan, int(s["layer"]), s["strategy"], config)
    return plan


def to_json(plan: Plan) -> dict[str, dict]:
    return {str(i): dict(p) for i, p in sorted(plan.items())}
