"""Validator (ARCHITECTURE.md §4.4, FR4, task 2.2): pure checks of an agent reply against the fixed rules.

Every reason is written for the agent, which gets them back verbatim: it says which candidate and choice is wrong,
what the rule is, and what would be allowed. Candidates and choices are numbered from 1 in the messages.

Replies are read in either form (`schema.normalize_reply`): the v1 instructions' {"candidates": [{"choices":
[{"layer": "layer 12", "method": ...}], "reason": ...}]} or {"rollback_to": n}, and the internal {"action",
"rationale", "candidates": [{"choices": [{"layer": 12, "strategy": ...}], "rationale"}], ...}. Messages use the
instructions' words ("method", "kept plan").

Rules, in the order they are checked:
- Structure: an action the agent may take (`Context.actions`; v1: propose candidates or roll back) with a
  rationale; each candidate is a list of choices (or {"choices", "rationale"}); each choice is exactly {"layer",
  "method"} with an existing layer and a method on the agent's menu (`Context.strategies`; v1: the three
  structured methods, not wanda_sparsity). Any other field (an amount, a ratio, a step count) is rejected: code
  sets every amount.
- Step sizes: each choice is one fixed step (§4.2); the same strategy on the same layer at most `max_same_steps`
  times per candidate; a layer is dropped at most once, and a dropped layer takes no other choice.
- Structure rules: at most `max_drops_per_candidate` drops per candidate and `max_drops_per_run` in the run; no
  choice on a layer the current plan has already dropped; every remaining layer keeps at least `min_mlp_fraction`
  of its original MLP width and `min_groups` key/value group(s), and unstructured sparsity at most
  `max_wanda_sparsity`.
- Budget: a candidate may not overshoot the memory limit by more than one step. Precisely: if the candidate's
  weights fit within the limit, every choice that frees memory must be needed, i.e. leaving out any one of them
  must leave the weights over the limit. A candidate that stays over the limit is allowed (later rounds continue
  from it). Memory is weight memory as `estimate.weight_bytes` counts it, the measure Baseline: FLAP is cut to.
- Candidates: 1 to `max_candidates`; no two give the same plan, and none repeats a plan already evaluated.

`check_reply` checks a whole reply; `check_candidates` and `check_candidate` the apply part. All are pure: they
read the model's config (shape only), the current plan and the limit, and return a `CheckResult`.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, get_args

from pruneagent import estimate as E
from pruneagent import plan as P
from pruneagent.schema import (
    V1_ACTIONS,
    Action,
    AgentReply,
    CandidateCheck,
    CheckResult,
    Rules,
    normalize_candidate,
    normalize_choice,
    normalize_reply,
)

ACTIONS = get_args(Action)
MB = 1e6
ROLLBACK_HINT = 'answer {"rollback_to": <index of an earlier kept plan>} to go back'


@dataclass(frozen=True)
class Context:
    """What the rules are checked against.

    config: the unpruned model's config (object or dict); plan: the current (last accepted) plan; limit_bytes:
    the memory limit; tried: plans already evaluated in earlier rounds; n_accepted: accepted plans in the history,
    counting the starting plan (rollback targets are 0..n_accepted-2); strategies: the methods the agent may
    propose; actions: what the agent may do."""

    config: Any
    plan: dict
    limit_bytes: int
    rules: Rules = field(default_factory=Rules)
    tried: tuple = ()
    n_accepted: int = 1
    only_layer: int | None = None  # sweep strategies: the one layer this round may change
    strategies: tuple = P.AGENT_STRATEGIES
    actions: tuple = V1_ACTIONS


def _mb(n: float) -> str:
    return f"{n / MB:,.3f} MB"


def _key(plan: dict) -> str:
    return json.dumps(P.to_json(plan), sort_keys=True)


def _times(n: int) -> str:
    return "once" if n == 1 else "twice" if n == 2 else f"{n} times"


# ------------------------------------------------------------------------------------------------- choices


def _choice(raw: Any, where: str, n_layers: int,
            strategies: tuple = P.AGENT_STRATEGIES) -> tuple[tuple[int, str] | None, list[str]]:
    if not isinstance(raw, dict):
        return None, [(f"{where} is {json.dumps(raw)}; a choice is an object {{\"layer\": \"layer <number>\", "
                      f"\"method\": <name>}}.")]
    raw = normalize_choice(raw)  # "method" -> "strategy", "layer 12" -> 12
    extra = sorted(set(raw) - {"layer", "strategy"})
    missing = [{"strategy": "method"}.get(k, k) for k in ("layer", "strategy") if k not in raw]
    reasons = []
    if extra:
        reasons.append(f"{where} has {'fields' if len(extra) > 1 else 'field'} {', '.join(map(repr, extra))}; give "
                       f"only \"layer\" and \"method\". Code sets every amount: each choice is one fixed step.")
    if missing:
        reasons.append(f"{where} is missing {' and '.join(map(repr, missing))}.")
    if missing:
        return None, reasons
    layer, strategy = raw["layer"], raw["strategy"]
    if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < n_layers:
        reasons.append(f"{where}: layer {json.dumps(layer)} does not exist; layers are numbered 0 to "
                       f"{n_layers - 1} (write \"layer 12\").")
    if strategy in P.STRATEGIES and strategy not in strategies:
        reasons.append(f"{where}: {strategy} is not one of the methods you can use; use one of "
                       f"{', '.join(strategies)}.")
    elif strategy not in strategies:
        reasons.append(f"{where}: unknown method {json.dumps(strategy)}; use one of {', '.join(strategies)}.")
    return (None, reasons) if reasons else ((layer, strategy), [])


def _layer_after(cur: P.LayerPlan, counts: Counter, width: int) -> P.LayerPlan:
    """A layer's settings after the given number of steps of each strategy (as `plan.step` applies them)."""
    p = dict(cur)
    for _ in range(counts["narrow_mlp"]):
        p["mlp_keep"] = (p["mlp_keep"] // P.MLP_ROUND) * P.MLP_ROUND - P.mlp_step(width)
    p["groups_keep"] -= counts["remove_attention_group"]
    p["wanda_sparsity"] = round(p["wanda_sparsity"] + P.WANDA_STEP * counts["wanda_sparsity"], 4)
    p["dropped"] = p["dropped"] or counts["drop_layer"] > 0
    return P.LayerPlan(**p)


def _more_steps(cur: P.LayerPlan, strategy: str, width: int, rules: Rules) -> int:
    """How many more steps of `strategy` the layer can take within the minimums."""
    k = 0
    while k < 100:
        nxt = _layer_after(cur, Counter({strategy: k + 1}), width)
        if _minimum_problem(nxt, strategy, width, rules):
            return k
        k += 1
    return k


def _minimum_problem(p: P.LayerPlan, strategy: str, width: int, rules: Rules) -> str | None:
    min_mlp = math.ceil(width * rules.min_mlp_fraction)
    if strategy == "narrow_mlp" and p["mlp_keep"] < min_mlp:
        return (f"would leave {max(p['mlp_keep'], 0):,} of its {width:,} MLP neurons, below the minimum of "
                f"{min_mlp:,} ({rules.min_mlp_fraction:.0%})")
    if strategy == "remove_attention_group" and p["groups_keep"] < rules.min_groups:
        return (f"would leave {max(p['groups_keep'], 0)} key/value groups; at least {rules.min_groups} must remain "
                f"(drop_layer removes the whole layer)")
    if strategy == "wanda_sparsity" and p["wanda_sparsity"] > rules.max_wanda_sparsity + 1e-9:
        return (f"would bring its unstructured sparsity to {p['wanda_sparsity']:.0%}, above the maximum of "
                f"{rules.max_wanda_sparsity:.0%}")
    return None


# ----------------------------------------------------------------------------------------------- candidates


def _choices_of(raw: Any, name: str) -> tuple[list | None, list[str]]:
    """The raw choices list of a candidate given as a list or as {"choices": [...], "reason": "..."}."""
    raw = normalize_candidate(raw)  # "reason" -> "rationale"
    if isinstance(raw, dict):
        extra = sorted(set(raw) - {"choices", "rationale"})
        if extra:
            return None, [(f"{name} has {'fields' if len(extra) > 1 else 'field'} {', '.join(map(repr, extra))}; a "
                          f"candidate has only \"choices\" and \"reason\". Code sets every amount.")]
        raw = raw.get("choices")
    if not isinstance(raw, list):
        return None, [f"{name} has no list of choices."]
    if not raw:
        return None, [f"{name} has no choices; give at least one {{\"layer\", \"method\"}}."]
    return raw, []


def check_candidate(raw: Any, ctx: Context, index: int = 0) -> CandidateCheck:
    """One candidate against every rule except duplicates and the candidate count (see `check_candidates`)."""
    name = f"candidate {index + 1}"
    rules = ctx.rules
    n_layers, width, _ = P.shape(ctx.config)

    def fail(reasons: list[str]) -> CandidateCheck:
        return CandidateCheck(index=index, ok=False, reasons=reasons)

    raw_choices, reasons = _choices_of(raw, name)
    if raw_choices is None:
        return fail(reasons)
    choices: list[tuple[int, str]] = []
    for k, ch in enumerate(raw_choices):
        parsed, why = _choice(ch, f"{name}, choice {k + 1}", n_layers, ctx.strategies)
        reasons += why
        if parsed:
            choices.append(parsed)
    if reasons:
        return fail(reasons)

    current = P.normalize(ctx.plan, ctx.config)
    counts = Counter(choices)
    by_layer: dict[int, Counter] = {}
    for (layer, strategy), n in counts.items():
        by_layer.setdefault(layer, Counter())[strategy] = n

    # step sizes: one fixed step per choice, a bounded number per layer
    for (layer, strategy), n in sorted(counts.items()):
        if strategy == "drop_layer" and n > 1:
            reasons.append(f"{name}: drop_layer on layer {layer} is chosen {_times(n)}; a layer can be dropped only "
                           f"once.")
        elif n > rules.max_same_steps:
            reasons.append(f"{name}: {strategy} on layer {layer} is chosen {_times(n)}; at most "
                           f"{rules.max_same_steps} steps of the same method on one layer per candidate.")
    for layer, c in sorted(by_layer.items()):
        if current[layer]["dropped"]:
            reasons.append(f"{name}: layer {layer} is already dropped in the current plan; choose a remaining layer.")
        elif c["drop_layer"] and len(c) > 1:
            others = ", ".join(sorted(s for s in c if s != "drop_layer"))
            reasons.append(f"{name}: layer {layer} is dropped and also given {others}; a dropped layer takes no "
                           f"other choice.")

    # search strategy (search.py): which layers, and how many, one candidate may change
    if ctx.only_layer is not None and set(by_layer) - {ctx.only_layer}:
        others = ", ".join(map(str, sorted(set(by_layer) - {ctx.only_layer})))
        reasons.append(f"{name} changes layer {others}; this round decides layer {ctx.only_layer} only. Every choice "
                       f"names layer {ctx.only_layer}.")
    if rules.max_layers_per_candidate is not None and len(by_layer) > rules.max_layers_per_candidate:
        layers = ", ".join(map(str, sorted(by_layer)))
        one = "one layer" if rules.max_layers_per_candidate == 1 else f"{rules.max_layers_per_candidate} layers"
        reasons.append(f"{name} changes {len(by_layer)} layers ({layers}); in this run a candidate changes at most "
                       f"{one}. Put the other layers in separate candidates or later rounds.")

    # structure rules: drops per candidate and per run, per-layer minimums
    drops = sorted(layer for layer, c in by_layer.items() if c["drop_layer"])
    if len(drops) > rules.max_drops_per_candidate:
        reasons.append(f"{name} drops {len(drops)} layers ({', '.join(map(str, drops))}); at most "
                       f"{rules.max_drops_per_candidate} drop_layer per candidate.")
    before = sum(p["dropped"] for p in current.values())
    new_drops = [la for la in drops if not current[la]["dropped"]]
    if new_drops and before + len(new_drops) > rules.max_drops_per_run:
        reasons.append(f"{name} would bring the run to {before + len(new_drops)} dropped layers; at most "
                       f"{rules.max_drops_per_run} per run ({before} already dropped).")
    for layer, c in sorted(by_layer.items()):
        if current[layer]["dropped"] or c["drop_layer"]:
            continue
        after = _layer_after(current[layer], c, width)
        for strategy in P.STRATEGIES:
            if c[strategy] and (problem := _minimum_problem(after, strategy, width, rules)):
                more = _more_steps(current[layer], strategy, width, rules)
                reasons.append(f"{name}: {c[strategy]} {strategy} step{'s' if c[strategy] > 1 else ''} on layer "
                               f"{layer} {problem}; this layer can take {more} more {strategy} "
                               f"step{'' if more == 1 else 's'}.")
    if reasons:
        return fail(reasons)

    try:
        plan = P.apply_steps(current, [{"layer": la, "strategy": s} for la, s in choices], ctx.config)
    except P.PlanError as e:  # not expected after the checks above; reported rather than raised
        return fail([f"{name} cannot be built: {e}."])

    # budget: no more than one step past the memory limit
    size = E.weight_bytes(ctx.config, plan)
    if size <= ctx.limit_bytes:
        for (layer, strategy), n in sorted(counts.items()):
            fewer = Counter(by_layer[layer])
            fewer[strategy] -= 1
            without = dict(plan)
            without[layer] = _layer_after(current[layer], fewer, width)
            freed = E.weight_bytes(ctx.config, without) - size
            if freed > 0 and size + freed <= ctx.limit_bytes:
                return fail([(f"{name} frees more memory than the limit needs: its weights would be {_mb(size)} "
                             f"against a limit of {_mb(ctx.limit_bytes)}, and it would still fit without one "
                             f"{strategy} on layer {layer} (which frees {_mb(freed)}). Remove choices until every "
                             f"choice that frees memory is needed to reach the limit.")])
    return CandidateCheck(index=index, ok=True, plan=plan, weight_bytes=size)


def check_candidates(raw: Any, ctx: Context) -> CheckResult:
    """The candidates of an apply reply: each one (`check_candidate`), their number, and no duplicates."""
    rules = ctx.rules
    if not isinstance(raw, list) or not raw:
        return CheckResult(ok=False, reasons=[f"No candidates; propose 1 to {rules.max_candidates}."])
    reasons: list[str] = []
    if len(raw) > rules.max_candidates:
        reasons.append(f"{len(raw)} candidates proposed; at most {rules.max_candidates} per round.")
    seen = {_key(p): None for p in ctx.tried}
    checks = []
    for k, cand in enumerate(raw):
        if k >= rules.max_candidates:
            checks.append(CandidateCheck(index=k, ok=False, reasons=[
                f"candidate {k + 1} is beyond the first {rules.max_candidates} and was not checked."]))
            continue
        c = check_candidate(cand, ctx, k)
        if c.ok:
            key = _key(c.plan)
            if key in seen:
                earlier = seen[key]
                why = (f"candidate {k + 1} gives the same plan as candidate {earlier + 1} (the same choices, perhaps "
                       f"in another order); propose a different candidate." if earlier is not None else
                       f"candidate {k + 1} gives a plan already evaluated in an earlier round; propose a different "
                       f"candidate.")
                c = CandidateCheck(index=k, ok=False, reasons=[why])
            else:
                seen[key] = k
        reasons += c.reasons
        checks.append(c)
    return CheckResult(ok=not reasons, reasons=reasons, candidates=checks)


# -------------------------------------------------------------------------------------------------- replies


def check_reply(reply: Any, ctx: Context) -> CheckResult:
    """A whole agent reply (a dict as parsed from JSON, in either form, or an `AgentReply`)."""
    if isinstance(reply, AgentReply):
        reply = reply.model_dump(exclude_defaults=True)
    if not isinstance(reply, dict):
        return CheckResult(ok=False, reasons=["The reply is not a JSON object."])
    reply = normalize_reply(reply)
    reasons: list[str] = []
    extra = sorted(set(reply) - set(AgentReply.model_fields))
    if extra:
        allowed = ("candidates, rollback_to" if set(ctx.actions) <= set(V1_ACTIONS)
                   else ", ".join(AgentReply.model_fields))
        reasons.append(f"The reply has unknown {'fields' if len(extra) > 1 else 'field'} {', '.join(map(repr, extra))}"
                       f"; allowed: {allowed}. Code sets every amount.")
    action = reply.get("action")
    if action is None and "action" not in reply:  # the v1 form, with both or neither of its two answers
        if reply.get("candidates") not in (None, []) and reply.get("rollback_to") is not None:
            reasons.append("The reply gives both candidates and rollback_to; give one: candidates to try, or "
                           "rollback_to alone to go back.")
        else:
            reasons.append(f"The reply has no candidates; propose 1 to {ctx.rules.max_candidates}, or "
                           f"{ROLLBACK_HINT}.")
        return CheckResult(ok=False, reasons=reasons)
    if action not in ACTIONS:
        reasons.append(f"Unknown action {json.dumps(action)}; use one of {', '.join(ctx.actions)}.")
        return CheckResult(ok=False, reasons=reasons)
    if action not in ctx.actions:
        reasons.append(f"Action {action!r} is not available in this version; propose 1 to "
                       f"{ctx.rules.max_candidates} candidates, or {ROLLBACK_HINT}.")
        return CheckResult(ok=False, reasons=reasons)
    if not isinstance(reply.get("rationale"), str) or not reply["rationale"].strip():
        reasons.append("The reply has no rationale; every action needs one.")
    uses = {"apply": "candidates", "probe": "probe", "rollback": "rollback_to"}
    for f in ("candidates", "probe", "rollback_to"):
        if reply.get(f) not in (None, []) and uses.get(action) != f:
            reasons.append(f"Action {action!r} takes no {f!r}.")

    n_layers = P.shape(ctx.config)[0]
    if action == "apply":
        res = check_candidates(reply.get("candidates"), ctx)
        return CheckResult(ok=not reasons and res.ok, reasons=reasons + res.reasons, candidates=res.candidates)
    if action == "probe":
        tests = reply.get("probe")
        if not isinstance(tests, list) or not tests:
            reasons.append("Action 'probe' needs a non-empty \"probe\" list of {\"layer\", \"method\"} pairs.")
        else:
            current = P.normalize(ctx.plan, ctx.config)
            for k, ch in enumerate(tests):
                parsed, why = _choice(ch, f"probe {k + 1}", n_layers, ctx.strategies)
                reasons += why
                if parsed and current[parsed[0]]["dropped"]:
                    reasons.append(f"probe {k + 1}: layer {parsed[0]} is already dropped in the current plan.")
    if action == "rollback":
        target = reply.get("rollback_to")
        latest = ctx.n_accepted - 2  # the most recent plan before the current one
        if latest < 0:
            reasons.append("Nothing to roll back to: no step has been kept yet, so the current plan is the "
                           "unpruned model.")
        elif target is not None and (isinstance(target, bool) or not isinstance(target, int)
                                     or not 0 <= target <= latest):
            reasons.append(f"rollback_to {json.dumps(target)} is not an earlier kept plan; use 0 to {latest} "
                           f"(0 is the unpruned model).")
    return CheckResult(ok=not reasons, reasons=reasons)
