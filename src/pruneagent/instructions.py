"""Agent instructions v1 (task 2.9): the prompt the agent sees each round, built from `prompts/instructions_v1.txt`.

The text was approved as written; `render` only fills its placeholders, every one from the code's real values:
the rules (`schema.Rules`) and the gate (`graph.choose`) for what is rejected, what is kept and when the run stops;
the estimator for the memory one step frees; the model config for the number of attention groups; Baseline:
FLAP's perplexity on the gate's 32 windows; and every earlier candidate, rejected ones included. The instructions
describe the evidence but hold no numbers, so the data (the model as it is now, the quick tests and the measured
damage) follows them in a "Data for round N" section, and an unusable last reply's reasons come last.

Layers are named "layer 12" (numbered from 0); a reply may give 12, "12" or "layer 12" (`schema.parse_layer`).
`build_prompt(request, feedback)` is `graph.Deps.build_prompt`'s default.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

from pruneagent import plan as P
from pruneagent import validator as V
from pruneagent.schema import AgentRequest, PastRound, Rules
from pruneagent.search import get as search_strategy

VERSION = "v1"
TEMPLATE_PATH = Path(__file__).with_name("prompts") / "instructions_v1.txt"
MB = 1e6
WIDTH = 78  # the template's line width
# a placeholder: {name} or {name: description}; the JSON in the answer format starts with a quote, so it never matches
PLACEHOLDER = re.compile(r"\{([a-z_]+)(?::[^{}]*)?\}")


def template(version: str = VERSION) -> str:
    """The instructions text of a version (`prompts/instructions_<version>.txt`; task 3.1 adds v2, v3, ...)."""
    return (TEMPLATE_PATH.parent / f"instructions_{version}.txt").read_text(encoding="utf-8")


def placeholders(text: str) -> list[str]:
    """The placeholder names left in `text` (none in a rendered prompt)."""
    return PLACEHOLDER.findall(text)


def _mb(n_bytes: float) -> str:
    """MB with enough digits to tell steps apart: 1,072.9, 31.5, 1.96, 0.025."""
    v = n_bytes / MB
    return f"{v:,.1f}" if abs(v) >= 10 else f"{v:.2f}" if abs(v) >= 1 else f"{v:.3f}"


def _ppl(x: float | None) -> str:
    if x is None:
        return "unknown"
    return f"{x:,.0f}" if x >= 1000 else f"{x:.2f}"


def _pct(x: float) -> str:
    return f"{x:+.1%}"


def _change(value: float, ref: float) -> str:
    """+9.2%, or 1,316,122x once the change is beyond +900%."""
    return _pct(value / ref - 1) if value / ref <= 10 else f"{value / ref:,.0f}x"


def layer_name(i: int) -> str:
    return f"layer {i}"


# ------------------------------------------------------------------------------------------------- rules


def rejection_rule(rules: Rules, current_ppl: float) -> str:
    """The gate's rejection, in the words of `graph.choose`: perplexity more than `step_limit` above the current
    kept model's. (A candidate that frees no memory is also rejected; every method on the v1 menu frees some.)"""
    return (f"its perplexity is more than {rules.step_limit:.0%} above the current model's (above "
            f"{current_ppl * (1 + rules.step_limit):.2f} this round)")


def selection_rule() -> str:
    """`graph.choose`: the smallest perplexity rise per MB freed; an exact tie goes to the faster plan by the speed
    estimate, then to the earlier candidate."""
    return ("the one with the smallest perplexity increase per MB freed; if two are exactly tied, the one "
            "estimated to run faster")


def stop_rule(rules: Rules, max_calls: int, since_kept: int) -> str:
    """`graph.stop_check`'s conditions, in its order."""
    return (f"The run ends as soon as a kept step brings the model within the memory limit. It also ends after "
            f"{rules.max_rounds_without_accept} rounds in a row without a kept step ({since_kept} so far), after "
            f"{rules.max_rounds} rounds or {max_calls} replies in all, or after {rules.max_invalid_in_row} unusable "
            f"replies in a row.")


def limits_text(rules: Rules, n_layers: int, mlp_width: int, n_groups: int) -> str:
    """The validator's rules (`validator.check_candidate`, `check_candidates`), in plain words."""
    min_mlp = -(-mlp_width * rules.min_mlp_fraction // 1)
    return " ".join([
        f"Layers are numbered 0 to {n_layers - 1}; write \"layer 12\".",
        f"Give 1 to {rules.max_candidates} candidates, each with at least one choice.",
        *(["Every choice in one candidate names the same layer."] if rules.max_layers_per_candidate == 1 else
          [f"One candidate changes at most {rules.max_layers_per_candidate} layers."]
          if rules.max_layers_per_candidate else []),
        f"In one candidate, the same method on the same layer at most {V._times(rules.max_same_steps)}.",
        (f"At most {rules.max_drops_per_candidate} drop_layer per candidate and {rules.max_drops_per_run} in the "
        f"whole run; a dropped layer takes no other choice, now or later."),
        (f"Every remaining layer keeps at least {rules.min_mlp_fraction:.0%} of its original feed-forward width "
        f"({int(min_mlp):,} of {mlp_width:,} neurons) and at least {rules.min_groups} of its {n_groups} attention "
        f"groups."),
        ("A candidate may not go more than one step past the memory limit: if it fits, leaving out any one of its "
        "choices must leave it over the limit."),
        "No two candidates may give the same model, and none may repeat a model already measured.",
    ])


# ------------------------------------------------------------------------------------------------ history


def _choices(choices) -> str:
    if not choices:  # start_from_flap (off in v1)
        return "FLAP's allocation"
    return ", ".join(f"{layer_name(c.layer)} {c.strategy}" for c in choices)


def _round_rows(r: PastRound) -> list[str]:
    if not r.candidates:
        return [f"{r.round} | - | {r.note} | | | | |"]
    rows = []
    for c in r.candidates:
        per_mb = "-" if c.rise_per_mb is None else f"{c.rise_per_mb:+.2%}/MB"
        rows.append(f"{r.round} | {c.candidate} | {_choices(c.choices)} | {c.val_ppl:.2f} | {_pct(c.step_rise)} | "
                    f"{_mb(c.freed_bytes)} | {per_mb} | {c.result}")
    return rows


def history_text(req: AgentRequest) -> str:
    """Every earlier candidate, rejected ones included, then the kept plans `rollback_to` can name."""
    if not req.past:
        return "none yet; this is round 1."
    lines = ["", "round | candidate | choices | perplexity | change | MB freed | change per MB | result"]
    for r in req.past:
        lines += _round_rows(r)
    lines.append("Change is against the model as it was that round.")
    if req.kept:
        kept = []
        for i, k in enumerate(req.kept):
            what = "the unpruned model" if i == 0 else f"kept in round {k.round}"
            now = " (current)" if i == len(req.kept) - 1 else ""
            kept.append(f"plan {i}: {what}, {_mb(k.weight_bytes)} MB, perplexity {k.val_ppl:.2f}{now}")
        lines.append("Kept plans you can return to: " + "; ".join(kept) + ".")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------- data


def model_now(req: AgentRequest) -> list[str]:
    """The layers the current plan changes; the rest are unpruned."""
    width, groups = req.mlp_width, req.n_groups
    changed = []
    for i, p in sorted(req.plan.items()):
        if p["dropped"]:
            changed.append(f"- {layer_name(i)}: dropped")
        elif p["mlp_keep"] != width or p["groups_keep"] != groups:
            changed.append(f"- {layer_name(i)}: feed-forward width {p['mlp_keep']:,} of {width:,}, "
                           f"{p['groups_keep']} of {groups} attention groups")
    if not changed:
        return ["The model now: unpruned (every layer at full size)."]
    return ["The model now (layers not listed are at full size):", *changed]


def quick_test_lines(req: AgentRequest) -> list[str]:
    """Quick tests: each layer's estimated damage after 1, 2 and 3 steps of one method alone (relative error in
    the layer's output x1000; drop_layer has one step)."""
    menu = [s for s in P.AGENT_STRATEGIES if s in req.step_bytes] or list(P.AGENT_STRATEGIES)
    lines = [("Quick tests: estimated damage to the layer's output after 1/2/3 steps of each method alone "
             "(x1000; lower is less damage; - when the steps are not possible)."),
             "layer | " + " | ".join(menu)]
    for i, row in sorted(req.sensitivity.rows.items()):
        by = {(t.strategy, t.steps): t.local_error for t in row.quick_tests}
        cells = []
        for s in menu:
            steps = sorted(k for (st, k) in by if st == s)
            if s == "drop_layer":
                steps = steps[:1]
            cells.append("/".join("-" if by[(s, k)] is None else f"{by[(s, k)] * 1000:.3g}" for k in steps) or "-")
        lines.append(f"{layer_name(i)} | " + " | ".join(cells))
    return lines


def measured_lines(req: AgentRequest) -> list[str]:
    """Measured damage: whole-model perplexity after one step on each layer alone, on the unpruned model."""
    table = req.sensitivity
    if not any(row.measured_ppl for row in table.rows.values()):
        return ["Measured damage: not available for this run."]
    menu = list(P.AGENT_STRATEGIES)
    ref = table.dense_ppl
    where = f" on {table.measured_windows} windows" if table.measured_windows else ""
    same = table.measured_windows == getattr(req, "gate_windows", 32)
    note = ("the same text the rounds are scored on" if same
            else "these windows differ from the rounds', so compare within this table")
    head = (f"Measured damage: perplexity after one step on each layer alone, measured on the unpruned model"
            f"{where} (unpruned: {_ppl(ref)}; {note}).")
    lines = [head, "layer | " + " | ".join(menu)]
    for i, row in sorted(table.rows.items()):
        cells = []
        for s in menu:
            v = row.measured_ppl.get(s)
            cells.append("-" if v is None else _ppl(v) + (f" ({_change(v, ref)})" if ref else ""))
        lines.append(f"{layer_name(i)} | " + " | ".join(cells))
    return lines


# ------------------------------------------------------------------------------------------------- render


def values(req: AgentRequest) -> dict[str, str]:
    """Every placeholder's value for this round."""
    rules = req.rules
    rounds_left = req.rounds_left if req.rounds_left is not None else max(0, rules.max_rounds - req.round + 1)
    n_layers, groups, width = req.n_layers or len(req.plan), req.n_groups, req.mlp_width
    return {
        "limit": _mb(req.limit_bytes),
        "flap_ppl": _ppl(req.baseline_ppl),
        "dense_ppl": _ppl(req.dense_ppl),
        "rejection_rule": rejection_rule(rules, req.current_ppl),
        "selection_rule": selection_rule(),
        "rounds_left": str(rounds_left),
        "stop_rule": stop_rule(rules, req.max_agent_calls, req.since_kept),
        "x": _mb(req.step_bytes["narrow_mlp"]),
        "y": _mb(req.step_bytes["remove_attention_group"]),
        "z": _mb(req.step_bytes["drop_layer"]),
        "n_groups": str(groups),
        "current_mb": _mb(req.current_weight_bytes),
        "mb_to_free": _mb(max(0, req.current_weight_bytes - req.limit_bytes)),
        "current_ppl": _ppl(req.current_ppl),
        "history": history_text(req),
        "limits": limits_text(rules, n_layers, width, groups),
        "candidate_rule": search_strategy(req.search).candidate_rule,
    }


def _reflow(text: str, width: int = WIDTH) -> str:
    """Re-wrap each paragraph (and each "- " item) to `width`, as the template is wrapped, once the filled-in
    values have changed line lengths. Paragraphs with a table or JSON are left as they are."""
    out = []
    for block in text.split("\n\n"):
        if " | " in block or '{"' in block:
            out.append(block)
            continue
        items: list[str] = []
        for line in block.splitlines():
            if line.startswith("- ") or not items:
                items.append(line.strip())
            else:
                items[-1] += " " + line.strip()
        out.append("\n".join(textwrap.fill(item, width, subsequent_indent="  " if item.startswith("- ") else "",
                                           break_on_hyphens=False) for item in items))
    return "\n\n".join(out)


def render(req: AgentRequest, text: str | None = None, version: str = VERSION) -> str:
    """The instructions with every placeholder filled. Raises KeyError on a placeholder the code cannot fill."""
    vals = values(req)
    out = _reflow(PLACEHOLDER.sub(lambda m: vals[m.group(1)], template(version) if text is None else text))
    return "\n".join(line.rstrip() for line in out.splitlines())


def build_prompt(req: AgentRequest, feedback: str | None = None, version: str = VERSION) -> str:
    """`graph.Deps.build_prompt`: the rendered instructions, the data for this round, and the reasons an
    unusable last reply was rejected."""
    parts = [render(req, version=version).rstrip(), "", f"Data for round {req.round}.", "", *model_now(req), "",
             *quick_test_lines(req), "", *measured_lines(req)]
    if feedback:
        parts += ["", "About your previous reply.", feedback]
    return "\n".join(parts) + "\n"


def __getattr__(name: str):
    """`build_prompt_v2`, `build_prompt_v3`, ...: the prompt builder for that instructions version, so a run's
    `setup.json` can name it ("pruneagent.instructions:build_prompt_v2") and a resume uses the same version."""
    m = re.fullmatch(r"build_prompt_(v\d+)", name)
    if m and (TEMPLATE_PATH.parent / f"instructions_{m.group(1)}.txt").exists():
        version = m.group(1)

        def build(req: AgentRequest, feedback: str | None = None) -> str:
            return build_prompt(req, feedback, version=version)

        build.__name__ = name
        return build
    raise AttributeError(name)
