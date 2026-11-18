"""One agent round wired by hand (task 1.11): prompt -> Codex CLI -> parse -> validate -> apply -> measure.

The agent sees the objective, the memory limit, the strategy menu (§4.2), the quick-test (probe) table, the
current plan and the current and target weight memory (§4.5). It answers with up to three candidates, each a
list of (layer, strategy) choices and a rationale, with no amounts: code applies one fixed step per choice
(`plan.step`), checks the fixed rules (§4.4), builds each valid candidate from cached scores and measures it.

Rules checked here (§4.4), on top of `plan.validate`:
- at most 2 steps of the same strategy on one layer per candidate;
- at most 1 layer dropped per candidate and 6 per run;
- at least 25% of the original MLP width and 1 key/value group per remaining layer; Wanda sparsity <= 70%;
- a dropped layer takes no other choice in the same candidate (the result would depend on the order);
- at most 3 candidates per round (extra ones are rejected).
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
import time
from collections import Counter

from pruneagent import estimate as E
from pruneagent import plan as P

MAX_CANDIDATES = 3
MAX_SAME_STEPS = 2  # same strategy on one layer, per candidate
MAX_DROPS_PER_STEP = 1
MAX_DROPS_PER_RUN = 6
MIN_MLP_FRACTION = 0.25
MAX_STEP_RISE = 0.15  # per-step limit (§4.4): told to the agent; not applied in this hand-wired round
DEFAULT_MEMORY_CUT = 0.10  # memory limit: 10% less weight memory than the unpruned model
MB = 1e6

OBJECTIVE = ("Fit the model within the memory limit while keeping the most quality "
             "(lowest perplexity on held-out text).")


def _mb(n_bytes: float, digits: int = 1) -> str:
    return f"{n_bytes / MB:,.{digits}f} MB"


def step_savings(config) -> dict[str, int]:
    """Weight bytes freed by one step of each strategy on one unpruned layer (the same on every layer)."""
    base = P.dense_plan(config)
    full = E.weight_bytes(config, base)
    return {s: full - E.weight_bytes(config, P.step(base, 0, s, config)) for s in P.STRATEGIES}


def strategy_menu(config) -> dict[str, str]:
    """Neutral, equal-detail description of each strategy: what one step does to the weights and the memory
    it frees. No ranking."""
    _, width, groups = P.shape(config)
    c = P._cfg(config)
    per_group = c["num_attention_heads"] // groups
    save = step_savings(config)
    return {
        "narrow_mlp": f"delete {P.mlp_step(width)} of the layer's {width:,} MLP neurons (10% of the original width), "
                      f"the lowest-scoring ones; frees {_mb(save['narrow_mlp'], 3)}.",
        "remove_attention_group": f"delete one of the layer's {groups} key/value groups (its key and value heads and "
                                  f"the {per_group} query heads that share them), the lowest-scoring one; frees "
                                  f"{_mb(save['remove_attention_group'], 3)}.",
        "drop_layer": f"delete the whole layer (attention and MLP); its input passes straight to the next layer; "
                      f"frees {_mb(save['drop_layer'], 3)}.",
        "wanda_sparsity": f"set a further {P.WANDA_STEP:.0%} of the layer's remaining weights to zero, the lowest-"
                          f"scoring ones (|weight| x input size); matrices keep their shape, so it frees "
                          f"{_mb(save['wanda_sparsity'], 3)}.",
    }


def _fmt_err(e: float | None) -> str:
    return "-" if e is None else f"{e * 1000:.3g}"


def sensitivity_lines(table: dict) -> list[str]:
    """Compact quick-test table: one row per layer, error x1000 at 1/2/3 steps per strategy."""
    head = ["layer", "block_influence"] + [s for s in P.STRATEGIES]
    lines = [" | ".join(head)]
    for key in sorted(table, key=int):
        lp = table[key]
        by = {(c["strategy"], c["steps"]): c["local_error"] for c in lp["candidates"]}
        cells = [str(int(key)), f"{lp['block_influence']:.3f}"]
        for s in P.STRATEGIES:
            steps = [k for (st, k) in by if st == s]
            if s == "drop_layer":
                cells.append(_fmt_err(by.get((s, 1))))
            else:
                cells.append("/".join(_fmt_err(by.get((s, k))) for k in sorted(steps)))
        lines.append(" | ".join(cells))
    return lines


def memory_limit(config, cut: float = DEFAULT_MEMORY_CUT) -> int:
    """Default memory limit: `cut` less weight memory than the unpruned model."""
    return int(E.weight_bytes(config, P.dense_plan(config)) * (1 - cut))


def build_prompt(config, table: dict, limit_bytes: int, plan: P.Plan | None = None) -> str:
    plan = P.normalize(plan or {}, config)
    n_layers, width, groups = P.shape(config)
    current = E.weight_bytes(config, plan)
    dense = P.dense_plan(config)
    pruned = {i: p for i, p in plan.items() if p != dense[i]}
    plan_text = ("all layers unpruned" if not pruned else
                 "; ".join(f"layer {i}: {json.dumps(p)}" for i, p in sorted(pruned.items())))
    menu = strategy_menu(config)
    rows = sensitivity_lines(table)
    return "\n".join([
        "You plan structured pruning of a language model. Answer with JSON only; do not run commands or read files.",
        "",
        f"Objective: {OBJECTIVE}",
        f"Model: {n_layers} layers, each with an MLP of {width:,} neurons and {groups} key/value attention groups.",
        (f"Weight memory: current {_mb(current, 3)}; limit {_mb(limit_bytes, 3)}; "
        f"to remove: {_mb(max(0, current - limit_bytes), 3)}."),
        f"Current plan: {plan_text}.",
        "",
        "Strategies (one step each; code picks what is removed and the amount):",
        *[f"- {s}: {t}" for s, t in menu.items()],
        "",
        ("Rules per candidate: each choice is one step; the same strategy on the same layer at most twice; "
        f"at most {MAX_DROPS_PER_STEP} drop_layer, and no other choice on a dropped layer; keep at least "
        f"{MIN_MLP_FRACTION:.0%} of MLP width and 1 key/value group per layer; sparsity at most {P.WANDA_MAX:.0%}."),
        (f"Code builds each candidate and keeps the one with the lowest perplexity, if that is at most "
         f"{MAX_STEP_RISE:.0%} above the current plan's; later rounds continue from it."),
        "",
        ("Quick-test table: change in the layer's output when the strategy is applied alone, relative error x1000 "
        "after 1/2/3 steps (drop_layer: 1 step only); block_influence = how much the layer changes its input."),
        *rows,
        "",
        f"Propose up to {MAX_CANDIDATES} alternative candidates. Reply with this JSON and nothing else:",
        '{"candidates": [{"choices": [{"layer": 0, "strategy": "narrow_mlp"}], "rationale": "..."}]}',
    ])


# ---------------------------------------------------------------------------------------------- Codex call


def call_codex(prompt: str, timeout: int = 900) -> dict:
    """`codex exec --skip-git-repo-check --sandbox read-only "<prompt>"` from a temporary empty directory.
    Returns stdout (the final message), stderr, exit code, seconds, model and total tokens reported by Codex."""
    with tempfile.TemporaryDirectory(prefix="pruneagent-codex-") as tmp:
        start = time.perf_counter()
        proc = subprocess.run(["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only", prompt],
                              cwd=tmp, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL,
                              check=False)
        seconds = time.perf_counter() - start
    return {"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode,
            "seconds": round(seconds, 1), "tokens_used": tokens_used(proc.stderr + "\n" + proc.stdout),
            "model": _header(proc.stderr, "model"), "reasoning_effort": _header(proc.stderr, "reasoning effort")}


def tokens_used(text: str) -> int | None:
    m = re.search(r"tokens used\s*:?\s*\n?\s*([\d,]+)", text, flags=re.IGNORECASE)
    return int(m.group(1).replace(",", "")) if m else None


def _header(text: str, key: str) -> str | None:
    m = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else None


# ------------------------------------------------------------------------------------------------- parsing


def extract_json(text: str) -> dict:
    """The first JSON object with a "candidates" key in `text` (bare, fenced, or surrounded by prose)."""
    dec = json.JSONDecoder()
    found = None
    for i in [m.start() for m in re.finditer(r"\{", text)]:
        try:
            obj, _ = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "candidates" in obj:
            return obj
        found = found or (obj if isinstance(obj, dict) else None)
    raise ValueError("no JSON object with a 'candidates' key in the reply"
                     + (f" (found keys {sorted(found)})" if found else ""))


def parse_reply(text: str) -> list[dict]:
    """Candidates from the agent's reply as [{"choices": [{"layer", "strategy"}], "rationale": str}].
    Raises ValueError when the reply has no usable candidates list. Choice contents are checked later."""
    obj = extract_json(text)
    cands = obj["candidates"]
    if not isinstance(cands, list):
        raise TypeError("'candidates' is not a list")
    out = []
    for c in cands:
        if isinstance(c, list):  # a bare list of choices
            c = {"choices": c, "rationale": obj.get("rationale", "")}
        if not isinstance(c, dict):
            out.append({"choices": c, "rationale": ""})
            continue
        out.append({"choices": c.get("choices", c.get("steps")), "rationale": str(c.get("rationale", ""))})
    return out


# ---------------------------------------------------------------------------------------------- validation


def check_candidate(choices, config, base: P.Plan | None = None, dropped_before: int = 0) -> tuple[P.Plan | None,
                                                                                                    list[str]]:
    """(plan after one step per choice, []) or (None, reasons) under the rules in the module docstring."""
    if not isinstance(choices, list) or not choices:
        return None, ["no choices"]
    reasons: list[str] = []
    clean: list[tuple[int, str]] = []
    n_layers, width, _ = P.shape(config)
    for k, ch in enumerate(choices):
        if not isinstance(ch, dict) or set(ch) - {"layer", "strategy"} or not {"layer", "strategy"} <= set(ch):
            reasons.append(f"choice {k}: expected {{'layer', 'strategy'}}, got {ch!r}")
            continue
        layer, strategy = ch["layer"], ch["strategy"]
        if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < n_layers:
            reasons.append(f"choice {k}: layer {layer!r} is not a layer index 0..{n_layers - 1}")
        elif strategy not in P.STRATEGIES:
            reasons.append(f"choice {k}: unknown strategy {strategy!r}")
        else:
            clean.append((layer, strategy))
    counts = Counter(clean)
    for (layer, strategy), n in sorted(counts.items()):
        if n > MAX_SAME_STEPS:
            reasons.append(f"layer {layer}: {strategy} {n} times (at most {MAX_SAME_STEPS} per candidate)")
    drops = sorted({layer for layer, s in clean if s == "drop_layer"})
    if len(drops) > MAX_DROPS_PER_STEP:
        reasons.append(f"drops {len(drops)} layers {drops} (at most {MAX_DROPS_PER_STEP} per candidate)")
    if dropped_before + len(drops) > MAX_DROPS_PER_RUN:
        reasons.append(f"would drop {dropped_before + len(drops)} layers in the run (at most {MAX_DROPS_PER_RUN})")
    for layer in drops:
        others = sorted({s for la, s in clean if la == layer and s != "drop_layer"})
        if others:
            reasons.append(f"layer {layer}: dropped and also {', '.join(others)}")
    if reasons:
        return None, reasons
    plan = P.normalize(base or {}, config)
    for layer, strategy in clean:
        try:
            plan = P.step(plan, layer, strategy, config)
        except P.PlanError as e:
            return None, [str(e)]
    try:
        plan = P.validate(plan, config)
    except P.PlanError as e:
        return None, [str(e)]
    min_mlp = math.ceil(width * MIN_MLP_FRACTION)
    for i, p in sorted(plan.items()):
        if not p["dropped"] and p["mlp_keep"] < min_mlp:
            reasons.append(f"layer {i}: MLP width {p['mlp_keep']} below {MIN_MLP_FRACTION:.0%} ({min_mlp})")
    return (None, reasons) if reasons else (plan, [])


def validate_candidates(cands: list[dict], config, base: P.Plan | None = None) -> list[dict]:
    """One record per candidate: valid or not, reasons, plan and weight memory."""
    dropped_before = sum(p["dropped"] for p in P.normalize(base or {}, config).values())
    out = []
    for k, c in enumerate(cands):
        if k >= MAX_CANDIDATES:
            out.append({"index": k, "valid": False, "reasons": [f"more than {MAX_CANDIDATES} candidates"]})
            continue
        plan, reasons = check_candidate(c["choices"], config, base, dropped_before)
        rec = {"index": k, "valid": plan is not None, "reasons": reasons}
        if plan is not None:
            rec["plan"] = P.to_json(plan)
            rec["weight_bytes"] = E.weight_bytes(config, plan)
            rec["steps"] = dict(Counter(s for s in (ch["strategy"] for ch in c["choices"])))
        out.append(rec)
    return out


# ---------------------------------------------------------------------------------------------- summary


def flap_weight_bytes(config, groups_kept: list[int], neurons_kept: list[int], dtype_bytes: int = 2) -> int:
    """Weight bytes of a FLAP-pruned model (any MLP width, as FLAP allocates), counted as `estimate.weight_bytes`
    counts them; `estimate.weight_bytes` itself needs a plan with widths in multiples of 64."""
    d = E._dims(config)
    values = sum(E._layer_counts(d, P.LayerPlan(mlp_keep=int(m), groups_keep=int(g), dropped=False,
                                                wanda_sparsity=0.0))[0]
                 for g, m in zip(groups_kept, neurons_kept, strict=True))
    values += d.vocab * d.hidden + d.hidden
    if not d.tied:
        values += d.vocab * d.hidden
    return values * dtype_bytes


def flap_group_weight(config) -> float:
    """`flap.group_weight` from the config alone."""
    c = P._cfg(config)
    head_dim = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
    q_heads = c["num_attention_heads"] // c["num_key_value_heads"]
    return (2 * q_heads + 2) * head_dim / 3


def flap_ratio_for_limit(attn, mlp, config, limit_bytes: int, lo: float = 0.0, hi: float = 0.5,
                         tol: float = 1e-4) -> tuple[float, int]:
    """Smallest FLAP ratio (share of block weights removed) whose model fits within `limit_bytes`, by bisection on
    the AL-AM allocation (no GPU). Returns (ratio, weight bytes)."""
    from pruneagent import flap

    def size(r: float) -> int:
        a, m = flap.allocate(attn, mlp, r, flap_group_weight(config))
        return flap_weight_bytes(config, a.sum(dim=1).tolist(), m.sum(dim=1).tolist())

    if size(hi) > limit_bytes:
        raise ValueError(f"FLAP at ratio {hi} does not fit within {limit_bytes} bytes")
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if size(mid) <= limit_bytes else (mid, hi)
    return round(hi, 6), size(hi)


def measure_flap(net, scores, config, limit_bytes: int, ids, seqlen: int, device: str, n_windows: int) -> dict:
    """Baseline: FLAP cut to the memory limit: the smallest ratio that fits (`flap_ratio_for_limit`), applied to a
    copy of the unpruned `net` with the cached FLAP statistics, and its subset perplexity."""
    import copy

    from pruneagent import evaluate as ev
    from pruneagent import flap

    ratio, est = flap_ratio_for_limit(scores.attn, scores.mlp, config, limit_bytes)
    start = time.perf_counter()
    pruned = copy.deepcopy(net)
    keep = flap.prune_flap_from_stats(pruned, scores.stats, ratio)
    stored = sum(p.numel() for p in pruned.parameters()) * next(pruned.parameters()).element_size()
    r = ev.subset_perplexity(pruned, ids, n_windows, seqlen, device)
    out = {"method": "FLAP (WIFV, AL-AM, bias compensation) from cached statistics", "ratio": ratio,
           "weight_bytes": est, "weight_bytes_stored": stored,
           "weight_bytes_removed": E.weight_bytes(config, P.dense_plan(config)) - est,
           "within_limit": est <= limit_bytes, "groups_kept": keep["attn_keep"].sum(dim=1).tolist(),
           "neurons_kept": keep["mlp_keep"].sum(dim=1).tolist(), "ppl": r.ppl, "n_windows": r.n_windows,
           "eval_seconds": round(r.seconds, 1), "total_seconds": round(time.perf_counter() - start, 1)}
    del pruned
    return out


def _vs(ppl: float | None, ref: float | None) -> str:
    return f"{ppl / ref - 1:+.0%}" if ppl and ref else "-"


def previous_numbers(record: dict, source: str, calibration: str) -> dict:
    """The perplexities of an earlier run of the same round, to show next to a re-measurement."""
    return {"source": source, "calibration": calibration,
            "flap_ppl": (record.get("baseline_flap") or {}).get("ppl"),
            "candidate_ppl": [v.get("measured", {}).get("ppl") for v in record.get("validation", [])]}


def _previous_table(record: dict, prev: dict) -> list[str]:
    def f(x):
        return f"{x:.2f}" if x else "-"

    now = record.get("calibration", {}).get("entry", "this run's calibration")
    now = "C4, 128 x 2,048 tokens" if now.startswith("c4_calib__") else now
    rows = [("Baseline: FLAP", prev.get("flap_ppl"), (record.get("baseline_flap") or {}).get("ppl"))]
    rows += [(f"Candidate {v['index'] + 1}", old, v.get("measured", {}).get("ppl"))
             for v, old in zip(record["validation"], prev.get("candidate_ppl", []), strict=False)]
    out = [(f"Compared with the earlier run ({prev['source']}; FLAP scores from {prev['calibration']}), "
            f"perplexity on the same 32 windows:"), "",
           f"| Model | Before ({prev['calibration']}) | Now ({now}) | Change |", "|---|---|---|---|"]
    out += [f"| {name} | {f(old)} | {f(new)} | {_vs(new, old)} |" for name, old, new in rows]
    return out + [""]


def summary_markdown(record: dict) -> str:
    m = record["memory"]
    ref = record.get("reference_ppl_32")
    fl = record.get("baseline_flap") or {}
    base = fl.get("ppl")
    lines = [
        f"# {record.get('title', 'Agent round 1 (task 1.11)')}",
        "",
        f"Model {record['model']}; agent: Codex CLI ({record['codex'].get('model') or 'default model'}). "
        f"Prompt: {record['prompt_chars']:,} characters, {record['prompt_words']:,} words"
        + (f", about {record['prompt_tokens_est']:,} tokens" if record.get("prompt_tokens_est") else "")
        + f". Codex reported {record['codex'].get('tokens_used') or 'unknown'} tokens in total for the call.",
        "",
        (f"Memory limit: {_mb(m['limit_bytes'])} of weights, {_mb(m['dense_bytes'] - m['limit_bytes'])} "
         f"({m['cut']:.0%}) less than the unpruned {_mb(m['dense_bytes'])}."),
        "",
        "Perplexity on the same 32 validation windows (lower is better):",
        f"- unpruned model: {record.get('dense_ppl_32', float('nan')):.2f}",
    ]
    if base:
        lines.append(f"- Baseline: FLAP cut to the same limit (ratio {fl['ratio']}, {_mb(fl['weight_bytes_removed'])} "
                     f"removed): {base:.2f}")
    if ref:
        lines.append(f"- Wanda-50% (secondary reference; frees no memory): {ref:.2f}")
    lines += [
        "",
        "| # | Valid | Choices | Perplexity | vs Baseline: FLAP | vs Wanda-50% | Memory removed | Within limit |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c, v in zip(record["candidates"], record["validation"], strict=True):
        ppl = v.get("measured", {}).get("ppl")
        removed = m["dense_bytes"] - v["weight_bytes"] if v.get("weight_bytes") else None
        steps = ", ".join(f"{n} x {s}" for s, n in v.get("steps", {}).items()) or f"{len(c['choices'] or [])} choices"
        lines.append(
            f"| {v['index'] + 1} | {'yes' if v['valid'] else 'no: ' + '; '.join(v['reasons'])} | {steps} | "
            f"{f'{ppl:.2f}' if ppl else '-'} | {_vs(ppl, base)} | {_vs(ppl, ref)} | "
            f"{_mb(removed) if removed is not None else '-'} | "
            f"{('yes' if v['weight_bytes'] <= m['limit_bytes'] else 'no') if removed is not None else '-'} |")
    lines.append("")
    if prev := record.get("previous"):
        lines += _previous_table(record, prev)
    for c, v in zip(record["candidates"], record["validation"], strict=True):
        lines += [f"**Candidate {v['index'] + 1}.** {c['rationale']}", ""]
        by: dict[str, list] = {}
        for ch in c["choices"] or []:
            if isinstance(ch, dict):
                by.setdefault(str(ch.get("strategy")), []).append(ch.get("layer"))
        lines += ["Choices (layers; a layer listed twice takes two steps): " + "; ".join(
            f"{s} on {sorted(ls, key=lambda x: (type(x).__name__, x if isinstance(x, int) else str(x)))}" for s, ls in by.items()), ""]
    return "\n".join(lines) + "\n"
