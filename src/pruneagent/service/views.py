"""Read-only views of runs for the web application (ADR-008).

Everything here reads files the tools already write (the run store, `decisions.jsonl`, `heldout.json`,
`accuracy.json`, the FLAP curve) and reshapes them as JSON for the browser. No pruning, evaluation or decision
logic: if a number is not in a tool's output, it is not shown.
"""

from __future__ import annotations

import functools
import json
import logging
import sqlite3
from pathlib import Path

from pruneagent import estimate as E
from pruneagent import mc
from pruneagent import runstore as R

MB = 1e6
log = logging.getLogger(__name__)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _layers(plan: dict) -> list[dict]:
    plan = {k: (v.model_dump() if hasattr(v, "model_dump") else v) for k, v in plan.items()}
    return [
        {"mlp": int(v["mlp_keep"]), "groups": int(v["groups_keep"]), "dropped": bool(v["dropped"])}
        for _, v in sorted(plan.items(), key=lambda kv: int(kv[0]))
    ]


def _summary(store: R.RunStore) -> dict:
    info = store.info()
    cfg = store.run_config()
    kept = store.accepted()
    dense, last = (kept[0], kept[-1]) if kept else (None, None)
    limit = cfg.device.memory_limit_gb * 1e9
    return {
        "id": info.run_id,
        "model": cfg.model_id,
        "state": info.state,
        "reason": info.reason,
        "round": info.round,
        "kept_steps": max(len(kept) - 1, 0),
        "created": info.created,
        "updated": info.updated,
        "limit_mb": limit / MB,
        "original_mb": dense.weight_bytes / MB if dense else None,
        "cut": 1 - limit / dense.weight_bytes if dense else None,
        "now_mb": last.weight_bytes / MB if last else None,
        "original_error": dense.val_ppl if dense else None,
        "now_error": last.val_ppl if last else None,
        "has_heldout": (store.dir / "heldout.json").exists(),
        "settings": {
            "device": cfg.device.name,
            "agent": " ".join(x for x in (cfg.agent_backend.capitalize(), cfg.agent_model, cfg.agent_reasoning_effort) if x),
            "ideas_per_round": cfg.rules.max_candidates,
            "max_drop_per_step": cfg.rules.step_limit,
            "max_rounds": cfg.rules.max_rounds,
            "max_ai_calls": cfg.max_agent_calls,
            "max_layers_removed": cfg.rules.max_drops_per_run,
            "min_layer_width": cfg.rules.min_mlp_fraction,
            "sample_text": f"{cfg.calibration.dataset.upper()}, {cfg.calibration.n_samples} passages",
            "test_passages": cfg.gate_windows,
        },
        "has_accuracy": (store.dir / "accuracy.json").exists(),
        "has_model": (store.dir / "model.zip").exists(),
        "model_mb": (store.dir / "model.zip").stat().st_size / MB if (store.dir / "model.zip").exists() else None,
    }


def list_runs(root: Path) -> list[dict]:
    out = []
    for info in R.list_runs(root):
        try:
            out.append(_summary(R.RunStore.open(info.run_id, root)))
        except (OSError, ValueError, sqlite3.Error) as e:  # a run being created, or a broken folder
            log.warning("skipping run %s: %s", info.run_id, e)
    return out


def _candidate(c: dict, check: dict | None) -> dict:
    ok = next((x for x in (check or {}).get("candidates", []) if x.get("index") == c.get("candidate_id")), None)
    return {
        "id": c.get("candidate_id"),
        "changes": c.get("choices", []),
        "why": c.get("rationale", ""),
        "error": c.get("val_ppl"),
        "rise": c.get("step_rise"),
        "freed_mb": (c.get("freed_bytes") or 0) / MB,
        "size_mb": (c.get("estimate") or {}).get("weight_bytes", 0) / MB or None,
        "refused": None if ok is None or ok.get("ok", True) else ok.get("reasons", []),
    }


def _round(r: dict) -> dict:
    check = r.get("check") or {}
    chosen = (r.get("gate") or {}).get("chosen_candidate")
    return {
        "round": r["round"],
        "outcome": r.get("outcome"),
        "why": r.get("rationale", ""),
        "chosen": chosen if r.get("outcome") == "ACCEPT" else None,
        "candidates": [_candidate(c, check) for c in r.get("candidates", [])],
        "refused_reasons": check.get("reasons", []) if not check.get("ok", True) else [],
        "agent_seconds": r.get("llm_seconds", 0.0),
        "check_seconds": r.get("gpu_seconds", 0.0),
    }


@functools.cache
def _config(model_id: str, revision: str | None):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_id, revision=revision)


def _mac_mb(cfg, plan) -> float | None:
    """Memory access cost (MAC) per generated token: bytes read from memory for one word, i.e. every kept weight
    plus the key/value cache, at the estimator's default context (`estimate.DEFAULT_CONTEXT`)."""
    try:
        return E.decode_bytes_per_token(cfg, plan, strict=False) / MB
    except (ValueError, KeyError) as e:
        log.warning("no MAC figure: %s", e)
        return None


def run_detail(root: Path, run_id: str) -> dict:
    store = R.RunStore.open(run_id, root)
    run_cfg = store.run_config()
    setup = _read_json(store.dir / "setup.json") or {}
    try:
        cfg = _config(run_cfg.model_id, setup.get("revision"))
    except OSError:  # offline and not cached: no MAC figures
        cfg = None
    kept = store.accepted()
    first = _layers(kept[0].plan) if kept else []
    state = store.load_checkpoint() or {}
    base = state.get("baseline")
    rounds = []
    if store.log_path.exists():
        rounds = [_round(json.loads(line)) for line in store.log_path.read_text().splitlines() if line.strip()]
    return {
        **_summary(store),
        "mac_context": E.DEFAULT_CONTEXT,
        # the device's memory speed: best-case speed = this / MAC (the estimator's bandwidth bound)
        "bandwidth_mb_s": run_cfg.device.bandwidth_gb_s * 1000 if run_cfg.device.bandwidth_gb_s else None,
        "shape": {
            "layers": len(first),
            "mlp_full": max((x["mlp"] for x in first), default=0),
            "groups_full": max((x["groups"] for x in first), default=0),
        },
        "flap": None if base is None else {
            "size_mb": base.weight_bytes / MB,
            "error": base.val_ppl_subset,
            "mac_mb": _mac_mb(cfg, base.plan) if cfg else None,
            "layers": _layers(base.plan),
        },
        "steps": [
            {"round": a.round, "error": a.val_ppl, "size_mb": a.weight_bytes / MB,
             "mac_mb": _mac_mb(cfg, a.plan) if cfg else None, "layers": _layers(a.plan)}
            for a in kept
        ],
        "rounds": rounds,
        "time_left": R.time_left(store, R.past_rates(root, run_cfg.model_id)) if store.info().state == "running" else None,
    }


def run_results(root: Path, run_id: str) -> dict:
    """Checks made after the run on text it never saw (`heldout.json`) and on question sets (`accuracy.json`)."""
    d = R.RunStore.open(run_id, root).dir
    held = _read_json(d / "heldout.json")
    acc = _read_json(d / "accuracy.json")
    out: dict = {"heldout": None, "accuracy": None}
    if held:
        out["heldout"] = {name: {"size_mb": r["weight_bytes"] / MB, "test_error": r["test_ppl"],
                                 "unseen_error": r["val_unseen_ppl"]}
                          for name, r in held["results"].items()}
    if acc:
        out["accuracy"] = {}
        for name, r in acc["results"].items():
            tasks = {t: {"acc": v["acc"], "acc_stderr": v.get("acc_stderr")} for t, v in r["tasks"].items()}
            out["accuracy"][name] = {"all_questions": mc.weighted_accuracy(tasks), "sets": tasks}
    return out


def compare(root: Path, curve_file: Path) -> dict:
    """Every finished run with a held-out check, next to FLAP's measured curve over memory cuts."""
    runs = []
    for s in list_runs(root):
        if not s["has_heldout"]:
            continue
        held = run_results(root, s["id"])["heldout"]
        runs.append({"id": s["id"], "model": s["model"], "cut": s["cut"],
                     "agent": held.get("agent"), "flap": held.get("Baseline: FLAP"),
                     "original": held.get("unpruned")})
    curve = _read_json(curve_file)
    flap_curve = [] if not curve else sorted(
        ({"cut": v["memory_cut"], "test_error": v["test_ppl"]} for v in curve["results"].values()),
        key=lambda p: p["cut"])
    return {"runs": sorted(runs, key=lambda r: r["cut"] or 0), "flap_curve": flap_curve}
