"""Held-out check of a finished run against Baseline: FLAP at the same memory limit.

    scripts/mac_gpu.sh uv run python scripts/heldout_check.py runs/<run-id>

The search chooses every step on the 32 validation windows (the gate), so its score there is optimistic. This
measures, for the unpruned model, the run's final plan and FLAP cut to the run's memory limit (same C4 scores):
- the validation windows the gate never used, and
- the full WikiText-2 test split (never seen by the search).
Writes <run-dir>/heldout.json.
"""

import json
import sys
import time
from pathlib import Path

from pruneagent import baselines, pruner, runstore
from pruneagent import data as d
from pruneagent import evaluate as ev
from pruneagent.cli import _scores
from pruneagent.realrun import Setup
from pruneagent.schema import RunConfig


def main() -> None:
    run_dir = Path(sys.argv[1])
    store = runstore.RunStore(run_dir)
    setup = Setup.load(run_dir)
    import sqlite3

    con = sqlite3.connect(run_dir / "run.sqlite")
    run_cfg = RunConfig.model_validate_json(con.execute("select config from run").fetchone()[0])
    final = store.accepted()[-1]

    data_dir = Path(setup.data_dir) if setup.data_dir else None
    device = setup.device
    dense = ev.load_model(run_cfg.model_id, setup.revision, run_cfg.dtype, device)
    scores = _scores(dense, run_cfg.model_id, setup.revision or "", device, run_cfg.dtype, data_dir,
                     Path(setup.scores), compute=False)
    seqlen = run_cfg.calibration.seqlen
    val = d.load_tokens(d.wikitext_name("validation", run_cfg.model_id), data_dir).reshape(-1)
    test = d.load_tokens(d.wikitext_name("test", run_cfg.model_id), data_dir).reshape(-1)
    n_val = ev.windows(val, seqlen).shape[0]
    gate_idx = set(ev.subset_indices(n_val, run_cfg.gate_windows))
    unseen = [i for i in range(n_val) if i not in gate_idx]

    target = baselines.flap_target_bytes(run_cfg.limit_bytes, final.weight_bytes)  # same size as the agent's

    def flap_model():
        model, out = baselines.flap_at_memory_limit(dense, scores.stats, target)
        return model, out.weight_bytes

    # built one at a time, each freed before the next: at 1.7B, three models at once is ~10 GB
    builders = {
        "unpruned": lambda: (dense, sum(p.numel() * p.element_size() for p in dense.parameters())),
        "agent": lambda: (pruner.prune(dense, final.plan, scores), final.weight_bytes),
        "Baseline: FLAP": flap_model,
    }
    out_path = run_dir / "heldout.json"
    old = json.loads(out_path.read_text()) if out_path.exists() else {}
    record = {"run": run_dir.name, "memory_limit_bytes": run_cfg.limit_bytes, "flap_target_bytes": target,
              "unseen_val_windows": len(unseen), "results": {}}
    if old.get("unseen_val_windows") == len(unseen):  # resume: keep what is already measured the same way
        record["results"] = {k: v for k, v in old.get("results", {}).items()
                             if k != "Baseline: FLAP" or old.get("flap_target_bytes", old.get("memory_limit_bytes")) == target}
    for name, build in builders.items():
        if name in record["results"]:
            continue
        model, nbytes = build()
        t = time.time()
        v = ev.perplexity(model, val, seqlen, device, window_idx=unseen)
        te = ev.perplexity(model, test, seqlen, device)
        record["results"][name] = {"weight_bytes": nbytes, "val_unseen_ppl": v.ppl, "test_ppl": te.ppl,
                                   "seconds": round(time.time() - t, 1)}
        print(f"{name}: {nbytes / 1e6:,.1f} MB  unseen validation {v.ppl:.2f}  test {te.ppl:.2f}", flush=True)
        out_path.write_text(json.dumps(record, indent=2) + "\n")
        if model is not dense:
            del model
            ev.free_memory(device)


if __name__ == "__main__":
    main()
