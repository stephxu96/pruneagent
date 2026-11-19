"""Tuning score of a finished run (task 3.1): the run's final model on the validation windows the search never used.

    scripts/mac_gpu.sh uv run python scripts/tuning_check.py runs/<run-id>

Tuning decisions use only this score. The WikiText-2 test split is not touched, so the final comparison (4.4)
stays clean (prompt_optimizer_plan.md, "no tuning to the test"). FLAP's score on the same windows at the same
memory limit comes from `results/flap_curve/` when the cut matches (same data, same code), so it is not recomputed.
Writes <run-dir>/tuning.json.
"""

import json
import sqlite3
import sys
import time
from pathlib import Path

from pruneagent import data as d
from pruneagent import evaluate as ev
from pruneagent import pruner, runstore
from pruneagent.cli import _scores
from pruneagent.realrun import Setup
from pruneagent.schema import RunConfig


def main() -> None:
    run_dir = Path(sys.argv[1])
    store = runstore.RunStore(run_dir)
    setup = Setup.load(run_dir)
    con = sqlite3.connect(run_dir / "run.sqlite")
    run_cfg = RunConfig.model_validate_json(con.execute("select config from run").fetchone()[0])
    final = store.accepted()[-1]
    data_dir = Path(setup.data_dir) if setup.data_dir else None
    dense = ev.load_model(run_cfg.model_id, setup.revision, run_cfg.dtype, setup.device)
    scores = _scores(dense, run_cfg.model_id, setup.revision or "", setup.device, run_cfg.dtype, data_dir,
                     Path(setup.scores), compute=False)
    seqlen = run_cfg.calibration.seqlen
    val = d.load_tokens(d.wikitext_name("validation", run_cfg.model_id), data_dir).reshape(-1)
    n_val = ev.windows(val, seqlen).shape[0]
    gate = set(ev.subset_indices(n_val, run_cfg.gate_windows))
    unseen = [i for i in range(n_val) if i not in gate]
    t = time.time()
    v = ev.perplexity(pruner.prune(dense, final.plan, scores), val, seqlen, setup.device, window_idx=unseen)
    record = {"run": run_dir.name, "instructions": setup.prompt_builder, "memory_limit_bytes": run_cfg.limit_bytes,
              "weight_bytes": final.weight_bytes, "unseen_val_windows": len(unseen), "val_unseen_ppl": v.ppl,
              "rounds": store.info().round, "seconds": round(time.time() - t, 1)}
    (run_dir / "tuning.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"{run_dir.name}: {final.weight_bytes / 1e6:,.1f} MB  unseen validation {v.ppl:.2f}", flush=True)


if __name__ == "__main__":
    main()
