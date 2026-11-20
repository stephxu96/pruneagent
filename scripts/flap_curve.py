"""Baseline: FLAP perplexity across memory cuts on one model, to see where it breaks down.

    scripts/mac_gpu.sh uv run python scripts/flap_curve.py runs/<run-id> 0.10 0.15 0.20 0.25 0.30

Uses the run's model, C4 scores and gate windows (as scripts/heldout_check.py). For each cut of total weight memory,
FLAP is pruned to that limit and measured on the validation windows the search never used and on the full test split.
Writes results/flap_curve/<model>.json after each cut.
"""

import json
import sqlite3
import sys
import time
from pathlib import Path

from pruneagent import baselines
from pruneagent import data as d
from pruneagent import evaluate as ev
from pruneagent.cli import _scores
from pruneagent.realrun import Setup
from pruneagent.schema import RunConfig


def main() -> None:
    run_dir, cuts = Path(sys.argv[1]), [float(c) for c in sys.argv[2:]]
    setup = Setup.load(run_dir)
    con = sqlite3.connect(run_dir / "run.sqlite")
    run_cfg = RunConfig.model_validate_json(con.execute("select config from run").fetchone()[0])
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
    dense_bytes = sum(p.numel() * p.element_size() for p in dense.parameters())

    out_path = Path("results/flap_curve") / f"{run_cfg.model_id.split('/')[-1].lower()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"model": run_cfg.model_id, "dense_bytes": dense_bytes, "calibration": "C4 128 x 2048",
              "unseen_val_windows": len(unseen), "results": {}}
    for cut in cuts:
        t = time.time()
        limit = int(dense_bytes * (1 - cut))
        model, out = baselines.flap_at_memory_limit(dense, scores.stats, limit)
        v = ev.perplexity(model, val, seqlen, device, window_idx=unseen)
        te = ev.perplexity(model, test, seqlen, device)
        record["results"][f"{cut:.2f}"] = {"memory_cut": cut, "limit_bytes": limit, "weight_bytes": out.weight_bytes,
                                           "val_unseen_ppl": v.ppl, "test_ppl": te.ppl,
                                           "seconds": round(time.time() - t, 1)}
        print(f"cut {cut:.0%}: {out.weight_bytes / 1e6:,.1f} MB  unseen validation {v.ppl:.2f}  test {te.ppl:.2f}",
              flush=True)
        out_path.write_text(json.dumps(record, indent=2) + "\n")
        del model
        ev.free_memory(device)


if __name__ == "__main__":
    main()
