"""Best of N: run the agent N times at the same limit and keep the run whose final model scores best on the run's
own sample text (the error each run measured on its last kept step).

The pick never looks at the unseen validation windows or the test text, so the winner's later scores on those are
not inflated by the pick. Runs differ by chance (the same instructions gave 22.98 and 25.20 on unseen text at a 20%
cut), and keeping the better of two turns that spread into a gain.
"""

from __future__ import annotations

import json
from pathlib import Path

from pruneagent import runstore as R


class PickError(ValueError):
    pass


def _measure(cfg) -> tuple:
    # runs are comparable only when they were measured the same way, at the same limit
    return (cfg.model_id, cfg.limit_bytes, cfg.calibration.model_dump_json(), cfg.gate_windows)


def pick(root: Path | str, run_ids: list[str]) -> dict:
    """The best of the given finished runs. Raises PickError if a run is unfinished, missed the limit, or was
    measured differently from the others."""
    if len(run_ids) < 2:
        raise PickError("give at least two runs")
    rows, keys = [], set()
    for rid in run_ids:
        store = R.RunStore.open(rid, root)
        cfg, info, kept = store.run_config(), store.info(), store.accepted()
        if info.state != "finished" or len(kept) < 2:
            raise PickError(f"{rid} has not finished")
        last = kept[-1]
        if last.weight_bytes > cfg.limit_bytes:
            raise PickError(f"{rid} stopped above the limit ({last.weight_bytes / 1e6:,.1f} MB)")
        keys.add(_measure(cfg))
        rows.append({"run": rid, "sample_error": last.val_ppl, "size_mb": last.weight_bytes / 1e6,
                     "rounds": info.round})
    if len(keys) > 1:
        raise PickError("the runs differ in model, limit or sample text, so their errors are not comparable")
    rows.sort(key=lambda r: r["sample_error"])
    return {"winner": rows[0]["run"], "runs": rows,
            "chosen_by": "lowest error on the run's own sample text (unseen and test text not used)"}


def write(root: Path | str, run_ids: list[str], name: str) -> Path:
    out = Path(root) / "best_of" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pick(root, run_ids), indent=2))
    return out
