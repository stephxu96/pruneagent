"""Multiple-choice accuracy of a finished run's final plan against Baseline: FLAP at the same size (the memory
limit, or the model's own size if the run ended over it).

    scripts/mac_gpu.sh uv run python scripts/accuracy_check.py runs/<run-id>

Same models as scripts/heldout_check.py (the run's final plan, and FLAP cut to the run's memory limit with the same
C4 scores), scored on the seven question sets in pruneagent.mc. Writes <run-dir>/accuracy.json after each model.
"""

import json
import sqlite3
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer

from pruneagent import baselines, mc, pruner, runstore
from pruneagent import evaluate as ev
from pruneagent.cli import _scores
from pruneagent.realrun import Setup
from pruneagent.schema import RunConfig


def flap(dense, scores, limit_bytes):
    model, out = baselines.flap_at_memory_limit(dense, scores.stats, limit_bytes)
    return model, out.weight_bytes


def main() -> None:
    run_dir = Path(sys.argv[1])
    store = runstore.RunStore(run_dir)
    setup = Setup.load(run_dir)
    con = sqlite3.connect(run_dir / "run.sqlite")
    run_cfg = RunConfig.model_validate_json(con.execute("select config from run").fetchone()[0])
    final = store.accepted()[-1]

    data_dir = Path(setup.data_dir) if setup.data_dir else None
    device = setup.device
    dense = ev.load_model(run_cfg.model_id, setup.revision, run_cfg.dtype, device)
    tok = AutoTokenizer.from_pretrained(run_cfg.model_id, revision=setup.revision)
    scores = _scores(
        dense,
        run_cfg.model_id,
        setup.revision or "",
        device,
        run_cfg.dtype,
        data_dir,
        Path(setup.scores),
        compute=False,
    )

    out_path = run_dir / "accuracy.json"
    target = baselines.flap_target_bytes(run_cfg.limit_bytes, final.weight_bytes)  # same size as the agent's
    record = (
        json.loads(out_path.read_text())
        if out_path.exists()
        else {"run": run_dir.name, "memory_limit_bytes": run_cfg.limit_bytes, "device": device, "results": {}}
    )
    if record.get("flap_target_bytes", record["memory_limit_bytes"]) != target:  # FLAP scored at another size
        record["results"].pop("Baseline: FLAP", None)
    record["flap_target_bytes"] = target
    builders = {
        "agent": lambda: (pruner.prune(dense, final.plan, scores), final.weight_bytes),
        "Baseline: FLAP": lambda: flap(dense, scores, target),
    }
    for name, build in builders.items():
        if name in record["results"]:
            continue  # resume
        t = time.time()
        model, nbytes = build()
        summary = mc.summarize(mc.evaluate(model, tok, device, batch_size=8))
        record["results"][name] = {"weight_bytes": nbytes, **summary, "seconds": round(time.time() - t, 1)}
        print(
            f"{name}: {nbytes / 1e6:,.1f} MB  mean accuracy {summary['mean_acc']:.4f}  "
            f"({time.time() - t:.0f} s)",
            flush=True,
        )
        out_path.write_text(json.dumps(record, indent=2) + "\n")
        del model
        ev.free_memory(device)  # before the next model is built: otherwise both stay in memory


if __name__ == "__main__":
    main()
