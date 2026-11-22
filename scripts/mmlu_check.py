"""Knowledge check (MMLU) of a finished run: the original model, the run's final plan and Baseline: FLAP at the
same size (the memory limit, or the agent's own size if the run ended over it).

    scripts/mac_gpu.sh uv run python scripts/mmlu_check.py runs/<run-id>

About 14,000 four-answer questions from 57 subjects (pruneagent.mc.evaluate_mmlu), no examples in the prompt.
Not part of every run: started on its own for the final run. Writes <run-dir>/mmlu.json after each model; a
restart skips models already scored.
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
    scores = _scores(dense, run_cfg.model_id, setup.revision or "", device, run_cfg.dtype, data_dir,
                     Path(setup.scores), compute=False)

    out_path = run_dir / "mmlu.json"
    target = baselines.flap_target_bytes(run_cfg.limit_bytes, final.weight_bytes)  # same size as the agent's
    record = json.loads(out_path.read_text()) if out_path.exists() else {
        "run": run_dir.name, "memory_limit_bytes": run_cfg.limit_bytes, "device": device, "results": {}}
    if record.get("flap_target_bytes", target) != target:  # FLAP scored at another size
        record["results"].pop("Baseline: FLAP", None)
    record["flap_target_bytes"] = target

    def flap():
        model, out = baselines.flap_at_memory_limit(dense, scores.stats, target)
        return model, out.weight_bytes

    builders = {
        "unpruned": lambda: (dense, sum(p.numel() * p.element_size() for p in dense.parameters())),
        "agent": lambda: (pruner.prune(dense, final.plan, scores), final.weight_bytes),
        "Baseline: FLAP": flap,
    }
    for name, build in builders.items():
        if name in record["results"]:
            continue  # resume
        t = time.time()
        model, nbytes = build()
        summary = mc.summarize_mmlu(mc.evaluate_mmlu(model, tok, device, batch_size=8))
        record["results"][name] = {"weight_bytes": nbytes, **summary, "seconds": round(time.time() - t, 1)}
        o = summary["overall"]
        print(f"{name}: {nbytes / 1e6:,.1f} MB  MMLU {o['acc']:.4f} on {o['questions']} questions "
              f"({time.time() - t:.0f} s)", flush=True)
        out_path.write_text(json.dumps(record, indent=2) + "\n")
        if model is not dense:
            del model
            ev.free_memory(device)


if __name__ == "__main__":
    main()
