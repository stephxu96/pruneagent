"""Does Codex's reasoning effort change the quality of its pruning proposals? (effort check)

    uv run python scripts/effort_check.py call       # 3 Codex calls: low, medium, high (no GPU)
    scripts/mac_gpu.sh uv run python scripts/effort_check.py measure --scores <path> --data-dir <path>

Everything is held fixed except the reasoning effort: the saved round-1 prompt (task 1.11, second attempt,
results/agent_round/round1_prompt.txt) verbatim, the trimmed Codex client (`agent_client.CodexBackend`, model
gpt-5.6-terra) with one answer schema, no retry (exactly one call per effort). Each reply is checked by
`validator.check_candidates` against the 10% memory limit, and each valid candidate is built from the
C4-calibrated cached scores and measured on the same 32 validation windows as round 1.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from pruneagent import agent_client as C
from pruneagent import plan as P
from pruneagent import validator as V

MODEL = "Qwen/Qwen3-0.6B-Base"
EFFORTS = ("low", "medium", "high")
PROMPT = Path("results/agent_round/round1_prompt.txt")
OUT = Path("results/effort_check/effort_check.json")
REFERENCE = Path("results/agent_round/round1_c4.json")  # Baseline: FLAP and unpruned on the same windows

SCHEMA = {
    "type": "object",
    "properties": {"candidates": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "choices": {"type": "array", "items": {
                "type": "object",
                "properties": {"layer": {"type": "integer"}, "strategy": {"type": "string", "enum": list(P.STRATEGIES)}},
                "required": ["layer", "strategy"], "additionalProperties": False}},
            "rationale": {"type": "string"}},
        "required": ["choices", "rationale"], "additionalProperties": False}}},
    "required": ["candidates"], "additionalProperties": False,
}


def _config(data_dir: Path | None):
    from transformers import AutoConfig

    from pruneagent import data as d

    manifest = d.read_manifest(data_dir)
    revision = manifest["entries"][d.calib_name(MODEL)]["tokenizer"]["revision"]
    return AutoConfig.from_pretrained(MODEL, revision=revision), revision


def call(args) -> None:
    prompt = PROMPT.read_text().removesuffix("\n")
    ref = json.loads(REFERENCE.read_text())
    assert ref["prompt"] == prompt, "saved prompt differs from the one round 1 measured"
    limit = ref["memory"]["limit_bytes"]
    config, _ = _config(args.data_dir)
    backends = {e: C.CodexBackend(reasoning_effort=e) for e in EFFORTS}
    request = C.AgentRequest(prompt, SCHEMA, tag="effort check")
    keys = {e: C.cache_key(request, b) for e, b in backends.items()}
    assert len(set(keys.values())) == len(EFFORTS), "cache keys do not differ by effort"
    cache = C.Cache(args.cache)
    for e, k in keys.items():
        assert cache.get(k) is None, f"cache already holds an answer for effort {e}"
    record = {"prompt_source": str(PROMPT), "prompt": prompt, "schema": SCHEMA, "memory": ref["memory"],
              "cache_keys": {e: k[:16] for e, k in keys.items()}, "runs": {}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for e, backend in backends.items():
        client = C.AgentClient(backend, cache=cache, ledger=C.Ledger(args.ledger), run_id="effort-check",
                               retries=0, timeout=args.timeout)
        print(f"[{time.strftime('%H:%M:%S')}] Codex, effort {e} ...", flush=True)
        run = {"effort": e, "identity": backend.identity(), "command": backend.command(Path("schema.json"))}
        try:
            reply = client.propose(request)
        except C.AgentError as err:
            run |= {"error": f"{type(err).__name__}: {err}"}
            print(f"  failed: {run['error']}", flush=True)
            record["runs"][e] = run
            OUT.write_text(json.dumps(record, indent=2) + "\n")
            continue
        res = V.check_candidates(reply.data.get("candidates"), V.Context(config, {}, limit))
        run |= {"seconds": reply.seconds, "cached": reply.cached, "usage": asdict(reply.usage), "raw_reply": reply.text,
                "candidates": reply.data.get("candidates"), "reply_valid": res.ok, "reasons": res.reasons,
                "validation": [{"index": c.index, "valid": c.ok, "reasons": c.reasons,
                                "plan": P.to_json(c.plan) if c.ok else None, "weight_bytes": c.weight_bytes}
                               for c in res.candidates]}
        for c, v in zip(run["candidates"], run["validation"], strict=False):
            if v["valid"]:
                v["steps"] = dict(Counter(ch["strategy"] for ch in c["choices"]))
        print(f"  {reply.seconds} s, usage {run['usage']}, {len(run['candidates'] or [])} candidates, "
              f"valid {[v['valid'] for v in run['validation']]}", flush=True)
        record["runs"][e] = run
        OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved {OUT}")


def measure(args) -> None:
    from pruneagent import data as d
    from pruneagent import evaluate as ev
    from pruneagent import pruner

    record = json.loads(OUT.read_text())
    _, revision = _config(args.data_dir)
    device = args.device or ev.pick_device()
    scores = pruner.load_scores(args.scores)
    ref = json.loads(REFERENCE.read_text())
    assert scores.meta.get("calib") == ref["scores"]["calib"], "scores are not the C4-calibrated ones round 1 used"
    net = ev.load_model(MODEL, revision, args.dtype, device)
    ids = d.load_tokens(d.wikitext_name("validation", MODEL), args.data_dir).reshape(-1)
    dense = record["memory"]["dense_bytes"]
    limit = record["memory"]["limit_bytes"]
    record["measure"] = {"device": device, "dtype": args.dtype, "scores": scores.meta, "scores_path": str(args.scores),
                         "n_windows": ev.GATE_WINDOWS, "seqlen": 2048,
                         "reference": {"source": str(REFERENCE), "dense_ppl_32": ref["dense_ppl_32"],
                                       "flap_ppl_32": ref["baseline_flap"]["ppl"]}}
    for e, run in record["runs"].items():
        for v in run.get("validation", []):
            if not v["valid"]:
                continue
            start = time.perf_counter()
            pruned = pruner.prune(net, v["plan"], scores)
            r = ev.subset_perplexity(pruned, ids, ev.GATE_WINDOWS, 2048, device)
            v["measured"] = {"ppl": r.ppl, "n_windows": r.n_windows, "eval_seconds": round(r.seconds, 1),
                             "total_seconds": round(time.perf_counter() - start, 1),
                             "weight_bytes_removed": dense - v["weight_bytes"],
                             "within_limit": v["weight_bytes"] <= limit}
            print(f"  {e} candidate {v['index'] + 1}: ppl {r.ppl:.4f} ({r.seconds:.0f} s)", flush=True)
            del pruned
            gc.collect()
            OUT.write_text(json.dumps(record, indent=2) + "\n")
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["call", "measure"])
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--cache", type=Path, default=Path("runs/effort_check/agent_cache"))
    ap.add_argument("--ledger", type=Path, default=Path("runs/effort_check/agent_usage.jsonl"))
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--scores", type=Path, default=Path("checkpoints/scores/qwen3-0.6b-base__mps__float16.safetensors"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args()
    call(args) if args.step == "call" else measure(args)


if __name__ == "__main__":
    main()
