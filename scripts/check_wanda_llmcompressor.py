"""Independent check of our Wanda against llm-compressor's WandaPruningModifier (task 1.1).

Runs outside the project environment so llm-compressor's own pins cannot touch uv.lock:

    uv run --no-project --python 3.12 --with "llmcompressor==0.8.1" \\
        python scripts/check_wanda_llmcompressor.py [--sparsity 0.5] [--device cuda]

Uses the same cached C4 calibration tokens (data_cache/, checked against the manifest hash), prunes the same
seven linear layers per block, and scores WikiText-2 with the project's perplexity code. Writes
results/wanda/<model>__wanda<NN>__llmcompressor__<device>.json. llm-compressor 0.8.1 is the last release
with Wanda (removed by 0.15).
"""

import argparse
import hashlib
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import Dataset
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pruneagent import evaluate as ev  # torch-only module, importable outside the project env

TARGETS = ["re:.*self_attn\\.(q|k|v|o)_proj$", "re:.*mlp\\.(gate|up|down)_proj$"]


def load_cached(manifest: dict, name: str) -> torch.Tensor:
    entry = manifest["entries"][name]
    path = ROOT / "data_cache" / entry["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        sys.exit(f"{path} does not match the manifest; run `uv run pruneagent data` first")
    return load_file(path)["input_ids"]


def main() -> None:
    from llmcompressor import oneshot
    from llmcompressor.modifiers.pruning import WandaPruningModifier

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--sparsity", type=float, default=0.5)
    ap.add_argument("--device", default=ev.pick_device())
    args = ap.parse_args()

    slug = args.model.split("/")[-1].lower()
    manifest = json.loads((ROOT / "data_cache" / "manifest.json").read_text())
    calib = load_cached(manifest, f"c4_calib__{slug}")
    revision = manifest["entries"][f"c4_calib__{slug}"]["tokenizer"]["revision"]

    model = AutoModelForCausalLM.from_pretrained(args.model, revision=revision, torch_dtype=torch.float16)
    model.to(args.device).eval()
    ds = Dataset.from_dict({"input_ids": calib.tolist(), "attention_mask": torch.ones_like(calib).tolist()})

    start = time.perf_counter()
    oneshot(
        model=model,
        dataset=ds,
        recipe=WandaPruningModifier(sparsity=args.sparsity, mask_structure="0:0", targets=TARGETS),
        num_calibration_samples=len(calib),
        max_seq_length=calib.shape[1],
        shuffle_calibration_samples=False,
    )
    prune_seconds = time.perf_counter() - start

    zeros = total = 0
    for name, p in model.named_parameters():
        if "layers." in name and name.endswith("proj.weight"):
            zeros += int((p == 0).sum())
            total += p.numel()
    record = {
        "method": "wanda (llm-compressor WandaPruningModifier)", "llmcompressor": version("llmcompressor"),
        "transformers": version("transformers"), "torch": version("torch"), "target_sparsity": args.sparsity,
        "achieved_linear_sparsity": zeros / total, "model": args.model, "revision": revision,
        "device": args.device, "dtype": "float16", "prune_seconds": round(prune_seconds, 1), "splits": {},
    }
    model.to(args.device)
    for split in ("validation", "test"):
        ids = load_cached(manifest, f"wikitext2_{split}__{slug}")
        r = ev.perplexity(model, ids, 2048, args.device)
        record["splits"][split] = {"ppl": r.ppl, "n_windows": r.n_windows, "seconds": round(r.seconds, 1)}
        print(f"{split:<10} ppl {r.ppl:.4f}", file=sys.stderr)

    out = ROOT / "results" / "wanda" / f"{slug}__wanda{round(args.sparsity * 100)}__llmcompressor__{args.device}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
