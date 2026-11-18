"""Check our uniform Wanda-sp against the reference code, adapted to Qwen3 (task 1.3).

    scripts/mac_gpu.sh uv run python scripts/check_wanda_sp_reference.py --ratio 0.2

`WrappedGPT` and the scoring and threshold lines of `prune_wanda_sp` are copied below from
github.com/CASIA-IVA-Lab/FLAP commit 3bb57db (lib/layerwrapper.py, lib/prune.py; Apache-2.0). Two constants are
adapted to Qwen3: 128 channels per head -> 256 per key/value group, and num_heads -> key/value groups. The
reference's `compress` is replaced by our removal functions (no bias, as the reference calls it). The reference
loop runs on one copy of the model, ours on another; the script reports how many groups and neurons the two keep
differently, block by block, and writes a JSON to results/wanda_sp/.

Calibration: Wanda-sp's published setting (128 x 128 tokens of C4, `d.wanda_sp_reference_calib_name`). Compared
or benchmarked Wanda-sp results use the shared C4 set (128 x 2,048, `d.wanda_sp_calib_name`).
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from pruneagent import data as d
from pruneagent import evaluate as ev
from pruneagent import flap, wanda_sp
from pruneagent.wanda import _block_inputs, _run_block


class WrappedGPT:
    """FLAP lib/layerwrapper.py `WrappedGPT`, unchanged."""

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples


@torch.no_grad()
def reference(model, calib, ratio, device):
    """FLAP lib/prune.py `prune_wanda_sp`, per-block body; Qwen3 constants adapted, `compress` -> our removal."""
    n_groups, per_group = flap.group_shape(model)
    inps, kwargs = _block_inputs(model, calib, device)
    layers = model.model.layers
    attn_masks, mlp_masks = [], []
    for i in range(len(layers)):
        layer = layers[i]
        subset = {"self_attn.o_proj": layer.self_attn.o_proj, "mlp.down_proj": layer.mlp.down_proj}
        wrapped_layers = {name: WrappedGPT(subset[name]) for name in subset}

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)  # noqa: B023 - reference, used within the loop

            return tmp

        handles = [subset[name].register_forward_hook(add_batch(name)) for name in wrapped_layers]
        for h in inps:
            _run_block(layer, h, kwargs)
        for h in handles:
            h.remove()

        for name in subset:  # noqa: PLC0206 - reference line
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            if name == "self_attn.o_proj":
                W_metric = W_metric.mean(axis=0).reshape(-1, per_group).sum(dim=1)  # importance score of each group
                thresh = torch.sort(W_metric)[0][int(ratio * n_groups)]
                W_mask = W_metric >= thresh
                flap.remove_attention_groups(layer, W_mask, None)
                attn_masks.append(W_mask.cpu())
            else:
                W_metric = W_metric.mean(axis=0)
                thresh = torch.sort(W_metric)[0][int(W_metric.numel() * ratio)]
                W_mask = W_metric >= thresh
                flap.remove_mlp_neurons(layer, W_mask, None)
                mlp_masks.append(W_mask.cpu())

        inps = [_run_block(layer, h, kwargs) for h in inps]
    return torch.stack(attn_masks), torch.stack(mlp_masks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--ratio", type=float, default=0.2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--results", type=Path, default=Path("results/wanda_sp"))
    args = ap.parse_args()
    device = args.device or ev.pick_device()
    manifest = d.read_manifest()
    entry = manifest["entries"][d.wanda_sp_reference_calib_name(args.model)]
    calib = d.load_tokens(d.wanda_sp_reference_calib_name(args.model))

    model = ev.load_model(args.model, entry["tokenizer"]["revision"], "float16", device)
    ref_attn, ref_mlp = reference(model, calib, args.ratio, device)
    del model
    model = ev.load_model(args.model, entry["tokenizer"]["revision"], "float16", device)
    ours = wanda_sp.prune_wanda_sp(model, calib, args.ratio, device)
    our_attn, our_mlp = ours["attn_keep"], ours["mlp_keep"]

    result = {
        "model": args.model, "ratio": args.ratio, "device": device, "dtype": "float16",
        "calibration": {"entry": d.wanda_sp_reference_calib_name(args.model), "sha256": entry["sha256"],
                        "shape": entry["shape"]},
        "groups_total": ref_attn.numel(),
        "groups_kept_ref": int(ref_attn.sum()),
        "groups_kept_ours": int(our_attn.sum()),
        "groups_differing": int((ref_attn != our_attn).sum()),
        "neurons_total": ref_mlp.numel(),
        "neurons_kept_ref": int(ref_mlp.sum()),
        "neurons_kept_ours": int(our_mlp.sum()),
        "neurons_differing": int((ref_mlp != our_mlp).sum()),
        "groups_differing_per_layer": (ref_attn != our_attn).sum(dim=1).tolist(),
        "neurons_differing_per_layer": (ref_mlp != our_mlp).sum(dim=1).tolist(),
    }
    print(json.dumps(result, indent=2))
    args.results.mkdir(parents=True, exist_ok=True)
    out = args.results / f"{d.slug(args.model)}__wandasp{round(args.ratio * 100)}__reference_check__{device}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
