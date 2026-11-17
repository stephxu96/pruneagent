"""Check our FLAP scores and allocation against the reference code, adapted to Qwen3 (task 1.2).

    uv run python scripts/check_flap_reference.py --ratio 0.2

Statistics come from the reference `BiasGPT` class (WIFV branch) and `metrics["WIFV"]`, copied below from
github.com/CASIA-IVA-Lab/FLAP commit 3bb57db (lib/layerwrapper.py, lib/prune.py; Apache-2.0); the AL-AM
allocation is the reference code with its two LLaMA constants replaced (128 channels per head -> 256 per Qwen3
key/value group; 512/3 neurons per head -> 256 per group). The calibration samples run through the unpruned model
block by block as in the reference. Prints how many groups and neurons the two versions keep differently.
"""

import argparse
import json

import torch

from pruneagent import data as d
from pruneagent import evaluate as ev
from pruneagent import flap
from pruneagent.wanda import _block_inputs, _run_block


class BiasGPT:
    """FLAP lib/layerwrapper.py `BiasGPT`, WIFV branch, unchanged apart from dropping the WIFN branch."""

    def __init__(self, layer, metric):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.in_dim = layer.weight.data.shape[1]
        self.type = metric
        self.nsamples = 0
        self.baseline_inp = torch.zeros((self.in_dim), device=self.dev)
        self.fluc_inp = torch.zeros((self.in_dim), device=self.dev)

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        old_baseline_inp = self.baseline_inp
        self.baseline_inp *= self.nsamples / (self.nsamples + batch_size)
        self.baseline_inp += torch.mean(inp, dim=1) / (self.nsamples + batch_size)
        if self.nsamples == 0:
            self.fluc_inp = 0
        else:
            self.fluc_inp *= (self.nsamples - 1) / (self.nsamples + batch_size - 1)
            self.fluc_inp += torch.sum(
                (inp - self.baseline_inp.unsqueeze(1)) * (inp - old_baseline_inp.unsqueeze(1)), dim=1
            ) / (self.nsamples + batch_size)
        self.nsamples += batch_size


metrics = {  # FLAP lib/prune.py, WIFV entry
    "WIFV": lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp
    * torch.sum(subset[name].weight.data.pow(2), dim=0),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--ratio", type=float, default=0.2)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ev.pick_device()
    manifest = d.read_manifest()
    revision = manifest["entries"][d.flap_calib_name(args.model)]["tokenizer"]["revision"]
    model = ev.load_model(args.model, revision, "float16", device)
    calib = d.load_tokens(d.flap_calib_name(args.model))
    _, per_group = flap.group_shape(model)
    layers = model.model.layers

    # ---- reference statistics and metric
    attn_metric_list, mlp_metric_list = [], []
    with torch.no_grad():
        hiddens, kwargs = _block_inputs(model, calib, device)
        for layer in layers:
            subset = {"self_attn.o_proj": layer.self_attn.o_proj, "mlp.down_proj": layer.mlp.down_proj}
            wrapped = {name: BiasGPT(subset[name], "WIFV") for name in subset}
            handles = [
                subset[name].register_forward_hook(lambda _m, inp, out, w=wrapped[name]: w.add_batch(inp[0].data, out.data))
                for name in subset
            ]
            hiddens = [_run_block(layer, h, kwargs) for h in hiddens]
            for h in handles:
                h.remove()
            attn_metric_list.append((metrics["WIFV"](wrapped, subset, "self_attn.o_proj") ** 2).float().cpu())
            mlp_metric_list.append(metrics["WIFV"](wrapped, subset, "mlp.down_proj").float().cpu())

    # ---- reference AL-AM allocation (constants adapted)
    standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)
    attn_metric = standarlization(torch.stack(attn_metric_list)).reshape(len(layers), -1, per_group).mean(dim=2)
    mlp_metric = standarlization(torch.stack(mlp_metric_list))
    prune_metric = torch.cat([attn_metric.view(-1), mlp_metric.view(-1)])
    sorted_prune, indices = torch.sort(prune_metric, descending=True)
    compression_weight = torch.ones_like(indices, dtype=torch.float)
    compression_weight[indices < attn_metric.numel()] = flap.group_weight(model)
    threshold = sorted_prune[
        torch.argmin(torch.abs(torch.cumsum(compression_weight, 0) - torch.sum(compression_weight) * (1 - args.ratio)))
    ]
    ref_attn, ref_mlp = attn_metric > threshold, mlp_metric > threshold

    # ---- ours
    stats = flap.collect(model, calib, device)
    attn, mlp = flap.scores(model, stats)
    our_attn, our_mlp = flap.allocate(attn, mlp, args.ratio, flap.group_weight(model))

    result = {
        "ratio": args.ratio,
        "groups_total": ref_attn.numel(),
        "groups_kept_ref": int(ref_attn.sum()),
        "groups_kept_ours": int(our_attn.sum()),
        "groups_differing": int((ref_attn != our_attn).sum()),
        "neurons_total": ref_mlp.numel(),
        "neurons_kept_ref": int(ref_mlp.sum()),
        "neurons_kept_ours": int(our_mlp.sum()),
        "neurons_differing": int((ref_mlp != our_mlp).sum()),
        "attn_score_max_abs_diff": float((attn - attn_metric).abs().max()),
        "mlp_score_max_abs_diff": float((mlp - mlp_metric).abs().max()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
