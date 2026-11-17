"""Save a pruned model as safetensors + plan JSON, and rebuild it without the unpruned weights.

A saved model is a directory:
    model.safetensors   the pruned weights (Wanda zeros stored as zeros; a tied lm_head is stored once)
    plan.json           {"format": "pruneagent-plan", "version": 1, "model_id", "revision", "dtype",
                         "dense_config": <the unpruned model's Hugging Face config>,
                         "plan": {"<layer>": {mlp_keep, groups_keep, dropped, wanda_sparsity}, ...},
                         "kept_layers": [...], "weights": {"file", "sha256"}, "extra": {...}, ...}

Hugging Face configs cannot express per-layer widths, so the plan is the source of truth: `load` builds the
unpruned architecture from `dense_config` with no memory allocated, reshapes each layer to its planned MLP width
and group count, removes dropped layers, then loads the weights strictly. o_proj / down_proj get a bias where the
weight file has one (FLAP bias compensation).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from importlib.metadata import version as pkg_version
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from pruneagent import plan as P
from pruneagent.data import sha256_file
from pruneagent.pruner import drop_layers

WEIGHTS = "model.safetensors"
PLAN = "plan.json"
FORMAT = "pruneagent-plan"
VERSION = 1


def _tied(model) -> bool:
    return model.lm_head.weight is model.model.embed_tokens.weight or (
        model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    )


def save(model, path, *, model_id: str | None = None, revision: str | None = None,
         extra: dict | None = None) -> Path:
    """Write `path/model.safetensors` and `path/plan.json` for a model made by `pruner.apply_plan` (or `load`)."""
    info = getattr(model, "pruneagent", None)
    if info is None:
        raise ValueError("model has no plan attached; build it with pruner.apply_plan or loader.load")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    if _tied(model):
        state.pop("lm_head.weight")
    save_file({k: v.detach().contiguous().cpu() for k, v in state.items()}, str(path / WEIGHTS))
    dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
    record = {
        "format": FORMAT, "version": VERSION,
        "model_id": model_id or info.get("model_id") or info["dense_config"].get("_name_or_path"),
        "revision": revision or info.get("revision"),
        "dtype": dtype,
        "plan": P.to_json(info["plan"]),
        "kept_layers": [i for i, p in sorted(info["plan"].items()) if not p["dropped"]],
        "dense_config": info["dense_config"],
        "weights": {"file": WEIGHTS, "sha256": sha256_file(path / WEIGHTS)},
        "extra": extra or info.get("extra") or {},
        "torch": pkg_version("torch"), "transformers": pkg_version("transformers"),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (path / PLAN).write_text(json.dumps(record, indent=2) + "\n")
    return path


def read_plan(path) -> dict:
    """The plan JSON of a saved model, with the plan validated against its config (layer keys as ints)."""
    record = json.loads((Path(path) / PLAN).read_text())
    if record.get("format") != FORMAT or record.get("version") != VERSION:
        raise ValueError(f"{path}: not a {FORMAT} v{VERSION} directory")
    record["plan"] = P.validate(record["plan"], record["dense_config"])
    return record


def build_config(dense_config: dict):
    from transformers import AutoConfig

    cfg = dict(dense_config)
    return AutoConfig.for_model(cfg.pop("model_type"), **cfg)


def _linear(in_f: int, out_f: int, bias: bool, dtype) -> nn.Linear:
    return nn.Linear(in_f, out_f, bias=bias, device="meta", dtype=dtype)


def load(path, device: str = "cpu", dtype: str | None = None, verify: bool = True):
    """Rebuild a saved pruned model; strict: every weight must match a parameter of the planned shape."""
    from transformers import AutoModelForCausalLM

    from pruneagent.evaluate import DTYPES

    path = Path(path)
    record = read_plan(path)
    if verify and sha256_file(path / WEIGHTS) != record["weights"]["sha256"]:
        raise ValueError(f"{path / WEIGHTS} does not match the hash in {PLAN}")
    torch_dtype = DTYPES[record["dtype"]]
    config = build_config(record["dense_config"])
    with safe_open(str(path / WEIGHTS), "pt") as f:
        keys = set(f.keys())
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch_dtype)

    plan = record["plan"]
    kept = [i for i, p in sorted(plan.items()) if not p["dropped"]]
    hidden = config.hidden_size
    for i, block in enumerate(model.model.layers):
        p = plan[i]
        if p["dropped"]:
            continue
        prefix = f"model.layers.{kept.index(i)}."
        a, m = block.self_attn, block.mlp
        q_dim = p["groups_keep"] * a.num_key_value_groups * a.head_dim
        kv_dim = p["groups_keep"] * a.head_dim
        for name, (in_f, out_f) in {"q_proj": (hidden, q_dim), "k_proj": (hidden, kv_dim),
                                    "v_proj": (hidden, kv_dim), "o_proj": (q_dim, hidden)}.items():
            has_bias = f"{prefix}self_attn.{name}.bias" in keys
            setattr(a, name, _linear(in_f, out_f, has_bias, torch_dtype))
        for name, (in_f, out_f) in {"gate_proj": (hidden, p["mlp_keep"]), "up_proj": (hidden, p["mlp_keep"]),
                                    "down_proj": (p["mlp_keep"], hidden)}.items():
            has_bias = f"{prefix}mlp.{name}.bias" in keys
            setattr(m, name, _linear(in_f, out_f, has_bias, torch_dtype))
        if hasattr(m, "intermediate_size"):
            m.intermediate_size = p["mlp_keep"]
    drop_layers(model, [i for i, p in plan.items() if p["dropped"]])

    state = load_file(str(path / WEIGHTS))
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    tied = "lm_head.weight" not in state
    if tied:
        missing = [k for k in missing if k != "lm_head.weight"]
        model.lm_head.weight = model.model.embed_tokens.weight
    if missing or unexpected:
        raise ValueError(f"weights do not match the plan: missing {missing}, unexpected {unexpected}")
    # non-persistent buffers (rotary inverse frequencies) are not in the file: rebuild them for real
    model.model.rotary_emb = type(model.model.rotary_emb)(config=model.config)
    for name, t in [*model.named_parameters(), *model.named_buffers()]:
        if t.is_meta:
            raise ValueError(f"{name} was not loaded")
    model.config.use_cache = False  # as evaluate.load_model; generate() still uses a cache by default
    model.pruneagent = {"dense_config": record["dense_config"], "plan": plan, "model_id": record["model_id"],
                        "revision": record["revision"], "extra": record.get("extra", {})}
    return model.to(device).eval()
