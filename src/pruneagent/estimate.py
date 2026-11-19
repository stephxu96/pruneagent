"""Device Estimator (ARCHITECTURE.md §6, ADR-006): memory and speed of a plan, from the unpruned model's config
and the plan alone. No weights, no hardware.

What is counted:
- Weight bytes: exactly the tensors `loader.save` writes to model.safetensors, at `dtype_bytes` per value: per kept
  layer the seven linear weights at their planned shapes, the o_proj / down_proj compensation bias when groups /
  neurons were removed (FLAP), q/k norms and the two layer norms; the token embeddings; the final norm; lm_head only
  when it is not tied to the embeddings (a tied lm_head is stored once). The file adds a small JSON header (~0.01% on
  Qwen3-0.6B), which is not counted.
- Unstructured Wanda sparsity saves nothing: zeros are stored densely and read at full cost, so `wanda_sparsity`
  changes none of these numbers. Only removed neurons, groups and layers shrink the model.
- Bytes read per generated token (batch 1): every kept block weight and norm, the final norm, the full lm_head
  matrix (the embedding matrix when tied) and one embedding row, plus the key/value cache of every kept layer at
  `context` tokens (keys and values, kept groups x head_dim each, same dtype as the weights).
- Prompt (prefill) FLOPs for `prompt` tokens: 2 x (block linear weights) per token, plus attention scores and the
  weighted sum over the causal half (2 matmuls x 2 FLOPs x query width x prompt(prompt+1)/2 per layer), plus
  lm_head for the last position only (generation needs logits for that position only). Norms, rotary embeddings,
  softmax and activations are ignored.
- Speed is a roofline estimate: time = max(bytes / bandwidth, FLOPs / compute). Decode is bandwidth-bound and
  prefill compute-bound on both measured machines. These are ideal upper bounds; how well they rank plans against
  measured tokens/s is checked in task 1.7 (A6, A8).
- The memory limit counts weights only (decided 2026-11-17): `check` compares `weight_bytes` with it. The key/value
  cache is reported beside it, at a stated text length, and is not part of the limit.
- Per-layer cost (G1, task 2.11): measured decode is far slower than the roofline estimate, and the extra time grows
  with the number of kept layers (a fixed cost per layer per token, mostly kernel launches). Decode speed for
  ranking plans and checking a minimum speed is 1 / (1 / estimated + layer_cost_s x kept layers), with
  `LAYER_COST_S` fitted on the task 1.7 measurements (`speed_check.fit_layer_cost`). Speed is reported only for
  the two machines where this was checked against measurements, the M4 and the T4 (`speed_machine`); other
  devices, such as a phone, get memory fit only (A8).

Units: GB = 1e9 bytes, GB/s = 1e9 bytes/s, TFLOPs = 1e12 FLOP/s (datasheet conventions).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from pruneagent import plan as P

GB = 1e9
DEFAULT_CONTEXT = 512  # §6: decode speed at a 512-token context
DEFAULT_PROMPT = 512  # §6: prompt time for a 512-token prompt
REPORT_CONTEXT = 2048  # text length at which the device report states the key/value cache size


@dataclass(frozen=True)
class DeviceTarget:
    """A device the pruned model must fit and run on (ARCHITECTURE.md §7 `DeviceTarget`)."""

    name: str
    memory_limit_gb: float
    bandwidth_gb_s: float
    compute_tflops: float
    backend: str = "gpu"  # "cpu" or "gpu"; recorded for the 1.7 check, not used in the formulas
    min_decode_tok_s: float | None = None

    def __post_init__(self):
        if self.backend not in ("cpu", "gpu"):
            raise ValueError(f"backend must be 'cpu' or 'gpu', not {self.backend!r}")
        for f in ("memory_limit_gb", "bandwidth_gb_s", "compute_tflops"):
            if not getattr(self, f) > 0:
                raise ValueError(f"{f} must be positive")

    @classmethod
    def from_dict(cls, d: dict) -> DeviceTarget:
        """Also reads a measured device file from `pruneagent device-test` (speed.device_target): bandwidth_gbs and
        kind stand for bandwidth_gb_s and backend. Such a file measures speed only, so without memory_limit_gb the
        limit is the same-named preset's (m4, t4: the memory a model may use there), else the reported
        memory_total_gb; with neither, it is an error rather than a guess."""
        d = dict(d)
        for alias, field in (("bandwidth_gbs", "bandwidth_gb_s"), ("kind", "backend")):
            if d.get(field) is None and d.get(alias) is not None:
                d[field] = d[alias]
        if d.get("memory_limit_gb") is None:
            preset = PRESETS.get(d.get("name"))
            d["memory_limit_gb"] = preset.memory_limit_gb if preset else d.get("memory_total_gb")
            if d["memory_limit_gb"] is None:
                raise ValueError(f"device target {d.get('name')!r} has no memory_limit_gb and no memory_total_gb")
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Datasheet values only; task 1.6 measures the real ones. Memory limit is the machine's memory (M4: the 24 GB
# laptop's unified memory; T4: what Colab exposes of its 16 GB).
PRESETS: dict[str, DeviceTarget] = {
    # Apple M4, 10-core GPU: 120 GB/s unified memory (Apple). Apple publishes no FLOPs figure; ~4.3 TFLOPs fp32 is
    # the commonly cited third-party number (10 cores x 128 ALUs x 2 x ~1.7 GHz).
    "m4": DeviceTarget("Apple M4 10-core GPU (datasheet)", memory_limit_gb=24, bandwidth_gb_s=120,
                       compute_tflops=4.3, backend="gpu"),
    # NVIDIA Tesla T4: 320 GB/s GDDR6, 65 TFLOPs fp16 tensor cores (NVIDIA datasheet).
    "t4": DeviceTarget("NVIDIA Tesla T4 (datasheet)", memory_limit_gb=15, bandwidth_gb_s=320,
                       compute_tflops=65, backend="gpu"),
}


# Seconds per kept layer per generated token, fitted by least squares through the origin on measured minus
# estimated seconds per token over the ten task 1.7 plans (results/speed_check/{m4,t4}.json; refit with
# `python -m pruneagent.speed_check`, which writes results/speed_check/layer_cost.json). The estimate the cost is
# added to depends on the bandwidth assumed, so there is one value per target as it was used in 1.7: the measured
# device files (named "m4" and "t4", from `pruneagent device-test`) and the datasheet presets.
LAYER_COST_S: dict[str, float] = {
    "m4": 2.331e-4,  # Apple M4 GPU, measured 94.8 GB/s: decode within 2% of measured on every plan
    "t4": 1.403e-3,  # Colab T4, measured 233.4 GB/s: within 10% (mean 4%)
    PRESETS["m4"].name: 3.209e-4,  # M4 datasheet, 120 GB/s: within 5% (mean 2%)
    PRESETS["t4"].name: 1.449e-3,  # T4 datasheet, 320 GB/s: within 10% (mean 5%)
}
SPEED_MACHINES = {"m4": "m4", "t4": "t4", PRESETS["m4"].name: "m4", PRESETS["t4"].name: "t4"}


def speed_machine(target: DeviceTarget) -> str | None:
    """"m4" or "t4" when speed estimates for `target` were checked against measured speed (task 1.7), else None:
    such a device (a phone) gets memory fit only."""
    return SPEED_MACHINES.get(target.name)


def layer_cost_s(target: DeviceTarget) -> float:
    """The fitted per-layer cost for `target`; 0 for a device without measurements (ranking then uses the roofline
    estimate alone, and no speed is reported)."""
    return LAYER_COST_S.get(target.name, 0.0)


def with_layer_cost(est_decode_tok_s: float, kept_layers: int, cost_s: float) -> float:
    """Decode tokens/s with `cost_s` seconds per kept layer added to each token's estimated time."""
    return 1 / (1 / est_decode_tok_s + cost_s * kept_layers)


def kept_layers(config, plan: dict) -> int:
    return len(_kept(P.validate(plan, config)))


def load_target(spec: str) -> DeviceTarget:
    """A preset name, a JSON file path, or an inline JSON object."""
    if spec in PRESETS:
        return PRESETS[spec]
    text = Path(spec).read_text() if Path(spec).is_file() else spec
    try:
        return DeviceTarget.from_dict(json.loads(text))
    except json.JSONDecodeError as e:
        raise ValueError(f"{spec!r} is not a preset ({', '.join(PRESETS)}), a JSON file or JSON") from e


@dataclass(frozen=True)
class Estimate:
    """ARCHITECTURE.md §7 `Estimate`, plus the parts the warnings and the 1.7 check need."""

    weight_bytes: int  # model.safetensors tensor data
    kv_cache_bytes: int  # at `context` tokens, batch 1
    memory_bytes: int  # weights + key/value cache at `context` (the limit counts weights only)
    decode_bytes_per_token: int
    decode_flops_per_token: int
    prefill_flops: int
    context: int
    prompt: int
    est_decode_tok_s: float | None = None
    est_prefill_s: float | None = None
    est_prefill_tok_s: float | None = None
    meets_target: bool | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Check:
    """A plan against a device target: what the early and final warnings (task 2.11) report."""

    target: DeviceTarget
    estimate: Estimate
    fits_memory: bool  # weights within the memory limit
    meets_speed: bool  # True when the target sets no minimum speed
    memory_gap_gb: float  # GB of weights over the memory limit (0 when it fits)
    speed_gap_tok_s: float  # tokens/s short of the minimum (0 when met or not set)
    decode_tok_s: float = 0.0  # estimated decode speed with the per-layer cost: the speed checked
    layer_cost_s: float = 0.0

    @property
    def meets_target(self) -> bool:
        return self.fits_memory and self.meets_speed

    def to_dict(self) -> dict:
        return {"target": asdict(self.target), "estimate": self.estimate.to_dict(), "fits_memory": self.fits_memory,
                "meets_speed": self.meets_speed, "meets_target": self.meets_target,
                "memory_gap_gb": self.memory_gap_gb, "speed_gap_tok_s": self.speed_gap_tok_s,
                "decode_tok_s": self.decode_tok_s, "layer_cost_s": self.layer_cost_s}


@dataclass(frozen=True)
class _Dims:
    vocab: int
    hidden: int
    width: int
    groups: int
    heads_per_group: int
    head_dim: int
    tied: bool
    attention_bias: bool
    mlp_bias: bool


def _dims(config) -> _Dims:
    c = P._cfg(config)
    heads, groups = c["num_attention_heads"], c["num_key_value_heads"]
    return _Dims(vocab=c["vocab_size"], hidden=c["hidden_size"], width=c["intermediate_size"], groups=groups,
                 heads_per_group=heads // groups, head_dim=c.get("head_dim") or c["hidden_size"] // heads,
                 tied=bool(c.get("tie_word_embeddings", False)), attention_bias=bool(c.get("attention_bias", False)),
                 mlp_bias=bool(c.get("mlp_bias", False)))


def _layer_counts(d: _Dims, p: P.LayerPlan) -> tuple[int, int]:
    """(values stored, of which linear weights) for one kept layer."""
    q = p["groups_keep"] * d.heads_per_group * d.head_dim
    kv, m, h = p["groups_keep"] * d.head_dim, p["mlp_keep"], d.hidden
    linear = h * q + 2 * h * kv + q * h + 3 * h * m
    other = 2 * d.head_dim + 2 * h  # q_norm, k_norm, input and post-attention layer norms
    if d.attention_bias:
        other += q + 2 * kv + h
    elif p["groups_keep"] < d.groups:
        other += h  # o_proj compensation bias
    if d.mlp_bias:
        other += 2 * m + h
    elif p["mlp_keep"] < d.width:
        other += h  # down_proj compensation bias
    return linear + other, linear


def _kept(plan: P.Plan) -> list[P.LayerPlan]:
    return [p for _, p in sorted(plan.items()) if not p["dropped"]]


def weight_bytes(config, plan: dict, dtype_bytes: int = 2) -> int:
    """Bytes of tensor data in the saved model.safetensors (see module docstring; Wanda zeros count in full)."""
    d, full = _dims(config), P.validate(plan, config)
    values = sum(_layer_counts(d, p)[0] for p in _kept(full))
    values += d.vocab * d.hidden + d.hidden  # embeddings, final norm
    if not d.tied:
        values += d.vocab * d.hidden  # lm_head
    return values * dtype_bytes


def _full(plan: dict, config, strict: bool) -> P.Plan:
    # strict=False accepts any MLP width (FLAP's plans), not only the agent's 64-neuron steps
    return P.validate(plan, config) if strict else P.normalize(plan, config)


def kv_cache_bytes(config, plan: dict, context: int = DEFAULT_CONTEXT, dtype_bytes: int = 2,
                   strict: bool = True) -> int:
    """Key/value cache for `context` tokens at batch 1: keys and values of the kept groups in every kept layer."""
    d, full = _dims(config), _full(plan, config, strict)
    return sum(2 * p["groups_keep"] * d.head_dim * context for p in _kept(full)) * dtype_bytes


def decode_bytes_per_token(config, plan: dict, context: int = DEFAULT_CONTEXT, dtype_bytes: int = 2,
                           strict: bool = True) -> int:
    """Bytes read from memory to generate one token at `context` tokens of history (batch 1): the memory access
    cost (MAC) per token. `strict=False` accepts any MLP width (for FLAP's plans)."""
    d, full = _dims(config), _full(plan, config, strict)
    values = sum(_layer_counts(d, p)[0] for p in _kept(full))
    values += d.hidden + d.vocab * d.hidden + d.hidden  # final norm, lm_head (or tied embeddings), one embedding row
    return values * dtype_bytes + kv_cache_bytes(config, full, context, dtype_bytes, strict=False)


def _matmul_params(d: _Dims, full: P.Plan) -> int:
    return sum(_layer_counts(d, p)[1] for p in _kept(full))


def decode_flops_per_token(config, plan: dict, context: int = DEFAULT_CONTEXT) -> int:
    """FLOPs for one generated token: block linear layers, attention over `context` cached tokens, lm_head."""
    d, full = _dims(config), P.validate(plan, config)
    attn = sum(2 * 2 * p["groups_keep"] * d.heads_per_group * d.head_dim * context for p in _kept(full))
    return 2 * _matmul_params(d, full) + attn + 2 * d.vocab * d.hidden


def prefill_flops(config, plan: dict, prompt: int = DEFAULT_PROMPT) -> int:
    """FLOPs to process a `prompt`-token prompt (causal attention; lm_head for the last position only)."""
    d, full = _dims(config), P.validate(plan, config)
    causal_pairs = prompt * (prompt + 1) // 2
    attn = sum(2 * 2 * p["groups_keep"] * d.heads_per_group * d.head_dim * causal_pairs for p in _kept(full))
    return 2 * _matmul_params(d, full) * prompt + attn + 2 * d.vocab * d.hidden


def estimate(config, plan: dict, target: DeviceTarget | None = None, *, dtype_bytes: int = 2,
             context: int = DEFAULT_CONTEXT, prompt: int = DEFAULT_PROMPT) -> Estimate:
    """Memory and FLOP counts for `plan` on the model described by `config` (the unpruned Hugging Face config, as
    an object or dict); with a target, also estimated speeds and whether the target is met."""
    full = P.validate(plan, config)
    w = weight_bytes(config, full, dtype_bytes)
    kv = kv_cache_bytes(config, full, context, dtype_bytes)
    dec_b = decode_bytes_per_token(config, full, context, dtype_bytes)
    dec_f = decode_flops_per_token(config, full, context)
    pre_f = prefill_flops(config, full, prompt)
    speeds: dict = {}
    if target is not None:
        bw, fl = target.bandwidth_gb_s * GB, target.compute_tflops * 1e12
        decode_s = max(dec_b / bw, dec_f / fl)
        prefill_s = max(pre_f / fl, (w + kv) / bw)
        speeds = {"est_decode_tok_s": 1 / decode_s, "est_prefill_s": prefill_s,
                  "est_prefill_tok_s": prompt / prefill_s}
        speeds["meets_target"] = (w <= target.memory_limit_gb * GB and (
            target.min_decode_tok_s is None or speeds["est_decode_tok_s"] >= target.min_decode_tok_s))
    return Estimate(weight_bytes=w, kv_cache_bytes=kv, memory_bytes=w + kv, decode_bytes_per_token=dec_b,
                    decode_flops_per_token=dec_f, prefill_flops=pre_f, context=context, prompt=prompt, **speeds)


def check(target: DeviceTarget, config, plan: dict, *, dtype_bytes: int = 2, context: int = DEFAULT_CONTEXT,
          prompt: int = DEFAULT_PROMPT, layer_cost: float | None = None) -> Check:
    """Do `plan`'s weights fit `target`'s memory limit, and does it reach the minimum decode speed? With the gaps
    when it does not. Speed includes the per-layer cost (`layer_cost`, default: the fitted one for the target)."""
    e = estimate(config, plan, target, dtype_bytes=dtype_bytes, context=context, prompt=prompt)
    cost = layer_cost_s(target) if layer_cost is None else layer_cost
    speed = with_layer_cost(e.est_decode_tok_s, kept_layers(config, plan), cost)
    over = max(0.0, e.weight_bytes / GB - target.memory_limit_gb)
    short = 0.0 if target.min_decode_tok_s is None else max(0.0, target.min_decode_tok_s - speed)
    return Check(target=target, estimate=e, fits_memory=over == 0, meets_speed=short == 0,
                 memory_gap_gb=over, speed_gap_tok_s=short, decode_tok_s=speed, layer_cost_s=cost)
