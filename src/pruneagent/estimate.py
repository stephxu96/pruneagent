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
    memory_bytes: int  # weights + key/value cache: compared with the memory limit
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
    fits_memory: bool
    meets_speed: bool  # True when the target sets no minimum speed
    memory_gap_gb: float  # GB over the memory limit (0 when it fits)
    speed_gap_tok_s: float  # tokens/s short of the minimum (0 when met or not set)

    @property
    def meets_target(self) -> bool:
        return self.fits_memory and self.meets_speed

    def to_dict(self) -> dict:
        return {"target": asdict(self.target), "estimate": self.estimate.to_dict(), "fits_memory": self.fits_memory,
                "meets_speed": self.meets_speed, "meets_target": self.meets_target,
                "memory_gap_gb": self.memory_gap_gb, "speed_gap_tok_s": self.speed_gap_tok_s}


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


def kv_cache_bytes(config, plan: dict, context: int = DEFAULT_CONTEXT, dtype_bytes: int = 2) -> int:
    """Key/value cache for `context` tokens at batch 1: keys and values of the kept groups in every kept layer."""
    d, full = _dims(config), P.validate(plan, config)
    return sum(2 * p["groups_keep"] * d.head_dim * context for p in _kept(full)) * dtype_bytes


def decode_bytes_per_token(config, plan: dict, context: int = DEFAULT_CONTEXT, dtype_bytes: int = 2) -> int:
    """Bytes read from memory to generate one token at `context` tokens of history (batch 1)."""
    d, full = _dims(config), P.validate(plan, config)
    values = sum(_layer_counts(d, p)[0] for p in _kept(full))
    values += d.hidden + d.vocab * d.hidden + d.hidden  # final norm, lm_head (or tied embeddings), one embedding row
    return values * dtype_bytes + kv_cache_bytes(config, full, context, dtype_bytes)


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
        speeds["meets_target"] = (w + kv <= target.memory_limit_gb * GB and (
            target.min_decode_tok_s is None or speeds["est_decode_tok_s"] >= target.min_decode_tok_s))
    return Estimate(weight_bytes=w, kv_cache_bytes=kv, memory_bytes=w + kv, decode_bytes_per_token=dec_b,
                    decode_flops_per_token=dec_f, prefill_flops=pre_f, context=context, prompt=prompt, **speeds)


def check(target: DeviceTarget, config, plan: dict, *, dtype_bytes: int = 2, context: int = DEFAULT_CONTEXT,
          prompt: int = DEFAULT_PROMPT) -> Check:
    """Does `plan` fit `target`'s memory and reach its minimum decode speed? With the gaps when it does not."""
    e = estimate(config, plan, target, dtype_bytes=dtype_bytes, context=context, prompt=prompt)
    over = max(0.0, e.memory_bytes / GB - target.memory_limit_gb)
    short = 0.0 if target.min_decode_tok_s is None else max(0.0, target.min_decode_tok_s - e.est_decode_tok_s)
    return Check(target=target, estimate=e, fits_memory=over == 0, meets_speed=short == 0,
                 memory_gap_gb=over, speed_gap_tok_s=short)
