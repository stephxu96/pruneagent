"""Speed meter: measured prefill and decode tokens/s and peak memory of one model on one device, plus a
device test (memory bandwidth and fp16 compute) recorded as a device target.

Fixed settings (WORKPLAN 1.6): batch 1, a 512-token prompt (the first 512 tokens of the cached WikiText-2
validation stream), 128 generated tokens, greedy decoding, key/value cache on, no early stop.

One run of `measure`:
    prefill  one forward pass over the prompt (filling the key/value cache, logits for the last position only,
             as `generate` does) and the argmax of the first new token. prefill tokens/s = prompt_len / time.
    decode   `gen - 1` forward passes of one token each, each reading the cache and appending to it, greedy
             argmax on the device. Together with the token from prefill that is exactly `gen` new tokens.
             decode tokens/s = (gen - 1) / time.
The device is synchronized before and after each timed section. `warmup` untimed runs come first (kernel
compilation, clock ramp-up); then `repeats` timed runs, reported as median, min, max and spread
= (max - min) / median.

Peak memory is taken in one extra, untimed run (sampling hooks would disturb the timings):
    cuda  torch.cuda.max_memory_allocated() after reset_peak_memory_stats(): the exact peak of memory held by
          tensors (weights + key/value cache + activations), excluding the allocator's unused cached blocks.
    mps   PyTorch has no peak counter on MPS, so torch.mps.current_allocated_memory() (memory held by tensors,
          the same quantity as on cuda) and torch.mps.driver_allocated_memory() (everything the Metal driver
          has given the process, including the allocator's cache) are sampled after every module's forward
          pass and the maximum kept. Activations freed inside a single module are not seen, so this can be
          slightly below the true peak.
    cpu   not measured (None).
Weight bytes = bytes of all distinct parameter and buffer storage (a tied lm_head counted once).
"""

from __future__ import annotations

import platform
import statistics
import subprocess
import time
from datetime import UTC, datetime

import torch

from pruneagent.evaluate import synchronize

PROMPT_LEN = 512
GEN = 128
BATCH = 1
SEED = 0


def weight_bytes(model) -> int:
    seen, total = set(), 0
    for t in [*model.parameters(), *model.buffers()]:
        key = t.untyped_storage().data_ptr() if t.device.type != "meta" else id(t)
        if key in seen:
            continue
        seen.add(key)
        total += t.numel() * t.element_size()
    return total


def summarize(values: list[float]) -> dict:
    med = statistics.median(values)
    return {"median": med, "min": min(values), "max": max(values),
            "spread": (max(values) - min(values)) / med if med else None, "runs": values}


@torch.no_grad()
def _run(model, prompt: torch.Tensor, gen: int, device: str) -> tuple[float, float, torch.Tensor]:
    """One prefill + decode run; returns (prefill seconds, decode seconds, all generated token ids)."""
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    out_ids = torch.empty((prompt.shape[0], gen), dtype=torch.long, device=device)
    synchronize(device)
    t0 = time.perf_counter()
    logits = model(input_ids=prompt, past_key_values=cache, use_cache=True, logits_to_keep=1).logits
    tok = logits[:, -1:].argmax(-1)
    out_ids[:, :1] = tok
    synchronize(device)
    t1 = time.perf_counter()
    for i in range(1, gen):
        logits = model(input_ids=tok, past_key_values=cache, use_cache=True).logits
        tok = logits[:, -1:].argmax(-1)
        out_ids[:, i : i + 1] = tok
    synchronize(device)
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1, out_ids


def generate_tokens(model, prompt: torch.Tensor, gen: int = GEN, device: str = "cpu") -> torch.Tensor:
    """The `gen` greedy tokens the meter produces for `prompt` ([batch, gen])."""
    return _run(model, prompt.to(device), gen, device)[2]


@torch.no_grad()
def _peak_memory(model, prompt: torch.Tensor, gen: int, device: str) -> dict:
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _run(model, prompt, gen, device)
        return {"peak_memory_bytes": torch.cuda.max_memory_allocated(),
                "peak_memory_method": "torch.cuda.max_memory_allocated (tensors only)"}
    if device == "mps":
        torch.mps.empty_cache()
        peak = {"current": torch.mps.current_allocated_memory(), "driver": torch.mps.driver_allocated_memory()}

        def sample(*_):
            peak["current"] = max(peak["current"], torch.mps.current_allocated_memory())
            peak["driver"] = max(peak["driver"], torch.mps.driver_allocated_memory())

        handles = [m.register_forward_hook(sample) for m in model.modules()]
        try:
            _run(model, prompt, gen, device)
        finally:
            for h in handles:
                h.remove()
        sample()
        return {"peak_memory_bytes": peak["current"], "peak_driver_memory_bytes": peak["driver"],
                "peak_memory_method": "max of torch.mps.current_allocated_memory (tensors only) sampled after "
                                      "every module forward; driver figure from torch.mps.driver_allocated_memory"}
    return {"peak_memory_bytes": None, "peak_memory_method": "not measured on cpu"}


def measure(model, device: str, prompt: torch.Tensor, prompt_len: int = PROMPT_LEN, gen: int = GEN,
            repeats: int = 5, warmup: int = 3, seed: int = SEED) -> dict:
    """Prefill and decode tokens/s (median and spread over `repeats`) and peak memory of `model` on `device`.

    `prompt`: token ids, [N] or [1, N] with N >= prompt_len; the first `prompt_len` are used (batch 1).
    """
    if gen < 2:
        raise ValueError("gen must be at least 2 (one token from prefill, at least one decode step)")
    torch.manual_seed(seed)
    ids = prompt.reshape(-1)[:prompt_len]
    if ids.numel() < prompt_len:
        raise ValueError(f"prompt has {ids.numel()} tokens, need {prompt_len}")
    ids = ids.reshape(BATCH, prompt_len).to(device)
    model.eval()

    for _ in range(warmup):
        _run(model, ids, gen, device)
    prefill, decode, outputs = [], [], None
    for _ in range(repeats):
        tp, td, out = _run(model, ids, gen, device)
        prefill.append(prompt_len * BATCH / tp)
        decode.append((gen - 1) * BATCH / td)
        out = out.cpu()
        if outputs is None:
            outputs = out
        elif not torch.equal(outputs, out):
            raise RuntimeError("greedy output changed between repeats")
    record = {
        "settings": {"prompt_len": prompt_len, "batch": BATCH, "gen": gen, "decode_steps": gen - 1,
                     "repeats": repeats, "warmup": warmup, "seed": seed, "decoding": "greedy, key/value cache"},
        "device": device,
        "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
        "parameters": sum(p.numel() for p in {id(p): p for p in model.parameters()}.values()),
        "weight_bytes": weight_bytes(model),
        "generated_tokens": int(outputs.shape[1]),
        "prefill_tok_s": summarize(prefill),
        "decode_tok_s": summarize(decode),
    }
    record |= _peak_memory(model, ids, gen, device)
    return record


# ---------------------------------------------------------------- device test


def _timed(device: str, iters: int, fn, *args, **kwargs) -> float:
    synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    synchronize(device)
    return (time.perf_counter() - t0) / iters


@torch.no_grad()
def device_test(device: str, copy_mb: int = 1024, matmul_n: int = 4096, iters: int = 20,
                repeats: int = 5, warmup: int = 3) -> dict:
    """Memory bandwidth from a device-to-device copy and fp16 compute from a square matmul.

    bandwidth GB/s = 2 x copy bytes / time (each byte read once and written once), 1 GB = 1e9 bytes.
    compute TFLOPs = 2 n^3 / time for an n x n fp16 matmul (fp32 on cpu, where fp16 matmul is emulated).
    Each figure is the median of `repeats` timings of `iters` back-to-back operations, after `warmup`.
    """
    n_elem = copy_mb * (1 << 20) // 2
    src = torch.randn(n_elem, dtype=torch.float16, device=device)
    dst = torch.empty_like(src)
    bw = []
    for r in range(warmup + repeats):
        s = _timed(device, iters, dst.copy_, src)
        if r >= warmup:
            bw.append(2 * src.numel() * src.element_size() / s / 1e9)
    del src, dst

    mm_dtype = torch.float32 if device == "cpu" else torch.float16
    a = torch.randn(matmul_n, matmul_n, dtype=mm_dtype, device=device)
    b = torch.randn(matmul_n, matmul_n, dtype=mm_dtype, device=device)
    c = torch.empty_like(a)
    tf = []
    for r in range(warmup + repeats):
        s = _timed(device, iters, torch.mm, a, b, out=c)
        if r >= warmup:
            tf.append(2 * matmul_n**3 / s / 1e12)
    del a, b, c
    return {
        "bandwidth": summarize(bw), "compute": summarize(tf),
        "method": {"copy_mb": copy_mb, "matmul_n": matmul_n, "matmul_dtype": str(mm_dtype).removeprefix("torch."),
                   "iters": iters, "repeats": repeats, "warmup": warmup,
                   "bandwidth": "2 x bytes copied / s (read + write), GB = 1e9 bytes",
                   "compute": "2 n^3 / s for an n x n matmul, TFLOPs = 1e12 FLOPs/s"},
    }


def machine_info(device: str) -> dict:
    """Device name and total memory as the OS or driver reports them."""
    info = {"platform": platform.platform(), "processor": platform.processor() or platform.machine(),
            "torch": torch.__version__}
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        info |= {"device_name": props.name, "memory_total_gb": props.total_memory / 1e9}
    else:
        try:
            brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                   text=True, check=True).stdout.strip()
            mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                                     check=True).stdout.strip())
            info |= {"device_name": brand + (" GPU" if device == "mps" else ""), "memory_total_gb": mem / 1e9}
        except (OSError, subprocess.CalledProcessError, ValueError):
            info |= {"device_name": platform.processor() or platform.machine(), "memory_total_gb": None}
    return info


def device_target(name: str, device: str, test: dict) -> dict:
    """Device-target JSON for the estimator: plain keys first, measurement details after."""
    return {
        "name": name,
        "kind": "cpu" if device == "cpu" else "gpu",
        "bandwidth_gbs": round(test["bandwidth"]["median"], 1),
        "compute_tflops": round(test["compute"]["median"], 2),
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "device": device,
        **machine_info(device),
        "test": test,
    }
