"""This computer's chip and memory, and each model's original size, for the web app's "Your computer" panel.

The app runs on the user's own machine, so the server can read the real hardware (a website could not).
Model sizes come from the same estimator the runs use (`estimate.weight_bytes` on the unpruned plan).
"""

from __future__ import annotations

import functools
import json
import os
import platform
import subprocess

WORKING_CONTEXT = 2048  # tokens of conversation the suggested setup leaves room for
FREE_SHARE = 0.5  # share of the device's memory left for the system and other apps


@functools.cache
def machine() -> dict:
    name, chip, memory = platform.node(), platform.processor() or platform.machine(), None
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["system_profiler", "-json", "SPHardwareDataType"], capture_output=True, text=True,
                                 timeout=10, check=True).stdout
            hw = json.loads(out)["SPHardwareDataType"][0]
            name, chip = hw.get("machine_name", name), hw.get("chip_type", chip)
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError):
            pass
    try:
        memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        memory = None
    return {"name": name, "chip": chip, "memory_gb": memory / 2**30 if memory else None}


@functools.cache
def _size(model_id: str) -> dict | None:
    from transformers import AutoConfig

    from pruneagent import estimate as E
    from pruneagent import plan as P

    try:
        cfg = AutoConfig.from_pretrained(model_id)
        dense = P.dense_plan(cfg)
    except (OSError, ValueError, KeyError):  # offline and not cached, or a layout we can't read
        return None
    # working memory while answering: the model's notes on the conversation so far (key/value cache),
    # for a conversation of WORKING_CONTEXT tokens, unpruned (an upper bound for any cut)
    return {"size_mb": E.weight_bytes(cfg, dense) / 1e6,
            "working_mb": E.kv_cache_bytes(cfg, dense, context=WORKING_CONTEXT) / 1e6}


def model_sizes(ids: tuple[str, ...] | list[str] | None = None) -> list[dict]:
    """Each model's size and whether it is prepared on this machine (read fresh: a preparation may just have
    finished). `ids` defaults to the built-in models."""
    from pruneagent import models

    out = []
    for m in ids or models.BUILT_IN:
        size = _size(m)
        if size:
            out.append({"id": m, "ready": ready(m), "built_in": m in models.BUILT_IN, **size})
    return out


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def live() -> dict:
    """What the machine is doing right now (macOS; no admin rights needed): memory in use, GPU busy and the memory
    the GPU is using. Fields are None where a reading is not available."""
    import re

    out: dict = {"memory_used_pct": None, "gpu_busy_pct": None, "gpu_memory_mb": None}
    if platform.system() != "Darwin":
        return out
    free = re.search(r"free percentage:\s*(\d+)%", _run(["memory_pressure"]))
    if free:
        out["memory_used_pct"] = 100 - int(free.group(1))
    gpu = _run(["ioreg", "-r", "-c", "IOAccelerator", "-d", "1", "-w0"])
    busy = re.search(r'"Device Utilization %"=(\d+)', gpu)
    used = re.search(r'"In use system memory"=(\d+)', gpu)
    if busy:
        out["gpu_busy_pct"] = int(busy.group(1))
    if used:
        out["gpu_memory_mb"] = int(used.group(1)) / 1e6
    return out


def ready(model_id: str, device: str = "mps", dtype: str = "float16") -> bool:
    """Whether the model has been prepared for a run on this machine: its quick tests exist (a run stops without
    them; see `realrun.make_setup`). `pruneagent prepare` makes them."""
    from pathlib import Path

    from pruneagent import realrun as RR

    try:
        return Path(RR.make_setup(model_id, device, dtype).probes).exists()
    except (OSError, KeyError, ValueError):
        return False
