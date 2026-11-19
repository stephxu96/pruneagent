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

from pruneagent.web.jobs import MODELS

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
def model_sizes() -> list[dict]:
    from transformers import AutoConfig

    from pruneagent import estimate as E
    from pruneagent import plan as P

    out = []
    for m in MODELS:
        try:
            cfg = AutoConfig.from_pretrained(m)
            dense = P.dense_plan(cfg)
            # working memory while answering: the model's notes on the conversation so far (key/value cache),
            # for a conversation of WORKING_CONTEXT tokens, unpruned (an upper bound for any cut)
            out.append({"id": m, "size_mb": E.weight_bytes(cfg, dense) / 1e6,
                        "working_mb": E.kv_cache_bytes(cfg, dense, context=WORKING_CONTEXT) / 1e6})
        except OSError:  # offline and not cached
            continue
    return out
