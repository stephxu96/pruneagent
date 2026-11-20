"""Bring your own model (FR17): check whether a Hugging Face model can be pruned here, estimate how long preparing
it takes, and keep the list of models the user added.

Supported families are the ones whose layers the pruner, loader and estimator handle exactly (tested on tiny random
models in tests/test_families.py): decoder layers with grouped-query attention and a gated MLP, as in Qwen3, Qwen2,
Llama and Mistral. Others are refused in plain words.

Preparing a model computes, on this machine: its tokenized text (C4 and WikiText-2), its neuron scores and quick
tests, and its measured damage table (`pruneagent prepare`). Nothing of it goes to git.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

BUILT_IN = ("Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base")
FAMILIES = {"qwen3": "Qwen3", "qwen2": "Qwen2", "llama": "Llama", "mistral": "Mistral"}
MODEL_ID = re.compile(r"^[A-Za-z0-9][\w.-]{0,95}/[A-Za-z0-9][\w.-]{0,95}$")
REGISTRY = "models.json"

# Measured on this Mac (M4, float16): the damage table took 65 s per entry on Qwen3-0.6B and 85 s on Qwen3-1.7B,
# 4 entries per layer; scores and quick tests took ~13 min on 1.7B. Fitted as a line in billions of parameters.
ENTRY_S = (49.0, 21.8)  # seconds per damage-table entry = a + b * billions of parameters
FIRST_MIN = (5.0, 5.0)  # minutes for text, scores and quick tests = a + b * billions
SPREAD = (0.8, 1.3)  # the range shown around the fitted time
MEMORY_SHARE = 0.6  # the model's weights may use at most this share of the computer's memory while preparing


class ModelError(ValueError):
    """A model that cannot be prepared, with a plain reason."""


def valid_id(model_id: str) -> bool:
    return bool(MODEL_ID.match(model_id))


def _config(model_id: str):
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    from transformers import AutoConfig

    if not valid_id(model_id):
        raise ModelError("Use the model's name as shown on Hugging Face, like Qwen/Qwen3-0.6B-Base.")
    try:
        return AutoConfig.from_pretrained(model_id)
    except GatedRepoError:
        raise ModelError("This model needs permission on Hugging Face first: accept its license on the model's "
                         "page and sign in on this computer (`hf auth login`).") from None
    except RepositoryNotFoundError:
        raise ModelError("No model with this name on Hugging Face.") from None
    except (OSError, ValueError) as e:
        msg = str(e)
        if "gated repo" in msg:
            raise ModelError("This model needs permission on Hugging Face first: accept its license on the "
                             "model's page and sign in on this computer (`hf auth login`).") from None
        if "not a valid model identifier" in msg:
            raise ModelError("No model with this name on Hugging Face. Check the spelling, or sign in if it "
                             "is private.") from None
        raise ModelError("Could not reach Hugging Face to read this model's details. Check the internet "
                         "connection.") from None


def prepare_minutes(params_b: float, layers: int) -> list[float]:
    fitted = FIRST_MIN[0] + FIRST_MIN[1] * params_b + 4 * layers * (ENTRY_S[0] + ENTRY_S[1] * params_b) / 60
    return [fitted * SPREAD[0], fitted * SPREAD[1]]


def check(model_id: str, memory_gb: float | None) -> dict:
    """Whether `model_id` can be prepared and pruned on this machine, and what it takes. Raises ModelError with a
    plain reason when it cannot."""
    from pruneagent import estimate as E
    from pruneagent import plan as P

    cfg = _config(model_id)
    kind = getattr(cfg, "model_type", None)
    if any(getattr(cfg, k, None) for k in ("num_experts", "num_local_experts", "n_routed_experts")):
        raise ModelError("This model is built from several expert parts, which isn't supported yet. Works with "
                         f"{', '.join(FAMILIES.values())} models.")
    if kind not in FAMILIES:
        raise ModelError(f"This kind of model isn't supported yet. Works with {', '.join(FAMILIES.values())} "
                         f"models (this one is {kind or 'unknown'}).")
    try:
        size = E.weight_bytes(cfg, P.dense_plan(cfg))
        layers = P.shape(cfg)[0]
    except (KeyError, ValueError, TypeError) as e:
        raise ModelError(f"This model's layout is not one PruneAgent can cut ({e}).") from None
    size_mb = size / 1e6
    if memory_gb and size_mb > memory_gb * 1000 * MEMORY_SHARE:
        raise ModelError(f"Too big for this computer: the model needs {size_mb / 1000:.1f} GB and this computer "
                         f"has {memory_gb:.0f} GB, so it can't be prepared here.")
    params_b = size / 2 / 1e9  # float16: 2 bytes per value
    return {"id": model_id, "family": FAMILIES[kind], "size_mb": size_mb, "layers": layers,
            "params_b": params_b, "prepare_minutes": prepare_minutes(params_b, layers)}


# ---------------------------------------------------------------- the user's models


def registry(root: Path | str) -> list[str]:
    path = Path(root) / REGISTRY
    try:
        ids = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    return [m for m in ids if isinstance(m, str) and valid_id(m) and m not in BUILT_IN]


def add(root: Path | str, model_id: str) -> None:
    ids = registry(root)
    if model_id not in ids and model_id not in BUILT_IN:
        path = Path(root) / REGISTRY
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([*ids, model_id], indent=2))


def known(root: Path | str) -> list[str]:
    """Built-in models first, then the ones the user added."""
    return [*BUILT_IN, *registry(root)]
