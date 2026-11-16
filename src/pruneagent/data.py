"""Calibration and evaluation data: downloaded once, cached as token ids, hashed in a manifest.

- Calibration: 128 random 2,048-token windows from the first C4 English training shard, sampled with
  the same procedure (and seed) as the Wanda and SparseGPT reference code, so their results reproduce.
- Evaluation: WikiText-2 (raw) validation and test splits, each joined with blank lines and tokenized
  as one sequence, as in the reference code.

Every cached file is recorded in `manifest.json` with its SHA-256, shape, source dataset revision and
tokenizer revision. Loading verifies the hash, so a changed or corrupted cache fails loudly.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

C4_REPO = "allenai/c4"
C4_SHARD = "en/c4-train.00000-of-01024.json.gz"
WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
WIKITEXT_JOIN = "\n\n"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
MANIFEST = "manifest.json"
TENSOR_KEY = "input_ids"


class CacheError(RuntimeError):
    """The cache is missing an entry, or a file does not match its recorded hash."""


def cache_dir() -> Path:
    """Cache location: $PRUNEAGENT_CACHE, else ./data_cache."""
    return Path(os.environ.get("PRUNEAGENT_CACHE", "data_cache")).resolve()


def slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def calib_name(model_id: str = DEFAULT_MODEL) -> str:
    return f"c4_calib__{slug(model_id)}"


def wikitext_name(split: str, model_id: str = DEFAULT_MODEL) -> str:
    return f"wikitext2_{split}__{slug(model_id)}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- sampling (pure, unit-tested)


def sample_calibration(
    texts: Sequence[str],
    tokenize: Callable[[str], list[int]],
    nsamples: int,
    seqlen: int,
    seed: int,
) -> torch.Tensor:
    """Wanda / SparseGPT `get_c4`: pick random documents until one has more than `seqlen` tokens,
    then take a random `seqlen` window from it. Returns int64 [nsamples, seqlen]."""
    rng = random.Random(seed)
    rows = []
    for _ in range(nsamples):
        while True:
            i = rng.randint(0, len(texts) - 1)
            ids = tokenize(texts[i])
            if len(ids) > seqlen:
                break
        start = rng.randint(0, len(ids) - seqlen - 1)
        rows.append(ids[start : start + seqlen])
    return torch.tensor(rows, dtype=torch.int64)


def tokenize_joined(lines: Sequence[str], tokenize: Callable[[str], list[int]]) -> torch.Tensor:
    """WikiText evaluation stream: all lines joined with blank lines, tokenized once. int64 [1, N]."""
    return torch.tensor([tokenize(WIKITEXT_JOIN.join(lines))], dtype=torch.int64)


# ---------------------------------------------------------------- manifest


def read_manifest(out: Path | None = None) -> dict:
    path = (out or cache_dir()) / MANIFEST
    if not path.exists():
        return {"version": 1, "entries": {}}
    return json.loads(path.read_text())


def write_entry(out: Path, name: str, tensor: torch.Tensor, meta: dict) -> dict:
    """Save `tensor` as `<name>.safetensors` and record it in the manifest."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.safetensors"
    save_file({TENSOR_KEY: tensor.contiguous()}, path)
    entry = {
        "file": path.name,
        "sha256": sha256_file(path),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        **meta,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    manifest = read_manifest(out)
    manifest["entries"][name] = entry
    manifest["libraries"] = {lib: _version(lib) for lib in ("torch", "transformers", "datasets", "safetensors")}
    tmp = out / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(tmp, out / MANIFEST)
    return entry


def load_tokens(name: str, out: Path | None = None) -> torch.Tensor:
    """Load a cached tensor after checking its hash against the manifest."""
    out = out or cache_dir()
    entry = read_manifest(out)["entries"].get(name)
    if entry is None:
        raise CacheError(f"'{name}' is not in {out / MANIFEST}. Run `pruneagent data` first.")
    path = out / entry["file"]
    if not path.exists():
        raise CacheError(f"{path} is listed in the manifest but missing.")
    if sha256_file(path) != entry["sha256"]:
        raise CacheError(f"{path} does not match its recorded SHA-256; rebuild with `pruneagent data`.")
    return load_file(path)[TENSOR_KEY]


def verify(out: Path | None = None) -> dict[str, bool]:
    """Check every manifest entry's hash. Returns {name: ok}."""
    out = out or cache_dir()
    result = {}
    for name, entry in read_manifest(out)["entries"].items():
        path = out / entry["file"]
        result[name] = path.exists() and sha256_file(path) == entry["sha256"]
    return result


def _version(lib: str) -> str | None:
    try:
        return version(lib)
    except Exception:  # noqa: BLE001 - any metadata failure just means "unknown"
        return None


# ---------------------------------------------------------------- build (network)


def build_cache(
    model_id: str = DEFAULT_MODEL,
    nsamples: int = 128,
    seqlen: int = 2048,
    seed: int = 0,
    out: Path | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Download C4 and WikiText-2, tokenize with the model's tokenizer, cache and hash. Network needed."""
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    out = out or cache_dir()
    api = HfApi()
    model_rev = api.model_info(model_id).sha
    c4_rev = api.dataset_info(C4_REPO).sha
    wt_rev = api.dataset_info(WIKITEXT_REPO).sha
    tokenizer_meta = {"id": model_id, "revision": model_rev}

    log(f"Tokenizer {model_id}@{model_rev[:8]}")
    tok = AutoTokenizer.from_pretrained(model_id, revision=model_rev)

    def tokenize(text: str) -> list[int]:
        return tok(text)["input_ids"]

    log(f"C4 shard {C4_SHARD}@{c4_rev[:8]} (first run downloads ~300 MB)")
    c4 = load_dataset(C4_REPO, data_files={"train": C4_SHARD}, split="train", revision=c4_rev)
    calib = sample_calibration(_Column(c4, "text"), tokenize, nsamples, seqlen, seed)
    write_entry(
        out,
        calib_name(model_id),
        calib,
        {
            "source": {"repo": C4_REPO, "revision": c4_rev, "data_files": C4_SHARD, "split": "train"},
            "params": {"nsamples": nsamples, "seqlen": seqlen, "seed": seed, "procedure": "wanda/sparsegpt get_c4"},
            "tokenizer": tokenizer_meta,
        },
    )
    log(f"  calibration {tuple(calib.shape)}")

    for split in ("validation", "test"):
        ds = load_dataset(WIKITEXT_REPO, WIKITEXT_CONFIG, split=split, revision=wt_rev)
        ids = tokenize_joined(ds["text"], tokenize)
        write_entry(
            out,
            wikitext_name(split, model_id),
            ids,
            {
                "source": {"repo": WIKITEXT_REPO, "revision": wt_rev, "config": WIKITEXT_CONFIG, "split": split},
                "params": {"join": WIKITEXT_JOIN, "windows_of_seqlen": seqlen},
                "tokenizer": tokenizer_meta,
            },
        )
        log(f"  wikitext-2 {split} {tuple(ids.shape)} → {ids.shape[1] // seqlen} windows of {seqlen}")

    return read_manifest(out)


class _Column(Sequence[str]):
    """Lazy view of one column of a Hugging Face dataset, so the C4 shard is not loaded into a list."""

    def __init__(self, ds, column: str):
        self._ds, self._col = ds, column

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, i):  # type: ignore[override]
        return self._ds[i][self._col]
