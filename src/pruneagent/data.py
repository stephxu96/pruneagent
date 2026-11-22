"""Calibration and evaluation data: downloaded once, cached as token ids, hashed in a manifest.

- Calibration: 128 random 2,048-token windows from the first C4 English training shard, sampled with
  the same procedure (and seed) as the Wanda and SparseGPT reference code, so their results reproduce.
  Every method calibrates on this one set (FLAP and Wanda-sp too, and the agent's scores), so compared methods
  see the same data, and it is a different source from the WikiText-2 evaluation text (decision of 2026-11-17).
- Gate: the keep-or-undo check and the damage table measure on 32 of 128 passages of C4's validation split
  (`c4_validation_name`): the calibration text's source, never in it. The other 96 are a held-out web-text check.
  Runs before 2026-11-21 used WikiText-2 validation windows (decision of 2026-11-21, ADR-004).
- Reference-only calibration sets, kept for the scripts that check our code against the authors' code at their
  published settings: Wanda-sp's 128 random 128-token C4 windows (`wanda_sp_reference_calib_name`), and FLAP's
  1,024 random 128-token windows of the WikiText-2 training split (`flap_reference_calib_name`, FLAP's
  `get_wikitext2`). No compared or benchmarked result uses them.
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
C4_VALIDATION_SHARD = "en/c4-validation.00000-of-00008.json.gz"  # C4's validation split: no overlap with C4_SHARD
WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
WIKITEXT_JOIN = "\n\n"
FLAP_JOIN = " "  # FLAP's get_wikitext2 joins the training split with single spaces
FLAP_NSAMPLES = 1024  # FLAP's published scripts: --nsamples 1024
FLAP_SEQLEN = 128  # FLAP's main.py: model.seqlen = 128
WANDA_SP_NSAMPLES = 128  # FLAP's prune_wanda_sp: get_loaders("c4", nsamples=128, seqlen=model.seqlen)
WANDA_SP_SEQLEN = 128
DEFAULT_MODEL = "Qwen/Qwen3-0.6B-Base"
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


def c4_validation_name(model_id: str = DEFAULT_MODEL) -> str:
    """128 x 2,048-token passages of C4's validation split: same source as the calibration text, never in it.
    32 of them (evaluate.subset_indices) are the gate; the other 96 are the held-out web-text check."""
    return f"c4_validation__{slug(model_id)}"


def flap_calib_name(model_id: str = DEFAULT_MODEL) -> str:
    """FLAP's calibration: the shared C4 set (`calib_name`), as every other method."""
    return calib_name(model_id)


def wanda_sp_calib_name(model_id: str = DEFAULT_MODEL) -> str:
    """Wanda-sp's calibration: the shared C4 set (`calib_name`), as every other method."""
    return calib_name(model_id)


def flap_reference_calib_name(model_id: str = DEFAULT_MODEL) -> str:
    """FLAP's published calibration (1,024 x 128 tokens of WikiText-2 train). Only for
    scripts/check_flap_reference.py, which checks our FLAP against the reference at its published settings."""
    return f"wikitext2_train_flap_calib__{slug(model_id)}"


def wanda_sp_reference_calib_name(model_id: str = DEFAULT_MODEL) -> str:
    """Wanda-sp's published calibration (128 x 128 tokens of C4). Only for scripts/check_wanda_sp_reference.py."""
    return f"c4_wanda_sp_calib__{slug(model_id)}"


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


def sample_stream(ids: torch.Tensor, nsamples: int, seqlen: int, seed: int) -> torch.Tensor:
    """FLAP `get_wikitext2`: `nsamples` random `seqlen` windows from one token stream [1, N]. int64 [nsamples, seqlen]."""
    rng = random.Random(seed)
    starts = [rng.randint(0, ids.shape[1] - seqlen - 1) for _ in range(nsamples)]
    return torch.stack([ids[0, i : i + seqlen] for i in starts]).to(torch.int64)


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

    wsp = sample_calibration(_Column(c4, "text"), tokenize, WANDA_SP_NSAMPLES, WANDA_SP_SEQLEN, seed)
    write_entry(
        out,
        wanda_sp_reference_calib_name(model_id),
        wsp,
        {
            "source": {"repo": C4_REPO, "revision": c4_rev, "data_files": C4_SHARD, "split": "train"},
            "params": {
                "nsamples": WANDA_SP_NSAMPLES, "seqlen": WANDA_SP_SEQLEN, "seed": seed,
                "procedure": "flap get_c4 (as wanda/sparsegpt), for wanda-sp",
            },
            "tokenizer": tokenizer_meta,
        },
    )
    log(f"  Wanda-sp reference calibration {tuple(wsp.shape)}")

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

    train = load_dataset(WIKITEXT_REPO, WIKITEXT_CONFIG, split="train", revision=wt_rev)
    stream = torch.tensor([tokenize(FLAP_JOIN.join(train["text"]))], dtype=torch.int64)
    flap = sample_stream(stream, FLAP_NSAMPLES, FLAP_SEQLEN, seed)
    write_entry(
        out,
        flap_reference_calib_name(model_id),
        flap,
        {
            "source": {"repo": WIKITEXT_REPO, "revision": wt_rev, "config": WIKITEXT_CONFIG, "split": "train"},
            "params": {
                "nsamples": FLAP_NSAMPLES, "seqlen": FLAP_SEQLEN, "seed": seed, "join": FLAP_JOIN,
                "procedure": "flap get_wikitext2",
            },
            "tokenizer": tokenizer_meta,
        },
    )
    log(f"  FLAP reference calibration {tuple(flap.shape)} from wikitext-2 train ({stream.shape[1]} tokens)")

    return read_manifest(out)


def build_c4_validation(
    model_id: str = DEFAULT_MODEL,
    nsamples: int = 128,
    seqlen: int = 2048,
    seed: int = 0,
    out: Path | None = None,
    revision: str | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Download C4's first validation shard (~40 MB), sample passages as the calibration set is sampled, cache them."""
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    out = out or cache_dir()
    api = HfApi()
    model_rev = revision or api.model_info(model_id).sha
    c4_rev = api.dataset_info(C4_REPO).sha
    tok = AutoTokenizer.from_pretrained(model_id, revision=model_rev)
    log(f"C4 validation shard {C4_VALIDATION_SHARD}@{c4_rev[:8]}")
    ds = load_dataset(C4_REPO, data_files={"validation": C4_VALIDATION_SHARD}, split="validation", revision=c4_rev)
    ids = sample_calibration(_Column(ds, "text"), lambda t: tok(t)["input_ids"], nsamples, seqlen, seed)
    entry = write_entry(
        out,
        c4_validation_name(model_id),
        ids,
        {
            "source": {"repo": C4_REPO, "revision": c4_rev, "data_files": C4_VALIDATION_SHARD, "split": "validation"},
            "params": {"nsamples": nsamples, "seqlen": seqlen, "seed": seed, "procedure": "wanda/sparsegpt get_c4"},
            "tokenizer": {"id": model_id, "revision": model_rev},
        },
    )
    log(f"  C4 validation {tuple(ids.shape)}")
    return entry


GATE_TEXTS = ("wikitext2", "c4")


def gate_text_name(gate_text: str, model_id: str = DEFAULT_MODEL) -> str:
    """The cached passages a run's keep-or-undo check (the gate) and the damage table measure on. "c4": C4
    validation passages, the calibration text's source (from 2026-11-21); "wikitext2": WikiText-2 validation
    (earlier runs), the source of the reported perplexity."""
    if gate_text == "c4":
        return c4_validation_name(model_id)
    if gate_text == "wikitext2":
        return wikitext_name("validation", model_id)
    raise ValueError(f"no gate text {gate_text!r}; choose one of {', '.join(GATE_TEXTS)}")


def gate_tokens(gate_text: str, model_id: str, out: Path | None = None, revision: str | None = None) -> torch.Tensor:
    """The gate's passages as one token stream [N] (C4 validation built on first use)."""
    if gate_text == "c4":
        return ensure_c4_validation(model_id, out, revision).reshape(-1)
    return load_tokens(gate_text_name(gate_text, model_id), out).reshape(-1)


def ensure_c4_validation(model_id: str, out: Path | None = None, revision: str | None = None) -> torch.Tensor:
    """The C4 validation passages, built on first use."""
    if c4_validation_name(model_id) not in read_manifest(out or cache_dir())["entries"]:
        build_c4_validation(model_id, out=out, revision=revision)
    return load_tokens(c4_validation_name(model_id), out)


class _Column(Sequence[str]):
    """Lazy view of one column of a Hugging Face dataset, so the C4 shard is not loaded into a list."""

    def __init__(self, ds, column: str):
        self._ds, self._col = ds, column

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, i):  # type: ignore[override]
        return self._ds[i][self._col]
