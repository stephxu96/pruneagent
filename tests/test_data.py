"""Tests for the data cache. Offline except the one marked network test (set PRUNEAGENT_NETWORK_TESTS=1)."""

import json
import os
import random

import pytest
import torch

from pruneagent import data as d


def fake_tokenize(text: str) -> list[int]:
    return [ord(c) for c in text]


def reference_get_c4(texts, tokenize, nsamples, seqlen, seed):
    """The Wanda / SparseGPT loop as written in their repos (global `random`, same calls in the same order)."""
    random.seed(seed)
    out = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(texts) - 1)
            ids = tokenize(texts[i])
            if len(ids) > seqlen:
                break
        i = random.randint(0, len(ids) - seqlen - 1)
        out.append(ids[i : i + seqlen])
    return out


@pytest.fixture
def texts():
    rng = random.Random(1)
    return ["".join(chr(97 + rng.randrange(26)) for _ in range(rng.randrange(5, 60))) for _ in range(200)]


def test_sampling_matches_reference(texts):
    got = d.sample_calibration(texts, fake_tokenize, nsamples=16, seqlen=20, seed=0)
    want = reference_get_c4(texts, fake_tokenize, 16, 20, 0)
    assert got.tolist() == want
    assert got.shape == (16, 20) and got.dtype == torch.int64


def test_sampling_skips_short_documents(texts):
    got = d.sample_calibration(texts, fake_tokenize, nsamples=32, seqlen=40, seed=3)
    assert got.shape == (32, 40)


def test_sampling_depends_on_seed(texts):
    a = d.sample_calibration(texts, fake_tokenize, 8, 20, seed=0)
    b = d.sample_calibration(texts, fake_tokenize, 8, 20, seed=1)
    assert not torch.equal(a, b)


def test_wikitext_join():
    ids = d.tokenize_joined(["a", "b"], fake_tokenize)
    assert ids.tolist() == [[ord("a"), 10, 10, ord("b")]]


def test_manifest_roundtrip(tmp_path):
    t = torch.arange(12, dtype=torch.int64).reshape(3, 4)
    entry = d.write_entry(tmp_path, "x", t, {"source": {"repo": "r"}})
    assert entry["shape"] == [3, 4] and entry["dtype"] == "int64" and len(entry["sha256"]) == 64
    assert torch.equal(d.load_tokens("x", tmp_path), t)
    assert d.verify(tmp_path) == {"x": True}
    manifest = json.loads((tmp_path / d.MANIFEST).read_text())
    assert "torch" in manifest["libraries"]


def test_same_tensor_same_hash(tmp_path):
    t = torch.arange(10, dtype=torch.int64)
    h1 = d.write_entry(tmp_path / "a", "x", t, {})["sha256"]
    h2 = d.write_entry(tmp_path / "b", "x", t, {})["sha256"]
    assert h1 == h2


def test_tampered_file_fails(tmp_path):
    d.write_entry(tmp_path, "x", torch.zeros(4, dtype=torch.int64), {})
    path = tmp_path / "x.safetensors"
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(bytes(raw))
    assert d.verify(tmp_path) == {"x": False}
    with pytest.raises(d.CacheError, match="SHA-256"):
        d.load_tokens("x", tmp_path)


def test_missing_entry(tmp_path):
    with pytest.raises(d.CacheError, match="not in"):
        d.load_tokens("nope", tmp_path)


@pytest.mark.skipif(not os.environ.get("PRUNEAGENT_NETWORK_TESTS"), reason="set PRUNEAGENT_NETWORK_TESTS=1")
def test_real_cache_verifies():
    result = d.verify()
    assert result and all(result.values())


def test_wanda_sp_sampling_matches_flap_get_c4(texts):
    """FLAP lib/data.py `get_c4` (commit 3bb57db), verbatim apart from the dataset rows and tokenizer."""

    def tokenizer(text, return_tensors):
        return type("Enc", (), {"input_ids": torch.tensor([fake_tokenize(text)])})

    traindata = [{"text": t} for t in texts]
    nsamples, seed, seqlen = 12, 0, 16
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        trainloader.append(inp)
    got = d.sample_calibration(texts, fake_tokenize, nsamples, seqlen, seed)
    assert torch.equal(got, torch.cat(trainloader))


def test_stream_sampling_matches_flap_reference():

    ids = torch.arange(1000).unsqueeze(0)
    random.seed(0)  # FLAP get_wikitext2, verbatim
    ref = []
    for _ in range(5):
        i = random.randint(0, ids.shape[1] - 16 - 1)
        ref.append(ids[:, i : i + 16])
    assert torch.equal(d.sample_stream(ids, 5, 16, 0), torch.cat(ref))
