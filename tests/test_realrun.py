"""`pruneagent run` (realrun.py) on a tiny random Qwen3 saved to disk, offline, on the CPU: the dry run renders round
1's prompt without an agent call, a real run builds every part (computing the scores), and `pruneagent resume`
rebuilds the run from its recorded factory. The agent is a fake backend behind the real `AgentClient`."""

import json
import os

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from typer.testing import CliRunner

from pruneagent import agent_client as AC
from pruneagent import data as d
from pruneagent import realrun as RR
from pruneagent import runstore as R
from pruneagent.cli import app
from pruneagent.schema import Calibration, SensitivityTable

N_LAYERS = 4


def make_model(path):
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=256, num_hidden_layers=N_LAYERS,
                      num_attention_heads=8, num_key_value_heads=4, head_dim=16, max_position_embeddings=2048)
    Qwen3ForCausalLM(cfg).eval().save_pretrained(path)
    return path


def make_data(path, model_id, calib_shape):
    torch.manual_seed(1)
    meta = {"tokenizer": {"revision": None}}
    d.write_entry(path, d.calib_name(model_id), torch.randint(0, 97, calib_shape), meta)
    d.write_entry(path, d.wikitext_name("validation", model_id), torch.randint(0, 97, (1, 4000)), meta)
    return path


def make_probes(path):
    table = {str(i): {"layer": i, "block_influence": 0.1 * (i + 1), "candidates": [
        {"strategy": s, "steps": 1, "setting": None, "local_error": 0.01 * (i + 1)}
        for s in ("narrow_mlp", "remove_attention_group", "drop_layer", "wanda_sparsity")]} for i in range(N_LAYERS)}
    path.write_text(json.dumps({"table": table, "ground_truth": {"dense": {"ppl": 99.0}}}))
    return path


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # runs/, the agent cache and the ledger go here
    model = str(make_model(tmp_path / "tiny-qwen3"))
    return {"model": model, "probes": make_probes(tmp_path / "probes.json"), "tmp": tmp_path}


def reply(*choices):
    return {"action": "apply", "rationale": "narrow", "probe": [], "rollback_to": None,
            "candidates": [{"choices": [{"layer": la, "strategy": "narrow_mlp"}], "rationale": "c"}
                           for la in choices]}


class FakeBackend:
    """Narrows one MLP per round, on the next layer; `fail_on` raises Ctrl-C on that call."""

    name = "codex"
    model = "fake"

    def __init__(self, fail_on=None):
        self.calls, self.fail_on = 0, fail_on

    def identity(self):
        return {"backend": "fake", "model": self.model}

    def complete(self, prompt, schema, timeout):
        self.calls += 1
        if self.calls == self.fail_on:
            raise KeyboardInterrupt
        layer = (self.calls - 1) % N_LAYERS
        return AC.BackendResult(text=json.dumps(reply(layer)), usage=AC.Usage(input_tokens=10, output_tokens=5),
                                model=self.model)


def use_fake_agent(monkeypatch, tmp_path, backend):
    def make_agent(run_cfg, run_id):
        return AC.AgentClient(backend, cache=AC.Cache(tmp_path / "cache"), ledger=AC.Ledger(tmp_path / "use.jsonl"),
                              run_id=run_id, sleep=lambda s: None)

    monkeypatch.setattr(RR, "make_agent", make_agent)


# ---------------------------------------------------------------------------------------------------- dry run


def test_dry_run_renders_round_one_without_an_agent_call_or_scores(tiny, monkeypatch):
    data = make_data(tiny["tmp"] / "data", tiny["model"], (128, 2048))  # the run's calibration: C4 128 x 2,048

    def no_call(*a, **k):
        raise AssertionError("a dry run must not call the agent")

    monkeypatch.setattr(AC.CodexBackend, "complete", no_call)
    out = CliRunner().invoke(app, ["run", "--model", tiny["model"], "--memory-cut", "0.10", "--device", "cpu",
                                   "--dry-run", "--data-dir", str(data), "--probes", str(tiny["probes"])])
    assert out.exit_code == 0, out.output
    assert "Round 1 prompt" in out.output
    assert "not cached" in out.output  # scores: a dry run never computes them
    assert "no measured per-layer damage" in out.output
    assert not (tiny["tmp"] / "runs").exists() or not any(p.is_dir() for p in (tiny["tmp"] / "runs").iterdir())
    assert not (tiny["tmp"] / "checkpoints").exists()


def test_dry_run_reports_an_early_stop_instead_of_a_prompt(tiny):
    data = make_data(tiny["tmp"] / "data", tiny["model"], (128, 2048))
    out = CliRunner().invoke(app, ["run", "--model", tiny["model"], "--memory-limit-mb", "1000", "--device", "cpu",
                                   "--dry-run", "--data-dir", str(data), "--probes", str(tiny["probes"])])
    assert out.exit_code == 0, out.output
    assert "No agent call would be made" in out.output


def test_memory_limit_needs_exactly_one_of_mb_and_cut(tiny):
    data = make_data(tiny["tmp"] / "data", tiny["model"], (128, 2048))
    args = ["run", "--model", tiny["model"], "--device", "cpu", "--dry-run", "--data-dir", str(data)]
    for extra in ([], ["--memory-cut", "0.1", "--memory-limit-mb", "1"]):
        out = CliRunner().invoke(app, args + extra)
        assert out.exit_code == 1 and "exactly one" in out.output


def test_a_wrong_calibration_set_is_refused(tiny):
    data = make_data(tiny["tmp"] / "data", tiny["model"], (8, 32))
    out = CliRunner().invoke(app, ["run", "--model", tiny["model"], "--memory-cut", "0.1", "--device", "cpu",
                                   "--dry-run", "--data-dir", str(data), "--probes", str(tiny["probes"])])
    assert out.exit_code == 1 and "128 x 2048" in out.output


# ------------------------------------------------------------------------------------------- run and resume


def test_real_run_on_a_tiny_model_then_resume_from_the_recorded_factory(tiny, monkeypatch):
    tmp = tiny["tmp"]
    data = make_data(tmp / "data", tiny["model"], (8, 32))
    setup = RR.make_setup(tiny["model"], "cpu", "float32", data, scores=tmp / "scores.safetensors",
                          probes=tiny["probes"])
    from transformers import AutoConfig

    run_cfg = RR.make_run_config(tiny["model"], AutoConfig.from_pretrained(tiny["model"]), "cpu", memory_cut=0.10,
                                 dtype="float32", calibration=Calibration(n_samples=8, seqlen=32), gate_windows=4)
    assert run_cfg.agent_model == AC.CODEX_MODEL and run_cfg.agent_reasoning_effort == AC.CODEX_REASONING_EFFORT

    use_fake_agent(monkeypatch, tmp, FakeBackend(fail_on=2))
    with pytest.raises(KeyboardInterrupt):
        RR.start(run_cfg, setup, tmp / "runs", "r")
    store = R.RunStore.open("r", tmp / "runs")
    assert store.info().deps_factory == RR.FACTORY
    assert store.info().state == "paused" and store.info().round == 1
    assert RR.Setup.load(store.dir) == setup
    assert (tmp / "scores.safetensors").exists()  # computed on the first start

    use_fake_agent(monkeypatch, tmp, FakeBackend())
    out = CliRunner().invoke(app, ["resume", "--root", str(tmp / "runs")])
    assert out.exit_code == 0, out.output
    assert "memory limit met" in out.output
    info = store.info()
    assert info.state == "finished" and info.n_accepted >= 2
    log = store.log()
    assert log[0].outcome == "ACCEPT" and all(r.candidates[0].val_ppl > 0 for r in log if r.candidates)


def test_the_mac_gpu_needs_the_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    with pytest.raises(RR.SetupError, match="mac_gpu.sh"):
        RR.check_gpu_lock("mps")
    RR.check_gpu_lock("cpu")  # only the Mac GPU is shared
    lock = tmp_path / "pruneagent-mac-gpu.lock"
    lock.mkdir()
    (lock / "pid").write_text(str(os.getpid()))
    RR.check_gpu_lock("mps")


def test_sensitivity_uses_quick_tests_and_the_measured_table_when_present(tmp_path):
    probes = make_probes(tmp_path / "probes.json")
    notes = []
    table = RR.load_sensitivity(probes, tmp_path / "missing.json", notes.append)
    assert set(table.rows) == set(range(N_LAYERS)) and table.dense_ppl is None  # the probe file's own measurement
    assert all(not r.measured_ppl for r in table.rows.values()) and "no measured" in notes[0]

    damage = tmp_path / "damage.json"
    damage.write_text(json.dumps({"dense": {"ppl": 10.0}, "windows": 32, "entries": {
        "0/narrow_mlp": {"layer": 0, "strategy": "narrow_mlp", "ppl": 10.5}}}))
    notes.clear()
    table = RR.load_sensitivity(probes, damage, notes.append)
    if hasattr(SensitivityTable, "with_measured_damage"):  # the damage-table loader is merged
        assert table.dense_ppl == 10.0 and table.rows[0].measured_ppl == {"narrow_mlp": 10.5}
        assert any("incomplete" in n for n in notes)
    else:
        assert "cannot read it yet" in notes[0]


def test_a_missing_prompt_builder_falls_back_to_the_placeholder():
    notes = []
    assert RR.prompt_builder("pruneagent.nothing_here:build_prompt", notes.append) is None
    assert "placeholder" in notes[0]
    assert RR.prompt_builder("pruneagent.graph:placeholder_prompt") is not None
