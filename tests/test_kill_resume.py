"""Task 3.3: a run killed part-way and resumed ends exactly where an uninterrupted run ends.

Same tiny random Qwen3 and real `pruneagent run` pieces as test_realrun.py (offline, CPU), with a deterministic fake agent
whose answer depends only on how many answers it has given. Each test makes one uninterrupted run and one run that is
killed (Ctrl-C) and then resumed with `pruneagent resume`, and compares round by round: outcome, chosen candidate,
every candidate's measured perplexity and memory, and the final plan. Two kill points:
- during an agent call (the answer is never received, so the round's call is made again on resume);
- while candidates are being measured (the answer is cached, so resume re-measures without a new call).
"""

import json

import pytest
from test_realrun import N_LAYERS, make_data, make_model, make_probes, reply
from transformers import AutoConfig
from typer.testing import CliRunner

from pruneagent import agent_client as AC
from pruneagent import candidate_eval as CE
from pruneagent import realrun as RR
from pruneagent import runstore as R
from pruneagent.cli import app
from pruneagent.schema import Calibration


class Agent:
    """Answer k narrows layer k % N_LAYERS, then the next one. `kill_on=n` raises Ctrl-C on the n-th call instead of
    answering (without counting it), as if the run were killed while waiting for the agent."""

    name = "codex"
    model = "fake"

    def __init__(self, kill_on=None):
        self.answers, self.kill_on, self.calls = 0, kill_on, 0

    def identity(self):
        return {"backend": "fake", "model": self.model}

    def complete(self, prompt, schema, timeout):
        self.calls += 1
        if self.calls == self.kill_on:
            raise KeyboardInterrupt
        k = self.answers
        self.answers += 1
        layers = (k % N_LAYERS, (k + 1) % N_LAYERS)
        return AC.BackendResult(
            text=json.dumps(reply(*layers)),
            usage=AC.Usage(input_tokens=10, output_tokens=5),
            model=self.model,
        )


@pytest.fixture
def setup_for(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = str(make_model(tmp_path / "tiny-qwen3"))
    data = make_data(tmp_path / "data", model, (8, 32))
    probes = make_probes(tmp_path / "probes.json")
    run_cfg = RR.make_run_config(
        model,
        AutoConfig.from_pretrained(model),
        "cpu",
        memory_cut=0.10,
        dtype="float32",
        calibration=Calibration(n_samples=8, seqlen=32),
        gate_windows=4,
    )

    def make(name):
        # each run gets its own agent cache, so the uninterrupted run cannot feed the resumed one
        setup = RR.make_setup(
            model, "cpu", "float32", data, scores=tmp_path / "scores.safetensors", probes=probes
        )
        return run_cfg, setup, tmp_path / name

    return make


def use(monkeypatch, cache_dir, agent):
    def make_agent(run_cfg, run_id):
        return AC.AgentClient(
            agent,
            cache=AC.Cache(cache_dir / "cache"),
            ledger=AC.Ledger(cache_dir / "use.jsonl"),
            run_id=run_id,
            sleep=lambda s: None,
        )

    monkeypatch.setattr(RR, "make_agent", make_agent)


def summary(store):
    rounds = [
        (
            r.round,
            r.outcome,
            r.gate.chosen_candidate if r.gate else None,
            [(round(c.val_ppl, 6), c.freed_bytes) for c in r.candidates],
        )
        for r in store.log()
    ]
    final = store.accepted()[-1]
    return rounds, json.dumps(final.plan, sort_keys=True, default=str), final.weight_bytes, store.info().state


def uninterrupted(setup_for, monkeypatch):
    run_cfg, setup, root = setup_for("plain")
    use(monkeypatch, root, Agent())
    RR.start(run_cfg, setup, root / "runs", "r")
    return summary(R.RunStore.open("r", root / "runs"))


def killed_then_resumed(setup_for, monkeypatch, agent, kill_eval_on=None):
    run_cfg, setup, root = setup_for("killed")
    use(monkeypatch, root, agent)
    if kill_eval_on is not None:
        real, n = CE.CandidateEvaluator.__call__, {"calls": 0}

        def call(self, *a, **k):
            n["calls"] += 1
            if n["calls"] == kill_eval_on:
                raise KeyboardInterrupt
            return real(self, *a, **k)

        monkeypatch.setattr(CE.CandidateEvaluator, "__call__", call)
    with pytest.raises(KeyboardInterrupt):
        RR.start(run_cfg, setup, root / "runs", "r")
    store = R.RunStore.open("r", root / "runs")
    assert store.info().state == "paused"
    killed_at = store.info().round
    monkeypatch.undo()  # drop the evaluator patch (and the agent); put the same agent back
    monkeypatch.chdir(root.parent)
    use(monkeypatch, root, agent)
    out = CliRunner().invoke(app, ["resume", "--root", str(root / "runs")])
    assert out.exit_code == 0, out.output
    return summary(store), killed_at


def test_killed_during_an_agent_call_resumes_to_the_same_result(setup_for, monkeypatch):
    agent = Agent(kill_on=3)
    resumed, killed_at = killed_then_resumed(setup_for, monkeypatch, agent)
    assert killed_at == 2  # two rounds finished before the kill
    assert resumed == uninterrupted(setup_for, monkeypatch)
    assert resumed[3] == "finished"


def test_killed_while_measuring_candidates_resumes_without_a_new_agent_call(setup_for, monkeypatch):
    agent = Agent()
    resumed, killed_at = killed_then_resumed(setup_for, monkeypatch, agent, kill_eval_on=6)
    assert killed_at >= 1  # killed after at least one finished round, inside a later one
    plain = uninterrupted(setup_for, monkeypatch)
    assert resumed == plain
    assert agent.calls == len(plain[0])  # one call per round: the killed round's answer came from the cache


def test_the_comparison_notices_a_different_run(setup_for, monkeypatch):
    """Guard: the round-by-round comparison is strict enough to tell two different runs apart."""
    plain = uninterrupted(setup_for, monkeypatch)
    assert len(plain[0]) >= 3 and plain[3] == "finished"
    run_cfg, setup, root = setup_for("other")
    agent = Agent()
    agent.answers = 1  # starts one layer later
    use(monkeypatch, root, agent)
    RR.start(run_cfg, setup, root / "runs", "r")
    assert summary(R.RunStore.open("r", root / "runs")) != plain
