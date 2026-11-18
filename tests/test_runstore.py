"""Checkpoints, rollback, decision log and resume (tasks 2.7, 2.8) with fakes only: the fake agent backend, fake
evaluator and fake Baseline: FLAP of test_graph. No model, GPU or real agent call."""

import json
import re
import sqlite3
import subprocess
import sys
from contextlib import closing

import pytest
from test_graph import CFG, FakeBackend, fake_baseline, fake_ppl, reply, run_config, table
from typer.testing import CliRunner

from pruneagent import agent_client as AC
from pruneagent import graph as G
from pruneagent import plan as P
from pruneagent import runstore as R
from pruneagent.cli import app
from pruneagent.schema import AcceptedPlan, StepRecord


class Killed(BaseException):
    """Stands in for the process being stopped (like KeyboardInterrupt, not an ordinary error)."""


def round_of(prompt: str) -> int:
    return int(re.findall(r'"round":(\d+)\}', prompt)[-1])  # the request's own round is its last field


# Round by round: ACCEPT; ROLLBACK by the gate (attention groups cost +50%); ACCEPT; the agent rolls back to the
# first accepted step; ACCEPT; ACCEPT, which reaches the memory limit.
SCRIPT = {1: reply([(0, "narrow_mlp")], [(0, "remove_attention_group")]),
          2: reply([(1, "remove_attention_group")], [(2, "remove_attention_group")]),
          3: reply([(1, "narrow_mlp")]),
          4: reply(action="rollback", rollback_to=1),
          5: reply([(2, "narrow_mlp")]),
          6: reply([(3, "narrow_mlp")])}
OUTCOMES = ["ACCEPT", "ROLLBACK", "ACCEPT", None, "ACCEPT", "ACCEPT"]


def narrowed(*layers):
    """The unpruned plan with one narrow_mlp step on each of `layers`."""
    plan = P.dense_plan(CFG)
    for i in layers:
        plan = P.step(plan, i, "narrow_mlp", CFG)
    return plan


def policy(n, prompt):
    return SCRIPT[round_of(prompt)]


def ppl(plan) -> float:
    return fake_ppl(plan) * (1.5 if any(p["groups_keep"] < 4 for p in plan.values()) else 1.0)


def make_deps(tmp, store, evaluate=ppl, run_id=None):
    backend = FakeBackend(policy)
    client = AC.AgentClient(backend, cache=AC.Cache(tmp / "cache"), ledger=AC.Ledger(tmp / "use.jsonl"),
                            sleep=lambda s: None, run_id=run_id)
    deps = G.Deps(config=CFG, agent=client, evaluate=evaluate, baseline=fake_baseline, load_sensitivity=table,
                  store=store)
    return deps, backend


class KillingStore(R.RunStore):
    """Dies at the checkpoint of `kill_round`: that round's log line and plan changes are already written."""

    kill_round = None

    def checkpoint(self, state):
        if state.get("round") == self.kill_round:
            raise Killed()
        super().checkpoint(state)


def reference(tmp_path):
    store = R.RunStore.create(run_config(), tmp_path / "ref_runs", "ref")
    deps, backend = make_deps(tmp_path / "ref", store)
    return G.run(deps, run_config()), store, backend


def essentials(rec: StepRecord) -> dict:
    """A log record without the timings, which differ between runs."""
    return rec.model_dump(exclude={"llm_seconds", "gpu_seconds"})


# ------------------------------------------------------------------------------------------- kill and resume


@pytest.mark.parametrize("kill", ["evaluate 5", "checkpoint 3", "checkpoint 4"])
def test_kill_and_resume_matches_an_uninterrupted_run(tmp_path, kill):
    ref_state, ref_store, ref_backend = reference(tmp_path)
    assert [r.outcome for r in ref_state["history"]] == OUTCOMES
    assert ref_state["stop_reason"] == "memory limit met"

    where, at = kill.split()
    store = R.RunStore.create(run_config(), tmp_path / "runs", "r")
    armed = {"on": True}

    def evaluate(plan):  # dies evaluating round 5's candidate, after the round's agent call
        if where == "evaluate" and armed["on"] and plan == narrowed(0, 2):
            raise Killed()
        return ppl(plan)

    if where == "checkpoint":
        store = KillingStore(store.dir)
        store.kill_round = int(at)
    deps, first = make_deps(tmp_path, store, evaluate)
    with pytest.raises(Killed):
        G.run(deps, run_config())
    info = store.info()
    assert info.state == "paused" and info.reason == "Killed" and info.resumable
    last = {"evaluate 5": 4, "checkpoint 3": 2, "checkpoint 4": 3}[kill]  # the last finished round
    assert store.load_checkpoint()["round"] == last
    if where == "checkpoint":  # the killed round's log line was written before its checkpoint
        assert [r.round for r in store.log()][-1] == last + 1

    armed["on"] = False
    store = R.RunStore(store.dir)  # a new process: plain store
    deps, second = make_deps(tmp_path, store, evaluate)
    state = G.resume(deps)

    # same result as the uninterrupted run
    assert state["plan"] == ref_state["plan"]
    assert state["accepted"] == ref_state["accepted"] and store.accepted() == ref_store.accepted()
    assert state["stop_reason"] == ref_state["stop_reason"]
    assert [essentials(r) for r in state["history"]] == [essentials(r) for r in ref_state["history"]]
    # the decision log has each round once, as in the uninterrupted run
    assert [essentials(r) for r in store.log()] == [essentials(r) for r in ref_store.log()]
    assert [r.round for r in store.log()] == [1, 2, 3, 4, 5, 6]
    # no agent call repeated: the killed round's call came from the cache
    assert len(first.prompts) + len(second.prompts) == len(ref_backend.prompts) == 6
    assert not set(first.prompts) & set(second.prompts)
    assert store.info().state == "finished" and store.info().reason == "memory limit met"


def test_resume_before_the_first_checkpoint_starts_over(tmp_path):
    store = KillingStore(R.RunStore.create(run_config(), tmp_path / "runs", "r").dir)
    store.kill_round = 1
    deps, _ = make_deps(tmp_path, store)
    with pytest.raises(Killed):
        G.run(deps, run_config())
    assert store.load_checkpoint() is None and len(store.log()) == 1
    deps, second = make_deps(tmp_path, R.RunStore(store.dir))
    state = G.resume(deps)
    assert [r.outcome for r in state["history"]] == OUTCOMES
    assert [r.round for r in deps.store.log()] == [1, 2, 3, 4, 5, 6]
    assert len(second.prompts) == 5  # round 1's reply was cached


def test_memory_store_resumes_too(tmp_path):
    store = G.MemoryStore()
    armed = {"on": True}

    def evaluate(plan):
        if armed["on"] and plan == narrowed(0, 1):
            raise Killed()
        return ppl(plan)

    deps, _ = make_deps(tmp_path, store, evaluate)
    with pytest.raises(Killed):
        G.run(deps, run_config())
    assert store.status == ("paused", "Killed")
    armed["on"] = False
    deps, second = make_deps(tmp_path, store, evaluate)
    state = G.resume(deps)
    assert [r.outcome for r in state["history"]] == OUTCOMES and [r.round for r in store.log] == [1, 2, 3, 4, 5, 6]
    assert len(second.prompts) == 3  # rounds 4-6; round 3's reply was cached


# ------------------------------------------------------------------------------------------------- rollback


def test_rollback_restores_the_exact_earlier_plan_and_discards_later_ones(tmp_path):
    store = R.RunStore.create(run_config(), tmp_path / "runs", "r")
    plans = [P.dense_plan(CFG)]
    for layer in (0, 1, 2):
        plans.append(P.step(plans[-1], layer, "narrow_mlp", CFG))
    for i, plan in enumerate(plans):
        store.save_accepted(i, AcceptedPlan(round=i, plan=plan, val_ppl=fake_ppl(plan), weight_bytes=1000 + i))

    restored = R.RunStore(store.dir).restore(1)  # read back from disk, as after a restart
    assert restored.plan == plans[1] and all(isinstance(k, int) for k in restored.plan)
    assert fake_ppl(restored.plan) == restored.val_ppl == fake_ppl(plans[1])  # rebuilds to the same perplexity
    assert [a.plan for a in store.accepted()] == plans[:2]
    assert sorted(p.name for p in (store.dir / R.PLANS).iterdir()) == ["accepted-000.json", "accepted-001.json"]
    with pytest.raises(KeyError):
        store.restore(2)
    # the next accepted plan takes index 2 again
    store.save_accepted(2, AcceptedPlan(round=5, plan=plans[3], val_ppl=11.0, weight_bytes=999))
    assert [a.round for a in store.accepted()] == [0, 1, 5]


def test_agent_rollback_in_a_run_uses_the_saved_plans(tmp_path):
    state, store, _ = reference(tmp_path)
    rb = state["history"][3]
    assert rb.action == "rollback" and rb.reply.rollback_to == 1
    # after the rollback, round 5 started from accepted plan 1 (layer 0 narrowed), not from round 3's plan
    acc = store.accepted()
    assert [a.round for a in acc] == [0, 1, 5, 6]
    assert acc[2].plan == narrowed(0, 2)
    assert all(ppl(a.plan) == a.val_ppl for a in acc)  # each saved plan rebuilds to its measured perplexity


# ------------------------------------------------------------------------------------------- decision log


def test_decision_log_and_plan_files(tmp_path):
    state, store, _ = reference(tmp_path)
    lines = store.log_path.read_text().splitlines()
    assert len(lines) == 6
    recs = [json.loads(line) for line in lines]
    assert [r["round"] for r in recs] == [1, 2, 3, 4, 5, 6]
    assert [r["outcome"] for r in recs] == OUTCOMES
    assert [r["action"] for r in recs] == ["apply", "apply", "apply", "rollback", "apply", "apply"]
    first = recs[0]
    assert set(first) == set(StepRecord.model_fields)
    assert first["rationale"] == "apply for a reason" and first["gate"] == {"decision": "ACCEPT",
                                                                              "chosen_candidate": 0}
    assert first["strategies_used"] == {"narrow_mlp": 1, "remove_attention_group": 1}
    assert first["llm_tokens"] == 120 and first["llm_cost_usd"] is None  # Codex backend: no cost
    c0 = first["candidates"][0]
    assert c0["val_ppl"] == pytest.approx(10.2) and c0["estimate"]["weight_bytes"] > 0
    assert c0["plan"]["0"]["mlp_keep"] == 576  # layer keys are strings in JSON
    assert recs[1]["gate"] == {"decision": "ROLLBACK", "chosen_candidate": None}
    assert recs[3]["reply"]["rollback_to"] == 1 and recs[3]["candidates"] == []
    assert store.log() == state["history"]

    plan_file = json.loads((store.dir / R.PLANS / "accepted-001.json").read_text())
    assert plan_file["format"] == R.PLAN_FORMAT and plan_file["index"] == 1 and plan_file["round"] == 1
    assert "weights" not in plan_file and set(plan_file["plan"]) == {"0", "1", "2", "3"}

    info = store.info()
    assert info.status == "finished" and info.round == 6 and info.n_accepted == 4
    assert store.load_checkpoint()["plan"] == state["plan"]
    assert store.run_config() == run_config()


def test_run_ids_are_readable_and_unique(tmp_path):
    from datetime import datetime

    cfg = run_config().model_copy(update={"model_id": "Qwen/Qwen3-0.6B-Base"})
    when = datetime(2026, 11, 16, 14, 32)
    assert R.make_run_id(cfg, tmp_path, when) == "20261116-1432-qwen3-0.6b-codex"
    (tmp_path / "20261116-1432-qwen3-0.6b-codex").mkdir()
    assert R.make_run_id(cfg, tmp_path, when) == "20261116-1432-qwen3-0.6b-codex-2"


# ------------------------------------------------------------------------------------- runs and resume (CLI)

ARMED = {"on": False}


def cli_deps(run_cfg, run_dir):
    """The deps factory recorded with each CLI-test run: dies in round 3 while `ARMED` is on."""

    def evaluate(plan):
        if ARMED["on"] and plan == narrowed(0, 1):
            raise Killed()
        return ppl(plan)

    deps, _ = make_deps(run_dir.parent.parent, None, evaluate, run_id=run_dir.name)
    return deps


def paused_run(root, run_id):
    ARMED["on"] = True
    with pytest.raises(Killed):
        G.start_run(run_config(), "test_runstore:cli_deps", root, run_id)
    ARMED["on"] = False


def test_resume_without_an_id_takes_the_only_paused_run(tmp_path):
    root = tmp_path / "runs"
    G.start_run(run_config(), "test_runstore:cli_deps", root, "done")  # finished: not a candidate
    paused_run(root, "a")
    out = CliRunner().invoke(app, ["runs", "--root", str(root)])
    assert out.exit_code == 0
    assert re.search(r"a\s+2\s+2\s+.*paused: Killed", out.output)
    assert re.search(r"done\s+6\s+4\s+.*finished: memory limit met", out.output)

    out = CliRunner().invoke(app, ["resume", "--root", str(root)])
    assert out.exit_code == 0, out.output
    assert "memory limit met" in out.output
    assert R.RunStore.open("a", root).info().state == "finished"
    assert [r.outcome for r in R.RunStore.open("a", root).log()] == OUTCOMES

    out = CliRunner().invoke(app, ["resume", "--root", str(root)])
    assert out.exit_code == 1 and "no paused run" in out.output


def test_resume_without_an_id_refuses_when_several_runs_are_paused(tmp_path):
    root = tmp_path / "runs"
    paused_run(root, "a")
    paused_run(root, "b")
    out = CliRunner().invoke(app, ["resume", "--root", str(root)])
    assert out.exit_code == 1
    assert "2 runs are paused" in out.output and "  a  (paused: Killed, round 2)" in out.output
    assert "  b  (paused: Killed, round 2)" in out.output
    assert R.RunStore.open("a", root).info().state == "paused"  # nothing was resumed

    out = CliRunner().invoke(app, ["resume", "b", "--root", str(root)])
    assert out.exit_code == 0, out.output
    assert R.RunStore.open("b", root).info().state == "finished"
    assert R.RunStore.open("a", root).info().state == "paused"

    out = CliRunner().invoke(app, ["resume", "b", "--root", str(root)])
    assert out.exit_code == 1 and "finished: memory limit met" in out.output


def test_a_run_whose_process_died_counts_as_interrupted(tmp_path):
    store = R.RunStore.create(run_config(), tmp_path / "runs", "r")
    assert store.info().state == "running" and not store.info().resumable  # this process: alive
    dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], check=True, capture_output=True,
                          text=True)
    with closing(sqlite3.connect(store.dir / R.DB)) as con, con:
        con.execute("UPDATE run SET pid = ?", (int(dead.stdout),))
    assert store.info().state == "interrupted" and store.info().resumable
    assert R.pick_run(tmp_path / "runs").run_id == "r"
