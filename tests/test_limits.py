"""Usage limits (task 2.12, FR15) with recorded limit readings: read, estimate (fits, warns), pause at 90% or on a
limit, resume without repeating a call. No real agent call."""

import json
import subprocess
from datetime import UTC, datetime, timedelta

import pytest
from test_graph import FakeBackend, fake_baseline, table
from test_runstore import CFG, OUTCOMES, essentials, policy, ppl, reference, run_config
from typer.testing import CliRunner

from pruneagent import agent_client as AC
from pruneagent import graph as G
from pruneagent import limits as L
from pruneagent import runstore as R
from pruneagent.cli import app

# Recorded 2026-11-17 from a saved session of `codex exec` (codex-cli 0.155.1, Plus plan, one-word reply). The
# credits balance is blanked; everything else is as Codex wrote it.
RECORDED_TOKEN_COUNT = {
    "timestamp": "2026-11-18T03:58:54.476Z", "type": "event_msg",
    "payload": {"type": "token_count",
                "info": {"total_token_usage": {"input_tokens": 3148, "cached_input_tokens": 0,
                                               "output_tokens": 5, "reasoning_output_tokens": 0,
                                               "total_tokens": 3153}},
                "rate_limits": {"limit_id": "codex", "limit_name": None,
                                "primary": {"used_percent": 7.0, "window_minutes": 300, "resets_at": 1794983552},
                                "secondary": {"used_percent": 1.0, "window_minutes": 10080,
                                              "resets_at": 1795384569},
                                "credits": {"has_credits": True, "unlimited": False, "balance": "0"},
                                "individual_limit": None, "spend_control_reached": None, "plan_type": "plus",
                                "rate_limit_reached_type": None}}}
# Also seen in saved sessions: a reading with no window at all, to be skipped.
RECORDED_PREMIUM = {
    "timestamp": "2026-11-18T03:58:55.000Z", "type": "event_msg",
    "payload": {"type": "token_count", "info": None,
                "rate_limits": {"limit_id": "premium", "limit_name": None, "primary": None, "secondary": None,
                                "credits": None, "individual_limit": None, "spend_control_reached": None,
                                "plan_type": "plus", "rate_limit_reached_type": None}}}
FIVE_HOUR_RESET = datetime.fromtimestamp(1794983552, UTC)  # 2026-11-18 06:32 UTC
WEEKLY_RESET = datetime.fromtimestamp(1795384569, UTC)
THREAD = "01a1ecce-b346-7c52-a8df-48c27cf7a88d"


def write_session(home, thread=THREAD, events=(RECORDED_TOKEN_COUNT, RECORDED_PREMIUM)):
    d = home / "sessions" / "2026" / "11" / "17"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"rollout-2026-11-17T22-58-51-{thread}.jsonl"
    meta = {"timestamp": "2026-11-18T03:58:51Z", "type": "session_meta", "payload": {"id": thread}}
    path.write_text("".join(json.dumps(e) + "\n" for e in (meta, *events)))
    return path


def reading(five_hour, weekly=1.0, at=None, reset=FIVE_HOUR_RESET, reached=None):
    rl = {"primary": {"used_percent": five_hour, "window_minutes": 300, "resets_at": int(reset.timestamp())},
          "secondary": {"used_percent": weekly, "window_minutes": 10080, "resets_at": int(WEEKLY_RESET.timestamp())},
          "rate_limit_reached_type": reached, "plan_type": "plus"}
    return L.from_rate_limits(rl, at or reset - timedelta(hours=4))


BEFORE = FIVE_HOUR_RESET - timedelta(hours=1)  # a time inside the recorded 5-hour window


# ----------------------------------------------------------------------------------------------- reading


def test_reads_the_recorded_session(tmp_path):
    path = write_session(tmp_path)
    assert L.find_session(tmp_path, THREAD) == path and L.find_session(tmp_path, "other") is None
    r = L.read_session(path)  # the windowless "premium" reading after it is skipped
    assert [(w.name, w.used_percent, w.resets_at) for w in r.windows] == [
        ("5-hour", 7.0, FIVE_HOUR_RESET), ("weekly", 1.0, WEEKLY_RESET)]
    assert r.reached is None and r.plan_type == "plus" and r.has_credits is True
    assert r.time == datetime(2026, 11, 18, 3, 58, 54, 476000, tzinfo=UTC)
    assert L.LimitReading.from_dict(json.loads(json.dumps(r.to_dict()))) == r  # as kept in the ledger
    assert "balance" not in json.dumps(r.to_dict())
    assert r.pause_reason(BEFORE) is None
    assert L.window_name(43200) == "30-day"  # the free plan's single window


def test_pause_point_reset_and_reached_flag():
    assert reading(89.0).pause_reason(BEFORE) is None
    why = reading(90.0).pause_reason(BEFORE)
    assert why.startswith("Codex 5-hour limit at 90% used (pause at 90%); resets at ")
    assert reading(90.0).resets_at(BEFORE) == FIVE_HOUR_RESET
    after = FIVE_HOUR_RESET + timedelta(minutes=1)
    assert reading(95.0).pause_reason(after) is None  # the window has reset
    weekly = reading(10.0, weekly=92.0)
    assert "weekly limit at 92%" in weekly.pause_reason(BEFORE) and weekly.resets_at(BEFORE) == WEEKLY_RESET
    hit = reading(40.0, reached="primary")
    assert hit.pause_reason(BEFORE).startswith("Codex usage limit reached (primary)")
    assert hit.resets_at(BEFORE) == FIVE_HOUR_RESET  # the window Codex names, not the fullest
    assert reading(100.0, reached="primary").pause_reason(after) is None  # it has reset since


# ---------------------------------------------------------------------------------------------- estimate


def test_estimate_fits():
    est = L.estimate_run(reading(7.0), 60, now=BEFORE)
    assert est.fits and est.calls_fit == 207  # (90 - 7) / 0.4% per call
    assert est.message().startswith("Usage limits: this run needs up to 60 agent calls (~6.1k tokens each")


def test_estimate_warns_with_the_reset_time_and_options():
    est = L.estimate_run(reading(80.0), 60, now=BEFORE)
    assert est.fits is False and est.calls_fit == 25
    msg = est.message()
    assert msg.startswith("Warning: this run may need up to 60 agent calls") and "only about 25 fit" in msg
    assert "5-hour limit reaches 90%" in msg and "resets" in msg and "API backend" in msg
    assert L.estimate_run(reading(80.0), 20, now=BEFORE).fits  # a shorter run fits
    after = FIVE_HOUR_RESET + timedelta(minutes=1)
    assert L.estimate_run(reading(80.0), 60, now=after).fits  # by then the window has reset


def test_estimate_learns_percent_per_token_from_readings():
    # 30 calls of 5k tokens each, 5-hour window rising 1% every 3 calls: 10% over 150k tokens
    t0 = BEFORE - timedelta(hours=3)
    history = [(5000, reading(20.0 + (i // 3), at=t0 + timedelta(minutes=i))) for i in range(31)]
    est = L.estimate_run(history[-1][1], 60, history, recent_tokens=[5000] * 30, now=BEFORE)
    five = est.windows[0]
    assert five.basis == "measured over 30 calls" and five.pct_per_call == pytest.approx(10 / 30)
    assert est.tokens_per_call == 5000 and est.windows[1].basis == "default"  # weekly never moved: not enough
    assert est.calls_fit == int((90 - 30) / (10 / 30))


def test_estimate_without_a_reading():
    est = L.estimate_run(None, 60)
    assert est.fits is None and "No Codex limit reading yet" in est.message()


# ------------------------------------------------------------------------------------- Codex backend


class SessionWritingRunner:
    """Stands in for `codex exec`: prints the recorded events and writes the session into $CODEX_HOME."""

    def __init__(self, events=(RECORDED_TOKEN_COUNT,), stdout_extra=(), returncode=0):
        self.events, self.extra, self.returncode = events, stdout_extra, returncode
        self.seen = []

    def __call__(self, cmd, **kw):
        from pathlib import Path

        self.seen.append((cmd, kw))
        write_session(Path(kw["env"]["CODEX_HOME"]), events=self.events)
        out = [{"type": "thread.started", "thread_id": THREAD}, *self.extra]
        if self.returncode == 0:
            out += [{"type": "item.completed", "item": {"type": "agent_message", "text": '{"a": 1}'}},
                    {"type": "turn.completed", "usage": {"input_tokens": 3148, "output_tokens": 5}}]
        return subprocess.CompletedProcess(cmd, self.returncode, "\n".join(map(json.dumps, out)), "")


def test_codex_backend_reads_the_limits_of_each_call(tmp_path):
    login = tmp_path / "login"
    login.mkdir()
    (login / "auth.json").write_text("{}")
    home = tmp_path / "agent-home"
    run = SessionWritingRunner()
    res = AC.CodexBackend(codex_home=home, login_home=login, run=run).complete("p", None, 60)
    assert res.limits.windows[0].used_percent == 7.0
    assert (home / "auth.json").is_symlink() and (home / "auth.json").resolve() == (login / "auth.json").resolve()
    assert run.seen[0][1]["env"]["CODEX_HOME"] == str(home)


def test_codex_backend_copies_a_refreshed_login_back(tmp_path):
    import os

    login = tmp_path / "login"
    login.mkdir()
    (login / "auth.json").write_text("old")
    home = tmp_path / "agent-home"
    b = AC.CodexBackend(codex_home=home, login_home=login)
    b.prepare_home()
    (home / "auth.json").unlink()  # Codex replaced the link with a refreshed login
    (home / "auth.json").write_text("new")
    os.utime(home / "auth.json", (2e9, 2e9))
    b.prepare_home()
    assert (login / "auth.json").read_text() == "new" and (home / "auth.json").is_symlink()


def test_codex_limit_error_carries_the_reset_time(tmp_path):
    hit = json.loads(json.dumps(RECORDED_TOKEN_COUNT))
    hit["payload"]["rate_limits"]["primary"]["used_percent"] = 100.0
    hit["payload"]["rate_limits"]["rate_limit_reached_type"] = "primary"
    failed = [{"type": "turn.failed", "error": {"message": "You've hit your usage limit."}}]
    run = SessionWritingRunner(events=(hit,), stdout_extra=failed, returncode=1)
    with pytest.raises(AC.UsageLimitReached) as e:
        AC.CodexBackend(codex_home=tmp_path / "h", run=run).complete("p", None, 60)
    assert e.value.resets_at == FIVE_HOUR_RESET and e.value.reading.reached == "primary"


# ------------------------------------------------------------------------------------- graph: fits, warns


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


class LimitedBackend(FakeBackend):
    """The fake agent of test_runstore, named codex, with a limit reading after every call: `percents[n]` of the
    5-hour window after call n (counted across instances through `counter`)."""

    name = "codex"

    def __init__(self, percents, counter, clock, fail_at=None):
        super().__init__(policy)
        self.percents, self.counter, self.clock, self.fail_at = percents, counter, clock, fail_at

    def complete(self, prompt, schema, timeout):
        self.counter["n"] += 1
        n = self.counter["n"]
        if n == self.fail_at:
            r = reading(100.0, at=self.clock(), reached="primary")
            raise AC.UsageLimitReached("You've hit your usage limit.", r.resets_at(self.clock()), r)
        res = super().complete(prompt, schema, timeout)
        reset = FIVE_HOUR_RESET if self.clock() < FIVE_HOUR_RESET else FIVE_HOUR_RESET + timedelta(hours=5)
        res.limits = reading(self.percents[n], at=self.clock(), reset=reset)
        return res


def limited_deps(tmp, store, percents, counter, clock, notes, fail_at=None):
    backend = LimitedBackend(percents, counter, clock, fail_at)
    client = AC.AgentClient(backend, cache=AC.Cache(tmp / "cache"), ledger=AC.Ledger(tmp / "use.jsonl"),
                            sleep=lambda s: None, now=clock)
    deps = G.Deps(config=CFG, agent=client, evaluate=ppl, baseline=fake_baseline, load_sensitivity=table,
                  store=store, notify=notes.append)
    return deps, backend


def seed_ledger(tmp, used, clock):
    AC.Ledger(tmp / "use.jsonl").append({"time": clock().isoformat(), "backend": "codex", "cached": False,
                                         "ok": True, "total_tokens": 6000,
                                         "limits": reading(used, at=clock()).to_dict()})


def test_run_that_fits_says_so_and_finishes(tmp_path):
    clock, notes = Clock(BEFORE), []
    seed_ledger(tmp_path, 10.0, clock)
    store = R.RunStore.create(run_config(), tmp_path / "runs", "r")
    deps, _backend = limited_deps(tmp_path, store, [None, 11, 12, 13, 14, 15, 16], {"n": 0}, clock, notes)
    state = G.run(deps, run_config())
    assert notes[0].startswith("Usage limits: this run needs up to 60 agent calls") and len(notes) == 1
    assert "paused" not in state and [r.outcome for r in state["history"]] == OUTCOMES
    assert store.info().state == "finished"


def test_run_that_does_not_fit_warns_then_starts(tmp_path):
    clock, notes = Clock(BEFORE), []
    seed_ledger(tmp_path, 80.0, clock)
    store = R.RunStore.create(run_config(), tmp_path / "runs", "r")
    deps, _ = limited_deps(tmp_path, store, [None, 81, 82, 83, 84, 85, 86], {"n": 0}, clock, notes)
    state = G.run(deps, run_config())
    assert notes[0].startswith("Warning: this run may need up to 60 agent calls (")
    assert "only about 25 fit" in notes[0]
    assert state["stop_reason"] == "memory limit met"  # a warning only: the run went ahead


# --------------------------------------------------------------------------------- graph: pause and resume


def test_pause_at_90_percent_then_resume_after_the_reset_without_repeating_a_call(tmp_path):
    ref_state, ref_store, ref_backend = reference(tmp_path)
    clock, notes, counter = Clock(BEFORE), [], {"n": 0}
    # 5-hour window after each call: round 4's call reaches 90%; calls 5 and 6 come after the reset
    percents = [None, 60, 70, 80, 90, 2, 3]
    store = R.RunStore.create(run_config(), tmp_path / "runs", "r")
    deps, first = limited_deps(tmp_path, store, percents, counter, clock, notes)
    state = G.run(deps, run_config())

    # paused cleanly after round 4, which finished and was saved
    assert state["paused"].startswith("Codex 5-hour limit at 90% used") and state["resets_at"] == FIVE_HOUR_RESET
    assert state["round"] == 4 and store.load_checkpoint()["round"] == 4 and len(first.prompts) == 4
    info = store.info()
    assert info.state == "paused" and "resets at" in info.reason and info.resumable
    assert notes[-1].startswith("Paused: Codex 5-hour limit at 90% used") and "pruneagent resume" in notes[-1]

    # resumed before the reset: pauses again at once, without a call
    store = R.RunStore(store.dir)
    deps, early = limited_deps(tmp_path, store, percents, counter, clock, notes)
    again = G.resume(deps)
    assert again["paused"] == state["paused"] and early.prompts == [] and store.info().state == "paused"
    assert store.load_checkpoint()["round"] == 4

    # resumed after the reset: finishes as the uninterrupted run did
    clock.t = FIVE_HOUR_RESET + timedelta(minutes=5)
    deps, second = limited_deps(tmp_path, store, percents, counter, clock, notes)
    final = G.resume(deps)
    assert "paused" not in final and final["stop_reason"] == ref_state["stop_reason"]
    assert final["plan"] == ref_state["plan"] and store.accepted() == ref_store.accepted()
    assert [essentials(r) for r in store.log()] == [essentials(r) for r in ref_store.log()]
    assert len(first.prompts) + len(second.prompts) == len(ref_backend.prompts) == 6  # no call repeated
    assert not set(first.prompts) & set(second.prompts)
    assert store.info().state == "finished"


def test_limit_error_pauses_and_the_round_is_replayed_on_resume(tmp_path):
    ref_state, ref_store, _ = reference(tmp_path)
    clock, notes, counter = Clock(BEFORE), [], {"n": 0}
    percents = [None, 20, 30, None, 40, 41, 42, 43]
    store = R.RunStore.create(run_config(), tmp_path / "runs", "r")
    deps, _first = limited_deps(tmp_path, store, percents, counter, clock, notes, fail_at=3)
    state = G.run(deps, run_config())
    assert state["paused"].startswith("You've hit your usage limit") and state["resets_at"] == FIVE_HOUR_RESET
    assert state["round"] == 2 and store.info().state == "paused"
    assert [r.round for r in store.log()] == [1, 2]  # the failed round left nothing behind

    clock.t = FIVE_HOUR_RESET + timedelta(minutes=5)
    deps, _second = limited_deps(tmp_path, store, percents, counter, clock, notes)
    final = G.resume(deps)
    assert final["stop_reason"] == "memory limit met" and final["plan"] == ref_state["plan"]
    assert [essentials(r) for r in store.log()] == [essentials(r) for r in ref_store.log()]
    answered = [e for e in AC.Ledger(tmp_path / "use.jsonl").entries() if e.get("ok") and not e.get("cached")]
    assert len(answered) == 6 and len({e["key"] for e in answered}) == 6  # every answered call made once


def test_api_cost_cap_pauses_the_run(tmp_path):
    class Priced(FakeBackend):
        name = "api"

        def complete(self, prompt, schema, timeout):
            res = super().complete(prompt, schema, timeout)
            res.usage.cost_usd = 0.4
            return res

    store = R.RunStore.create(run_config(agent_backend="api"), tmp_path / "runs", "r")
    client = AC.AgentClient(Priced(policy), cache=AC.Cache(tmp_path / "cache"),
                            ledger=AC.Ledger(tmp_path / "use.jsonl"), max_cost_usd=1.0)
    deps = G.Deps(config=CFG, agent=client, evaluate=ppl, baseline=fake_baseline, load_sensitivity=table,
                  store=store, notify=lambda m: None)
    state = G.run(deps, run_config(agent_backend="api"))
    assert state["paused"].startswith("API cost cap reached") and state["round"] == 3
    assert store.info().state == "paused"


# --------------------------------------------------------------------------------------------------- CLI

LIMITED = {"clock": None, "counter": None}


def limited_cli_deps(run_cfg, run_dir):
    tmp = run_dir.parent.parent
    deps, _ = limited_deps(tmp, None, [None, 60, 70, 80, 90, 2, 3], LIMITED["counter"], LIMITED["clock"], [])
    del deps.store
    return deps


def test_cli_resume_reports_a_pause_then_finishes(tmp_path):
    root = tmp_path / "runs"
    LIMITED.update(clock=Clock(BEFORE), counter={"n": 0})
    state = G.start_run(run_config(), "test_limits:limited_cli_deps", root, "a")
    assert state["paused"] and R.RunStore.open("a", root).info().state == "paused"
    out = CliRunner().invoke(app, ["runs", "--root", str(root)])
    assert "paused: Codex 5-hour limit at 90% used (pause at 90%); resets at" in out.output
    out = CliRunner().invoke(app, ["resume", "--root", str(root)])
    assert out.exit_code == 0 and "Paused after round 4: Codex 5-hour limit" in out.output
    LIMITED["clock"].t = FIVE_HOUR_RESET + timedelta(minutes=5)
    out = CliRunner().invoke(app, ["resume", "--root", str(root)])
    assert out.exit_code == 0 and "Finished after round 6: memory limit met" in out.output


def test_resume_on_the_other_backend_records_it(tmp_path):
    cfg = G.switch_backend(run_config(), "api")
    assert cfg.agent_backend == "api" and cfg.max_cost_usd == 5.0
    assert G.switch_backend(cfg, "codex").max_cost_usd is None
    store = R.RunStore.create(run_config(), tmp_path, "r")
    store.set_config(cfg)
    assert store.run_config().agent_backend == "api"
