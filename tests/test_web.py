"""Task 3.6: the web application's API (ADR-008) reads tool outputs and runs fixed commands only."""

import json
import signal
import sys
import time

import pytest
from fastapi.testclient import TestClient
from test_kill_resume import Agent, setup_for, use  # noqa: F401  (setup_for is a fixture)

from pruneagent import realrun as RR
from pruneagent.web import jobs as J
from pruneagent.web import machine as M
from pruneagent.web.app import create_app


@pytest.fixture
def finished_run(setup_for, monkeypatch):  # noqa: F811
    """One real (tiny, CPU) run with a fake agent, plus a held-out file as the check would write it."""
    run_cfg, setup, root = setup_for("web")
    use(monkeypatch, root, Agent())
    RR.start(run_cfg, setup, root / "runs", "r1")
    held = {"results": {name: {"weight_bytes": b, "val_unseen_ppl": p + 1, "test_ppl": p}
                        for name, b, p in [("unpruned", 2_000_000, 10.0), ("agent", 1_800_000, 12.0),
                                           ("Baseline: FLAP", 1_800_000, 15.0)]}}
    (root / "runs" / "r1" / "heldout.json").write_text(json.dumps(held))
    return root / "runs"


def client(runs, tmp_path, build=J.command):
    runner = J.JobRunner(tmp_path / "jobs.sqlite", tmp_path, tmp_path / "logs", build=build)
    return TestClient(create_app(root=runs, repo=tmp_path, runner=runner, dist=None, start_worker=False)), runner


def test_runs_detail_and_results_come_from_the_run_files(finished_run, tmp_path):
    api, _ = client(finished_run, tmp_path)
    runs = api.get("/api/runs").json()
    assert [r["id"] for r in runs] == ["r1"] and runs[0]["state"] == "finished"
    d = api.get("/api/runs/r1").json()
    assert d["shape"]["layers"] == len(d["steps"][0]["layers"])
    assert len(d["rounds"]) == d["round"] and d["steps"][0]["round"] == 0
    kept = [r for r in d["rounds"] if r["outcome"] == "ACCEPT"]
    assert len(kept) == len(d["steps"]) - 1  # every kept step is one accepted round
    assert d["now_mb"] <= d["limit_mb"]
    res = api.get("/api/runs/r1/results").json()
    assert res["heldout"]["agent"]["test_error"] == 12.0 and res["accuracy"] is None
    comp = api.get("/api/compare").json()
    assert comp["runs"][0]["flap"]["test_error"] == 15.0
    assert api.get("/api/runs/nope").status_code == 404
    assert api.get("/api/runs/..%2Fetc").status_code == 404


def test_actions_become_fixed_commands_only(finished_run, tmp_path):
    api, runner = client(finished_run, tmp_path)
    job = api.post("/api/runs", json={"cut": 0.2}).json()
    argv = runner.get(job["id"]).argv
    assert argv.startswith("scripts/mac_gpu.sh caffeinate -i uv run pruneagent run")
    assert "--memory-cut 0.2000" in argv and "--backend codex" in argv
    assert "--dry-run" in runner.get(api.post("/api/precheck", json={"cut": 0.1}).json()["id"]).argv
    assert api.post("/api/runs", json={"cut": 0.2, "model": "evil; rm -rf /"}).status_code == 422
    assert api.post("/api/runs", json={"cut": 5}).status_code == 422
    held = api.post("/api/runs/r1/checks/heldout").json()
    assert runner.get(held["id"]).argv.endswith("python scripts/heldout_check.py runs/r1")
    assert api.post("/api/runs/r1/checks/rm").status_code == 404
    assert api.post("/api/runs/r1/resume").status_code == 409  # finished runs are not resumed
    assert api.post("/api/runs/r1/pause").status_code == 409


def test_job_runner_runs_in_order_and_stop_sends_ctrl_c(tmp_path):
    script = ("import signal,sys,time\n"
              "signal.signal(signal.SIGINT, lambda *a: (print('paused'), sys.exit(3)))\n"
              "print('started', flush=True)\ntime.sleep(30)\n")

    def build(kind, params):
        return [sys.executable, "-c", "print('ok')"] if kind == "quick" else [sys.executable, str(tmp_path / "s.py")]

    (tmp_path / "s.py").write_text(script)
    runner = J.JobRunner(tmp_path / "jobs.sqlite", tmp_path, tmp_path / "logs", build=build, poll=0.05)
    a, b = runner.submit("quick", {}), runner.submit("slow", {})
    assert runner.step() and runner.get(a.id).status == "done"
    assert "ok" in runner.get(a.id).public()["output"]
    runner.start()
    for _ in range(100):
        if "started" in runner.get(b.id).public()["output"]:
            break
        time.sleep(0.05)
    runner.stop(b.id)
    for _ in range(100):
        if runner.get(b.id).status != "running":
            break
        time.sleep(0.05)
    runner.shutdown()
    job = runner.get(b.id)
    assert job.status == "stopped" and "paused" in job.public()["output"] and job.exit_code == 3
    assert signal.SIGINT  # Ctrl-C, not a kill


def test_a_job_running_when_the_server_stopped_is_marked_lost(tmp_path):
    runner = J.JobRunner(tmp_path / "jobs.sqlite", tmp_path, tmp_path / "logs", build=lambda k, p: ["true"])
    job = runner.submit("x", {})
    runner._set(job.id, status="running")
    again = J.JobRunner(tmp_path / "jobs.sqlite", tmp_path, tmp_path / "logs", build=lambda k, p: ["true"])
    assert again.get(job.id).status == "lost"


def test_machine_reports_memory_and_model_sizes(finished_run, tmp_path, monkeypatch):
    from pruneagent.web import machine as M

    monkeypatch.setattr(M, "machine", lambda: {"name": "MacBook Pro", "chip": "Apple M4", "memory_gb": 24.0})
    monkeypatch.setattr(M, "model_sizes", lambda ids=None: [{"id": J.MODELS[0], "ready": True, "size_mb": 1192.1, "working_mb": 117.4}])
    api, _ = client(finished_run, tmp_path)
    got = api.get("/api/machine").json()
    assert got["machine"]["memory_gb"] == 24.0 and got["models"][0]["size_mb"] == 1192.1 and got["max_cut"] == 0.6 and got["free_share"] == 0.5


def test_this_machine_is_read_for_real():
    from pruneagent.web import machine as M

    m = M.machine()
    assert m["memory_gb"] and m["memory_gb"] > 1


def test_live_usage_is_read_on_this_mac():
    import platform

    from pruneagent.web import machine as M

    got = M.live()
    assert set(got) == {"memory_used_pct", "gpu_busy_pct", "gpu_memory_mb"}
    if platform.system() == "Darwin":
        assert 0 <= got["memory_used_pct"] <= 100


def test_feasibility_result_is_read_from_the_last_json_line():
    out = 'Loading ...\nnotes\n{"feasible": false, "stop_reason": "limit below the smallest model allowed"}\n'
    assert J._last_json(out) == {"feasible": False, "stop_reason": "limit below the smallest model allowed"}
    assert J._last_json("no json here") is None
    assert J.command("precheck", {"cut": 0.2})[-2:] == ["--dry-run", "--json-result"]


def test_past_rates_give_time_per_mb_from_finished_runs(finished_run):
    from pruneagent import runstore as R

    rates = R.past_rates(finished_run, R.RunStore.open("r1", finished_run).run_config().model_id)
    assert len(rates) == 1 and rates[0]["rounds_per_mb"] > 0 and rates[0]["minutes_per_mb"] >= 0
    assert R.past_rates(finished_run, "some/other-model") == []


def test_time_estimate_gives_a_range_and_the_round_cap():
    from pruneagent import runstore as R

    rates = [{"minutes_per_mb": 0.25, "rounds_per_mb": 0.08, "minutes_per_round": 2.5},
             {"minutes_per_mb": 0.32, "rounds_per_mb": 0.12, "minutes_per_round": 3.0}]
    t = R.time_estimate(rates, 200, 40)
    assert t["minutes"] == [50, 64] and t["worst_minutes"] == 120 and t["max_rounds"] == 40
    assert R.time_estimate(rates, 1000, 40)["minutes"] == [120, 120]  # capped by the rounds
    assert R.time_estimate([], 200, 40) is None


def test_time_left_reads_a_runs_own_progress(finished_run):
    from pruneagent import runstore as R

    t = R.time_left(R.RunStore.open("r1", finished_run), [])
    assert t is not None and t["mb_left"] >= 0 and t["rounds_left"] >= 0 and t["worst_minutes"] >= 0



def test_prepare_jobs_check_the_name_and_report_progress():
    assert J.command("prepare", {"model": "HuggingFaceTB/SmolLM2-360M"})[-3:] == ["pruneagent", "prepare",
                                                                                  "HuggingFaceTB/SmolLM2-360M"]
    for bad in ("", "no-slash", "a/b; rm -rf ~", "../x/y"):
        with pytest.raises(J.JobError):
            J.command("prepare", {"model": bad})
    # a run may use an added model only when the app passes it as known
    with pytest.raises(J.JobError):
        J.command("start", {"model": "HuggingFaceTB/SmolLM2-360M", "cut": 0.2})
    assert "HuggingFaceTB/SmolLM2-360M" in J.command("start", {"model": "HuggingFaceTB/SmolLM2-360M", "cut": 0.2},
                                                     known=[*J.MODELS, "HuggingFaceTB/SmolLM2-360M"])
    log = ("Step 1 of 3: text\nStep 2 of 3: neuron scores and quick tests\n[3/16] probe\n"
           "Step 3 of 3: damage table\n  [1/64] layer 0 narrow_mlp\n  [32/64] layer 7 drop_layer\n")
    p = J.prepare_progress(log)
    assert (p["step"], p["done"], p["total"]) == (3, 32, 64) and abs(p["share"] - 0.55) < 1e-9
    assert J.prepare_progress("")["step"] == 0


def test_models_api_checks_and_prepares(finished_run, tmp_path, monkeypatch):
    from pruneagent import models as MD

    monkeypatch.setattr(M, "machine", lambda: {"name": "Mac", "chip": "M4", "memory_gb": 16.0})
    monkeypatch.setattr(M, "ready", lambda m, *a: False)

    def fake_check(model_id, memory_gb):
        if model_id != "org/good-model":
            raise MD.ModelError("not supported")
        return {"id": model_id, "family": "Llama", "size_mb": 720.0, "layers": 32, "params_b": 0.36,
                "prepare_minutes": [100.0, 160.0]}

    monkeypatch.setattr(MD, "check", fake_check)
    api, _ = client(finished_run, tmp_path)
    assert api.get("/api/models/check", params={"id": "org/bad"}).json() == {"id": "org/bad", "ok": False,
                                                                           "reason": "not supported"}
    ok = api.get("/api/models/check", params={"id": "org/good-model"}).json()
    assert ok["ok"] and not ok["prepared"] and not ok["added"]
    assert api.post("/api/models/prepare", json={"model": "org/bad"}).status_code == 422
    job = api.post("/api/models/prepare", json={"model": "org/good-model"}).json()
    assert job["kind"] == "prepare" and job["progress"]["step"] == 0
    assert MD.known(finished_run)[-1] == "org/good-model"
