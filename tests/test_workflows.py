"""Task 3.11: the background workflow service runs long work apart from the web app (docs/APP_DESIGN.md §4)."""

import sys
import time
from pathlib import Path

import pytest

from pruneagent.service import workflows as W

PY = sys.executable


def py(code: str):
    return lambda ctx: [PY, "-c", code]


@pytest.fixture
def service(tmp_path, monkeypatch):
    marker = tmp_path / "marker"
    monkeypatch.setitem(W.WORKFLOWS, "two", [W.StepDef("first", False, py("print('one')")),
                                             W.StepDef("second", False, py("print('two')"))])
    monkeypatch.setitem(W.WORKFLOWS, "fails", [W.StepDef("boom", False, py("import sys; print('no space left'); sys.exit(3)"))])
    monkeypatch.setitem(W.WORKFLOWS, "slow", [W.StepDef("sleep", False, lambda c: [
        PY, "-c", f"import time; print({'continuing' if c.continuing else 'fresh'!r}, flush=True); time.sleep(30)"])])
    monkeypatch.setitem(W.WORKFLOWS, "gpu", [W.StepDef("work", True, py("import time; time.sleep(1.5)"))])
    monkeypatch.setitem(W.WORKFLOWS, "marked", [W.StepDef(
        "work", False, lambda c: [PY, "-c", f"import time; time.sleep(1.5); open({str(marker)!r}, 'w').write('ok')"],
        lambda c: ("done", None) if marker.exists() else ("failed", "no marker"))])
    return W.Service(tmp_path, tmp_path / "runs", poll=0.05, gpu_wrap=[])


def run_until(svc, wid, states, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        svc.step()
        wf = svc.store.get(wid)
        if wf.state in states:
            return wf
        time.sleep(0.05)
    raise AssertionError(f"workflow {wid} stuck in {svc.store.get(wid).state}")


def test_steps_run_in_order_and_the_workflow_finishes(service):
    wf = service.store.submit("two", {})
    done = run_until(service, wf.id, ("finished",))
    assert [s["state"] for s in done.steps] == ["done", "done"]
    assert "two" in W.read_log(done.steps[1]["log"])


def test_a_failed_step_says_why_in_one_line(service):
    wf = service.store.submit("fails", {})
    out = run_until(service, wf.id, ("failed",))
    assert out.reason == "no space left" and out.steps[0]["exit_code"] == 3


def test_pause_stops_the_step_and_continue_resumes_it(service):
    wf = service.store.submit("slow", {})
    run_until(service, wf.id, ("running",))
    time.sleep(0.3)
    service.store.request(wf.id, "pause")
    paused = run_until(service, wf.id, ("paused",))
    assert paused.reason == "Paused by you"
    service.store.resume(wf.id)
    again = run_until(service, wf.id, ("running",))
    time.sleep(0.5)
    assert again.steps[0]["attempts"] == 2 and "continuing" in W.read_log(again.steps[0]["log"])
    service.store.request(wf.id, "cancel")
    assert run_until(service, wf.id, ("cancelled",)).reason == "Cancelled"


def test_one_gpu_step_at_a_time_in_submission_order(service):
    a, b = service.store.submit("gpu", {}), service.store.submit("gpu", {})
    service.step()
    assert service.store.get(a.id).state == "running" and service.store.get(b.id).state == "waiting"
    run_until(service, a.id, ("finished",))
    run_until(service, b.id, ("finished",))


def test_waiting_work_can_be_cancelled_at_once(service):
    a, b = service.store.submit("gpu", {}), service.store.submit("gpu", {})
    service.step()
    assert service.store.request(b.id, "cancel").state == "cancelled"
    run_until(service, a.id, ("finished",))


def test_a_restarted_service_picks_up_a_step_that_is_still_running(service, tmp_path):
    wf = service.store.submit("marked", {})
    service.step()  # launches the step, then this service "stops"
    assert service.store.get(wf.id).state == "running"
    fresh = W.Service(tmp_path, tmp_path / "runs", poll=0.05, gpu_wrap=[])
    fresh.step()
    assert fresh.store.get(wf.id).state == "running"  # still alive: watched, not restarted
    out = run_until(fresh, wf.id, ("finished", "failed"))
    assert out.state == "finished" and out.steps[0]["attempts"] == 1


def test_pruning_runs_the_final_check_after_the_run(tmp_path):
    steps = W.steps_for("pruning", {"model": "Qwen/Qwen3-0.6B-Base", "cut": 0.2, "run_id": "r1"})
    assert [s.name for s in steps] == ["run", "heldout", "accuracy"]
    ctx = W.StepCtx(tmp_path, tmp_path, {"model": "m/x", "cut": 0.2, "run_id": "r1"}, W.Tools(tmp_path / "rel"),
                    continuing=False, exit_code=None, started=None, log="")
    argv = steps[0].argv(ctx)
    assert argv[0].endswith("rel/.venv/bin/pruneagent") and argv[-2:] == ["--run-id", "r1"]
    cont = W.StepCtx(**{**ctx.__dict__, "continuing": True})
    assert steps[0].argv(cont)[1:] == ["resume", "r1"]
    assert steps[1].argv(ctx)[-1] == "runs/r1" and steps[1].argv(ctx)[1].endswith("rel/scripts/heldout_check.py")


def test_service_status_reads_the_heartbeat(service):
    assert not W.service_status(service.runs)["running"]
    service.step()
    assert W.service_status(service.runs)["running"]


def test_a_step_that_ended_while_the_service_was_off_reports_its_own_exit_code(service, tmp_path):
    wf = service.store.submit("fails", {})
    service.step()  # launches, then this service "stops" before seeing the end
    log = service.store.get(wf.id).steps[0]["log"]
    for _ in range(100):
        if Path(f"{log}.exit").exists():
            break
        time.sleep(0.05)
    fresh = W.Service(tmp_path, tmp_path / "runs", poll=0.05, gpu_wrap=[])
    out = run_until(fresh, wf.id, ("failed", "finished"))
    assert out.state == "failed" and out.steps[0]["exit_code"] == 3 and out.reason == "no space left"


def test_a_step_that_cannot_start_fails_without_stopping_the_service(service, tmp_path, monkeypatch):
    monkeypatch.setitem(W.WORKFLOWS, "nope", [W.StepDef("x", False, lambda c: [str(tmp_path / "missing-program")])])
    service.gpu_wrap = []
    wf = service.store.submit("nope", {}, release="abc")  # a release that was never built
    service.step()
    out = service.store.get(wf.id)
    assert out.state == "failed" and "Release abc is missing" in out.reason
    ok = service.store.submit("two", {})
    assert run_until(service, ok.id, ("finished",)).state == "finished"
