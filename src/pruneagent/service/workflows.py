"""Background workflow service (docs/APP_DESIGN.md §4): long-running work, separate from the web app.

Model setup, pruning runs and checks run here as **workflows** made of **steps**. A step is one command, the same a
user would type; it saves its result before the next starts, so a workflow can stop after any step and continue.

- The web app and the service share only the workflow store (`runs/workflows.sqlite`): the app writes requests
  (`submit`, `request`, `resume`) and reads state; the service does the work. Restarting or updating the app never
  touches a running workflow.
- Each step runs as its own process group, so it also outlives a restart of the service; on start the service
  re-attaches to steps that are still running and reads the outcome of those that ended while it was down.
- Each workflow runs on the **release** it was submitted with (`releases/<commit>/`, made by `scripts/deploy.sh`): a
  fixed copy of the code with its own environment, so an update never changes a running workflow, AI instructions
  included. Without releases (development), steps run with `uv run` from the data root.
- One GPU step at a time, in submission order; steps also take the Mac GPU lock (`scripts/mac_gpu.sh`), which they
  share with commands started from a terminal.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

STORE = "workflows.sqlite"
LOGS = "workflow_logs"
HEARTBEAT = "workflow_service.json"
RELEASES = "releases"
CURRENT = "current"  # releases/current -> releases/<commit>
ACTIVE = ("waiting", "running")
SIGINT_CODES = (130, -signal.SIGINT)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS workflows (
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, params TEXT NOT NULL, state TEXT NOT NULL,
    reason TEXT, run_id TEXT, release TEXT, step INTEGER NOT NULL DEFAULT 0, want TEXT,
    created TEXT NOT NULL, updated TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS steps (
    workflow_id INTEGER NOT NULL, idx INTEGER NOT NULL, name TEXT NOT NULL, state TEXT NOT NULL,
    pid INTEGER, started TEXT, ended TEXT, exit_code INTEGER, log TEXT, attempts INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (workflow_id, idx));
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------- what each workflow does


@dataclass(frozen=True)
class Tools:
    """How a step calls the code: a release's own environment, or `uv run` in development."""

    release: Path | None = None

    def cli(self) -> list[str]:
        return [str(self.release / ".venv" / "bin" / "pruneagent")] if self.release else ["uv", "run", "pruneagent"]

    def script(self, name: str) -> list[str]:
        if self.release:
            return [str(self.release / ".venv" / "bin" / "python"), str(self.release / "scripts" / name)]
        return ["uv", "run", "python", f"scripts/{name}"]


@dataclass(frozen=True)
class StepCtx:
    root: Path  # the data root: runs/, checkpoints/, results/, data_cache/
    runs: Path
    params: dict
    tools: Tools
    continuing: bool  # this step ran before and stopped; continue it rather than start over
    exit_code: int | None  # None when the step ended while the service was down
    started: str | None
    log: str  # the step's output so far


@dataclass(frozen=True)
class StepDef:
    name: str
    gpu: bool
    argv: Callable[[StepCtx], list[str]]
    # (state, reason): done, paused or failed. The default reads the exit code.
    outcome: Callable[[StepCtx], tuple[str, str | None]] | None = None


def _last_line(text: str) -> str | None:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip() and not ln.startswith("[mac_gpu]")]
    return lines[-1][:300] if lines else None


def default_outcome(ctx: StepCtx) -> tuple[str, str | None]:
    if ctx.exit_code == 0:
        return "done", None
    if ctx.exit_code in SIGINT_CODES:
        return "paused", "Paused by you"
    if ctx.exit_code is None:
        return "failed", "Stopped while the background service was off"
    return "failed", _last_line(ctx.log) or f"exit code {ctx.exit_code}"


def _run_step_argv(ctx: StepCtx) -> list[str]:
    p = ctx.params
    if ctx.continuing or p.get("resume"):
        return [*ctx.tools.cli(), "resume", p["run_id"]]
    return [*ctx.tools.cli(), "run", "--model", p["model"], "--memory-cut", f"{float(p['cut']):.4f}", "--device",
            "mps", "--backend", p.get("backend", "codex"), "--run-id", p["run_id"]]


def _run_step_outcome(ctx: StepCtx) -> tuple[str, str | None]:
    """A run step is done when the run has finished (target met or not); a run that paused itself (usage limit) or
    was interrupted is paused, with the run's own reason."""
    from pruneagent import runstore as R

    try:
        info = R.RunStore.open(ctx.params["run_id"], ctx.runs).info()
    except (OSError, ValueError, KeyError):
        return default_outcome(ctx)
    if info.state == "finished":
        return "done", None
    if info.state == "paused":
        return "paused", info.reason or "Paused"
    if ctx.exit_code in SIGINT_CODES:
        return "paused", "Paused by you"
    if ctx.exit_code is None:
        return "paused", "Interrupted while the background service was off"
    return "failed", _last_line(ctx.log) or f"exit code {ctx.exit_code}"


def _check_step(kind: str) -> StepDef:
    script = {"heldout": "heldout_check.py", "accuracy": "accuracy_check.py"}[kind]
    return StepDef(kind, True, lambda c: [*c.tools.script(script), f"runs/{c.params['run_id']}"])


def _json_outcome(ctx: StepCtx) -> tuple[str, str | None]:
    if ctx.exit_code in (0, None) and last_json(ctx.log) is not None:
        return "done", None
    return default_outcome(ctx)


WORKFLOWS: dict[str, list[StepDef]] = {
    "setup": [StepDef("setup", True, lambda c: [*c.tools.cli(), "setup", c.params["model"]])],
    # run, then evaluate: the unseen-text test and the question sets, each against the original and FLAP
    "pruning": [StepDef("run", True, _run_step_argv, _run_step_outcome), _check_step("heldout"),
                _check_step("accuracy")],
    "check": [],  # one step, chosen by params["check"]: see steps_for
    "precheck": [StepDef("precheck", False, lambda c: [
        *c.tools.cli(), "run", "--model", c.params["model"], "--memory-cut", f"{float(c.params['cut']):.4f}",
        "--device", "mps", "--backend", c.params.get("backend", "codex"), "--dry-run", "--json-result"],
        _json_outcome)],
}


def steps_for(kind: str, params: dict) -> list[StepDef]:
    if kind == "check":
        return [_check_step(params["check"])]
    if kind not in WORKFLOWS:
        raise ValueError(f"unknown workflow: {kind}")
    return WORKFLOWS[kind]


def last_json(text: str) -> dict | None:
    """The last line of the output that is a JSON object (`--json-result`)."""
    for line in reversed(text.strip().splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                return None
    return None


# ---------------------------------------------------------------- the store (shared with the web app)


def current_release(root: Path) -> str | None:
    """The release new workflows use (`releases/current`, when complete), or None in development."""
    link = Path(root) / RELEASES / CURRENT
    return link.resolve().name if (link / ".ready").exists() else None


@dataclass
class Workflow:
    id: int
    kind: str
    params: dict
    state: str  # waiting, running, paused, finished, failed, cancelled
    reason: str | None
    run_id: str | None
    release: str | None
    step: int
    want: str | None
    created: str
    updated: str
    steps: list[dict] = field(default_factory=list)


class Store:
    def __init__(self, runs: Path | str):
        self.runs = Path(runs)
        self.runs.mkdir(parents=True, exist_ok=True)
        self.path = self.runs / STORE
        with closing(self._con()) as con, con:
            con.executescript(_SCHEMA)

    def _con(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def submit(self, kind: str, params: dict, release: str | None = None) -> Workflow:
        names = [s.name for s in steps_for(kind, params)]
        now = _now()
        with closing(self._con()) as con, con:
            cur = con.execute("INSERT INTO workflows (kind, params, state, run_id, release, created, updated) "
                              "VALUES (?, ?, 'waiting', ?, ?, ?, ?)",
                              (kind, json.dumps(params), params.get("run_id"), release, now, now))
            wid = cur.lastrowid
            con.executemany("INSERT INTO steps (workflow_id, idx, name, state) VALUES (?, ?, ?, 'waiting')",
                            [(wid, i, n) for i, n in enumerate(names)])
        return self.get(wid)

    def get(self, wid: int) -> Workflow:
        with closing(self._con()) as con:
            row = con.execute("SELECT * FROM workflows WHERE id = ?", (wid,)).fetchone()
            if row is None:
                raise KeyError(wid)
            steps = [dict(r) for r in con.execute("SELECT * FROM steps WHERE workflow_id = ? ORDER BY idx", (wid,))]
        return self._wf(row, steps)

    def list(self, limit: int = 100, states: tuple[str, ...] | None = None) -> list[Workflow]:
        q, args = "SELECT * FROM workflows", []
        if states:
            q += f" WHERE state IN ({','.join('?' * len(states))})"
            args = list(states)
        with closing(self._con()) as con:
            rows = con.execute(q + " ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
            out = []
            for row in rows:
                steps = [dict(r) for r in con.execute("SELECT * FROM steps WHERE workflow_id = ? ORDER BY idx",
                                                      (row["id"],))]
                out.append(self._wf(row, steps))
        return out

    @staticmethod
    def _wf(row, steps) -> Workflow:
        return Workflow(id=row["id"], kind=row["kind"], params=json.loads(row["params"]), state=row["state"],
                        reason=row["reason"], run_id=row["run_id"], release=row["release"], step=row["step"],
                        want=row["want"], created=row["created"], updated=row["updated"], steps=steps)

    def request(self, wid: int, want: str) -> Workflow:
        """Ask the service to pause or cancel a workflow. A waiting or paused one changes at once."""
        if want not in ("pause", "cancel"):
            raise ValueError(want)
        wf = self.get(wid)
        with closing(self._con()) as con, con:
            if wf.state == "running":
                con.execute("UPDATE workflows SET want = ?, updated = ? WHERE id = ?", (want, _now(), wid))
            elif wf.state == "waiting":
                state = "paused" if want == "pause" else "cancelled"
                con.execute("UPDATE workflows SET state = ?, reason = ?, updated = ? WHERE id = ?",
                            (state, "Paused by you" if want == "pause" else "Cancelled", _now(), wid))
            elif wf.state == "paused" and want == "cancel":
                con.execute("UPDATE workflows SET state = 'cancelled', reason = 'Cancelled', updated = ? WHERE id = ?",
                            (_now(), wid))
        return self.get(wid)

    def resume(self, wid: int) -> Workflow:
        """Continue a paused or failed workflow from the step it stopped at."""
        wf = self.get(wid)
        if wf.state in ("paused", "failed"):
            with closing(self._con()) as con, con:
                con.execute("UPDATE workflows SET state = 'waiting', reason = NULL, want = NULL, updated = ? "
                            "WHERE id = ?", (_now(), wid))
        return self.get(wid)

    def set(self, wid: int, **fields) -> None:
        fields["updated"] = _now()
        cols = ", ".join(f"{k} = ?" for k in fields)
        with closing(self._con()) as con, con:
            con.execute(f"UPDATE workflows SET {cols} WHERE id = ?", (*fields.values(), wid))

    def set_step(self, wid: int, idx: int, **fields) -> None:
        cols = ", ".join(f"{k} = ?" for k in fields)
        with closing(self._con()) as con, con:
            con.execute(f"UPDATE steps SET {cols} WHERE workflow_id = ? AND idx = ?", (*fields.values(), wid, idx))


def read_log(path: str | None, tail: int | None = None) -> str:
    if not path:
        return ""
    try:
        text = Path(path).read_text(errors="replace")
    except FileNotFoundError:
        return ""
    return text[-tail:] if tail else text


def service_status(runs: Path | str, stale_s: float = 30) -> dict:
    """Whether the background service is running (its heartbeat is recent)."""
    try:
        beat = json.loads((Path(runs) / HEARTBEAT).read_text())
        age = time.time() - beat["beat"]
    except (OSError, ValueError, KeyError):
        return {"running": False, "pid": None, "release": None}
    return {"running": age < stale_s and _alive(beat.get("pid")), "pid": beat.get("pid"),
            "release": beat.get("release")}


def _exit_code(log: str | None) -> int | None:
    try:
        return int(Path(f"{log}.exit").read_text().strip())
    except (OSError, ValueError, TypeError):
        return None


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:  # our own child that has ended (not yet reaped) counts as gone
        if os.waitpid(pid, os.WNOHANG)[0]:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---------------------------------------------------------------- the service


class Service:
    """Runs workflows from the store. `step()` does one pass; `serve()` loops until stopped."""

    def __init__(self, root: Path | str, runs: Path | str | None = None, poll: float = 1.0,
                 gpu_wrap: list[str] | None = None):
        self.root = Path(root).resolve()
        self.runs = Path(runs) if runs else self.root / "runs"
        self.store = Store(self.runs)
        self.logs = self.runs / LOGS
        self.logs.mkdir(parents=True, exist_ok=True)
        self.poll = poll
        # GPU steps wait for the Mac GPU lock (shared with the terminal) and keep the Mac awake
        self.gpu_wrap = gpu_wrap if gpu_wrap is not None else [str(self.root / "scripts" / "mac_gpu.sh"),
                                                                 "caffeinate", "-i"]
        self._procs: dict[tuple[int, int], subprocess.Popen] = {}
        self._signalled: set[tuple[int, int]] = set()
        self._stop = threading.Event()

    # -- one pass

    def step(self) -> None:
        self._beat()
        for wf in self.store.list(limit=500, states=("running",)):
            self._watch(wf)
        self._start_next()

    def _tools(self, wf: Workflow) -> Tools:
        return Tools(self.root / RELEASES / wf.release if wf.release else None)

    def _ctx(self, wf: Workflow, idx: int, exit_code: int | None) -> StepCtx:
        st = wf.steps[idx]
        return StepCtx(root=self.root, runs=self.runs, params=wf.params, tools=self._tools(wf),
                       continuing=st["attempts"] > 1, exit_code=exit_code, started=st["started"],
                       log=read_log(st["log"], tail=20000))

    def _watch(self, wf: Workflow) -> None:
        idx = wf.step
        st = wf.steps[idx]
        key = (wf.id, idx)
        proc = self._procs.get(key)
        if proc is not None:
            code = proc.poll()
            alive = code is None
        else:  # started by an earlier service: re-attach by its process id; the step wrote its exit code
            alive, code = _alive(st["pid"]), _exit_code(st["log"])
        if alive:
            if wf.want and key not in self._signalled and st["pid"]:
                try:
                    os.killpg(st["pid"], signal.SIGINT)  # the step checkpoints and stops cleanly
                except (ProcessLookupError, PermissionError):
                    pass
                self._signalled.add(key)
            return
        self._procs.pop(key, None)
        self._signalled.discard(key)
        self._finish(wf, idx, code)

    def _finish(self, wf: Workflow, idx: int, code: int | None) -> None:
        sdef = steps_for(wf.kind, wf.params)[idx]
        if code is None:  # ended while no service watched it: it wrote its own exit code
            code = _exit_code(wf.steps[idx]["log"])
        state, reason = (sdef.outcome or default_outcome)(self._ctx(wf, idx, code))
        if wf.want == "cancel":
            state, reason = "cancelled", "Cancelled"
        elif wf.want == "pause" and state != "done":
            state, reason = "paused", "Paused by you"
        self.store.set_step(wf.id, idx, state=state if state != "cancelled" else "paused", ended=_now(), exit_code=code)
        if state == "done":
            last = idx + 1 >= len(wf.steps)
            if last:
                self.store.set(wf.id, state="finished", reason=None, want=None)
            elif wf.want == "pause":
                self.store.set(wf.id, state="paused", reason="Paused by you", step=idx + 1, want=None)
            else:
                self.store.set(wf.id, state="waiting", step=idx + 1, want=None)
        else:
            self.store.set(wf.id, state=state, reason=reason, want=None)

    def _start_next(self) -> None:
        active = self.store.list(limit=500, states=ACTIVE)
        gpu_busy = any(w.state == "running" and steps_for(w.kind, w.params)[w.step].gpu for w in active)
        for wf in sorted((w for w in active if w.state == "waiting"), key=lambda w: w.id):
            sdef = steps_for(wf.kind, wf.params)[wf.step]
            if sdef.gpu and gpu_busy:
                continue  # waits its turn; earlier GPU workflows go first
            self._launch(wf, sdef)
            gpu_busy = gpu_busy or sdef.gpu

    def _launch(self, wf: Workflow, sdef: StepDef) -> None:
        idx = wf.step
        attempts = wf.steps[idx]["attempts"] + 1
        log = self.logs / f"{wf.id:05d}-{idx}-{sdef.name}.log"
        ctx = StepCtx(root=self.root, runs=self.runs, params=wf.params, tools=self._tools(wf),
                      continuing=attempts > 1, exit_code=None, started=None, log="")
        argv = (self.gpu_wrap if sdef.gpu else []) + sdef.argv(ctx)
        # the step writes its own exit code, so a restarted service still learns how it ended; the INT trap lets
        # Ctrl-C reach the step and still records its code
        exit_file = Path(f"{log}.exit")
        exit_file.unlink(missing_ok=True)
        argv = ["/bin/bash", "-c", 'trap "true" INT; "$@"; c=$?; echo $c > "$0"; exit $c', str(exit_file), *argv]
        release = self._tools(wf).release
        if release is not None and not (release / ".ready").exists():
            self.store.set_step(wf.id, idx, state="failed", log=str(log), attempts=attempts, ended=_now())
            self.store.set(wf.id, state="failed", reason=f"Release {wf.release} is missing; run scripts/deploy.sh")
            return
        with open(log, "a") as out:
            out.write(f"$ {' '.join(argv)}\n")
            out.flush()
            try:
                proc = subprocess.Popen(argv, cwd=self.root, stdout=out, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, start_new_session=True)
            except OSError as e:  # one bad step never stops the service
                self.store.set_step(wf.id, idx, state="failed", log=str(log), attempts=attempts, ended=_now())
                self.store.set(wf.id, state="failed", reason=f"Could not start: {e}")
                return
        self._procs[(wf.id, idx)] = proc
        self.store.set_step(wf.id, idx, state="running", pid=proc.pid, started=_now(), ended=None, exit_code=None,
                            log=str(log), attempts=attempts)
        self.store.set(wf.id, state="running", reason=None)

    def _beat(self) -> None:
        (self.runs / HEARTBEAT).write_text(json.dumps({"pid": os.getpid(), "beat": time.time(),
                                                       "release": current_release(self.root)}))

    # -- loop

    def serve(self) -> None:
        if threading.current_thread() is threading.main_thread():  # in the tools service it runs in a thread
            signal.signal(signal.SIGTERM, lambda *_: self._stop.set())
        import traceback

        while not self._stop.is_set():
            try:
                self.step()
            except Exception:  # noqa: BLE001  a bad pass is logged; the service keeps running
                traceback.print_exc()
            self._stop.wait(self.poll)

    def stop(self) -> None:
        self._stop.set()
