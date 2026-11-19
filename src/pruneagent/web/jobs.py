"""Job runner for the web application (ADR-008): runs the same commands a user would type, one at a time.

Each job is one fixed command built here from a few checked fields; the browser never sends a command line.
Jobs that use the GPU go through `scripts/mac_gpu.sh`, so they also wait for anything started from a terminal.
The queue lives in SQLite next to the runs, so it survives a server restart (a job that was running when the
server stopped is marked `lost`; the run itself can be resumed as usual).
"""

from __future__ import annotations

import json
import os
import re
import signal
import sqlite3
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

MODELS = ("Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base")
MAX_CUT = 0.6  # largest size cut the app offers; the run's own early warning says if it is reachable
RUN_ID = re.compile(r"^[A-Za-z0-9._-]{1,120}$")
GPU = ["scripts/mac_gpu.sh"]
UV = ["uv", "run"]


class JobError(ValueError):
    """A job request that does not pass the checks."""


def _run_id(run_id: str) -> str:
    if not RUN_ID.match(run_id or ""):
        raise JobError("not a run id")
    return run_id


def command(kind: str, params: dict) -> list[str]:
    """The command line for one job. Only these kinds exist; every field is checked."""
    if kind in ("start", "precheck"):
        model = params.get("model", MODELS[0])
        if model not in MODELS:
            raise JobError(f"model must be one of {', '.join(MODELS)}")
        try:
            cut = float(params["cut"])
        except (KeyError, TypeError, ValueError):
            raise JobError("cut must be a number, e.g. 0.10") from None
        if not 0.01 <= cut <= MAX_CUT:
            raise JobError(f"cut must be between 1% and {MAX_CUT:.0%}")
        backend = params.get("backend", "codex")
        if backend not in ("codex", "api"):
            raise JobError("backend must be codex or api")
        args = ["pruneagent", "run", "--model", model, "--memory-cut", f"{cut:.4f}", "--device", "mps",
                "--backend", backend]
        if kind == "precheck":
            return [*UV, *args, "--dry-run", "--json-result"]
        return [*GPU, "caffeinate", "-i", *UV, *args]
    if kind == "resume":
        return [*GPU, "caffeinate", "-i", *UV, "pruneagent", "resume", _run_id(params.get("run_id", ""))]
    if kind in ("heldout", "accuracy"):
        script = {"heldout": "scripts/heldout_check.py", "accuracy": "scripts/accuracy_check.py"}[kind]
        return [*GPU, *UV, "python", script, f"runs/{_run_id(params.get('run_id', ''))}"]
    raise JobError(f"unknown job kind: {kind}")


@dataclass
class Job:
    id: int
    kind: str
    run_id: str | None
    argv: str
    status: str  # queued, running, done, failed, stopped, lost
    created: str
    started: str | None
    ended: str | None
    exit_code: int | None
    log: str
    params: str | None = None  # the request's fields, as JSON

    def public(self, tail: int = 4000) -> dict:
        try:
            text = Path(self.log).read_text(errors="replace")[-tail:]
        except FileNotFoundError:
            text = ""
        out = {"id": self.id, "kind": self.kind, "run_id": self.run_id, "status": self.status,
               "params": json.loads(self.params) if self.params else {},
               "created": self.created, "started": self.started, "ended": self.ended,
               "exit_code": self.exit_code, "output": text, "result": None}
        if self.kind == "precheck" and self.status in ("done", "failed"):
            out["result"] = _last_json(text)
        return out


def _last_json(text: str) -> dict | None:
    """The last line of the output that is a JSON object (`--json-result`)."""
    import json

    for line in reversed(text.strip().splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                return None
    return None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class JobRunner:
    def __init__(self, db: Path, cwd: Path, log_dir: Path,
                 build: Callable[[str, dict], list[str]] = command, poll: float = 1.0):
        self.db, self.cwd, self.log_dir, self.build, self.poll = Path(db), Path(cwd), Path(log_dir), build, poll
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._proc: subprocess.Popen | None = None
        self._current: int | None = None
        self._stopping: set[int] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        with closing(self._con()) as con, con:
            con.execute("""CREATE TABLE IF NOT EXISTS jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT,
                run_id TEXT, argv TEXT, status TEXT, created TEXT, started TEXT, ended TEXT, exit_code INTEGER,
                log TEXT)""")
            if "params" not in {r[1] for r in con.execute("PRAGMA table_info(jobs)")}:
                con.execute("ALTER TABLE jobs ADD COLUMN params TEXT")
            con.execute("UPDATE jobs SET status='lost', ended=? WHERE status='running'", (_now(),))

    def _con(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db, timeout=10)
        con.row_factory = sqlite3.Row
        return con

    # ------------------------------------------------------------------ queue

    def submit(self, kind: str, params: dict) -> Job:
        argv = self.build(kind, params)
        with closing(self._con()) as con, con:
            cur = con.execute(
                "INSERT INTO jobs (kind, run_id, argv, status, created, log, params) VALUES (?,?,?,?,?,?,?)",
                (kind, params.get("run_id"), " ".join(argv), "queued", _now(), "", json.dumps(params)))
            job_id = cur.lastrowid
            con.execute("UPDATE jobs SET log=? WHERE id=?", (str(self.log_dir / f"job-{job_id}.log"), job_id))
        return self.get(job_id)

    def get(self, job_id: int) -> Job:
        with closing(self._con()) as con:
            row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return Job(**dict(row))

    def list(self, limit: int = 50) -> list[Job]:
        with closing(self._con()) as con:
            rows = con.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [Job(**dict(r)) for r in rows]

    def stop(self, job_id: int) -> Job:
        """Ctrl-C for a running job (a run checkpoints and pauses); a queued job is just taken off the queue."""
        job = self.get(job_id)
        if job.status == "queued":
            self._set(job_id, status="stopped", ended=_now())
        elif job.status == "running" and self._current == job_id and self._proc is not None:
            self._stopping.add(job_id)
            os.killpg(self._proc.pid, signal.SIGINT)
        return self.get(job_id)

    def _set(self, job_id: int, **fields) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        with closing(self._con()) as con, con:
            con.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    # ------------------------------------------------------------------ worker

    def step(self) -> bool:
        """Start the oldest queued job and wait for it. False when the queue is empty."""
        with closing(self._con()) as con:
            row = con.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY id LIMIT 1").fetchone()
        if row is None:
            return False
        job = Job(**dict(row))
        self._set(job.id, status="running", started=_now())
        with open(job.log, "ab") as log:
            self._current = job.id
            self._proc = subprocess.Popen(job.argv.split(" "), cwd=self.cwd, stdout=log, stderr=subprocess.STDOUT,
                                          stdin=subprocess.DEVNULL, start_new_session=True)
            code = self._proc.wait()
        stopped = job.id in self._stopping or code in (-signal.SIGINT, 130)
        self._set(job.id, status="stopped" if stopped else ("done" if code == 0 else "failed"),
                  ended=_now(), exit_code=code)
        self._proc, self._current = None, None
        return True

    def start(self) -> None:
        def loop():
            while not self._stop.is_set():
                if not self.step():
                    time.sleep(self.poll)

        self._thread = threading.Thread(target=loop, name="pruneagent-jobs", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
