"""Run storage (tasks 2.7 and 2.8): checkpoints, accepted plans for rollback, and the decision log, on disk.

One directory per run, under `runs/` on the local drive or Google Drive:

    runs/<run-id>/
        run.sqlite          run status, one checkpoint per round, the accepted-plan history
        plans/accepted-000.json, accepted-001.json, ...
                            each accepted plan as plan JSON (no weights: a plan rebuilds from the cached scores,
                            ADR-002); index 0 is the unpruned model
        decisions.jsonl     the decision log: one `StepRecord` per line, appended and flushed every round

The run id is readable, e.g. `20261116-1432-qwen3-0.6b-codex` (ADR-005).

`RunStore` has the same methods as `graph.MemoryStore`, so the graph uses either. The graph checkpoints at the
end of every round (`stop_check`) and once more in `final`. A run killed mid-round resumes from its last
checkpoint: `sync` first removes what the unfinished round wrote (its log lines, a plan it accepted or a rollback
it made), then the round is replayed; its agent call comes from the agent cache, so no call is repeated.

The checkpointer is our own: LangGraph's SQLite saver is a separate package that stores every node's writes in a
binary format, while resume needs only the state at the end of a round, kept here as readable JSON.

Status: `running` while the graph runs; `paused: <reason>` when it stopped on an error or an interruption it could
catch (a usage limit, Ctrl-C); `finished: <stop reason>` at the end. A run whose process died without a word (a
Colab runtime stopped) still says `running`; `RunInfo.state` reports it as `interrupted` once its process is gone
(same machine) or it has not written for `STALE_MINUTES` (another machine). Paused and interrupted runs can resume.
"""

from __future__ import annotations

import json
import os
import re
import socket
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypedDict

from pydantic import TypeAdapter

from pruneagent import plan as P
from pruneagent.schema import (
    AcceptedPlan,
    Baseline,
    CandidateResult,
    DeviceWarning,
    Plan,
    RunConfig,
    SensitivityTable,
    StepRecord,
    Usage,
)

DEFAULT_ROOT = Path("runs")
DB = "run.sqlite"
LOG = "decisions.jsonl"
PLANS = "plans"
PLAN_FORMAT = "pruneagent-accepted-plan"
STALE_MINUTES = 30  # a run on another machine that has not written for this long counts as interrupted

_SCHEMA = """
CREATE TABLE IF NOT EXISTS run (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    run_id TEXT NOT NULL, status TEXT NOT NULL, reason TEXT, created TEXT NOT NULL, updated TEXT NOT NULL,
    host TEXT, pid INTEGER, round INTEGER NOT NULL DEFAULT 0, config TEXT NOT NULL, deps_factory TEXT);
CREATE TABLE IF NOT EXISTS checkpoint (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, round INTEGER NOT NULL, created TEXT NOT NULL, state TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS accepted (
    idx INTEGER PRIMARY KEY, round INTEGER NOT NULL, val_ppl REAL NOT NULL, weight_bytes INTEGER NOT NULL,
    file TEXT NOT NULL, created TEXT NOT NULL);
"""


class Saved(TypedDict, total=False):
    """The part of the graph state a checkpoint keeps: everything the next round reads. Per-round fields (the
    reply, its check, the gate) are rebuilt by the replayed round."""

    run: RunConfig
    baseline: Baseline
    sensitivity: SensitivityTable
    plan: Plan
    accepted: list[AcceptedPlan]
    candidates: list[CandidateResult]
    history: list[StepRecord]
    warnings: list[DeviceWarning]
    usage: Usage
    eval_seconds: list[float]
    round: int
    invalid_in_row: int
    since_accept: int
    feedback: str | None
    stop_reason: str | None
    final: dict | None


_SAVED = TypeAdapter(Saved)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def make_run_id(run_cfg: RunConfig, root: Path | str = DEFAULT_ROOT, now: datetime | None = None) -> str:
    """`<yyyymmdd-hhmm>-<model>-<backend>`, e.g. `20261116-1432-qwen3-0.6b-codex`; `-2`, `-3` ... when taken."""
    return new_run_id(run_cfg.model_id, run_cfg.agent_backend, root, now)


def new_run_id(model_id: str, backend: str, root: Path | str = DEFAULT_ROOT, now: datetime | None = None,
               taken: set[str] | frozenset[str] = frozenset()) -> str:
    """A free run id for this model and backend (`make_run_id`); `taken` adds ids reserved but not yet on disk."""
    now = now or datetime.now()
    model = model_id.rsplit("/", 1)[-1].lower().removesuffix("-base")
    base = re.sub(r"[^a-z0-9.\-]+", "-", f"{now:%Y%m%d-%H%M}-{model}-{backend}")
    run_id, n = base, 1
    while (Path(root) / run_id).exists() or run_id in taken:
        n += 1
        run_id = f"{base}-{n}"
    return run_id


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass
class RunInfo:
    run_id: str
    status: str  # as stored: running, paused, finished
    reason: str | None
    round: int
    created: str
    updated: str
    host: str | None
    pid: int | None
    n_accepted: int
    deps_factory: str | None

    @property
    def state(self) -> str:
        """running, interrupted (a `running` run whose process is gone), paused or finished."""
        if self.status != "running":
            return self.status
        if self.host == socket.gethostname():
            return "running" if self.pid == os.getpid() or _alive(self.pid) else "interrupted"
        idle = datetime.now(UTC) - datetime.fromisoformat(self.updated)
        return "interrupted" if idle > timedelta(minutes=STALE_MINUTES) else "running"

    @property
    def resumable(self) -> bool:
        return self.state in ("paused", "interrupted")

    def describe(self) -> str:
        s = self.state
        if s == "running" and self.host != socket.gethostname():
            s = f"running on {self.host}"
        return f"{s}: {self.reason}" if self.reason and s in ("paused", "finished") else s


class RunStore:
    """Checkpoints, accepted plans and the decision log of one run, under `runs/<run-id>/`."""

    def __init__(self, run_dir: Path | str):
        self.dir = Path(run_dir)
        if not (self.dir / DB).is_file():
            raise FileNotFoundError(f"{self.dir} is not a run directory (no {DB})")

    # ------------------------------------------------------------------------------------------ open and create

    @classmethod
    def create(cls, run_cfg: RunConfig, root: Path | str = DEFAULT_ROOT, run_id: str | None = None,
               deps_factory: str | None = None) -> RunStore:
        """A new run directory. `deps_factory` ("module:function", called as fn(RunConfig, run_dir) -> Deps)
        lets `pruneagent resume` rebuild the model, evaluator and agent for this run."""
        run_id = run_id or make_run_id(run_cfg, root)
        run_dir = Path(root) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / PLANS).mkdir()
        (run_dir / LOG).touch()
        with closing(sqlite3.connect(run_dir / DB)) as con, con:
            con.executescript(_SCHEMA)
            con.execute("INSERT INTO run (id, run_id, status, created, updated, host, pid, config, deps_factory) "
                        "VALUES (1, ?, 'running', ?, ?, ?, ?, ?, ?)",
                        (run_id, _now(), _now(), socket.gethostname(), os.getpid(), run_cfg.model_dump_json(),
                         deps_factory))
        return cls(run_dir)

    @classmethod
    def open(cls, run_id: str, root: Path | str = DEFAULT_ROOT) -> RunStore:
        return cls(Path(root) / run_id)

    def _db(self) -> sqlite3.Connection:
        # one connection per operation: graph nodes may run on different threads
        return sqlite3.connect(self.dir / DB)

    @property
    def run_id(self) -> str:
        return self.dir.name

    @property
    def log_path(self) -> Path:
        return self.dir / LOG

    def run_config(self) -> RunConfig:
        with closing(self._db()) as con:
            (config,) = con.execute("SELECT config FROM run").fetchone()
        return RunConfig.model_validate_json(config)

    def info(self) -> RunInfo:
        with closing(self._db()) as con:
            row = con.execute("SELECT run_id, status, reason, round, created, updated, host, pid, deps_factory "
                              "FROM run").fetchone()
            (n,) = con.execute("SELECT COUNT(*) FROM accepted").fetchone()
        run_id, status, reason, rnd, created, updated, host, pid, factory = row
        return RunInfo(run_id=run_id, status=status, reason=reason, round=rnd, created=created, updated=updated,
                       host=host, pid=pid, n_accepted=n, deps_factory=factory)

    def set_config(self, run_cfg: RunConfig) -> None:
        """Replace the run's config (resuming on the other agent backend)."""
        with closing(self._db()) as con, con:
            con.execute("UPDATE run SET config = ?, updated = ?", (run_cfg.model_dump_json(), _now()))

    def set_status(self, status: str, reason: str | None = None) -> None:
        """running (also records this process, so another machine can take a run over), paused or finished."""
        with closing(self._db()) as con, con:
            con.execute("UPDATE run SET status = ?, reason = ?, updated = ?, host = ?, pid = ?",
                        (status, reason, _now(), socket.gethostname(), os.getpid()))

    # ------------------------------------------------------------------------- the graph's four store methods

    def append_log(self, record: StepRecord) -> None:
        """One line of the decision log, flushed to disk before the graph moves on."""
        with self.log_path.open("a") as f:
            f.write(record.model_dump_json() + "\n")
            f.flush()
            os.fsync(f.fileno())

    def checkpoint(self, state: dict) -> None:
        saved = {k: state[k] for k in Saved.__annotations__ if k in state}
        blob = _SAVED.dump_json(saved).decode()
        rnd = state.get("round") or 0
        with closing(self._db()) as con, con:
            con.execute("INSERT INTO checkpoint (round, created, state) VALUES (?, ?, ?)", (rnd, _now(), blob))
            con.execute("UPDATE run SET round = ?, updated = ?", (rnd, _now()))

    def save_accepted(self, index: int, accepted: AcceptedPlan) -> None:
        """Accepted plan `index` as plan JSON plus its row in SQLite; any later index is discarded."""
        self._discard_after(index - 1)
        name = f"accepted-{index:03d}.json"
        record = {"format": PLAN_FORMAT, "version": 1, "index": index, "round": accepted.round,
                  "val_ppl": accepted.val_ppl, "weight_bytes": accepted.weight_bytes,
                  "plan": P.to_json(accepted.plan)}
        path = self.dir / PLANS / name
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=1) + "\n")
        tmp.replace(path)
        with closing(self._db()) as con, con:
            con.execute("INSERT OR REPLACE INTO accepted (idx, round, val_ppl, weight_bytes, file, created) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (index, accepted.round, accepted.val_ppl, accepted.weight_bytes, f"{PLANS}/{name}", _now()))

    def restore(self, index: int) -> AcceptedPlan:
        """Rollback: accepted plan `index`, read back from its plan JSON; every later accepted plan is discarded."""
        with closing(self._db()) as con:
            row = con.execute("SELECT file FROM accepted WHERE idx = ?", (index,)).fetchone()
        if row is None:
            raise KeyError(f"no accepted plan {index} in {self.dir}")
        restored = self._read_plan(row[0])
        self._discard_after(index)
        return restored

    def _read_plan(self, file: str) -> AcceptedPlan:
        r = json.loads((self.dir / file).read_text())
        if r.get("format") != PLAN_FORMAT:
            raise ValueError(f"{self.dir / file} is not a {PLAN_FORMAT} file")
        return AcceptedPlan(round=r["round"], plan=r["plan"], val_ppl=r["val_ppl"], weight_bytes=r["weight_bytes"])

    # ---------------------------------------------------------------------------------------- reading back

    def _discard_after(self, index: int) -> None:
        with closing(self._db()) as con, con:
            files = [f for (f,) in con.execute("SELECT file FROM accepted WHERE idx > ?", (index,))]
            con.execute("DELETE FROM accepted WHERE idx > ?", (index,))
        for f in files:
            (self.dir / f).unlink(missing_ok=True)

    def accepted(self) -> list[AcceptedPlan]:
        """The accepted-plan history, oldest first (index 0 is the unpruned model)."""
        with closing(self._db()) as con:
            files = [f for (f,) in con.execute("SELECT file FROM accepted ORDER BY idx")]
        return [self._read_plan(f) for f in files]

    def log(self) -> list[StepRecord]:
        """The decision log, one record per round."""
        return [StepRecord.model_validate_json(line) for line in self.log_path.read_text().splitlines()
                if line.strip()]

    def load_checkpoint(self) -> dict | None:
        """The state at the last checkpoint, or None when the run never finished a round."""
        with closing(self._db()) as con:
            row = con.execute("SELECT state FROM checkpoint ORDER BY seq DESC LIMIT 1").fetchone()
        return dict(_SAVED.validate_json(row[0])) if row else None

    def sync(self, state: dict) -> None:
        """Before resuming from `state`: drop what an unfinished round wrote after it. Log lines of later rounds
        go, and the accepted plans on disk are made to match the checkpoint's."""
        last = state.get("round") or 0
        lines = [line for line in self.log_path.read_text().splitlines()
                 if line.strip() and json.loads(line)["round"] <= last]
        tmp = self.log_path.with_suffix(".tmp")
        tmp.write_text("".join(line + "\n" for line in lines))
        tmp.replace(self.log_path)
        accepted = state.get("accepted") or []
        for i, a in enumerate(accepted):
            self.save_accepted(i, a)
        self._discard_after(len(accepted) - 1)


# ------------------------------------------------------------------------------------------------ all runs


def list_runs(root: Path | str = DEFAULT_ROOT) -> list[RunInfo]:
    """Every run under `root`, oldest first."""
    root = Path(root)
    if not root.is_dir():
        return []
    return [RunStore(d).info() for d in sorted(root.iterdir()) if (d / DB).is_file()]


class ResumeError(Exception):
    """No run to resume, or more than one and no id given. The message says which and lists them."""


def pick_run(root: Path | str = DEFAULT_ROOT, run_id: str | None = None) -> RunStore:
    """The run to resume: `run_id`, or the only paused (or interrupted) run when no id is given. Refuses, listing
    the candidates, when several are paused; refuses a finished run or one still running."""
    if run_id is not None:
        try:
            store = RunStore.open(run_id, root)
        except FileNotFoundError:
            raise ResumeError(f"no run {run_id!r} under {root}") from None
        info = store.info()
        if not info.resumable:
            raise ResumeError(f"run {run_id} is {info.describe()}: nothing to resume")
        return store
    paused = [r for r in list_runs(root) if r.resumable]
    if not paused:
        raise ResumeError(f"no paused run under {root}")
    if len(paused) > 1:
        listing = "\n".join(f"  {r.run_id}  ({r.describe()}, round {r.round})" for r in paused)
        raise ResumeError(f"{len(paused)} runs are paused; give the id of the one to resume:\n{listing}")
    return RunStore.open(paused[0].run_id, root)


def past_rates(root: Path | str, model_id: str) -> list[dict]:
    """Active time and rounds per MB cut in this model's finished runs (agent time + candidate checks, from the
    decision logs; pauses not counted). Used to say how long a new run will take."""
    out = []
    for info in list_runs(root):
        if info.state != "finished":
            continue
        store = RunStore.open(info.run_id, root)
        try:
            if store.run_config().model_id != model_id:
                continue
            kept = store.accepted()
            log = store.log()
        except (OSError, ValueError):
            continue
        mb = (kept[0].weight_bytes - kept[-1].weight_bytes) / 1e6 if len(kept) > 1 else 0
        if mb <= 0 or not log:
            continue
        seconds = sum(r.llm_seconds + r.gpu_seconds for r in log)
        out.append({"run": info.run_id, "minutes_per_mb": seconds / 60 / mb, "rounds_per_mb": len(log) / mb,
                    "minutes_per_round": seconds / 60 / len(log)})
    return out


def time_estimate(rates: list[dict], to_free_mb: float, max_rounds: int, sweep_layers: int | None = None) -> dict | None:
    """How long a new run will take. `minutes` is the usual range (fastest and slowest past pace per MB cut);
    `worst_minutes` is the round cap at the slowest past pace per round, which a run reaches when it undoes its own
    work or stalls near the limit. Both are capped by the round cap.

    A sweep (search.py) has exactly one round per layer (`sweep_layers`), so its range is that many rounds at the
    fastest and slowest past pace per round."""
    if not rates:
        return None
    if sweep_layers:
        paces = [r["minutes_per_round"] for r in rates]
        return {"minutes": [min(paces) * sweep_layers, max(paces) * sweep_layers],
                "rounds": [sweep_layers, sweep_layers], "worst_minutes": max(paces) * sweep_layers,
                "max_rounds": sweep_layers, "based_on_runs": len(rates)}
    per_round = max(r["minutes_per_round"] for r in rates)
    worst = max_rounds * per_round
    return {"minutes": [min(min(r["minutes_per_mb"] for r in rates) * to_free_mb, worst),
                        min(max(r["minutes_per_mb"] for r in rates) * to_free_mb, worst)],
            "rounds": [min(min(r["rounds_per_mb"] for r in rates) * to_free_mb, max_rounds),
                       min(max(r["rounds_per_mb"] for r in rates) * to_free_mb, max_rounds)],
            "worst_minutes": worst, "max_rounds": max_rounds, "based_on_runs": len(rates)}


RECENT_ROUNDS = 10


def _sweep_layers(cfg, kept) -> int | None:
    """The number of layers when the run is a sweep (search.py: one round per layer), else None."""
    from pruneagent.search import get

    return len(kept[0].plan) if kept and get(cfg.search).order == "sweep" else None


def time_left(store: RunStore, rates: list[dict]) -> dict | None:
    """How long a running run still needs, from its own progress: MB still to cut times its own pace so far, and
    the slowest past pace for the upper end. Pace per round is the last RECENT_ROUNDS rounds (the agent slows as
    its history grows). Capped by the rounds left."""
    cfg = store.run_config()
    kept, log = store.accepted(), store.log()
    if len(kept) < 1 or not log:
        return None
    limit_mb = cfg.device.memory_limit_gb * 1e9 / 1e6
    left_mb = max(0.0, kept[-1].weight_bytes / 1e6 - limit_mb)
    recent = log[-RECENT_ROUNDS:]
    per_round = sum(r.llm_seconds + r.gpu_seconds for r in recent) / 60 / len(recent)
    sweep = _sweep_layers(cfg, kept)
    if sweep:  # one round per layer: what's left is the layers not yet decided
        decided = sum(r.outcome in ("ACCEPT", "ROLLBACK", "LEFT") for r in log)
        rounds_left = max(0, sweep - decided)
        return {"minutes": [rounds_left * per_round, rounds_left * per_round], "worst_minutes": rounds_left * per_round,
                "rounds_left": rounds_left, "mb_left": left_mb}
    rounds_left = max(0, cfg.rules.max_rounds - len(log))
    cap = rounds_left * per_round
    cut_mb = (kept[0].weight_bytes - kept[-1].weight_bytes) / 1e6
    paces = [r["minutes_per_mb"] for r in rates]
    if cut_mb > 0:
        paces.append(sum(r.llm_seconds + r.gpu_seconds for r in log) / 60 / cut_mb)
    if not paces:
        return {"minutes": None, "worst_minutes": cap, "rounds_left": rounds_left, "mb_left": left_mb}
    return {"minutes": [min(min(paces) * left_mb, cap), min(max(paces) * left_mb, cap)],
            "worst_minutes": cap, "rounds_left": rounds_left, "mb_left": left_mb}
