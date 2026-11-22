"""The tools API (docs/APP_DESIGN.md §4.1): the one way into PruneAgent's tools from outside, for the app.

The tools service (`pruneagent tools-service`) runs this API and the workflow runner together, natively, on the
machine with the GPU. It owns the run folders, the model assets, the workflow store and the hardware details; the
app (a container) only calls these endpoints. Local and single user: it listens on 127.0.0.1 only.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from pruneagent import models as MD
from pruneagent import runstore as R
from pruneagent.service import machine as M
from pruneagent.service import requests as J
from pruneagent.service import views as V
from pruneagent.service import workflows as W

REPO = Path(__file__).resolve().parents[3]


KIND_FOR_UI = {"setup": "prepare", "precheck": "precheck"}
STATUS_FOR_UI = {"waiting": "queued", "running": "running", "finished": "done", "failed": "failed",
                 "paused": "stopped", "cancelled": "cancelled"}


def public(wf: W.Workflow, tail: int = 4000) -> dict:
    """A workflow in the shape the interface reads (one card in the history and on the Models page)."""
    if wf.kind == "pruning":
        kind = "resume" if wf.params.get("resume") else "start"
    elif wf.kind == "check":
        kind = wf.params.get("check", "check")
    else:
        kind = KIND_FOR_UI.get(wf.kind, wf.kind)
    step = wf.steps[min(wf.step, len(wf.steps) - 1)] if wf.steps else {}
    full = W.read_log(step.get("log"))
    started = next((s["started"] for s in wf.steps if s["started"]), None)
    out = {"id": wf.id, "kind": kind, "workflow": wf.kind, "run_id": wf.run_id, "status": STATUS_FOR_UI[wf.state],
           "reason": wf.reason, "params": wf.params, "release": wf.release, "created": wf.created,
           "started": started, "ended": step.get("ended"), "exit_code": step.get("exit_code"),
           "output": full[-tail:], "result": None,
           "steps": [{"name": s["name"], "state": s["state"]} for s in wf.steps], "step": wf.step}
    if wf.kind == "precheck" and wf.state in ("finished", "failed"):
        out["result"] = W.last_json(full)
    if wf.kind == "setup":
        out["progress"] = J.prepare_progress(full)
    return out


class PrepareModel(BaseModel):
    model: str


class NewRun(BaseModel):
    model: str = J.MODELS[0]
    cut: float
    backend: str = "codex"
    search: str = "multi_layer"  # search strategy (search.py)


def create_app(root: Path = REPO / "runs", repo: Path = REPO, store: W.Store | None = None,
               curve: Path | None = None) -> FastAPI:
    """The tools API. Long work is handed to the workflow runner through the workflow store and read back."""
    root, repo = Path(root), Path(repo)
    curve = curve or repo / "results" / "flap_curve" / "qwen3-0.6b-base.json"
    root.mkdir(parents=True, exist_ok=True)
    store = store or W.Store(root)
    app = FastAPI(title="PruneAgent tools API", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.store = store

    def store_or_404(run_id: str) -> R.RunStore:
        if not J.RUN_ID.match(run_id) or not (root / run_id / "run.sqlite").exists():
            raise HTTPException(404, "no such run")
        return R.RunStore.open(run_id, root)

    def submit(kind: str, params: dict) -> dict:
        """Check the request's fields (the same checks as the command line), then hand it to the service."""
        try:
            J.command(kind, params, known=MD.known(root))
        except J.JobError as e:
            raise HTTPException(422, str(e)) from None
        release = W.current_release(repo)
        if kind == "precheck":
            wf = store.submit("precheck", {k: params[k] for k in ("model", "cut", "backend", "search") if k in params}, release)
        elif kind == "start":
            taken = {w.run_id for w in store.list(limit=500) if w.run_id}
            run_id = R.new_run_id(params["model"], params.get("backend", "codex"), root, taken=taken)
            wf = store.submit("pruning", {"model": params["model"], "cut": float(params["cut"]),
                                          "backend": params.get("backend", "codex"),
                                          "search": params.get("search", "multi_layer"), "run_id": run_id}, release)
        elif kind == "resume":
            wf = store.submit("pruning", {"run_id": params["run_id"], "resume": True}, release)
        elif kind == "save":
            wf = store.submit("save", {"run_id": params["run_id"]}, release)
        elif kind in J.CHECK_SCRIPTS:
            wf = store.submit("check", {"run_id": params["run_id"], "check": kind}, release)
        elif kind == "prepare":
            wf = store.submit("setup", {"model": params["model"]}, release)
        else:
            raise HTTPException(422, f"unknown request: {kind}")
        return public(wf)

    def workflow_for(run_id: str, states: tuple[str, ...]) -> W.Workflow | None:
        return next((w for w in store.list(limit=500, states=states) if w.kind == "pruning" and w.run_id == run_id),
                    None)

    # ---------------------------------------------------------------- runs (read only)

    @app.get("/api/runs")
    def runs() -> list[dict]:
        return V.list_runs(root)

    @app.get("/api/runs/{run_id}")
    def run(run_id: str) -> dict:
        store_or_404(run_id)
        return V.run_detail(root, run_id)

    @app.get("/api/runs/{run_id}/removal")
    def removal(run_id: str) -> dict:
        store_or_404(run_id)
        out = V.removal_order(root, run_id)
        if out is None:
            raise HTTPException(404, "This run's scores aren't on this computer.")
        return out

    @app.get("/api/runs/{run_id}/results")
    def results(run_id: str) -> dict:
        store_or_404(run_id)
        return V.run_results(root, run_id)

    @app.get("/api/compare")
    def compare() -> dict:
        return V.compare(root, curve)

    @app.get("/api/machine")
    def this_machine() -> dict:
        """This computer's chip and memory, and each model's original size."""
        return {"machine": M.machine(), "models": M.model_sizes(MD.known(root)), "max_cut": J.MAX_CUT,
                "free_share": M.FREE_SHARE}

    # ---------------------------------------------------------------- bring your own model (FR17)

    @app.get("/api/models/check")
    def check_model(id: str) -> dict:
        """Whether a Hugging Face model can be prepared and pruned here; reads its details, not its weights."""
        try:
            info = MD.check(id, M.machine()["memory_gb"])
        except MD.ModelError as e:
            return {"id": id, "ok": False, "reason": str(e)}
        return {**info, "ok": True, "prepared": M.ready(id), "added": id in MD.known(root)}

    @app.post("/api/models/prepare")
    def prepare(req: PrepareModel) -> dict:
        """Add the model to this computer's list and queue its preparation (resumes a stopped one)."""
        try:
            MD.check(req.model, M.machine()["memory_gb"])
        except MD.ModelError as e:
            raise HTTPException(422, str(e)) from None
        MD.add(root, req.model)
        return submit("prepare", {"model": req.model})

    @app.get("/api/machine/live")
    def machine_live() -> dict:
        """Memory in use, GPU busy and GPU memory right now."""
        return M.live()

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str) -> StreamingResponse:
        """Live updates: one message whenever the run records something new (server-sent events)."""
        store = store_or_404(run_id)

        async def stream():
            last = None
            while True:
                info = store.info()
                now = store.now() or {}
                mark = (info.updated, info.round, info.status, now.get("phase"), now.get("round"))  # also a round's phase
                if mark != last:
                    last = mark
                    yield f"data: {json.dumps({'state': info.state, 'round': info.round})}\n\n"
                if info.state == "finished":
                    return
                await asyncio.sleep(2)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/api/runs/{run_id}/download/{what}")
    def download(run_id: str, what: str) -> FileResponse:
        """The smaller model (`model`: a zip made by `pruneagent save`) or the kept plan alone (`plan`)."""
        store = store_or_404(run_id)
        if what == "model" and (store.dir / "model.zip").exists():
            return FileResponse(store.dir / "model.zip", filename=f"{run_id}-model.zip",
                                media_type="application/zip")
        kept = sorted((store.dir / "plans").glob("accepted-*.json"))
        if what == "plan" and kept:
            return FileResponse(kept[-1], filename=f"{run_id}-plan.json")
        raise HTTPException(404, "nothing to download")

    @app.post("/api/runs/{run_id}/save")
    def save(run_id: str) -> dict:
        """Build the download of a finished run (pruning's last step; on its own for runs from before it)."""
        if store_or_404(run_id).info().state != "finished":
            raise HTTPException(409, "only finished runs have a model to download")
        return submit("save", {"run_id": run_id})

    # ---------------------------------------------------------------- actions (each is one tool command)

    @app.post("/api/precheck")
    def precheck(req: NewRun) -> dict:
        """Dry run: the early warning and the usage-limit estimate, with no agent call and nothing saved."""
        return submit("precheck", req.model_dump())

    @app.post("/api/runs")
    def start(req: NewRun) -> dict:
        return submit("start", req.model_dump())

    @app.post("/api/runs/{run_id}/pause")
    def pause(run_id: str) -> dict:
        """Pause a run: through its workflow, or Ctrl-C to a run started from a terminal (it checkpoints and stops)."""
        wf = workflow_for(run_id, ("running", "waiting"))
        if wf:
            return public(store.request(wf.id, "pause"))
        info = store_or_404(run_id).info()
        if info.state != "running" or info.host != socket.gethostname() or not info.pid:
            raise HTTPException(409, f"the run is {info.describe()}, not running on this computer")
        os.kill(info.pid, signal.SIGINT)
        return {"ok": True}

    @app.post("/api/runs/{run_id}/resume")
    def resume(run_id: str) -> dict:
        wf = workflow_for(run_id, ("paused", "failed"))
        if wf:
            return public(store.resume(wf.id))
        info = store_or_404(run_id).info()
        if not info.resumable:
            raise HTTPException(409, f"the run is {info.describe()}")
        return submit("resume", {"run_id": run_id})

    @app.post("/api/runs/{run_id}/checks/{kind}")
    def check(run_id: str, kind: str) -> dict:
        if kind not in J.CHECK_SCRIPTS:  # mmlu: not part of every run, started on its own
            raise HTTPException(404, "unknown check")
        if store_or_404(run_id).info().state != "finished":
            raise HTTPException(409, "checks run on finished runs")
        return submit(kind, {"run_id": run_id})

    # ---------------------------------------------------------------- workflows ("jobs" in the interface)

    @app.get("/api/service")
    def service() -> dict:
        """Whether the background workflow service is running; without it nothing starts."""
        return W.service_status(root)

    @app.get("/api/jobs")
    def jobs() -> list[dict]:
        return [public(w, tail=1500) for w in store.list()]

    def get_or_404(job_id: int) -> W.Workflow:
        try:
            return store.get(job_id)
        except KeyError:
            raise HTTPException(404, "no such job") from None

    @app.get("/api/jobs/{job_id}")
    def job(job_id: int) -> dict:
        return public(get_or_404(job_id))

    @app.post("/api/jobs/{job_id}/stop")
    def stop(job_id: int) -> dict:
        """Pause a running workflow (at a safe point), or cancel one that hasn't started."""
        wf = get_or_404(job_id)
        return public(store.request(wf.id, "pause" if wf.state == "running" else "cancel"))

    @app.post("/api/jobs/{job_id}/continue")
    def continue_job(job_id: int) -> dict:
        """Continue a paused or failed workflow from the step it stopped at."""
        return public(store.resume(get_or_404(job_id).id))

    return app


TOOLS_PORT = 8124


def serve(port: int = TOOLS_PORT, root: Path = REPO) -> None:
    """The tools service: the workflow runner in a background thread and the tools API in front of it."""
    import threading

    import uvicorn

    runner = W.Service(root)
    threading.Thread(target=runner.serve, name="workflow-runner", daemon=True).start()
    uvicorn.run(create_app(root=runner.runs, repo=root), host="127.0.0.1", port=port, log_level="warning")
