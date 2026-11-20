"""PruneAgent web application: an HTTP API over the tools, plus the built browser interface (ADR-008).

Start it with `pruneagent web` (http://127.0.0.1:8125). Local, single user: it binds to 127.0.0.1 only.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pruneagent import models as MD
from pruneagent import runstore as R
from pruneagent.web import jobs as J
from pruneagent.web import machine as M
from pruneagent.web import views as V

REPO = Path(__file__).resolve().parents[3]


class PrepareModel(BaseModel):
    model: str


class NewRun(BaseModel):
    model: str = J.MODELS[0]
    cut: float
    backend: str = "codex"


def create_app(root: Path = REPO / "runs", repo: Path = REPO, runner: J.JobRunner | None = None,
               dist: Path | None = REPO / "web" / "dist", curve: Path | None = None,
               start_worker: bool = True) -> FastAPI:
    root, repo = Path(root), Path(repo)
    curve = curve or repo / "results" / "flap_curve" / "qwen3-0.6b-base.json"
    root.mkdir(parents=True, exist_ok=True)
    runner = runner or J.JobRunner(root / "web_jobs.sqlite", repo, root / "web_jobs",
                                   build=lambda kind, params: J.command(kind, params, known=MD.known(root)))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if start_worker:
            runner.start()
        yield
        runner.shutdown()

    app = FastAPI(title="PruneAgent", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan)
    app.state.runner = runner

    def store_or_404(run_id: str) -> R.RunStore:
        if not J.RUN_ID.match(run_id) or not (root / run_id / "run.sqlite").exists():
            raise HTTPException(404, "no such run")
        return R.RunStore.open(run_id, root)

    def submit(kind: str, params: dict) -> dict:
        try:
            return runner.submit(kind, params).public()
        except J.JobError as e:
            raise HTTPException(422, str(e)) from None

    # ---------------------------------------------------------------- runs (read only)

    @app.get("/api/runs")
    def runs() -> list[dict]:
        return V.list_runs(root)

    @app.get("/api/runs/{run_id}")
    def run(run_id: str) -> dict:
        store_or_404(run_id)
        return V.run_detail(root, run_id)

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
                mark = (info.updated, info.round, info.status)
                if mark != last:
                    last = mark
                    yield f"data: {json.dumps({'state': info.state, 'round': info.round})}\n\n"
                if info.state == "finished":
                    return
                await asyncio.sleep(2)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/api/runs/{run_id}/download/{what}")
    def download(run_id: str, what: str) -> FileResponse:
        store = store_or_404(run_id)
        kept = sorted((store.dir / "plans").glob("accepted-*.json"))
        if what != "plan" or not kept:
            raise HTTPException(404, "nothing to download")
        return FileResponse(kept[-1], filename=f"{run_id}-plan.json")

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
        """Ctrl-C to the run's process: it saves a checkpoint and pauses, as from a terminal."""
        info = store_or_404(run_id).info()
        if info.state != "running" or info.host != socket.gethostname() or not info.pid:
            raise HTTPException(409, f"the run is {info.describe()}, not running on this computer")
        os.kill(info.pid, signal.SIGINT)
        return {"ok": True}

    @app.post("/api/runs/{run_id}/resume")
    def resume(run_id: str) -> dict:
        info = store_or_404(run_id).info()
        if not info.resumable:
            raise HTTPException(409, f"the run is {info.describe()}")
        return submit("resume", {"run_id": run_id})

    @app.post("/api/runs/{run_id}/checks/{kind}")
    def check(run_id: str, kind: str) -> dict:
        if kind not in ("heldout", "accuracy"):
            raise HTTPException(404, "unknown check")
        if store_or_404(run_id).info().state != "finished":
            raise HTTPException(409, "checks run on finished runs")
        return submit(kind, {"run_id": run_id})

    @app.get("/api/jobs")
    def jobs() -> list[dict]:
        return [j.public(tail=1500) for j in runner.list()]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: int) -> dict:
        try:
            return runner.get(job_id).public()
        except KeyError:
            raise HTTPException(404, "no such job") from None

    @app.post("/api/jobs/{job_id}/stop")
    def stop(job_id: int) -> dict:
        try:
            return runner.stop(job_id).public()
        except KeyError:
            raise HTTPException(404, "no such job") from None

    # ---------------------------------------------------------------- the browser interface

    if dist is not None and (dist / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def page(path: str) -> FileResponse:
            f = dist / path
            return FileResponse(f if path and f.is_file() and dist in f.resolve().parents else dist / "index.html")

    return app


def serve(port: int = 8125) -> None:
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=port, log_level="warning")
