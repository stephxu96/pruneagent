"""The PruneAgent app's server (docs/APP_DESIGN.md §4.1): serves the browser interface and forwards `/api/*` to the
tools API. It keeps nothing and reads no project files, so it runs in a container anywhere.

Standalone on purpose: it needs only fastapi, uvicorn and httpx (no PyTorch, no `pruneagent` package), so the app's
image stays small. Settings come from the environment:

- `PRUNEAGENT_TOOLS_URL`: the tools API (default `http://127.0.0.1:8124`; from a container on a Mac,
  `http://host.docker.internal:8124`)
- `PRUNEAGENT_DIST`: the built interface (default `web/dist` next to this repo, `/app/dist` in the image)
- `PORT`: where the app listens (default 8125)
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

TOOLS_URL = os.environ.get("PRUNEAGENT_TOOLS_URL", "http://127.0.0.1:8124")


def _default_dist() -> Path | None:
    """`PRUNEAGENT_DIST` (set in the image), else `web/dist` of the repo this file is in."""
    if os.environ.get("PRUNEAGENT_DIST"):
        return Path(os.environ["PRUNEAGENT_DIST"])
    parents = Path(__file__).resolve().parents
    return parents[3] / "web" / "dist" if len(parents) > 3 else None


DEFAULT_DIST = _default_dist()
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "content-length", "host",
       "content-encoding"}


def create_front(tools_url: str = TOOLS_URL, dist: Path | None = DEFAULT_DIST,
                 transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    app = FastAPI(title="PruneAgent", docs_url=None, redoc_url=None, openapi_url=None)
    client = httpx.AsyncClient(base_url=tools_url, transport=transport, timeout=httpx.Timeout(60, read=None))

    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE"], include_in_schema=False)
    async def forward(path: str, request: Request) -> Response:
        req = client.build_request(request.method, f"/api/{path}", params=request.query_params,
                                   headers={k: v for k, v in request.headers.items() if k.lower() not in HOP},
                                   content=await request.body())
        try:
            resp = await client.send(req, stream=True)
        except httpx.TransportError:
            # the one thing the app knows on its own: whether the tools answer at all
            if path == "service":
                return JSONResponse({"running": False, "reachable": False, "pid": None, "release": None})
            return JSONResponse({"detail": "The PruneAgent tools aren't running. Start them with scripts/start.sh."},
                                status_code=503)
        headers = {k: v for k, v in resp.headers.items() if k.lower() not in HOP}
        # everything streams straight through: live updates, and model downloads of several GB
        return StreamingResponse(resp.aiter_raw(), status_code=resp.status_code, headers=headers,
                                 background=BackgroundTask(resp.aclose))

    if dist is not None and (dist / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def page(path: str) -> FileResponse:
            f = dist / path
            if path and f.is_file() and dist in f.resolve().parents:
                return FileResponse(f)
            # the page itself is never kept by the browser, so a reload always gets the deployed version (the
            # scripts it names have the build's hash in their names, so they can be kept)
            return FileResponse(dist / "index.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

    return app


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8125"))
    host = os.environ.get("HOST", "127.0.0.1")  # 0.0.0.0 inside the container; compose publishes it on 127.0.0.1
    uvicorn.run(create_front(), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
