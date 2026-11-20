"""The app's server (docs/APP_DESIGN.md §4.1) uses only the tools API: it forwards /api/* and keeps nothing."""

import httpx
from fastapi.testclient import TestClient
from test_kill_resume import Agent, setup_for, use  # noqa: F401  (setup_for is a fixture)
from test_web import finished_run  # noqa: F401  (a fixture)

from pruneagent.service import workflows as W
from pruneagent.service.api import create_app
from pruneagent.web.front import create_front


def test_the_app_shows_what_the_tools_api_answers(finished_run, tmp_path):  # noqa: F811
    tools = create_app(root=finished_run, repo=tmp_path, store=W.Store(finished_run))
    front = TestClient(create_front("http://tools", dist=None, transport=httpx.ASGITransport(app=tools)))
    direct = TestClient(tools)
    assert front.get("/api/runs").json() == direct.get("/api/runs").json()
    assert front.get("/api/runs/r1").json()["id"] == "r1"
    assert front.get("/api/runs/nope").status_code == 404
    job = front.post("/api/precheck", json={"cut": 0.1})
    assert job.status_code == 200 and job.json()["kind"] == "precheck"
    assert front.post("/api/runs", json={"cut": 5}).status_code == 422


def test_the_app_says_when_the_tools_are_not_running(tmp_path):
    front = TestClient(create_front("http://127.0.0.1:9", dist=None))  # nothing listens there
    assert front.get("/api/service").json()["running"] is False
    down = front.get("/api/runs")
    assert down.status_code == 503 and "scripts/start.sh" in down.json()["detail"]


def test_the_app_reads_no_project_files():
    from pathlib import Path

    import pruneagent.web.front as f

    src = Path(f.__file__).read_text()
    assert "import pruneagent" not in src and "from pruneagent" not in src  # standalone: forwards, never reads


def test_the_app_server_starts_as_a_lone_file_like_in_the_image(tmp_path):
    import shutil
    import subprocess
    import sys

    import pruneagent.web.front as f

    shutil.copy(f.__file__, tmp_path / "front.py")  # in the image it is /app/front.py, nothing around it
    code = "import front; print(front.DEFAULT_DIST)"
    env = {"PATH": "/usr/bin:/bin", "PRUNEAGENT_DIST": "/app/dist"}
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True,
                         check=False)
    assert out.returncode == 0 and out.stdout.strip() == "/app/dist", out.stderr


def test_the_page_is_never_cached_but_its_scripts_can_be(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html></html>")
    (dist / "assets" / "index-abc.js").write_text("x")
    front = TestClient(create_front("http://tools", dist=dist))
    for path in ("/", "/run/r1"):
        assert "no-store" in front.get(path).headers["cache-control"]
    assert "no-store" not in front.get("/assets/index-abc.js").headers.get("cache-control", "")
