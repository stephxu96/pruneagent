#!/usr/bin/env python3
"""PruneAgent project board: local server.

Run:  python3 board/server.py            (then open http://127.0.0.1:8765)
      python3 board/server.py --port 9000

Serves index.html and reads/writes board.json next to this file. Standard library only.
Writes are guarded by an ETag (hash of board.json), so an edit made on disk (by hand or by
board/cli.py) is never silently overwritten by a stale browser tab.
"""
import argparse
import hashlib
import json
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "board.json"
LOCK = threading.Lock()
STATIC = {"/": "index.html", "/index.html": "index.html"}
ALLOWED_HOSTS = {"127.0.0.1", "localhost"}


def read_board() -> tuple[bytes, str]:
    raw = DATA.read_bytes()
    return raw, '"' + hashlib.sha256(raw).hexdigest()[:16] + '"'


def write_board(obj: dict) -> str:
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode()
    tmp = DATA.with_suffix(".json.tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, DATA)
    return '"' + hashlib.sha256(raw).hexdigest()[:16] + '"'


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        if host in ALLOWED_HOSTS:
            return True
        self.send_error(403, "Board is local only")
        return False

    def _send(self, code: int, body: bytes, etag: str | None = None, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if etag:
            self.send_header("ETag", etag)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._host_ok():
            return
        if self.path == "/api/board":
            with LOCK:
                raw, etag = read_board()
            return self._send(200, raw, etag)
        if self.path in STATIC:
            self.path = "/" + STATIC[self.path]
            return super().do_GET()
        self.send_error(404)

    def do_PUT(self):
        if not self._host_ok():
            return
        if self.path != "/api/board":
            return self.send_error(404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            new = json.loads(self.rfile.read(length))
            assert isinstance(new, dict) and isinstance(new.get("cards"), list)
        except (ValueError, AssertionError):
            return self._send(400, b'{"error": "invalid board JSON"}')
        with LOCK:
            raw, etag = read_board()
            if self.headers.get("If-Match") != etag:
                return self._send(409, raw, etag)
            new_etag = write_board(new)
        self._send(200, b'{"ok": true}', new_etag)

    def log_message(self, fmt, *args):
        if self.command == "PUT":
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if not DATA.exists():
        raise SystemExit(f"Missing {DATA}")
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"PruneAgent board: http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
