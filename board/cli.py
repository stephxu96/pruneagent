#!/usr/bin/env python3
"""Update the project board from the terminal (for you or the coding agent).

  python3 board/cli.py list [--phase 1] [--status doing]
  python3 board/cli.py status 1.4 done          # todo | doing | blocked | done
  python3 board/cli.py note 1.4 "Round trip exact on T4"
  python3 board/cli.py check G1 2 go "rank correlation 0.81"   # gate check: pending | go | fallback
  python3 board/cli.py spend 1 --units 6.5 --llm 12               # add compute units / agent calls to a phase

An open browser tab picks up these changes within a few seconds.
"""
import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

DATA = Path(__file__).resolve().parent / "board.json"
STATUSES = ["todo", "doing", "blocked", "done"]


def load():
    return json.loads(DATA.read_text())


def save(board):
    tmp = DATA.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(board, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, DATA)


def card(board, cid):
    for c in board["cards"]:
        if c["id"] == cid:
            return c
    sys.exit(f"No card {cid}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.add_argument("--phase", type=int); p.add_argument("--status", choices=STATUSES)
    p = sub.add_parser("status"); p.add_argument("id"); p.add_argument("status", choices=STATUSES)
    p = sub.add_parser("note"); p.add_argument("id"); p.add_argument("text")
    p = sub.add_parser("check"); p.add_argument("id"); p.add_argument("index", type=int)
    p.add_argument("result", choices=["pending", "go", "fallback"]); p.add_argument("text", nargs="?", default="")
    p = sub.add_parser("spend"); p.add_argument("phase", type=int)
    p.add_argument("--units", type=float, default=0); p.add_argument("--llm", type=float, default=0)
    a = ap.parse_args()

    board = load()
    if a.cmd == "list":
        done = {c["id"] for c in board["cards"] if c["status"] == "done"}
        for c in board["cards"]:
            if a.phase is not None and c["phase"] != a.phase:
                continue
            if a.status and c["status"] != a.status:
                continue
            waiting = [d for d in c.get("deps", []) if d not in done]
            extra = f"  (waiting on {', '.join(waiting)})" if waiting and c["status"] != "done" else ""
            print(f'{c["id"]:>5}  {c["status"]:<8} {c["owner"]:<14} {c["title"]}{extra}')
        return
    if a.cmd == "status":
        card(board, a.id)["status"] = a.status
    elif a.cmd == "note":
        c = card(board, a.id)
        stamp = date.today().isoformat()
        c["notes"] = (c.get("notes", "") + f"\n[{stamp}] {a.text}").strip()
    elif a.cmd == "check":
        c = card(board, a.id)
        checks = c.get("checks") or sys.exit(f"{a.id} has no gate checks")
        if not 1 <= a.index <= len(checks):
            sys.exit(f"{a.id} has checks 1–{len(checks)}")
        checks[a.index - 1]["result"] = a.result
        if a.text:
            checks[a.index - 1]["observed"] = a.text
    elif a.cmd == "spend":
        row = next((r for r in board["budget"] if r["phase"] == a.phase), None) or sys.exit(f"No phase {a.phase}")
        row["units_used"] = round(row.get("units_used", 0) + a.units, 2)
        row["llm_used"] = round(row.get("llm_used", 0) + a.llm, 2)
    save(board)
    print("Saved.")


if __name__ == "__main__":
    main()
