# Project board

A local kanban board for the phased work plan (`../WORKPLAN.md`). Python standard library only; nothing to install.

## Run

```bash
python3 board/server.py          # open http://127.0.0.1:8765
python3 board/server.py --port 9000
```

## Use

- **Move a task:** drag it to another column, or focus it and press ← / →.
- **Edit a task:** click it (or press Enter). Gates (G0–G4) list their checks; set each to *go* or *fallback* and note what you observed.
- **Phases:** click a phase to filter to it; the current phase is highlighted from the start date (`start_date` in `board.json`).
- **Ready to start only:** shows tasks in *To do* whose dependencies are all done.
- **Budget:** type compute units and LLM spend used per phase; totals turn red past the caps.

Everything saves to `board/board.json`, so the board is versioned with the repo.

## From the terminal (you or the coding agent)

```bash
python3 board/cli.py list --phase 1
python3 board/cli.py status 1.4 done
python3 board/cli.py note 1.4 "Round trip exact on T4"
python3 board/cli.py check G1 3 go "rank correlation 0.81"
python3 board/cli.py spend 1 --units 6.5 --llm 0.40
```

An open browser tab picks up these changes within a few seconds. If the browser and the file both changed, the browser does not overwrite the file; it asks you to reload.

## Change the start date

Edit `"start_date"` in `board.json` (YYYY-MM-DD). Day numbers and the current phase follow from it.
