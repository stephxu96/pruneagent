# ADR-008: A Web Application on Top of the Existing Tools

- **Status:** Accepted (v0.9, 2026-11-19)
- **Date:** 2026-11-19
- **Deciders:** Steph Xu
- **Related:** ADR-001, ADR-005, ADR-006; ARCHITECTURE.md §1.2 (FR16), §2, §3, §5, §8b, §13 (Q2)

## Context

Until v0.8, the only way to use PruneAgent is the `pruneagent` command line, plus scripts for the checks after a run (`heldout_check.py`, `accuracy_check.py`, `flap_curve.py`). The architecture's client part is one box ("CLI / Colab notebook"). It does not say what a user sees or does. The only thing shown to a user is the report written at the end. A researcher who wants to prune a model for a device should not need to know the commands, read SQLite files or follow a JSONL log to see why the agent did what it did.

The tools already do everything that is needed:
- start, list and resume runs (`run`, `runs`, `resume`);
- the early warning (`estimate`);
- usage limits (`limits`);
- comparisons (`compare`, the held-out, accuracy and FLAP-curve checks).

The run store keeps the status, every round's record and every kept plan.

## Options

| Option | What the user gets | Build cost | Problem |
|---|---|---|---|
| A. Command line only, with a better report | A file at the end | Low | Nothing to see during a run; not an application |
| B. Notebook with live charts (the v0.8 default for Q2) | A demo | Low | Runs one way, top to bottom; not usable as a tool |
| C. Single-file dashboard (Streamlit or Gradio) | Pages over the run files | Low to moderate | Its rerun-the-script model fits forms and charts, not long-running jobs with pause and resume; hard to shape the round-by-round view |
| **D. Web application: browser interface + application server that calls the existing tools** | Start, watch, pause, resume and compare runs in a browser | Moderate | A second language (TypeScript) in the repo |

## Decision

**D.** The web application is a layer on top of the tools. It adds no pruning, evaluation or decision logic.

- **Web interface** (React + TypeScript, built with Vite): five screens.
  - **New run:** model, memory limit, optional device target and agent backend. It shows the early warning and the usage-limit estimate before anything starts.
  - **Runs:** every run with its state (running, paused, finished, failed).
  - **Live run:** a timeline of rounds: the agent's proposals and reasons, each candidate's memory and perplexity, the outcome (kept, undone, refused), and a memory-against-perplexity trace moving toward the limit. Pause and resume buttons.
  - **Results:** the agent against Baseline: FLAP and the unpruned model, on held-out perplexity and accuracy (weighted over all questions, then per set). It shows the final warning if there is one, and a download of the pruned model and plan.
  - **Tools:** the follow-up checks (held-out, accuracy, FLAP curve) started on a finished run.
- **Application server** (FastAPI, Python): an HTTP API with one endpoint per tool, plus read-only views of the run store.
  - **Job runner.** Each tool call becomes a job. The job runner starts the same command a user would type, as a separate process, under the Mac GPU lock (`scripts/mac_gpu.sh`), so one GPU job runs at a time and others wait in a queue.
  - **Live updates.** The server sends live updates to the browser (server-sent events) by watching the run store.
  - **Pause.** Pause sends Ctrl-C (SIGINT) to the job, which already checkpoints and stops cleanly (FR11).
- **One source of truth.** Every number on screen is read from the files the tools write (run store, `decisions.jsonl`, `heldout.json`, `accuracy.json`). A run started in the browser and one started on the command line look the same in both.
- **Local, single user.** The server binds to 127.0.0.1 on the machine with the GPU. API keys and the Codex login stay on the server and never reach the browser. Hosting it for other users is out of scope.

## Consequences

- **Gains:**
  - an actual application to show and use, which also serves as the live demo (Q2 closed; replaces the notebook in task 3.6);
  - past runs can be replayed with no GPU or Codex use.
- **Costs:**
  - a frontend build (Node) and a second test suite (Vitest for the interface, pytest for the API);
  - about 2 days of work.
- **Rules that keep it thin:**
  - the server may only call tools and read their files;
  - any new number shown must first exist as a tool output, so it is also available on the command line.
- **Reversal test:** if the application layer turns out to need logic of its own (for example, computing a comparison the tools do not), that logic moves into a tool first.
