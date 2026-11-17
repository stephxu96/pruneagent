# ADR-005: Logging and Agent Calls — Decision Log with Rationale and Strategy Choices; Swappable Agent Backend (Codex CLI or LLM API); Cache, Usage Ledger, Pinned Versions

- **Status:** Proposed (v0.6; agent calls through a swappable backend: Codex CLI on the ChatGPT plan, or an LLM API)
- **Date:** 2026-11-16
- **Deciders:** Steph Xu
- **Related:** ADR-001, ADR-003, ADR-004, ADR-007; ARCHITECTURE.md §10

## Context

The agent needs one model call per round that returns a JSON action. Two ways to make it:

| Backend | Cost | Speed and size (measured 2026-11-16, codex-cli 0.155.1) | Catch |
|---|---|---|---|
| Codex CLI (`codex exec`, ChatGPT plan) | No per-call charge | ~11 s per call; ~20k input tokens, mostly Codex's own instructions | A coding agent with tools, so it must be locked down; plan usage limits; must be installed and logged in where the graph runs |
| LLM API (provider SDK) | Per token | Typically a few seconds; only the prompt's own tokens | Needs an API key and a spending cap |

Neither is clearly better for every run: Codex costs nothing for tuning; an API is faster, has no plan limits, and is the usual choice in the literature.

## Decision

- **One interface, two backends.** `Backend.complete(prompt, schema) -> (json, usage)`, chosen per run by `agent_backend = codex | api` in the run config. Everything else is shared: the prompt and agent instructions, the action JSON schema, the validator (every answer is still checked by code; ADR-001), the cache, the decision log and the usage ledger.
- **`CodexBackend`:** `codex exec` with `--sandbox read-only`, an empty working directory, sessions saved (Codex writes the limit status into each saved session; read by thread id), `--output-schema` (answer must match the schema), `--json` (events, including token usage), and a pinned `--model` and reasoning effort. The prompt forbids running commands; the read-only sandbox enforces it.
- **`ApiBackend`:** the provider's SDK with its structured-output mode bound to the same schema; provider and model set in the run config; key from `.env` (ignored by git) or Colab Secrets.
- **Same contract tests for both:** a fake backend and both real backends must return schema-valid actions for the same fixed prompts; a malformed answer is rejected the same way.
- **Decision log:** one JSONL file per run: every agent action with its rationale, probe results, validation result, every candidate with its memory, estimated speed and perplexity (how well the model predicts real text; lower is better), the gate decision, device-target warnings, strategies proposed and accepted, backend, tokens, seconds, cost (API), GPU time and evaluation count. This log is the audit trail and the source of the search measures and strategy counts (ADR-004).
- **Manifest:** model id and revision, agent backend, provider or Codex CLI version, agent model and reasoning effort, instruction version, seed, dtype, quality-bar values, device target, git commit, lock-file hash, GPU type.
- **LLM cache:** keyed by prompt, schema, backend and model; a rerun makes no calls and gives the same decisions.
- **Usage ledger:** stops agent calls at 60 per run on both backends; on the API backend also at $5 per run and $100 total. Records calls, tokens, seconds and cost. Tracks compute units against the 200-unit cap.
- **Usage limits (Codex) and caps (API): tell, pause, resume.**
  - *Read:* after every Codex call, read the limit status from that call's saved session: 5-hour and weekly windows (% used, reset time), whether a limit was reached, and the credits balance. A non-zero exit or a limit error from `codex exec` counts as reached.
  - *Predict:* keep a running % per call for each window (measured 2026-11-16: ~0.4% of the 5-hour window per call; weekly < 0.2%). Before a run: calls that fit = (90% − used) ÷ % per call. If fewer than the run needs (up to 60), warn with the reset time and the options: start anyway (it will pause), wait, or use the API backend.
  - *Pause:* at 90% of either window, or when a limit or the API cost cap is reached, the graph checkpoints (it already does every round) and stops with status `paused: <reason>, resets at <time>`. Stopping at 90% also keeps runs from silently drawing on paid credits if the plan would.
  - *Run ids:* each run gets a readable id (e.g. `20261116-1432-qwen3-0.6b-codex`), used as the LangGraph checkpoint key and in the log and result file names. `pruneagent runs` lists runs with status (running, paused and why, finished), last round and reset time.
  - *Resume:* `pruneagent resume <run-id>`, or `pruneagent resume` alone when exactly one run is paused (with several paused, it lists them and asks for an id rather than guessing). It continues from that run's last checkpoint: after the reset on the same backend, or now on the other backend (the switch is logged; mixed-backend runs are not used for reported results). Cached calls are never repeated.
- **Credentials:** never in git (`.env` is ignored) and never in a file on Drive; Colab uses Colab Secrets for both the Codex access token and API keys.
- **Storage:** checkpoints, logs and outputs (pruned model as safetensors plus plan JSON that the loader rebuilds from; HTML report) saved to Google Drive or local drive.

## Consequences

- Positive: tuning can run free on Codex and the production run can use either; a backend that hits limits or misbehaves is swapped without touching the graph; results record which backend made each decision.
- Positive: a limit never loses work: the run pauses at a checkpoint and resumes, and the pre-run estimate says in advance whether a run fits.
- Negative: two backends to test; decisions are not comparable across backends unless the model is the same, so the production run fixes one backend and model before it starts.

## Evidence That Would Reverse This Decision

- One backend is never used after phase 3 → drop it and its tests.
- Writing to Drive slows the loop noticeably → write locally, sync to Drive every 5 rounds.
