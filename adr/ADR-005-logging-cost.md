# ADR-005: Logging and Cost — Decision Log with Rationale and Strategy Choices, LLM Cache, Cost Ledger, Pinned Versions

- **Status:** Proposed (v0.5; logs candidates and strategy choices; effort measures removed)
- **Date:** 2026-10-31
- **Deciders:** Steph Xu
- **Related:** ADR-001, ADR-003, ADR-004, ADR-007; ARCHITECTURE.md §10

## Decision

- **Decision log:** one JSONL file per run: every agent action with its rationale, probe results, validation result, every candidate with its memory, estimated speed and perplexity, the gate decision, device-target warnings, strategies proposed and accepted, LLM cost, GPU time and evaluation count. This log is the audit trail and the source of the search measures and strategy counts (ADR-004).
- **Manifest:** model id and revision, agent model id, instruction version, seed, dtype, quality-bar values, device target, git commit, lock-file hash, GPU type.
- **LLM cache:** requests hashed; a rerun costs nothing and gives the same decisions.
- **Cost ledger:** stops LLM calls at $5 per run and $100 total; tracks compute units against the 200-unit cap.
- **Storage:** checkpoints, logs and outputs (pruned model as safetensors plus plan JSON that the loader rebuilds from; HTML report) saved to Google Drive or local drive.

## Consequences

- Any run can be replayed from the log and cache; any reported number traces to a log line; a researcher can inspect why each step was taken and which strategies the agent favoured.

## Evidence That Would Reverse This Decision

- Writing to Drive slows the loop noticeably → write locally, sync to Drive every 5 rounds.
