# ADR-005: Logging and Cost — Decision Log with Rationale, LLM Cache, Cost Ledger, Pinned Versions

- **Status:** Proposed (v0.3)
- **Date:** 2026-10-30
- **Deciders:** Steph Xu
- **Related:** ADR-001, ADR-003, ADR-004; ARCHITECTURE.md §10

## Decision

- **Decision log:** one JSONL file per run: every agent action with its rationale, probe results, validation result, plan, gate result (per-step and cumulative), ACCEPT / ROLLBACK decision, LLM cost, GPU time and evaluation count. This log is the audit trail and the source of the effort measures (ADR-004).
- **Manifest:** model id and revision, agent model id, instruction version, seed, dtype, git commit, lock-file hash, GPU type.
- **LLM cache:** requests hashed; a rerun costs nothing and gives the same decisions.
- **Cost ledger:** stops LLM calls at $5 per run and $100 total; tracks compute units against the 200-unit cap.
- **Storage:** checkpoints, logs and outputs (pruned model in a sparse format, plan JSON, HTML report) saved to Google Drive or local drive.

## Consequences

- Any run can be replayed from the log and cache; any reported number traces to a log line; a researcher can inspect why each step was taken.

## Evidence That Would Reverse This Decision

- Writing to Drive slows the loop noticeably → write locally, sync to Drive every 5 rounds.
