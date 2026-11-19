# PruneAgent v0.8: Phased Work Plan

Companion to `ARCHITECTURE.md` (v0.8). Four build phases of 1–2 days each, one builder plus a coding agent. Every phase ends at a gate: a fixed check that decides whether to continue as planned or take the listed fallback.

Days are calendar days from the start of the build: D1 = Monday 16 November 2026, D9 = Tuesday 24 November (buffer). The plan assumes 2 days per phase; a phase finished in 1 day pulls the rest forward.

**Pace.** Overtime compresses the building, not the machine time. GPU runs (tuning runs, the production run, final evaluation) take as long as they take, so queue them back to back and overnight, and run the Mac and Colab in parallel. Rough machine time per phase is in the table below; check it against the phase-1 timings.

**Baseline (v0.8, 2026-11-17):** the agent is compared with FLAP cut to the same memory limit (Baseline: FLAP). Wanda-50% and uniform Wanda-sp are optional backups, reported only if useful; the ShortGPT comparison (task 3.4) is dropped. Tasks 1.1, 1.9 and 1.10 were run when Wanda-50% was the reference and are kept as done.

**Calibration (2026-11-17):** every compared method calibrates on the same example text, 128 passages of 2,048 tokens from C4 (the set Wanda uses): the agent's scores, Baseline: FLAP and the backups. Reasons: every method sees the same data, and the example text comes from a different source than the WikiText-2 evaluation text. FLAP's own WikiText-2 training set is kept only for the check against the authors' code (task 1.2). FLAP numbers measured before 2026-11-17 (task 1.9's 15.2% of weight memory removed, and the 11.1% run with no allowance quoted in A7) used FLAP's own WikiText-2 calibration text, and are not directly comparable with later numbers. Agent round 1 (task 1.11) was re-measured with the shared set: `results/agent_round/round1_c4.md`.

**Models:** Qwen3-0.6B-Base for all development and tuning; Qwen3-1.7B-Base only for the production run (phase 4). "Qwen3-0.6B" and "Qwen3-1.7B" elsewhere mean these Base checkpoints.

---

## Overview

| Phase | Days | Goal | Gate at the end |
|---|---|---|---|
| 0. Setup | D1 (morning) | Repo, data and dense reference numbers ready | G0: dense 0.6B perplexity (how well the model predicts real text; lower is better) reproduces |
| 1. Feasibility | D1–D2 | Every assumption the design rests on checked on 0.6B | G1: go / fallback per assumption |
| 2. Agent graph | D3–D4 | End-to-end agent run on 0.6B | G2 (midpoint): one ACCEPT and one ROLLBACK in a real run |
| 3. Tuning | D5–D6 | Agent instructions tuned on 0.6B and frozen | G3: instructions frozen at a tagged commit |
| 4. Production and evaluation | D7–D8, buffer D9 | 1.7B production run, final evaluation, report | G4: report reproduces from the README |
| 5. Weight quantization (optional) | Only from leftover buffer | One-time 8-/4-bit weight quantization of the final model | Same checks as phase 4 |

---

## Phase 0: Setup (D1 morning)

| # | Task | Owner | Done when |
|---|---|---|---|
| 0.1 | Repo, `uv` lock file (PyTorch, Transformers, LangGraph, lm-eval), pytest, GitHub Actions | Coding agent | CI runs an empty test suite |
| 0.2 | Colab job runner with the Google Colab CLI (`scripts/colab_job.sh`): named GPU session, clone + `uv sync --locked`, reconnect on a dropped connection, download the result, always stop the session | Coding agent | A job runs on the T4 end to end and the session is released |
| 0.3 | Download and cache C4 calibration data (example text run through the model to measure which inputs are large; it decides which weights are removed, and is kept separate from the test text) set (128 × 2,048 tokens, fixed seed) and WikiText-2 validation and test splits | Coding agent | Cached files hashed in the manifest |
| 0.4 | Dense Qwen3-0.6B: validation and test perplexity on Mac and T4 | Coding agent | Numbers recorded |

**G0:** Mac and T4 perplexity for the dense model agree within 1%. If not, fix the evaluation code before anything else.

---

## Phase 1: Feasibility (D1–D2)

Everything runs on Qwen3-0.6B except task 1.10.

### Tasks

| # | Task | Owner | Depends on | Done when |
|---|---|---|---|---|
| 1.1 | Adapt Wanda to Qwen3 layer names; run Wanda at 50% on 0.6B, recorded as the Wanda-50% reference (a backup since v0.8) | Coding agent | 0.3 | Reference recorded: validation and test perplexity, 7 multiple-choice question sets |
| 1.2 | Adapt FLAP scoring and bias compensation to grouped-query attention (prune key/value groups with their query heads) | Coding agent | 0.3 | FLAP at its own allocation matches the adapted reference |
| 1.3 | Uniform Wanda-sp on Qwen3 | Coding agent | 1.2 | Matches its reference at a uniform cut |
| 1.4 | Pruner from a per-layer plan (all four strategies) + loader (safetensors + plan JSON) | Coding agent | 1.1, 1.2 | Save-and-reload gives identical perplexity |
| 1.5 | Device estimator from a plan: weight bytes, bytes read per generated token, prompt FLOPs; estimated tokens/s for a device target | Coding agent | 1.4 | Weight bytes match file size within 1% |
| 1.6 | Speed meter on the M4 (Mac GPU) and T4: peak memory, prefill and decode tokens/s; fix settings (prompt 512, batch 1, 128 generated); record each machine's bandwidth and compute as a device target | Coding agent; Steph fixes settings (Q3) | 1.4 | Repeat runs vary < 5% on each machine |
| 1.7 | Estimate check: ~10 plans mixing all strategies; estimated vs. measured decode speed on M4 and T4 | Coding agent | 1.5, 1.6 | Rank correlation and % error recorded per machine (A6, A8) |
| 1.8 | Layer-local probes for all four strategies; compare their layer ranking with full-model perplexity | Coding agent | 1.4 | Rank correlation recorded (A1) |
| 1.9 | FLAP and uniform Wanda-sp pruned to the largest ratio that keeps Wanda-50% quality, by bisection on validation perplexity (run with each method's published calibration; see the 2026-11-17 calibration note) | Coding agent | 1.1–1.3 | Savings at that ratio recorded (A7) |
| 1.10 | Wanda at 50% on Qwen3-1.7B (A100, one session), recorded as the Wanda-50% reference for the production run (a backup since v0.8) | Coding agent | 1.1 | Reference recorded for 1.7B |
| 1.12 | Gate subset: 32-window subset vs. full validation split on ~10 plans (A9) | Coding agent | 1.4 | Rank correlation recorded |
| 1.11 | One agent round wired by hand (no graph): sensitivity table → LLM → JSON with 3 candidates → evaluate | Steph (prompt draft), coding agent | 1.5, 1.8 | Valid JSON; candidates evaluate |

### G1: Go / fallback

| Check | Go if | Otherwise |
|---|---|---|
| References reproduce (1.1–1.3) | Within 2% of reference perplexity | Reimplement fluctuation metric and bias compensation directly (A4); spend at most half a day on it |
| Save-and-reload (1.4) | Identical perplexity | Block phase 2 until fixed |
| Probes rank like full evaluation (A1) | Rank correlation ≥ 0.7 | Probes stay as hints only; give the agent more full-model evaluations per round |
| Speed estimate ranks like measured speed on M4 and T4 (A6) | Rank correlation ≥ 0.7 on both | Measure tokens/s on the accepted plan every 5 rounds and give the agent that instead |
| Speed estimate close enough for phone estimates (A8) | Within ~25% of measured speed on both machines | Report phone targets as memory fit only |
| Structured pruning saves memory near Wanda-50% quality (A7) | FLAP saves ≥ 5% of weight memory within 10% of Wanda-50% perplexity and 2 points of its accuracy (the v0.6 allowance this check was set with) | Continue; headline result becomes how the agent mixes strategies |
| 32-window subset ranks like the full split (A9) | Rank correlation ≥ 0.9 | Gate on 64 windows |
| The Wanda-50% reference is not trivial | Wanda-50% validation perplexity > 1.3 × dense | Fall back to a weaker reference only if structured pruning matches it at > 30% saving; fixed before phase 4 |

---

## Phase 2: Agent Graph (D3–D4)

### Tasks

| # | Task | Owner | Done when |
|---|---|---|---|
| 2.1 | State types (`RunConfig`, `Baseline`, `LayerPlan`, `PruneState`, …) per ARCHITECTURE.md §4.3 and §7 | Coding agent | Type-checked |
| 2.2 | Validator: structure, step-size and budget rules (§4.4) | Coding agent | Unit tests for every rule |
| 2.3 | Agent client with two backends behind one interface: Codex CLI (`codex exec`, read-only, sessions saved for usage-limit readings, JSON schema, pinned model) and LLM API (structured output, cost cap); shared cache and usage ledger (60 calls per run; $5 per run on the API) | Coding agent | Same contract tests pass for a fake and both real backends; cached rerun makes no calls; malformed answers rejected |
| 2.4 | Graph: score → feasibility → baseline (FLAP cut to the memory limit) → probe fan-out → reduce → plan → validate → candidate fan-out → gate → rollback / stop → final | Coding agent | Graph compiles; each edge tested with a fake LLM |
| 2.5 | Candidate evaluation (≤ 3): memory and speed estimate from the plan; perplexity on the 32-window subset, skipping the unchanged layers before each candidate's first change (their output is saved once from the accepted model; every layer from the first change onward is recomputed); candidates spread across Mac and Colab | Coding agent | Skipping unchanged early layers gives the same perplexity as running every layer; same results in parallel and sequential mode |
| 2.6 | Gate: 15% per-step limit; among candidates within it, pick the smallest perplexity rise per MB of memory freed (one that frees nothing cannot win), ties to the faster plan | Coding agent | Gate unit tests |
| 2.7 | Rollback to any earlier accepted plan | Coding agent | Rollback restores exact earlier perplexity |
| 2.8 | Checkpoints (SQLite + plan JSON on Drive) and decision log (JSONL) | Coding agent | Log schema test |
| 2.9 | Agent instructions v1 (`src/pruneagent/prompts/instructions_v1.txt`, approved by Steph): objective and Baseline: FLAP stated, three methods described neutrally (Wanda sparsity off the agent's menu: it frees no memory), the gate's real rules (15% per-step limit, smallest perplexity rise per MB freed, stop rules), full candidate history; every placeholder filled by code (`instructions.py`); replies in either field naming; v1 actions are propose candidates or roll back (probe, start from FLAP and stop off behind a flag) | Steph (text), coding agent (code) | Reviewed against ADR-007 neutrality rule; tests: no unfilled placeholder, numbers match the gate, aliases, Wanda and disabled actions rejected, fake run |
| 2.12 | Usage limits: pre-run estimate, live limit readings after every Codex call, pause at 90% or on limit / API cap, `pruneagent resume` after reset or on the other backend (FR15) | Coding agent | Tests with recorded limit readings: fits, warns, pauses, resumes without repeating a call |
| 2.10 | Pick the default backend and pin its model (and Codex reasoning effort); trim Codex's context with a dedicated profile | Steph | Pinned in config; tokens per call recorded |
| 2.11 | Device target: feasibility check with early warning, stop when the target is met, final warning with closest plan and gap | Coding agent | Unit tests with an impossible, an already-met and a reachable target |

**G2 (midpoint):** one real end-to-end run on Qwen3-0.6B with at least one ACCEPT and one ROLLBACK, under 2 h and 60 agent calls. If not reached by end of D4, cut to one candidate per round (descoping step 3) and retry the next morning.

---

## Phase 3: Tuning (D5–D6)

| # | Task | Owner | Done when |
|---|---|---|---|
| 3.1 | Tuning runs on 0.6B (~8–10 runs, queued overnight; Mac and Colab in parallel): adjust instructions, compare quality with Baseline: FLAP cut to the same memory limit | Steph | Each run logged with instruction version |
| 3.2 | Neutrality check on every instruction change: no strategy ranked or discouraged | Steph | Diff reviewed |
| 3.3 | Kill-and-resume test (stop the Colab runtime mid-run) | Coding agent | Resumed run matches an uninterrupted run |
| 3.4 | ~~ShortGPT cut to the same memory limit~~ Dropped 2026-11-17 (v0.8): one primary baseline, FLAP | — | — |
| 3.5 | Report writer: quality, memory, speed, strategy counts, rationale log, charts | Coding agent | Report builds from a tuning run |
| 3.6 | Web application on top of the tools (ADR-008, FR16): FastAPI server with job runner, React interface (new run, runs, live run, results, tools); also the demo (Q2), replaying a finished 0.6B run. Replaces the demo notebook | Coding agent | A 0.6B run started, watched, paused and resumed in the browser matches the same run from the command line; API and interface tests pass |
| 3.7 | 1.7B evidence: C4 scores, quick tests and the measured per-layer damage table for Qwen3-1.7B, then a round-1 dry run (no agent call) | Coding agent | Mac GPU, ~6 h; evidence saved; dry run renders round 1 |

**G3:** instructions frozen at a tagged commit by end of D6, whatever the tuning results. No changes after this point. A tuning result below FLAP is reported, not chased (S4).

---

## Phase 4: Production and Evaluation (D7–D9)

| # | Task | GPU | Done when |
|---|---|---|---|
| 4.1 | Baseline: FLAP cut to the memory limits used on 1.7B (the production run's and the out-of-reach run's); uniform Wanda-sp at the same limits optional | A100 | Plans saved; each fits its limit |
| 4.2 | Production run on Qwen3-1.7B with frozen instructions at a set memory limit | A100 | Run complete; stops at the limit; log and plan saved |
| 4.3 | Device-target run on Qwen3-1.7B at a memory limit pruning cannot reach | A100 | Ends with a warning and the gap |
| 4.4 | Final evaluation of the agent, Baseline: FLAP and the unpruned model (and any backup run): test perplexity, 7 multiple-choice question sets, MMLU | A100 | Scores recorded |
| 4.5 | Memory and tokens/s for every method evaluated in 4.4 | M4 + T4 | Measurement per method per machine |
| 4.6 | Score S1–S4 and prediction P1; device reports for 4.3 | — | Table in the report |
| 4.7 | Report, README with reproduction steps, demo | — | A fresh clone reproduces the report from saved logs |
| 4.8 | Buffer | — | D9 kept free |

**G4:** report reproduces from the README.

---

## Phase 5: Weight Quantization (optional)

Only if the D9 buffer is unused. A one-time step after the search: quantize the final model's weights to 8-bit and 4-bit (activations and key/value cache stay in fp16), then repeat test perplexity, memory and tokens/s on the M4 and T4. Wanda is not quantized.

## Later (bonus, not in this build)

Sweep the tool across a range of device targets to map how much quality pruning alone keeps at each target and where it warns: a sensitivity analysis of the tool itself.

---

## Budget by Phase

| Phase | Machine time (rough) | Colab compute units | Agent calls (API backend: ~$1–5 per run) |
|---|---|---|---|
| 0 | < 1 h | ~2 | 0 |
| 1 | ~4–6 h GPU (Wanda-50% references, FLAP and Wanda-sp searches, probe and speed checks) | ~25 (incl. one A100 session for 1.10) | ~10 |
| 2 | ~2–3 h GPU (test runs) | ~20 | ~200 |
| 3 | ~8–10 h GPU (tuning runs) | ~45 | ~600 |
| 4 | ~10–12 h GPU (production run, 1 out-of-reach device-target run, Baseline: FLAP, final evaluation) | ~90 (A100 for 4.1–4.4) | ~150 |
| Reserve | | ~18 | |
| **Cap** | | **200** | **60 per run** |

**Early warning:** more than 50 units used by the end of phase 2 → T4 only for the rest of the project, with candidates evaluated in sequence (ADR-003).

## Descoping Order

Applied in this order when a gate slips by more than half a day:

1. Backup baselines (uniform Wanda-sp, Wanda-50%)
2. Device-target runs on 0.6B instead of 1.7B
3. One candidate per round instead of three
4. MMLU
5. Qwen3-0.6B only (no production run on 1.7B)

## Steph's Decisions by Phase

| Phase | Decision |
|---|---|
| 1 | Speed settings (Q3); accept or apply each G1 fallback |
| 2 | Default agent backend and model; instructions v1; device targets for the two phase-4 runs (one reachable, one not), set from phase-1 results |
| 3 | When to freeze instructions (no later than end of D6) |
| 4 | ~~Demo format (Q2)~~ Decided 2026-11-19: the web application (ADR-008) |
