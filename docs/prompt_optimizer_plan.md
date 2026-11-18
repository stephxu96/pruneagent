# Prompt optimizer for tuning (task 3.1)

Agreed with Steph on 2026-11-17. Used in phase 3 to improve the agent's instructions, one iteration at a time. Build it after the midpoint run (G2) shows the loop works end to end.

## How it works
Each iteration, an LLM receives the current instructions, the change history and evidence from recent runs (decision logs, final perplexity against Baseline: FLAP, memory reached, rounds used, rejected candidates, invalid answers, the agent's reasons). It returns JSON with a diagnosis (quoting the log lines), at most 3 edits (each with the behaviour it should change and the log signal that would confirm it), problems that belong in code instead of the instructions, the revised instructions, and a change-log entry.

Constraints it must follow: never tell the agent which method or layer to choose; never steer it toward copying FLAP; every statement must be true of the system (no advice that assumes fine-tuning); don't undo an earlier edit unless the change log shows it hurt; no model-specific numbers; flag edits backed by one model only; if the evidence is too thin, propose a measurement instead of an edit.

The full draft prompt Steph supplied is kept in `docs/prompt_optimizer_draft.txt`.

## Changes agreed before use
1. **Match the current system.** No `wanda_sparsity` (removed from the agent's menu in v1); the rejection, selection and stop rules and the per-step memory are filled by code from the real values, as in the v1 instructions, instead of the draft's FILL notes.
2. **No tuning to the test.** Edits are proposed from runs scored on the gate's 32 validation windows, then confirmed on different held-out validation windows before being kept. Instructions are frozen before the 1.7B production run.
3. **Repeats.** At least 2–3 runs per instruction version before an edit is judged.
4. **Steph approves every edit** (the human check of task 3.2).

## Cost
About 2–3 full agent runs on Qwen3-0.6B per iteration, roughly 2 hours each on the Mac GPU plus Codex calls; no Colab. A few iterations fill the overnight queue planned in 3.1.
