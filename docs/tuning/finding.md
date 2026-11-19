# Rewording the agent's instructions doesn't reliably help

**Question.** Do small changes to the agent's instructions give a better pruned model?

**What we did.** Qwen3-0.6B, 20% smaller (about 954 MB). We changed one paragraph of the instructions (v1 → v2): smaller
steps near the target, and choices in one step add up. We ran each version and scored the result on text the agent
never saw (lower is better). Each run asked the AI afresh: no saved answers reused.

**What we found.**

| Instructions | Run | Score on unseen text |
|---|---|---|
| v1 | 20261118-1718 | 23.22 |
| v1 | 20261119-tune-v1-c | (running) |
| v2 | 20261119-tune-v2-a | 22.98 |
| v2 | 20261119-tune-v2-b | 25.20 |
| FLAP, same size | | 175.07 |

The same instructions gave scores 2.2 apart. That chance spread is bigger than the gap between versions, so a small
rewording can't be shown to help: telling a 1-point gain from chance would take about 30 runs per version. Every
run, lucky or not, beats FLAP by about 7×.

**What we'll do.** Keep the v1 instructions. No more rewording. Report agent results with their run-to-run spread
where we have repeats.
