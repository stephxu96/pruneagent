# Tuning round 2: best of 2 instead of rewording

Decided 2026-11-19 with Steph, after tuning round 1.

## Why we stopped rewording

The same instructions (v2) at a 20% cut gave 22.98 and 25.20 on unseen text. That 2-point spread from chance is
larger than any gap between instruction versions so far (v1 scored 23.22), and telling a 1-point gain apart from
chance would take about 30 runs per version (~40 hours of Mac time each). Runs differ because each is 20 to 40
decisions in a row, and one early unlucky cut changes every choice after it (v2-b undid rounds 10 to 29 at round 30).

## What we do instead

Run the agent twice at the same limit and keep the run with the lower error on its own sample text
(`pruneagent best-of RUN_A RUN_B --name NAME`, record in `runs/best_of/NAME.json`).

- The pick uses only the error each run measured during the run. The unseen validation windows and the test text
  are not used, so the winner's later scores are fair, and the comparison with FLAP is unchanged.
- The held-out check and the question sets run on the winner only.
- Cost: twice the Mac time per result (about 2.5 to 3 hours at a 20% cut on Qwen3-0.6B).

## Evidence so far

| Pair | Run | Sample-text error | Unseen-text score | Kept |
|---|---|---|---|---|
| v2 | 20261119-tune-v2-a | 22.54 | 22.98 | yes |
| v2 | 20261119-tune-v2-b | 24.37 | 25.20 | |

Average of a single run: 24.09. Best of 2: 22.98. The pick chose the run that was also better on unseen text.

Next: the v1 pair (20261118-1718 and the fresh repeat 20261119-tune-v1-c) checks the same thing for v1.

## Later options (not started)

- Keep the best two versions each round, so a bad step doesn't trap the run.
- A final touch-up at the target size: put back neurons where quality fell most, cut elsewhere.
