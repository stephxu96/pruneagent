# Does thinking harder make Codex a better pruning planner?

**Headline: yes, somewhat. When we asked Codex to think hard ("high" effort), it gave the best plan of the three
settings. The pruned model's error score was 20.45, against 21.52 for medium and 21.95 for low. That narrows the gap to
the standard method we compare against (Baseline: FLAP, 18.84) from 17% to 9%, but no setting beat it. High effort
took about 80 seconds longer per answer and wrote about 80% more text.**

## The question and why it matters

PruneAgent asks Codex, an AI assistant, to plan which parts of a language model to cut so that the model fits a
memory limit and loses as little quality as possible. We can tell Codex how much effort to put into reasoning before
it answers: low, medium or high. More effort costs more time and more of our Codex allowance in every round of every
run, so we want to know whether it produces better plans. So far we have used low.

## What we did

We sent Codex the same request three times, once at each effort level, and changed nothing else. All three used the
same written request (the one from the first agent round, with memory figures given to three decimal places), the
same Codex model (gpt-5.6-terra), the same settings and the same required answer format. The request gives Codex:

- the target: remove at least 119.21 MB, 10% of the model's 1,192.1 MB;
- four ways to cut: narrow part of a layer, remove part of a layer's attention, delete a whole layer, or set some
  weights to zero (which saves no memory);
- a table of quick tests showing how much each cut, on its own, disturbs each of the model's 28 layers.

Each time, Codex answered with three alternative plans. Code checked every plan against the fixed rules, built the
pruned model and measured its error score on the same 32 passages of text used in earlier rounds. The error score
(perplexity) measures how surprised the model is by text it has not seen: lower is better. The unpruned model scores
13.15 on these passages.

## Results

| Effort | Best plan's error score | vs Baseline: FLAP (18.84) | Other two plans | Tokens read / written (of which thinking) | Seconds per answer |
|---|---|---|---|---|---|
| Low | 21.95 | 17% worse | 22.54, 22.56 | 5,191 / 4,190 (2,528) | 105 |
| Medium | 21.52 | 14% worse | 21.69, 22.06 | 5,191 / 4,797 (2,566) | 118 |
| High | **20.45** | **9% worse** | 20.97, 21.22 | 5,191 / 7,649 (5,779) | 186 |

Tokens are the units of text Codex reads and writes; they are what counts against our allowance. All nine plans
followed the rules and each removed between 119.5 and 120.3 MB, just over the 119.21 MB needed.

## What we found

- **Every high-effort plan beat every low- and medium-effort plan.** The worst high-effort plan (21.22) scored better
  than the best medium-effort plan (21.52).
- **Low and medium were close.** Medium spent almost exactly as much on thinking as low (2,566 tokens against 2,528),
  and its best plan was only slightly better. High spent more than twice as much.
- **All three settings used the quick-test table sensibly.** None of the nine plans touched the first two layers,
  which the table shows are the most easily disturbed. All of them spread small cuts over the middle layers, where the
  table shows cuts do the least harm. When a plan deleted a whole layer, it chose layer 3, 4 or 5, the three layers the
  table shows can be removed with the least disturbance.
- **The settings differed in style.** At low effort, all three plans deleted one layer and made almost every other
  cut by narrowing. Medium never deleted a layer and removed a lot of attention instead (19 to 21 removals per plan).
  High mixed the two: two plans deleted a layer and one did not, each with a moderate amount of attention removed. Its
  best plan deleted layer 4.
- **Codex's own memory totals were slightly off.** Each plan's explanation states how much memory it frees, and these
  figures were up to 0.06 MB too low. Code computes the real figure, so the plans were unaffected, but the
  explanations are not reliable for exact numbers.

## Recommendation

Make **high** the default effort. It gave the best plans by a clear margin, for about 80 extra seconds and about
3,500 extra tokens per round. Over a 20-round run that adds roughly half an hour of waiting, compared with about an
hour spent measuring plans. Medium is not worth choosing over low: it cost a little more and gave almost the same
plans.

## Caveat

This is one answer per setting. Codex does not give the same answer twice, so part of the difference could be chance.
The best high-effort plan beat the best medium-effort plan by more than the spread among the three plans within any
single answer. Only repeating each setting a few times would show how reliable the difference is.

## Side note: trimming

This note is separate from the effort comparison. The first agent round asked the same question at low effort through
the older Codex setup, before we removed Codex's built-in instructions and tools from each request. Its best plan
scored 19.69 (5% worse than Baseline: FLAP), better than any plan here. We measured that plan again with this check's
code and got 19.69 again, so the measurement is not the cause. Unlike every plan here, that answer also cut the
second layer. A few repeats would show whether the trimmed setup gives worse plans or that answer was simply lucky.

Raw data: `effort_check.json` (replies, choices, explanations, token counts and every measurement). Code:
`scripts/effort_check.py`.
