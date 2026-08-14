# Raw-completion continuation style flips with concurrency (DSv4, TP=8)

## Context

Re-anchoring the DSv4-Flash-FP8 8×H20 DSpark baseline at `fad8f4d5b`
(120 s/point, greedy, seed 42, 64 synthetic prompts over raw
`/v1/completions`). The c=16 point reported 7/241 `correctness_failed`; c=1 and
c=8 reported none.

## Phenomenon

All seven failures come from one prompt, `Explain bloom filters and their use
cases.`. Its continuation differs by concurrency, at temperature 0:

| c | first tokens of the completion |
|---|---|
| 1 | ` A: A Bloom filter is a space-efficient probabilistic data structure…` |
| 8 | `. 2023-05-23. https://…/bloom-filters-introduction… - [2] Bloom, B. H. (1970)…` |
| 16 | `. C. 2024-05-15: 2024-05-15: 2024-05-15: …` |

The divergence starts at the first token — ` A` against `.` — so the whole
continuation follows from one flip, not from drift partway through. At c=16 the
same prompt runs 23 times and 16 of those pass the runner's diversity check;
those 16 are the citation-list form, degenerate in the same way but varied
enough to clear the heuristic. The gate's 7 is the tail of a distribution, not
the whole of it.

## What is established

The runner posts raw `/v1/completions` with no chat template, so a bare
question has two genuinely competing continuations: answer form (`A: …`) and
document form (a Wikipedia body plus reference list). `ignore_eos=true` then
forces 128 tokens, so the document form runs into repeated citation dates.

## Cause — resolved 2026-08-15

The logit-bias threshold measurement
(wins/2026-08-15-dsv4-first-token-flip-near-tied-pair.md) established the
near-tied arm: c=1 top-2 margin < 0.125 logit units, c=16 unstable at zero
bias (6/10 vs 4/10), and the winner also flips between TP=8 and TP=4. The
pair is tied within FP8 MoE + reduction-order noise; no batching defect.
The section below is the pre-measurement state, kept as filed.

## Cause unknown (as filed 2026-08-14)

Whether the first-token flip is batch-dependent numerics on a near-tied logit
pair, or a batching defect, is **not** established. The two carry opposite
remedies, and nothing measured here separates them. What would: dump the top-2
logits for that first position at c=1 and c=16 and compare the margin against
the observed FP8 MoE noise floor. A tie inside the noise floor makes it a
knife-edge prompt; a wide margin that still flips makes it a defect.

## Rule

A `correctness_failed` count from the throughput runner bounds the failures
from below, never from above — it is a diversity heuristic, so the degenerate
responses that stay varied are not counted. Read the decoded text of the
flagged prompt across the whole grid before assigning a verdict, and do not
treat a clean count at low concurrency as evidence that the same prompt is
clean at high concurrency.
