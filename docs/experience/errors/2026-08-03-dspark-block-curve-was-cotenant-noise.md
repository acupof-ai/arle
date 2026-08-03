# The DSpark block-size curve was co-tenant noise — 2026-08-03

> Retracts an unshipped same-day claim ("block 16 leaves 34% on the table,
> 4–8 is the optimum"). The claim was never licensed and no default moved,
> so nothing in the tree needs reverting. The measurement method does.

## Context

A sweep of `--dspark-block-size` on Qwen3.6-27B-FP8 + DFlash, c=1, produced a
curve where the shipped default (16) placed **last** at 1.17× while block 4 hit
1.57×. I wrote it up as "the default is mistuned", flagged n=1 as the one gap,
and noted a confound in what I argued was the safe direction: the block-16 point
ran with 1 co-tenant serve on the box and the 4/8 points with 4, so the *loser*
had the friendlier conditions.

The matched repeat inverts the entire ordering. Empty box, temperature 0
(so every arm emits exactly 768 tokens and wall-clock is the only free
variable), 2 interleaved reps, one serve at a time:

| `--dspark-block-size` | rep 1 | rep 2 | tok/s | vs off |
|---|---:|---:|---:|---:|
| off | 14.20 s | 14.23 s | 54.0 | 1.00× |
| 4 | 8.00 | 8.09 | 95.4 | 1.77× |
| 6 | 6.69 | 6.62 | 115.3 | 2.13× |
| 8 | 6.71 | 6.62 | 115.2 | 2.13× |
| **16 (default)** | **6.15** | **6.29** | **123.4** | **2.29×** |

**The default is the optimum**, and wall-clock is monotone in block size over
the measured range — the same direction `tok/step` already pointed. The
"marginal draft forward stops paying past block 8" mechanism I wrote down
was an explanation invented for an artifact.

## Root cause

Two noise sources, and I reasoned about only one.

**Co-tenancy was not a safe-direction confound, it was the signal.** I checked
the sign of the bias assuming a co-tenant is a uniform tax. It isn't: DSpark's
block step is one large verify forward, and the SM contention a co-tenant
creates lands proportionally harder on the arm doing the most work per step —
which is precisely the arm with the largest block. So the confound scaled *with
the treatment variable*. That is not a bias with a knowable direction; it is
confounded design.

**temp 0.7 let output length float.** Each arm generated a different number of
tokens over a different trajectory, so tok/s mixed a rate difference with a
length difference. At temp 0 the trunk is greedy and every arm emits the
identical 768 tokens, which also gave a free determinism check — the md5 was
stable across reps within an arm.

(Aside, expected: md5 differs *across* block sizes. Verify runs `chain_rows`
rows through batched GEMMs whose reduction order differs from a 1-row decode,
so a near-tie can land differently and the tail diverges. Correct inference is
not byte-identity — the reps-within-arm identity is the check that matters.)

## Rule

**A confound that scales with the treatment variable has no safe direction.**
Before writing "confound, in the safe direction", ask whether the nuisance
factor interacts with the treatment or merely adds to it. If a co-tenant, a
cache state, or a thermal condition hits the arms *unequally in proportion to
what you are varying*, the sign argument is worthless — rerun clean.

And: for a throughput sweep, pin the output length (temp 0) so wall-clock is
the only free variable. It costs nothing and removes a whole noise axis.
