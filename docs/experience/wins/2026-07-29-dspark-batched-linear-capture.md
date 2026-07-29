# The spec linear capture batches too — c=16 DSpark reaches +17% over no-spec

## Context

The [batched snapshot](2026-07-29-dspark-batched-state-snapshot.md) took 54% of
the tick's D2D calls. The other 46% is the spec linear capture: three copies
(qkv, b_proj, a_proj) per row per layer, pulling each chain's rows out of the
packed verify projections — 48 × 3 × B per tick.

## What Worked

Their sources are computed before the row loop, so the copies hoist out of it.
Chain lengths differ per row, so `batched_copy_uniform` gained an optional
per-buffer 16B-word count; one launch per projection replaces `3 * B` memcpys
per layer.

## Measurement

Matched, one binary per arm, GPU 0, ThinkingCap-Qwen3.6-27B-FP8 + 27B-DFlash,
block 6, 48 req/point, max_tokens 214, seed 20260416. No `--spec-max-batch` —
the shipped default of 16.

| c | no-spec | +batched snapshot | **+batched capture** | TPOT no-spec | snapshot | **capture** |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 13.9 | 18.9 | **19.0** | 28.86 | 10.40 | **10.28** |
| 2 | 47.1 | 86.2 | **88.6** | 37.38 | 18.76 | **18.27** |
| 4 | 60.7 | 106.6 | **111.1** | 59.26 | 31.56 | **30.63** |
| 8 | 94.5 | 122.9 | **129.7** | 71.93 | 55.89 | **52.94** |
| 16 | 122.7 | 133.3 | **143.2** | 101.88 | 95.04 | **88.51** |

The step is +0.5 / +2.8 / +4.2 / +5.5 / **+7.4%** — monotone in concurrency,
which is the shape a `3 * B`-per-layer cost predicts. Gate exact=3 DET at
512/4k/16k, 0 errors.

Shipped, against no-spec: **c=1 +37% / c=2 +88% / c=4 +83% / c=8 +37% /
c=16 +17%** tok/s; TPOT −64 / −51 / −48 / −26 / **−13%**.

`output_tokens_per_s` is not decode speed on this fingerprint: 48 requests carry
1.53M prompt tokens against 7,700 output tokens, so at c=1 each request spends
6.2 s in prefill and 1.9-5.0 s decoding — the column measures prefill there. Per
request, decode is `1000 / TPOT`:

| c | no-spec | DSpark | |
|---|---:|---:|---:|
| 1 | 34.6 | **97.3** | 2.81× |
| 2 | 26.8 | 54.7 | 2.04× |
| 4 | 16.9 | 32.6 | 1.93× |
| 8 | 13.9 | 18.9 | 1.36× |
| 16 | 9.8 | 11.3 | 1.15× |

## Still open

Speculation now pays everywhere, but the margin still falls with concurrency
(+88% at c=2, +17% at c=16) because verify rows are the one cost that scales
with B and cannot be batched away — block 6 commits 2.21 tokens per 6 verify
rows against plain decode's 1 token per row. The next axis is acceptance, not
launches: the DSpark head's train sidecar, whose `prob_match` normalization fix
is still owed an acceptance A/B.

## Rule

**Do not headline `output_tokens / wall` on a prefill-heavy fingerprint.** At
c=1 this set is 55-76% prefill, so the column reads as a +37% decode result
where decode actually went 2.81×. TTFT and decode are separate SLOs; a table
that mixes them hides the win it is reporting.

**A per-(row, layer) copy is a `B * L` launch pile wearing a memcpy's clothes.**
Three of them looked like bookkeeping next to a 27B forward and cost 7.4% of
c=16 throughput. The tell is the shape: a fix whose gain rises monotonically
with concurrency was a per-row cost, and one that is flat was not.
