# Prefix sidecar serialize −9.5% per snapshot, end-to-end null — CUDA, 2026-08-07

> Status: **mechanism confirmed, end-to-end null. Kept, not reverted.**
> Counterbalanced A/B, single-mechanism delta, identical event count and payload
> in all four arms. The operation is measurably cheaper; it is too small a share
> of wall clock for that to reach throughput or latency.

## Problem

`Qwen35RecurrentSnapshot` is written at every stride boundary of every prefill so
a later conversation can restore the hybrid prefix. The payload is the whole
recurrent state — 48 linear layers × (3 MiB gdr f32 + 60 KiB conv bf16) =
146.8 MiB, independent of context length.

`to_bytes` walked it one element at a time (`extend_from_slice(&x.to_le_bytes())`
per f32, 37M calls per snapshot) and `from_bytes` rebuilt it with
`chunks_exact(4).map().collect()`. Both are bulk byte copies in `d626a1b03`, the
idiom `attention/prefix_state.rs::push_bf16` already used.

## Parameters

```bash
arle serve --backend cuda --model-path ThinkingCap-Qwen3.6-27B-FP8 \
  --spec-type dspark --mtp-draft-model Qwen3.6-27B-DFlash \
  --dspark-block-size 6 --max-running-requests 16 --port 18701

python3 scripts/bench_throughput.py --url http://127.0.0.1:18701 \
  --prompts-jsonl bench-short-16x2.jsonl --concurrency-grid 1,8,16 \
  --requests-per-concurrency 96 --max-tokens 512 --temperature 0 \
  --seed 20260416 --timeout-seconds 900
```

- 1× H20 GPU 1, TP=1, eager, 16 slots, `RUST_LOG=info`.
- Arm D = `d626a1b03^` serializer (per element), arm E = `d626a1b03` (bulk).
  Both arms carry the same `to_bytes` timing log, so the mechanism is read
  directly off the two serve logs rather than inferred from throughput.
- **Four sweeps in the order D, E, E, D** so each arm runs once in each half.
- Decode-heavy short prompts on purpose: the snapshot fires ~48× per 20 s here
  against ~9 on the 32K anchor, which would bury the effect under its 5.7%
  same-config spread.
- 90/96 complete at every point in all four arms — uniform, cause unknown, and
  identical across arms so it does not bias the contrast.

## Results

**Mechanism** — `n` and payload identical in all four arms, so the contrast is clean:

| arm | n | payload | mean serialize | total |
|---|---:|---:|---:|---:|
| D1 | 578 | 146.8 MiB | 83.5 ms | 48.25 s |
| E1 | 578 | 146.8 MiB | 75.5 ms | 43.65 s |
| E2 | 578 | 146.8 MiB | 77.3 ms | 44.66 s |
| D2 | 578 | 146.8 MiB | 85.4 ms | 49.35 s |

Position-averaged **D 84.45 ms → E 76.40 ms, −9.5%**. Both halves agree in sign
and magnitude (−8.0 ms, −8.1 ms) and the D arms bracket the E arms with no
overlap.

**End to end** — position-averaged, null on every metric:

| c | D itl_mean | E itl_mean | Δ | D itl_p99 | E itl_p99 | Δ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.92 | 8.92 | 0.0% | 40.5 | 40.4 | −0.2% |
| 8 | 30.85 | 31.05 | +0.6% | 496.1 | 498.6 | +0.5% |
| 16 | 57.07 | 58.58 | +2.6% | 671.2 | 610.2 | −9.1% |

Within-arm spread at c=16 is 2.7% (D 56.31/57.82, E 57.79/59.36), so every one
of these sits inside the noise.

The arithmetic says it has to: bench wall is ~512 s per arm, the sidecar
serialize is **9.4% of it** (48.25 s), and −9.5% of that is **0.9% of wall**.

## Two wrong predictions, and why

**Predicted the per-element loop was ~half the event; it was 9.5%.** 146.8 MiB
in 76 ms is 1.9 GB/s — the residual is allocating and first-touching 147 MiB of
fresh heap plus the copy itself. The bulk change removes the loop's call
overhead and nothing else. Making this materially cheaper means not making the
copy at all (serialize straight out of the pinned D2H staging into the tier
buffer), not making the copy faster.

**Predicted the sidecar was ~45% of wall; it is 9.4%.** The 45% came from an
nsys window of 19.92 s in which 8.98 s sat in 79 stalls. That window was
*aimed* at decode, where the cost concentrates. A window selected for a
phenomenon reports that phenomenon's share of the window, not of the run — a
selected window is not a sample. This is the same failure as the earlier
"profiler share-of-time" null, one level up: there the profiler distorted the
denominator, here the denominator was chosen.

## Kept, not reverted

Strictly less work, cannot regress, and −9.5% on the operation is real and
measured. Recorded as an end-to-end null so the next change is not sized off it.

## What the measurement actually surfaced

578 snapshots × 146.8 MiB = **83 GB of host serialization per 512 s bench**, and
that is 9.4% of wall clock before counting the D2H and the tier insert. The
open question is not how fast the copy is but whether the sidecar earns it —
its restore hit rate is unmeasured, and the payload is fixed at 146.8 MiB by the
model's 48 linear layers regardless of how much prefix is being cached.

## Learnings

**A window selected for a phenomenon overstates that phenomenon's share.** Size
a fix against the run, not against the window where you went looking for it.

**An instrument in both arms beats an inference from throughput.** The timing
log made a 0.9%-of-wall change readable as a clean −9.5% on the operation; no
throughput A/B on this box could have resolved it against a 2.7% spread.
`kv_tier_stats.demoted_slots` existed and was never surfaced, and the sidecar
had no counter at all — three costs today with no instrument that could notice
them.
