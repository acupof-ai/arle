# DSpark long-agent anchor re-measured: c=1 +40%, c≥4 down — CUDA, 2026-08-06

> Status: **c=1 accepted, c≥4 open.** The anchor audit against the archived
> champion binary is the deciding measurement and is recorded below once it
> lands. Do not edit the `docs/baselines.md` row until it does.

## Goal

`docs/baselines.md`'s DSpark long-agent row is `51985031d` (2026-07-30) and is
three accepted prefill changes stale:

| change | commit | measured effect |
|---|---|---|
| chunked GDR default-on | `c2eb5de9e` (08-02) | 33K prefill −27% |
| FlashQLA actually compiled into the pod binary | `0ac780495` (08-05) | TTFT 31.08 → 25.01 s |
| snapshot stride 2048 → 8192 | `301d0c074` (08-06) | TTFT 25.3 → 23.0 s |

Rule 1 of that file replaces a row on any effect over ~10%. Three landed and
the row did not move.

## Parameters

```bash
python3 scripts/gen_bench_prompts.py bench-agent-32k-16x8.jsonl 16 32000 214 8

arle serve --backend cuda --model-path ThinkingCap-Qwen3.6-27B-FP8 \
  --spec-type dspark --mtp-draft-model Qwen3.6-27B-DFlash \
  --dspark-block-size 6 --max-running-requests 16

python3 scripts/bench_throughput.py --prompts-jsonl bench-agent-32k-16x8.jsonl \
  --concurrency-grid 1,2,4,8,16 --requests-per-concurrency 128 \
  --max-tokens 214 --temperature 0 --seed 20260416 --timeout-seconds 900
```

- Binary: `b8d390bf3`. 1× H20 GPU 6, TP=1, eager, 16 slots.
- Prompt tokens: 34782/request against a 32000 target — **+8.7%, inside the
  ±10% bar**. The recorded row's p50 was 34963, so the dataset matches.
- 128/128 complete at every point, 0 errors, both sweeps.

## Results

Two identical sweeps. Sweep 1 ran with a co-tenant holding 62–63 GB on GPUs 4
and 5; sweep 2 ran on an idle box.

| c | sweep 1 (contended) | sweep 2 (idle) | spread |
|---:|---:|---:|---:|
| 1 | 10444.2 | 10406.8 | −0.4% |
| 2 | 20484.9 | 20752.5 | +1.3% |
| 4 | 24666.9 | 24327.3 | −1.4% |
| 8 | 29450.8 | 28669.5 | −2.7% |
| 16 | 31313.7 | 30486.2 | −2.6% |

**The co-tenant hypothesis is dead**: the idle box measured marginally slower.
Run-to-run spread on this workload is **±2.7%**, which confirms the file's
stated ±3% drift band on this workload rather than assuming it.

Against the recorded row, median of the two sweeps:

| c | recorded (`51985031d`) | measured | Δ | vs ±2.7% band |
|---:|---:|---:|---:|---|
| 1 | 7440.7 | **10425** | **+40.1%** | far outside |
| 2 | 8292.3 | 20619 | — | `pt` differs, not comparable |
| 4 | 25432.8 | 24497 | −3.7% | just outside |
| 8 | 31754.1 | 29060 | **−8.5%** | outside |
| 16 | 32559.0 | 30900 | −5.1% | outside |

TTFT, c=1: cold **19.3 → 10.8 s (−44%)**, warm 1.1 → 0.94 s. Both sweeps agree
to 15 ms on the cold slice (p90 10760 / 10788 ms).

TTFT, c=16: warm flat at 1.2 → 1.24 s, cold **6.8 → 8.27 s (+22%)**.

`accept_rate` 0.3074 / 0.3085 across the two sweeps.

## Problems

**`pt` is not matched to the recorded row.** That row measured c=1 and c=2 as a
fresh serve's first point, c=4 and c=8 as second points, and c=16 as a third;
both sweeps here run one serve ascending, so the points are 1st through 5th.
Its own note records that `accept` and prefix hit track `pt` rather than `c`,
with +70% accept at matched c=16 from cache state alone.

The direction of that difference works against the measured arm: points 3–5
inherit a warmer cache than points 2–3 and should be faster. They are slower.
So `pt` does not explain the c≥4 deficit, which is why the audit below is
needed rather than a re-run with matched `pt`.

**c=2's +148% is not a result.** Recorded c=2 was a fresh serve's sole point
(cold), measured c=2 is the second point of a warm serve. Two different
workloads.

## Anchor audit

Rule 4 of `docs/baselines.md`: one A/B against the archived binary bounds
accumulated drift. `51985031d`'s binary is archived at
`/host/spec-phase/arle-mk`, so this needs no rebuild and no bisect across the
~800 commits between the two shas. Same box, same dataset file, same serve
flags, same grid, back to back.

Two outcomes, two different actions:

- `arle-mk` reproduces ~31754 at c=8 today → the deficit is a real regression;
  the row records it and the next step is bisecting the ~800 commits.
- `arle-mk` also lands near 29060 → the recorded row was measured under a
  different fingerprint; the row is replaced and no regression is filed.

Result: **pending**, this session.

## Learnings

**A stale anchor's step budget is worse than its throughput row.** The
throughput numbers are merely old. The budget section still ranks
`gated_delta_rule_prefill_recurrent` as prefill #1 at 9.37 s / 33%, and that
kernel is off the default path since 08-02 — its replacement measured 1.06 s.
Anyone ranking prefill work off that table optimizes a kernel that no longer
runs.

**Measure the drift band, don't inherit it.** The ±3% in the file is a stated
constant. Running the same sweep twice cost 45 minutes and turned "−7.3%, is
that noise?" into "−8.5% against a measured ±2.7% spread", which is the
difference between a finding and a guess.
