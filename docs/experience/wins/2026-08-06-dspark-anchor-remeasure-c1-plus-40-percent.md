# DSpark long-agent anchor re-measured: c=1 +40%, c≥4 down — CUDA, 2026-08-06

> Status: **c=1 accepted (+41.6%), c≥4 is a confirmed regression (−6.0 / −10.1
> / −7.2% at c=4/8/16).** The archived champion reproduces its own recorded row
> today, so the deficit is not a fingerprint artifact. Bisect target below.

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

**The champion reproduces its own row**, which is what makes the rest of this
readable:

| c | recorded (07-30) | champion today | drift |
|---:|---:|---:|---:|
| 1 | 7440.7 | 7287.4 | −2.1% |
| 4 | 25432.8 | 25265.7 | −0.7% |
| 8 | 31754.1 | 30621.3 | −3.6% |
| 16 | 32559.0 | 31890.9 | −2.1% |

Six days and ~800 commits of accumulated drift on the champion binary itself:
within the ±2.7% band at c=1/4/16, at its edge at c=8. The recorded row, the
box, and the regenerated dataset all agree.

Back to back, same shell:

| c | champion `51985031d` | HEAD `b8d390bf3` | Δ |
|---:|---:|---:|---:|
| 1 | 7287.4 | **10321.5** | **+41.6%** |
| 2 | 20740.4 | 19967.0 | −3.7% |
| 4 | 25265.7 | 23750.0 | −6.0% |
| 8 | 30621.3 | 27517.6 | **−10.1%** |
| 16 | 31890.9 | 29583.1 | **−7.2%** |

`accept_rate` 0.3121 vs 0.3091 — spec decode is not the difference.

The champion completed 126/128 at every point against HEAD's 128/128. An
incomplete request spends wall clock without contributing tokens, so the
champion's throughput is understated if anything; the deficit is a floor, not
a ceiling.

**Verdict: a real concurrency regression**, reproducible against the binary
that set the row, on a quiet box, an order of magnitude outside the measured
drift band.

## Bisect target

Not yet run. The cheapest first probe, before touching the ~800-commit range:

`301d0c074` (today) raised `SIDECAR_SNAPSHOT_STRIDE_PAGES` 128 → 512 in
`crates/infer-cuda/src/executor.rs:49`. Its own commit message states the
tradeoff — a cross-conversation prefix hit now re-prefills up to 8K tokens
instead of 2K — and it was licensed on a single-request 33K TTFT measurement,
which is the c=1 regime. This workload is 16 sessions × 8 turns against 16
slots, so a session's later turn can find its slot taken and land on the
restore path; that is where the stated cost falls.

**This is a hypothesis, not a root cause.** It is one compile-time constant, so
the test is a rebuild at 128 and one sweep — about an hour, against a
multi-hour bisect. If it moves c=8 back toward 30621, the mechanism is
confirmed and the constant becomes a per-workload decision rather than a global
default. If it does not, the range still has to be bisected.

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

**An archive turns a six-day-old row into a live control arm.** The deficit had
three candidate explanations — a regression, accumulated drift in the recorded
row, or a fingerprint difference in how it was measured. Running the archived
binary back to back settled all three in 90 minutes and needed no rebuild and
no bisect. The rule that every accepted binary gets archived is what made the
row falsifiable; without `/host/spec-phase/arle-mk` the only path was bisecting
~800 commits against a number nobody could reproduce.

**A prefill license does not cover concurrency.** The three changes that landed
since the row were each licensed on single-request 33K TTFT. They delivered:
c=1 is up 41.6%. The same window lost 10.1% at c=8, on a workload the licensing
benches never ran.
