# Rolling performance baselines

> Status: Active

One table per config fingerprint. Screening compares new runs against the
champion row — no second arm. Rules:

1. **Effect > ~10% (2× the measured cross-session drift band)**: rolling
   baseline verdict is valid; update the champion row, archive the binary.
2. **Effect inside the drift band (±3% measured 2026-07-16: same binary
   lineage, different session + GPU set): never kill on ambiguity — every
   stable positive gain is kept.** Escalate to a same-shell A/B against the
   archived champion binary (≥3 trials/arm, median + range) to resolve sign.
3. **Fingerprint change re-anchors**: any change to model, TP/EP, GPU set,
   serve flags, num_slots line, dataset, driver/CUDA invalidates the row —
   re-measure the champion before comparing.
4. **Anchor audit**: every ~5 accepted updates (and before any default flip),
   one A/B against the oldest archived binary bounds accumulated drift.

## DSv4-Flash-FP8 · 4×H20 GPUs 3,5,6,7 · TP=4/EP=4 · eager · port 8000

**RE-ANCHOR 2026-07-19 (`45dd64bd2`, production-all-on)** — dataset
`bench-prompts-64.jsonl` (~2.8k unique docs), 120 s/point, max_tokens 256,
seed 20260416. Binary `c7730414…`, kernel `bundle:7eef1a90…`.
Needle ×3 passes (15/15 strict). Raw: pod
`/host/arle-evidence/prod-allon-45dd64bd-dsv4-{base,mtp,dspark}-20260719T*/`.

| c | Base out tok/s | MTP out tok/s | MTP Δ | DSpark out tok/s | DSpark Δ |
|---|---:|---:|---:|---:|---:|
| 1 | 38.0 | **46.2** | **+21.6%** | 38.1 | +0.3% |
| 4 | 74.6 | 70.2 | -5.9% | 74.3 | -0.4% |
| 8 | **123.7** | 72.0 | -41.8% | 121.9 | -1.5% |
| 16 | **195.7** | 69.7 | -64.4% | 117.6 | -39.9% |

- **Base = production champion.** c16 195.7 vs old chunk-2048 anchor 142.9
  (different GPU set + 120s vs 90s → re-anchor, not a strict Δ).
- **MTP: c1-only win.** accept_rate 0.704; draft verification serializes under
  concurrency, c4+ regresses. Not a default-flip candidate.
- **DSpark: not triggered.** `--dspark-max-prompt-tokens 64` routes all >64-tok
  prompts to no-spec; bench prompts ~2.8k tok → 100% target decode. Needs
  short-prompt workload to measure gain.

## DSv4-Flash-FP8 · 4×H20 GPUs 0–3 · TP=4/EP=4 · eager · port 8000

**RE-ANCHOR 2026-07-17 (chunk-2048 default, `0904a50cc`)** — dataset
`bench-prompts-64.jsonl` (~2.8k unique docs), cap32 serve, 90 s/point:
c1 38.9 out / TTFT 1093 ms · c4 75.0 · c16 142.9 · c32 **209.9 out / 2474
total tok/s**. Cold prefill ~2560 tok/s. Raw: pod
`bench-output/2026-07-17-p3-sweep/`. Rows below are the chunk-128 era.

Champion (chunk-128 era): `00b301643` (grid-parallel FP32 + slot hoist + carry coherence +
plan repair), build `--features cuda,nccl`. **RE-ANCHORED fingerprint
2026-07-17**: runner = `bench_throughput.py`, **max_tokens 256** (the earlier
16-out era below is retired), dataset `bench-prompts.jsonl` (20×~3352 tok),
60 s/point, seed 20260416, GPUs 0–3 TP=4/EP=4 eager, slot line `256 clamped
to 59, per_slot 338MB, budget 20582MB, comp capacity 83968 tok`. Needle ×2
passes (prefix-restore lane) zero-miss. Raw:
`bench-output/2026-07-16-accept/`.

| c | complete | out tok/s | total tok/s | TTFT p50/p99 ms | ITL p50/p99 ms |
|---|---:|---:|---:|---|---|
| 1 | 11 | 42.3 | 605 | 442 / 457 | 21.8 / 41.9 |
| 4 | 20 | 73.3 | 1032 | 1532 / 7863 | 43.4 / 88.4 |
| 8 | 40 | 137.3 | 1935 | 2726 / 2923 | 46.8 / 92.7 |
| 16 | 48 | 169.6 | 2392 | 5294 / 5637 | 71.6 / 124.4 |
| 32 (bench-prompts-64, 300 s, `--max-running-requests 32`) | 121 | 91.9 | 1090 | 83500 / 91900 | 134.6 / 308 |

c32 champion requires `--max-running-requests 32` (slot line: `num_slots 32,
comp capacity 1048576 tokens`) — the recommended DSv4 serve flag per
[the slot-budget entry](experience/wins/2026-07-17-max-running-requests-caps-slot-budget.md).
Crash-era and default-slots rows retired. Grid-time prefix hit_rate 0.925.

Retired 16-out-era rows (guidellm, 2026-07-16, kept for the same-era A/B
deltas only): rate 1/4/8/16 total 4514/6663/7985/8130; var-c1 850.

c32 crash RESOLVED 2026-07-17 (#164/#162 closed): oversubscription now
degrades to preemption
([errors](experience/errors/2026-07-16-dsv4-c32-hostpagedkvpool-fatal.md)),
and `--max-running-requests 32` removes the pressure entirely.

**WORKLOAD CORRECTION (2026-07-16)**: every 2026-07-16 run actually generated
**16 output tokens per request**, not 256 — guidellm never sent `max_tokens`
(the server's /v1/completions default is 16) because the dataset column wasn't
mapped. All same-day Δ% comparisons remain valid (identical workload both
arms), and per-request decode speed is 1/ITL ≈ 46.5 tok/s, but `out tok/s`
aggregates are prefill-diluted and entries saying "256 output" describe
3352-in/16-out. guidellm is REMOVED entirely (2026-07-16, user call — this silent default plus
its synthetic-data and accounting quirks); `run_dsv4_bench.sh` now drives the
canonical native runner `bench_throughput.py` with explicit `--max-tokens`.
The NEXT anchored run (native runner, 256-out) is a **fingerprint change:
re-anchor the champion**.

Pure-prefill anchor (c1, no queueing): ~3352-tok prompt / 446 ms TTFT ≈
**7516 tok/s prefill**.

Archived arms (pod): `arle-armA-serialprobe` (serial probe),
`arle-armB-gridpar` (2e635eda3, per_slot 9618MB/2 slots). Raw:
`/host/arle-build/bench-output/2026-07-16-{fp32par,fp32serialA,fp32slots}-*`.

## DSv4-Flash-FP8 · 4×H20 GPUs 1,2,3,5 · TP=4/EP=4 · eager · DSpark batched verify

**2026-07-21 (`13fe251cb` + `9edfcb234` + `4e2a852b0` + `dspark-fix3`)** — dataset
`bench-prompts-64.jsonl` (~3.4k tok), 60 s/point, max_tokens 256, seed 20260416.
Serve: `--max-running-requests 32`. DSpark `--dspark-conf-threshold 0`, block 5,
greedy. Raw: pod `/tmp/nospec_c8c16.{json,csv}`.
[win](experience/wins/2026-07-21-dspark-batched-verify-c8-c16.md)

| c | No-spec out tok/s | DSpark out tok/s | DSpark Δ | Errors (no-spec / dspark) |
|---|---:|---:|---:|---:|
| 8 | **101.2** | n/a (OOM) | n/a | 0 / n/a |
| 16 | **146.7** | n/a (OOM) | n/a | 0 / n/a |

- **No-spec re-measured with correct config**: GPUs 1,2,3,5 (GPU 5 had 8 GB stale
  memory from a dead process, not 37 GB as the 2026-07-20 run assumed) +
  `--max-running-requests 32` (the 2026-07-20 c=16 run omitted this → 82286
  connection errors, 15/82303 complete). c=16 now valid: 48/48 complete, 0 errors.
- **DSpark OOM on GPU 5**: draft model needs ~10 GB more than no-spec; GPU 5's 8 GB
  stale memory (PID 1499243, unkillable, `nvidia-smi --gpu-reset` refused) leaves
  insufficient headroom. 3 GPUs fully free (1,2,3) is not enough for TP=4 DSpark.
- **Prev (2026-07-20) no-spec rows invalid**: c=8 146.5 ran on 3 effective GPUs
  (GPU 5 occupied); c=16 32.0 was server collapse. Both superseded.

## Qwen3.6-27B-W4A16 · 1×V100 (sm_70) · eager · port 8080

**2026-07-21 (`aec71ef16`, V100 kernel opts + KV pool floor fix)** — synthetic
prompts 64, 60 s/point, max_tokens 256, seed 20260416. KV pool 16384 tok BF16
(1.1 GB), 86 slots (clamped from 256 by VRAM budget). Serve:
`--max-total-tokens 16384`. Raw: V100 `/tmp/v100_nospec_bench.{json,csv}`.

| c | complete | out tok/s | total tok/s | TTFT p50/p99 ms | ITL p50/p99 ms |
|---|---:|---:|---:|---|---|
| 1 | 11 | 22.8 | 24.4 | 251 / 304 | 40.4 / 41.6 |
| 4 | 12 | 25.5 | 27.4 | 17799 / 25769 | 0.02* / 270 |
| 8 | 17 | 28.4 | 30.4 | 30818 / 54318 | 0.02* / 335 |
| 16 | 16 | 30.1 | 32.1 | 72270 / 72933 | 0.02* / 452 |

\* ITL p50 ≈ 0.02 ms is a bench-script artifact (streaming inter-token
sampling undercounts at c≥4); the out tok/s column is the valid throughput
metric. c=1 ITL 40.4 ms ≈ 24.7 tok/s decode, matches the c=1 out tok/s.

- **Decode-bound at all concurrencies.** out tok/s scales weakly (22.8 → 30.1,
  +32% from c=1 to c=16) — V100 sm_70 W4A16 decode is the bottleneck, not
  scheduler/KV. TTFT grows linearly with concurrency (queueing).
- **DSpark not measured**: DFlash draft checkpoint on V100 has only
  `config.json`, no weight files — needs a full DFlash model download before
  the spec-decode arm can run.


