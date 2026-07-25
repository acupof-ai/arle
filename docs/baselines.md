# Rolling performance baselines

> Status: Active — one champion table per config fingerprint, newest first.

Screening compares new runs against the champion row — no second arm. Rules:

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

## DSv4-Flash-FP8 · 4×H20 · TP=4/EP=4 · eager · port 8000

### CHAMPION — Base, `d0525cb06` (re-anchored 2026-07-25, #180)

GPUs 0-3 (H20 indices don't re-anchor — same silicon). Dataset
`bench-prompts-20.jsonl`, sha256
`e095ddf1fcc9325a43bb510b36e2afcb6c56d68af3ecc032503b8430b4f3fc49`,
**reproducible byte-for-byte from the repo** (verified local vs pod):

```
python3 scripts/gen_bench_prompts.py bench-prompts-64.jsonl 64 13400 256   # sha256 3543ac33…
head -20 bench-prompts-64.jsonl > bench-prompts-20.jsonl
```

Runner `bench_throughput.py` via `run_dsv4_bench.sh`, 60 s/point, seed
20260416, max_tokens 256, no `--max-running-requests`. Slot line `59 slots /
per_slot 338MB / budget 20584MB / 84736 tok`. chunk-2048 default. Prefix
hit_rate 0.048 (c1) → 0.767 (c16).

| c  | complete | out tok/s | total tok/s | TTFT p50/p99 ms | ITL p50/p99 ms |
|----|---------:|----------:|------------:|-----------------|----------------|
| 1  | 10       | 38.66     | 456         | 1085 / 1113     | 21.9 / 41.0    |
| 4  | 20       | 74.67     | 876         | 1447 / 2985     | 43.8 / 89.2    |
| 8  | 40       | 152.82    | 1793        | 1069 / 1204     | 47.5 / 93.2    |
| 16 | 48       | 197.51    | 2319        | 2238 / 2265     | 71.4 / 119.0   |

0 errors / 0 incomplete / 0 correctness_failed at every point. Raw: pod
`bench-output/2026-07-24-b156-d0525cb0/`. Reproduced by the `d0525cb06`
runtime verifies (recompute-resume, band-exhaustion park-gate, both
2026-07-25).

- **c32**: needs `--max-running-requests 32` (`num_slots 32, comp capacity
  1048576 tok`); without it, host-admission oversubscription degrades to
  preemption, not a crash (#164/#162 closed). No reproducible-dataset c32
  throughput point yet — the retired 209.9 out tok/s ran on the lost
  `bench-prompts.jsonl`.
- **Why this anchor**: the prior champion's `bench-prompts.jsonl`
  (repeated-filler, prefix hit_rate 0.925) no longer exists and has no
  generator; `gen_bench_prompts.py` deliberately produces the non-degenerate
  variant. Rule 3 re-anchored on the reproducible dataset. `run_dsv4_bench.sh`
  now fails loudly on a missing dataset and records `dataset.sha256` next to
  every result.

### Spec-decode arms — `45dd64bd2` (2026-07-19, production-all-on)

Different dataset (`bench-prompts-64.jsonl` ~2.8k tok, 120 s/point) → Δ is vs
each run's own Base, not the champion. Needle 15/15 strict. MTP accept_rate
0.704. Raw: pod `/host/arle-evidence/prod-allon-45dd64bd-dsv4-*`.

| c  | Base  | MTP      | MTP Δ      | DSpark | DSpark Δ |
|----|------:|---------:|-----------:|-------:|---------:|
| 1  | 38.0  | **46.2** | **+21.6%** | 38.1   | +0.3%    |
| 4  | 74.6  | 70.2     | −5.9%      | 74.3   | −0.4%    |
| 8  | 123.7 | 72.0     | −41.8%     | 121.9  | −1.5%    |
| 16 | 195.7 | 69.7     | −64.4%     | 117.6  | −39.9%   |

- **MTP: c1-only win** — draft verification serializes under concurrency, c4+
  regresses. Not a default-flip candidate.
- **DSpark: not triggered** — `--dspark-max-prompt-tokens 64` routes all
  ~2.8k-tok bench prompts to no-spec. Needs a short-prompt workload to measure
  gain. Batched-verify c8/c16 OOM'd on a stale-memory GPU (inconclusive,
  [win](experience/wins/2026-07-21-dspark-batched-verify-c8-c16.md)).

## Qwen3.6-27B-W4A16 · 1×V100 (sm_70) · eager · port 8080

**2026-07-21 (`aec71ef16`, V100 kernel opts + KV pool floor fix)** — synthetic
prompts 64, 60 s/point, max_tokens 256, seed 20260416. KV pool 16384 tok BF16
(1.1 GB), 86 slots (clamped from 256 by VRAM budget). Serve:
`--max-total-tokens 16384`. Raw: V100 `/tmp/v100_nospec_bench.{json,csv}`.

| c  | complete | out tok/s | total tok/s | TTFT p50/p99 ms | ITL p50/p99 ms |
|----|---------:|----------:|------------:|-----------------|----------------|
| 1  | 11       | 22.8      | 24.4        | 251 / 304       | 40.4 / 41.6    |
| 4  | 12       | 25.5      | 27.4        | 17799 / 25769   | 0.02* / 270    |
| 8  | 17       | 28.4      | 30.4        | 30818 / 54318   | 0.02* / 335    |
| 16 | 16       | 30.1      | 32.1        | 72270 / 72933   | 0.02* / 452    |

\* ITL p50 ≈ 0.02 ms is a bench-script artifact (streaming inter-token
sampling undercounts at c≥4); the out tok/s column is the valid throughput
metric. c=1 ITL 40.4 ms ≈ 24.7 tok/s decode, matches the c=1 out tok/s.

- **Decode-bound at all concurrencies.** out tok/s scales weakly (22.8 → 30.1,
  +32% from c=1 to c=16) — V100 sm_70 W4A16 decode is the bottleneck, not
  scheduler/KV. TTFT grows linearly with concurrency (queueing).

### DSpark arm — KILLED (−91% at c=1)

Same serve + `--spec-type dspark --mtp-draft-model
z-lab/Qwen3.6-27B-DFlash` (DFlash drafter, block=16, taps=[1,16,31,46,61]).
39 slots (clamped by draft model VRAM +96MB/slot). Raw: V100
`/tmp/v100_dspark_bench.{json,csv}`.

| c  | complete | errors | out tok/s | ITL p50/p99 ms | Δ vs no-spec |
|----|---------:|-------:|----------:|----------------|-------------:|
| 1  | 2        | 0      | 2.0       | 499 / 507      | **−91.2%**   |
| 4  | 4        | 1      | 2.0       | 0.02* / 1990   | **−92.2%**   |
| 8  | 0        | 8      | 0.0       | n/a            | n/a          |
| 16 | 0        | 131204 | 0.0       | n/a            | n/a          |

\* ITL p50 artifact (see no-spec note); c=1 ITL 499 ms vs no-spec 40 ms =
12.5× slower per decode step.

- **KILL.** DSpark draft+verify path on V100 sm_70 adds ~460 ms/step overhead
  (ITL 40 → 499 ms). c=8 all 8 requests error (1543 s wall); c=16 connection
  storm (131204 errors in 60 s). Serve log: `[coordinator] lockstep stalled:
  tick #2232128 awaiting acks (elapsed=10s)` — the TP lockstep mechanism
  deadlocks under DSpark's multi-step proposal on single-GPU V100.
- **Root cause hypothesis**: DSpark's `tp_lockstep_proposal/accept` was designed
  for TP≥2 (H20); on TP=1 V100 the lockstep coordinator stalls waiting for
  cross-rank acks that never arrive. Needs a TP=1 fast path or the lockstep
  disabled when world_size=1.
