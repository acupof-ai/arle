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
5. **One workload**: every champion row runs the multi-turn long-agent dataset
   at the TraceLab medians (bench spec §3.3), and reports cold vs warm turns
   separately. Rows below the 2026-07-26 line predate that rule — short-prompt
   fingerprints, or the one-shot 32k dataset that could never hit the prefix
   cache. Historical evidence, not comparison targets; re-anchor first.

```
python3 scripts/gen_bench_prompts.py bench-agent-119k-16x8.jsonl 16 119000 214 8
```

## ThinkingCap-Qwen3.6-27B-FP8 · 1×H20 · single-GPU · eager — LONG-AGENT ANCHOR

### CHAMPION — `c1c05d61c` (2026-07-27) · FA3 paged decode

Dataset `bench-agent-32k-16x8.jsonl`, sha256
`8867f63eaac2f053...`, regenerable via
`gen_bench_prompts.py bench-agent-32k-16x8.jsonl 16 32000 214 8` (16 sessions ×
8 turns; sessions ≥ max concurrency, rule 5). Runner `bench_throughput.py`,
128 req/point, max_tokens 214, greedy, seed 20260416,
`--max-running-requests 16`, GPUs 0 (no-spec) and 2 (DSpark), same binary and
session. `prompt_tokens` p50 34959. 0 errors.

| c | arm | wall s | total tok/s | TTFT p50 | ITL p50 | ITL p90 |
|---|---|---:|---:|---:|---:|---:|
| 1 | no-spec | 3273.9 | 1344.4 | 14.1 s | **28.71 ms** | 29.80 |
| 1 | DSpark 16 | 3227.7 | 1364.2 (+1.5%) | 14.4 s | (burst) | 80.28 |
| 8 | no-spec | 1788.3 | **2460.8** | 12.2 s | **65.92 ms** | 3808.6 |
| 8 | DSpark 16 | 1910.5 | 2304.6 (**−6.3%**) | 13.1 s | (burst) | 256.2 |
| 16 | no-spec | 1770.1 | **2486.0** | 16.1 s | **107.67 ms** | 8128.1 |
| 16 | DSpark 16 | 1905.9 | 2310.1 (**−7.1%**) | 27.4 s | (burst) | 8351.5 |

`ITL p50` is meaningless on a spec arm (a whole accepted chain lands per step);
compare those rows on wall clock and total tok/s.

- **FA3 paged decode is the delta vs everything before 2026-07-27**: ITL p50
  76.98 → 28.71 ms at c=1 (**2.68×**) and 140.49 → 65.92 ms at c=8 (**2.13×**).
  c=1 reproduces the 8-session run (27.94 ms) to 0.8% on a different dataset.
  [win](experience/wins/2026-07-27-fa3-paged-decode-32k-2.76x.md)
- **DSpark is net-negative at serving concurrency.** +1.5% at c=1, −6.3% at c=8,
  −7.1% at c=16. Before FA3 it was +57.5% at c=1 on short prompts; the edge was
  never mostly speculation — it was paying for a 2.7× too expensive decode step.
  [repricing](research/2026-07-27-dspark-repriced-after-fa3.md)
- **The machine saturates at c=8**: 2460.8 → 2486.0 tok/s from c=8 to c=16
  (+1.0%) while ITL p50 goes 65.9 → 107.7 ms and ITL p90 hits 8.1 s. Past c=8
  concurrency buys queueing, not throughput. That is the next bottleneck and it
  is a scheduler problem.
- sm_90 only: FA3 hopper is Hopper-only; other targets keep the TileLang
  `batch_decode_paged_hd256` kernel.
- c=4 not measured (grid was 1,8,16).

### `f4f419629` (2026-07-26) · RETIRED cold-cache fingerprint — NOT a champion

Dataset `bench-agent-32k-64.jsonl`, sha256
`683b3736b2b162a07e419bf8ed8639fb70e6bc4f9a2cd8c5c586b39060ab8ef5`, reproducible
from the repo (rule 5). Runner `bench_throughput.py`, 8 req/point, max_tokens
256, seed 20260416, `--max-running-requests 16`, GPU 0, no spec decode.
Measured `prompt_tokens` 33000 (target 32768, +0.7%). Prefix hit rate 0 by
construction. 0 errors / 0 incomplete / 0 correctness_failed at every point.

| c | complete | out tok/s | total tok/s | wall s |
|---|---------:|----------:|------------:|-------:|
| 1 | 8        | 3.4       | 486.0       | 547.0  |
| 4 | 8        | 12.9      | 1748.9      | 152.0  |
| 8 | 8        | 14.3      | 1906.7      | 139.5  |

- **Retired the same day: prefix hit rate 0 by construction.** Each request got
  a unique 32k context, so every one paid a full cold prefill. Real coding-agent
  serving hits the prefix cache 95.7% of the time (TraceLab arXiv:2606.30560),
  which makes this row a measurement of a machine nobody runs.
- The "~89% is prefill, decode is the ~10% slice" reading taken from it is
  **withdrawn**. At the trace medians a step is TTFT 3.1 s vs 4.6 s of decode —
  decode is ~60% of per-step wall clock. Do not cite this row for scoping.
- Kept as evidence of cold-prefill cost at 33k (~540 tok/s, degrading from
  ~1270 tok/s at ~5k), which is still the right number for a cache miss.
- c=16 not yet measured on this dataset (KV budget check pending).

## DSv4-Flash-FP8 · 4×H20 · TP=4/EP=4 · eager · port 8000

### CHAMPION — Base, `d0525cb06` (re-anchored 2026-07-25, #180)

> **Short-prompt fingerprint, retired 2026-07-26 (rule 5).** ~3.4k-token docs
> from the pre-long-agent `gen_bench_prompts.py`; that generator now emits the
> 32k agent shape, so this dataset is no longer reproducible from the repo and
> the row cannot be re-measured. Kept as evidence for the numbers it licensed.

GPUs 0-3 (H20 indices don't re-anchor — same silicon). Dataset
`bench-prompts-20.jsonl`, sha256
`e095ddf1fcc9325a43bb510b36e2afcb6c56d68af3ecc032503b8430b4f3fc49`
(first 20 lines of `bench-prompts-64.jsonl`, 64 docs × 13400 chars).

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

### Spec-decode arms — `6aa4ca6d1` (2026-07-25) · RETIRED short-prompt fingerprint

Retired 2026-07-26 (rule 5): 128-token prompts, not a serving shape. Dataset
`dspark_natural_128in_128out.jsonl` (sha `169b7c78…`, 20 prompts),
**max_tokens 128**, 60 s/point, c=1,4,8,16, GPUs 4-7 TP=4/EP=4. Same binary,
same session; Δ is vs this run's own no-spec (NOT the champion — different
workload). 0 errors all arms. Raw: pod `bench-output/2026-07-25-R2-dsv4-*`.

| c  | no-spec | MTP   | MTP Δ  | DSpark | DSpark Δ |
|----|--------:|------:|-------:|-------:|---------:|
| 1  | 42.39   | 36.51 | −13.9% | **44.52** | **+5.0%** |
| 4  | 79.46   | 42.57 | −46.4% | 61.00  | −23.2%   |
| 8  | 136.44  | 51.74 | −62.1% | 76.57  | −43.9%   |
| 16 | 174.46  | 61.94 | −64.5% | 90.66  | −48.0%   |

accept_rate (server-stats): MTP ~0.15, DSpark ~0.30. Slot lines: no-spec
`per_slot 338MB → 59 slots`, MTP `381MB → 49`, DSpark `607MB → 22`
(`stages=3 block=5 target_layers=[40,41,42]`).

- **Both spec arms are c=1-only** — DSpark +5.0% at c=1, net-negative at c≥4;
  MTP negative everywhere on this shape. The crossover is the spec-decode
  compute-bound transition, one mechanism (verify cost is free only while the
  GPU has idle compute):

  ```
  each step: draft block=5 → target verifies 6 positions → commit ~2.5 tok (accept 0.30, flat vs c)
                                      │
              ┌───────────────────────┴───────────────────────┐
          c=1: batch small                              c=16: batch full
          GPU memory-bound                              GPU compute-bound
          verify 6 pos ≈ free                           verify 6 pos = ~6× time
              │                                                │
        2.5 tok / ~1× time                            2.5 tok / ~6× time
              ▼                                                ▼
        ✅ +5%  (27B c=1 +57.5%)                    ❌ 2.5/6 ≈ 0.42 → measured 90.66/174.46 = 0.52
  ```

  Not default-flip candidates.
- **#183 fix confirmed** — the earlier c=16 −49.9% "DSpark collapse" was the
  train-capture per-step 2×D2H+2×sync serializing the TP=4 NCCL pipeline
  (default-off consumer). Gated on the train sidecar now; the curve above is
  the real spec-decode shape, no crash.
- **#184 confirmed** — spec scratch sized to the real verify width (6/9 rows),
  DSpark per_slot 645→607MB.

## ThinkingCap-Qwen3.6-27B-FP8 · 1×H20 · single-GPU · eager · port 8200

**2026-07-25 (`6aa4ca6d1`) · RETIRED short-prompt fingerprint** (rule 5) —
canonical CUDA agentic model
(`bottlecapai/ThinkingCap-Qwen3.6-27B-FP8`, ~29 GB, qwen35 hybrid, TP=1).
Dataset `dspark_natural_128in_128out.jsonl` (sha `169b7c78…`), **max_tokens
128**, 60 s/point, c=1,4,8,16, GPU 4. Same binary, same session; Δ vs this
run's own no-spec. 0 errors all arms. Coherent (thinking model, answer in
`reasoning_content`). Raw: pod `bench-output/2026-07-25-R2-tc-*`.

| c  | no-spec | MTP   | MTP Δ  | DSpark | DSpark Δ  |
|----|--------:|------:|-------:|-------:|----------:|
| 1  | 38.62   | 42.31 | +9.6%  | **60.83** | **+57.5%** |
| 4  | 75.10   | 40.29 | −46.4% | 52.09  | −30.6%    |
| 8  | 126.01  | 40.59 | −67.8% | 52.52  | −58.3%    |
| 16 | 150.89  | 40.37 | −73.3% | 53.02  | −64.9%    |

accept_rate (server-stats): MTP ~0.17, DSpark ~0.10–0.13. Slot lines: no-spec
`343360 tok / 22.5 GB`, MTP `330560 tok / 21.7 GB`, DSpark `121920 tok / 8.0
GB` (DFlash drafter `z-lab/Qwen3.6-27B-DFlash`, `block=16 taps=[1,16,31,46,61]`).

- **DSpark c=1 +57.5%** is the strongest spec-decode signal in either model —
  27B single-GPU at c=1 is decode-bound, exactly where spec decode wins. Both
  spec arms go net-negative at c≥4 (batch saturates the single GPU).
- **TP=1 lockstep deadlock FIXED** — 60 s liveness probe returned coherent
  output, 0 `lockstep stalled` lines across the sweep. The `world_size ≤ 1`
  early-return holds; the V100 W4A16 −91% hang below does NOT reproduce here.

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
