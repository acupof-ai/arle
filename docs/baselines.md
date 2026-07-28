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

## Qwen3.6 on 1×H20 · single-GPU · eager — LONG-AGENT ANCHOR

### CHAMPION — `55bf627bc` (2026-07-28) · FA3 paged, qlen ≤ 64

Three arms, one binary (`arle-fa3c`), one session, one dataset. Models:
`bottlecapai/ThinkingCap-Qwen3.6-27B-FP8` (~29 GB dense, canonical CUDA agentic
model) and `Qwen3.6-35B-A3B-FP8` (MoE, 3B active). Both TP=1, qwen35 hybrid,
head_dim 256 — they share `full_attention_paged` and diverge only at the FFN.

Dataset `bench-agent-32k-16x8.jsonl`, sha256 `8867f63eaac2f053…`, regenerable via
`gen_bench_prompts.py bench-agent-32k-16x8.jsonl 16 32000 214 8` (16 sessions ×
8 turns; sessions ≥ max concurrency, rule 5). Runner `bench_throughput.py`,
128 req/point, `--concurrency-grid 1,8,16`, max_tokens 214, greedy, seed
20260416, `--max-running-requests 16`. `prompt_tokens` p50 34828. Needle gate
exact=3 DET at 512/4k/16k/32k on every arm. 0 errors at all 9 points.

**Prefill and decode are separate SLOs — never averaged into one tok/s.**
Prefill = TTFT and `prompt_tokens / TTFT`. Decode = token-weighted mean ITL
(`Σ itl_s / count`), which is the only decode metric valid on a spec arm: a
whole accepted chain lands per step, so ITL p50 there is ~0.02 ms and
meaningless. Never use `e2e − ttft` — this harness carries ~4.7 s of
post-stream teardown in `e2e`, inflating TPOT ~1.85×. Cold = each session's
turn 0 (nothing to reuse); warm = turns 1-7. `total tok/s` counts prompt +
generated tokens over wall time: a capacity number, not a latency one.

| c | arm | TTFT cold | TTFT warm | prefill tok/s | TPOT | decode tok/s | total tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | dense no-spec | 48.7 s | 14.2 s | 4252 | 28.71 ms | 34.8 | 1403.3 |
| 1 | dense DSpark 16 | 48.3 s | 13.3 s | 3801 | **9.39 ms** | **106.6** | 1504.8 |
| 1 | MoE no-spec | 18.6 s | 4.8 s | **10991** | 15.93 ms | 62.8 | 3174.0 |
| 8 | dense no-spec | 21.9 s | 15.3 s | 4416 | 908.39 ms | 1.1 | 2007.7 |
| 8 | dense DSpark 16 | 22.7 s | 15.7 s | 4301 | 838.64 ms | 1.2 | 2140.7 |
| 8 | MoE no-spec | 10.9 s | 3.4 s | 11268 | 224.29 ms | 4.5 | **5689.0** |
| 16 | dense no-spec | 75.2 s | 26.9 s | 4071 | 1822.09 ms | 0.5 | 2027.4 |
| 16 | dense DSpark 16 | 72.3 s | 23.8 s | 4231 | 1755.26 ms | 0.6 | 2132.3 |
| 16 | MoE no-spec | 27.5 s | 8.1 s | 10701 | 453.79 ms | 2.2 | 5759.6 |

Cold/warm decode at c=1: dense no-spec 27.58/28.99 ms, DSpark 11.91/**8.77 ms
(114.1 tok/s warm)**, MoE 14.98/16.10 ms. DSpark is the only arm that gets
*faster* warm — a hit prefix leaves the drafter more context to work with.

- **DSpark is 3.06× per token at c=1** (9.39 vs 28.71 ms), up from 1.48× before
  the verify path reached FA3. A verify step costs 9.39 × E[k+1]=3.19 ≈ 30.0 ms
  against a 28.71 ms decode step — **1.04×**, the physical floor, since
  verifying 17 tokens reads the same KV bytes as verifying 1.
  [entry](experience/wins/2026-07-28-fa3-covers-every-query-length.md)
- **The spec win decays with concurrency: 3.06× (c=1) → 1.08× (c=8) → 1.04×
  (c=16).** Expected, not a defect — once the batch is full the GPU is already
  saturated with real tokens and there is no idle capacity for drafts to
  reclaim. Total tok/s still moves +6.6% at c=8.
- **MoE beats dense on every axis at equal spec setting** — 1.80× decode
  (62.8 vs 34.8), 2.6× prefill, 2.26× total at c=1 — on the *identical* FA3
  kernel at GQA 8. Comparing MoE against the DSpark row is apples to oranges.
- **All three arms hit the same wall at c=8**: c=8→16 buys +0.7% (dense
  no-spec), −0.4% (DSpark), +1.2% (MoE) total tok/s while TPOT doubles. Model-
  and spec-independent, so it is queueing, not compute — ITL p50 at c=8 is
  66.07 ms (no-spec) vs 66.19 ms (DSpark), and p90 is 4035 vs 3823 ms. The
  bimodal ITL is the scheduler, and it is the next thing worth attacking.
- **FA3 on prefill chunks is a wash and is capped anyway** — `FA3_MAX_QLEN = 64`
  keeps prefill on TileLang; FA3 needs one launch per request when `seqused_k`
  is set, which cost 51% of TTFT at c=8.
  [entry](experience/errors/2026-07-28-fa3-prefill-per-request-launch-regression.md)
- **FA3 paged is the delta vs anything before 2026-07-27**: decode mean ITL
  76.98 → 28.64 ms at c=1 on the same dataset.
  [entry](experience/wins/2026-07-27-fa3-paged-decode-32k-2.76x.md)
- sm_90 only: FA3 hopper is Hopper-only; other targets keep the TileLang
  `batch_decode_paged_hd256` kernel.
- c=4 not measured (grid is 1,8,16). MoE has no DSpark arm (no drafter).

**Superseded fingerprints** (deleted 2026-07-28 — one dataset per question,
rule 5). What survives them:

- 2026-07-25 short-prompt row (`6aa4ca6d1`, `dspark_natural_128in_128out.jsonl`):
  DSpark accept_rate 0.10–0.13, MTP ~0.17; per-slot KV `343360 tok / 22.5 GB`
  no-spec vs `121920 tok / 8.0 GB` DSpark. Its headline "DSpark c=1 +57.5%" is
  explained, not reproduced — the drafter was carrying a decode kernel that
  cost 2.7× too much.
- 2026-07-26 cold-cache row (`f4f419629`, `bench-agent-32k-64.jsonl`): cold
  prefill at 33k ran ~540 tok/s, degrading from ~1270 at ~5k. Prefix hit rate
  was 0 by construction, so it measured a machine nobody runs; the "~89% is
  prefill" reading taken from it was already withdrawn.

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
