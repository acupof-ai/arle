# Metal Qwen3.5 — flip concurrent DFlash to default-on

**Date**: 2026-04-19
**Machine**: Apple M4 Max (40 GPU cores, ~400 GB/s UMA), macOS 26.3.1
**Model**: `mlx-community/Qwen3.5-4B-MLX-4bit` + DFlash draft `z-lab/Qwen3.5-4B-DFlash`
**Commits**: `308d427` (toolchain), this commit (flip + doc)

## Goal

Make concurrent DFlash the production default. Until today the path was
gated behind `MetalSchedulerConfig.metal_dflash_concurrency_off = true`
*and* `runtime::admit_request` only enabled DFlash for solo requests
(`active.is_empty()`), so any second concurrent request demoted both into
a plain-decode batch — the legacy "downgrade" — which actively regressed
throughput at c≥2.

Two changes:

1. `MetalSchedulerConfig::default().metal_dflash_concurrency_off = false`
2. `admit_request` sets `enable_dflash = true` unconditionally (the
   `from_incoming` constructor still no-ops when no draft model is loaded)

## Hypothesis

Removing the downgrade should make concurrent DFlash beat concurrent
plain-decode at c≥2. Per-row eligibility filter (round-3 [P2], `f09b25a`)
ensures rows that *can't* dflash this tick still serialize via
`execute_decode_single` while the rest go through the packed batch.

## Setup

- Server: `metal_serve` (rebuild `cargo build --release -p infer --bin metal_serve --no-default-features --features metal`)
- Bench tool: guidellm 0.6.0 under Python 3.11 (see `scripts/setup_bench_toolchain.sh`)
- Workload: `prompt_tokens=1024,output_tokens=256`, concurrent profile, `--rate 1,2,4`, `--max-seconds 30`

A and B were collected with the same guidellm command, on consecutive
warm-server runs. The wrapper now exports `GUIDELLM__MP_CONTEXT_TYPE=forkserver`
because guidellm 0.6.0's default `fork` deadlocks on macOS.

## Results

`--max-seconds 30, prompt_tokens=1024, output_tokens=256`:

| streams | A gen/s | B gen/s | Δ gen/s | A tot/s | B tot/s | A TPOT | B TPOT |
|---|---|---|---|---|---|---|---|
| 1 | 65.0 | 64.1 | -1% (noise) | 325 | 321 | 5.1 ms | 5.1 ms |
| 2 | 36.3 | 64.0 | **+76%** | 182 | 320 | 42.4 ms | 21.1 ms |
| 4 | 22.9 | 64.0 | **+180%** | 115 | 320 | 55.1 ms | 44.3 ms |

Source data: `/tmp/bench_baseline_A/benchmarks.json` (legacy default),
`/tmp/bench_B_full/benchmarks.json` (flipped default).

## Interpretation

* The flip eliminates the legacy collapse: at c=4 each stream now runs
  at ~solo-DFlash speed (64 gen/s) instead of crashing to 22.9.
* Aggregate `tot/s` plateaus at ~320 across c=1/2/4, matching the
  `archived terminal-state` ceiling — the GDR kernel `6.1 ms/row` wall
  still binds aggregate throughput. We are *not* harvesting concurrency,
  but we *did* stop self-sabotaging it.
* TPOT halves at c=2 and drops 20% at c=4, so user-visible latency
  improves materially under bursty load.

## Problems

* guidellm 0.6.0 worker_group hangs under macOS default `fork` mp
  context. Worked around by exporting `GUIDELLM__MP_CONTEXT_TYPE=forkserver`
  in `scripts/bench_guidellm.sh`. Documented in
  `scripts/setup_bench_toolchain.sh` and the toolchain commit (`308d427`).
* No additional aggregate throughput gain past c=1 — concurrency is
  capacity-saving, not a speedup. Next non-trivial upside still requires
  Xcode Metal capture of the GDR kernel (out of /loop scope), per the
  archived terminal-state doc.

## Rule

When evaluating the "is concurrent batching helping?" question on Metal
Qwen3.5, compare **per-stream gen/s vs c=1**, not aggregate `tot/s`.
A flat `tot/s` curve across c=1..N with non-degrading per-stream gen/s
means the GPU is saturated by a single decode tick and concurrency is
buying you queueing fairness, not throughput. The wrong reading is
"concurrent DFlash didn't help because tot/s didn't grow."
