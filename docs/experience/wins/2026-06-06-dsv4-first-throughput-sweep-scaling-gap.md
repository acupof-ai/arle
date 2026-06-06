# DSv4 first throughput sweep — concurrency scaling is the big untouched axis (1.63× at c=8, per-req collapse, c=32 OOM)

**Date:** 2026-06-06. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Status:** the **first DSv4 throughput measurement** (all prior work was B=1
latency). ckl: "到最后最后也得压一下吞吐." Ran on the **allreduce** MoE backend (the
only debug-runnable serving lane — see Serving gap below), `bench_direct_http.py`,
ISL=1024 OSL=512, num_slots=32, mem_fraction_static=0.60.

## Results

| concurrency | TTFT | ITL | per-request decode tok/s | aggregate output tok/s |
|---|---|---|---|---|
| 1 | 2391.5 ms | 30.28 ms | **33.09** | 28.66 |
| 8 | (mean) | (mean) | **7.15** (collapsed) | **46.61** |
| 32 | — | — | prefill **OOM** | 0 |

## The finding — throughput does NOT scale

- **Aggregate throughput improves only 1.63× from c=1→c=8** (28.66 → 46.61 tok/s).
  Ideal batched decode would approach ~8× (the weights + TP-comm are loaded once per
  step and should amortize over concurrent rows).
- **Per-request decode collapses 33.09 → 7.15 tok/s** at c=8 — each concurrent
  request runs ~4.6× slower, i.e. the 8 rows are NOT being batched into one efficient
  forward; they contend.
- **c=32 OOMs** at prefill (KV for 32 × 1600-tok slots doesn't fit at mem=0.60).
- c=1 decode 33 tok/s ≈ the parity-harness 37.6 (serving + allreduce overhead).
  TTFT 2391 ms for a 1024-tok prefill.

**So the batched-decode / continuous-batching efficiency is essentially
unoptimized** — the largest remaining DSv4 perf opportunity, orthogonal to the B=1
latency work (fused-wqkv, on-device route, the prefill DeepGEMM −5%). Getting
throughput to scale needs profiling the c>1 step (is the MoE/attention/all-reduce
batched forward inefficient, or is the scheduler running rows semi-serially?).

## Serving gap (why allreduce, and the distance to "beat-SGLang")

DSv4 serving is gated through a feature program, mapped while standing this up:
- `arle serve` does NOT wire DSv4 (`infer-api/loaded.rs:436` bails "DSv4 multi-GPU
  only; use dsv4_multigpu_parity.sh").
- `dsv4_beat_sglang_bench.sh` gates on `ARLE_DSV4_PERFORMANCE_PROFILE=sglang`, which
  the binary REJECTS: requires MTP/EAGLE spec ON, **full_decode** CUDA graph (have
  piecewise), owner-group communicators, and a DeepEP/NCCL capture/replay contract —
  **none implemented**.
- native-DeepEP serving is guarded behind that profile; **allreduce** is the only
  debug-runnable MoE backend (lower-bound throughput vs DeepEP).
- mem-fraction is fragile: 0.45/0.55 hit `ncclAllReduce unhandled cuda error` at
  prefill (0 tokens); only 0.60 served c=1/c=8.

## Rule

DSv4 perf has two orthogonal axes: **B=1 latency** (optimized this session) and
**throughput/batched-decode** (just measured — poor, 1.63× at c=8, the big
opportunity). A latency-only optimization campaign can leave throughput unscaled —
measure BOTH ([[feedback_correct_inference_not_baseline_identity]] is about
correctness; this is the perf-axis analog). The "beat-SGLang/6ms" headline target is
a **feature program** (MTP spec + full_decode graph + native-DeepEP + owner-group
comms + capture/replay), not a tuning knob — that's the honest distance to the goal.
