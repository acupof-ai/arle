# INT8/FP8 paged quant-KV wired into the dense-Qwen3 CUDA path (#68 T3)

**Date:** 2026-06-12. **Backend:** CUDA, dense Qwen3 (gate model Qwen3-0.6B).
**Scope:** `cuda-kernels` (2 new per-channel-K dequant kernels + wrappers),
`infer-cuda` (executor dtype admission, loader PageMeta quant fields,
attention refill/calibrate/quantize/fused-decode dispatch), `infer-api`/`cli`
threading. **Commits:** `d9f930b6` (port), `b2415c01` (codex-review P2:
workspace gate on the 1-split floor).
**Status: VERIFIED on H20** (same day) — **correctness LICENSED for INT8 and
FP8; decode perf −77% at B=1, optimization owed before any wider rollout.**

## Context

#68 left Qwen3.5's BF16/INT8/FP8/TQ4 matrix BLOCKED until the monolith's
quant-KV dispatch was re-ported to the rewrite's paged-KV path. The pool
plumbing (T2) and `--kv-cache-dtype` threading (T4) were already in; T3 is
the missing op layer: prefix refill, KIVI calibration, row quantize, and
fused-dequant decode behind the seam dispatch.

## What Worked

Not a 1:1 monolith port — four design corrections over the template:

1. **NEW per-channel-K dequant kernels**
   (`dequantize_paged_kv_{fp8,int8}_per_channel_k_to_hnd`): the monolith
   refilled K through the per-token dequant path, but under per-channel K
   quantize the per-token K scales are never written → prefix refill read
   zeros. K refill now consumes `k_static_scales[kv_head*head_dim+d]`.
2. **Latch-once KIVI calibration** (`k_kivi_calibrated` AtomicBool per
   layer) for both INT8 and FP8 — recalibrating on a later chunk would
   silently corrupt earlier chunks' already-quantized K under chunked
   prefill.
3. **Decode CUDA-graph hard-disabled for quant pools** (warmup early-return
   before `decode_graph_enabled()`); decode runs eager through
   `decode_attention_{fp8,int8}_per_channel_k`. V1 tradeoff, revisit only
   with a wall-clock license.
4. **1-token-prompt edge:** decode path calibrates-from-single-row when the
   latch is unset, so a prompt that never hits the prefill finalize still
   gets valid scales.

Explicitly deferred: **TQ4** (TurboQuant pools are page_size=1; the TileLang
paged prefill kernels are compiled for PAGE_SIZE=16 — no paged-prefill path
ever existed; `resolve()` bails with a pointer to #68). Accepted dead
weight: per-token `k_scales` plane still allocated under per-channel K
(unused; removal is a pool-layout change, not V1).

Pool transports needed **zero changes**: tier store
(`copy_pages_to_host/from_host`) and radix COW
(`copy_pages_device_to_device`) already carry data plane + scales + norms
per layer.

## Verification

Mac-side: `CUDARC_CUDA_VERSION=12060 cargo check -p infer-api` (cuda,no-cuda)
PASS; `cargo test -p agent-infer` (cpu,no-cuda,cli) PASS; `cargo test -p cli`
147/147; clippy `-D warnings` clean; fmt clean.

H20 pod (sm_90a, CUDA 12.9, `scripts/dsv4_toolchain.sh build`, 2026-06-12):

- **nvcc first compile of both new kernels: PASS** (full `cuda,nccl` link).
- **GPU kernel tests 3/3**: `hnd_refill_{fp8,int8}_per_channel_k_matches_scale_table`
  (exact bf16-bit match vs the scale table; head_dim 8 = vectorized path,
  head_dim 7 = scalar fallback) + the pre-existing reference test.
- **Needle gates** (`GATE_PROFILE=generic MODEL=/data01/models/Qwen3-0.6B
  SERVE_FLAGS="--kv-cache-dtype <dt>" scripts/lever_gate.sh qwen06b_<dt>_raw
  RAW=1`; RAW=1 mandatory — chat template burns the budget in `<think>`):

  | dtype | len 115/300/446/2000/8000 | verdict |
  |---|---|---|
  | bf16 (envelope, hd128 entry) | exact=3 partial=0 miss=0 DET ×5 | baseline |
  | int8 | exact=3 partial=0 miss=0 DET ×5 | **LICENSED (correctness)** |
  | fp8 | exact=3 partial=0 miss=0 DET ×5 | **LICENSED (correctness)** |
  | tq4 | — | DEFERRED (`resolve()` bails; page_size=1 vs PAGE_SIZE=16) |

- **Path probe** (RUST_LOG=info, not inferred): pool boots `format=INT8`
  (data 268.4 MB/layer vs 536.9 MB bf16 — the 2× KV-memory win) and warmup
  logs "decode graph disabled for quantized KV … fused-dequant kernels".

## Perf row (same-binary, same-shell, same-prompt, side-by-side ×3)

B=1 decode, 256 tokens, Qwen3-0.6B, GPU 0, no decode graph in any lane
(INFER_CUDA_DECODE_GRAPH unset):

| dtype | tok/s (runs 2–3 steady) | Δ vs bf16 |
|---|---|---|
| bf16 | 439 | — |
| int8 | ~100 | **−77%** |
| fp8 | ~102 | **−77%** |

The quant decode lane pays ~3 extra launches/layer (calibrate-check +
row-quantize + two-phase split-KV) against TileLang's single tuned bf16
decode kernel — at 0.6B/B=1 that's launch-bound territory. **No default
flip is proposed (bf16 stays default), so correctness licensing stands; the
perf gap is the cost of the opt-in lane today and the optimization target
(fused quantize+attend, graph re-enable for quant) before any wider
rollout.** guidellm is absent on the pod; this same-session side-by-side
A/B is the Δ% row.

## Rule

- A quant-KV port is four dataflows (refill, calibrate, quantize, decode),
  not one kernel swap — enumerate every buffer each dataflow writes and
  prove the scale plane it READS is the one the quantize path WRITES.
  The monolith's refill-reads-unwritten-per-token-K-scales bug survived
  there because the monolith never refilled a per-channel pool.
