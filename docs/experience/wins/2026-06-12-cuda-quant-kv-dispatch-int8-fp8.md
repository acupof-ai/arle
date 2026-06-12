# INT8/FP8 paged quant-KV wired into the dense-Qwen3 CUDA path (#68 T3)

**Date:** 2026-06-12. **Backend:** CUDA, dense Qwen3 (gate model Qwen3-0.6B).
**Scope:** `cuda-kernels` (2 new per-channel-K dequant kernels + wrappers),
`infer-cuda` (executor dtype admission, loader PageMeta quant fields,
attention refill/calibrate/quantize/fused-decode dispatch), `infer-api`/`cli`
threading. **Commit:** `d9f930b6`.
**Status: pending-remote** — the two new `.cu` kernels have never been
compiled by nvcc (no CUDA toolchain on the Mac), the device-gated GPU tests
(`cargo test -p cuda-kernels --release --features cuda hnd_refill`) need an
H20, and the int8/fp8 needle gates vs the BF16 envelope are owed (T5).

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

## Verification (Mac-side; device work pending-remote)

- `CUDARC_CUDA_VERSION=12060 cargo check -p infer-api --release
  --no-default-features --features cuda,no-cuda --lib` PASS.
- `cargo test -p agent-infer` (cpu,no-cuda,cli) PASS; `cargo test -p cli`
  147/147 PASS; clippy `-D warnings` clean on the touched crates; fmt clean.
- New GPU tests committed (exact bf16-bit comparison vs the scale table,
  head_dim 8 and 7 covering the vectorized + scalar kernel paths) — run on
  pod.
- BF16 default path byte-for-byte unchanged (quant ops are format-gated; the
  BF16 envelope `exact=3 partial=0 miss=0 DET` at len 115/300/446/2000/8000
  from the 06-12 hd128 fix entry is the T5 reference).

## Pending (T5 — license-or-kill per dtype)

`GATE_PROFILE=generic MODEL=/data01/models/Qwen3-0.6B
SERVE_FLAGS="--kv-cache-dtype <dt>" scripts/lever_gate.sh qwen06b_<dt>_raw
RAW=1` for int8/fp8. **RAW=1 mandatory** (chat template burns the token
budget in `<think>` → all-miss). PASS = within ±1 of the BF16 envelope per
length, zero garbage. Plus one `bench_guidellm.sh` run per dtype for the
perf row.

## Rule

- A quant-KV port is four dataflows (refill, calibrate, quantize, decode),
  not one kernel swap — enumerate every buffer each dataflow writes and
  prove the scale plane it READS is the one the quantize path WRITES.
  The monolith's refill-reads-unwritten-per-token-K-scales bug survived
  there because the monolith never refilled a per-channel pool.
