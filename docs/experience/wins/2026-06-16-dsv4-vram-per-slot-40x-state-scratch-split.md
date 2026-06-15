# DSv4 VRAM per-slot 40× — STATE/SCRATCH split + bit-exact ledger

**Date:** 2026-06-16 · **Backend/model:** CUDA / DeepSeek-V4-Flash · **SKU:** 8×H20
TP=8/EP=8, CUDA 12.9 sm_90a (node 192.168.12.61, isolated from eic-test) · **Track:**
serving capacity / VRAM accounting.

## Goal
DSv4 serving OOM'd at `--num-slots 32` (booted only at 16) — a hard concurrency
cap. ckl: make VRAM "绝对严苛、来源绝对清晰、每一比特申请都要清楚" (every bit
accounted), fix the budget systematically, build an adversarial reconciliation loop.

## Root cause (bit-exact, measured — NOT inferred)
Per-slot cost was **2.9 GB**, but genuine per-slot KV STATE is only ~74 MB. The rest
was forward COMPUTE SCRATCH wrongly allocated PER (slot, layer) instead of once:
- `prefill_linear` (`Dsv4PrefillDeepGemmLinearScratch`): ~1.29 GB/slot (config-sized,
  layer-uniform → 1 shared instance suffices).
- `Dsv4FlashMlaDecodeState` SCRATCH (`o_accum` = accum_rows×global-h_q_d×f32 ≈
  33.7 MB/layer, + lse/tp_*/indices): ~1.39 GB/slot (95% of per-slot cost).
The codebase already classified every buffer STATE vs SCRATCH at
`attention.rs` `swap_out_image` doc-block — the SCRATCH set is overwrite-before-read,
carries no cross-call/cross-slot state, so ONE shared instance (like the existing
`dsa_shared`/`flashmla_batch`) is correct: DSv4 runs layers sequentially, one forward
at a time on `ctx.stream`.

## What worked
1. **prefill_linear → shared** (`Dsv4KvAdapter.prefill_linear`, ×1): per-slot 2.9→1.47 GB.
2. **flashmla single-row SCRATCH → shared** (`Dsv4FlashMlaDecodeScratch`, ×1, sized
   for worst-case layer): per-slot 1.47 GB→**74 MB**. STATE/CONSTANT
   (scalars/sched_meta/topk_length/num_splits/fp8 counters) stays per-slot.
3. **Bit-exact ledger**: `device_bytes()` on every DSv4 device struct + per-phase
   `mem_get_info` reconciliation in the engine build (`[vram-ledger]`).

## Results (16-slot post-build VRAM, gpu0)
| stage | per-slot | post-build | max slots |
|---|---|---|---|
| original | 2.9 GB | 84.4 GB | 16 (32 OOM) |
| + prefill hoist | 1.47 GB | 64.8 GB | 32 |
| + flashmla hoist | **74 MB (40×↓)** | **42.5 GB** | **128+ (44.5 GB free)** |

- **128 slots BOOTS** (52.9 GB used / 44.5 GB free) — was OOM at 32.
- **Ledger residual −375 MB / −0.9%** of 42.5 GB → every byte has a named source; no
  cudaMalloc-rounding bloat. The 74 MB/slot = compressor 32 + fused_wqkv 22 +
  dsa/indexer/sw_window 5ea + spec_normed 3 (all genuine KV state).
- **Correctness:** needle 8/8 @c8, 16/16 @c16, 32/32 @c32 — zero cross-slot contam.
- **Throughput-neutral** (memory-only change): c=8 85.3 tok/s vs pre-change 86.1 — no regression.

## Problems / open
- **Concurrency throughput is NOT slot-bound — it's kernel-bound.** c-sweep at
  128 slots: agg **85→96→105 tok/s @ c=8/32/64** (1.2× for 8× concurrency); per-req
  collapses 11.7→2.0. The VRAM cap is gone but aggregate plateaus ~100 tok/s, far
  from SGLang V4-Flash 4×H20-3e = 1297@c97. Decode step-time grows ~linearly with
  batch → per-row work not amortizing. Next lever is the batched-decode KERNEL
  (continuous batching + chunked prefill are already on), NOT DP-attn.
- **MTP red flag:** serve log shows `[dsv4-mtp-batched] accepted=0` per step on the
  filler prompt — spec decode rejecting heavily → pure overhead inflating per-token
  cost. Isolate (spec off) before kernel work.

## Rule
- **Per-slot = STATE only; forward compute SCRATCH is shared ×1.** Allocating
  overwrite-before-read scratch per (slot,layer) is the default concurrency killer;
  the swap-image STATE/SCRATCH classification is the authority for what's safe to share.
- **A budget that re-derives bytes by hand drifts; the allocator's own `device_bytes()`
  + a per-phase `mem_get_info` reconciliation (residual→0) is the only bit-exact check.**
- **Removing a VRAM OOM cap ≠ throughput.** Validate the slot unlock translates to
  aggregate tok/s on the SLO workload before claiming the concurrency win.

pending-remote: bench A/B ran on node .61 (8×H20); guidellm sweep TODO.
