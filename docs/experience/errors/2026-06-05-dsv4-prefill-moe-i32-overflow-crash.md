# DSv4 prefill broken at all production shapes — MoE padded-layout i32 work-size overflow

**Date:** 2026-06-05. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.

## Context

The session goal is "prefill 高性能 + decode ~6ms". Decode had a strong arc
(scalar 23.7 → 33.0 tok/s) but **prefill was never assessed this session** — a
Stop-hook caught the gap. The assumed prefill blocker was the FlashMLA-prefill
>24K TMA-OOB (`wins/2026-05-27-dsv4-flashmla-v2-22x-prefill-22x-pre-crash`). That
was **wrong**.

## Root Cause

A ladder probe (5-tok / 8K) found prefill **crashes at any production shape**,
not just >24K, and not in FlashMLA at all:

- **FlashMLA prefill isn't even wired** — `dsv4_parity` exports only the *decode*
  sparse-decode symbol; `attention.rs` keeps multi-token prefill on the scalar
  reference path; there is no `ARLE_DSV4_FLASHMLA_PREFILL` gate. The 2026-05-27
  22× was a prior experiment not in the current tree.
- **8K prompt → all 8 ranks `CUDA_ERROR_INVALID_VALUE`**, backtrace
  `infer_cuda::moe::dsv4_gpu::deepgemm_grouped_experts`. The MoE prefill uses the
  **padded masked layout**: `max_m = prompt_len × topk`; at 8K, `max_m = 8192×6 =
  49152`. The pack/swiglu/unpad kernels compute the work-size
  `active_count × max_m × hidden_dim` (and `scale_k_blocks × max_m × active_count`
  for the launch grid) **as `int`**: `32 × 49152 × 7168 = 11.27B > INT_MAX` →
  overflow → garbage grid / index → `CUDA_ERROR_INVALID_VALUE`.
- Threshold ≈ **1560 tokens** → 8K / 16K / 24K / 32K all crash. So DSv4 prefill at
  every binding SLO shape (32K input) **crashes** — not "slow", broken.

## Fix

Two layers (verified: 4K prefill completes, no `CUDA_ERROR_INVALID_VALUE`, 8 ranks
clean, first prefill-argmax token sensible):

1. **`dsv4_deepgemm_ops.cu`** (pack/swiglu/unpad): compute the flat index / total /
   grid in **`int64_t`** with a `if (grid > INT_MAX) return CUDA_ERROR_INVALID_VALUE`
   guard before the `int` launch cast; flatten the `grid.y = max_m` launches.
   Defensive — no more silent overflow.
2. **`infer-cuda/src/moe.rs`** (the real fix): prefill/non-decode MoE switches from
   the padded masked layout to the **contiguous active-row layout** (same as the
   +12.78% decode win, `wins/2026-06-05-dsv4-moe-contiguous-decode-layout-13pct`) —
   `rows = seq_len`, no `num_groups × max_m` padded slab, no unpad. This roots out
   the overflow at the source (the work-size never reaches `max_m`-padded scale).
   New `dsv4_fill_m_indices_from_counts_cuda` builds the contiguous `m_indices`.
   Prefill default flips to contiguous (the padded path *crashed*, so it cannot
   remain the default).

## Rule

When a forward path crashes only at large shapes with `CUDA_ERROR_INVALID_VALUE`
in a launch wrapper, suspect an **`int` work-size / grid-dim overflow** before
suspecting the kernel: a padded layout multiplies `tokens × topk × experts ×
hidden`, which blows past `INT_MAX` fast (here at ~1560 tokens). And **probe the
ladder before trusting a stale "known blocker"** — the assumed >24K FlashMLA-OOB
was a prior-session artifact; the real blocker was an i32 overflow at 8K and
FlashMLA prefill wasn't wired at all. Perf is the *next* step: 4K prefill is
27.8 s (scalar attention, FlashMLA prefill still unwired) — unblocking ≠ fast.
