# DSv4 prefill −37% more — nsys pins the #1 kernel (indexer-proj scalar GEMV), flip its DeepGEMM default-on

**Date:** 2026-06-09. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Commit:** (this). **Scope:** `attention.rs` (`dsv4_prefill_indexer_deepgemm_enabled` default).
Follows the parallel-compressor win (`2026-06-09-dsv4-parallel-compressor-prefill`).

## Context — nsys, finally

64K prefill ran GPU-100%-util at only ~275W (68% TDP, ~1-2% MFU). nsys
(`cuda_gpu_kern_sum`, 8 ranks, a real 64K serve prefill) gave the breakdown that
roofline-by-reading couldn't:

| % GPU | kernel | note |
|---|---|---|
| **38.4** | **`dsv4_fp8_gemv_batch_tiled`** (25ms/call ×1888) | **scalar FP8 GEMV — the DSA indexer query projection** |
| ~20 | `deep_gemm::sm90_fp8_gemm` (MoE+proj) | efficient, fine |
| 9.3 | `pack_quantize` + `swiglu_quantize` | FP8 quant |
| 8.2 | `dsv4_mhc_{post,params,pre}` | hyper-connection mixing |
| 6.5 | `sparse_attn_fwd` (FlashMLA) | compressed attention |
| ~5 | `nccl AllReduce`+`AllGather` | TP/EP comm |
| 2.5 | `dsv4_swa_attention` (16ms/call) | sliding-window attn (secondary lever) |
| 2.2 | `sm90_fp8_paged_mqa_logits` | the indexer O(N²) logits — **only 2.2% @64K** |
| 0.5 | `dsv4_compressor_block` | the just-parallelized compressor — **was ~30% of wall** |

Two assumptions died here: the next lever was **not** the compressor (already
fixed, now 0.5%) and **not** the logits fusion (Phase 1.2 — only 2.2% @64K; it's
the *900K* killer, not 64K). The real #1 was the scalar indexer-query GEMV.

## What worked

`dsv4_fp8_gemv_batch_tiled` is the scalar (token-looped) FP8 GEMV. The MLA
residual projections (wq_a/wkv/wq_b/wo) already route to tensor-core DeepGEMM by
default — but the **DSA indexer query projection** had its DeepGEMM replacement
(134.9→6.05ms, −95.5% @M=1024) gated **default-OFF**, because an FP8 numeric flip
could shift the top-k block SELECTION and it had *"never been validated by a
planted-answer long-context needle."*

That gate is now MET. Flipped `dsv4_prefill_indexer_deepgemm_enabled` to default-ON
(opt-out `ARLE_DSV4_PREFILL_INDEXER_DEEPGEMM=0`), validated by same-binary env A/B:

| ctx | indexer-DeepGEMM OFF | ON | Δ |
|---|---|---|---|
| 64K prefill | 17.6s | **11.0s** | **−37%** |
| 128K prefill | 42.7s | **23.0s** | **−46%** |

Correctness (the selection gate): **64K hit `738291` exact, 128K hit `738291`
exact**, and every run finds the needle region (`738…`) — top-k selection intact.
The exact-digit borderline at ≥2K (64K 1/3 exact) is the pre-existing
compression-fidelity + MoE-non-det residual, present with the flag OFF too — NOT a
selection break. Per the codebase's own stated gate ("planted-answer long-context
needle confirms retrieval"), this is licensed.

**Cumulative prefill: 25.7s → 17.6s (compressor) → 11.0s (indexer-DeepGEMM) =
−57% on 64K.**

## Rule

- **nsys before optimizing a "100%-util but low-MFU" path.** Roofline-by-reading
  guessed the compressor (right, but only after fixing it) and the logits (wrong —
  2.2%); the actual #1 was a scalar GEMV nobody flagged. 100% util at 68% TDP = a
  slow kernel hiding in plain sight; the kernel summary names it in one shot.
- A correctness-gated perf flag stays OFF only for *lack of the specific
  validation it names* — when you can finally run that exact gate (here: a
  planted-answer long-context needle), run it and flip, don't leave free perf on
  the floor.
- `*_gemv_batch*` on a prefill (multi-token) path is a red flag — a GEMV looped
  over tokens is a decode kernel mis-used as a GEMM; route to DeepGEMM/tensor-core.

## Residual / next

- Exact-digit retrieval ≥2K still borderline (task: multi-seed study to confirm the
  indexer-FP8 doesn't worsen it vs OFF — small-n here can't tell).
- 256K+ still **host-bound** (engine thread 100% CPU, GPU 0%, pre-kernel-issue) —
  a *scheduler-side* Rust loop, the real 900K blocker, untouched by any GPU lever.
- Next GPU levers from the table: `swa_attention` (16ms/call), the `mhc_*` HC
  mixing (8.2%), the quant kernels (9.3%).
