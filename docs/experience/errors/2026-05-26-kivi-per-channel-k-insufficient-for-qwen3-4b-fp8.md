# KIVI per-channel K insufficient as sole FP8 fix on Qwen3-4B / A100

## Context

After the 2026-05-26 FP8 KV step-1 catastrophic divergence root-cause
investigation (f50dd674) traced the failure to **precision-floor
compounding through 36 dense full-attention layers** — K/V scales
shrinking 10× from layer 0 to layer 17, FP8 E4M3's quant step at
deeper layers reaching `scale/448 ≈ 4e-5`, and intermediate
softmax×V values truncated to zero by the BF16 attn-output write —
we implemented the **KIVI** scheme (per-channel K, per-token V) as
the first license-or-kill candidate fix.

KIVI is the established academic fix for the *K-outlier-channel*
half of the FP8 KV quality problem on dense decoders.

## Implementation

V1 landed across 8 files (commit 8c6d92db, ~960 insertions):

- `crates/cuda-kernels/csrc/kv/kv_quant.cu` — three new kernels:
  - `quantize_paged_kv_fp8_per_channel_kernel`: consumes a pre-computed
    `[num_kv_heads, head_dim]` f32 scale table.
  - `compute_k_per_channel_absmax_kernel`: per-(kv_head, head_dim)
    absmax via `atomic_max_float`.
  - `finalize_k_per_channel_scales_kernel`: divides by FP8 E4M3 max
    (448), floors at 1e-30.
- `crates/cuda-kernels/csrc/attention/decode_attention_quantized.cu`
  — `decode_attention_fp8_per_channel_k_partial_kernel<HEAD_DIM>`:
  pre-loads `k_scale_reg[EPT]` once per warp from
  `K_static_scales[kv_head * HEAD_DIM + d]`, then multiplies per-dim
  during the QK dot.
- `crates/cuda-kernels/src/paged_kv.rs` — new `k_static_scales:
  Option<Vec<CudaSlice<f32>>>` field + `k_kivi_calibrated:
  Vec<AtomicBool>` latch (per-layer, batch-local recalibration).
- `infer/src/model/qwen3/prefill.rs::finalize_paged_prefill_kv_layer`
  — calibration path: memset scales to 0 → absmax accumulate →
  finalize → set latch → quantize K with static scales.
- `infer/src/model/qwen3/batch_decode.rs::decode_batch_layer_inner`
  — dispatch fork on `pool.k_static_scales_ptr.is_some()`.
- V branch unchanged — `quantize_paged_kv_fp8` (per-(token, head))
  on the V side, KIVI's documented asymmetric choice.

Three CUDA kernel unit tests pass in isolation.

## Audit (A100 sm_80, Qwen3-4B, kv_precision_parity)

```text
[kivi-prefill] layer=0 token_count=4096 KIVI per-channel K path engaged
[kivi-decode]  layer=0 batch=1     KIVI per-channel K decode-attn engaged
[kivi-scales]  layer=0 n=1024 min=1.035e-5 max=3.973e-1 mean=5.190e-3 near_zero=0/1024
[kivi-scales]  layer=0 head=0 dims[0..8]=[0.019, 0.0096, 0.0072, 0.0076, 0.012, 0.0012, 0.0027, 0.0034]
[kivi-scales]  layer=0 head=7 dims[0..8]=[0.0012, 0.0013, 0.00090, 0.0013, 0.0020, 0.0015, 0.00032, 0.0019]

kv-parity: bf16 mean_match=1.0000 first_div=None/None     passed=true
kv-parity: int8 mean_match=1.0000 first_div=None/None     passed=true
kv-parity: fp8  mean_match=0.0156 first_div=Some(0)/Some(1)
kv-parity: tq4  mean_match=0.0000 first_div=Some(0)/Some(0)
```

Calibration produces **non-degenerate** scales:
- 0/1024 near-zero (no floor truncation)
- 38× dynamic range across the table (min 1e-5 → max 4e-1)
- ~10× head-0 vs head-7 attenuation (consistent with the 10× layer-0
  to layer-17 shrinkage observed in f50dd674's multi-layer dump —
  this matches deeper-head behavior, calibration is "seeing" the
  real activation magnitudes)

But fp8 `mean_match=0.0156` is **bit-identical** to the pre-KIVI
baseline. KIVI changes the K representation, the dispatched kernel
is provably engaged, scales are reasonable — and yet the
catastrophic divergence is unchanged.

## Root cause of the kill

KIVI fixes the **K outlier channel** half of the FP8 quality
problem. The Qwen3-4B catastrophic divergence is **downstream** of
K quantization:

1. **V is still legacy per-(token, head).** KIVI's asymmetric
   choice keeps V on per-token scales because V doesn't show the
   same outlier-channel structure K does. But on Qwen3-4B, V scales
   *do* shrink ~10× across depth, and at quant step `scale/448 ≈ 4e-5`
   intermediate `softmax × V` values fall below BF16's representable
   minimum (~6e-5) and round to zero before reaching the attention
   output.

2. **BF16 attn-out write truncation.** Even with FP32 inside the
   attention kernel, the write back to BF16 erases the small
   contributions that survive V dequant. f50dd674's layer 17/35 dump
   showed `attn_out = [0.0, 0.0, 0.0, 0.0]` for the first 4 dims —
   not a K-side problem, an output-precision problem.

Per the §0 SOLID rule: **wall-clock framing is ground truth.**
mean_match across all decoded tokens is the user-visible quality.
KIVI changes K math but doesn't move this metric → KIVI is not the
fix for this failure mode.

## Decision: KILL KIVI as a sole FP8 fix; KEEP as scaffolding

KIVI stays in the tree (correct math, working dispatch, unit-tested)
as the foundation any per-channel K scheme will need. It is opt-in
via the FP8 format dispatch — no production cost.

Three structural recovery paths remain for FP8 KV usability on
deep dense models:

1. **KVLinC Hadamard pre-rotation** (paper-recommended for
   Qwen3-4B specifically) — applies a Hadamard transform to flatten
   the per-channel K *and* V distributions before quantization.
   Bounded-cost, but cross-layer rotation invariant must hold.

2. **FP32 attention accumulator + FP32 attn-out write** — addresses
   the BF16 truncation directly. Higher SRAM cost, may need TileLang
   re-codegen. Closest to the root cause.

3. **Per-group V quantization** — extend KIVI scaffolding to V with
   group_size=32 or 64 (between per-token and per-channel).

## Rule

When a KV-quant scheme's debug dump shows **sensible scales but
unchanged downstream metric**, the kernel math is correct but the
*scheme itself* is not addressing the failure mode. Don't iterate on
floor / threshold / group-size knobs — pivot to a different
structural fix (precision of downstream ops, pre-rotation, etc.).

## Related

- `docs/experience/errors/2026-05-26-fp8-kv-step1-divergence-known-deferred.md`
  — the multi-layer attn_out dump and three-recovery-path framing.
- f50dd674 — root-cause investigation closing commit.
- 8c6d92db — KIVI v1 implementation commit.
- 0ef57994 — KIVI scale dump instrumentation.
