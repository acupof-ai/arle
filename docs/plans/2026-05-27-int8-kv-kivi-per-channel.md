# INT8 KV: extend KIVI per-channel K calibration from FP8 to INT8

**Status: DONE 2026-05-27** (commits `8afecffe` + `ba74dd49`).
V100 Qwen3.5-4B audit `int8 mean_match=1.0000` bit-identical with BF16
across 32 tokens × 2 prompts. Wins entry:
[`2026-05-27-int8-kv-kivi-per-channel-k-fix.md`](../experience/wins/2026-05-27-int8-kv-kivi-per-channel-k-fix.md).

## The "right problem"

V100 audit on Qwen3.5-4B (2026-05-27, see
[`wins/2026-05-27-v100-kv-precision-parity-qwen35-4b.md`](../experience/wins/2026-05-27-v100-kv-precision-parity-qwen35-4b.md)):

| KV format | mean_match | first_div_step | Decoded text shape |
|---|---:|---:|---|
| BF16 (ref) | 1.0000 | None | "KV caching and attention mask…" / "n-th Fibonacci…" |
| FP8 E4M3 | **1.0000** | None | bit-identical to BF16 |
| **INT8** | **0.0938** | **1** | coherent but semantically different text from step 1 |
| TQ4 | n/a | n/a | V100 sm_70 not supported |

INT8 step-1 deterministic divergence reproduces across both prompts and
matches the historical May 26 audits (mean_match ≈ 0.03–0.06,
first_div_step=1). FP8 reaches bit-identical because of the KIVI
per-channel K calibration landed in commit `8c6d92db` (2026-05-26).

The actual bug: **KIVI per-channel K calibration is gated to FP8 only**.
`crates/cuda-kernels/src/paged_kv.rs:495`:

```rust
let (k_static_scales, k_kivi_calibrated) =
    if matches!(format, KVFormat::FP8E4M3) && pool_bytes_per_layer > 0 {
        // alloc per-(kv_head, head_dim) static scales
    } else {
        (None, Vec::new())   // ← INT8 lands here
    };
```

INT8 quantize/dequantize then falls back to per-(token, head) absmax.
K activations in Qwen3.5 (and most modern LLMs) have channel-wise
outliers that per-token absmax silently flattens — the same precision
loss KIVI was added to fix for FP8. INT8 has the same axis of error
and just hasn't been wired up.

This is not a kernel correctness bug, it is a **feature gap**. The
existing INT8 per-(token, head) path is computing exactly what its
contract says, that contract just isn't sufficient for K cache on
modern dense transformers.

## What FP8 already has (template for INT8)

End-to-end FP8 KIVI pipeline:

1. **Allocate**: `k_static_scales: Vec<CudaSlice<f32>>` per layer, shape
   `[num_kv_heads, head_dim]`, zero-initialised
   (`paged_kv.rs:494-510`).
2. **Calibrate during prefill**: first prefill batch invokes
   `compute_k_per_channel_absmax_cuda` per layer, atomic-max running
   absmax into the table; then once at the end
   `finalize_k_per_channel_scales_cuda` divides by 448 (FP8 max) and
   floors at 1e-30.
3. **Quantize-on-append (decode)**:
   `quantize_paged_kv_fp8_per_channel_cuda` — uses static
   `[num_kv_heads, head_dim]` scales instead of computing absmax per
   token.
4. **Decode attention**:
   `decode_attention_fp8_per_channel_k_cuda` reads `k_static_scales`
   in lieu of `K_scales` for K dequant; V stays per-(token, head).
5. **Dispatch**: `batch_decode.rs:2398-2425` and `:2521-2545` check
   `k_static_scales_ptr.is_some()` and route to the per-channel
   kernels.

## INT8 mirror plan

Per-step changes, each independently shippable + revertable behind
the same `k_static_scales_ptr` Some/None gate already present.

**Step 1 — `paged_kv.rs:495`**: extend the gate:
```rust
if matches!(format, KVFormat::FP8E4M3 | KVFormat::INT8) && pool_bytes_per_layer > 0
```
Side effect: INT8 pool now allocates one `[num_kv_heads, head_dim]`
f32 buffer per layer (~16 KB × 36 layers = 576 KB for Qwen3-4B).

**Step 2 — prefill calibration**: confirm the existing
`populate_kivi_k_scales_from_prefill` reads bf16 K (format-agnostic).
It already does because the work buffer is bf16 for both FP8 and
INT8. The only INT8-specific part is the divisor — FP8 divides by
448, INT8 needs 127. Approach:
- Add `finalize_k_per_channel_scales_int8_cuda` (mirror of FP8 with
  divisor = 127.0f, same 1e-30 floor).
- OR parameterise `finalize_k_per_channel_scales_cuda(divisor)` and
  have two thin Rust wrappers. (Choose this — less code duplication.)

**Step 3 — quant kernel**: add
`quantize_paged_kv_int8_per_channel_kernel` (one-shot port of the FP8
template; replace `__nv_fp8_e4m3(val * inv_scale)` with
`max(-127, min(127, __float2int_rn(val * inv_scale)))`, dst type
`int8_t`).

**Step 4 — decode attention**: add
`decode_attention_int8_per_channel_k_partial_kernel` (mirror of the
FP8 per-channel K decoder; replaces per-(token, head) scale lookup
with per-channel scale lookup). V kernel side stays per-(token,
head). Same split-KV merge pass.

**Step 5 — Rust FFI**: add `quantize_paged_kv_int8_per_channel`
and `decode_attention_int8_per_channel_k` in
`crates/cuda-kernels/src/kv_quant.rs`, mirror the FP8 signatures.

**Step 6 — dispatch wiring**: in `batch_decode.rs` for Qwen3 and
Qwen3.5, the existing `if let Some(k_static_scales_ptr) = ... {}`
branch already handles per-channel K dispatch — extend the match
arm so `KVFormat::INT8` also follows it (currently only `FP8E4M3`).

## Validation

Re-run V100 audit:
```
KV_PARITY_PROMPTS=2 KV_PARITY_MAX_TOKENS=16 \
  cargo test --release -p infer --features cuda \
  --test kv_precision_parity_qwen35 -- --nocapture --test-threads=1
```

Pre-fix: `int8 mean_match=0.0938, first_div=step 1`.

Post-fix expected: `int8 mean_match ≥ 0.95` (parity with FP8) or
identifiable per-token drift that needs further investigation but is
NOT step-1 deterministic divergence. Decoded INT8 tokens should
match BF16 reference text closely.

## Trip wires

- **If INT8 KIVI mean_match > 0.99**: ship. INT8 KV is now
  production-equivalent to FP8.
- **If INT8 KIVI improves but stays < 0.9**: K is only half the
  precision story. Investigate V quant separately (KIVI keeps V per-
  token; might need per-channel V too on Qwen3.5 hybrid layers).
- **If INT8 KIVI shows no improvement**: the per-channel calibration
  isn't actually being applied. Verify by dumping
  `k_static_scales` content for layer 0 and confirming non-zero
  values after first prefill.

## Effort

~1–2 days focused. Most code is direct copy-with-substitution from
the FP8 KIVI commit. Higher-risk parts:
- Make sure `populate_kivi_k_scales_from_prefill` runs for INT8
  contig prefill (the bf16 work buffer might be allocated only
  conditionally for INT8 — needs verification).
- Per-decode-step quantize was added for both formats already
  (`quantize_paged_kv_single` for INT8); only the per-channel variant
  needs adding.

## Related

- [`docs/experience/wins/2026-05-27-v100-kv-precision-parity-qwen35-4b.md`](../experience/wins/2026-05-27-v100-kv-precision-parity-qwen35-4b.md) — the V100 audit that surfaced this gap.
- [`docs/plans/2026-05-26-fp8-kv-per-channel-k-fix.md`](2026-05-26-fp8-kv-per-channel-k-fix.md) — the FP8 KIVI plan; INT8 follows the same template.
- Commit `8c6d92db` (FP8 KIVI implementation V1, gated) — the source
  of truth for the FP8 per-channel infrastructure being ported here.
- [KIVI paper](https://arxiv.org/abs/2402.02750) — ICML 2024, per-
  channel K + per-token V asymmetric quantization for KV cache.
