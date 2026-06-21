# OptiQ Metal speedup — keep mixed-bit MLP quantized via separate-matmul path (7.96 → 12.4 tok/s)

## Context
Follow-up to [2026-06-21-metal-optiq-per-weight-quant-load.md](2026-06-21-metal-optiq-per-weight-quant-load.md),
which loaded `mlx-community/Qwen3.6-27B-OptiQ-4bit` but ran **7.96 tok/s vs plain 4bit's 14.71
(−46%)**. Root cause: the 18 mixed-bit MLP layers (`gate=4-bit, up=8-bit`, `down=4-bit`) can't
row-merge into one quantized `gate_up`, so `weights.rs::concat_weight_rows` **dequantized both to
dense bf16** — heavier reads (weights 18 GB vs 14) and ineligible for the compiled MLP path.

**Diagnosis correction:** the prior entry assumed the 18 mixed-bit layers were dense full-attn
layers. They are not — **14 are GDR (linear-attn) layers, 4 are full-attn**. The fix had to cover
both attention families.

## What Worked
Keep mixed-bit gate/up **quantized** and run them as two separate quantized matmuls (the existing
`mlp_separate` path), instead of dequant-merging to bf16.

1. **Rust `weights.rs`** — `MlpInputProjection` gained a `Separate { gate_dim }` variant + a
   `WeightTensor::quant_bits()` accessor. `qwen35.rs::build_qwen35_dense_mlp` detects mixed-bit
   (`gate.bits != up.bits`) and emits `Separate` (no dequant) instead of `MergedQuantized`.
2. **Rust `qwen35.rs` push** — for a `Separate` layer, `gate_up_id = -1` (no merged projection);
   full-attn layers call the new `qwen35_compiled_set_full_separate_mlp_v2(gate_id, up_id)`; GDR
   layers reuse the existing `set_separate_mlp_v2`/`set_separate_proj_v2` registration.
3. **C++ `mlx_qwen35_model.cpp`** — `FullAttnLayerWeights` gained `gate_proj/up_proj/has_separate_mlp`
   (mirroring `GdrLayerWeights`). MLP routing sends a full-attn layer with `has_separate_mlp` to
   `mlp_separate` (unconditional — a mixed-bit layer has no merged fallback). GDR routing now also
   fires unconditionally when no merged `gate_up` exists (`gate_dim == 0` sentinel), still env-gated
   when a merged path is present (no same-bit regression).
4. **C++ down-projection fix (the bug that blocked it)** — `push_full_attn_v2`/`push_gdr_v2` set
   `down` only when `gate_up_id >= 0 && down_id >= 0`. With `gate_up_id = -1`, `down` stayed the
   default `array(0)` int32 scalar → `[quantized_matmul] scales.dtype()==int32` crash at warmup.
   Split so `down` is set whenever `down_id >= 0`, independent of `gate_up_id`.
5. **Compiled `mlp_separate`** — added `compiled_mlp_separate_fn` keyed by
   `(gate_dim, gate_bits, up_bits, down_bits, gs)`: two `quantized_matmul` + inlined swiglu + down
   matmul, no split (gate/up already separate). Shaped, decode S=1 only, gated under the same
   `INFER_METAL_NO_MLP_COMPILE` env as `compiled_mlp_fn` for A/B.

## Results (M-series Metal, same-binary A/B, max_tokens 96, temp 0, prompt "Explain how transformer
attention works, step by step.")

| Config | tok/s | Δ vs OptiQ-before | notes |
|--------|-------|-------------------|-------|
| Plain 4bit baseline (before & after) | 14.5–14.7 | — | **no regression**; "Paris" correct |
| OptiQ **before** (dequant-to-bf16) | 7.96 | — | prior entry |
| OptiQ **after**, compile ON (default) | **12.4** | **+56%** | "Paris", primes correct |
| OptiQ after, compile OFF (`INFER_METAL_NO_MLP_COMPILE=1`) | 12.31 | +55% | bit-identical to ON |

- **Correctness gate — bit-identical:** OptiQ compile-ON output == compile-OFF output, byte-for-byte
  (390 bytes, temp 0). The compiled `mlp_separate` is numerically equivalent to the per-op path.
- The win is from **keeping weights quantized** (compile ON vs OFF is a wash here, ~0.1 tok/s):
  the dequant-to-bf16 read cost was the dominant penalty, exactly as the prior entry predicted.
- Build clean (`cargo build --release --features metal,no-cuda`); 30/30 infer-metal tests pass.

## Rule
- **Verify the model's actual layer topology before assuming where a fix lands.** The diagnosis
  pinned "18 dense full-attn layers"; the config showed 14 GDR + 4 full-attn. A fix scoped to one
  family would have half-worked and looked like a partial regression.
- **Enumerate every device buffer a state change reads.** Dropping the merged `gate_up` (id = -1)
  silently un-set the `down` projection too (same `if` guard) → an int32 scalar fed to
  `quantized_matmul`. The default-`array(0)` placeholder masks the gap until it's used as a real
  weight; split the guard so each buffer is set on its own precondition.
