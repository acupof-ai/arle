# Plan: Gemma 4 Model + GGUF Loading

**Status**: Split — GGUF **shipped**, Gemma model **still queued**
**Date**: 2026-04-10 (original) / 2026-04-15 (status update)
**Goal**: Minimal-effort multi-architecture support via Gemma 4 + GGUF weight format

> **2026-04-15 status update.** The two halves of this plan diverged:
>
> - **GGUF loading (§2): shipped.** The minimal parser in
>   [`infer/src/gguf.rs`](../../infer/src/gguf.rs) landed, `weight_loader.rs`
>   has BF16 / F16 / Q8_0 / Q4_K_M fast paths, and
>   [`plans/q4k-native-gpu.md`](q4k-native-gpu.md) describes the native Q4_K
>   kernel that is now in production. See the three 2026-04-10 experience
>   errors under `docs/experience/errors/2026-04-10-*gguf*.md` for the debug
>   trail that preceded it.
> - **Gemma 4 model (§1): still not shipped.** `model_registry.rs` detects
>   `Gemma{,2,3,4}ForCausalLM` architectures and maps them to `ModelArch::Gemma`,
>   but `backend/cuda/bootstrap.rs` has no Gemma load path and no
>   `model/gemma*.rs` files exist. The Gemma-specific forward-pass work
>   (pre+post RMSNorm, logit soft-cap, sliding-window attention) below is
>   still accurate as a forward-looking design.

---

## 1. Gemma 4 Model

### What exists
- `model_registry.rs`: `Gemma4ForCausalLM` → `ModelArch::Gemma` (detected)
- `bootstrap.rs`: no Gemma load path (not wired)

### Architecture differences from Qwen3

| Feature | Qwen3 | Gemma 4 |
|---------|-------|---------|
| QK norm | Yes (per-head weight) | No |
| Attention bias | Yes | No |
| RMSNorm | Standard | **Pre + post** (2× per layer) |
| Logit soft-capping | No | **Yes** (tanh gate before softmax) |
| Sliding window | No | **Alternating** (local 4K + global) |
| Tied embeddings | Model-dependent | Yes |
| RoPE | Standard (θ=1M) | Standard (θ=10K default) |

### Implementation plan

**Reuse from Qwen3** (via common.rs):
- MLP loading + forward (SwiGLU, same structure)
- Embedding + output projection
- RoPE precomputation
- CUDA Graph infrastructure

**New code needed**:

| File | Lines (est.) | Key difference |
|------|-------------|----------------|
| `model/gemma.rs` | 15 | Module file |
| `model/gemma/config.rs` | 120 | Parse Gemma config (sliding_window, attn_logit_softcapping) |
| `model/gemma/weights.rs` | 200 | Load weights — no QK norm, identity norm vectors |
| `model/gemma/forward.rs` | 250 | ModelForward impl |
| `model/gemma/decode.rs` | 130 | Decode — reuse Qwen3 with logit softcap |
| `model/gemma/prefill.rs` | 200 | Prefill — pre+post norm per layer |
| `model/gemma/decode_buffers.rs` | 80 | Simpler than Qwen3 (no QK norm buffers) |
| `model/gemma/batch_decode.rs` | 600 | Adapt from Qwen3 — pre+post norm |
| `bootstrap.rs` | +20 | Wire ModelType::Gemma |

**Sliding window attention**: For MVP, ignore sliding window (use full attention for all layers). Add sliding window as a later optimization — it only matters for contexts > window size.

**Logit soft-capping**: `score = tanh(score / softcap_value) * softcap_value` applied in attention. Can be fused into the existing attention kernel with a config flag, or applied as a post-processing step.

### Estimated effort: 1 session (~1600 lines, mostly adapted from Qwen3)

---

## 2. GGUF Loading — **Shipped**

> Phase 1 (BF16/F16), Phase 2 (Q8_0), and Phase 3 (Q4_K_M, via the native
> q4k_gemv path, not dequant-to-BF16 as originally sketched) all landed.
> The design sections below describe the pre-implementation plan; the
> implementation matches the Phase 1/2 shape but Phase 3 chose the **keep
> Q4_K packed on GPU** path documented in
> [`q4k-native-gpu.md`](q4k-native-gpu.md) instead of the "dequant at load
> to BF16" option in §2 below.

### What exists (2026-04-15)
- `infer/src/gguf.rs`: minimal parser + tensor directory + per-tensor
  readers (`read_tensor_bf16`, `read_tensor_q4k_packed`, Q8_0 fast path)
- `infer/src/weight_loader.rs`: `load_tensor_2d_gguf` with BF16 / F16 /
  Q8_0 / Q4_K_M fast paths, plus `load_tensor_2d_gguf_v_reorder_rows`
  for V-head row permutation on packed Q4_K
- GGUF → HF name mapping handled inline in `weight_loader.rs`
- Integrated with the 3 currently-supported models (Qwen3, Qwen3.5, GLM4)

### Original plan (preserved for historical rationale)

### What existed (2026-04-10, at plan-write time)
- `quant.rs`: `QuantFormat::Gguf`, `GgufConfig`, `try_load_gguf()` (finds .gguf file)
- `weight_loader.rs`: no GGUF tensor reading (only safetensors)

### GGUF format
- Binary header: magic, version, tensor count, metadata KV pairs
- Tensor info: name, dims, dtype, offset into data section
- Data section: contiguous tensor data (various quantization formats)

### Quantization block formats (llama.cpp)

| Format | Bits | Block size | Structure |
|--------|------|-----------|-----------|
| Q4_0 | 4 | 32 | 1× f16 scale + 16× u8 (2 nibbles each) |
| Q4_K_M | 4 | 256 | min/scale f16 + sub-block scales + packed nibbles |
| Q8_0 | 8 | 32 | 1× f16 scale + 32× i8 |
| F16 | 16 | 1 | Standard IEEE FP16 |
| BF16 | 16 | 1 | Standard BFloat16 |

### Implementation plan

**Phase 1: GGUF parser + BF16/F16 loading**
- Parse GGUF header + tensor directory
- Load F16/BF16 tensors as DeviceMatrix (dequant-at-load for F16→BF16)
- Map GGUF tensor names → HuggingFace names (e.g., `blk.0.attn_q.weight` → `model.layers.0.self_attn.q_proj.weight`)

**Phase 2: Q8_0 loading**
- Dequant Q8_0 blocks at load: `val = int8 * scale` → BF16
- Reuse existing W8A16 GEMV kernel for runtime

**Phase 3: Q4_K_M loading (stretch)**
- Dequant at load to BF16 (simplest) or reuse existing W4A16 kernel
- Q4_K_M has sub-block structure that doesn't map cleanly to our W4 kernel

### Rust crates
- `gguf`: Pure Rust GGUF parser (limited quality)
- Better: write minimal parser (header + tensor directory only, ~200 lines)

### Files to create/modify

| File | Action |
|------|--------|
| `infer/src/gguf.rs` (new) | GGUF header parser, tensor reader, name mapping |
| `infer/src/weight_loader.rs` | Add `load_from_gguf()` alternative to safetensors path |
| `infer/src/lib.rs` | `pub mod gguf;` |

### Estimated effort: 1 session for Phase 1 (BF16/F16), Phase 2 (Q8_0) adds ~200 lines

---

## 3. Priority

1. **Gemma 4** first — more user demand, reuses existing kernel infrastructure
2. **GGUF Phase 1** next — enables loading llama.cpp community models
3. **GGUF Q4/Q8** — enables quantized GGUF models (biggest community segment)

---

## 4. Generalization opportunity

Both Gemma and future models (Llama, Mistral) share 90% of the forward path. Consider extracting:

```rust
/// Generic transformer forward: embed → norm → attn → norm → mlp → logits
pub trait TransformerConfig {
    fn has_qk_norm(&self) -> bool;
    fn has_post_attn_norm(&self) -> bool;   // Gemma: yes, Qwen3: yes
    fn logit_softcap(&self) -> Option<f32>; // Gemma: Some(30.0), others: None
    fn rope_theta(&self) -> f32;
    // ...
}
```

This would reduce each new model to ~200 lines of config + wiring instead of ~1600 lines of copy-paste. **But**: the current approach (per-model files) is simpler to debug and profile. Generalize only after 3+ models share the pattern.
