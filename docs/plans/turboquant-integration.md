# TurboQuant Integration — Unified Rotation-Based Quantization

**Status**: Design  
**Date**: 2026-04-09  
**Scope**: KV cache compression (Phase 1) + weight quantization ITQ3_S (Phase 2)

---

## 1. Motivation

Current quantization in agent-infer:

| Target | Formats | Best compression | Limitation |
|--------|---------|-----------------|------------|
| Weights | W2/W4/W8 scalar, Marlin W4 TC | 4-bit (4x) | No rotation → outlier sensitivity |
| KV cache | INT8 symmetric, FP8 E4M3 | 8-bit (2x) | Scalar quantization, no distribution shaping |

TurboQuant (Google Research, ICLR 2026) introduces **rotation-based quantization** that shapes the data distribution before quantizing, achieving near-optimal distortion at much lower bit widths:

| Target | TurboQuant | Compression vs BF16 | Quality |
|--------|-----------|---------------------|---------|
| KV cache | 3-bit MSE | **5.3x** | Lossless at 3.5-bit |
| Weights | ITQ3_S (3-bit + Hadamard) | **5.3x** | 57% less PPL gap vs IQ3_S baseline |

Combined with existing W4 weight quant: **W4 weights + 3-bit KV** → enables 70B models on 24GB GPUs.

---

## 2. Core Algorithm

### 2.1 TurboQuant MSE (for KV cache values + keys)

```
quantize(x, bits):
  norm = ||x||_2                        // f16 scalar per token per head
  x_unit = x / norm                     // project onto unit sphere
  y = x_unit @ Pi^T                     // random orthogonal rotation (D×D)
  indices = searchsorted(boundaries, y)  // per-coordinate Lloyd-Max quantization
  packed = bitpack(indices, bits)        // compact storage
  → (packed, norm)

dequantize(packed, norm):
  indices = unpack(packed, bits)
  y_hat = codebook[indices]             // gather from 2^b centroids
  x_hat = y_hat @ Pi                    // inverse rotation (Pi is orthogonal)
  → x_hat * norm
```

**Why it works**: Random orthogonal rotation maps any unit vector's coordinates to a known Beta distribution. Lloyd-Max quantization on this known distribution is provably optimal — no calibration data needed.

### 2.2 Lloyd-Max Codebook (precomputed)

After rotation, each coordinate follows `Beta((D-1)/2, (D-1)/2)` rescaled to [-1, 1].

For `D=128, bits=3`, the 8 centroids are:
```
[-0.1884, -0.1181, -0.0666, -0.0216, 0.0216, 0.0666, 0.1181, 0.1884]
```

Pre-computed at init via 200 iterations of Lloyd-Max on the analytic PDF. Stored in CUDA constant memory (32 bytes for 3-bit).

### 2.3 ITQ3_S (for weights, Phase 2)

Same rotation principle but uses **Fast Walsh-Hadamard Transform (FWHT)** instead of full matrix multiply:
- FWHT is O(D log D) vs O(D²) for matmul — critical for weight GEMM where rotation is fused
- Interleaves 4-bit nibbles for DP4A hardware compatibility
- Inverse FWHT fused into shared memory during GEMM (2.1% overhead)

---

## 3. System Design

### 3.1 Design Principle: Quantization as a Storage Transform

All quantization in this engine follows the same pattern:

```
                 ┌──────────┐
  BF16 data ───→│ quantize  │───→ packed storage
                 └──────────┘
                 ┌──────────┐
  packed     ───→│dequantize │───→ BF16 data (or fused into compute)
                 └──────────┘
```

TurboQuant adds a **pre-transform** (rotation) before scalar quantization and an **inverse transform** after dequantization. The key insight: for attention, we can rotate Q instead of dequantizing K, avoiding materialization entirely.

### 3.2 New Abstractions

#### `KVFormat` enum extension

```rust
pub enum KVFormat {
    BF16,                                    // existing
    FP8E4M3,                                 // existing
    INT8,                                    // existing
    TurboQuant { key_bits: u8, val_bits: u8 }, // NEW
}
```

#### `QuantFormat` enum extension (for weights)

```rust
pub enum QuantFormat {
    None, Gptq, Awq, Fp8, Int8, Gguf,      // existing
    TurboQuant,                              // NEW (ITQ3_S)
}
```

#### Per-layer rotation state

```rust
/// Pre-computed rotation state for one layer (shared across heads).
/// Generated deterministically from seed + layer_idx at model load.
pub struct TurboQuantState {
    /// Random orthogonal matrix (D×D), f32, device memory.
    /// For KV: full matmul rotation. For weights: Hadamard (implicit, no matrix stored).
    rotation: CudaSlice<f32>,     // D*D elements
    /// Lloyd-Max centroids, f32, in device constant memory.
    /// Index: centroids[2^bits]. Shared across all layers (same D → same distribution).
    centroids: *const f32,        // pointer into constant memory
    /// Lloyd-Max boundaries for searchsorted, f32.
    boundaries: *const f32,       // pointer into constant memory
    bits: u8,
    head_dim: usize,
}
```

### 3.3 Memory Layout

#### KV Cache (per token per head, D=128)

| Format | Data | Scales | Total/token | vs BF16 |
|--------|------|--------|-------------|---------|
| BF16 | 256 B | — | 256 B | 1.0x |
| INT8 | 128 B | 4 B (f32) | 132 B | 1.9x |
| FP8 | 128 B | — | 128 B | 2.0x |
| **TQ 3-bit** | **48 B** | **2 B (f16 norm)** | **50 B** | **5.1x** |
| **TQ 2-bit** | **32 B** | **2 B** | **34 B** | **7.5x** |

Packed layout in pool buffer (NHD, per token):
```
offset 0:                packed_indices [ceil(D * bits / 8) bytes]
offset packed_size:       norm           [2 bytes, f16]
```

Total bytes per token per KV head: `ceil(D * bits / 8) + 2`.

#### Weights (ITQ3_S, Phase 2)

3-bit packed with 4-bit nibble interleaving for DP4A:
- 2 weights per byte (rounded up from 1.5)
- Stored as `[N, K/2]` uint8 (same density as existing W4)
- Plus per-group f16 scales: `[N, K/group_size]`

### 3.4 File Structure

```
infer/
├── src/
│   ├── ops/
│   │   ├── kv_turboquant.rs         # NEW — TQ quantize/dequantize/init FFI wrappers
│   │   └── linear.rs                # MODIFIED — add TQ weight dispatch (Phase 2)
│   ├── model/
│   │   ├── kv_cache.rs              # MODIFIED — TurboQuant format in KVCache
│   │   └── turboquant_state.rs      # NEW — TurboQuantState, codebook precompute
│   ├── paged_kv.rs                  # MODIFIED — TQ pool layout + pointer accessors
│   ├── quant.rs                     # MODIFIED — TurboQuant format detection
│   ├── ffi.rs                       # MODIFIED — TQ FFI declarations
│   └── ops.rs                       # MODIFIED — register kv_turboquant module
├── csrc/cuda/
│   └── turboquant.cu                # NEW — all TQ CUDA kernels
└── build.rs                         # MODIFIED — compile turboquant.cu
```

---

## 4. CUDA Kernel Design

### 4.1 Kernel 1: `turboquant_quantize_kv`

**Purpose**: BF16 KV → TurboQuant packed (called by `commit_layer`).

```
Grid:  (num_kv_heads, batch_size)
Block: (D)  // one thread per coordinate, D=128

Per-thread:
  1. Load bf16 value from KV buffer
  2. Shared memory reduction for ||x||_2
  3. Normalize: x_unit = x / norm
  4. Matrix-vector product: y = sum_j(Pi[d][j] * x_unit[j])
     → Pi loaded via shared memory tiling (D×D is 64KB, use 4 tiles of 32 columns)
  5. Searchsorted against boundaries (binary search, 2^b-1 comparisons = 7 for 3-bit)
  6. Cooperative bitpack: threads collaborate to pack indices into bytes
  7. Store packed bytes + f16 norm to output buffer
```

**Performance target**: <2µs per token per layer (rotation dominates at O(D²)).

### 4.2 Kernel 2: `turboquant_dequantize_kv`

**Purpose**: TurboQuant packed → BF16 KV (called by `prepare_layer`).

```
Grid:  (num_kv_heads, token_count)
Block: (D)

Per-thread:
  1. Cooperative unpack: extract index for coordinate d
  2. Gather centroid value from constant memory
  3. Inverse rotation: x_hat[d] = sum_j(Pi[j][d] * centroid[idx_j])
     → Note: Pi^T^{-1} = Pi for orthogonal matrices
  4. Scale by norm: x_hat[d] *= norm
  5. Store bf16 to output buffer
```

### 4.3 Kernel 3: `turboquant_fused_decode_attention` (optimization, later)

**Purpose**: Compute attention scores directly from TQ-packed keys without dequantizing.

**Key insight**: Instead of dequantizing K, rotate Q:
```
score = q · k_dequant
      = q · (norm * Pi @ codebook[indices])
      = norm * (q @ Pi) · codebook[indices]       // rotate Q once
      = norm * sum_j(q_rot[j] * codebook[idx_j])  // no full dequant needed
```

This avoids materializing D-dimensional dequantized vectors entirely.

```
Grid:  (num_splits, batch_size * num_qo_heads)
Block: (128)  // 4 warps

Input:
  - q_rot: pre-rotated query (D floats, computed outside kernel as GEMV)
  - K packed: (seq_len, packed_bytes) per head
  - K norms: (seq_len,) f16 per head
  - V packed: separate format (group quant or TQ)

Loop over KV tokens in blocks of BLOCK_N:
  1. Load K packed bytes for BLOCK_N tokens
  2. Unpack indices, gather centroids
  3. Dot product: score = norm * sum(q_rot[j] * centroid[idx_j])
  4. Online softmax (flash-attention style: running max + sum-exp)
  5. Load + dequant V, weighted accumulate

Final: cross-warp reduction → output bf16
```

This kernel is the **performance-critical path** for decode and should match or beat the current INT8 fused-dequant attention (which does 1 multiply per element vs TQ's 1 gather + 1 multiply).

---

## 5. Integration Points

### 5.1 KV Cache (`kv_cache.rs`)

```rust
// Existing interface — TurboQuant integrates transparently:

fn prepare_layer(&mut self, ctx, layer_idx) -> (&DeviceVec, &DeviceVec)
  // TQ path: dequantize TQ packed → bf16 working buffer (Kernel 2)
  // Later: skip this entirely when using fused decode attention (Kernel 3)

fn commit_layer(&mut self, ctx, layer_idx, start_pos, count)
  // TQ path: quantize bf16 → TQ packed (Kernel 1)
```

### 5.2 Paged Pool (`paged_kv.rs`)

New pool layout for TurboQuant:
```rust
// Per-layer buffers (TQ format):
k_data: CudaSlice<u8>,     // [max_total_tokens, packed_bytes_per_head * num_kv_heads]
v_data: CudaSlice<u8>,     // same
k_norms: CudaSlice<u16>,   // [max_total_tokens, num_kv_heads] f16
v_norms: CudaSlice<u16>,   // same
```

Pointer accessors: `k_data_ptr()`, `k_norms_ptr()`, etc.

### 5.3 Model Forward (no changes needed)

The `prepare_layer` / `commit_layer` abstraction means model forward code (`prefill.rs`, `decode.rs`, `batch_decode.rs`) needs **zero changes** for Phase 1. The format choice flows through `KVFormat`.

### 5.4 CLI Flag

```
--kv-quant tq3      # TurboQuant 3-bit K + 3-bit V (default)
--kv-quant tq3:2    # TurboQuant 3-bit K + 2-bit V
--kv-quant int8     # existing INT8
--kv-quant fp8      # existing FP8
```

---

## 6. Performance Analysis

### 6.1 Memory Savings (Qwen3-4B, 36 layers, 8 KV heads, D=128)

| Format | Per-token KV | 8K context | 32K context | 128K context |
|--------|-------------|------------|-------------|--------------|
| BF16 | 18.4 KB | 147 MB | 589 MB | 2.3 GB |
| INT8 | 9.5 KB | 76 MB | 304 MB | 1.2 GB |
| FP8 | 9.2 KB | 74 MB | 294 MB | 1.15 GB |
| **TQ3** | **3.6 KB** | **29 MB** | **115 MB** | **0.45 GB** |
| **TQ2** | **2.5 KB** | **20 MB** | **79 MB** | **0.31 GB** |

**At 128K context**: TQ3 saves **1.85 GB** vs BF16, **0.75 GB** vs INT8. This is the difference between fitting on a 24GB GPU or not.

### 6.2 Compute Cost

| Operation | Cycles/token/head | Notes |
|-----------|-------------------|-------|
| Rotation (quantize) | ~D²=16K FMAs | Amortized over tokens; only 1 token/step in decode |
| Searchsorted | 7 comparisons | 3-bit = 7 boundaries, trivial |
| Bitpack | ~D/8 ops | Cooperative warp ops |
| **Total quantize** | **~20µs** | Dominated by rotation matmul |
| Dequantize | ~D²=16K FMAs | Same rotation cost |
| **Fused attention** | **~0** | Rotation moved to Q (1 GEMV); score = gather + FMA |

### 6.3 Decode Latency Impact

The fused attention kernel (§4.3) is key: it avoids the D² dequantization cost entirely by rotating Q once (a single D×D GEMV, ~2µs) and computing scores via centroid gather. Expected decode latency impact: **<5%** vs INT8 fused-dequant.

---

## 7. Implementation Phases

### Phase 1: KV Cache TurboQuant ✅

1. ✅ **Core CUDA kernel**: `turboquant.cu` + `turboquant_fast.cu` (full + Hadamard rotation)
2. ✅ **Rust bindings**: `kv_turboquant.rs` FFI wrappers + `turboquant_state.rs` init
3. ✅ **Integration**: `KVFormat::TurboQuant` in `kv_cache.rs` + `paged_kv.rs`
4. ✅ **CLI**: `--kv-quant tq3` flag
5. ✅ **Tests**: GPU roundtrip + codebook + signs tests

### Phase 2: Weight TurboQuant / ITQ3_S ✅ (2026-04-09)

1. ✅ **Format detection**: `turboquant_config.json` → `QuantMeta::TurboQuant`
2. ✅ **Offline conversion**: `scripts/turboquant_weights.py`
3. ✅ **GPU-packed storage**: `DeviceMatrix.tq_packed/tq_scales/tq_signs/tq_centroids`
4. ✅ **Decode GEMV**: `turboquant_weight_gemv.cu` — fused unpack → centroid gather → iFWHT → sign flip → dot
5. ✅ **Prefill GEMM**: bulk dequant kernel + cuBLAS GEMM (Marlin pattern)
6. ✅ **Dispatch**: `ops/linear.rs` TQ path before INT quant dispatch
7. **Future**: Fused Tensor Core MMA kernel for higher prefill throughput

### Phase 3: Fused Decode Attention ✅

1. ✅ **Fused kernel**: `tq_decode_attention_cuda` — score from packed K, no dequant
2. ✅ **Q rotation**: `tq_rotate_query_cuda` — sign flip + FWHT per-layer
3. ✅ **Wired into all 3 models**: Qwen3, Qwen3.5, GLM4

---

## 8. Engineering Findings from Reference Implementations

Critical lessons from scos-lab/turboquant and tonbistudio/turboquant-pytorch:

1. **MSE-only often beats Prod (QJL) for keys** — QJL variance hurts softmax more than MSE bias. Start with MSE-only.

2. **K and V need different bit budgets** — Qwen models have K/V norm ratios of 100-1000x (RoPE). Optimal: high bits for K (3-4), lower for V (2-3).

3. **Buffer recent tokens unquantized** — Keep last 128 tokens in BF16, only flush to TQ when buffer fills. Most critical context stays lossless.

4. **Initial layers need more bits** — First 4 layers carry more information. Use `key_bits + 1` for initial layers.

5. **Value quantization can use simpler group quant** — TurboQuant rotation is optional for V; simple group quantization with min-max scaling is sufficient since V only needs reconstruction, not inner products.

---

## 9. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Rotation O(D²) too slow for quantize | Amortized: only 1 token/step in decode; prefill batches rotation as GEMM |
| Quality regression on small models | Configurable bits + BF16 recent-token buffer + per-layer bit allocation |
| Paged pool complexity | Reuse existing `TokenKVPool` layout patterns; TQ is just a different bytes_per_element |
| Fused attention kernel correctness | Verify against dequantize+FlashInfer reference path; share test infrastructure with INT8 |

---

## 10. Success Criteria

- [ ] TQ3 KV cache: **5x memory reduction** vs BF16, quality-neutral on E2E baselines
- [ ] Decode latency: **<10% regression** vs INT8 (Phase 1), **<5%** (Phase 3 fused)
- [ ] Max context: **4x increase** in max concurrent tokens at same GPU memory budget
- [ ] Clean integration: no changes to model forward code (prepare_layer/commit_layer abstraction)
