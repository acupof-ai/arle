> **Archived 2026-04-15** — design spec **was not implemented as written**.
> The `LinearWeight` enum + dispatch architecture proposed in this doc
> was never built; `grep -rn 'enum LinearWeight' infer/` returns only
> this archived doc itself. The actual production quantization path
> landed via **TurboQuant** (Phases 1–3) instead — see
> [`../plans/turboquant-integration.md`](../plans/turboquant-integration.md)
> for the shipped design, and
> [`../experience/wins/2026-04-08-weight-quantization.md`](../experience/wins/2026-04-08-weight-quantization.md)
> /
> [`../experience/wins/2026-04-09-marlin-prefill.md`](../experience/wins/2026-04-09-marlin-prefill.md)
> /
> [`../experience/wins/2026-04-09-w4-int8kv-combo.md`](../experience/wins/2026-04-09-w4-int8kv-combo.md)
> for the shipped numbers. The `Status: Complete` line below was
> aspirational; line 362's "Waiting for design review approval. On
> approval, start P1 (LinearWeight enum)" reflects the true terminal
> state. Preserved for the rationale + four-risk analysis (CUDA Graph
> compat, merged QKV, desc_act, prefill perf), which is still useful
> background reading for any future revisit of typed-weight dispatch.

---

# Quantization Architecture Design

**Status**: ~~Complete~~ **Archived — design spec, never implemented (see banner above)**
**Date**: 2026-04-02 (design) / ~~2026-04-09 (implemented)~~ implementation never started
**Goal**: Support GPTQ/AWQ INT4 quantized models, unlocking 4x weight compression with minimal throughput regression

---

## TL;DR

Introduce `LinearWeight` enum abstraction to decouple model forward from quantization format. INT4 GEMV kernel for decode (CUDA Graph safe), dequant+cuBLAS for prefill MVP. Four risks analyzed: CUDA Graph compat (safe), merged QKV (use separate projections), desc_act (defer to v2), prefill perf (acceptable with dequant+cuBLAS).

---

## 1. Current State

| Layer | Status | Notes |
|-------|--------|-------|
| Format detection (`quant.rs`) | Done | GPTQ/AWQ/FP8/INT8/GGUF metadata parsing, 582 lines, zero consumers |
| Quantized tensor type | Missing | Only `DeviceMatrix` (bf16) exists |
| Quantized weight loading | Missing | `load_tensor_2d` assumes bf16 |
| Quantized GEMV/GEMM kernel | Missing | `gemv.cu` only has bf16 GEMV + cuBLAS GEMM |
| Ops dispatch | Missing | `gemv()` / `gemm_into()` take `DeviceMatrix` directly |
| Model forward integration | Missing | `Attention`/`MLP` structs hold `DeviceMatrix` |

`quant.rs` is imported in `lib.rs` but **not referenced from any other module** — a detection island waiting for the execution pipeline.

---

## 2. Architecture Design

### 2.1 LinearWeight Enum (Core Abstraction)

```rust
// New: infer/src/quant_tensor.rs (or extend tensor.rs)

/// A linear projection weight — dense or quantized.
/// Model code operates on this enum; dispatch happens in ops layer.
pub enum LinearWeight {
    /// Dense BF16 weight [out_dim, in_dim]
    Dense(DeviceMatrix),

    /// GPTQ/AWQ packed INT4
    Int4 {
        qweight: CudaSlice<u32>,   // packed [out_dim, in_dim/8] — 8 INT4 per u32
        scales: CudaSlice<bf16>,   // [num_groups, out_dim]
        qzeros: CudaSlice<u32>,    // packed [num_groups, out_dim/8]
        out_dim: usize,
        in_dim: usize,
        group_size: usize,
    },

    /// FP8 E4M3 (Hopper+, future)
    Fp8 {
        weight: CudaSlice<u8>,
        scale: CudaSlice<f32>,
        out_dim: usize,
        in_dim: usize,
    },
}

impl LinearWeight {
    pub fn out_dim(&self) -> usize { ... }
    pub fn in_dim(&self) -> usize { ... }
    pub fn is_quantized(&self) -> bool { !matches!(self, Self::Dense(_)) }
}
```

**Why enum, not trait**: Variants are exhaustively known at compile time. No dyn dispatch overhead. Match arms in ops make the dispatch explicit and auditable.

### 2.2 Ops Dispatch

```rust
// ops/linear.rs — new public API

/// Decode path: y = weight @ x (single token)
pub fn linear_forward(
    ctx: &DeviceContext, w: &LinearWeight, x: &DeviceVec, y: &mut DeviceVec,
) -> Result<()> {
    match w {
        LinearWeight::Dense(m) => gemv(ctx, m, x, y),
        LinearWeight::Int4 { qweight, scales, qzeros, out_dim, in_dim, group_size } =>
            gemv_int4(ctx, qweight, scales, qzeros, *out_dim, *in_dim, *group_size, x, y),
        LinearWeight::Fp8 { .. } => todo!("FP8 GEMV"),
    }
}

/// Prefill path: Y = weight @ X (batched)
pub fn linear_forward_batch(
    ctx: &DeviceContext, w: &LinearWeight, x: &HiddenStates, out: &mut HiddenStates,
) {
    match w {
        LinearWeight::Dense(m) => gemm_into(ctx, m, x, out),
        LinearWeight::Int4 { .. } => gemm_int4_dequant(ctx, w, x, out),  // dequant + cuBLAS
        LinearWeight::Fp8 { .. } => todo!("FP8 GEMM"),
    }
}
```

Model forward code changes:
- `gemv(ctx, &layer.attention.q_proj, ...)` → `linear_forward(ctx, &layer.attention.q_proj, ...)`
- `gemm_into(ctx, &layer.attention.qkv_proj, ...)` → `linear_forward_batch(ctx, &layer.attention.qkv_proj, ...)`

### 2.3 Fused MLP Strategy

Current `fused_mlp_into` is a single CUDA kernel (gate+up+SiLU+down all fused). INT4 weights cannot use this kernel.

**Design: dispatch at ops level, decompose for quantized path**

```rust
pub fn fused_mlp_dispatch(
    ctx: &DeviceContext, x: &DeviceVec,
    gate: &LinearWeight, up: &LinearWeight, down: &LinearWeight,
    act: &mut DeviceVec, up_buf: &mut DeviceVec, out: &mut DeviceVec,
) -> Result<()> {
    match (gate, up, down) {
        (Dense(g), Dense(u), Dense(d)) => {
            fused_mlp_into(ctx, x, g, u, d, act, out)  // existing fused kernel, zero regression
        }
        _ => {
            linear_forward(ctx, gate, x, act)?;     // gate = gate_proj @ x
            linear_forward(ctx, up, x, up_buf)?;     // up = up_proj @ x
            silu_mul(ctx, act, up_buf)?;              // act = SiLU(gate) * up
            linear_forward(ctx, down, act, out)?;     // out = down_proj @ act
            Ok(())
        }
    }
}
```

Dense path: zero regression (same fused kernel). Quantized path: 4 kernel launches instead of 1, but each INT4 GEMV is ~2x faster (half bandwidth), net neutral or faster.

### 2.4 Weight Loading (QuantMeta-driven)

```rust
// weight_loader.rs — new

pub fn load_linear_weight(
    ctx: &DeviceContext, shards: &[SafeTensors], weight_map: &HashMap<String, usize>,
    name: &str, quant: &QuantMeta,
) -> Result<LinearWeight> {
    match quant {
        QuantMeta::None => {
            Ok(LinearWeight::Dense(load_tensor_2d(ctx, shards, weight_map, name)?))
        }
        QuantMeta::Gptq(cfg) => {
            let qweight = load_raw_u32(ctx, shards, weight_map, &format!("{name}.qweight"))?;
            let scales  = load_raw_bf16(ctx, shards, weight_map, &format!("{name}.scales"))?;
            let qzeros  = load_raw_u32(ctx, shards, weight_map, &format!("{name}.qzeros"))?;
            // Derive dims from tensor shapes
            Ok(LinearWeight::Int4 { qweight, scales, qzeros, out_dim, in_dim, group_size: cfg.group_size as usize })
        }
        QuantMeta::Awq(cfg) => {
            // AWQ uses same INT4 packing, different naming convention
            ...
        }
        _ => bail!("Quantization format {} not yet supported for GPU inference", quant.format()),
    }
}
```

GPTQ safetensors naming: `model.layers.{i}.self_attn.q_proj.qweight` / `.scales` / `.qzeros`.

### 2.5 INT4 GEMV Kernel (Decode)

```cuda
// csrc/cuda/gemv_int4.cu

// Each thread block processes ROWS_PER_BLOCK output rows
// Threads within a block collaborate on the dot product along K dimension
// Dequantization is inline: no intermediate buffer

__global__ void gemv_int4_kernel(
    const uint32_t* __restrict__ qweight,   // [out_dim, in_dim/8] packed
    const __nv_bfloat16* __restrict__ scales, // [num_groups, out_dim]
    const uint32_t* __restrict__ qzeros,    // [num_groups, out_dim/8] packed
    const __nv_bfloat16* __restrict__ x,    // [in_dim]
    __nv_bfloat16* __restrict__ y,          // [out_dim]
    int out_dim, int in_dim, int group_size
) {
    // 1. Each thread loads u32 from qweight → extract 8 INT4 values
    // 2. Load scale + zero for the current group
    // 3. Dequant: val_fp = (int4_val - zero) * scale
    // 4. FMA with corresponding x elements
    // 5. Warp shuffle + shared memory reduction → write y[row]
}
```

**CUDA Graph safe**: no alloc, no host sync, fixed pointers. Follows same pattern as existing `gemv_handwritten_kernel`.

### 2.6 INT4 GEMM for Prefill (Dequant + cuBLAS)

MVP approach: dequantize INT4 → BF16 workspace, then call existing `gemm_cuda()`.

```rust
fn gemm_int4_dequant(ctx: &DeviceContext, w: &LinearWeight, x: &HiddenStates, out: &mut HiddenStates) {
    // 1. Dequant kernel: INT4 → BF16 into pre-allocated workspace
    //    Workspace: [out_dim, in_dim] bf16, allocated once, reused across layers
    int4_dequant_cuda(qweight, scales, qzeros, workspace, out_dim, in_dim, group_size, stream);

    // 2. Standard cuBLAS GEMM (uses prefill handle with workspace)
    gemm_cuda(workspace, x_ptr, out_ptr, M, N, K, stream);
}
```

Workspace sizing for Qwen3-8B: `4096 × 14336 × 2 = 112 MB` (largest linear layer). Allocated once, reused across all layers.

---

## 3. Risk Analysis

### Risk 1: CUDA Graph Compatibility — LOW RISK

**Finding: INT4 GEMV is fully CUDA Graph safe.**

Current CUDA Graph mechanism (`cuda_graph.rs:10-68`):
- Captures all kernel launches inside `run_or_capture()` closure
- Constraint: "pure GPU kernel sequence — no CPU-GPU sync, no allocation"
- All decode buffers pre-allocated in `DecodeBuffers::new()`

INT4 GEMV kernel satisfies all constraints:
- No dynamic allocation (weights/scales/zeros pre-loaded)
- No host sync (pure kernel launch)
- Fixed pointers (buffers don't move after allocation)
- No workspace (dequant inline in dot product loop)

Graph-safe cuBLAS pattern already established: `gemm_graphsafe_cuda` uses workspace-free handle (`g_cublas_handle`). INT4 GEMV doesn't even need cuBLAS — it's a standalone kernel.

**Caveat**: Quantized path uses separate Q/K/V projections (3 GEMV) instead of merged QKV (1 GEMM). This means the captured graph for quantized models has ~3 more kernel launches per layer (96 total for 32 layers). The overhead is ~48us per decode step vs ~2ms total — negligible.

### Risk 2: Merged QKV / GateUp Projections — LOW RISK

**Finding: Cannot merge INT4 packed weights. Use separate projections — zero overhead under CUDA Graph.**

Current state:
- **Prefill**: Already uses 3 separate GEMM calls for Q/K/V (`prefill.rs:200-217`). No impact.
- **Single decode**: Already uses 3 separate GEMV calls (`decode.rs:99-116`). No impact.
- **Batched decode**: Uses merged `qkv_proj` + `split_qkv_batch()` (`batch_decode.rs:372-385`). **Needs change.**
- **MLP**: Gate+up merged in batched decode via `silu_mul_fused_batch_into()`. **Needs change.**

Why merging INT4 is infeasible:
- GPTQ/AWQ safetensors store separate `q_proj.qweight`, `k_proj.qweight`, `v_proj.qweight`
- Each has independent per-group scales and zeros
- Concatenating packed INT4 data would misalign group boundaries unless `group_size` divides `out_dim` exactly AND scale layouts are contiguous — not guaranteed

**Kernel launch count (32-layer model):**

| Path | BF16 (merged) | INT4 (separate) | Delta |
|------|---------------|-----------------|-------|
| Attention (QKV+O) | 2 launches | 4 launches | +2 |
| MLP (gate_up+down) | 2 launches | 4 launches | +2 |
| **Per layer** | **4** | **8** | **+4** |
| **32 layers** | **128** | **256** | **+128** |

**Why this is a non-issue**: Quantization format is known at model load time. CUDA Graph warmup captures the full INT4 kernel sequence (256 nodes) once. Replay is a single `cudaGraphLaunch` — graph node count has negligible effect on replay latency (~5us delta between 128 vs 256 nodes). No CPU-side per-step launch overhead.

**Implementation**: Batched decode for quantized models skips merged weight creation in `weights.rs`. The `LinearWeight` enum naturally handles this — `concat_rows` is only called for `Dense` variants.

### Risk 3: desc_act (GPTQ Activation Order) — LOW RISK (Defer)

**Finding: ~20-30% of GPTQ models use `desc_act=true`. Defer to v2.**

What desc_act does:
- Reorders input columns by decreasing Hessian diagonal importance before quantization
- At inference: requires `g_idx` tensor mapping each input channel to its quantization group
- Without desc_act: group assignment is trivial (`group_i = channel / group_size`)
- With desc_act: arbitrary mapping stored in `g_idx`, adds memory bandwidth + kernel complexity

Prevalence:
- TheBloke models (dominant producer): default `desc_act=false`, offer `_actorder_True` variants
- ~20-30% of popular GPTQ models on HuggingFace offer `desc_act=true` variants
- Most deployment guides recommend `desc_act=false` due to stability concerns (NaN on outliers)

Industry approach:
| Engine | desc_act Support |
|--------|-----------------|
| vLLM | Full (via g_idx lookup in Marlin kernel) |
| ExLlama | Partial (weight reordering, NaN risk) |
| GPTQModel | Full (active fork of AutoGPTQ) |

**Recommendation**: v1 detects `desc_act=true` and errors with a clear message. Config parsing already captures the field (`quant.rs:73`). Support `g_idx` lookup in v2 when there's user demand.

If forced to support early: **repack weights at load time** (convert `desc_act=true` → `false` by applying permutation to packed weights + rebuilding trivial group assignments). CPU-only, ~5s per model, no kernel changes.

### Risk 4: Prefill Performance (Dequant + cuBLAS) — LOW RISK

**Finding: 10-20% prefill slowdown is acceptable. Decode is the bottleneck.**

Bandwidth analysis for Qwen3-8B layer (4096 × 4096):
| Approach | Weight Read | Dequant Write | cuBLAS Read | Total Bandwidth |
|----------|------------|---------------|-------------|----------------|
| BF16 baseline | 32 MB | — | — | 32 MB |
| Dequant + cuBLAS | 8 MB (INT4) | 32 MB | 32 MB | 72 MB |
| Marlin (future) | 8 MB (INT4) | — | — | 8 MB |

Latency impact (2048-token prefill, 32 layers):
- BF16 baseline: ~150ms total
- Dequant + cuBLAS: ~165ms (+10%)
- Marlin: ~145ms (-3%)

**Why 10% is fine:**
1. Prefill happens once per request, amortized over generation
2. Decode (ITL) is the throughput bottleneck at concurrency ≥ 2 (confirmed by profiling: `experience/wins/2026-03-31-throughput-profiling.md`)
3. INT4 models exist to save memory (run larger models on fewer GPUs), not to speed up prefill

Workspace sizing:
- Largest linear layer (MLP gate/up): `14336 × 4096 × 2 = 112 MB`
- Allocated once, reused across all layers in a single prefill step
- Current cuBLAS prefill handle already has 32 MB workspace — total 144 MB

**Upgrade path**: Replace dequant+cuBLAS with Marlin kernel when profiling shows prefill is >5% of total latency. Marlin is MIT-licensed, integrated in vLLM, supports `group_size=128` and `-1`.

---

## 4. Implementation Phases

| Phase | Scope | Files Changed | GPU? | Est. |
|-------|-------|--------------|------|------|
| **P1** | `LinearWeight` enum + helpers | `quant_tensor.rs` (new), `tensor.rs` | No | 1d |
| **P2** | Ops dispatch (`linear_forward`, `linear_forward_batch`, `fused_mlp_dispatch`) | `ops/linear.rs` | No (stubs for INT4) | 1d |
| **P3** | Quantized weight loading | `weight_loader.rs` | No | 1d |
| **P4** | Qwen3 model integration (weights → `LinearWeight`, forward → dispatch) | `qwen3/weights.rs`, `decode.rs`, `prefill.rs`, `batch_decode.rs` | No | 2d |
| **P5** | INT4 GEMV kernel + FFI + E2E test | `csrc/gemv_int4.cu`, `ffi.rs`, `build.rs` | Yes | 3d |
| **P6** | INT4 dequant kernel + prefill GEMM | `csrc/int4_dequant.cu`, `ops/linear.rs` | Yes | 2d |
| **P7** | CUDA Graph warmup for INT4 decode | `cuda_graph.rs`, `decode_buffers.rs` | Yes | 1d |
| **P8** | AWQ support (weight naming adapter) | `weight_loader.rs` | No | 1d |

**Critical path**: P1 → P2 → P3 → P4 → P5 → P7 (decode E2E). P6 can parallel with P7.

---

## 5. Key Design Invariants

1. **Model code never matches on `LinearWeight`** — only `ops/linear.rs` does dispatch
2. **Dense path has zero regression** — fused kernels + merged weights untouched
3. **All INT4 buffers pre-allocated before CUDA Graph capture** — graph safety by construction
4. **QuantMeta detected once at model load** — threaded through weight loading, not stored in model struct
5. **desc_act=true → early error** — not silent degradation, clear user message
6. **Dequant workspace allocated once** — reused across layers, freed after model unload

---

## 6. Files Reference

| File | Role | Lines of Interest |
|------|------|------------------|
| `infer/src/quant.rs` | Format detection (done) | 280: `detect_quant_format`, 287: `load_quant_meta` |
| `infer/src/tensor.rs` | Tensor types | 59: `DeviceVec`, 215: `DeviceMatrix` |
| `infer/src/ops/linear.rs` | Linear ops | 9: `gemv`, 93: `gemm`, 42: `fused_mlp_into` |
| `infer/src/ffi.rs` | CUDA FFI bindings | 95: `gemv_cuda`, 104: `gemm_cuda` |
| `infer/csrc/cuda/gemv.cu` | GEMV + cuBLAS wrappers | 16: kernel, 139: workspace comments |
| `infer/src/weight_loader.rs` | Safetensors loading | 93: `load_tensor_1d`, 103: `load_tensor_2d` |
| `infer/src/model/qwen3/weights.rs` | Qwen3 weight structs | 26: `Attention`, 39: `MLP`, 140: QKV merge |
| `infer/src/model/qwen3/decode.rs` | Decode forward | 99-116: GEMV calls |
| `infer/src/model/qwen3/prefill.rs` | Prefill forward | 200-217: GEMM calls |
| `infer/src/model/cuda_graph.rs` | Graph capture | 37: constraint comment |

---

## Next Action

Waiting for design review approval. On approval, start P1 (LinearWeight enum).
