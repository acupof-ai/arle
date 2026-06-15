# Qwen3.6 CUDA Native Quant Plan

Date: 2026-06-15

Status: implementation plan, not shipped.

Goal: run `Qwen3.6-35B-A3B` FP8 and NVFP4 checkpoints on CUDA with resident
quantized weights. The CUDA forward path must consume the checkpoint's
quantized representation directly. It must not load-time materialize full model
weights to dense BF16.

## Decision

The previous load-time BF16 materialization idea is rejected. It does not solve
the two problems that matter for this model: HBM residency and weight bandwidth.

The accepted first version is:

- Qwen3.6 quantized checkpoint tensors stay resident as quantized device
  buffers.
- Qwen35 CUDA linear and MoE paths dispatch on a Qwen-specific resident
  `WeightFormat`.
- Kernels dequantize at tile/group granularity inside the operator.
- Full-weight dense BF16 expansion is forbidden.

A100 note: A100 has no Hopper/Blackwell hardware FP8/FP4 tensor-core path. In
this plan, "native" means resident checkpoint-native quant buffers plus ARLE CUDA
kernels that consume them directly. It does not mean hardware-native FP8/FP4 MMA.

## Non-Goals

- Do not reuse `Dsv4Fp8BlockScaled` or `Dsv4Fp4BlockScaled` for Qwen3.6.
  DSv4 uses an E8M0 scale ABI; Qwen FP8/NVFP4 does not.
- Do not add Qwen quant types above `infer-cuda` / `cuda-kernels`.
- Do not make a broad model-free quant promise. The codec can be generic, but
  model tensor-role mapping and operator ABI stay explicit.
- Do not claim MMLU, SWE Pro, or performance from source review or Mac
  typecheck. A100 serve/eval is the gate.

## Current Blockers

1. `crates/infer-cuda/src/loader.rs` has an incomplete WIP that does not
   compile. It references undefined helpers such as `QWEN35_FP8_BLOCK`,
   `qwen35_weight_prefix`, `tensor_to_f32_vec`, and
   `materialize_fp8_blocks_to_bf16`.
2. The WIP loader functions are not reached by the real Qwen35 load path.
   `crates/infer-cuda/src/qwen35.rs` still calls BF16-only `load_matrix`,
   `load_matrix_sharded`, `load_qkv_head_sharded`, and the BF16 MoE loader.
3. `load_linear_qkv_sharded` and `load_conv1d_sharded` in `qwen35.rs` still
   read raw BF16 bytes directly. The qkv weight helper must become
   quant-aware; conv/norm/bias should remain strict BF16/F32.
4. Qwen3.6 MoE stacked experts are the critical path. Handling only dense
   2D matrices is insufficient because production experts use stacked/fused
   tensors such as `experts.gate_up_proj` and `experts.down_proj`.
5. Existing unrelated dirty files must stay untouched. Stage by explicit path
   only.

## Format Contracts

### Official FP8

Storage:

- weight: `F8_E4M3`
- sibling scale: `weight_scale_inv`
- scale shape follows 128x128 weight blocks in observed Qwen3.6 checkpoints

Resident ABI:

- upload FP8 weight bytes as-is
- upload decoded or raw scale buffer according to the kernel ABI chosen in
  `cuda-kernels`
- kernel computes `value = fp8(weight) * scale_inv_block`

Open item before coding: confirm whether the on-disk field name is always an
inverse scale for every official Qwen3.6 FP8 tensor class used by CUDA.

### ModelOpt FP8

Storage:

- weight: `F8_E4M3`
- sibling scales: `weight_scale`, `input_scale`
- scales may be per logical shard, not necessarily one global scalar

Resident ABI:

- preserve the per-logical-shard scale semantics from the checkpoint loader
- do not collapse scales to `max()` unless implementing a specific native-kernel
  remap that proves equivalence for that tensor role
- `input_scale` is activation metadata; first version may store it but should
  fail closed if a kernel would need dynamic activation quantization that is not
  implemented

### ModelOpt NVFP4

Storage:

- weight: `U8`, two FP4 E2M1 values per byte
- weight scale: `F8_E4M3`, one scale per group of 16 input elements
- second scale: `weight_scale_2`, F32 scalar or per logical shard
- `input_scale`: activation scale metadata

Resident ABI:

- upload packed U8 weight as-is
- upload FP8 E4M3 scale bytes or a decoded scale buffer, whichever the kernel
  consumes
- upload `weight_scale_2`
- kernel computes from packed FP4 nibbles and group scale without expanding the
  full matrix
- A100 path must not call SGLang/ModelOpt SM100-only native NVFP4 kernels

## File-Level Implementation

### P0: Restore a compiling tree

Files:

- `crates/infer-cuda/src/loader.rs`

Work:

- Remove the incomplete BF16-materialization WIP.
- Keep existing BF16 and DSv4 loaders unchanged.
- Add no new behavior in this step.

Exit gate:

```bash
CUDARC_CUDA_VERSION=12060 cargo check -p infer-api --release \
  --no-default-features --features cuda,no-cuda --lib
```

### P1: Checkpoint codec layer

Files:

- `crates/infer-cuda/src/lib.rs`
- `crates/infer-cuda/src/quant_format.rs`

Types:

```rust
pub(crate) enum Qwen35QuantFormat {
    DenseBf16,
    DenseF32,
    OfficialFp8BlockScaleInv { block_m: usize, block_k: usize },
    ModelOptFp8,
    ModelOptNvFp4 { group_size: usize },
}

pub(crate) struct Qwen35QuantTensorView {
    pub name: String,
    pub logical_shape: Vec<usize>,
    pub storage_dtype: safetensors::tensor::Dtype,
    pub format: Qwen35QuantFormat,
    pub weight: OwnedTensor,
    pub scales: Vec<OwnedTensor>,
}
```

Functions:

- `detect_qwen35_quant_format(name, tensor, siblings, manifest)`.
- `decode_f8_e4m3fn(byte) -> f32`.
- `decode_fp4_e2m1(nibble) -> f32`.
- `read_qwen35_quant_manifest(model_path)`.
- `validate_qwen35_scale_shapes(view)`.

Rules:

- Detection must use sibling tensor names and quant config, not dtype alone.
- The codec returns views and metadata. It does not allocate a dense BF16
  matrix.

### P2: Resident weight ABI

Files:

- `crates/cuda-kernels/src/tensor.rs`

Add `WeightFormat` variants:

```rust
QwenFp8BlockScaled,
QwenModelOptFp8,
QwenNvFp4,
```

Extend `DeviceMatrix` with Qwen-specific optional fields:

- `qwen_qweight: Option<CudaSlice<u8>>`
- `qwen_qscale: Option<CudaSlice<u8>>`
- `qwen_scale_f32: Option<CudaSlice<f32>>`
- `qwen_scale2_f32: Option<CudaSlice<f32>>`
- `qwen_block_m`, `qwen_block_k`, `qwen_group_size`

Constructors:

- `DeviceMatrix::from_qwen_fp8_block_scaled(...)`
- `DeviceMatrix::from_qwen_modelopt_fp8(...)`
- `DeviceMatrix::from_qwen_nvfp4(...)`

Invariants:

- `data` remains a dummy 1-element BF16 buffer for quant resident matrices.
- `rows` and `cols` are logical matrix dimensions.
- Shape validation lives next to the constructors.
- DSv4 constructors and DeepGEMM caches are unchanged.

### P3: Loader native path

Files:

- `crates/infer-cuda/src/loader.rs`

New methods:

- `load_qwen35_matrix_native(ctx, name) -> DeviceMatrix`
- `load_qwen35_matrix_sharded_native(ctx, name, kind, tp) -> DeviceMatrix`
- `load_qwen35_qkv_head_sharded_native(ctx, name, local_heads, head_dim, tp)`
- `load_qwen35_stacked_expert_native(ctx, name, expert, row_offset, rows)`
- `load_qwen35_moe_layer_experts_native(...) -> MoeLayerWeights`

Sharding rules:

- BF16/F32 dense tensors may reuse existing byte-slice helpers.
- FP8 row/column sharding must slice weight and scale tensors together.
- NVFP4 row sharding slices packed columns and scale groups together.
- Column sharding for NVFP4 must be group-aligned. Otherwise fail closed.
- Stacked expert slicing must work on the packed tensor dimensions, not a
  materialized dense shape.

### P4: GEMV/GEMM kernels

Files:

- `crates/cuda-kernels/csrc/gemm/qwen35_quant_gemm.cu`
- `crates/cuda-kernels/src/ffi/gemm.rs`

FFI:

- `qwen35_fp8_gemv_cuda`
- `qwen35_modelopt_fp8_gemv_cuda`
- `qwen35_nvfp4_gemv_cuda`
- `qwen35_quant_gemm_batch_cuda`

Priority:

1. Decode GEMV, because Qwen35 decode is the immediate eval path.
2. Small prefill GEMM with tiled dequant.
3. Batched/large prefill optimization.

Kernel rule:

- No full-matrix BF16 side buffer.
- Tile/register/shared-memory dequant is allowed.
- Accumulation may be FP32 with BF16 output.

### P5: Operator dispatch

Files:

- `crates/infer-cuda/src/ops.rs`

Update:

- `gemv` dispatches by `DeviceMatrix::weight_format()`.
- `gemm_batch` dispatches by `DeviceMatrix::weight_format()`.
- Dense BF16 keeps the current path.
- DSv4 quant keeps the DSv4-specific path.
- Qwen quant variants call the new Qwen kernels.

Fail closed:

- Unsupported format/shape returns an error naming the exact matrix and format.
- No implicit dense fallback.

### P6: Qwen35 model load wiring

Files:

- `crates/infer-cuda/src/qwen35.rs`

Replace all matrix loads in `from_safetensors_with_tp`:

- `embed_tokens`
- `lm_head`
- full attention `q_proj`, `k_proj`, `v_proj`, `o_proj`
- linear attention `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`,
  `out_proj`
- dense MLP `gate_proj`, `up_proj`, `down_proj`
- MoE routed experts, router gate, shared expert

Keep strict BF16/F32 for:

- norms
- `dt_bias`
- `A_log`
- conv1d weights unless a checkpoint proves quantized conv exists and a kernel
  contract is added

OPD and LoRA:

- `remerge_student_lora` should fail closed for quant resident matrices in the
  first version.
- weight offload/reload snapshots must either support quant buffers or fail
  closed before offload. Do not silently clone dummy `data`.

### P7: Native MoE

Files:

- `crates/infer-cuda/src/moe.rs`
- `crates/infer-cuda/src/loader.rs`
- `crates/cuda-kernels/csrc/gemm/qwen35_quant_gemm.cu`

Work:

- Extend `MoeLayerWeights` to carry quant expert matrices.
- Route expert GEMV/GEMM through Qwen quant kernels.
- Disable BF16 DeepGEMM grouped cache when expert matrices are quant resident.
- Shared expert must use the same quant dispatch as routed experts.

Exit condition:

- Qwen3.6 MoE layers do not allocate dense BF16 expert weights.
- Decode path runs routed and shared expert quant kernels.

## Tests

Pure tests:

- `decode_f8_e4m3fn_known_values`
- `decode_fp4_e2m1_known_values`
- `official_fp8_block_scale_shape_validation`
- `modelopt_fp8_per_logical_shard_scale_is_preserved`
- `nvfp4_group16_shape_validation`
- `qwen_quant_rejects_dsv4_e8m0_scale_abi`

Loader tests:

- fake safetensors with FP8 linear weight reaches `QwenFp8BlockScaled`
- fake safetensors with NVFP4 expert reaches `QwenNvFp4`
- TP shard rejects unaligned NVFP4 column shards

Kernel tests on A100:

- tiny FP8 GEMV vs CPU reference
- tiny NVFP4 GEMV vs CPU reference
- Qwen35 quant GEMM batch vs CPU reference for small matrices
- MoE top-k one-token quant expert path vs CPU reference

## Verification Gates

Local Mac:

```bash
git diff --check
cargo fmt --all -- --check
CUDARC_CUDA_VERSION=12060 cargo check -p infer-api --release \
  --no-default-features --features cuda,no-cuda --lib
cargo test -p infer-cuda --release --no-default-features --features no-cuda --lib
```

A100:

```bash
CUDA_HOME=/usr/local/cuda cargo build --release --features cuda
target/release/arle serve --backend cuda --model-path "$MODEL" --port 8123 \
  --num-slots 1 --kv-cache-dtype bf16
curl -sf http://127.0.0.1:8123/v1/models
```

Smoke:

```bash
python scripts/arle_capability_eval.py --backend arle \
  --base-url http://127.0.0.1:8123 \
  --model-id "$MODEL_ID" \
  --tasks mmlu \
  --n-samples 50 \
  --seed 0 \
  --output bench-output/qwen36-a100-mmlu-smoke
```

Bench:

```bash
scripts/bench_guidellm.sh qwen36-cuda-native-quant-a100 \
  --target http://127.0.0.1:8123 \
  --model "$MODEL_ID" \
  --processor "$MODEL"
```

## Hard Acceptance

- `dense_materialized_weight_bytes=0` in loader/runtime logs.
- `nvidia-smi` memory is incompatible with full dense BF16 weight residency.
- Serve log prints resident bytes by format: FP8, NVFP4, BF16.
- `strings target/release/arle | grep qwen35_quant` finds linked kernels.
- FP8 checkpoint passes serve + curl smoke + MMLU 50 plumbing.
- NVFP4 checkpoint passes serve + curl smoke + MMLU 50 plumbing.
- No performance claim until `guidellm` completes.
- No capability claim from MMLU 50 or SWE Pro limit-3 smoke.

## Documentation And Commit Policy

Runtime changes under `crates/infer-cuda` or `crates/cuda-kernels` require a
wins/errors entry. If A100 is not available locally, write a `pending-remote`
wins stub and name the exact remote gate.

Commit in small tranches:

1. P0 compile restoration.
2. P1/P2 codec + resident ABI tests.
3. P3 loader native path.
4. P4/P5 kernels and op dispatch.
5. P6/P7 Qwen35 and MoE wiring.
6. A100 eval evidence entry.

Stage by explicit path only. Do not stage unrelated dirty files.

## Cross-Review Notes

Architecture review:

- Keep Qwen quant support in `infer-cuda` and `cuda-kernels`.
- Do not touch `infer-seam`, `infer-core`, or `infer-api` in the first tranche.
- Do not reuse DSv4 resident quant formats.

Quant format review:

- Dtype-only detection is unsafe.
- ModelOpt FP8 scale semantics must preserve logical shard scale behavior.
- NVFP4 BF16-style fallback is rejected; native resident path must consume packed
  U8 weights directly.

Verification review:

- Loader reachability must be proved by runtime logs or tests.
- A100 serve/eval is required before declaring support.
- MMLU 50 and SWE Pro limit-3 are plumbing checks only.

## Self-Review

What is SOLID:

- Existing code confirms Qwen35 currently loads matrices through BF16-only
  loader methods.
- Existing `cuda-kernels::WeightFormat` confirms DSv4 quant formats are
  DSv4-specific resident ABIs.
- Existing eval docs require ARLE to produce the candidate artifact and graders
  to score only.

What remains hypothesis until implementation:

- A100 quant GEMV/GEMM throughput.
- Whether all target Qwen3.6 FP8 and NVFP4 tensors share the probed scale shapes.
- Exact VRAM fit with resident quant weights plus Qwen35 KV/recurrent state.

Deferred uncertainty:

- Native Hopper/Blackwell FP8/FP4 MMA path.
- Full multi-concurrency performance.
- Official MMLU/SWE Pro capability score.
