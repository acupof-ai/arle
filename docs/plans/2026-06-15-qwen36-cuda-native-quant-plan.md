# Qwen3.6 CUDA Native Quant Plan

Date: 2026-06-15

Status: implementation plan, not shipped.

Goal: run `Qwen3.6-35B-A3B` FP8 and NVFP4 checkpoints on CUDA with the quantized
representation **resident** in HBM. The CUDA forward path consumes the
checkpoint's quantized bytes directly. It must not load-time materialize full
model weights to dense BF16.

## Why this is forced (budget — read before any code)

The justification is not "bandwidth matters" in the abstract; it is two
measurable numbers. Both are estimates to be **confirmed in P1** against a real
checkpoint, but they already decide the architecture.

Residency (35B total / ~3B active, A100 80 GB):

| Format | bytes/param | resident weights | fits with real KV + activations on 80 GB? |
|--------|-------------|------------------|-------------------------------------------|
| BF16   | 2.0         | ~70 GB           | No — ~10 GB left, infeasible for real ctx |
| FP8    | ~1.0        | ~35 GB           | Yes, comfortable |
| NVFP4  | ~0.56       | ~18–20 GB        | Yes (matches the ~19 GB MLX-4bit residency cited in CLAUDE.md) |

BF16 not fitting is the **forcing function**. The real comparison for the quant
path is "serve at all" vs "cannot serve", not "quant vs BF16 on the same box".

Decode ceiling (B=1, weight-read-bound, A100 HBM ≈ 2.0 TB/s, **upper bound only**
— ignores KV/attention/routing reads):

| Format | active bytes/token | tok/s ceiling |
|--------|--------------------|---------------|
| BF16   | ~6 GB              | ~330          |
| FP8    | ~3 GB              | ~660          |
| NVFP4  | ~1.7 GB            | ~1190         |

So quant buys ~2–4× decode headroom **and** is the only way the model fits. That
pair of numbers is the entire license. P1 confirms the exact param count, active
set, and scale overhead before any kernel is written.

A100 hardware reality: A100 is `sm_80`. It has **no** Hopper/Blackwell FP8/FP4
tensor-core MMA. In this plan "native" means *resident checkpoint-native quant
buffers consumed directly by ARLE CUDA kernels that dequantize inside the
operator*, then accumulate in BF16/FP32 on CUDA cores (or a `sm_80`-ported
Marlin mixed-input mainloop). It does **not** mean hardware FP8/FP4 MMA. This
distinction drives the decode-vs-prefill split below.

## Decision

The previous load-time BF16 materialization idea is rejected — it solves neither
residency nor weight bandwidth.

A second WIP (currently in `loader.rs`, see Blockers) re-implemented exactly that
rejected approach (`materialize_qwen35_fp8_to_bf16`, `materialize_qwen35_nvfp4_to_bf16`).
It is deleted in P0.

The accepted first version:

- Quantized checkpoint tensors stay resident as quantized device buffers.
- The Qwen3.5/3.6 CUDA linear and MoE paths dispatch on a **numeric-ABI** resident
  `WeightFormat` (named by scale ABI, not by model — see §Format Contracts).
- Kernels dequantize at tile/group granularity inside the operator.
- Full-weight dense BF16 expansion is forbidden.
- **New kernels are written only for the genuine gap.** Quant decode and quant MoE
  reuse and extend the kernels already vendored/wired in-tree (§Existing assets).
  This is mandated by CLAUDE.md "先用最好的再自己写" and the
  `kernels-align-sglang-no-handwrite` rule.

## Existing assets (survey result — do not re-invent)

The first version of this plan proposed a new `qwen35_quant_gemm.cu` and three new
`Qwen*` formats. A tree survey shows most of that already exists; the plan now
extends it instead of duplicating it.

Already in-tree and relevant:

- **Marlin stack, vendored from vLLM/Neural-Magic** (`csrc/gemm/marlin_pf8/`,
  "Adapted from IST-DASLab/marlin", Apache-2.0): `marlin_kernel.cu`,
  `marlin_repack.cu`, `marlin_w4a8_kernel.cu`, `marlin_w4_fp8_kernel.cu`,
  `quantized_gemv.cu`, `quantized_gemv_mma.cu`.
  - Caveat: `marlin_w4_fp8_kernel.cu` is **`sm_89`-only** and is **INT4-weight
    (GPTQ U4B8) + FP8-activation**, not FP8-weight. It serves neither A100 nor the
    Official-FP8-weight case as-is.
  - `quantized_gemv.cu` has no arch gate → CUDA-core, A100-viable.
- **`csrc/gemm/moe_grouped_gemm.cu`** — its own header states it is the Qwen3.5/3.6
  single-GPU MoE path (permute → grouped GEMM → combine), `sm_70`-safe CUDA-core,
  and that "**W4 nibble-decode variant is an explicit follow-up (the Qwen3.6
  production checkpoint ships 4-bit experts)**". The 4-bit MoE kernel architecture
  is therefore already decided.
- **`crates/infer-cuda/src/moe.rs`** — the live, wired BF16 grouped path:
  `moe_bf16_grouped_gemm_swiglu_decode` / `_decode` (R≤256), `_pair_batch` /
  `_batch` (R>256), with `ARLE_QWEN35_DEEPGEMM` (SM90 BF16) opt-in. `moe.rs:30`:
  "W4/4-bit is a separate follow-up: the two `moe_bf16_grouped_gemm_*` call sites."
- **`cuda-kernels::WeightFormat`** (`tensor.rs:909`) already has `W8A16`, `W4A16`,
  `MarlinW4A8`, `W2A16`, GGUF `Q{3,4,5,6}_K`, `Dsv4Fp8BlockScaled`,
  `Dsv4Fp4BlockScaled`, plus matching `DeviceMatrix` quant constructors. These
  constructors are **not yet called by any `infer-cuda` loader** (dormant, not
  missing).
- **Offload infra**: `DeviceMatrix`/raw-slice offload replaces live buffers with a
  1-element placeholder (`tensor.rs:756`, `:1514`, `:1613`). Reconcile with this;
  do not invent a parallel offload path.

DSv4 FP8 dispatches through DeepGEMM + `dsv4_grouped_gemm.cu` + `dsv4_fp8_decode_moe.cu`
(Hopper tensor-core), **not** Marlin. That lane is unchanged by this plan.

## Non-Goals

- Do not reuse `Dsv4Fp8BlockScaled` / `Dsv4Fp4BlockScaled` resident matrices for
  Qwen3.6. DSv4 uses an E8M0 scale ABI; Qwen FP8/NVFP4 does not.
- Do not add per-model format variants (`QwenFp8*`). Add **ABI-named** formats and
  keep the model tensor-role→format mapping in the Qwen loader.
- Do not add Qwen quant types above `infer-cuda` / `cuda-kernels`.
- Do not write a new `qwen35_quant_gemm.cu`. Extend `quantized_gemv.cu` and
  `moe_bf16_grouped_gemm_*`.
- Do not use SGLang/ModelOpt `SM100`-only NVFP4 kernels, or Hopper DeepGEMM, on A100.
- Do not claim MMLU / SWE Pro / throughput from source review or Mac typecheck.
  A100 serve/eval is the gate.

## Current Blockers

1. `crates/infer-cuda/src/loader.rs` has an incomplete WIP (the rejected
   BF16-materialization path) that does not compile. It references 8 undefined
   symbols — confirmed: `QWEN35_FP8_BLOCK`, `QWEN35_NVFP4_GROUP`,
   `qwen35_weight_prefix`, `tensor_to_f32_vec`, `materialize_fp8_blocks_to_bf16`,
   `materialize_fp8_scalar_to_bf16`, `materialize_nvfp4_to_bf16`,
   `tensor_scalar_f32` (0 definitions each).
2. The WIP functions are unreachable from the real load path. `qwen35.rs` still
   calls BF16-only `load_matrix`, `load_matrix_sharded`, `load_qkv_head_sharded`,
   and the BF16 MoE loader.
3. `load_linear_qkv_sharded` / `load_conv1d_sharded` in `qwen35.rs` read raw BF16
   bytes directly. The qkv helper must become quant-aware; conv/norm/bias stay
   strict BF16/F32.
4. Qwen3.6 MoE stacked experts are the critical path (they dominate both residency
   and decode bandwidth). Production experts ship stacked/fused tensors
   (`experts.gate_up_proj`, `experts.down_proj`); a dense-2D-only loader is
   insufficient.
5. Unrelated dirty files (Gemma/Metal work in `git status`) stay untouched. Stage
   by explicit path only.

## Format Contracts

Formats are named by **scale ABI**, generic across models. Detection uses sibling
tensor names + `quantization_config`, never dtype alone.

### Official FP8 → `Fp8BlockScaled { block_m, block_k, inverse }`

- weight: `F8_E4M3`
- sibling scale: `weight_scale_inv` (observed 128×128 weight blocks)
- kernel computes `value = e4m3(weight) * scale_block` (or `/ scale_block` if the
  on-disk field is an inverse scale)

Resolved in P1, not deferred: whether `*.weight_scale_inv` is always an inverse
scale and always 128×128 for every CUDA-consumed Qwen3.6 FP8 tensor class. This is
a header read; it gates the kernel's multiply-vs-divide.

### ModelOpt FP8 → `Fp8PerShard`

- weight: `F8_E4M3`
- sibling scales: `weight_scale`, `input_scale` (may be per logical shard, not one
  global scalar)
- preserve per-logical-shard scale semantics; do **not** collapse to `max()` unless
  a native-kernel remap proves equivalence for that tensor role
- `input_scale` is activation metadata; first version may store it but **fails
  closed** if a kernel would need unimplemented dynamic activation quantization

### ModelOpt NVFP4 → `Fp4E2M1Group { group_size: 16 }`

- weight: `U8`, two FP4 `E2M1` values per byte
- weight scale: `F8_E4M3`, one per group of 16 input elements
- second scale: `weight_scale_2`, F32 scalar or per logical shard
- `input_scale`: activation scale metadata
- kernel computes from packed FP4 nibbles + group scale + global scale, no
  full-matrix expansion
- E2M1 is a float micro-format ({0, .5, 1, 1.5, 2, 3, 4, 6} ± sign) — **not** INT4.
  The existing W4A16 INT4 dequant LUT must not be reused blind; the E2M1 LUT
  replaces only the dequant step of the shared Marlin/grouped mainloop.

The numeric decode tables (`decode_f8_e4m3fn`, `decode_fp4_e2m1`) are pure Rust,
authored once, and **shared** by loader-side scale validation and kernel reference
tests — one source of truth for E4M3/E2M1.

## File-Level Implementation

### P0 — Restore a compiling tree

Files: `crates/infer-cuda/src/loader.rs`

- Delete the incomplete BF16-materialization WIP (the 8 undefined-symbol functions).
- Keep existing BF16 and DSv4 loaders unchanged. No new behavior.

Exit gate:

```bash
CUDARC_CUDA_VERSION=12060 cargo check -p infer-api --release \
  --no-default-features --features cuda,no-cuda --lib
```

### P1 — Checkpoint inspection + budget (BLOCKING, no code)

Files: none (produces a facts note appended to this plan).

- Read the real Qwen3.6 FP8 and NVFP4 checkpoint safetensors headers + each
  `config.json` `quantization_config`. Enumerate every `*.weight`, `*.weight_scale*`
  tensor: dtype, shape, scale block/group shape, inverse-or-direct, per-shard-or-scalar.
- Resolve every open item in §Format Contracts. No assumption survives into P2.
- Confirm the §Budget numbers: exact total/active param count, exact scale overhead,
  exact resident bytes per format, recomputed decode ceiling.

Exit: a one-page "format facts + budget" block with no remaining `?`.

### P2 — Generic quant codec + ABI-named formats

Files: `crates/infer-cuda/src/lib.rs`, `crates/infer-cuda/src/quant_format.rs`

```rust
pub(crate) enum QuantFormat {
    DenseBf16,
    DenseF32,
    Fp8BlockScaled { block_m: usize, block_k: usize, inverse: bool },
    Fp8PerShard,
    Fp4E2M1Group { group_size: usize },
}

pub(crate) struct QuantTensorView {
    pub name: String,
    pub logical_shape: Vec<usize>,
    pub storage_dtype: safetensors::tensor::Dtype,
    pub format: QuantFormat,
    pub weight: OwnedTensor,    // packed bytes, never dense BF16
    pub scales: Vec<OwnedTensor>,
}
```

Functions: `detect_quant_format(name, tensor, siblings, manifest)`,
`decode_f8_e4m3fn(byte)->f32`, `decode_fp4_e2m1(nibble)->f32`,
`read_quant_manifest(model_path)`, `validate_scale_shapes(view)`.

Rules: detection from sibling names + config; the codec returns views + metadata
and allocates **no** dense BF16 matrix. Reconcile names with the existing
`WeightFormat` enum so there is one format vocabulary, not two.

### P3 — Resident weight ABI (extend, don't fork)

Files: `crates/cuda-kernels/src/tensor.rs`

- Add `WeightFormat` variants named by ABI: `Fp8BlockScaled`, `Fp8PerShard`,
  `Fp4E2M1Group`. Do not prefix `Qwen`.
- Extend `DeviceMatrix` with the genuinely-new scale fields only:
  `qweight: Option<CudaSlice<u8>>`, `qscale_fp8: Option<CudaSlice<u8>>`,
  `scale_f32: Option<CudaSlice<f32>>`, `scale2_f32: Option<CudaSlice<f32>>`,
  `block_m`/`block_k`/`group_size`. Reuse existing quant-constructor scaffolding
  where the shape rules already match.
- Constructors: `from_fp8_block_scaled(...)`, `from_fp8_per_shard(...)`,
  `from_fp4_e2m1_group(...)`. Shape validation next to the constructors.

Invariants: `data` stays a dummy 1-element BF16 buffer for resident quant matrices
(same pattern DSv4 uses); `rows`/`cols` are logical dims; **offload/reload must use
the existing 1-element-placeholder path** (`tensor.rs:1613`) — fail closed before
offload if a quant buffer cannot be snapshotted, never clone the dummy `data`.
DSv4 constructors and DeepGEMM caches unchanged.

### P4 — Decode kernels: extend `quantized_gemv.cu`

Files: `crates/cuda-kernels/csrc/gemm/quantized_gemv.cu`, `crates/cuda-kernels/src/ffi/gemm.rs`

- Add E4M3 block/per-shard and E2M1 group **dequant-then-dot** variants to the
  existing CUDA-core GEMV (A100-viable, weight-read-bound — the bandwidth win is here).
- FFI: `fp8_block_gemv_cuda`, `fp8_per_shard_gemv_cuda`, `fp4_e2m1_gemv_cuda` (or one
  entry dispatching on a format tag, matching the file's existing convention).
- Kernel rule: no full-matrix BF16 side buffer; tile/register/shared dequant only;
  FP32 accumulate, BF16 output.

This is the immediate eval path (decode). Do **not** open a new `.cu` file.

### P5 — Operator dispatch

Files: `crates/infer-cuda/src/ops.rs`

- `gemv` / `gemm_batch` branch on `DeviceMatrix::weight_format()` **before**
  dereferencing `weight.data` (which is the dummy buffer for quant matrices).
- Dense BF16 keeps the current path; DSv4 keeps the DSv4 path; ABI quant formats
  call the new variants.
- Fail closed: unsupported format/shape returns an error naming the exact matrix and
  format. No implicit dense fallback.

### P6 — Qwen3.6 linear load wiring

Files: `crates/infer-cuda/src/qwen35.rs`

Route every matrix load in `from_safetensors_with_tp` through the quant-aware
loader helpers (ABI-named, e.g. `load_quant_matrix`, `load_quant_matrix_sharded`,
`load_quant_qkv_head_sharded`): `embed_tokens`, `lm_head`, attention
`q/k/v/o_proj`, linear-attn `in_proj_{qkv,z,b,a}` + `out_proj`, dense MLP
`gate/up/down_proj`, MoE routed experts + router gate + shared expert.

Sharding rules: BF16/F32 dense may reuse byte-slice helpers; FP8 row/col sharding
slices weight **and** scale together; NVFP4 row sharding slices packed columns +
scale groups together; NVFP4 column sharding must be group-aligned else fail closed;
stacked-expert slicing works on packed dims, never a materialized dense shape.

Keep strict BF16/F32 for: norms, `dt_bias`, `A_log`, conv1d weights (unless a
checkpoint proves quantized conv exists and a kernel contract is added).

OPD/LoRA: `remerge_student_lora` fails closed for resident quant matrices in v1.

### P7 — MoE: extend `moe_bf16_grouped_gemm_*`

Files: `crates/infer-cuda/src/moe.rs`, `crates/infer-cuda/src/loader.rs`,
`crates/cuda-kernels/csrc/gemm/moe_grouped_gemm.cu`

- Add E4M3 / E2M1-group **nibble/byte-decode** variants to the two
  `moe_bf16_grouped_gemm_*` call sites (the path `moe.rs:30` already prescribes).
  Same grouping/permute/combine; the only change is the per-element decode in the MAC.
- Extend `MoeLayerWeights` to carry resident quant expert matrices.
- Route routed **and** shared expert through the same quant dispatch.
- Disable the Hopper DeepGEMM grouped lane when experts are resident quant (it is a
  separate FP8-tensor-core lane, not the A100 path).

Exit: Qwen3.6 MoE layers allocate no dense BF16 expert weights; decode runs routed +
shared expert quant kernels.

### P8 — Prefill GEMM tensor-core (split out, license-or-kill)

Files: `crates/cuda-kernels/csrc/gemm/marlin_*` (port) or `moe_grouped_gemm.cu` /
`quantized_gemv_mma.cu` (CUDA-core batch)

This is the only part with real kernel risk and is isolated accordingly. A100
prefill is compute-bound; a CUDA-core dequant GEMM (~3.9 TFLOP/s class per
`moe.rs`) likely **loses** to BF16 + tensor cores.

- Option A: port the vendored Marlin mixed-input mainloop from `sm_89` to `sm_80`,
  swapping the dequant step to the E4M3 / E2M1 LUT.
- Option B: accept CUDA-core dequant batch GEMM, decode-residency-only.

Decision gate: a same-binary, same-shape A/B of quant prefill vs BF16 prefill at the
SLO prompt length. Ship whichever wins; if neither beats "model does not fit in
BF16", the quant path stands on residency alone and that is stated explicitly. No
default flip without this measured A/B (per CLAUDE.md license-or-kill on wall-clock).

## Tests

Pure (CPU, `--features no-cuda`):

- `decode_f8_e4m3fn_known_values`, `decode_fp4_e2m1_known_values`
- `fp8_block_scale_shape_validation`, `fp8_per_shard_scale_is_preserved`
- `fp4_e2m1_group16_shape_validation`
- `quant_format_rejects_dsv4_e8m0_scale_abi`
- loader: fake FP8 linear reaches `Fp8BlockScaled`; fake NVFP4 expert reaches
  `Fp4E2M1Group`; TP shard rejects unaligned NVFP4 column shards
- **round-trip**: kernel-reference dequant == loader-validation dequant, sharing the
  P2 LUT (catches a divergent second table)

Kernel (A100):

- tiny FP8 GEMV vs CPU reference; tiny NVFP4 GEMV vs CPU reference
- grouped quant GEMM vs CPU reference, small matrices
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

Smoke + bench:

```bash
python scripts/arle_capability_eval.py --backend arle \
  --base-url http://127.0.0.1:8123 --model-id "$MODEL_ID" \
  --tasks mmlu --n-samples 50 --seed 0 \
  --output bench-output/qwen36-a100-mmlu-smoke

scripts/bench_guidellm.sh qwen36-cuda-native-quant-a100 \
  --target http://127.0.0.1:8123 --model "$MODEL_ID" --processor "$MODEL"
```

## Hard Acceptance

- `dense_materialized_weight_bytes=0` in loader/runtime logs.
- `nvidia-smi` resident memory is incompatible with full dense BF16 residency and
  is consistent with the P1 budget table for the served format.
- Serve log prints resident bytes by format: FP8, NVFP4, BF16.
- `strings target/release/arle | grep -E 'fp8_block_gemv|fp4_e2m1_gemv|moe_.*_decode'`
  finds the linked quant kernels.
- FP8 checkpoint passes serve + curl smoke + MMLU-50 plumbing.
- NVFP4 checkpoint passes serve + curl smoke + MMLU-50 plumbing.
- No throughput claim until `guidellm` completes; no default prefill-path flip
  without the P8 A/B.
- No capability claim from MMLU-50 or SWE Pro limit-3 (plumbing only).

## Commit Policy

Runtime changes under `crates/infer-cuda` / `crates/cuda-kernels` require a
wins/errors entry. If A100 is unavailable locally, write a `pending-remote` wins
stub naming the exact remote gate.

Tranches (stage by explicit path; do not stage unrelated dirty files):

1. P0 compile restoration.
2. P1 facts note (docs-exempt from bench entry).
3. P2/P3 codec + resident ABI + pure tests.
4. P4/P5 decode kernels + op dispatch.
5. P6/P7 linear + MoE wiring.
6. P8 prefill A/B + A100 eval evidence entry.

## Cross-Review Notes

Architecture: keep quant in `infer-cuda` / `cuda-kernels`; do not touch
`infer-seam`/`-core`/`-api` in the first tranche; do not reuse DSv4 resident formats;
**extend existing kernels, name formats by ABI**.

Quant format: dtype-only detection is unsafe; ModelOpt FP8 must preserve per-shard
scale; NVFP4 BF16-fallback rejected — native resident path consumes packed U8
directly; E2M1 ≠ INT4 (LUT, not the W4A16 integer path).

Verification: loader reachability proved by runtime logs or tests; A100 serve/eval
required before declaring support; MMLU-50 and SWE Pro limit-3 are plumbing checks.

## Self-Review

SOLID:

- WIP confirmed non-compiling (8 undefined symbols, 0 definitions each) → P0 real.
- Existing Marlin/`quantized_gemv`/`moe_bf16_grouped_gemm_*` confirmed in-tree;
  `moe.rs:30` + `moe_grouped_gemm.cu` header confirm the 4-bit-MoE follow-up is the
  intended extension point.
- `WeightFormat` already carries DSv4 + W*A16 + Marlin + GGUF variants; ABI-naming
  avoids a second per-model axis.
- A100 = `sm_80`, no FP8/FP4 MMA; `marlin_w4_fp8_kernel.cu` is `sm_89` + W4A8 →
  cannot serve A100 FP8-weight as-is.

Hypothesis until measured:

- A100 quant GEMV/GEMM throughput, and whether P8 prefill quant beats BF16.
- Whether all target FP8/NVFP4 tensors share the probed scale shapes (P1 resolves).
- Exact VRAM fit with resident quant weights + Qwen3.6 KV/recurrent state.

Deferred (declared, not silent):

- Hopper/Blackwell FP8/FP4 hardware-MMA path.
- Full multi-concurrency performance.
- Official MMLU/SWE Pro capability score.
