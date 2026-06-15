# CUDA Native Weight Quant Subsystem Plan

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

A100 phase split (do not read the NVFP4 decode ceiling as a global throughput claim):
the ceilings above are **decode GEMV** (memory-bound — NVFP4's 4-bit read is genuinely
fewer bytes, so it can beat FP8 here; measure). **Prefill GEMM is different**: A100 has
no FP4 MMA, so NVFP4 runs through the Marlin-FP4 fallback (dequant→FP16 MMA), which vLLM
reports as **≈ FP8 throughput — memory savings only, no prefill speedup**. Net: NVFP4's
A100 value is residency (+ a possible decode-bandwidth edge), not prefill compute.

A100 hardware reality: A100 is `sm_80`. It has **no** Hopper/Blackwell FP8/FP4
tensor-core MMA. In this plan "native" means *resident checkpoint-native quant
buffers consumed directly by ARLE CUDA kernels that dequantize inside the
operator*, then accumulate in BF16/FP32 on CUDA cores (or a `sm_80`-ported
Marlin mixed-input mainloop). It does **not** mean hardware FP8/FP4 MMA. This
distinction drives the decode-vs-prefill split below.

## Decision

The previous load-time BF16 materialization idea is rejected — it solves neither
residency nor weight bandwidth.

A now-deleted WIP in `loader.rs` re-implemented exactly that rejected approach
(`materialize_qwen35_fp8_to_bf16`, `materialize_qwen35_nvfp4_to_bf16`). P0 removed
it and restored the CUDA/no-cuda typecheck; future work starts from the native
resident path below.

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

- **The FP8 + FP4 dequant-GEMV kernels already exist** in `quantized_gemv.cu`
  (CUDA-core, no arch gate → A100-viable). `:50` `DSV4_FP4_E2M1_LUT`, `:249`
  `dsv4_decode_fp8_e4m3`, `:259` `dsv4_decode_fp4_e2m1`;
  `dsv4_fp8_gemv{,_batch,_batch_tiled}_kernel`, `dsv4_fp4_gemv{,_batch,_batch_tiled}_kernel`;
  and the **MoE grouped/routed family** `dsv4_fp{8,4}_{grouped,route}_gemv_batch_cuda`
  + `_pair_batch_cuda`. They compute generic `decode(weight) * scale` — **the only
  model-specific line is the scale fetch `dsv4_decode_e8m0(scales[...])`**. Weight
  decode, block indexing (`block_h`/`block_w`), batch tiling, warp reduction are all
  generic. This is the load-bearing finding: the Qwen quant path is not new kernels,
  it is parameterizing the scale-decode of these.
- **Marlin stack, vendored from vLLM/Neural-Magic** (`csrc/gemm/marlin_pf8/`,
  "Adapted from IST-DASLab/marlin", Apache-2.0): `marlin_kernel.cu`,
  `marlin_repack.cu`, `marlin_w4a8_kernel.cu`, `marlin_w4_fp8_kernel.cu`.
  - `marlin_template.h` carries the **full NVFP4 GEMM path**: `kFE2M1f` weight +
    `kFE4M3fn` group scale + `global_scale` (`:254,:300,:327,:351,:1655`), FP4→FP16/BF16
    dequant in `dequant.h:400+`. But `marlin_w4_fp8_kernel.cu` only **instantiates** the
    `sm_89` W4A8 shape (`global_scale=nullptr`) — the NVFP4 kernel exists in template,
    just uninstantiated. (This is exactly vLLM's A100 NVFP4-Marlin fallback path.)
  - So the genuine prefill-GEMM gap is *instantiating* a vendored template for `sm_80`,
    not writing a kernel.
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

1. `qwen35.rs` still
   calls BF16-only `load_matrix`, `load_matrix_sharded`, `load_qkv_head_sharded`,
   and the BF16 MoE loader.
2. `load_linear_qkv_sharded` / `load_conv1d_sharded` in `qwen35.rs` read raw BF16
   bytes directly. The qkv helper must become quant-aware; conv/norm/bias stay
   strict BF16/F32.
3. Qwen3.6 MoE stacked experts are the critical path (they dominate both residency
   and decode bandwidth). Production experts ship stacked/fused tensors
   (`experts.gate_up_proj`, `experts.down_proj`); a dense-2D-only loader is
   insufficient.
4. Stage by explicit path only. Do not mix this plan's native-quant work with
   unrelated backend or release changes.

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

## Generic subsystem & model onboarding (the "通用 / 无缝接入" axis)

The variation axis across DSv4 / Qwen-FP8 / Qwen-NVFP4 / INT-affine is the **scale
ABI**, not the model and not the weight element type (the kernels already prove this:
one `decode(weight) * scale` body, only `dsv4_decode_e8m0` is model-specific). The
subsystem is layered so a new model touches only the top layer:

1. **Numeric codec (pure, model-free)** — decode LUTs E4M3 / E2M1 / E8M0 / INT4/8,
   asserted byte-identical to the CUDA LUTs by a round-trip test.
2. **ScaleAbi (the real variation)** — the kernels gain this as a parameter, replacing
   the hardcoded `dsv4_decode_e8m0`: `E8M0Block{block_m,block_k}` (DSv4) ·
   `Fp32InvBlock{block_m,block_k}` (Qwen Official FP8, `w / s_block`) ·
   `Fp8E4M3Group{group}+Fp32Global` (NVFP4 two-level) · `Bf16AffineGroup{group}`
   (W4A16/W8A16 `(q-zp)*s`).
3. **Resident `WeightFormat` = storage × ScaleAbi**, ABI-named (§Format Contracts).

Above these, a **model adapter is data, not code**:

```rust
pub struct QuantWeightMap { pub model: &'static str, pub rules: Vec<QuantRule> }
pub struct QuantRule { pub pattern: TensorPattern, pub disposition: Disposition }
// Disposition ∈ { KeepBf16, KeepF32, Quant { role, format } }
// role        ∈ { Dense, ShardCol, ShardRow, QkvHead, StackedExpert }
```

DSv4 and Qwen3.6 register one `QuantWeightMap` each over the same kernels. **Onboarding
a new model:**

1. Dump its checkpoint header + `quantization_config`; classify each tensor →
   keep-BF16 / keep-F32 / quant(role, format, scale ABI). Note the ignore-list.
2. Map each scale ABI to a layer-2 `ScaleAbi`. **New ABI → add it to layers 1–2 + one
   kernel instantiation; otherwise zero kernel work.**
3. Author the `QuantWeightMap` (data) + loader tests (fake safetensors → expected
   format; TP-shard alignment rejects).
4. Run §Verification Gates. A model whose ABIs already exist needs only steps 1–3.

## File-Level Implementation

### P0 — Restore a compiling tree (done 2026-06-15)

Files: `crates/infer-cuda/src/loader.rs`

- Delete the incomplete BF16-materialization WIP (the 8 undefined-symbol functions).
- Keep existing BF16 and DSv4 loaders unchanged. No new behavior.

Exit gate:

```bash
CUDARC_CUDA_VERSION=12060 cargo check -p infer-api --release \
  --no-default-features --features cuda,no-cuda --lib
```

Result: passed locally on 2026-06-15 after deleting the rejected BF16
materialization WIP.

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

- **Parameterize, don't add.** Lift the hardcoded `dsv4_decode_e8m0(scale)` in the
  existing `dsv4_fp{8,4}_gemv_*` kernels into a `ScaleAbi` scale-decode policy (template
  param or `__device__` functor) covering E8M0 / Fp32InvBlock / Fp8E4M3Group+Global.
  **The DSv4 (E8M0) instantiation must stay byte-identical** (regression-gated).
- Add the Qwen instantiations + generic-named launchers in `ffi/gemm.rs`
  (`gemv_fp8_*`, `gemv_fp4_*`, or one tag-dispatched entry, matching file convention).
- Kernel rule: no full-matrix BF16 side buffer; tile/register/shared dequant only;
  FP32 accumulate, BF16 output.

This is the immediate eval path (decode) and is A100-viable today (CUDA-core, no arch
gate). Do **not** open a new `.cu` file.

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

Drive `from_safetensors_with_tp` from the Qwen3.6 `QuantWeightMap`, not a hand list.
The dispositions are **grounded in the RedHatAI NVFP4 recipe ignore-list**, not guessed:

- **Quantize** (route through `load_quant_matrix{,_sharded}`, `load_quant_qkv_head_sharded`):
  full-attention `q/k/v/o_proj`, MoE routed experts (stacked `experts.gate_up_proj` /
  `experts.down_proj`), shared-expert up/down.
- **Keep BF16/F32** (recipe ignores these): `embed_tokens`, `lm_head`, router
  `mlp.gate`, `shared_expert_gate`, **all `linear_attn.*`** (`in_proj_{qkv,z,b,a}`,
  `out_proj`, `A_log`, `dt_bias`, `conv1d`), and norms.

Confirm the ignore-list against the *FP8* checkpoint too in P1 (it may differ from NVFP4).

Sharding rules: BF16/F32 dense may reuse byte-slice helpers; FP8 row/col sharding
slices weight **and** scale together; NVFP4 row sharding slices packed columns +
scale groups together; NVFP4 column sharding must be group-aligned else fail closed;
stacked-expert slicing works on packed dims, never a materialized dense shape.

(conv1d stays BF16 unless a checkpoint proves quantized conv exists and a kernel
contract is added.)

OPD/LoRA: `remerge_student_lora` fails closed for resident quant matrices in v1.

### P7 — MoE: extend `moe_bf16_grouped_gemm_*`

Files: `crates/infer-cuda/src/moe.rs`, `crates/infer-cuda/src/loader.rs`,
`crates/cuda-kernels/csrc/gemm/moe_grouped_gemm.cu`

- **Decode-shape routed experts**: route through the **already-existing**
  `dsv4_fp{8,4}_route_gemv_*` / `*_pair_batch` family in `quantized_gemv.cu` — same
  P4 `ScaleAbi` parameterization (these are the kernels `moe.rs:30`'s "W4 follow-up"
  pointed at; they exist, they just decode E8M0 today).
- **Prefill-shape (large R)**: add the E4M3/E2M1-group decode to the
  `moe_bf16_grouped_gemm_*` call sites (same grouping/permute/combine; only the
  per-element decode in the MAC changes).
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

- Option A: **instantiate the already-vendored Marlin NVFP4 template** (`kFE2M1f` +
  `kFE4M3fn` + `global_scale`, present in `marlin_template.h`) for `sm_80`, plus the
  FP8-weight Marlin shape. This *is* vLLM's A100 fallback — instantiation, not a
  from-scratch port.
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
- `strings target/release/arle | grep -E 'gemv_fp8|gemv_fp4|route_gemv|marlin.*fp4'`
  finds the linked quant kernels.
- DSv4 regression: post-P4 output byte-identical to pre-P4 (the E8M0 path unchanged).
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

1. P1 facts note (docs-exempt from bench entry).
2. P2/P3 codec + resident ABI + pure tests.
3. P4/P5 decode kernels + op dispatch.
4. P6/P7 linear + MoE wiring.
5. P8 prefill A/B + A100 eval evidence entry.

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

- Historical WIP confirmed non-compiling (8 undefined symbols, 0 definitions each);
  P0 deleted it and the CUDA/no-cuda typecheck now passes.
- Existing Marlin/`quantized_gemv`/`moe_bf16_grouped_gemm_*` confirmed in-tree;
  `moe.rs:30` + `moe_grouped_gemm.cu` header confirm the 4-bit-MoE follow-up is the
  intended extension point.
- `WeightFormat` already carries DSv4 + W*A16 + Marlin + GGUF variants; ABI-naming
  avoids a second per-model axis.
- The FP8 **and** FP4 dequant-GEMV + routed-MoE-GEMV kernels already exist in
  `quantized_gemv.cu`; `dsv4_decode_e8m0` is the **sole** model-specific line → the
  Qwen path is scale-ABI parameterization, not new kernels.
- The Marlin **NVFP4** GEMM is already in `marlin_template.h` (`kFE2M1f`+`kFE4M3fn`+
  `global_scale`), uninstantiated → P8 is an `sm_80` instantiation, not a port.
- Checkpoints real: `Qwen/…-FP8` (block 128), `nvidia/RedHatAI/unsloth …-NVFP4`;
  `qwen3_5_moe` rides `qwen35.rs`; the NVFP4 recipe ignore-list grounds the P6 map.
- A100 = `sm_80`, no FP8/FP4 MMA; `marlin_w4_fp8_kernel.cu` is `sm_89` + W4A8 →
  cannot serve A100 FP8-weight as-is; vLLM-confirmed NVFP4-on-A100 ≈ FP8 (residency win).

Hypothesis until measured:

- A100 quant GEMV/GEMM throughput, and whether P8 prefill quant beats BF16.
- Whether all target FP8/NVFP4 tensors share the probed scale shapes (P1 resolves).
- Exact VRAM fit with resident quant weights + Qwen3.6 KV/recurrent state.

Deferred (declared, not silent):

- Hopper/Blackwell FP8/FP4 hardware-MMA path.
- Full multi-concurrency performance.
- Official MMLU/SWE Pro capability score.
