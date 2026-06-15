# CUDA Native Weight Quant Subsystem Plan

Date: 2026-06-15

Status: P0-P5 partially shipped; P6/P7 loader and MoE wiring still pending.

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

### Scale-decode ABI × kernel FFI (line-level — author before P4)

The existing GEMV FFI is single-level, 1-byte scale, `decode_e8m0` baked in:
`dsv4_fp8_gemv_batch_cuda(weight:*u8, scales:*u8, input:*Half, output:*mut Half, B, N, K, scale_rows, scale_cols, stream)`.
"Parameterize the scale-decode" therefore **changes the FFI signature per ABI** — it is
not just a device-function swap. Each row is one kernel instantiation + one launcher:

| ScaleAbi | weight | scale buffer | `scale_rows × scale_cols` | decode | apply | global arg | FFI launcher |
|----------|--------|--------------|---------------------------|--------|-------|-----------|--------------|
| `E8M0Block` (DSv4) | u8 E4M3 / packed E2M1 | `*u8` (1B) | `N/blk_m × K/blk_k` (2D) | `decode_e8m0` | × | — | existing `dsv4_fp{8,4}_*` (unchanged) |
| `Fp32InvBlock` (Qwen FP8, 128²) | u8 E4M3 | `*f32` (4B) | `N/128 × K/128` (2D) | identity-f32 | **÷** *(or pre-reciprocal at load → ×)* | — | **new** `gemv_fp8_blockinv_*` |
| `Fp8PerShard` (ModelOpt FP8) | u8 E4M3 | `*f32` (4B) | `shards × 1` | identity-f32 | × | — | reuse `gemv_fp8_blockinv_*` (scale_cols=1) |
| `Fp8E4M3Group+Global` (NVFP4) | u8 (2× E2M1) | `*u8` E4M3 (1B) | `N × K/16` (1D group along K) | `decode_e4m3` | × | **`*f32` scalar/per-shard** | **new** `gemv_fp4_group_*` |

Locked in P1/P2: **pre-reciprocal the FP8 inverse-block scale at load** so the kernel
always multiplies (one apply-op for every ABI) — confirm no precision loss vs ÷.
`Fp8PerShard` is `Fp32InvBlock` with `scale_cols=1`, so it needs no new kernel. NVFP4's
`global` is one extra pointer arg; the rest reuses the FP4 kernel body. The `*_pair_batch`
(gate+up) and `*_route_gemv_*` (MoE) variants take the same per-ABI treatment.

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

## Opt-in & activation policy (baseline stays byte-identical)

- **Auto-selected, not a manual flag.** The quant path is chosen from the checkpoint's
  `quantization_config` at load. A non-quant (BF16) checkpoint takes the existing path
  **byte-for-byte unchanged** — no quant code on its hot path. Kill-switch
  `ARLE_DISABLE_RESIDENT_QUANT=1` makes the loader reject a quant checkpoint (fail closed)
  rather than silently materialize — there is no BF16-materialize fallback.
- **v1 is weight-only.** Weights dequant inside the operator; **activations stay BF16**
  (`input_scale` is read + stored but not applied). This is the correctness-safe superset:
  BF16 activations × dequant-BF16 weights is *more* precise than the calibrated W4A4/W8A8,
  so it is the numerical reference. The executor must **not** implement dynamic activation
  quantization in v1 — a kernel that would need it fails closed naming the tensor.
- True activation quant (to reach FP8/FP4 tensor cores) is a Blackwell/`sm_90+` lane,
  explicitly deferred.

## File-Level Implementation

Dependency DAG / critical path: **P1 blocks everything** (no ABI assumption survives it),
then `P2 → P3 → P4 → P5 → P6`; `P7` follows P4 (reuses the P4 ScaleAbi kernels) and `P8`
follows P6 (needs a working quant forward to A/B against). P2 (pure-Rust codec/registry)
and the P3 `WeightFormat`/`DeviceMatrix` scaffolding parallelize once P1 lands; the kernel
work (P4) is the long pole. P0 is done.

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
- **Expert-tensor layout (blocks P3/P6/P7).** Record the exact rank/shape of
  `experts.gate_up_proj` / `experts.down_proj` (fused `[E, 2·I_moe, H]` or split? row-major?)
  and **whether the matching `weight_scale*` tensors are stacked per-expert `[E, …]`** — the
  per-expert slice must cut weight and scale together on the E axis.
- **Both ignore-lists.** Record the un-quantized tensor set for FP8 *and* NVFP4 separately
  (they may differ); this is the P6 `QuantWeightMap` ground truth.
- Resolve every open item in §Format Contracts, incl. the inverse-vs-÷ / pre-reciprocal
  decision in the §Scale-decode ABI table. No assumption survives into P2.
- Confirm the §Budget numbers: exact total/active param count, exact scale overhead,
  exact resident bytes per format, recomputed decode ceiling.

Exit: a one-page "format facts + budget" block with no remaining `?`.

Result (2026-06-15, range-read safetensors headers only; no weight payloads
downloaded):

- Scope pinned for v1 implementation:
  `Qwen/Qwen3.6-35B-A3B-FP8@95a723d08a9490559dae23d0cff1d9466213d989`
  and
  `RedHatAI/Qwen3.6-35B-A3B-NVFP4@e850c696e6d75f965367e816c16bc7dacd955ffa`.
  `nvidia/Qwen3.6-35B-A3B-NVFP4` and
  `unsloth/Qwen3.6-35B-A3B-NVFP4` were sampled as ABI variants only: nvidia
  uses `.weight` + scalar `.weight_scale_2`/`.input_scale`, while
  RedHat/unsloth use `.weight_packed` + `.weight_global_scale` /
  `.input_global_scale`. Do not silently treat these as one loader ABI.
- Shared config facts: 40 layers, hidden 2048, MoE intermediate 512,
  256 routed experts, `num_experts_per_tok=8`. Language-model logical weights
  excluding visual/MTP are 34.6606B params. Per-token full-weight-read active
  set excluding the token-embedding table is 2.9465B logical params
  (lm_head still counts; embedding full matrix does not).
- Official FP8 main language files are `layers-0..39.safetensors` +
  `outside.safetensors`. Header count: 62,636 tensors. Language resident
  bytes, excluding the visual weights in `outside.safetensors`: **35.708 GB
  / 33.26 GiB** = 33.617 GB `F8_E4M3` weights + 4.10 MB BF16
  `weight_scale_inv` + 2.087 GB dense BF16 language weights. The text decode
  weight-read estimate is **3.481 GB/token** → **~575 tok/s ceiling at
  2.0 TB/s** before KV/attention/routing overhead.
- Official FP8 scale ABI: every quantized language weight has a BF16
  `.weight_scale_inv` with 128x128 block shape (examples:
  `[2048,512] -> [16,4]`, `[512,2048] -> [4,16]`,
  `[8192,2048] -> [64,16]`). Despite the suffix, the checkpoint's numeric
  semantics are **direct multiply**, not divide: a range-read A/B against the
  dense BF16 `Qwen/Qwen3.6-35B-A3B` tensor
  `layers.0.mlp.experts.gate_up_proj[expert=0, gate, 0:128, 0:128]` gave
  mean abs error `fp8 * scale = 6.09e-5` vs
  `fp8 / scale = 2.30e5`. P2 stores this as a direct f32 multiplier after
  BF16->f32 decode; do **not** pre-reciprocal it.
- Official FP8 expert layout is split per expert, not stacked:
  `model.language_model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight`.
  Dense BF16 baseline uses stacked `experts.gate_up_proj` /
  `experts.down_proj`, so loader tests must cover split-vs-stacked mapping.
  FP8 quantizes full-attention projections, shared expert projections, and
  routed experts. FP8 leaves norms, router `mlp.gate`, `shared_expert_gate`,
  linear-attn state/norm/conv/small projections BF16, but
  `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` are FP8.
- RedHat NVFP4 main file is `model.safetensors`. Header count: 123,973
  tensors. Language resident bytes: **22.444 GB / 20.90 GiB** =
  16.305 GB packed U8 E2M1 weights + 2.038 GB `F8_E4M3` group scales +
  0.247 MB F32 global scales + 4.100 GB dense BF16 language weights. The
  text decode weight-read estimate is **3.785 GB/token** →
  **~528 tok/s ceiling at 2.0 TB/s** before KV/attention/routing overhead.
  This is lower than the draft table's 1.7 GB/token because RedHat keeps all
  `linear_attn.*` projections BF16 and NVFP4 carries large per-group scales.
- RedHat NVFP4 scale ABI: quantized weights are split per expert as
  `.weight_packed` with two E2M1 values per byte, low nibble first. Shapes:
  gate/up `[512,1024]` packed + `weight_scale [512,128]`; down
  `[2048,256]` packed + `weight_scale [2048,32]`. Formula confirmed by the
  same dense BF16 range-read A/B: `e2m1(weight) *
  decode_e4m3(weight_scale) / weight_global_scale` is correct
  (`mean_abs_err=2.43e-4` for the first 128x128 gate block); direct multiply
  by `weight_global_scale=31104.0` is wrong. P2 should normalize this to a
  direct multiplier by storing `1.0 / weight_global_scale` as f32 if the
  kernel wants a multiply-only ABI. `input_global_scale` is activation
  metadata; v1 reads/stores it but does not apply activation quantization.
- RedHat NVFP4 ignore-list/header facts: all `linear_attn.*` weights are BF16,
  router `mlp.gate` and `shared_expert_gate` are BF16, norms are BF16; full
  attention projections, routed experts, and shared expert projections are
  NVFP4. RedHat `model_mtp.safetensors` and `model_visual.safetensors` are
  auxiliary and were not included in the first text-runtime resident budget.

Implementation consequence: P2/P3 must support ABI aliases rather than one
idealized NVFP4 naming convention:

- FP8 official: `.weight` + BF16 `.weight_scale_inv`, 128x128, direct multiply.
- NVFP4 RedHat/unsloth: `.weight_packed` + F8 `.weight_scale` + inverse F32
  `.weight_global_scale`, low-nibble-first.
- NVFP4 nvidia: `.weight` + F8 `.weight_scale` + scalar F32
  `.weight_scale_2` / `.input_scale`; full-budget validation remains a
  separate gate before enabling that repo id.

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

Result (2026-06-15, P4/P5 tranche):

- Added ABI-generic decode GEMV launchers in the existing
  `crates/cuda-kernels/csrc/gemm/quantized_gemv.cu`, without opening a new
  `.cu` file:
  `gemv_fp8_block_scaled{,_batch}_cuda` for E4M3 weights with f32 block/per-shard
  scales, and `gemv_fp4_e2m1_group{,_batch}_cuda` for packed E2M1 weights with
  E4M3 group scales plus an f32 global multiplier.
- Kept the DSv4 E8M0 symbols intact. DSv4 uses the existing
  `dsv4_fp{8,4}_*` launchers; Qwen-format ABI entrypoints are separate symbols.
- Split CUDA GEMV FFI tests out of
  `crates/cuda-kernels/src/ffi/gemm.rs` into
  `crates/cuda-kernels/src/ffi/gemm_tests.rs`.
- Split resident quant linear dispatch out of `ops.rs` into
  `crates/infer-cuda/src/ops/quant_linear.rs`. `gemv` and `gemm_batch` now branch
  on `weight_format.is_quantized()` before dereferencing the dense dummy
  `DeviceMatrix::data`.
- H20 CUDA reference tests passed for FP8 block-scaled and FP4 group-scaled GEMV
  against CPU references. DSv4 post-P4 parity matched the pre-P4 byte-parity
  reference for batch decode validation at batch 2 and batch 4.
- This is a correctness/build gate only. It is not yet a Qwen3.6 serve or
  throughput claim because P6 loader wiring and P7 quant MoE routing remain open.

### P6 — Qwen3.6 linear load wiring

Files: `crates/infer-cuda/src/qwen35.rs`

Drive `from_safetensors_with_tp` from the Qwen3.6 `QuantWeightMap`, not a hand list.
The dispositions are grounded in checkpoint headers plus the RedHatAI NVFP4
recipe ignore-list; FP8 and NVFP4 differ for linear attention, so do not reuse
one format's ignore-list as the other's truth:

- **Quantize** (route through `load_quant_matrix{,_sharded}`, `load_quant_qkv_head_sharded`):
  full-attention `q/k/v/o_proj`, MoE routed experts (stacked `experts.gate_up_proj` /
  `experts.down_proj`), shared-expert up/down, and FP8
  `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`.
- **Keep BF16/F32**: `embed_tokens`, `lm_head`, router `mlp.gate`,
  `shared_expert_gate`, `linear_attn.{in_proj_b,in_proj_a,A_log,dt_bias,conv1d}`,
  norms, and all NVFP4 `linear_attn.*` weights when the checkpoint keeps them dense.

Sharding rules: BF16/F32 dense may reuse byte-slice helpers; FP8 row/col sharding
slices weight **and** scale together; NVFP4 row sharding slices packed columns +
scale groups together; NVFP4 column sharding must be group-aligned else fail closed;
stacked-expert slicing works on packed dims, never a materialized dense shape.

(conv1d stays BF16 unless a checkpoint proves quantized conv exists and a kernel
contract is added.)

OPD/LoRA: `remerge_student_lora` fails closed for resident quant matrices in v1.

Instrumentation (feeds Hard Acceptance): the loader accumulates resident bytes per
`WeightFormat` plus a `dense_materialized_weight_bytes` total (must stay 0 for quant
ckpts), logged once at engine-ready. This is the runtime proof that no dense BF16
expansion happened — not an inference.

### P7 — MoE: extend `moe_bf16_grouped_gemm_*`

Files: `crates/infer-cuda/src/moe.rs`, `crates/infer-cuda/src/loader.rs`,
`crates/cuda-kernels/csrc/gemm/moe_grouped_gemm.cu`

- **R-band selector (mirror the BF16 one in `moe.rs`, do not invent).** The quant path
  reuses the existing routed-row thresholds: `R ≤ 256` → quant `route_gemv` decode
  (+ `pair_batch` for gate/up); `R > 256` → quant grouped GEMM; the `R ≥ 1024` DeepGEMM
  band is **disabled** for quant (Hopper FP8-tensor-core lane, not A100). Confirm the 256
  crossover still holds for the quant kernels in the P7 A/B.
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

Decision gate: a same-binary, same-shape A/B of quant prefill/decode vs BF16 at
the SLO prompt length. The dated bench entry must include Delta% for output
tok/s, TTFT, ITL, and peak VRAM. Ship whichever wins; if resident quant beats
neither throughput nor peak VRAM on any binding shape, the result is
KILL/iterate. If BF16 cannot fit a binding shape, residency may be the license,
but that must be stated as a memory win with the measured VRAM table. No default
flip, "best kernel", or throughput claim without this measured A/B (per
CLAUDE.md license-or-kill on wall-clock).

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
- fixture helper `fake_quant_safetensors(format, shape, scales)` — writes a minimal
  safetensors + `quantization_config` with the sibling scale tensors, so the loader tests
  above run with no real checkpoint.

DSv4 regression (the byte-identity proof for P4): **before** touching `quantized_gemv.cu`,
capture greedy decode for a fixed prompt + fixed seed on a DSv4 FP8 checkpoint into a
committed reference; the P4 acceptance re-runs it and asserts byte-equality (the E8M0
instantiation must be untouched).

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
- DSv4 regression: post-P4 output byte-identical to pre-P4 (the E8M0 path unchanged),
  via the saved fixed-prompt fixed-seed greedy reference captured before P4 (see §Tests).
- **Correctness gate = `scripts/needle_gate.py`**, not MMLU: needle ladder ×3 same-config
  repeats inside the baseline envelope for FP8 and NVFP4 (per CLAUDE.md KV-parity gate —
  needle + same-config-twice + self-consistency, NOT byte-vs-BF16, which MoE
  non-determinism confounds).
- FP8 and NVFP4 each pass serve + curl + the needle gate; MMLU-50 / SWE-Pro limit-3 are
  **plumbing checks only** (the pipe works), never a capability claim.
- **Perf gate is REQUIRED, not optional.** Correctness only licenses the next
  experiment. After coherent FP8 serve output, run a same-binary, same-shape
  `scripts/bench_guidellm.sh` A/B: BF16 Qwen3.6 baseline vs resident FP8 quant,
  wall-clock framing per `docs/bench-and-trace-spec.md`, and report Delta% for
  output tok/s, TTFT, ITL, and peak VRAM. Quant must beat BF16 on tok/s OR
  peak-VRAM on at least one binding shape. If it beats neither, verdict is
  KILL/iterate, not support/default.
- The current grouped quant-GEMV is a correctness kernel only. It must be A/B'd
  against adopt-first routes (DeepGEMM / CUTLASS / Marlin / vendor, aligned to
  SGLang where applicable) before any "best kernel" or default claim. Hand-rolled
  kernels stay provisional unless an explicit measured gap licenses them.
- No throughput claim until `guidellm` completes; no default prefill-path flip
  without the P8 A/B and the BF16-vs-quant Delta% table.

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
