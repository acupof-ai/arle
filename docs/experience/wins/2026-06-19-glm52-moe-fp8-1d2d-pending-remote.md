# GLM-5.2 MoE FP8 1D2D (revert bf16 detour) — pending-remote

## Context

Tranche C/D (`b41fa075` / `7157f721` / `bfc530ba`) dequantized GLM-5.2's MoE
experts (routed + shared) to **bf16**. That was wrong on two axes:

- **Memory:** 256 experts in bf16 ≈ **1.5 TB** — does not fit 8×H20. FP8 e4m3 =
  **755 GB**, fits. The bf16 detour doubled the routed-expert footprint.
- **Compute:** bf16 grouped GEMM throws away the FP8 tensor-core path the
  vendored DeepGEMM already provides.

The bf16 detour was rationalized as "GLM's `weight_scale_inv` blocks are general
F32 (non-pow2), so the E8M0 re-encode is lossy." True — but the fix is NOT bf16;
it is the **1D2D** DeepGEMM scheme, which consumes **F32 block scales directly**
(`sm90_fp8_gemm_1d2d`, `const float* sfb`). No E8M0 re-encode, no dequant,
lossless.

## What changed

Routed + shared GLM experts now load as **FP8 e4m3 + F32 `weight_scale_inv`**
(128×128 block) and ride the SAME `Dsv4Fp8DeepGemmWeightCache` grouped GEMM DSv4
uses:

- `loader.rs`: new `load_dsv4_glm_fp8_as_block_scaled` — reads raw FP8 bytes (no
  dequant) + F32 `weight_scale_inv` into a `WeightFormat::Fp8BlockScaled`
  `DeviceMatrix`; built into the grouped cache via
  `from_fp8_block_scaled_weight{,_pair_rows}` (the 1D2D F32-scale builders).
  GLM's `[N/128, K/128]` F32 scale grid matches the cache's `[scale_rows,
  scale_cols]` directly — a shape assertion fails loudly pre-pod on mismatch.
  Deleted the bf16 routed/shared host-dequant + the lossy E8M0 re-encode
  (`load_dsv4_glm_fp8_as_dsv4`, `load_dsv4_glm_fp8_as_bf16_host`) and the now-dead
  `encode_f8_e4m3fn`.
- `moe.rs`: removed every `is_bf16` branch (`GroupedCache.is_bf16`,
  `build_grouped_cache_bf16`, `bf16_grouped_experts`, `dsv4_shared_expert_bf16`)
  and the GLM bf16 guards on the FP8 decode/GEMV/pooled lanes. GLM routes the
  same FP8 grouped path as DSv4.
- `dsv4.rs`: dropped the `w13_up_grouped` + `shared_*_bf16` `Dsv4MoeLayer` fields;
  **re-enabled the decode-graph for GLM** (the `!plain_o_proj` gate existed ONLY
  because the bf16 shared GEMMs alloc per-call scratch — gone now; the FP8 shared
  path has no per-call alloc, exactly like DSv4 under capture).
- `glm.rs`: GLM has an **unclamped** SwiGLU, but the DSv4 FP8 swiglu kernel clamps
  `gate=min(gate,limit)` / `up=clamp(up,±limit)` and rejects `limit<=0`. Set
  `swiglu_limit = f32::MAX` so the clamp is a guaranteed no-op (faithful to GLM's
  no-clamp) and satisfies the kernel precondition. (Was `0.0`, which both errored
  the kernel and would have clamped to 0.)

## Why no local bench

Runtime change in `crates/infer-cuda/src/` + `crates/deepseek-spec/`, but GLM-5.2
is CUDA + sm_90 (8×H20) only — cannot bench/forward on a Mac. GPU validation is a
pod task (load → forward → `needle_gate` vs the envelope, mirroring the DSv4 FP8
MoE gate). **pending-remote.**

Verified on Mac (no-cuda):
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib` — green.
- `cargo clippy -p infer-cuda` (cuda,no-cuda) — 0 infer-cuda warnings.
- `cargo test -p deepseek-spec` — 9 passed.

Open pod-verify points (`// ponytail:` in source):
1. GLM `weight_scale_inv` resolves to `[N/128, K/128]` F32 at load (asserted — fails loudly if not).
2. GLM V32 SparseIndexed decode captures + replays cleanly under the re-enabled decode-graph.

## Rule

A lossy-quant symptom (E8M0 round-to-pow2 on general F32 block scales) does not
license a 2× memory + dropped-tensor-core detour. The vendored DeepGEMM already
has the **1D2D F32-block-scale** path — `先用最好的`: route GLM's native FP8 +
`weight_scale_inv` straight through it, same cache + same grouped GEMM as DSv4.

Cross-link: supersedes the bf16 MoE in
[`2026-06-19-glm52-tranche-d-forward-pending-remote.md`](2026-06-19-glm52-tranche-d-forward-pending-remote.md).
