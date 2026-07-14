# W4A16 (INT4) MoE grouped GEMV — CUDA, pending-remote

> Status: build PASS (H20 sm_90, 2026-07-14); end-to-end 4-bit
> MoE smoke DEFERRED — no W4A16/INT4 MoE model on the box and the HF proxy
> tunnel is down (127.0.0.1:1080 refused + direct HF timeout), so no model
> could be fetched. V100 (sm_70) is the target workload: a 4-bit MoE fits
> 32 GB VRAM where the FP8 variant (32.43 GB) does not.

## Goal

Add INT4/W4A16 weight quantization to the CUDA executor's MoE grouped
path so a 4-bit MoE model fits and runs on V100 32 GB. Dense W4A16
(`quant_linear` gemm/gemv) landed in the prior session; this change
closes the MoE grouped-GEMV lane.

## What changed

- `crates/cuda-kernels/csrc/gemm/quantized_gemv.cu`:
  `w4a16_grouped_gemv_batch_kernel` + `w4a16_grouped_gemv_pair_batch_kernel`
  (gate+up fused) and their `moe_w4a16_grouped_gemv_*_cuda` launchers.
  BF16 storage + FP32 accumulate, per-group BF16 scale, zero-point 8,
  packed 2×INT4/byte — sm_70 compatible (no BF16 compute).
- `crates/cuda-kernels/src/ffi/gemm.rs`: FFI decls for the two launchers.
- `crates/cuda-kernels/src/moe.rs`: `moe_w4a16_grouped_gemv_batch` /
  `_pair_batch` wrappers + `build_expert_qweight_i8_ptr_table` /
  `build_expert_qscale_bf16_ptr_table` pointer-table builders.
- `crates/infer-cuda/src/loader.rs`: `build_moe_layer_pointer_tables`
  routes W4A16 to the i8 qweight table + BF16 scale table (was wrongly
  falling into the u8/FP8 path).
- `crates/infer-cuda/src/moe.rs`: W4A16 arms in `grouped_pair_batch`
  (gate+up) and `grouped_down_batch` (down); shared experts already go
  through the dense W4A16 `gemm_batch`/`gemv` lane.

## Results

- **Build: PASS** (H20 sm_90, `cargo build --release --features cuda`,
  8 crates, 2m 54s). Both `w4a16_grouped_gemv_batch_kernel` and
  `w4a16_grouped_gemv_pair_batch_kernel` confirmed in the binary `.text`.
- **Smoke: DEFERRED** — no W4A16/INT4 MoE model on the box; HF proxy
  tunnel down + direct HF timeout blocked fetch. V100 end-to-end
  (serve + chat on a 4-bit MoE) to follow once a model is available.
- One pre-existing unrelated warning: `unused variable: i` at
  `crates/infer-cuda/src/qwen35.rs:6207` (not in scope).

## Rule

- Match the proven FP4 grouped dispatch pattern (pointer tables +
  offsets/counts/expert_indices) with the existing dense W4A16 nibble
  math — no new abstraction.
- Correctness gate before any perf claim: needle/lever vs the BF16
  envelope on ≥2 prompts (per `feedback_spec_decode_gate_needs_multi_prompt`).
