# Qwen3.6 27B FP8 Loader Adapter Phase 0

## Goal

Type: diagnosis / substrate bring-up. Start Qwen3.6 ModelScope adaptation with normal safetensors and quantized non-GGUF safetensors, using the smallest available Qwen3.6 target when no 9B checkpoint is available.

## Hypothesis

Qwen/Qwen3.6-27B and Qwen/Qwen3.6-27B-FP8 can reuse the existing Qwen3.5 dense hybrid runtime because their configs advertise `Qwen3_5ForConditionalGeneration`; the missing piece for FP8 is ModelOpt-style `quant_method=fp8` plus `.weight_scale_inv` load-time dequantization.

## Command

```bash
cargo test -p infer --no-default-features --features no-cuda parse_fp8_modelopt_quant_method -- --nocapture
CUDARC_CUDA_VERSION=13010 cargo check -p infer --no-default-features --features cuda,no-cuda
CUDARC_CUDA_VERSION=13010 NVCC_CCBIN=/usr/bin/g++-14 INFER_TILELANG_PYTHON=$PWD/.venv/bin/python TORCH_CUDA_ARCH_LIST=8.9 cargo check -p infer --features cuda
CUDARC_CUDA_VERSION=13010 NVCC_CCBIN=/usr/bin/g++-14 INFER_TILELANG_PYTHON=$PWD/.venv/bin/python TORCH_CUDA_ARCH_LIST=8.9 cargo test -p infer --features cuda --lib fp8_weight_scale_inv -- --nocapture
cargo clippy -p infer --no-default-features --features no-cuda -- -D warnings
CUDARC_CUDA_VERSION=13010 NVCC_CCBIN=/usr/bin/g++-14 INFER_TILELANG_PYTHON=$PWD/.venv/bin/python TORCH_CUDA_ARCH_LIST=8.9 cargo clippy -p infer --features cuda -- -D warnings
```

## Environment

- Host: CachyOS Linux, CUDA 13.2 install with `CUDARC_CUDA_VERSION=13010`.
- GPU: local RTX 4070 Ti SUPER / SM89 for compile and unit validation.
- Model target: ModelScope `Qwen/Qwen3.6-27B` normal BF16 safetensors and `Qwen/Qwen3.6-27B-FP8` quantized safetensors.
- Commit: lands with this entry.

## Results

Source evidence:

- ModelScope exact probes found `Qwen/Qwen3.6-27B`, `Qwen/Qwen3.6-27B-FP8`, `Qwen/Qwen3.6-35B-A3B`, and `Qwen/Qwen3.6-35B-A3B-FP8`; no official 9B Qwen3.6 normal/FP8/GPTQ/AWQ/Int4/Int8 repo was found in the same probe set.
- `Qwen/Qwen3.6-27B` config: `architectures=["Qwen3_5ForConditionalGeneration"]`, `model_type="qwen3_5"`, 64 dense hybrid layers, vocab 248320.
- `Qwen/Qwen3.6-27B-FP8` config: same architecture plus `quantization_config.quant_method="fp8"`, `fmt="e4m3"`, `weight_block_size=[128,128]`.
- `Qwen/Qwen3.6-27B-FP8` shard header sample: linear/MLP projection weights are `F8_E4M3`; matching `.weight_scale_inv` tensors are BF16 block scales. Outside tensors used by the language runtime (`embed_tokens`, `lm_head`, final norm) are BF16.

Local validation:

- PASS: FP8 ModelOpt config parser test.
- PASS: `cargo check -p infer --no-default-features --features cuda,no-cuda`.
- PASS: `cargo check -p infer --features cuda` with CUDA env above.
- PASS: CUDA feature unit tests for E4M3 + BF16 `weight_scale_inv` load-time dequantization, including indexed and single-file safetensors lookup.
- PASS: no-cuda clippy with `-D warnings`.

Implementation result:

- Normal BF16 Qwen3.6-27B remains on the existing safetensors loader.
- FP8 Qwen3.6-27B now parses `quant_method=fp8`, validates `fmt=e4m3`, carries `weight_block_size`, and dequantizes `F8_E4M3` matrices with `.weight_scale_inv` into BF16 `DeviceMatrix` at load time.
- Qwen3.5/Qwen3.6 single-rank linear-attention QKV loading now routes through the quant-aware loader; TP remains on the fused segment loader.
- New CUDA kernel count: 0.

## Problems

- Full-model smoke is pending-H20. This local 16 GB GPU cannot validate Qwen3.6-27B-FP8 after load-time dequantization because the adapter intentionally materializes BF16 matrices for correctness-first bring-up.
- `cargo clippy -p infer --features cuda -- -D warnings` is blocked on pre-existing CUDA-main warnings outside this tranche, primarily DSv4 unused imports/dead code and scheduler/runtime style lints. The Qwen3.6-specific no-cuda clippy gate passes.
- FP8 TP sharded load is still unsupported by the existing Qwen3.5 runtime guard for quantized TP. This tranche is single-rank loader substrate, not H20x8 tensor-parallel enablement.

## Learnings

- Qwen3.6 dense 27B is the smallest non-GGUF Qwen3.6 adaptation target currently found on ModelScope; the 35B-A3B line adds MoE and is a larger follow-up.
- Official Qwen3.6 FP8 uses ModelOpt `quant_method=fp8` plus `weight_scale_inv`, not GGUF or Marlin. Correct first step is a loader adapter, not a new GEMM kernel.
- Load-time BF16 dequant is a correctness substrate only. Performance work should add native FP8 matmul after full-model correctness is validated on the target H20 class machine.

## Delta vs Baseline

First Qwen3.6 ModelScope non-GGUF loader entry. No throughput benchmark yet; status is pending-H20/full-model-smoke.
