# PF8.3 — W4+FP8 Marlin Prefill Kernel Substrate

## Context

PF8.1 and PF8.2 already landed activation FP8 quantization and INT4 weight
preprocessing. PF8.4 added the opt-in dispatch stub
`INFER_MARLIN_W4_FP8_PREFILL=1`, but the real GEMM was still missing.

This entry records the PF8.3 substrate only: a sm_89 W4-weight + FP8-activation
Marlin prefill GEMM wired through the existing hybrid W4 checkpoint path. The
throughput and PPL license gates remain assigned to PF8.5.

## What Worked

- Added `crates/cuda-kernels/csrc/gemm/marlin_w4_fp8_kernel.cu`, using the
  vLLM/Marlin FP8 template path with the Ada `m16n8k32` FP8 MMA instruction.
- Added a local `marlin_pf8/` template shard with explicit Apache-2.0/vLLM
  attribution and a minimal scalar-type shim so `cuda-kernels` does not depend
  on vLLM, Torch, or ATen.
- Added `gemm_w4_fp8_marlin_cuda` FFI and a `DeviceMatrix` hybrid W4 sidecar
  holding PF8.2-preprocessed U4B8 weights.
- Replaced the PF8.4 dispatch bail with
  `run_marlin_w4_fp8_prefill(...)`, still gated behind
  `INFER_MARLIN_W4_FP8_PREFILL=1` and still restricted to hybrid W4 prefill.
- Decode remains on the existing W4A16/W4A8 paths; PF8 does not touch decode.

## Verification

Raw smoke output from `/tmp/pf83_mma_smoke.cu`:

```text
Status::Success fp8 m16n8k32 smoke out=0.000000
```

Build and test gates:

```bash
cargo fmt --all --check
git diff --check
CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 TORCH_CUDA_ARCH_LIST=8.9 \
  INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  cargo check --release -p infer --features cuda
CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 TORCH_CUDA_ARCH_LIST=8.9 \
  INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
  cargo clippy --release -p infer --features cuda --lib -- -D warnings
INFER_MARLIN_W4_FP8_PREFILL=1 INFER_HYBRID_W4A8_PREFILL=1 \
  INFER_TEST_W4A8_MODEL_PATH=/home/ckl/projects/arle/infer/models/Qwen3-4B-W4-hybrid-zpfix \
  cargo test --release -p infer --features cuda --test greedy_consistency \
  test_greedy_w4a8_marlin_optional -- --nocapture
INFER_MARLIN_W4_FP8_PREFILL=1 INFER_HYBRID_W4A8_PREFILL=1 \
  INFER_TEST_W4A8_MODEL_PATH=/home/ckl/projects/arle/infer/models/Qwen3-4B-W4-hybrid-zpfix \
  cargo test --release -p infer --features cuda --test e2e \
  test_e2e_w4a8_marlin_optional -- --nocapture
```

All gates passed.

`codex review --uncommitted` caught five pre-commit issues, all fixed before
landing:

- Parallel-M launch consumed multiple M-block groups but advanced the loop by
  one group, which could issue an extra out-of-range launch for larger M.
- The PF8 wrapper was raising `max_par` after Rust had sized the lock workspace,
  creating a potential workspace underrun; the wrapper now honors the caller's
  workspace contract.
- Hybrid W4 graph capture now excludes PF8 prefill while PF8 still owns per-call
  quant/reduce scratch; a later scratch-hoist tranche can re-enable capture.
- The PF8 kernel now instantiates FP16 output/scales to match the existing
  `.marlin_scales` tensor, then converts the FP16 output scratch back to BF16.
- PF8 weight preprocessing is gated behind `INFER_MARLIN_W4_FP8_PREFILL=1`, so
  default hybrid W4 loads do not pay the extra sidecar allocation or preprocess.

## Bench Status

Status: `pending-pf8.5`.

No throughput license is claimed in this entry. PF8.5 must run the
`INFER_MARLIN_W4_FP8_PREFILL=1` A/B with n=3, sigma under 5%, plus the PPL
gate. License target remains TTFT p50 at least 8% better for prefill-heavy
traffic without decode/ITL regression.

## Rule

For architecture-level quant work, land the hardware substrate only after a
real kernel smoke and a model-path correctness smoke. Keep the perf claim
separate until the production-shape bench and PPL gates run.
