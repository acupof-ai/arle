# Path B Phase 1.1 — vLLM Marlin dequant.h Port

## Context

Path B-Phase2' FP8 decode was killed on sm_89 because W4 decode is HBM-bound, not MMA-bound. Phase 1 is the conservative Marlin fallback: port vLLM-current dequant helpers first, then benchmark any follow-up reduction changes separately.

This entry covers Substep 1.1 only.

## What Worked

- Added `crates/cuda-kernels/csrc/gemm/marlin_dequant.h`, adapted from vLLM `csrc/quantization/marlin/dequant.h` under Apache 2.0.
- Kept ARLE integration narrow: `marlin_kernel.cu` now calls `arle::marlin::dequant<half2, arle::marlin::vllm::kU4B8.id(), false>()` instead of carrying its local inline INT4 unpack sequence.
- Used a local scalar-tag shim instead of importing vLLM headers, so future upstream cherry-picks stay isolated inside `crates/cuda-kernels/csrc/gemm/`.

## Verification

```bash
NVCC_CCBIN=/usr/bin/g++-14 CUDA_HOME=/opt/cuda TORCH_CUDA_ARCH_LIST=8.9 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
cargo build --release -p infer --features cuda
# PASS: Finished release profile in 4m 43s
```

```bash
cargo fmt --all --check
# PASS
```

```bash
NVCC_CCBIN=/usr/bin/g++-14 CUDA_HOME=/opt/cuda TORCH_CUDA_ARCH_LIST=8.9 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
cargo clippy --release -p infer --features cuda -- -D warnings
# PASS: Finished release profile in 3m 47s
```

```bash
NVCC_CCBIN=/usr/bin/g++-14 CUDA_HOME=/opt/cuda TORCH_CUDA_ARCH_LIST=8.9 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
INFER_TEST_MODEL_PATH=/home/ckl/projects/arle/infer/models/Qwen3-4B-GPTQ-W4A16-marlin-zpfix \
cargo test --release -p infer --features cuda --test greedy_consistency \
  test_greedy_solo_vs_concurrent -- --test-threads=1 --nocapture
# PASS: 1 passed; 0 failed; finished in 10.83s
```

Manual output inspection from the W4A16 targeted greedy run remained coherent:

```text
" about a boy who is a dragon tamer, and he is on a quest to find a dragon egg. The story should be in the style of"
```

Full `greedy_consistency` still fails in `test_w4a8_vs_bf16_token_diff`; that is the existing W4A8 accuracy gate, not this W4A16 Marlin dequant path. The targeted W4A16 path passed.

## Bench Status

No throughput license is claimed in this substep. The port is a correctness substrate for Path B Phase 1; A/B GuideLLM benchmarking should be run after the next scoped performance change.

Substep 1.2 needs re-scope before implementation: current W4A16 `marlin_kernel.cu` uses the output buffer plus lock workspace for global reduction, while the `max_par * 64 * n` INT32 reduce buffer is on the W4A8 kernel path. The pre-drafted atomic-add brief should not be applied to W4A16 as written.

## Rule

For upstream Marlin ports, keep imported implementation details behind a local namespace shim and prove the exact quant path with a targeted checkpoint. Do not let unrelated W4A8 accuracy-gate failures block W4A16-only changes, but document that boundary explicitly.
