# Path B Phase 1.1 — vLLM Marlin dequant.h Port

## Context

Path B-Phase2' FP8 decode was killed on sm_89 because W4 decode is HBM-bound, not MMA-bound. Phase 1 is the conservative Marlin fallback: port vLLM-current dequant helpers first, then benchmark any follow-up reduction changes separately.

This entry covers Substep 1.1 only.

## What Worked

- Added `crates/cuda-kernels/csrc/gemm/marlin_dequant.cuh`, adapted from vLLM `csrc/quantization/marlin/dequant.h` under Apache 2.0.
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

This substep includes a one-run local regression check against the closest
published W4A16 baseline (`docs/experience/wins/2026-05-08-m_quant-w4a16-marlin-bench.md`).
The benchmark used the same checkpoint and 4k/c=4 shape as that baseline.

Server:

```bash
CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
./target/release/infer \
  --model-path infer/models/Qwen3-4B-W4A16-sym-g128-marlin \
  --port 8000 --num-slots 8 --max-seq-len 5120
```

Bench:

```bash
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
scripts/bench_guidellm.sh path-b-p1-newdequant-r1 \
  --model Qwen3-4B-W4A16-sym-g128-marlin \
  --processor infer/models/Qwen3-4B \
  --concurrencies 4 --max-seconds 120 --warmup 10 \
  --data 'prompt_tokens=4096,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_min=256,output_tokens_max=256'
```

| Metric | 2026-05-08 W4A16 baseline median/mean | This run | Delta |
|---|---:|---:|---:|
| TTFT p50 | 2565.4 ms | 2386.3 ms | -7.0% |
| ITL p50 | 11.76 ms | 11.38 ms | -3.2% |
| out tok/s | 191.16 | 195.17 | +2.1% |

Raw artefacts:

- `bench-output/2026-05-10-path-b-p1-newdequant-r1/benchmarks.json`
- `bench-output/2026-05-10-path-b-p1-newdequant-r1/benchmarks.csv`
- `bench-output/2026-05-10-path-b-p1-newdequant-r1/headline_table.md`
- `bench-output/2026-05-10-path-b-p1-newdequant-r1/service_stats_trace_summary.md`

This is not an n=3 license run, but it clears the required local regression
check: the dequant header port does not regress W4A16 Marlin throughput or TTFT
on the canonical 4k/c=4 shape.

Substep 1.2 needs re-scope before implementation: current W4A16 `marlin_kernel.cu` uses the output buffer plus lock workspace for global reduction, while the `max_par * 64 * n` INT32 reduce buffer is on the W4A8 kernel path. The pre-drafted atomic-add brief should not be applied to W4A16 as written.

## Rule

For upstream Marlin ports, keep imported implementation details behind a local namespace shim and prove the exact quant path with a targeted checkpoint. Do not let unrelated W4A8 accuracy-gate failures block W4A16-only changes, but document that boundary explicitly.
