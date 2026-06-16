# A1 Attention At-Scale Microbench

## Context

A1 already had finite-diff coverage for Qwen3.6 head_dim=256 and a CUDA
device-residency gate for `dq/dk/dv`. The remaining gap before moving to A2 was
scale evidence: the production-shape attention op needed a cheap wall-clock
microbench that fails if the CUDA gradients fall back to host.

## What Worked

Added `crates/train/examples/bench_attention_a1.rs`, a focused harness around
`causal_sdpa_recompute(q, k, v) -> mul(probe) -> sum -> backward_profiled`.
On CUDA it asserts `q/k/v` gradients keep device handles before the final
readback/reporting path.

Local CPU smoke:

```text
cargo run -p train --release --example bench_attention_a1 --no-default-features --features no-cuda
a1_attention_bench backend=cpu batch=1 heads=2 seq=128 head_dim=256 warmup=1 repeats=3 avg_forward_seconds=0.003521 avg_backward_seconds=0.007199 avg_attention_backward_seconds=0.007088
```

Remote H20 CUDA gate on `.62` GPU4:

```text
source=/data01/arle-a1-bench-current
target=/data01/arle-target-a1-bench-noflash-20260617
log=/data01/arle-a1-bench-current/a1_bench_gpu4_20260617.log
env=CUDARC_CUDA_VERSION=12090 CUDA_HOME=/usr/local/cuda ARLE_CUDA_DISABLE_FLASHMLA=1 INFER_TILELANG_PYTHON=/root/tl-venv/bin/python
```

FlashMLA was disabled for this build because vendor FlashMLA failed on the pod
toolchain (`__nv_fp8_e8m0` / inline asm compile errors). This bench does not use
FlashMLA; it exercises the autograd CUDA causal-SDPA recompute path.

| batch | heads | seq | head_dim | avg forward ms | avg backward ms | avg attention backward ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 128 | 256 | 0.911 | 0.485 | 0.412 |
| 1 | 16 | 256 | 256 | 1.771 | 0.679 | 0.599 |
| 1 | 16 | 512 | 256 | 3.520 | 1.600 | 1.416 |
| 1 | 16 | 1024 | 256 | 8.014 | 5.026 | 4.805 |
| 1 | 16 | 2048 | 256 | 38.199 | 20.799 | 20.406 |

## Verification

```text
rustfmt --check crates/train/examples/bench_attention_a1.rs
cargo run -p train --release --example bench_attention_a1 --no-default-features --features no-cuda
cargo clippy -p train --release --example bench_attention_a1 --no-default-features --features no-cuda -- -D warnings
```

Remote build:

```text
ARLE_CUDA_DISABLE_FLASHMLA=1 INFER_TILELANG_PYTHON=/root/tl-venv/bin/python CUDARC_CUDA_VERSION=12090 CUDA_HOME=/usr/local/cuda CARGO_TARGET_DIR=/data01/arle-target-a1-bench-noflash-20260617 CUDA_VISIBLE_DEVICES=4 ARLE_CUDA_TEST_DEVICE=0 cargo build -p train --release --no-default-features --features cuda --example bench_attention_a1
```

## Limits

This closes the A1 scale/residency evidence for the current recompute
composition. It is still not a native FlashAttention backward kernel and does
not by itself prove full 35B OPD step throughput. A2 can proceed because the
next named blocker is TP gradient aggregation, not attention gradient
correctness.

## Rule

Do not promote a finite-diff-only gradient milestone to a training-runtime
milestone. Pair the numeric gate with a scale bench and an explicit
device-residency assertion at the op boundary.
