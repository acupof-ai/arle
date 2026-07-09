# CUDA greedy consistency fixed with deterministic per-row BF16 GEMM

## Context

Track A investigated the deferred B=1 vs B=3 Qwen3-4B greedy split from
`docs/experience/errors/2026-04-13-batched-decode-high-concurrency.md`.
CUDA Graph replay bounds had already been fixed, but graph-off decode still
split into coherent but different continuations.

## What Worked

First-divergence tracing showed the first token split at generated token index
4. The first numeric difference was not in TileLang attention: layer 0 Q
projection differed by one BF16 LSB before `decode_prep_paged`, then q_norm /
RoPE amplified it and logits diverged.

The fix makes `INFER_DETERMINISTIC=1` route BF16 batched dense GEMM through
per-row graph-safe N=1 calls. That gives B=3 rows the same numeric path as solo
B=1 decode while remaining CUDA Graph capturable.

## Verification

Passed:

```bash
CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
cargo test --release -p infer --features cuda --test greedy_consistency -- --nocapture
```

CUDA Graph was enabled. Solo and concurrent generated token IDs matched exactly.

## Bench

Status: pending follow-up bench. This is a correctness-first deterministic mode
change in the BF16 decode GEMM path; throughput impact should be measured
against the latest CUDA guidellm baseline before making deterministic mode a
default serving policy outside parity tests.

## Rule

For exact greedy parity tests, skipping cublasLt autotune is insufficient.
Batch-invariant numerics require the same effective GEMM shape and API path per
request row.
