# Multi-GPU F0 Foundation — Pending Remote Verification

Status: `pending-remote` (per CLAUDE.md §Benchmarks "If the bench can't
run locally")
Date: 2026-04-28
Author: claude+ckl

## Context

Landed the F0 foundation of the multi-GPU plan
([`docs/plans/2026-04-28-single-node-multi-gpu.md`](../../plans/2026-04-28-single-node-multi-gpu.md)).
F0 is architecture-only — no caller in the serving path yet wires up
NCCL, sharded weights, or per-rank threads. F1 (TP dense) is the first
phase that exercises these surfaces end-to-end.

Six commits, all on a Mac dev box without CUDA/NCCL:

1. `feat(cuda): explicit device ordinal via INFER_CUDA_DEVICE` (M0)
2. `feat(infer): multi-axis rank-layout math (SGLang port)` (24 tests)
3. `feat(infer): TCP rendezvous for unique-id broadcast` (5 tests)
4. `feat(cuda): NCCL FFI + CollectiveBackend trait skeleton`
5. `feat(qwen3,qwen35): per-tensor Shard annotations for TP loader` (32 tests)
6. `docs(plans): multi-GPU full-parallelism plan + F0 verification guide`

## Why pending-remote

The Mac dev box has neither nvcc nor libnccl. Mac-side gates that pass:

- `cargo check -p infer --no-default-features --features cuda,no-cuda` clean
- `cargo check -p cuda-kernels --no-default-features --features cuda,nccl,no-cuda` clean
- `cargo test -p infer --release tensor_parallel` — 24/24 pass
- `cargo test -p infer --release distributed::init_method` — 5/5 pass
- `cargo test -p qwen3-spec -p qwen35-spec --release` — 32/32 pass
- `cargo clippy -p cuda-kernels -p qwen3-spec -p qwen35-spec` clean

CUDA-host gates that ckl runs (full procedure in
[`2026-04-28-multi-gpu-f0-verification.md`](../../plans/2026-04-28-multi-gpu-f0-verification.md)):

- `cargo build --release --features cuda` (regression on existing path)
- `cargo build --release --features cuda,nccl` (FFI + linker)
- M0 behavior parity: `INFER_CUDA_DEVICE` unset vs `=0` within ±2% on
  `scripts/bench_guidellm.sh`
- M0 pinning: `INFER_CUDA_DEVICE=1` actually binds to GPU 1 (verify via
  `nvidia-smi` during the run)

## Expected result

This phase is a no-op for the single-GPU serving path — same kernel
launches, same KV pool, same scheduler tick. Bench numbers should
match the most recent CUDA L4 baseline (122 tok/s c=16 on Qwen3-4B per
[2026-04-27-bench-guidellm-cuda-l4-budget-fix.md](2026-04-27-bench-guidellm-cuda-l4-budget-fix.md))
within ±2%.

If they don't, M0 is not actually a no-op — investigate before F1.

## What worked

- **Multi-axis rank math is verbatim from SGLang** with line-number
  citations in code comments. Two SGLang docstring scenarios
  (`parallel_state.py:1749–1756` and `:1758–1769`) are reproduced as
  unit tests with the exact group lists, so future drift against
  upstream is caught immediately.
- **TCP rendezvous tested with 2-thread and 4-thread localhost
  scenarios** (and a "client retries when server late" scenario). Same
  protocol works unchanged for multi-host once F9 wires
  `MASTER_ADDR:MASTER_PORT` instead of `127.0.0.1:0`.
- **Spec `Shard` annotations live in the spec crates** (not in `infer`),
  so the per-tensor sharding contract sits next to the tensor-name
  contract — single source of truth for both train and infer.
- **Codex review caught the one feature-gating bug**
  (`nccl = []` should imply `cuda`; fixed before push).

## Rule

When a foundation phase ships compile-only / test-only changes on a
non-target dev box, document the verification path explicitly and mark
the wins entry `pending-remote` rather than skipping it. The remote run
is the real "done."
