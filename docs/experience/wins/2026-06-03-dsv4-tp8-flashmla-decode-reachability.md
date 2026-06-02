# DSv4 TP8 FlashMLA Decode Reachability

## Goal

Move ARLE toward the matched target workload:
DSv4-Flash, TP8, EAGLE, 256K/1500, hot GPU cache, with the SGLang reference
lane at about TTFT 0.44s, TPOT 4.85ms, E2E 7.7s, and 196 output tok/s.

## Hypothesis

ARLE's FlashMLA decode path was structurally unreachable at TP8 because the
MODEL1 decode shape gate checked `local_heads`, which is 8 for a 64-head model
under TP8. SGLang-style FlashMLA sees global `h_q=64` by all-gathering Q across
TP ranks, then slices the rank-local output slab back after the kernel.

## Params

Code tranche only:

- batch HCA FlashMLA decode computes `h_q = local_heads * tp_world`;
- TP>1 all-gathers `q_prepared`, repacks rank-major Q into
  `[B, h_global, d]`, runs FlashMLA at global `h_q`, and slices back to local
  heads;
- single-token CSA/HCA FlashMLA decode uses the same TP-aware all-gather,
  repack, and output-slice pattern;
- output projection remains the validated per-row path.

## Env

Local checks on macOS with no CUDA runtime execution:

- `cargo fmt --check`
- `git diff --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`

## Results

Local checks passed. Remote CUDA build, decode correctness, operator-trace
reachability, and target workload TPOT are pending.

## Problems

This tranche intentionally allocates TP all-gather/repack/full-output scratch in
the decode body. That is acceptable for reachability and correctness, but it is
not the final high-performance CUDA-graph-compatible implementation.

## Learnings

Do not gate FlashMLA decode by the rank-local head count. The correct contract is
the head count passed to FlashMLA after any TP request-ownership transform.
