# DSv4 FlashMLA Decode TP Workspace

## Context

The matched target lane is DSv4-Flash, TP8, EAGLE, CUDA graph, 256K/1500, hot
GPU cache. The reference target is about TTFT 0.44s, TPOT 4.85ms, E2E 7.7s, and
196 output tok/s.

The previous TP8 FlashMLA decode tranche made the path reachable, but it still
allocated Q all-gather, packed-Q, and full-output buffers inside the decode
body. That is not graph-compatible and adds pure hot-path overhead before any
kernel optimization can be trusted.

## What Worked

Move TP FlashMLA decode scratch out of the per-step body:

- batch HCA FlashMLA decode now reuses scheduler-owned arena buffers for
  gathered Q, packed Q, and full output;
- single-token CSA/HCA FlashMLA decode now reuses per-slot attention-cache arena
  buffers for the same TP scratch;
- NCCL BF16 all-gather gained a raw-pointer helper for pre-allocated arena
  memory, avoiding temporary `CudaSlice` ownership churn in the graph-critical
  path.

Local checks passed:

- `cargo fmt --check`
- `git diff --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`

Remote validation is pending. This entry cannot claim TPOT improvement until the
same DSv4 pod rebuilds this commit and passes decode correctness.

## Rule

Decode graph work must first make the hot decode body allocation-free for the
target shape. A reachable kernel path that still allocates per token is only a
correctness milestone, not a high-performance implementation.
