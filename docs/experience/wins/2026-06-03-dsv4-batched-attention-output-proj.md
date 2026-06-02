# DSv4 Batched Attention Output Projection

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, 256K/1500, hot GPU cache:
TTFT ~0.44s, TPOT ~4.85ms, E2E ~7.7s, output throughput ~196 tok/s.

ARLE is not at that target yet. This tranche does not make the high-performance
`sglang` profile runnable by itself. It removes another per-row section from
batched decode for compressed attention layers.

## What Worked

In `compute_top_level_logits_incremental_batch`, compressed attention layers no
longer run attention output projection once per row.

The row loop now stops at `local_attn` for compressed layers. After the row loop,
the batched path runs:

- `wo_a` over `[B, local_attn_width]`;
- `wo_b` over `[B, output_latent]`;
- exact-row `attn_all_reduce` over each `[1, hidden]` row.

The row all-reduce is intentional in this tranche: the current
`LayerCommunicator` NCCL path rejects oversized capacity buffers where
`CudaSlice::len()` does not match the logical `[B, hidden]` length. Keeping the
collective on exact row buffers preserves batched/fanout correctness while
isolating the next optimization to the all-reduce slice contract.

Sliding-window-only layers keep the existing projected per-row path for this
tranche. That keeps the first behavioral change narrow: compressed layers get a
real batch projection/all-reduce path; SWA refactoring remains separate.

## Verification

Local checks:

- `cargo fmt --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`
- `git diff --check`

Remote build:

- `/tmp/dsv4_batched_attn_out_proj_20260603/build.log` (`release-fast`, passed
  before the exact-row all-reduce fix)

Remote correctness verification is pending for the exact-row all-reduce follow
up at the time of writing.

## Rule

When removing DSv4 row-loop work, first split the output contract cleanly:
attention core may return `local_attn`, but output projection and collectives
must each satisfy their runtime buffer contracts before claiming per-token
launch or collective reduction.
