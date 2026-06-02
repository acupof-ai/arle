# DSv4 Batched Attention Output Projection

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, 256K/1500, hot GPU cache:
TTFT ~0.44s, TPOT ~4.85ms, E2E ~7.7s, output throughput ~196 tok/s.

ARLE is not at that target yet. This tranche does not make the high-performance
`sglang` profile runnable by itself. It removes another per-row section from
batched decode for compressed attention layers.

## What Worked

In `compute_top_level_logits_incremental_batch`, compressed attention layers no
longer run attention output projection and attention all-reduce once per row.

The row loop now stops at `local_attn` for compressed layers. After the row loop,
the batched path runs:

- `wo_a` over `[B, local_attn_width]`;
- `wo_b` over `[B, output_latent]`;
- one `attn_all_reduce` over `[B, hidden]`.

Sliding-window-only layers keep the existing projected per-row path for this
tranche. That keeps the first behavioral change narrow: compressed layers get a
real batch projection/all-reduce path; SWA refactoring remains separate.

## Verification

Local checks:

- `cargo fmt --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`
- `git diff --check`

Remote verification is pending for this entry at the time of writing.

## Rule

When removing DSv4 row-loop work, first split the output contract cleanly:
attention core may return `local_attn`, but output projection and collectives
must run as one batched stage before claiming per-token launch or collective
reduction.
