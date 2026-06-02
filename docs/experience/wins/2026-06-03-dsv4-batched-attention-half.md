# DSv4 Batched Decode Attention Half

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, 256K/1500, hot GPU cache:
TTFT ~0.44s, TPOT ~4.85ms, E2E ~7.7s, output throughput ~196 tok/s.

ARLE is not at that target yet. This tranche is a graph-enablement and launch
reduction step, not a performance claim. The high-performance route still needs
full decode CUDA graph replay, DeepEP/NCCL capture safety, EAGLE/MTP graph
replay, and full batched FlashMLA attention before it can be compared against
the target number.

## What Worked

`compute_top_level_logits_incremental_batch` now batches the row-independent
attention half work across the active decode rows:

- MHC parameter generation for the attention half.
- HC pre projection from stream space.
- RMSNorm on the attention input.
- HC post projection back to stream space.

The per-slot KV attention core remains row-looped. That keeps the existing
per-slot cache ownership path intact while preparing a batch-shaped
`attn_normed` and `attn_out` surface for the later full batched FlashMLA wiring.

The scratch layout was adjusted to remove the old `row_in` / `row_out`
temporary stream buffers and add explicit batch attention scratch:
`attn_pre`, `attn_normed`, `attn_out`, and single-row views for the remaining
row-looped KV core.

## Verification

Local checks:

- `cargo fmt --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`
- `git diff --check`

Remote verification is pending. Required before treating this as landed in the
DSv4 execution path:

- release-fast CUDA build on `/data01/build/arle`.
- Debug-fallback real decode smoke with non-empty, non-degenerate output.
- TP8 + EAGLE high-performance startup probe still fails closed only on the
  known full-graph blockers.
- No `infer` process or GPU compute app left after probes.

## Rule

Batching the attention half is not equivalent to full batched attention. Do not
claim progress toward the 4.85ms TPOT target until the KV attention core itself
is no longer row-looped and the graph replay contract is validated with real
decode output.
