# DSv4 Batched Decode Attention Projection

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, 256K/1500, hot GPU cache:
TTFT ~0.44s, TPOT ~4.85ms, E2E ~7.7s, output throughput ~196 tok/s.

ARLE is not at that target yet. This tranche removes another row-looped part of
the decode attention path. It is not a performance claim and it does not make
the high-performance `sglang` profile runnable by itself.

## What Worked

`compute_top_level_logits_incremental_batch` now batches the DSv4 decode
attention Q/K/V projection work across active decode rows:

- `wq_a` GEMM over `[N, hidden]`.
- `q_norm` over `[N, c_q]`.
- `wq_b` GEMM over `[N, c_q]`.
- `wkv` GEMM over `[N, hidden]`.
- `kv_norm` over `[N, head_dim]`.

The remaining row loop now extracts the projected per-row tensors and calls the
same `forward_attention_gpu_into` cache-bound core used by the old incremental
path. That keeps per-slot KV cache semantics intact while shrinking the
row-looped region to the FlashMLA/SWA/C4/C128 metadata and cache core.

Trace labeling was tightened at the same time: the row-looped section is now
reported as `attn_core`, while the newly batched projection section is reported
as `attn_proj`.

## Verification

Local checks:

- `cargo fmt --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`
- `git diff --check`

Remote verification is pending for this tranche. Required gate before keeping
the change:

- release-fast build on `/data01/build/arle`.
- Debug-fallback TP8 + EAGLE smoke with FP8 KV and `fanout=4`, proving real
  decode output for batched `N > 1`.
- High-performance TP8 + EAGLE `ARLE_DSV4_PERFORMANCE_PROFILE=sglang` startup
  probe must still fail closed only on the known full-graph blockers, not on
  this projection refactor.
- After probes, no `infer` process and no GPU compute app may remain.

## Rule

Only compare ARLE against the 4.85ms TPOT target after the full DSv4 target
workload is runnable. A projection-only batching win is graph-enablement
evidence, not an end-to-end throughput result.
