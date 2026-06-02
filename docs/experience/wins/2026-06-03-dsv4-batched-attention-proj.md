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

Remote verification on `/data01/build/arle`, final checked commit
`62c06c285e5f55626d156940d1fa39c9909b9f81`:

- release-fast build passed in 19.59s. The build used the DSv4 prebuilt CUDA
  artifact fast path and skipped nvcc / TileLang AOT, so this Rust-only tranche
  did not pay a CUDA rebuild.
- Debug-fallback TP8 + EAGLE with FP8 KV, shared KV pool, DeepGEMM experts, and
  MTP weights loaded returned real decode output:
  - `decode64`: HTTP 200, 64 completion tokens, non-empty CUDA graph text.
  - `math32`: HTTP 200, 32 completion tokens, output contained `406`.
  - `fanout=4` decode32: all four requests returned HTTP 200 and non-empty
    text, and the server log showed decode steps with `batch=4`.
- The debug-fallback smoke used `ignore_eos=true` for fixed-token coverage, so
  repeated tail tokens in some outputs are not treated as a quality result.
  They only prove this refactor can process real tokens through the N>1 path.
- High-performance TP8 + EAGLE `ARLE_DSV4_PERFORMANCE_PROFILE=sglang` startup
  still failed closed on the known full-graph blockers: full-decode CUDA graph,
  DeepEP/NCCL collective capture/replay, EAGLE/MTP graph replay,
  graph-captured FlashMLA/SWA/C4/C128 metadata replay, and the remaining
  row-looped cache/metadata attention core.
- The startup blocker text was tightened after this validation so future logs
  state the precise remaining gap: the cache/metadata attention core still
  loops per row, while Q/K/V projections are already batched.
- After both probes, no `infer` process remained and `nvidia-smi` reported no
  compute apps.

Artifacts:

- `/tmp/dsv4_batched_attn_proj_20260603/build.log`
- `/tmp/dsv4_batched_attn_proj_20260603/debug_fallback_server.log`
- `/tmp/dsv4_batched_attn_proj_20260603/debug_fallback_smoke.log`
- `/tmp/dsv4_batched_attn_proj_20260603/startup_tp8_eagle_sglang.log`

## Rule

Only compare ARLE against the 4.85ms TPOT target after the full DSv4 target
workload is runnable. A projection-only batching win is graph-enablement
evidence, not an end-to-end throughput result.
