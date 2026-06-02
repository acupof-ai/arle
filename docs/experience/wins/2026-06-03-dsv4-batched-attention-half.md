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

Remote verification on `/data01/build/arle`, final checked commit
`f409a0a40b85d6edb0ae1e791f832689d8ed721d`:

- release-fast CUDA build passed. The first remote build for this Rust-only
  tranche used the prebuilt fast path and completed in 18.6 s; the later
  DeepGEMM JIT-lock tranche rebuilt CUDA in 6m57s and refreshed the prebuilt
  archive.
- Debug-fallback TP8 + EAGLE with FP8 KV, shared KV pool, DeepGEMM experts, and
  MTP weights loaded returned real decode output:
  `decode64` produced 64 completion tokens, `math32` produced `406`, and
  `fanout=4` decode requests all returned non-empty, non-degenerate text.
- A cold `DG_JIT_CACHE_DIR` 1K prefill smoke also returned HTTP 200 with
  `prompt_tokens=1016`, `completion_tokens=1`, proving the decode correctness
  smoke was not relying on a broken long-prompt path.
- TP8 + EAGLE `ARLE_DSV4_PERFORMANCE_PROFILE=sglang` startup still failed
  closed on the known full-graph blockers: full-decode CUDA graph,
  DeepEP/NCCL collective capture/replay, EAGLE/MTP graph replay,
  graph-captured FlashMLA/SWA/C4/C128 metadata replay, and batched decode
  attention still looping per row.
- After the probes, no `infer` process remained and `nvidia-smi` reported no
  compute apps.

Artifacts:

- `/tmp/dsv4_batched_attn_half_20260603/build.log`
- `/tmp/dsv4_deepgemm_jit_lock_20260603/build.log`
- `/tmp/dsv4_deepgemm_jit_lock_20260603/cold_jit_eagle_smoke.log`
- `/tmp/dsv4_deepgemm_jit_lock_20260603/cold_jit_eagle_smoke.json`
- `/tmp/dsv4_deepgemm_jit_lock_20260603/startup_tp8_eagle_sglang.log`

## Rule

Batching the attention half is not equivalent to full batched attention. Do not
claim progress toward the 4.85ms TPOT target until the KV attention core itself
is no longer row-looped and the graph replay contract is validated with real
decode output.
