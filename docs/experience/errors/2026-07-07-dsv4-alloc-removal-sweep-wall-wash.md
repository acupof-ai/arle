# DSv4 decode alloc-removal (commit 1+2) — A/B wall-clock unchanged

> Status: measured — 2026-07-07. Code landed; sweep halted.

## Changes landed
- commit `de6fc4fd`: `forward_decode_batch_stream_impl` shared-expert switched from
  per-layer `dsv4_shared_expert_forward` (`HiddenStates::uninit` + 6 allocs + 4 H2D)
  to `dsv4_shared_expert_forward_decode_scratch` reusing `kv_adapter.shared_expert_scratch`.
- commit `4f589cfb`: `dsv4_moe_forward_decode_fp8` 8 per-layer buffers replaced by
  `Dsv4MoeTailScratch` (band ceiling 128 rows) on the kv_adapter; per step re-init
  counts=0, cursors=0, route_out=0, packed_route_slot=-1.
- Both compiled BUILD_EXIT=0 (cuda,nccl,deepep), clippy-clean.

## Correctness (greedy, TP=4/EP=4, DSv4-Flash-FP8, GPU 4-7, MTP-on)
- commit 1: "capital of France" → reasoning_content "…the answer is straightforward: Paris."
- commit 2: "capital of France" → content "Paris"; "three primary colors" → coherent reasoning.
- No garbage / NaN / empty generation.

## A/B (same prompt, max_tokens=256, temperature=0, TP=4, MTP-on, 3 runs each)
| | runs (s) | mean (s) | tok/s |
|---|---|---|---|
| baseline `c59aab9c` | 5.573 / 5.602 / 5.631 | 5.602 | 45.70 |
| c1+c2 | 5.557 / 5.632 / 5.573 | 5.587 | 45.82 |

Δ mean wall −0.27%. Per-group run-to-run spread ±0.7%.

## nsys (existing `/host/kern141_decode2.nsys-rep`, 07-03, TP=4, MTP-on)
- `cudaLaunchKernel` 39.8% wall, `cuStreamSynchronize` 26.6%, zero `cuGraphLaunch`.
- `cuMemAllocAsync`+`cuMemFreeAsync` 12.2M calls / 7.7% wall; `cuMemsetD8Async` 2.4M / 9.1%.
- `ctx.sync()` per decode step at `crates/infer-cuda/src/ops.rs:467`.

## Disposition
- commit 1+2 kept.
- commit 3 (attn/ffn stream double-buffer + N-ring) not implemented.
- Prior related entries: `errors/2026-06-20-host-launch-bound-misinference-decode-is-foundation-bound.md`,
  `errors/2026-06-08-dsv4-decode-graph-wash-launch-gap-is-framing.md`.
