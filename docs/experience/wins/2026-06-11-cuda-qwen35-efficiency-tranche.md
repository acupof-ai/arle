# Qwen3.5/3.6 CUDA efficiency tranche: workspace reuse, chunk floor, zero-copy loads, MoE scratch

**Date:** 2026-06-11. **Backend:** CUDA, Qwen3.5/3.6 (single GPU + TP=2).
**Scope:** `infer-cuda` (workspace.rs new, qwen35.rs, moe.rs, loader.rs,
executor.rs, lib.rs), `infer-api/loaded.rs`.
**Status: pending-remote** — same-binary before/after decode tok/s + prefill
TTFT A/B on the pod (baseline `fc5b48ed` numbers: TP=1 36.0 tok/s decode,
TP=2 45.5; needle 3k prefill 9.85 s at chunk=64).

## Context

The 2026-06-11 five-lane audit quantified the orchestration overhead around a
numerically-licensed forward: ~425 fresh device allocations per forward call
(3 `HiddenStates::zeros` per layer × 40 + per-full-attn + tails — audit
QW-KV-08), CUDA Qwen kinds inheriting the Metal-tuned `chunked_prefill_size:
64` (≈32× tick/launch overhead vs 2048 at equal KV-read volume — QW-KV-07),
~60 GiB of avoidable host memcpy loading stacked experts (`to_vec` per tensor
— MOE-P2-1), and ~10 fresh device buffers per MoE layer per step (MOE-P3-1).
Roofline framing: decode measured 36 tok/s vs ~750 tok/s HBM ceiling (4.8%) —
host orchestration, not kernels, is the binding constraint.

## What Worked

- **`workspace.rs`** — exact-shape buffer slots: reuse skips alloc+memset when
  the shape matches; shape change re-allocates zeroed (byte-identical to the
  old per-call `*::zeros`). Exact-shape (NOT capacity-arena) is deliberate:
  `TpRuntime::all_reduce_sum` derives the collective message length AND the
  one-shot-vs-NCCL path choice from `CudaSlice::len()`, so capacity-sized
  buffers would change TP≥2 reduction semantics. Decode steady-state hits
  zero allocations; chunked prefill re-allocates once per shape flip.
- **MoE scratch** — `MoeForwardScratch` + `moe_forward_into`: router logits /
  pack buffers / expert outs persist across layers and steps; the dense-path
  `moe_forward` wrapper keeps DSv4/dense byte-identical.
- **Chunk floor** — CUDA `Qwen3Dense|Qwen3Moe` floor `chunked_prefill_size`
  at 2048 (Metal keeps 64 — its single-threaded encode-loop interactivity
  tune; DSv4 keeps its 4096 override).
- **Zero-copy stacked-expert load** — read-once `Rc` shard cache +
  `SharedTensor` byte-range views; gate_up/down slices upload straight from
  the cache (no per-tensor `to_vec` of GiB-scale buffers).

## Verification

- Mac: infer-cuda/infer-api checks clean; infer-core 33/33; qwen35-spec
  37/37; clippy: 0 new warnings (3 `rc_buffer` suggestions suppressed with
  rationale — converting `Rc<Vec<u8>>`→`Rc<[u8]>` would re-copy the shard,
  the very copy this removes).
- **pending-remote:** pod same-binary A/B vs `fc5b48ed` — decode tok/s
  (expect the ~425-alloc and 40-roundtrip share of 27.8 ms/token to shrink),
  needle-prefill wall (expect ~chunk-count-bound improvement), plus
  numerics re-gate (smoke ×3 + needle) since allocation lifetime changed.

## Rule

- Buffer reuse caches must preserve every consumer-visible property of the
  allocation they replace — including `len()`-derived collective message
  sizes and path selection, not just contents.
- A lint that suggests an allocation-shape change (rc_buffer) can directly
  contradict a zero-copy optimization; suppress with the formula, don't obey.
