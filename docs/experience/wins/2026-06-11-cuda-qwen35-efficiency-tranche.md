# Qwen3.5/3.6 CUDA efficiency tranche: workspace reuse, chunk floor, zero-copy loads, MoE scratch

**Date:** 2026-06-11. **Backend:** CUDA, Qwen3.5/3.6 (single GPU + TP=2).
**Scope:** `infer-cuda` (workspace.rs new, qwen35.rs, moe.rs, loader.rs,
executor.rs, lib.rs), `infer-api/loaded.rs`.
**Status: LICENSED** (pod A/B 2026-06-11, single-variable vs `fc5b48ed`,
same box/driver/shapes, n=1 each — see Verification).

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
- **Pod A/B vs `fc5b48ed`** (H20, contention-free, same drivers/shapes;
  n=1 each — directionally above the 10% license bar at TP=2, TP=1 within
  it; multi-run σ deferred):

  | metric | fc5b48ed | 1e0f05e1 | Δ |
  |---|---|---|---|
  | decode tok/s TP=1 (gen128) | 36.01 | 39.39 | **+9.4%** |
  | decode tok/s TP=2 (gen128) | 45.47 | 57.52 | **+26.5%** |
  | needle 3k wall TP=1 | 9.85 s | 8.08 s | **−18.0%** |
  | needle 3k wall TP=2 | 5.78 s | 4.70 s | **−18.7%** |

  TP=2 gains more than TP=1: the host-side alloc/roundtrip work was
  serialized across BOTH ranks by lockstep, so removing it compounds.
- **Numerics re-gate**: smoke ×3 byte-consistent, gen128 head strings
  byte-equal to the fc5b48ed run ("Unit 734 … Seven/Artie"); needle-64
  retrieval PASS ×2 + self-consistent (5.88 s vs 7.29 s pre-tranche). The
  8-token needle probe now enters think-mode at TP=1 too — chunk 64→2048
  shifts MoE routing on knife-edge tokens (expected MoE non-determinism;
  retrieval gate is the licensed check, not token-identity).
- c=2 side-effect: 2k-token concurrent prefills now run ~11 s before the
  known single-row decode-tick engine death — previously they died at
  admission-time instantly. Same known limitation, different timing.

## Rule

- Buffer reuse caches must preserve every consumer-visible property of the
  allocation they replace — including `len()`-derived collective message
  sizes and path selection, not just contents.
- A lint that suggests an allocation-shape change (rc_buffer) can directly
  contradict a zero-copy optimization; suppress with the formula, don't obey.
