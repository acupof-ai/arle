# SGLang backend-seam scan → ARLE one-scheduler redesign

## Context

ckl: ARLE can't horizontally scale to HIP / new backends; architecture has
problems; optimize bottom-up; reference SGLang. Source survey via
`arle-upstream-runtime-scan` to extract SGLang's device-neutral-scheduler /
device-specific-execution layering, then audit ARLE's actual coupling.

## What Worked

**Upstream read:** `github.com/sgl-project/sglang@3e681d7` —
`layers/attention/base_attn_backend.py`, `model_executor/forward_batch_info.py`,
`mem_cache/memory_pool.py`, `model_executor/model_runner.py`, attention/ listing.

SGLang splits scheduling from execution with **three narrow contracts**:
`ForwardBatch`+`ForwardMode` (logical plan, data), `AttentionBackend` (~8-method
compute seam: `forward_decode/extend/mixed(q,k,v,layer,forward_batch)` + graph
hooks), `KVCache`/`TokenToKVPool` (KV memory seam). **24** attention-backend
impls — NVIDIA flashinfer/flashmla/trtllm, **AMD aiter/hip_radix/wave**, **Intel
xpu/amx**, triton/torch_native, NPU — all behind **one** scheduler. HIP =
cuda-alike (`_is_hip`, `_use_aiter`): swap attention backend, scheduler unchanged.

**ARLE audit (local truth):** `scheduler/cuda/` 13.7k lines + a *separate*
`MetalScheduler` 1.1k; seam is the `ModelForward` 50-method god-trait (forward +
KV-pool migrate + DeviceContext/DeviceVec + cuda_graph + ep_nccl + async
launch/readback + sampling + spec); scheduler holds `paged_kv_pool: PagedKVPool`
directly. **CUDA-touchpoint audit: only 506/13,638 lines (~3.7%) reference any
CUDA symbol** — the scheduling logic is already backend-agnostic in expression;
coupling is thin and concentrated (tiered-KV 248, KV pool 120, exec-ctx 71, rest
<65). The fix is threading ~500 touchpoints through 3 traits, not a rewrite.

Design + migration plan: [`docs/projects/2026-06-03-backend-seam-redesign.md`](../../projects/2026-06-03-backend-seam-redesign.md).

## Kept / Killed

- **Kept:** 3-seam split (ForwardBatch / AttentionBackend / KVCache); HIP =
  cuda-alike + swap attention backend; orthogonal sampling/comm/graph layers;
  bottom-up sequence (normalize ForwardPlan → KvPool trait → split ModelForward →
  de-CUDA scheduler → collapse Metal → HIP validates).
- **Killed for ARLE:** runtime `device`-string dispatch (ARLE = Rust traits +
  compile-time generics / cargo features); torch device abstraction.

## Rule

Adding a backend must implement a **narrow compute seam + KV-pool trait**, never a
new scheduler. When the seam is a god-trait or the scheduler holds a backend
concrete type, that is the architecture bug — measure the actual touchpoint % before
assuming a port is a rewrite (here: 3.7%, not 100%). Source survey is hypothesis-grade;
license L1+ with the open hypotheses in the project doc (tiered-KV placement, async
overlap behind the seam, CUDA/Metal plan field diff).

## Status

Docs-only scan; no runtime change → **bench-exempt** (per CLAUDE.md §Benchmarks).
Next patch boundary: L0.2 (extract `KvPool` trait), Mac-typecheckable.
