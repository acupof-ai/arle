# DSv4 shared persistent FP8 decode KV pool — env-gated re-implementation (pending-remote)

**Status: pending-remote.** Mac has no nvcc; CUDA build + pod parity/bench are
separate. Typecheck-only locally. This re-implements the pool reverted in
`9264c83a` (first attempt `4088da43`), corrected for the prefill-bind gap and
gated behind a default-OFF env knob.

## Context

DSv4 FlashMLA FP8 decode allocates its KV pool **per-(slot, layer)** lazily at
first decode, sized to the full `max_position_embeddings / ratio` (~1M / ratio
compressed rows). At c≥8 the `num_slots × layers` separate unbudgeted
`alloc_zeros` pools OOM (~18 GB), blocking concurrent throughput
(`docs/projects/2026-05-29-dsv4-beat-sglang-30pct-campaign.md` I3).

The first shared-pool attempt (`4088da43`) replaced the per-state pool with one
shared bound-view pool and required the (slot, layer) sub-range to be bound
**before** the SW-bootstrap / compressor-pack hooks. But those hooks also run on
the **PREFILL** path — prefill is routed through
`compute_top_level_logits_incremental` by the P→D fix (`token_count > 1`,
`weights.rs:3907`), which reaches `forward_attention_gpu_cached:1740` where the
compressor pack fired unconditionally. Prefill has no decode context to bind
from → "DSv4 FlashMLA FP8 KV pool sub-range not bound" → HTTP 500. Reverted.

## What changed (this commit)

Two hard requirements, both met:

1. **ENV-GATED, DEFAULT OFF** — `ARLE_DSV4_SHARED_KV_POOL` (`weights.rs`,
   `dsv4_shared_kv_pool_enabled`). OFF (default) keeps the per-(slot, layer)
   lazy pool (`ensure_dsv4_flashmla_fp8_kv_pool`), **byte-identical to `main`**.
   ON allocates ONE shared persistent pool in `DeepseekBatchDecodeBuffers`,
   sized `num_slots × layers × slot_blocks × 37376 B`, `comp_blocks` bounded by
   the served `max_seq_len` (not `max_position_embeddings`). Each (slot, layer)
   owns a fixed byte sub-range `[(s*layers + l) * slot_layer_bytes, …)`.

2. **PREFILL NEVER BINDS** — root-cause fix at
   `forward_attention_gpu_cached`. The SW bootstrap was already decode-only
   (`token_count == 1`); the compressor pack fired every step. When the shared
   pool is ON, the compressor pack is now **also gated to `token_count == 1`**:
   idempotent via the `fp8_kv_comp_packed_rows` high-water mark, so the first
   decode packs `[0, compressed_rows)` in one shot before Step 6 reads it
   (byte-identical result, only pack timing moves). Prefill then only writes the
   bf16 SW/compressed buffers and never touches the FP8 pool, so no bind is
   needed at prefill. When OFF, the compressor pack still runs every step
   (lazy-alloc), preserving `main`.

Mode dispatch is centralised in two helpers so OFF stays byte-identical:
- `dsv4_flashmla_fp8_kv_pool_base_ptr`: OFF → `ensure_*` lazy-alloc byte-0 ptr;
  ON → bound sub-range ptr (asserts capacity). The three pack helpers now take a
  `u64` base ptr (they used to derive the same `u64` internally → behaviour-id).
- `dsv4_flashmla_decode_pool_layout`: OFF → `max_position_embeddings`-based
  blocks (same values `ensure_*` stamps); ON → the bound layout stamped at bind.

The bind happens in `forward_decode_batch` (the one site owning both the decode
context and slot identity), covering the N≥2 batched path and the N==1 per-row
fallback. The pool is allocated in `create_decode_context` and reserved in
`scheduler_runtime_workspace_bytes` — both no-ops when OFF / decode knob off / no
layers. Slot reuse resets via `incremental.clear()`.

Files: `infer/src/model/deepseek/{batch_decode,forward,state,weights}.rs`.

## Typecheck (this worktree, Mac)

```
CUDARC_CUDA_VERSION=12090 cargo check -p infer --lib \
  --no-default-features --features cuda,no-cuda,nccl   # clean (0 errors)
```
My new code (the helpers + the 4 mode-dispatched sites + the forward.rs bind
loop / create_decode_context / scheduler_runtime_workspace_bytes) is
warning-clean; remaining clippy warnings in the module are pre-existing.

## Bench to run on pod (8×H20 TP=8, DSv4-Flash) — TWO independent runs ("各自验证")

Working serving config per `wins/2026-05-29-dsv4-true-batched-decode.md`, with
`--num-slots ≥ 8`. Run the **same binary, same shell, same prompt** twice, one
env flip:

**(A) OFF — must equal current `main` (non-negotiable):**
```
INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7 \
ARLE_DSV4_SHARED_KV_POOL=0 \
ARLE_DSV4_LOAD_LAYER_WEIGHTS=1 ARLE_DSV4_GPU_FULL_LAYERS=43 \
ARLE_DSV4_INCREMENTAL_KV=1 ARLE_DSV4_FLASHMLA_PREFILL=1 ARLE_DSV4_FLASHMLA_DECODE=1 \
ARLE_DSV4_MOE_BACKEND=allreduce ARLE_DSV4_EXPERT_BACKEND=native \
  infer serve ... --num-slots 8 --max-seq-len 4096 --mem-fraction-static 0.10 \
  --kv-cache-dtype fp8 --deepseek-distributed-layers 43 --port 18300
python3 scripts/dsv4_batched_decode_validate.py 18300   # expect PARITY_PASS, c=8 completes
```

**(B) ON — c=1/c=4 byte-identical to OFF, AND c=8 stops OOMing (the point):**
```
INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7 \
ARLE_DSV4_SHARED_KV_POOL=1 \
ARLE_DSV4_LOAD_LAYER_WEIGHTS=1 ARLE_DSV4_GPU_FULL_LAYERS=43 \
ARLE_DSV4_INCREMENTAL_KV=1 ARLE_DSV4_FLASHMLA_PREFILL=1 ARLE_DSV4_FLASHMLA_DECODE=1 \
ARLE_DSV4_MOE_BACKEND=allreduce ARLE_DSV4_EXPERT_BACKEND=native \
  infer serve ... --num-slots 8 --max-seq-len 4096 --mem-fraction-static 0.10 \
  --kv-cache-dtype fp8 --deepseek-distributed-layers 43 --port 18301
python3 scripts/dsv4_batched_decode_validate.py 18301   # expect PARITY_PASS, c=8 completes (was OOM)
```

**Gates (both must hold):**
- OFF: `PARITY_PASS`, identical to a `main`-binary baseline at c={1,4,8}.
- ON: c=1/c=4 greedy byte-identical to OFF; c=8 no longer OOMs (was `~18 GB`
  per-state OOM). Capture peak `nvidia-smi` mem ON vs OFF at c=8.

## Rule

A bound-view pool's bind requirement may only be enforced on the path that owns
the binding context. DSv4's pack hooks are shared by prefill-via-incremental and
decode; gate any decode-context-dependent step to `token_count == 1` and keep a
default-OFF env flip so the new path is validated against the working default
("各自验证") before any default flip.
