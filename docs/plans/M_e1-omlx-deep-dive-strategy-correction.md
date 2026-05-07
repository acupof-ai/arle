# oMLX deep dive — strategic correction for M_e.1 paged-KV — 2026-05-07

After 19 commits of paged-KV preparation (P2.0 → P3.1c.3a' + flush) on
the assumption that "paged-KV unlocks c=4 by replacing left-pad cache
with block-table indirection at SDPA", a deep dive on jundot/omlx
v0.3.9.dev1 (the closest production prior art for Apple-Silicon
serving with paged-KV) found the assumption is wrong about how the
unlock actually works.

## The misalignment

ARLE's M_e.1 plan said:
> Replace `slice_update + slice` on per-state kv_capacity caches
> with `gather_kv` from a token-slot pool, feed the gathered K/V to
> SDPA directly. SDPA reads from pool, kernel cutover skips left-pad.

oMLX does **not** do this. Their "paged" is metadata-only (block hash
+ ref count + SSD bridge for prefix dedup). The Metal SDPA hot path
operates on **contiguous per-batch KV buffers**, exactly like
mlx-lm's `KVCache.update_and_fetch`.

## What oMLX actually does (verified)

Source: `jundot/omlx@main`, files cited inline.

### 1. Paged-cache is bookkeeping, not a runtime KV pool

`omlx/cache/paged_cache.py` (1732 lines) — `CacheBlock`, `BlockTable`,
`PagedCacheManager`, `FreeKVCacheBlockQueue`. Block size 16 (vLLM-style).
**`grep mx. paged_cache.py` returns zero hits.** Pure-Python metadata
that assigns block IDs and tracks ref counts. Never owns mlx tensors.

### 2. Per-step KV write is direct slice-assign on a pre-allocated buffer

mlx-lm's `BatchKVCache.update_and_fetch` (used by oMLX) does:

```python
self.keys[..., prev : self.offset, :] = keys
```

where `keys` shape is `[B, H_kv, T_new, D]` covering the whole batch's
new tokens. The buffer pre-allocates capacity in step=256 increments;
typical decode step = zero allocation, just an in-place slice assign on
already-evaluated storage. Lazy-graph chaining is broken structurally
(by step-boundary eval), not by periodic flushing.

### 3. Eval barriers per step are the contract, not a workaround

oMLX explicitly evals at:
- `scheduler.py:1533` after each chunked prefill: `mx.eval([c.state ...])`
- `scheduler.py:3524, 3536` for sp_cache (speculative)
- `turboquant_kv.py:293, 396` after finalize/merge

`scheduler.py:48-58` documents that `mx.async_eval` **crashes M4
drivers** (issues #300, #888) and warns against it. They use sync
`mx.eval`. ARLE's `pool.flush()` using `async_eval` is on the wrong
side of this boundary — **switch to `eval`**.

### 4. SDPA reads contiguous tensors, no block-table indirection

Decode attention runs against `cache.keys[..., :offset, :]` returned by
`update_and_fetch` — a normal contiguous `[B, H, T, D]`. There is no
paged-attention Metal kernel in oMLX. Block indirection ENDS at the
cache-construction boundary. When SSD prefix is restored,
`prefix_cache.py:1361 reconstruct_cache` rebuilds via
`mx.concatenate(layer_keys, axis=2)` (line 2062) and hands a fresh
contiguous tensor to a normal mlx-lm cache.

### 5. CoW + prefix sharing is metadata-level, not GPU-level

Block hashing (`paged_cache.py:78 compute_block_hash`) →
`BlockHashToBlockMap` for prefix dedup. `_cow_copy_block:1223`
materializes a private copy when shared block needs mutation.
**No GPU memcpy in CoW** — tensors are reconstructed at request-start
from SSD, never shared in GPU memory across live requests.

## Reframing M_e.1 against oMLX

| Original M_e.1 plan | oMLX-correct shape |
|---|---|
| Pool gathers K/V; SDPA reads from pool | Pool is metadata only; SDPA reads contiguous per-batch buffer |
| `gather_kv_rows` indirection at SDPA | `cache.keys[..., :offset, :]` slice |
| `slice_update` per-token write at `[1, kv_dim]` | Slice-assign at `[B, H, 1, D]` per layer per step |
| `pool.flush()` with `async_eval` | `mx.eval` (sync) at step boundary; flush is canonical |
| Block-paged kernel cutover (P3.1c.3d) | **Doesn't exist**; the win lives elsewhere |

Specifically: the c=4 ITL 19→9 ms unlock that the master analysis
attributed to "skipping left-pad via paged-KV" **isn't on this
trajectory**. oMLX runs SDPA on the same shape ARLE does (contiguous
batched KV); their batched cache uses left-pad + per-row offset
(`BatchTurboQuantKVCache.make_mask:237-265`) — same design as
ARLE's `Qwen35PackedDecodeBatch.left_padding`. The 2.09× batching
multiplier we attribute to "left-pad" must come from somewhere else
(per-step concat-then-split at line 3049-3122 of
`request_state.rs::decode_qwen35_batch`? Actually that's dead code.
Real path is `decode_qwen35_packed_batch` — would need profiling to
confirm).

## What to do with the 19 commits

### Live (correct & useful)

- Pool data structure (`MetalKVPool`) — metadata-aligned with oMLX's
  block manager; useful for SSD prefix dedup
- Path probes (`DECODE_QWEN35_*_PROBE`) — keep as permanent diagnostic
- Single-stream c=1 path integration (P2.0/P2.2/P3.1a/b/P3.1c.1/c.2)
  — bench-validated, no regression

### Live but not the unlock

- Pool dual-write in `decode_qwen35_packed_batch` (commit `f9f3f7e`)
  + `pool.flush()` — pool stays in sync with the session, useful for
  *future* SSD persistence + prefix dedup, NOT for c=4 SDPA cutover

### Dead-code (targeted wrong path)

- P3.1c.3a/b/c on `decode_qwen35_batch` — scheduler doesn't call that
  function at c≥2. Leave in tree as harmless preparation but
  acknowledge.

## Top 3 ARLE adoptions, ranked

### A. Switch `pool.flush()` from `async_eval` to `eval` (5 LOC)

oMLX's `scheduler.py:48-58` warns `async_eval` crashes M4 drivers.
ARLE adopts it without that signal. Risk: if we hit a crash on M4 in
production, it's traceable to this. **Change**: edit
`MetalKVPool::flush()` body to call `eval` not `async_eval`. Bench
for ITL impact (likely +0.5-2 ms p50; sync vs async).

### B. Restructure `pool.write_kv` to slice-assign on `[B, H, T, D]` (M effort)

ARLE's per-token `slice_update(self.k_pool[L], row_k, [pi, 0], ...)`
on a `[max_total_tokens, kv_dim]` flat layout requires reshape +
per-token call. oMLX's `cache.keys[..., prev:offset, :] = keys`
writes `[B, H, T_new, D]` in one assign per layer per step on a
pre-allocated `[B, H, capacity, D]` buffer with step-grow.

Translated to ARLE: per-state pool keeps current shape but write API
takes `[1, n_kv_heads, T_new, head_dim]` (not flattened
`[T_new, kv_dim]`); slice-assign at `[..., offset:offset+T, :]`
without per-token loop. Combined with the eval-barrier boundary,
this should make per-step pool maintenance ~free (in-place slice).

This is a real refactor. P3.1c.3 follow-up.

### C. Keep paged-cache as metadata-only for prefix dedup + SSD (long-term)

Don't try to make Metal SDPA read from a block-indirect pool. Mirror
oMLX: hash blocks for prefix sharing/SSD persistence,
`reconstruct_cache` to contiguous at admission, normal SDPA.

The c=4 ITL gap closure must come from a different lever — most likely
the BatchPackedDecode left-pad mask + scheduler timing; needs profile.

## Action items

1. **Profile c=4 first** (Task #16 still pending): metal capture +
   per-phase Rust timing in `decode_qwen35_packed_batch`. Without this,
   any "unlock" claim is unfalsifiable. Per
   `feedback_path_probe_before_perf_claim.md` and
   `feedback_perf_model_unverified.md`, profile is the prerequisite.
2. **Switch `pool.flush()` to sync `eval`** (5 LOC, low risk, prevents
   future M4 driver crash).
3. **Update master analysis + M_e.1 plan** with oMLX correction. The
   "kernel cutover unlocks c=4" framing is misaligned; what actually
   moves c=4 is unverified.
4. **Don't pursue P3.1c.3d** as a kernel cutover (block-table
   indirection at SDPA). oMLX shows this isn't the path. Use the
   commits already shipped as substrate for SSD prefix dedup
   (oMLX-style metadata layer) instead — that's where the substrate
   actually composes well.

## References

- jundot/omlx @ main:
  - `omlx/cache/paged_cache.py` (block metadata, no mlx ops)
  - `omlx/cache/prefix_cache.py:1361` (`reconstruct_cache`)
  - `omlx/turboquant_kv.py:189-265` (BatchTurboQuantKVCache, mask
    assembly)
  - `omlx/scheduler.py:48-58` (async_eval danger), 1533, 3524, 3536
    (eval barriers)
- mlx-lm `mlx_lm/models/cache.py` — upstream KVCache.update_and_fetch
- ARLE master analysis (now needs correction):
  [`docs/projects/2026-05-07-metal-optimization-master-analysis.md`](../projects/2026-05-07-metal-optimization-master-analysis.md)
- M_e.1 plan parent:
  [`M_e1-metal-paged-kv-hot-path.md`](M_e1-metal-paged-kv-hot-path.md)
- Audit chain:
  [`2026-05-07-three-layer-audit-miss-c4-real-path-is-packed-batch.md`](../experience/errors/2026-05-07-three-layer-audit-miss-c4-real-path-is-packed-batch.md)
