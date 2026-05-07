# M_e.2 — PromptTrie + extend() prefix-cache integration for Metal

**Owner:** ckl · **Status:** designed 2026-05-07 (subagent), awaiting impl tick
**Track:** Metal scheduler · **Predecessor:** M_e.1 oMLX-C v3 (default-on)

## Goal

Adopt mlx-lm technique #2 from
[`docs/research/2026-05-07-mlx-ecosystem-survey-c4-itl-gap.md`](../research/2026-05-07-mlx-ecosystem-survey-c4-itl-gap.md):
shared-prefix detection at request admit + GPU-side cached-KV merge into the
running packed batch. Target workload: c=8+ chat with shared 800-token
system prompts.

## 1. Map mlx-lm semantics → ARLE state

| mlx-lm | ARLE today | Gap |
|---|---|---|
| `PromptTrie` keyed on token IDs (PR #1019) | `crate::prefix_cache::RadixCache` ([`infer/src/prefix_cache.rs:244-301`](../../infer/src/prefix_cache.rs)) — token-keyed, block-aligned, namespace-pinned (M_d.1) | None semantically. Block-resolution mismatch (mlx-lm trie is per-token; ours per-block) just lowers our resolution; v1 tolerates it. |
| `LRUPromptCache` recency eviction | RadixCache has soft-pin + keepalive ticks (`with_soft_pin_keepalive_namespaced:326`, `bump_soft_pin:2095`) and `evict_with_policy_for_intent:1227` — recency-aware | Reuse as-is for v1. mlx-lm's pure LRU is a strict subset. |
| `BatchKVCache.update_and_fetch + extend` (PR #1141, `mlx_lm/models/cache.py:912-967`) | `Qwen35PackedDecodeBatch::admit_rows` ([`request_state.rs:891-1040`](../../infer/src/backend/metal/request_state.rs)) appends *empty/left-padded* per-row slabs | **NEW**. ARLE has no path to seed the appended row's first L columns from a cached prefix tensor. `left_pad_kv_cache_row` ([`helpers.rs:19-52`](../../infer/src/backend/metal/request_state/helpers.rs)) zero-fills — needs a prefix-fill variant. |
| Bridge: trie ↔ Metal slot ledger | `MetalPrefixCache` ([`backend/metal/prefix_cache.rs:25-44`](../../infer/src/backend/metal/prefix_cache.rs)) exists, but only consumed by `MetalKVPool::share_prefix_from` ([`kv_pool.rs:140,433`](../../infer/src/backend/metal/kv_pool.rs)) and the Qwen3.5 **SSD** snapshot path (`runtime.rs:780-890`) — never wired into the packed-decode admit hot path. | **NEW WIRING**. Lookup at admit time + tensor handoff into `admit_rows`. |

## 2. Integration point

- **Lookup** at admit-to-active (top of `execute_qwen35_packed_decode_batch`,
  [`runtime.rs:2403-2453`](../../infer/src/backend/metal/runtime.rs)),
  **before** the new request runs `prefill_chunk`
  ([`runtime.rs:155, 1994, 2063`](../../infer/src/backend/metal/runtime.rs)).
  Hook in the same control-flow position where today a freshly-admitted
  request goes through `enqueue_request → execute_prefill_chunk`
  ([`runtime.rs:1886, 1994`](../../infer/src/backend/metal/runtime.rs)).
- **Merge** during the `admit_row_indices` branch at
  [`runtime.rs:2419-2446`](../../infer/src/backend/metal/runtime.rs).
  `Qwen35PackedDecodeBatch::admit_rows`
  ([`request_state.rs:891`](../../infer/src/backend/metal/request_state.rs))
  already appends new rows; we replace the zero-filled
  `left_pad_kv_cache_row` call ([`request_state.rs:986-991`](../../infer/src/backend/metal/request_state.rs))
  with a **prefix-fill** variant for rows whose `MetalRequestState`
  carries a new `cached_prefix_kv: Option<Vec<MlxArray>>` field.
- The request still needs to run a **partial** prefill for tokens after
  the cached prefix. At admit, slice the prompt into
  `[cached_prefix | residual_prompt]`, seed KV from the trie hit, and
  let the existing chunked-prefill loop handle the residual.
  `cache_len` on the driver is initialized to `L` instead of `0` before
  the first `prefill_chunk` call.

## 3. extend() spec for `Qwen35PackedDecodeBatch`

mlx-lm pattern (`mlx_lm/models/cache.py:946-967`, paraphrased):

```python
def extend(self, kv: BatchKVCache):
    # self.keys: [B0, n_kv, cap, d];  kv.keys: [1, n_kv, L_new, d]
    pad = self.offset - kv.offset
    new_k = mx.concatenate([mx.zeros(...pad...), kv.keys, mx.zeros(...rest...)], axis=2)
    self.keys = mx.concatenate([self.keys, new_k], axis=0)
```

Rust + ARLE's MLX wrappers (existing `concatenate_axis`, `slice`,
`slice_update`, `zeros` from
[`infer/src/backend/metal/mlx.rs:556-618`](../../infer/src/backend/metal/mlx.rs)
and [`ops.rs:254`](../../infer/src/ops.rs)):

1. For each new row: build `prefix_row_k: [1, n_kv, L_cached, d]` and
   `prefix_row_v: …` from the trie hit (already lives as `MlxArray`
   in the `MetalPrefixCache` payload — needs a slot→tensor materializer
   added there).
2. **Replace** the
   `left_pad_kv_cache_row(slot, left_pad, qwen35.driver.cache_len, kv_capacity)`
   call at [`request_state.rs:986-991`](../../infer/src/backend/metal/request_state.rs)
   with a `pad_kv_cache_row_with_prefix(prefix_row, left_pad, L_cached, kv_capacity)`:
   same skeleton as
   [`helpers.rs:19-52`](../../infer/src/backend/metal/request_state/helpers.rs)
   but the `slice_update` source is `prefix_row` instead of nothing
   (the `zeros` base stays — it covers `left_pad` and the
   `[L_cached, kv_capacity)` tail).
3. Existing `concatenate_axis(&concatenated, 0)` at
   [`request_state.rs:1011, 1021`](../../infer/src/backend/metal/request_state.rs)
   is the B0+1 concat — unchanged.

Shape check: ARLE's `packed_kv_flat[layer]: [B, n_kv_heads, kv_capacity, head_dim]`
([`qwen35.rs:1703`](../../infer/src/backend/metal/qwen35.rs)) matches
mlx-lm `BatchKVCache.keys` exactly — no transpose needed.

## 4. v1 NON-goals (defer to v2)

- Trie pruning policy beyond existing soft-pin/keepalive.
- Compression of cached prefix tensors (FP8, paged spill).
- GPU-side trie traversal — keep CPU-side token-id walk.
- Cross-request **partial** prefix sharing (v1 only longest-block-aligned).
- Prefix fingerprint validation against weight version **at lookup** —
  rely on existing `model_fingerprint` namespace pin
  ([`runtime.rs:986, 887-890`](../../infer/src/backend/metal/runtime.rs)).
- Speculative-decode interaction (DFlash already disables `kv_pool`;
  treat as mutually exclusive in v1).

## 5. Acceptance bench

`scripts/bench_guidellm.sh prompttrie-c8-shared-sysprompt`. Workload:
c=8 chat with a fixed 800-token system prompt + ~50-token user turn.
Expectation: TTFT for second-and-later requests drops from
`prefill(850)` to `prefill(50)` — at Qwen3-4B Metal numbers
(~8000 tok/s prefill on M2 Ultra) that's ~106ms → ~6ms TTFT for
shared-prefix hits. ITL (decode) unchanged.

## 6. Risk register

| ID | Risk | Mitigation |
|----|------|------------|
| R1 | Cached KV misaligns with current model weights (quant, RoPE base, KV layout) | Reuse existing `model_fingerprint` namespace on RadixCache (`runtime.rs:986`) + Qwen3.5 SSD snapshot's metadata checksum (`request_state.rs:550-622`). Drop on mismatch; don't try to reproject. |
| R2 | LRU eviction storm: chat fan-in admits N rows in one tick, each inserting a near-duplicate prefix → trie thrashes | v1 only **inserts** at request *finish* (already RadixCache convention via `lookup_or_stage:674`); **lookups** at admit are read-only. Cap insertions per tick at high-water (`prefix_cache_high_water:0.75`). |
| R3 | oMLX-C `prev_sampled` shape interaction (PR #1072 BatchGenerator compaction analog) | Already handled: `admit_rows` sets `self.prev_sampled = None` (`request_state.rs:1027-1031`). Prefix-seeded admits hit the same path — no extra change. |
| R4 | Dual-write with `MetalKVPool` slot ledger: pool refcounts must increment when prefix is consumed by an admitted request | Route prefix consumption through existing `MetalKVPool::share_prefix_from` (`kv_pool.rs:433-440`) so the slot refcount path is exercised. The packed-decode batch consumes MLX tensors; the pool ledger is the bookkeeping owner. |

## References

- mlx-lm: `mlx_lm/models/cache.py:912-967` (`BatchKVCache`, `update_and_fetch`,
  `extend`); PRs ml-explore/mlx-lm#1019 (PromptTrie), #1141 (extend),
  #1072 (BatchGenerator compaction)
- ARLE survey: [`docs/research/2026-05-07-mlx-ecosystem-survey-c4-itl-gap.md`](../research/2026-05-07-mlx-ecosystem-survey-c4-itl-gap.md)
- Predecessor: [`docs/experience/wins/2026-05-07-bench-c4-omlx-c-v3.md`](../experience/wins/2026-05-07-bench-c4-omlx-c-v3.md)
