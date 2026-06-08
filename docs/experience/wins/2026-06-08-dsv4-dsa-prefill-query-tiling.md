# DSv4 DSA prefill query-tiling — eliminates the O(N²) indexer-logits OOM

**Date:** 2026-06-08. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Status:** landed in `crates/infer-cuda/src/attention.rs`; verified on-pod that the
DSA-logits OOM is eliminated. **Long-context prefill is not yet end-to-end** — the
next per-layer scratch (and one-shot driving) still block 900K; see Next.

## Context

900K-token needle test OOM'd in the DSv4 "official DSA" indexer prefill. Root cause
(`attention.rs:932-935`): the logits scratch is `token_count × roundup(compressed_capacity,256)`,
i.e. **dense `O(N²/compress_ratio)`** f32. DSv4-Flash `compress_ratios` alternate 4/128,
so cr=4 layers dominate: at N=900072, cr=4 → `900072 × 225024 × 4B ≈ 810 GB/rank` →
`CUDA_ERROR_OUT_OF_MEMORY` at the first cr=4 layer. KV itself fit (21.5 GB/rank).

Upstream check (SGLang `dsv4/indexer.py` + `metadata.py`, the reference ARLE mirrors):
the indexer does **not** tile internally — it processes the current `forward_batch`'s
tokens, and SGLang bounds the logits via **chunked prefill** at the scheduler. ARLE's
parity path drove prefill one-shot (`token_count` = full N) and allocated the scratch by
`max_seq_len`, so it materialized the full-N logits.

## What worked

Query-axis tiling inside `csa_select_official` (the only path — no one-shot fallback):
- New `const DSV4_DSA_PREFILL_QUERY_TILE: usize = 4096`. `Dsv4DsaOfficialState` query-dim
  scratch (`logits`, `q_fp8`, `weights`, `context_lens`, `positions`, `page_table_identity`)
  sized by `query_tile = TILE.min(max_seq_len)` instead of `max_seq_len`. Key-dim buffers
  (`rotated_keys`, `cache_locs`, `freqs_cis`, key cache) and the full-N output `raw_indices`
  unchanged.
- The per-query compute loops over query sub-chunks `[t0, t0+tlen)`, writing the disjoint
  `selected`/`raw_indices[t0*topk..]` slices. Key-packing (incremental compressed cache)
  is query-independent and untouched.
- **Correctness:** each query's indexer logits/top-k is independent of the others, so tiling
  is a pure batching change — output is bit-identical to the untiled path. For
  `token_count ≤ TILE` the loop is a single iteration, behavior-identical to before.
- Logits scratch at cr=4 @900K: `4096 × 225024 × 4B ≈ 3.7 GB` (was ~810 GB).

`CUDARC_CUDA_VERSION=12060 cargo check -p infer-cuda --features cuda,no-cuda` PASS;
rebuilt the `dsv4_parity` example on the H20 pod (8×H20).

## Verified (on-pod, 8×H20)

900K parity run **no longer fails on `DSv4 official DSA logits alloc`**. It now OOMs
*earlier* — in `from_dsv4_fp8_safetensors` (slot/scratch construction) on the next
per-layer `max_seq_len`-sized scratch (`Dsv4PrefillDeepGemmLinearScratch`:
`input_fp8 = max_seq_len × hidden ≈ 3.7 GB` × 43 layers ≈ 250 GB). The DSA quadratic
blocker is gone; the remaining ones are the same "sized by full seq, not chunk" anti-pattern.

## Next / Rule

- **Prefill scratch must be chunk-bounded systematically**, not just DSA: every per-layer
  prefill scratch sized by `max_seq_len` (`Dsv4PrefillDeepGemmLinearScratch`, fused-wqkv,
  flashmla, compressor, indexer) should bound the query/token dimension by the prefill chunk.
- **Prefill must be driven in chunks** (the SGLang-canonical mechanism): the one-shot example
  also blows transient activations (hidden/MoE) at O(N). The DSv4 executor already supports
  chunked prefill (`executor.rs:556 ensure_slot_ready_for_prefill` + `forward_prefill_tokens(..,
  final_prefill)`); the gap is the serve frontend (`arle serve` bails on DSv4 multi-GPU).
- **pending-remote:** full 900K-needle end-to-end (chunked prefill + decode + retrieval)
  blocked on the above. This entry covers the DSA-logits quadratic fix only.
- Rule: feature-detect long-context memory by **chunk**, never `max_seq_len`; the indexer
  logits being dense `O(N²/cr)` is the headline long-context-prefill memory trap.
