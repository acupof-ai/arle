# DSv4 batched FlashMLA decode — the #1 concurrency lever (wiring, not new infra)

Status: **design** (2026-06-14). The measured #1 concurrency lever
([throughput plan](dsv4-concurrency-throughput.md)): batched decode (B>1) loses
FlashMLA at the `seq_len==1` gate and falls to the general/prefill attention path
(`dsv4_hybrid_attention_*`), so attention — 41% of B=1 GPU time — doesn't
amortize across the batch (1.40×/16× cap). Fix: make batched decode use a
**batched FlashMLA decode** call.

## Core finding: the pieces already exist (medium task, ~2-3 days)

Code-level research verdict — this is **wiring orphaned/B=1-only infra**, not a
new kernel or new KV infra:

| Piece | State | Evidence |
|---|---|---|
| Batched indices builder kernel | **EXISTS, orphaned** (never called) | `dsv4_flashmla_decode_build_indices.cu:173`; wrapper `cuda-kernels/src/attention.rs:340` `dsv4_flashmla_decode_build_indices_batched_raw` |
| Sparse decode kernel | **Takes `b`, batched-capable** (called with b=1) | `ffi/misc.rs:498` `arle_flashmla_sm90_sparse_decode_fwd`; shim loops over batch + split-KV |
| Scheduler-meta builder | **Batched-capable** (called b=1) | `ffi/misc.rs:566` `..._sched_meta` |
| KV pool | **UNIFIED, block-table-addressable** — no new infra | `paged_kv.rs:38` one contiguous `k_data` per layer + per-slot `page_indices`; `attention.rs:614` `flashmla_pool_data()`/`flashmla_pages_byte_range(slot)` |

## The gap (what to wire)

1. **The B=1 gate** — `attention.rs:4907` `if q_prepared.seq_len != 1 { return Ok(false) }`.
   Today `forward_decode_batch` (`dsv4.rs:1874`) loops mla_attention PER ROW
   (seq_len=1 each); a true seq_len=B call is rejected → prefill fallback.
2. **Batched state buffers** — `Dsv4FlashMlaDecodeState` (`attention.rs:786`) is
   per-(slot,layer), sized for seq_len=1: `indices[topk_unified]`,
   `topk_length[1]`, `lse_out/lse_accum/o_accum` no batch dim. Need `[B,…]`.
3. **Block table** (missing): build `block_table[B, max_pages]` per forward by
   copying each row's slot `page_indices` (the batched indices builder already
   applies `slot_layer_block_offsets[row]` to translate slot→pool coords).
4. **Batched metadata call** (was cached): `sched_meta`/`num_splits` are computed
   ONCE at state init for B=1 — must become a per-forward
   `..._sched_meta(b=B, topk_length[B])` call (the cached-constant pitfall).
5. **Q/O gather-scatter**: gather B rows → `[B,1,h_q,d_qk]` (the kernel's
   `stride_q_b = h_q*d_qk` matches `normed` viewed as `[B, h_q*d_qk]`), call once
   b=B, scatter output `[B,1,h_q,d_v]` back to per-row `attn_out`.

## SGLang target shape (1:1 map)
`flash_mla_with_kvcache(q[bs,1,h,d], block_table[bs,max_pages],
cache_seqlens[bs], tile_scheduler_metadata, num_splits[bs+1])`
(`flashmla_backend.py:355`). Ours: block_table ← B slots' `page_indices`;
cache_seqlens ← `start_pos[B]+1`; metadata ← the `..._sched_meta(b=B)` call;
q ← reshaped batched `normed`.

## Execution phases (each gated, B=1 must never regress)

**Phase A — batched metadata + block_table (no B>1 yet).** Add the batched state
buffers; build `block_table[B]` + call the batched indices builder + per-forward
`sched_meta(b=B)`. Keep the per-row kernel call for now. Verify **B=1 byte-path
unchanged** (the new buffers are allocated but the b=1 call still runs) — needle
512/6000 + B=1 53.3 unchanged.

**Phase B — the batched kernel call.** Raise the gate (`seq_len==1` →
`seq_len <= max_decode_batch`); replace the per-row loop with one gather →
`arle_flashmla_sm90_sparse_decode_fwd(b=B)` → scatter. Gate: **needle ×3 at B=1
AND a concurrent c=8 run** (same-config-twice floor, NOT byte-identity — MoE
non-determinism), self-consistency (the batched-FlashMLA output is the reference,
not the prefill-path output). Edge cases: TP gather (tp_world>1), DSA/CSA vs HCA
mode, the MTP-verify per-row path (stays per-row — it's tree-shaped, separate).

**Phase C — perf license + default.** c-sweep wall-clock (c=1/4/8/16): TTFT + ITL
+ agg vs the 44→62 batched baseline. License a default flip only if it beats the
baseline at c≥4 AND holds B=1. Multi-shape (short + long prompt) per the bench
spec. Wins entry with Δ%.

## Risks
- **The cached sched_meta pitfall** (gap #4) — recomputing per-forward is
  mandatory; a stale cached B=1 meta on a B>1 call = wrong split-KV merge.
- **Mode coverage**: SlidingWindow vs CompressedSparse vs HybridCompressed —
  the batched indices builder must handle all three (the B=1 path does).
- This is the genuine concurrency lever; do NOT couple with CUDA-graph (#70) or
  DP-attn (#89) — land batched decode first, re-baseline, then those.

## Out of scope
Whole-step CUDA graph (#70), DP-attn (#89) — see throughput plan; both are
*after* this and re-baselined against the faster batched step.
