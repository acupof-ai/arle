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

## Line-level implementation spec (gate passed 2026-06-14)

Measured decomposition (nsys, c=1 vs c=8, 32K): the batched lane step is **58%
attn-half (7.27× ∝c = the per-row loop), 35% MoE (3.70× sub-linear — amortizes,
NOT the wall), 7% tail**. Batching the attention loop alone: step 130.9→65.3 ms,
aggregate ~61→~123 tok/s (≈2×). MoE becomes the next (softer) wall.
([campaign](../experience/wins/2026-06-14-dsv4-longctx-concurrency-serial-capped.md),
[step-profile](../experience/wins/2026-06-14-dsv4-config-mechanism-classification.md)).

**The one thing to change:** the per-row attention loop
`dsv4.rs:1872-1911` (`for r in 0..n { memcpy→single-row mla_attention→memcpy }`)
→ one batched attention over the N rows. The single-row core is
`try_flashmla_decode_attention` (`attention.rs:4879`); make it process `seq_len=N`.

Per-row state ops inside it + their batched disposition:
| op | line | batched plan |
|---|---|---|
| gate `seq_len != 1` | 4907 | → `seq_len <= max_decode_batch` |
| `flashmla_pack_sw_ring` / `pack_one_sw_token` / `pack_compressed_delta` | 4934/4939/4943 | per-row (each writes its slot's KV into the unified pool) — keep the loop; cheap (one token/row). Batch later if profiled. |
| `dsv4_flashmla_decode_build_indices_start_pos_ptr_raw` | 4969 | → orphaned **`dsv4_flashmla_decode_build_indices_batched_raw`** (`cuda-kernels/src/attention.rs:333`) → `indices[N, topk_unified]` |
| sched_meta (cached constant) | state init `attention.rs:929` | → **per-forward** `arle_flashmla_sm90_sparse_decode_sched_meta(b=N, topk_length[N])` — the cached-constant PITFALL for N>1 |
| `arle_flashmla_sm90_sparse_decode_fwd` | 5084 | already takes `b` → call `b=N` with `block_table[N]`, `cache_seqlens[N]=start_pos[r]+1`, gathered `q[N,1,h,d]` |

State buffers to resize (`Dsv4FlashMlaDecodeState`, `attention.rs:786`, sized for
s_q=1 today): `indices[N×topk_unified]`, `topk_length[N]`, `lse_out[N×h_q]`,
`lse_accum[N×accum_rows×h_q]`, `o_accum[N×accum_rows×h_q_d]`, `num_splits[N+1]`,
`sched_meta` (per-forward), new `block_table[N×max_pages]`. Q gather + O scatter
[N,1,h,d]↔attn_out[N,hidden].

**Increments + gate (each: B=1 ms/step 42.0 unchanged + needle exact 512/6000):**
- **A** — add the N-sized buffers + `block_table[N]` + per-forward `sched_meta(b=N)`;
  keep the per-row kernel call. B=1 byte-path unchanged.
- **B** — raise the gate; replace the per-row loop with gather → one
  `sparse_decode_fwd(b=N)` → scatter. Gate = needle ×3 (B=1 + c=8 self-consistency,
  NOT byte-parity — MoE non-det) **AND c=8 aggregate decode tok/s RISES** (the
  acceptance bar from the long-ctx campaign: aggregate-must-rise-with-c).
- **C** — c-sweep wall-clock license → default flip.

Coverage: SW / CompressedSparse / HybridCompressed modes (`mode_int`, 4954); TP
gather when `tp_world>1`; the MTP-verify per-row path stays per-row (tree-shaped).
**Wrinkle:** `--spec-type mtp` disables batched decode (`executor.rs:1563`
`rows>1 && !spec_decode`) — the concurrency lane is the no-MTP path; c=1 keeps MTP
for latency. Reconciling batched+MTP is a later item.

## Out of scope
Whole-step CUDA graph (#70), DP-attn (#89) — see throughput plan; both are
*after* this and re-baselined against the faster batched step. Batched MoE
(the 35%, sub-linear) — the next wall after attention, separate lever.
