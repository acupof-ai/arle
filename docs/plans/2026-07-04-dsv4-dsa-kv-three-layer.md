# DSv4 DSA KV storage — three-layer redesign (Shape / Index / Physical)

> Status: Active — 2026-07-04. Design ratified with ckl. Deletion-style refactor:
> the bug CLASS is already attributed (rounds 2-4 — corruption is upstream of the
> decode read, in the value staging/mapping, **layout-dependent** ⇒ not compute
> math, which is deterministic across restarts). Deleting the drift-prone parallel
> maps removes the class whatever the exact off-by; no separate attribution pod
> round — the new `BlockMap` invariant assert + per-phase needle gate verify it.
> Supersedes the ad-hoc bf16-shadow + FP8-pool + dsa-key-cache addressing.

## Why (the problem in one paragraph)

DSv4 sparse attention stores the same logical KV in **four** physical layouts,
each with its own block→location math maintained by a separate write path
(`sw_window_cache`, `compressor.compressed`, `indexer.compressed`,
`dsa_key_cache`, plus the FP8 paged pool). The invariant "block `k` is the same
token range everywhere" is implicit and asserted nowhere. Below 2048 tokens the
top-k selection is exhaustive (`deepseek_v4_topk_transform`
`naive_paged_transform`, `dsv4_dsa_official.cu:646`), so the block→location map
is never exercised; above 2048 real `radix_topk` selection runs and the four
maps drift — #146: retrieval silently corrupts past `index_topk × cr = 512 × 4
= 2048`, layout-dependent, identical wrong bytes on both read lanes (round 4).

## The three-layer contract

Every KV access crosses exactly these layers, **one direction, one map per hop,
no layer-skipping**:

```
Shape    "compressed block k  ==  tokens [k·cr, k·cr+cr-1]"     pure semantics, no bytes
   │  BlockMap (the ONE index)
Index    "block k  →  page p, row r"                            the single block↔location truth
   │  page_table (dynamic draw)
Physical "page p  →  pool byte offset, FP8 record"              real memory
```

- **Shape** is length-agnostic: a longer sequence is just a larger `k`. No pool,
  no page, no bytes at this layer.
- **Index** is the *single* block↔location authority. Both the value pool and the
  indexer-key side store resolve a block through the SAME `BlockMap`. This is the
  "块 k ⟺ 值池第 k 块页" contract, made explicit.
- **Physical** knows bytes/FP8/page draw, never what a "block" means.

Rule: cross a layer only through its one map. Shape never sees pages; Physical
never sees blocks. Drift becomes structurally impossible.

## What goes through paged attention — and what does not

| Storage | Role | Layer path | Paged? |
|---|---|---|---|
| SW keys (FP8 pool SW region) | attention reads | Shape→Index→Physical, `page_table` | **yes** |
| Compressed values (FP8 pool comp region) | attention reads | Shape→Index→Physical, same `page_table` | **yes** |
| CSA indexer keys (`dsa_key_cache`) | scoring / block selection only | Shape→(own side map) | **no** — separate small store, shares Shape's block count |
| bf16 shadows ×3 | prefill pack + scalar lane | (legacy, deleted) | — |

The indexer key is a 128-dim scoring aid, not attention content — it stays a
non-paged side pool by design. Its only tie to the value pool is the Shape layer:
indexer block `k` and value block `k` are the same token range, asserted.

## Target vs current

- **Value side collapses to ONE paged pool** (the existing `flashmla_kv_pool`).
  Delete `compressor.compressed` as a value store and the scalar hybrid value
  lane (`dsv4_hybrid_attention_*`, already the quality-defective inferior lane per
  `wins/2026-07-03-dsv4-138-decoded-case-context129-wall.md`). One value
  coordinate, one consumer (FlashMLA).
- **`BlockMap` is the only block→(page,row) function**, owned by
  `Dsv4LayerKvLayout`, reused by: pool pack (prefill + decode delta), selection
  output translation, value gather, and the indexer-key band. No more four
  parallel computations.
- **Invariants asserted at the select→gather boundary** (turn silent >2048
  garble into a loud fail): `indexer.compressed.seq_len ==
  compressor_block_count`, `packed_rows == compressed_count` (pool fully packed
  before any read), `selected[k] ∈ [0, compressed_count)` (already sane per round
  3 — keep it enforced).

## Scope: what is touched, what is NOT

**Compute kernels are untouched.** Attention math (`flash_fwd_*`), the compressor
kernels (`dsv4_compressor_block/finalize`), MQA scoring, and every GEMM stay
byte-for-byte — #146 is a storage/mapping bug, not a math bug. Only two kernel
classes move, both toward deletion:

- **Mapping kernels** (`dsv4_flashmla_decode_build_indices.cu`,
  `arle_flashmla_csa_prep.cu` index build): they wrongly recompute block→slot
  *inside the kernel*. Correct paged-attention form is a host-built page table the
  kernel blindly consumes (the SW region already does this). These shrink to a
  page-table lookup or vanish.
- **The redundant scalar attention lane** (`dsv4_hybrid_attention_*`,
  `dsv4_swa_attention_*`): deleted wholesale.

Net: Rust-side `BlockMap` + delete/shrink mapping kernels + delete the scalar
lane. No compute-logic rewrite — the surface is large in file count, small in
risk.

## Implementation DAG (file:line)

**P1 — introduce the `BlockMap` type (no behavior change).**
`crates/infer-cuda/src/attention/kv_layout.rs` — add `Dsv4BlockMap { cr, sw_blocks,
page_size }` with `block_to_page_row(k) -> (page, row)` and `token_range(k)`. Wire
it into `Dsv4LayerKvLayout` (`kv_layout.rs:171`). Replace the inline math in
`flashmla_pack_compressed_delta` (`attention.rs:1948`), `arle_flashmla_csa_prep.cu`
index build, and `dsv4_flashmla_decode_build_indices.cu:118-151` with `BlockMap`
lookups. Byte-identical; the point is one source. Gate: needle 3/3 ≤2048 unchanged.

**P2 — route value reads through the pool only.** Make `csa_select_official`
(`attention.rs:8747`) emit block ids in Shape coordinate; the value gather resolves
through `BlockMap`→`page_table`. Remove the `selected`-as-two-coordinates ambiguity
(`raw_indices` logical vs `selected` slot) — one coordinate downstream.

**P3 — delete the bf16 value shadow + scalar value lane.** Remove
`compressor.compressed` value role and `dsv4_swa/hybrid_attention_*`
(`dsv4_attention.cu:717,1723`) once P2 lands. The compressor still produces
compressed KV, but it packs straight into the FP8 pool (Physical) via `BlockMap`,
never a parallel bf16 buffer. Keeps `pending_kv`/`prev_overlap` (cross-chunk
accumulator state — irreducible). Gate: needle 3/3 at 8K/32K + MTP-on.

**P4 — indexer-key side pool shares Shape.** `dsa_key_cache` stays non-paged but
its block count is derived from the same `BlockMap`; assert
`indexer_rows == compressor_rows` at the select boundary (`attention.rs:5224`).

**P5 — (follow-on) page-granular L2/L3.** Once value KV is pure pages, the tier
store can demote/promote per page instead of whole-slot blobs
(`executor.rs:2469` prefix store) — unblocks >200K contexts. Out of this plan's
scope; noted as the payoff.

## Verification (replaces the dropped P0 attribution round)

The `BlockMap` invariant assert is the standing substitute for a one-shot
attribution dump: a residual drift fails LOUD at the boundary instead of
silently garbling, and it keeps guarding forever. Plus a correct-inference gate
per phase: needle retrieval ×3 at the phase's target length (2K→8K→32K), MTP-on +
spec-none, SGLang as the reference oracle (exact on the same checkpoint/quad). A
phase that regresses ≤2048 or fails its new length is reverted, not patched
forward — and it is debugged on the CLEAN single-map baseline (which is what
"root-cause on a clean baseline" wants). Bench entry per landed phase (`wins/`),
decode tok/s A/B to prove the deletion didn't regress the fast path.

## Non-goals

- Not unifying the indexer key INTO the value pool (different data — 128-dim
  scores vs 512-dim values; forced union adds entropy).
- Not touching the SW ring's fixed 128-token semantics.
- Not the FP4 lane (#137, fail-closed).
