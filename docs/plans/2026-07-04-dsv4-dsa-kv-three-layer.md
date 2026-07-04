# DSv4 DSA KV storage — three-layer redesign (Shape / Index / Physical)

> Status: Active — 2026-07-04. **Deletion-style refactor that eliminates #146 as
> a side effect.** DSv4 stores the same logical KV in 4 physical layouts, each
> with its own block→(page,row) write map on a separate path. Above 2048 tokens
> the maps drift → #146: needle garble past `index_topk·cr = 2048`, depth-locked,
> identical wrong bytes on both read lanes (round 4 — a single shared *write*
> point writing to the wrong row, read back by both lanes). Collapse to ONE
> asserted `BlockMap` and ONE FP8 value pool; the drift class is gone whatever
> the exact off-by, and the invariant assert + per-phase needle gate verify it.
> No attribution A/B — the deletion removes the class; the assert catches any
> residual LOUD. (SGLang is exact on the same ckpt with the same vendored
> compressor kernel ⇒ the compressor math is not the ARLE-vs-SGLang delta; the
> excess deep-pos error lives in ARLE's own storage/pack/mapping layer, which is
> exactly what this deletes. Residual guard: if needle still garbles >2048 after
> P3 lands on the clean single-map baseline, the root cause is compressor
> precision, not mapping — file separately. Low prior, not silent.)

## The chain, top to bottom

```
Shape      block k  ==  tokens [k·cr, k·cr+cr-1]        pure semantics, no bytes
   │  Dsv4BlockMap  (the ONE index — new)
Index      block k  →  (page p, row r)                  single block↔location truth
   │  page_table    (host-built, drawn at admission)
Physical   page p   →  FP8 pool byte offset, record     real memory
   │  budget + pre-allocation
Backing    pool.max_total_pages  ←  kv_budget_plan       VRAM ledger, admission gate
```

### 1. Shape — the KV abstraction (length-agnostic)

"Compressed block `k` covers tokens `[k·cr, k·cr+cr-1]`." A longer sequence is
just a larger `k`. No pool, no page, no bytes here. Both the value pool and the
indexer-key side share this one block count — indexer block `k` and value block
`k` are the same token range, asserted (`attention.rs:5224`).

### 2. Index — `Dsv4BlockMap`, the ONE block→location authority

Today 4 write paths each recompute `block→(page,row)` (`sw_window_cache`,
`compressor.compressed`, `dsa_key_cache`, FP8 pool). Below 2048 the top-k is
exhaustive (`naive_paged_transform`, `dsv4_dsa_official.cu:646`) so the map is
never exercised and they agree by accident; above 2048 real `radix_topk` runs and
they drift → #146. **`Dsv4BlockMap { cr, sw_blocks, page_size }`
(`kv_layout.rs`) becomes the single function** `block_to_page_row(k)->(page,row)`
+ `token_range(k)`, owned by `Dsv4LayerKvLayout` (`kv_layout.rs:189`), consumed by
every write path. One computation, asserted, no drift possible.

Layout: SW ring at slot-logical pages `[0, sw_blocks)`, compressed region at
`[sw_blocks, sw_blocks+comp_blocks)`. Fixed slot-logical block ids — `BlockMap`
is the only place that math lives.

### 3. Physical — one FP8 paged pool

The existing `flashmla_kv_pool` (`kv_layout.rs:199`): a `TokenKVPool` of opaque
PackedBytes records, **584 B/token, page = FlashMLA block = 64 tokens**, single
plane. Every slot's band is addressed ONLY through its page table
(`flashmla_page_table`), **never `slot_idx × slot_bytes` arithmetic**. The value
side collapses to this ONE pool — `compressor.compressed`'s value role and the 3
bf16 shadows are deleted. One value coordinate, one consumer (FlashMLA).

### 4. Budget + pre-allocation — the backing that makes paging real

**Budget (`kv_budget_plan`, `dsv4.rs:1669`, #67).** Slots are not fixed — a fixed
count OOM-crashes at high `c` × long `max_seq_len`. Dynamic:
`affordable = cudaMemGetInfo × 0.9 ÷ per_slot_bytes`, clamp `requested`. The
per-slot ledger (`per_slot_device_bytes`, `dsv4.rs:1612`, MUST track
`Dsv4SlotState::new`): FP8 arena `max_seq_len × 584 × num_layers` ×2 (compressor/SW
+ activations) **+** DSA indexer scratch (`logits` tile ~`max_seq/cr`, dwarfs the
arena at long ctx). **NCCL min-reduced across ranks** — per-rank `mem_get_info`
diverges, and a divergent slot count desyncs the deterministic planner →
NCCL deadlock; a rank that can't query contributes `i32::MAX` (doesn't bind) but
still joins the collective.

**Pre-allocation (the non-obvious part).** The FlashMLA band is NOT a growing
`ceil(seq_len/page)` cache — **all `sw_blocks + comp_blocks` pages are resident
from token 1** (`flashmla_alloc_append`, `kv_layout.rs:239`). Because the SW ring
and compressed region sit at FIXED slot-logical block ids, the pack/read kernels
need every page mapped before the first write; a growing draw would leave the SW
ring / comp region unmapped. So `flashmla_alloc_append` only advances the logical
cursor — the band is drawn once at admission from the host slot page table
(`prepare_kv_batch`), never re-drawn. Truncate on MTP reject
(`flashmla_truncate_slot`, `kv_layout.rs:290`) is cursor-only; band pages stay
resident (never recycled).

**Admission gate.** `pool.max_total_pages` → `flashmla_total_pages()`
(`kv_layout.rs:211`) → `effective_total_pages()` (`lib.rs:387`) → the scheduler's
host admission pool. The budget is the single number the scheduler and the pool
agree on.

## Relation to paged attention — and to the model kernels

**To paged attention:** the SW region ALREADY does it right — host builds the page
table, the kernel blindly consumes it. This refactor makes the compressed value
region do the same. The correct paged form is: `BlockMap` (host) → page_table
(host) → kernel reads by table index, zero block math in the kernel.

**To the model kernels — untouched.** Attention math (`flash_fwd_*`), compressor
(`dsv4_compressor_block/finalize`), MQA scoring, every GEMM stay byte-for-byte.
Only two kernel classes move, both toward deletion:

- **Mapping kernels** (`dsv4_flashmla_decode_build_indices.cu:118-151`,
  `arle_flashmla_csa_prep.cu` index build) wrongly recompute `block→slot` INSIDE
  the kernel — the drift source. Replace with a `BlockMap`-built host page table
  the kernel indexes blindly. They shrink to a lookup or vanish.
- **Redundant scalar lane** (`dsv4_hybrid_attention_*`, `dsv4_swa_attention_*`,
  `dsv4_attention.cu:717,1723`): deleted wholesale — already the quality-defective
  inferior lane (`wins/2026-07-03-dsv4-138-decoded-case-context129-wall.md`).

Net: Rust-side `BlockMap` + shrink/delete mapping kernels + delete scalar lane.
No compute rewrite — large in file count, small in risk.

## Implementation DAG (file:line)

**P1 — `Dsv4BlockMap`, no behavior change.** Add the type to `kv_layout.rs`, wire
into `Dsv4LayerKvLayout` (`kv_layout.rs:189`). Replace inline math in
`flashmla_pack_compressed_delta` (`attention.rs:1948`), `arle_flashmla_csa_prep.cu`
index build, `dsv4_flashmla_decode_build_indices.cu:118-151` with `BlockMap`
lookups. Byte-identical. Gate: needle 3/3 ≤2048 unchanged.

**P2 — value reads through the pool only.** `csa_select_official`
(`attention.rs:8747`) emits block ids in Shape coordinate; value gather resolves
`BlockMap`→page_table. Kill the `raw_indices`(logical)/`selected`(slot)
two-coordinate ambiguity — one coordinate downstream.

**P3 — delete the bf16 value shadow + scalar value lane (this is where #146
dies).** Remove `compressor.compressed`'s value role and `dsv4_swa/hybrid_
attention_*` once P2 lands. Compressor packs straight into the FP8 pool via
`BlockMap`, never a parallel bf16 buffer. Keep `pending_kv`/`prev_overlap`
(cross-chunk accumulator — irreducible). Gate: needle 3/3 at 8K/32K + MTP-on.

**P4 — indexer-key side shares Shape.** `dsa_key_cache` stays non-paged
(128-dim scores ≠ 512-dim values, forced union adds entropy), block count derived
from `BlockMap`; assert `indexer_rows == compressor_rows` at the select boundary
(`attention.rs:5224`).

## Invariants asserted (turn silent >2048 garble into a loud fail)

At the select→gather boundary: `indexer.compressed.seq_len ==
compressor_block_count`, `packed_rows == compressed_count` (pool fully packed
before any read), `selected[k] ∈ [0, compressed_count)`. Any residual drift fails
LOUD at the boundary and keeps guarding forever — this is the standing substitute
for a one-shot attribution dump.

## Verification

Correct-inference gate per phase: needle ×3 at the phase length (2K→8K→32K),
MTP-on + spec-none, SGLang as reference oracle (exact on same ckpt/quad). A phase
that regresses ≤2048 or fails its length is reverted, not patched forward, and
debugged on the clean single-map baseline. Bench entry per landed phase (`wins/`),
decode tok/s A/B to prove the deletion didn't regress the fast path.

## L2/L3 tier adaptation (the payoff, once value KV is pure pages — P5 follow-on)

Today the tier store demotes/promotes whole-slot blobs (`executor.rs:2469` prefix
store) because the value KV is a tangle of 4 layouts — you can't move half of it.
After P3, value KV is a flat sequence of 64-token FP8 pages addressed by one
`page_table`. That is exactly the granularity a tier store wants: **demote/promote
per page**, not per slot. The path becomes `BlockMap`(which blocks are cold) →
page_table(their pages) → tier transport(those page byte ranges via
`kv-native-sys`). Out of this plan's scope — noted as what the three-layer split
unblocks: page-granular L2/L3 → >200K contexts without whole-slot copies.

## Non-goals

- Not unifying indexer key INTO the value pool (128-dim scores vs 512-dim values).
- Not touching the SW ring's fixed 128-token semantics.
- Not the FP4 lane (#137, fail-closed).
