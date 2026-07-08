# DSv4 Page-Granular Prefix Reuse (Route A) — #85, SGLang `CompressStatePool` precedent

> Status: Active — design only, no code yet

Owner directive (ckl, 2026-07-08): DSv4's prefix reuse must stop snapshotting
at arbitrary positions. Capture and restore only at the minimal reusable
boundary — page-granular, not whole-slot. Confirmed against SGLang's actual
DeepSeek-V4 implementation before committing to a design (don't hand-roll
when upstream already solved it).

## Why Route B (current, shipped) must go

Route B (`docs/plans/2026-06-11-dsv4-whole-slot-kv-swap.md`, #84/#85) gives
DSv4 "KV stays hot" semantics by snapshotting an **entire slot's** state at
whatever arbitrary position a request happened to stop, and restoring it
verbatim at whatever arbitrary position a later request's prefix match
reaches. Three bugs, all rooted in the same mismatch, were found and fixed
2026-07-06/08 (`docs/experience/errors/2026-07-06-dsv4-concurrent-decode-digit-corruption-unresolved.md`,
issues #151, #152):

1. **CUDA-graph device-page-table UAF** (#151) — a per-call temporary's
   device pointer baked into a captured graph, freed, then overwritten by a
   later restore. Fixed by making the buffer slot-persistent.
2. **Scheduler wrong-seed-token on full match** (#151) — a full-length
   restore skipped the real sampling step, silently reseeding decode with
   the prompt's own last token. Fixed by never restoring the exact last
   block.
3. **`truncate()` straddled-restore staleness** (#152, current stopgap) — a
   partial restore (`matched_len < image_len`) truncates two length counters
   but cannot rewind the compressor/indexer's `pending_kv`/`prev_overlap_*`
   (single-register "most recently completed block" carry-state, provably
   unrecoverable once truncated past a block boundary) or `sw_window_cache`
   (a `pos % sliding_window` ring — same problem). **Bit-correct repair is
   structurally impossible** without a full from-position-0 recompute.
   Current fix rejects any non-exact-length restore and falls back to full
   reprefill — correctness-safe, but an unmeasured performance regression
   that discards most of DSv4's prefix-cache benefit (#152).

The common root cause: compressor/indexer carry-state and the SW ring are
**only well-defined at compress-ratio-block / sliding-window-aligned
boundaries**. An "image" that snapshots wherever a request stops and restores
wherever a later request matches will always eventually straddle a boundary.
The fix isn't a better truncate — it's never taking a snapshot at a position
where restoring it later could straddle one.

## What SGLang actually does (confirmed from source, not summary)

`sgl-project/sglang`, `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`
+ `python/sglang/srt/layers/attention/dsv4/compressor.py`:

- The compressor/indexer's carry-state (`kv_score_buffer`, same shape as
  ARLE's `pending_kv`+`pending_score`+`prev_overlap_kv`+`prev_overlap_score`)
  lives in `CompressStatePool`, a **first-class, page-addressable pool** —
  the same species of object as plain KV, not a bolted-on whole-request
  snapshot.
- **Dual addressing**, both resolving to the same `state_loc` space
  (`compressor.py:282-290`, `create_paged_compressor_data`):
  ```python
  # ephemeral / non-cacheable decode:
  state_loc = req_pool_indices * ring_size + positions % ring_size
  # paged / cacheable — the reuse-relevant path:
  swa_pages = swa_loc // swa_page_size
  state_loc = swa_pages * ring_size + swa_loc % ring_size
  ```
  The paged form keys state by **SWA page identity** (`swa_loc //
  swa_page_size`), not by which request/slot produced it — the same
  addressing scheme plain KV pages use, so a page's compressor state is
  found by position, not by trusting whichever slot last wrote it.
- **The "previous block" overlap contribution is resolved by position, not
  by a trusted register**: `write_overlap_loc = get_raw_loc(write_positions
  - compress_ratio)` (`compressor.py:334`) — looks up `state_loc` at
  `position - compress_ratio` through the same addressing function. As long
  as that page-aligned position's state was ever written once, correctly, by
  ANY request, any later request matching up to that same aligned position
  finds the identical, correct value. No snapshot/restore step, no
  staleness — because it was never captured at an unaligned position to
  begin with.
- **Wired into the same hierarchical/host-tier cache as plain KV** —
  `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`,
  `build_deepseek_v4_hicache_stack`: builds `DeepSeekV4StateHostPool` entries
  (`PoolName.DEEPSEEK_V4_C4_STATE`, `PoolName.DEEPSEEK_V4_C4_INDEXER_STATE`)
  page-sized at `slot_page_size=kvcache.swa_page_size`, participating in the
  **same** `hicache_ratio`/`hicache_storage_backend`/prefetch/write-policy
  machinery as plain KV — one coherent cache system with one demotion path,
  not a parallel bespoke mechanism for the carry-state.

Assessment: SGLang's strategy is "real boundary-aligned carry-state
snapshot/restore, made page-addressable and folded into the normal
hierarchical cache" — not "recompute from scratch on every hit" and not
"true zero-snapshot re-derivation from raw KV bytes" (no backend anywhere,
including SGLang, does the latter for carry-state this shape — it would mean
re-feeding reused KV bytes into the compressor kernel as if they were a live
forward's fresh projections, an unproven numerics path this investigation's
whole day of sub-ULP restore-vs-live chasing argues strongly against).

## ARLE mapping

| SGLang | ARLE today (Route B) | ARLE target (Route A) |
|---|---|---|
| `CompressStatePool` | `Dsv4CompressorState` fields embedded in `Dsv4LayerImage`/whole-slot snapshot | New standalone page-addressable pool, one per (layer, compress_ratio class) |
| `translate_from_swa_loc_to_state_loc` | none — state keyed by `slot_idx` only | New `state_loc(page_id, ring_size)` fn, keyed by page identity like `flashmla_page_table` |
| `swa_page_size` (paged boundary unit) | `page_block_size=64` fixed (`flashmla.rs:58-59`), independent of `compress_ratios` | Derived: `lcm(page_block_size, active compress_ratios for this checkpoint)` — computed once at model load from `DeepSeekV4Config.compress_ratios` (`crates/deepseek-spec/src/v4.rs:64`), not a fixed constant |
| `get_raw_loc(write_positions - compress_ratio)` | `prev_overlap_kv`/`prev_overlap_score` single register, written in-place | Position-indexed lookup into the new pool, same function used for both the current-block write and the prior-block read |
| `hybrid_pool_assembler.py`'s hicache wiring | `Dsv4LayerImage` capture/restore is DSv4-only, separate from `CudaKvTierStore`'s page-key namespace | The new pool's pages participate in the **same** `CudaKvTierStore` demote/promote path plain FlashMLA KV pages already use (see below) |

## L2/L3 tiering — the new pool is not GPU-only (ckl, 2026-07-08)

SGLang's own design explicitly folds the compress-state pool into the same
`hicache_ratio`/`hicache_storage_backend` machinery as plain KV — it is not
solved once page-addressable in GPU memory alone. ARLE already has this
tiering: **L1 GPU HBM → L2 host-pinned DRAM (#82, `CudaKvTierStore`) → L3
NVMe local-disk (#83, `kv-native-sys`, fingerprint-keyed, restart-surviving)**.
The new compressor/indexer pool must compose with this, not bypass it:

- Namespace it as a **third key range** in `CudaKvTierStore` alongside the
  existing `NS_SLOT`/`NS_PREFIX` (`executor.rs:1811-1816`) — e.g.
  `NS_COMPRESS_STATE` — so demote-on-evict/promote-on-use (#82's existing
  machinery) applies to it for free, rather than inventing a second
  eviction/promotion path.
- Size accounting: the compress-state pool is tiny per page relative to
  plain KV (`last_dim = 2*(1+overlap)*head_dim`, a few hundred bytes vs.
  FlashMLA's 584 B/token × up to 128 tokens/page) — cheap to keep L2-resident
  even when the corresponding KV pages have sunk to L3, since it's the
  bottleneck for whether a boundary is reusable at all. Worth checking
  whether it should have a **more generous retention policy** than plain KV
  (evict compress-state last, or pin it independently) — a page whose KV
  sank to L3 but whose compress-state was evicted is a page that must fully
  recompute anyway, defeating the point.
- L3 (NVMe, `kv-native-sys`) persistence must cover the compress-state pool
  too if the "restart-surviving prefix cache" framing (#83's own title) is
  to hold for DSv4 — a restart that recalls plain KV bytes from disk but not
  the compressor/indexer state at those same positions is only half-restored
  and would need the same page recomputed anyway.
- **Sequencing implication**: don't build the GPU-resident page-addressable
  pool as a standalone piece and bolt on tiering later — design the
  page-key/state_loc scheme so it's `CudaKvTierStore`-compatible (same key
  space semantics as plain FlashMLA pages) from the first cut, per §0's
  "budget whole chain no half-steps."

## Buffer enumeration — what changes, what stays

Per (layer, compress_ratio) class, contrasted with Route B's per-slot
enumeration (`docs/plans/2026-06-11-dsv4-whole-slot-kv-swap.md` table):

| Buffer | Route B (today) | Route A (target) |
|---|---|---|
| FlashMLA FP8 KV pool | slot-ranged, whole-band snapshot on restore | **unchanged** — already page-addressable via `flashmla_page_table`/`mirror_band`; Route A reuses this as-is for the plain-KV half |
| compressor `pending_kv`/`pending_score` | `Dsv4CompressorImage`, captured/restored with the whole slot | New page-addressable pool, keyed by `state_loc(page_id)`; capture happens naturally as a byproduct of the block completing during ANY forward pass (fresh or cache-extending), not as a separate snapshot step |
| compressor `prev_overlap_kv`/`prev_overlap_score` | same | same new pool, resolved via position lookup at `write_pos - compress_ratio`, not a trusted "last write" register |
| `sw_window_cache` (ring) | whole-ring snapshot | Needs the SAME treatment — ring residue is position-dependent, not slot-dependent; likely a second page-addressable pool keyed by `pos % sliding_window` at the SWA-page granularity SGLang uses (`swa_page_size`), not the raw ring index |
| `dsa_official` state | whole-band snapshot | Same pattern as compressor — page-addressable, boundary-keyed |
| flashmla bootstrap scalars (`fp8_kv_sw_bootstrapped`, `fp8_kv_comp_packed_rows`) | slot-scoped booleans | Re-derivable from whether the relevant page range is resident in the new pool — likely eliminable rather than ported |
| MTP draft rollback (`spec_rollback`, `restore_spec_ring_tail`) | position-exact, just-captured, immediate use | **Unchanged, out of scope** — a different, still-correct mechanism (small-depth, immediate capture-then-restore around one perturbation, never a stale arbitrarily-old snapshot); confirmed by #152 not to generalize to or need touching for the prefix-cache case |

## What's NOT solved by this design — name it, don't discover it late

- Reuse granularity is checkpoint-dependent (`lcm` of active `compress_ratios`
  across layers) — must be computed and validated once at model load, not
  assumed constant across models. A checkpoint with a much larger ratio
  spread than today's `[4, 128]` could produce an impractically large
  minimum reuse unit.
- The new pool's content is only as good as the FIRST forward pass that ever
  wrote a given page-aligned position — a bug in that write path corrupts
  every future reuse of that boundary silently (same class of risk this
  whole investigation exists because of; needs its own correctness gate,
  not just a perf bench, before any default flip).
- Multi-rank (TP=8/EP=8) lockstep: Route B's whole-slot swap already solved
  demote/promote lockstep via the multiproc relay (`docs/plans/2026-06-11-dsv4-whole-slot-kv-swap.md`
  §"TP=8/EP=8 lockstep"). Route A's page-level granularity means MANY more,
  smaller lockstep events instead of one whole-slot event — needs its own
  design pass, not an assumed drop-in of the existing relay pattern.

## Sequencing (proposed, not yet started)

1. Compute-and-validate the checkpoint-derived reuse granularity
   (`lcm(page_block_size, compress_ratios)`) — source-only, no GPU needed.
2. Single-rank, GPU-resident-only (no L2/L3 yet) page-addressable compressor
   pool for ONE layer class (e.g. compress_ratio=4 only) — smallest possible
   slice to prove the `state_loc`/overlap-lookup scheme against a real
   needle-gate correctness check (not byte-identity — this investigation's
   whole day argues for the correct-inference gate, `needle_gate.py`).
3. Extend to `sw_window_cache` (the ring — same boundary-keyed pattern,
   likely reusable machinery from step 2).
4. Wire into `CudaKvTierStore`'s existing L2/L3 demote/promote path (this
   doc's L2/L3 section) — namespace it, do not build a parallel tier.
5. Multi-rank lockstep design pass, informed by step 2-4's actual event
   granularity.
6. Delete Route B's `Dsv4LayerImage`/`swap_out_image`/`swap_in_image`/
   `capture_cached_prefix`/`restore_cached_prefix`/`truncate` machinery
   (~1075-1100 lines across `attention/dsa.rs`, `dsv4.rs`, `executor.rs`,
   `infer-core/src/prefix.rs`+`lib.rs` — full inventory in the investigation
   doc's "Route A terrain map" section) once Route A's gate passes and
   composes cleanly with #150's still-open concurrent-decode investigation
   (verify Route A doesn't reintroduce or mask that bug before deleting the
   fallback).

KILL criteria: if the page-addressable pool's needle-gate correctness check
ever fails in a way Route B didn't, stop, do not proceed to L2/L3 wiring or
deletion — the whole point is a design that's *more* provably correct than
Route B, not just faster.
