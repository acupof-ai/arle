# DSv4 Page-Granular Prefix Reuse (Route A) — #85, SGLang `CompressStatePool` precedent (compressor half only)

> Status: Active — step 4 (compress_ratio==4 compressor pool) implemented,
> compile-verified only; kernel numerics + needle_gate.py unverified (no nvcc
> locally). Steps 5-8 (ring/`dsa_official`, L2/L3 wiring, multi-rank, bench)
> not started.

Owner directive (ckl, 2026-07-08): DSv4's prefix reuse must stop snapshotting
at arbitrary positions. Capture and restore only at the minimal reusable
boundary — page-granular, not whole-slot. Confirmed against SGLang's actual
DeepSeek-V4 implementation before committing to a design (don't hand-roll
when upstream already solved it). **Scope of that confirmation**: covers the
compressor/indexer half only (see "What SGLang actually does" below) — the
SW-window-ring half has no confirmed upstream precedent and is ARLE-original
design, flagged where it appears.

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
| `swa_page_size` (paged boundary unit) | `page_block_size=64` fixed (`flashmla.rs:58-59`), a FlashMLA GPU-kernel tiling constant, independent of `compress_ratios` | **Decoupled from FlashMLA's 64** (ckl, 2026-07-08): the new pool's own page unit = one compress_ratio-block (i.e. page size 1 row, not borrowed from FlashMLA) — nothing about the new pool's storage needs 64-alignment. Reuse-granularity for a request = `lcm(active compress_ratios for this checkpoint, across all layers)` — computed once at model load from `DeepSeekV4Config.compress_ratios` (`crates/deepseek-spec/src/v4.rs:64`), no `page_block_size` term. Plain KV's own 64-page-granularity is enforced separately, only at the final min-available-length step (L2/L3 section) — floor the KV component to the nearest whole FlashMLA page, don't bake 64 into the new pool's own design |
| `get_raw_loc(write_positions - compress_ratio)` | `prev_overlap_kv`/`prev_overlap_score` single register, written in-place | Position-indexed lookup into the new pool, same function used for both the current-block write and the prior-block read |
| `hybrid_pool_assembler.py`'s hicache wiring | `Dsv4LayerImage` capture/restore is DSv4-only, whole-slot; `CudaKvTierStore`'s DSv4 usage (`slot_tier`) is *also* whole-slot today — no existing page-granular KV path to "join" | New `NS_COMPRESS_STATE` namespace, page-granular from day one, evicted **independently** of KV's (still whole-slot) tiering; a restore-time min-available-length rule reconciles the two (see L2/L3 section) |

## L2/L3 tiering — independent per-tier eviction, no coupling infra (ckl's design call, 2026-07-08 — same-day correction after adversarial review)

ARLE already has three tiers: **L1 GPU HBM → L2 host-pinned DRAM (#82,
`CudaKvTierStore`) → L3 NVMe local-disk (#83, `kv-native-sys`)**. The
original draft of this section rested on three separate legs: **(1)** adding
a namespace is cheap, **(2)** doing so makes reuse page-granular "for free"
via the same path plain FlashMLA KV pages already use, **(3)** L3 already
makes DSv4 prefix cache restart-surviving (by analogy with SGLang's unified
hicache). **Adversarial review against source confirmed (1) and found (2)
and (3) both false** — corrected below.

- **Leg (1), holds: namespace addition is cheap.** `NS_SLOT=1,
  NS_SLOT_CHUNK=2, NS_PREFIX=3, NS_PREFIX_CHUNK=4` are plain `u64` consts in
  an 8-bit (256-way) open namespace field (`executor.rs:1808-1816`,
  `TIER_NS_SHIFT=56` at `kv_tier.rs:156-167`) — no hardcoded assumption of
  exactly N namespaces. Adding `NS_COMPRESS_STATE`/`NS_COMPRESS_STATE_CHUNK`
  is a two-constant change.
- **Leg (2), false: plain FlashMLA KV pages do *not* already flow through a
  page-granular `CudaKvTierStore` path.** DSv4's existing `slot_tier`
  demote/promote
  (`demote_slot`/`promote_slot`, `executor.rs:2428`/`2462`) moves an
  **entire slot's** serialized KV+state as one blob;
  `BLOB_CHUNK_BYTES=16MiB` (`kv_tier.rs:159`,
  `insert_chunked`/`read_chunked` `kv_tier.rs:694-741`) is IO-layer chunking
  for the mmap/network transport, not a selectively-evictable unit. Only
  Qwen-dense's `recall_tier` is genuinely page-granular
  (`bytes_per_page = kv.storage_bytes_per_page()`, `executor.rs:908,4201`).
  `flashmla_page_table`/`mirror_band` is GPU-kernel addressing (which
  physical page backs which logical position, for the attention kernel) — a
  different axis from tier-store eviction granularity; conflating the two
  was the original draft's mistake.
- **Design decision (ckl, 2026-07-08): each tier/namespace evicts
  independently, no cross-namespace coupling.** `NS_SLOT`/`NS_PREFIX` KV
  (today whole-slot granularity), `NS_COMPRESS_STATE` (compressor pool, page
  granularity from day one), and the ring's own namespace (its
  overwrite-in-place lifecycle differs from the compressor's append-only
  one — see buffer table — so it must not share `NS_COMPRESS_STATE`'s LRU)
  each get independent LRU, independent promotion/demotion timing, **no
  shared validity bit**. Correctness is enforced entirely at restore time:
  **reuse length for a request = min over every buffer class that
  position's forward pass depends on** — KV, compressor state, ring state,
  and `dsa_official` state once its own namespace membership is resolved
  (open question, see buffer table) — whichever buffer's resident tail is
  shortest sets the actual boundary; anything past that recomputes. Simpler
  than coupling eviction policies, and the same "reject rather than patch"
  posture #152's fix already established for whole-slot restores, extended
  to per-buffer granularity.
- **Two different granularities feed the min, on purpose.** The
  compress-state/ring pools report resident length in units of one
  compress_ratio-block (their own native page size, decoupled from
  FlashMLA's 64 — see ARLE mapping table). Plain KV reports resident length
  floored to the nearest whole FlashMLA page (a multiple of 64) — its own
  granularity, unrelated to compress_ratio. The min-rule takes all
  candidates in **token units**, not page-count units, so mixing native
  granularities per buffer is fine; no forced shared page size across pools.
- **Consequence: plain KV stays whole-slot — an explicit scope boundary,
  not a silent gap, and not assumed temporary.** Because reuse length is the
  *min* across buffers, compress-state's page granularity only improves
  end-to-end reuse once plain KV is *also* evictable at compatible
  granularity — until then a preempted slot's KV is either fully resident
  or fully absent, which caps the real-world benefit regardless of how
  fine-grained compress-state gets. Making DSv4 plain KV page-granular in
  the tier store is separate, unscoped follow-on work (call it **Route
  A.5** if it's ever prioritized) — this doc's near-term win is the
  compress-state pool's own L2/L3 footprint reduction, not a full
  page-granular reuse story end to end.
- **Leg (3), false: L3 persistence does not already make DSv4 prefix cache
  restart-surviving, with or without a compress-state pool.** DSv4's
  `slot_tier` is never durably attached today — `set_kv_tier_disk` calls
  only ephemeral
  `set_disk` (`executor.rs:2390-2397`); `slot_tier.load()` has zero call
  sites. Only Qwen-dense's `recall_tier` has the manifest/epoch persistence
  path (`kv_tier.rs:255-316,407-490,788-829`). DSv4 prefix cache does not
  survive restart today, with or without this design — restart-survival for
  DSv4 is new work on the plain-KV side first; the compress-state pool
  inherits whatever persistence plain KV eventually gets, it cannot provide
  restart-survival on its own.
- **Sequencing implication (unchanged)**: design the page-key/`state_loc`
  scheme so it's `CudaKvTierStore`-namespace-compatible from the first cut —
  just don't assume that alone delivers page-granular *or*
  restart-surviving reuse for DSv4 end-to-end.

## KV budget — one path (`kv_budget_plan`), one pattern, not two (ckl, 2026-07-08)

**Correction from an earlier draft of this doc**: an earlier pass described
"fix the existing FlashMLA per-layer split bug" and "size Route A's new
pools" as separate concerns, one of them "land as its own PR, separate from
Route A." That framing was wrong. Checked directly: DSv4 has **exactly one**
budget-planning path — `Dsv4::kv_budget_plan()` (`dsv4.rs:1727-2011`), the
sole caller of which is `executor.rs:2277`, feeding the one
`Dsv4KvAdapter`/`Dsv4LayerKvLayout` construction chain. There is no second
DSv4 budgeting mechanism to reconcile — so this isn't "two paths, pick one,"
it's "one function, one term inside it currently breaks the pattern every
other term already uses, and Route A is about to add two more terms that
need to follow that same pattern." Fixing the outlier and adding Route A's
terms are the **same convergence**, not two independent changes that happen
to be nearby.

**Are there other paths? Checked, and named precisely:**
- Three doc comments near this code are **stale** — leftover names from
  before a rename, not evidence of a second live function:
  `Dsv4Model::kv_budget_num_slots` (`attention/dsa.rs:344-346,388,421-422`)
  and `crate::dsv4::Dsv4Model::dsv4_kv_budget_plan` (`attention.rs:1706`).
  Neither name exists on `Dsv4Model` today — only `kv_budget_plan` does.
  Fix these three comments in the same pass (delete-drift-first, `AGENTS.md`
  §Editing) — trivial, but a stale function name in a doc comment is exactly
  the kind of thing that makes a reader believe a second path exists when it
  doesn't.
- `Qwen3.5` has its **own**, separately-named `kv_budget_num_slots`
  (`qwen35.rs:2120`) — a genuinely different model family with a uniform
  per-slot KV cost (`per_slot_kv_bytes()`, no per-layer division at all,
  because Qwen3.5 has no DSv4-style compress-ratio/SlidingWindow layer
  heterogeneity). **Not a duplicate to merge** — it has no analogue of the
  bug below and needs no change. "One path" applies within DSv4's own
  budget function, not across unrelated model architectures.

**Why the FlashMLA term is the one outlier, not a separate bug class**:
inside `kv_budget_plan()`, every genuinely *per-layer, N-separate-buffers*
term already sums real per-layer cost — `mla_decode_bytes: usize =
self.layers.iter().map(...).sum()` (`dsv4.rs:1811-1821`);
`dsa_key_cache_per_slot`/`dsa_batched_per_slot` accumulate
(`saturating_add`) across the relevant layers (`dsv4.rs:1830-1863`). (The
model-wide `dsa_shared_bytes` scratch, `dsv4.rs:1751-1756`, is a different
thing entirely — ONE shared buffer for the whole model, not N per-layer
buffers, so computing it once from a representative layer is correct there,
not a related bug.) The **FlashMLA plain-KV band is structurally the same
kind of term as `mla_decode_bytes`** — one separate `TokenKVPool` per layer
— but is the only one using `.max()` + uniform-divide instead of sum. Route
A's new compressor/ring pools will be the same kind of term again (one
separate pool per layer, heterogeneous per-layer need). All three —
`mla_decode_bytes` (already correct), the FlashMLA band (currently wrong),
Route A's new pools (not yet written) — belong to the **same one pattern**:
sum real per-layer need, allocate each layer exactly what it needs. Fixing
the FlashMLA outlier isn't incidental cleanup alongside Route A — it's
making the function internally consistent *before* two more terms are added
that must follow the pattern it's supposed to already have.

**Pro-checkpoint generalization** (owner question: does this hold across
DSv4-family checkpoint sizes — today's `DeepSeek-V4-Flash` and a
hypothetical larger future "Pro" variant?). Grounded against
`dsv4.rs:1727-2011` — not `examples/dsv4_resident_ab.rs`'s own
`DSV4_FLASH_KV_BYTES_PER_TOKEN=584`/`DSV4_FLASH_BASE_LAYERS=43`, which are
illustrative harness-only constants, not production logic:

- Every term in `kv_budget_plan()` already reads `&self.config`/
  `&layer.attention`/`&self.moe_config` (`hidden_size`/`head_dim`/
  `kv_lora_rank`/`compress_ratios`/etc.) — a differently-sized Pro
  checkpoint's own `config.json` drives every term correctly, no code
  change needed for that part.
- **One real lookup-table gap, already fail-closed**: plain FlashMLA KV's
  `bytes_per_token` is a 2-entry match on `(qk_nope_head_dim,
  qk_rope_head_dim)` (`dsv4.rs:84-97`): `(448,64)→584` (Flash),
  `(kv_lora_rank,64)→656` (V32/GLM); any other combo `bail!`s at load
  time — a Pro checkpoint with a third combo **refuses to load, doesn't
  silently miscompute**, but needs a new arm added. Same pattern in
  `kv_types.rs:71-73`'s disk-fingerprint `stable_tag()` (no tag yet for the
  656 shape either, comment at `kv_types.rs:67-70`).
- `flashmla.rs:1055-1060`'s batched-decode fast lane hardcodes
  `head_dim==512` — a Pro checkpoint with a different head_dim falls back
  to the always-correct single-row path (no correctness bug, just no
  batched-decode speedup until someone adds that shape). Plain-KV, out of
  Route A's scope, a separate "Pro readiness" tracking item.
- Route A's own compressor/ring page byte size is already a pure formula
  from config (`last_dim = 2*(1+overlap)*head_dim`, no lookup table needed)
  — easier than the FlashMLA case, no Pro-readiness gap to track there.
- If/when L3 persistence for the compress-state pool is eventually built
  (doesn't exist for DSv4 today — see L2/L3 section above), its on-disk
  fingerprint needs the same per-shape tag treatment as `stable_tag()`.

**The fix — converge `kv_budget_plan()` onto the one pattern** (3 file:line
edits to the existing FlashMLA term, then Route A's new terms follow the
same shape from their first line of code; sequence the FlashMLA fix as the
first commit of this convergence — small, self-contained, provably correct
below — so Route A's new terms are written against an already-consistent
function rather than next to a known outlier that still needs a second
pass):

1. `dsv4.rs:1951-1966` — keep per-layer page counts as a `Vec<usize>`
   instead of collapsing via `.max()`; add `total_slot_pages: usize =
   per_layer_pages.iter().sum()`.
2. `dsv4.rs:1867-1992` — replace the `pool_budget_bytes_per_layer`
   (divide-by-layer-count) computation with a single
   `pool_affordable_slots = pool_budget_total_reduced / (total_slot_pages *
   flashmla_page_bytes)`. `pool_budget_total` still needs the existing
   per-rank `mem_get_info`-based computation and cross-rank min-reduce
   (`all_reduce_min_scalar_i32`, same mechanism — just reduce the
   pre-division total instead of the post-division per-layer figure);
   `total_slot_pages * flashmla_page_bytes` is config-derived and already
   rank-uniform, no separate reduce needed for that factor.
3. `Dsv4KvAdapter::new`/`Dsv4LayerKvLayout::new`
   (`kv_layout.rs:411-453,980-1043`) — **drop the
   `flashmla_pool_budget_bytes_per_layer` parameter entirely** (net
   simplification, not just a fix): each layer already has
   `num_slots`/its own `flashmla_slot_pages`/`flashmla_page_bytes` locally
   (`kv_layout.rs:991-1008`), so `let budget_bytes =
   num_slots.saturating_mul(flashmla_slot_pages).saturating_mul(flashmla_page_bytes);`
   replaces the passed-in uniform value directly. `Dsv4KvBudgetPlan`
   (`dsv4.rs:51-53`) drops the `pool_budget_bytes_per_layer` field —
   downstream only ever needed `num_slots`.
4. **Then** — Route A's compressor/ring pools plug into the same, now
   fully-converged function as new summed-per-layer terms, exactly as
   described in the Pro-checkpoint bullet above. No third pattern, ever.

**Expected gain** (illustrative only, from the test fixture — **not** a
production-checkpoint measurement): old scaling factor `num_layers ×
max_layer_pages` vs new `Σ per_layer_pages`. For the fixture's 6-layer cycle
(`{64,576,64,80,64,192}`): `6×576=3456` vs `Σ=1040` — a **~3.3× larger
`num_slots`** for this specific layer-type mix. Re-measure against a real
DSv4 checkpoint's actual layer distribution before treating the magnitude as
real; only the fix's *correctness* (allocations become exact, proven below)
is established by the math alone, not the *size* of the win.

**Risk resolutions**:

- *Does `ensure!(pool.max_total_pages >= num_slots*flashmla_slot_pages)`
  (`kv_layout.rs:1048-1055`) survive the fix?* **Proven, not just hoped.**
  With `budget_bytes = num_slots*flashmla_slot_pages*flashmla_page_bytes`,
  and `PackedBytes` confirmed to have `has_scales()==has_norms()==
  needs_work_buffer()==false` (`kv_types.rs:203-209`), `total_bytes_per_token`
  in `compute_budget_breakdown` (`paged_kv.rs:185-186`) reduces to exactly
  `bytes_per_token` — making `max_total_tokens =
  budget_bytes/bytes_per_token` an **exact** integer division (no
  remainder, since `budget_bytes` is constructed as a multiple of
  `bytes_per_token`), and `max_total_pages =
  max_total_tokens.div_ceil(page_block_size)` likewise exact. The `ensure!`
  becomes a true equality, zero slack, mathematically guaranteed — keep it
  as a cheap defensive invariant (catches a future regression); it will not
  fire spuriously.
- *Does this need its own correctness gate before landing?* Yes: run
  `needle_gate.py` at 3 context lengths (small, `max_seq_len/2`,
  near-`max_seq_len`) comparing decode correctness before/after, plus one
  admission-boundary case (a `max_seq_len` sized to nearly exhaust the new,
  larger `num_slots` budget) to confirm the tighter fit doesn't starve the
  last slot by one page. Pure allocation-accounting change, expected
  byte-identical decode output — the gate is about catching an off-by-one
  in the new arithmetic, not numerics drift.
- *Can the FlashMLA fix and Route A's new terms still land as separate
  commits?* Yes — "one path/one pattern" is a design and sequencing
  decision, not a mandate to squash everything into one diff; small,
  self-contained commits per `AGENTS.md` §Git still apply. The FlashMLA fix
  lands first, alone, gated by the correctness check above; Route A's new
  terms land afterward, already written to match.

## Buffer enumeration — what changes, what stays

Per (layer, compress_ratio) class, contrasted with Route B's per-slot
enumeration (`docs/plans/2026-06-11-dsv4-whole-slot-kv-swap.md` table):

| Buffer | Route B (today) | Route A (target) |
|---|---|---|
| FlashMLA FP8 KV pool | slot-ranged, whole-band snapshot on restore | **unchanged, and still whole-slot in `CudaKvTierStore`** — `flashmla_page_table`/`mirror_band` page-addresses it for the GPU attention kernel, but that's a different axis from tier-store eviction granularity (see L2/L3 section); Route A does not change plain-KV tiering, only compress-state's |
| compressor `pending_kv`/`pending_score` | `Dsv4CompressorImage`, captured/restored with the whole slot | New page-addressable pool, keyed by `state_loc(page_id)`; capture happens naturally as a byproduct of the block completing during ANY forward pass (fresh or cache-extending), not as a separate snapshot step |
| compressor `prev_overlap_kv`/`prev_overlap_score` | same | same new pool, resolved via position lookup at `write_pos - compress_ratio`, not a trusted "last write" register |
| `sw_window_cache` (ring) | whole-ring snapshot | **Not confirmed against SGLang source** (unlike the compressor row above — no `swa_page_size`/ring-equivalent citation found in `compressor.py`/`hybrid_pool_assembler.py`; may be an ARLE-only design, not a port). Semantically different from the compressor pool: `compressed` rows are append-only (write once, read forever), the ring is **overwrite-in-place** (`ring_idx = pos % sliding_window`, `dsa.rs:589,1595` — a later write at the same residue destroys the earlier value once `seq_len > sliding_window`). A page-addressable ring pool needs an explicit **materialize/copy-out step at page-completion time**, not a pure position lookup like the compressor's overlap — this reintroduces a snapshot step, the exact class of thing #152 broke. MTP's `capture_sw_slot`/`restore_sw_slot` (`dsa.rs:1582-1616`) already touches this same array for its own one-row rollback; any storage-layout change here must keep that path in lockstep. |
| `dsa_official` state | whole-band snapshot | Same *pattern* as compressor — page-addressable, boundary-keyed. **In scope, not deferred** (ckl, 2026-07-08: GLM-5.2 must be covered, not punted): GLM-5.2 uses `dsa_official`/SparseIndexed as its primary path (`compress_ratios` all-zero, `glm.rs:262`), so this row is GLM's equivalent of the compressor row above, not an optional extra. Still must resolve, as part of step 1's source-only investigation (not left dangling): is its write semantics append-only (compressor-like — shares `NS_COMPRESS_STATE`'s eviction lifecycle) or overwrite-in-place (ring-like — needs its own materialize step and namespace)? |
| flashmla bootstrap scalars (`fp8_kv_sw_bootstrapped`, `fp8_kv_comp_packed_rows`) | slot-scoped booleans | Re-derivable from whether the relevant page range is resident in the new pool — likely eliminable rather than ported |
| MTP draft rollback (`spec_rollback`, `restore_spec_ring_tail`) | position-exact, just-captured, immediate use | **Unchanged, out of scope** — a different, still-correct mechanism (small-depth, immediate capture-then-restore around one perturbation, never a stale arbitrarily-old snapshot); confirmed by #152 not to generalize to or need touching for the prefix-cache case |

## What's NOT solved by this design — name it, don't discover it late

- Reuse granularity is checkpoint-dependent (`lcm(active compress_ratios
  across ALL layers)` — no `page_block_size` term, see ARLE mapping table
  above for why the new pool's page unit is decoupled from FlashMLA's 64) —
  must be computed and validated once at model load, not assumed constant
  across models. Corrected against source: the test fixture
  (`deepseek-spec/v4.rs:1320`) shows real per-layer heterogeneity — `{0, 4,
  16, 128}` (0 = pure SlidingWindow, no compressor state at all for that
  layer) — not a uniform `[4, 128]`; no confirmed production DSv4 checkpoint
  value found in-repo, treat `[4,128]` as illustrative only. **GLM-5.2 is
  in scope for the design overall (its `dsa_official` row must be covered,
  see buffer table) but out of scope for *this specific* lcm derivation** —
  it sets `compress_ratios` all-zero and uses `dsa_official`/SparseIndexed
  instead (`glm.rs:262`), a structurally different mechanism with its own
  boundary question, not this formula. The lcm-across-all-active-ratios math
  is still correct in principle (a request's prefix-match position must be
  uniform across layers, so the tightest global constraint governs). Caution
  — "the coarsest ratio governs" is only true when every active ratio evenly
  divides the largest one: for the fixture, 4∣128, 16∣128, so
  `lcm(4,16,128)=128` happens to equal the largest ratio. That's a
  coincidence of this fixture's divisor structure, not a general property of
  lcm — a checkpoint with, say, ratios `{4, 6}` gives `lcm=12`, larger than
  either ratio alone. Re-derive per actual checkpoint; ratio spread is
  config-driven, not bounded by anything in code.
- **Chunked prefill — resolved (ckl, 2026-07-08): bubbles are acceptable,
  no special handling.** A chunk boundary that doesn't land on a
  compress_ratio-aligned position simply doesn't trigger a capture for that
  partial block — the page-addressable pool only ever gets written once a
  block is genuinely complete (buffer table, compressor row: "capture
  happens naturally as a byproduct of the block completing during ANY
  forward pass"), whether that completion happens within one chunk or is
  finished off by a later chunk. No accommodation needed in the chunked-
  prefill path itself; the cost is a small, accepted "bubble" (a
  not-yet-reusable tail) rather than reuse at every possible chunk boundary.
- **Radix-cache match-length quantization — must be added, not yet
  designed.** The existing prefix-match layer (`infer-core/src/prefix.rs`)
  reports a matched length at token granularity; Route A's restore boundary
  is only valid at the reuse-granularity computed above (per-checkpoint
  `lcm` of active `compress_ratios`). Whoever calls into Route A's restore
  path must floor the radix-cache-reported match length down to the nearest
  multiple of that granularity before using it — this floor step doesn't
  exist yet anywhere in the codebase and needs its own file:line home
  (likely the same call site that today enforces the exact-length-only
  restore in `restore_cached_prefix`, `executor.rs`).
- The new pool's content is only as good as the FIRST forward pass that ever
  wrote a given page-aligned position — a bug in that write path corrupts
  every future reuse of that boundary silently (same class of risk this
  whole investigation exists because of; needs its own correctness gate,
  not just a perf bench, before any default flip).
- **"Pro" checkpoint readiness is out of Route A's scope but tracked here so
  it isn't discovered late**: a future larger DSv4-family checkpoint with a
  new `(qk_nope_head_dim, qk_rope_head_dim)` combo needs a new arm in
  `Dsv4MlaKvArena::from_config` (`dsv4.rs:84-97`) and a new disk-fingerprint
  tag in `kv_types.rs:71-73`'s `stable_tag()` before it can load at all —
  both fail closed today (load-time `bail!`, not silent miscomputation), so
  this is an additive checklist item, not a latent correctness bug (see KV
  budget section above).
- **#150 (concurrent-decode digit corruption, still open,
  `docs/experience/errors/2026-07-06-dsv4-concurrent-decode-digit-corruption-unresolved.md`)
  is not provably independent of Route A.** Its one surviving lead —
  `radix_topk`'s unbounded write (`csrc/misc/dsv4_dsa_official.cu:613`) —
  sits in the same per-row page/index-slice territory Route A's
  page-addressable pool restructures. This is a **hard precondition, not a
  footnote**: root-cause #150 to file:line, or demonstrate the two memory
  regions are disjoint, *before* touching page/row layout (see sequencing
  step 0) — otherwise a post-Route-A change in #150's symptom is
  unattributable (fixed vs. blast-radius-masked, indistinguishable without
  decoding cases per CLAUDE.md §0's case-as-fact rule).
- Multi-rank (TP=8/EP=8) lockstep is **designed, not solved** — corrected
  from the original draft's claim that Route B already solved this.
  `docs/plans/2026-06-11-dsv4-whole-slot-kv-swap.md` *proposes* riding the
  `multiproc_relay.rs` control-plane relay
  (`RelayCoordinator::broadcast()`, `multiproc_relay.rs:887`) via new
  `SwapOut`/`SwapIn` `RelayEnvelope` variants, but as of 2026-07-08
  `RelayEnvelope` (`multiproc_relay.rs:439-494`) has **no such variants
  yet** — this is still an open Route B sequencing item, not shipped.
  Per-event cost, once built, is cheap (one-directional broadcast, no
  cross-rank barrier/gather — observed 18 demote/18 promote events across a
  stress run,
  `docs/experience/wins/2026-07-07-prefix-cache-graph-page-table-fix.md:140-146`),
  but page granularity multiplies event count by pages-per-slot — needs its
  own design pass against real measured event rates once Route B's relay
  lands, not an assumed drop-in.

## Sequencing — from-scratch rewrite, not staged coexistence (ckl, 2026-07-08)

**Sequencing philosophy, stated explicitly**: this is a clean rewrite, not
an incremental migration — delete Route B early, build Route A clean, then
verify and fix what the verification finds. Not "build Route A alongside
Route B, prove it, then delete Route B once proven" (the original draft's
posture, step 6 previously deleted Route B *last*). Consequence, named so
it isn't discovered later: between the deletion step below and Route A's
completion, `main` has **zero DSv4 prefix-cache reuse** — every request
fully re-prefills. Correctness-safe (full re-prefill is always correct,
same posture as #152's stopgap) but a real, if temporary, throughput
regression on `main` for however long steps 3-6 take — accepted deliberately
per this directive, not an oversight.

0. **Gate: root-cause #150 to file:line, or prove disjointness** from the
   page/row memory layout Route A will restructure. Blocks step 2 onward
   (any code touching page/row layout) — step 1 is pure source/config
   reading and may proceed in parallel, it does not depend on this gate.
1. Source-only investigation, no GPU needed, both items required before step
   3 starts:
   - Compute-and-validate the checkpoint-derived reuse granularity
     (`lcm(active compress_ratios across ALL layers)` — no `page_block_size`
     term, per the ARLE-mapping-table correction above), confirmed against
     an actual production checkpoint's ratio values, not just the test
     fixture.
   - Resolve `dsa_official`'s write semantics (append-only vs
     overwrite-in-place, buffer table) — required for GLM-5.2 coverage, not
     optional.
   - Design the radix-cache match-length floor (What's-NOT-solved section
     above): the exact file:line where a token-granular matched length gets
     floored to the nearest multiple of the computed reuse granularity,
     before Route A's restore path ever sees it.
2. **Prerequisite, land first, alone**: fix the FlashMLA per-layer budget
   split to the sum-of-real-per-layer-need pattern (KV budget section
   above, "The fix" — 3 file:line edits in `dsv4.rs`/`kv_layout.rs`), gated
   by the `ensure!`-exactness proof + 3-context-length `needle_gate.py`
   check described there. This makes `kv_budget_plan()` internally
   consistent (one pattern, no outlier) *before* step 3 adds a new term to
   it.
3. **Delete Route B now — DONE 2026-07-08, scope corrected from this
   step's original wording.** `Dsv4LayerImage`/`swap_out_image`/
   `swap_in_image`/`Dsv4SlotSnapshot` are **NOT** Route-B-specific — grepping
   the actual call graph before deleting found `demote_slot`/`promote_slot`
   (`executor.rs`, the L2/L3 idle-slot host-tier eviction this doc's own
   L2/L3 section keeps out of scope) share the exact same serialization
   machinery. Deleting them per this step's original literal wording would
   have broken the L2/L3 tier. **Actual Route-B-specific surface deleted**:
   `PrefixIndex`/`PrefixIndexEntry` + `NS_PREFIX`/`NS_PREFIX_CHUNK`
   (`executor.rs`), the DSv4 arms of `cached_prefix_match_len`/
   `capture_cached_prefix`/`restore_cached_prefix` (now no-ops/bail,
   `executor.rs`), `Engine::attach_cached_prefix` and its sole caller
   (`infer-core/src/prefix.rs`+`lib.rs`), and the `PrefixStoreMockExecutor`
   test harness + its two tests (dead once the engine no longer drives
   capture/restore). `swap_out_image`/`swap_in_image`/`Dsv4LayerImage`/
   `Dsv4SlotSnapshot`/`demote_slot`/`promote_slot`/`mirror_restore_pages`
   all **stay untouched** — still load-bearing for the L2/L3 slot-tier path.
   Verified: `cargo check`/`clippy` clean under `cuda,no-cuda` (Mac,
   `CUDARC_CUDA_VERSION=12080`, no nvcc) and under `cpu,no-cuda`;
   `cargo test -p infer-core` 88 passed (2 Route-B-only tests removed, not
   patched to pass). Gated on step 0 (confirmed disjoint, see step 0 above).
4. Single-rank, GPU-resident-only (no L2/L3 yet) page-addressable compressor
   pool for ONE layer class (e.g. compress_ratio=4 only) — smallest possible
   slice to prove the `state_loc`/overlap-lookup scheme against a real
   needle-gate correctness check (not byte-identity — this investigation's
   whole day argues for the correct-inference gate, `needle_gate.py`). Size
   the pool as a new config-derived term in `Dsv4::kv_budget_plan()`
   (`dsv4.rs:1727-2011`) from day one, following the same sum-per-layer
   pattern step 2 converged the function onto — not a fixed constant sized
   for today's one shipped checkpoint.

   **Design pass 2026-07-08 (source-only, corrects this step's own buffer
   scope before any code is written):**
   - **Only 2 of the 4 buffer-table buffers actually need the new pool.**
     `pending_kv`/`pending_score` do NOT: `dsv4_attention.cu:1115` —
     `pending_len = start_pos % ratio` — is provably `0` at every
     block-aligned position, which is the ONLY position Route A ever
     restores at (the reuse-granularity floor). The kernel
     (`dsv4_attention.cu:940-954`) only touches rows `< pending_len`, so at
     every valid restore boundary this buffer holds zero live rows by
     construction, not "usually empty." `pending_kv`/`pending_score`
     (`kv_layout.rs:4-5`, `ratio × width`) stay ordinary per-slot embedded
     scratch, unchanged — no capture, no restore, no pool membership.
     `state.compressed` (the per-row compressed output, `kv_layout.rs:8`)
     is separately already covered: it flows into the FlashMLA
     page-addressable band via `flashmla_pack_compressed_delta`
     (`attention.rs:1926`, called at `attention.rs:2711,3133`) — out of
     Step 4's scope, already handled by the (unchanged, still-whole-slot)
     FlashMLA pool. **Net: Step 4's new pool holds only
     `prev_overlap_kv`/`prev_overlap_score`** (`ratio × head_dim` bf16 each,
     `kv_layout.rs:42-49`) — the single-block "most recently completed
     block" carry register, read as `has_prev_overlap` input then
     overwritten once the current block completes
     (`dsv4_attention.cu:1059-1060,1165`) — ARLE's existing equivalent of
     SGLang's `write_overlap_loc = get_raw_loc(write_positions -
     compress_ratio)`.
   - **Addressing to mirror**: `Dsv4BlockMap::comp_row(r)`
     (`kv_layout.rs:202-230`, `sw_blocks + r/page_size, r%page_size`) is
     ARLE's existing "one compressed-row → page" function (the #146 fix).
     Specialized to page_size=1 (one page = one compress_ratio-block, per
     this doc's page-unit decoupling from FlashMLA's 64), the split
     collapses to a scalar: `state_loc(block_index) = block_index` where
     `block_index = start_pos / ratio`. Simpler than SGLang's own formula,
     which needs the split because SGLang's compressor state shares a ring
     with ephemeral decode addressing — ARLE's pool here is
     compressor-overlap-only, no ephemeral/paged dual-mode.
   - **`TokenKVPool` reuse, partial.** Its mutation API
     (`alloc_tokens`/`attach_pages`/`free_slot`/etc., `paged_kv.rs`) is
     entirely `slot`-keyed — not directly usable, since `state_loc` keys by
     absolute block position across ALL requests (cross-request-shared,
     the whole point of Route A), not per-slot. Its **physical-storage
     primitives ARE reusable slot-less**: `alloc_detached_pages`,
     `retain_pages`/`release_pages`, `copy_pages_to_host`/`_from_host`, raw
     `k_data_slice`/`v_data_slice` all operate on physical page ids with no
     slot involved. **New code still needed**: the `state_loc(block_index)
     → physical page` mapping table itself — nothing in `TokenKVPool`
     provides a shared, slot-less logical→physical map; it must be built
     new, modeled on the existing `Dsv4BlockMap`/`flashmla_page_table`
     pattern, and (per step 6) will need its own LRU/eviction identity
     separate from any per-slot table.
   - **Write/read redirect (file:line, not yet implemented)**:
     `attention.rs:7596-7597` resolves `prkv_ptr`/`prsc_ptr` from
     `state.prev_overlap_kv/_score.device_ptr_mut(...)` today — redirect to
     resolve from `compress_state_pool.page_ptr_mut(state_loc(start_pos /
     ratio))` instead; the kernel call itself
     (`attention.rs:7606-7660`) is unchanged, only where the pointers come
     from changes. **This is a real behavior change, flagged for whoever
     implements**: today's kernel reads-then-overwrites ONE register
     because it's always exactly one block behind. Route A makes this a
     genuine two-address op — READ `state_loc(block_index - 1)` (the block
     that just completed, feeding this call's `has_prev_overlap` input),
     WRITE `state_loc(block_index)` (this call's own completed block, for
     the NEXT caller). Collapsing these back to one address reintroduces
     the #151/#152 staleness species this whole doc exists to eliminate.
   - **`kv_budget_plan()` term**: `ratio × head_dim × 2 (kv+score) × 2
     (bf16) × compressed_capacity` per compress_ratio=4 layer, where
     `compressed_capacity = max_seq_len.div_ceil(ratio)` (mirrors
     `Dsv4CompressorState::device_bytes_for`, `kv_layout.rs:117-135`, which
     computes this exact quantity for the OLD per-slot sizing) —
     **`num_slots`-independent**, since this pool is now shared across all
     slots, not per-slot; insert alongside `mla_decode_bytes`/
     `dsa_key_cache_per_slot` (`dsv4.rs:1811-1863`) as a new summed term.
   - **Open, NOT resolvable from source alone — blocks implementation,
     named so it isn't discovered late**:
     1. Does the new pool fully replace the old per-slot
        `prev_overlap_kv`/`_score` fields for EVERY forward pass (not just
        cross-slot restore), or keep them as a same-slot-continuing fast
        path and only consult the pool on restore? The "capture happens
        naturally as a byproduct of ANY forward pass" phrasing implies full
        replacement, but that adds one page-pool indirection to every
        live decode step at a block boundary — a real perf question,
        needs a bench (H20 pod), not resolvable by reading source.
     2. How this term coexists in `kv_budget_plan()` with layers NOT yet
        covered (ratio=16, ratio=128, SlidingWindow ratio=0, per the
        fixture's `{0,4,16,128}` heterogeneity) — needs step 5's design
        before `kv_budget_plan()` can be written for real, not stubbed to
        one ratio.
     3. `needle_gate.py` correctness verification itself — requires the
        H20 pod, not resolvable from source.
5. Extend to `sw_window_cache` (the ring) and, per step 1, `dsa_official`
   (GLM-5.2's equivalent). **Not the same machinery as step 4** — both are
   overwrite-in-place, no confirmed SGLang precedent; each needs its own
   materialize/copy-out design and its own correctness gate. Update MTP's
   `capture_sw_slot`/`restore_sw_slot` (`dsa.rs:1582-1616`) in lockstep,
   since it touches the same array as the ring.

   **Design pass 2026-07-08 (source-only, corrected against production
   config values pulled 2026-07-08):**
   - **Correction, implementation pass 2026-07-08: `dsa_official` did NOT
     need zero new design — that earlier claim in this doc was wrong.**
     `packed_rows` is append-only (confirmed, `attention.rs:7987,8002`), but
     `dsa_key_cache` itself is a full-history buffer with **slot-keyed
     physical layout baked into a compiled CUDA-graph kernel**
     (`dsv4_dsa_build_select_meta_cuda`) — not shape-equivalent to the
     compressor's single-register `prev_overlap_kv/score`, which is why
     Step 4's direct-redirect pattern didn't transfer. Built
     `Dsv4DsaKeyCachePool` instead as a **write-mirrored shadow**, not a
     live-path replacement: every cache-write D2D-mirrors its newly-packed
     rows into the shared pool right after the existing per-slot kernel
     write; restore D2D-copies the matched prefix back into the freshly
     assigned slot's own band. The slot-keyed live kernels are untouched —
     more design work than planned, but no kernel-signature change needed
     (unlike the compressor pool's `overlap_page_stride`).
   - **The ring's write cadence is per-token, not per-block.**
     `update_bf16_sw_window` (`attention.rs:2024-2068`) writes ring slot
     `(start_pos + i) % sliding_window` exactly once when position
     `start_pos+i` is first generated; the slot is untouched again until
     absolute position `start_pos+i+sliding_window` (the wraparound). MTP's
     `capture_sw_slot`/`restore_sw_slot` (`dsa.rs:1582-1645`) is the right
     COPY PRIMITIVE (one-row D2D copy) but the wrong CADENCE — it captures
     immediately before a same-request speculative perturbation and
     restores within the same forward pass; Route A needs an earlier
     request's ring content readable by a LATER, different request, long
     after the ring may have wrapped past that position.
   - **Materialize design: periodic full-ring snapshot at
     reuse-granularity (128-token) boundaries, not per-token.** Since
     restore only ever happens at `lcm(active compress_ratios)`-aligned
     positions (128 for the real checkpoint, step 1's result), the ring
     only needs to be durably readable at 128-aligned positions. At each
     `pos` where `pos % 128 == 0` (checked right after
     `update_bf16_sw_window` returns, in whichever caller already tracks
     absolute position — `attention.rs:2600,3007,4443/4542`), copy the
     ENTIRE current ring (not just the newly-completed span) into a new
     pool keyed by `block_index = pos / 128`. This makes the ring's
     capture cadence materially different from the compressor's (coarse
     periodic full-copy vs. fine natural per-block append) — same tier
     eventually, different mechanism.
   - **Correction to this doc's own step-6 framing**: once materialized
     this way, a given `block_index`'s ring-snapshot entry is written
     exactly once and never touched again — just as LRU-safe as the
     compressor's append-only pool. The "ring's validity is invalidated by
     overwrite events, not LRU recency" reasoning in step 6 below is
     **wrong** as a justification for a separate namespace; the real
     reason for a separate namespace is size/ownership, not
     overwrite-unsafety (see step 6's own corrected design pass).
   - **Sizing — corrected against real production config (pulled
     2026-07-08 via pod read of `/host/DeepSeek-V4-Flash-FP8/config.json`):
     `sliding_window = 128`, MLA `head_dim = 512`** (`qk_rope_head_dim =
     64`; indexer `index_head_dim = 128` is a separate, smaller sizing for
     `dsa_official`'s own pool). Per-snapshot cost = `sliding_window *
     head_dim * bf16` = `128 * 512 * 2` = 128 KiB; total ring pool =
     that × `max_seq_len / 128`. Compressor pool total (compress_ratio=4) =
     `2 * head_dim * bf16 * max_seq_len` (the `ratio` and its own
     `max_seq_len/ratio` page count cancel). Ring-pool-to-compressor-pool
     ratio = `sliding_window / (2 * 128)` = `128/256` = **0.5 — the ring
     pool is about HALF the compressor pool's size**, not the "1-2 orders
     of magnitude bigger" this doc originally speculated before the real
     `sliding_window` value was known. Also confirmed: 44 `compress_ratios`
     entries (43 hidden layers + 1 MTP layer); **3 layers are pure
     SlidingWindow** (ratio==0, indices 0/1/43 — need the ring pool, no
     compressor pool) and **41 are CompressedSparse** (ratio 4 or 128,
     alternating — need the compressor pool, per step 4's existing scope
     only ratio==4 is built; ratio==128 layers are still step-5-adjacent
     unfinished scope, not yet covered by either step 4 or this ring
     design).
   - **Not resolvable from source alone**: whether copying the FULL ring
     at each boundary (simple, always correct, some redundant bytes when
     `sliding_window > 128` — here `sliding_window == 128` so a full copy
     is actually no more than one reuse-granularity's worth, making this
     point mostly moot for the real checkpoint) is preferable to an
     incremental copy of only the newly-written span; GPU verification
     that the copy timing never races a concurrent read of the ring by the
     SAME forward pass.
6. Wire the compressor pool into `CudaKvTierStore` as `NS_COMPRESS_STATE`,
   and the ring/`dsa_official` pools (step 5) as their own **separate**
   namespaces (e.g. `NS_SW_RING_STATE`) — none share an LRU, since the
   ring's/`dsa_official`'s validity is invalidated by overwrite events, not
   LRU recency, unlike the append-only compressor pool.
   **Independent eviction per namespace**, no coupling infrastructure;
   enforce the min-available-length restore rule (this doc's L2/L3 section,
   mixed granularities, floored per-buffer) at request-admission time.
   Plain KV's own tiering stays whole-slot — out of scope here (Route A.5
   if ever prioritized).

   **Design pass 2026-07-08 (source-only, grounded in Step 4's actual
   shipped `Dsv4CompressStatePool`, not the abstract description above):**
   - **Correction — the ring's "not LRU-safe" framing above is wrong**,
     per step 5's own design pass: once the ring's materialize step copies
     out a periodic full-snapshot at a 128-boundary (not raw overwriting
     ring content), the MATERIALIZED entry is immutable and just as
     LRU-safe as the compressor's. The real distinguishing property between
     namespaces is size/ownership, not overwrite-safety — still separate
     namespaces, different reason.
   - **`CudaKvTierStore`'s chunking (`insert_chunked`/`read_chunked`,
     `kv_tier.rs:694-757`) already works at sub-16MiB blob sizes** — no new
     plumbing needed for that. But `host_capacity_pages =
     budget_bytes/bytes_per_page` (`kv_tier.rs:325`) counts **keys**, not
     bytes, and `bytes_per_page` is ONE value per store INSTANCE. DSv4's
     existing `slot_tier` is sized for 16 MiB whole-slot chunks
     (`BLOB_CHUNK_BYTES`, `kv_tier.rs:159`) — sharing that instance with a
     few-KB compress-state page would count it as "1 of N 16-MiB slots,"
     wildly under-representing real DRAM use and evicting far too
     aggressively. **`NS_COMPRESS_STATE`/`NS_SW_RING_STATE` each need their
     OWN separate `CudaKvTierStore` instance**, constructed with that
     namespace's own real page byte size — a new field on the DSv4
     executor (alongside `slot_tier: CudaKvTierStore`,
     `executor.rs:1798`), not just a namespace constant on the existing
     store.
   - **Real architectural gap, decided (ckl, 2026-07-08): build true
     eviction, not a write-behind backup.** `Dsv4CompressStatePool` as
     shipped (Step 4) has ZERO page-tracking — one big `alloc_zeros`'d flat
     buffer, always fully GPU-resident, addressed by direct array index
     (`compressed_base ± block` offset). Making it genuinely evictable
     needs: replace the flat buffer with a `TokenKVPool`-backed physical
     store (same primitive `flashmla_kv_pool` already uses,
     `KVFormat::PackedBytes`, `page_size=1`) plus a new
     `block_index → physical_page_id` table — `state_loc` stops being a
     direct array index and becomes a real page-table lookup. Comparable
     lift to Step 4 itself. (Rejected alternative: keep the flat buffer
     permanently resident and use the tier store only as an opportunistic
     write-behind backup for restart-survival — smaller change, but
     delivers no HBM footprint reduction, which is this doc's own stated
     near-term win for the compress-state pool.)
   - **Simplification found — Route A needs no bespoke restore function at
     all.** `CudaExecutorReal::reusable_prefix_blocks`
     (`executor.rs:339-345`) is hardcoded `Self::Dsv4(_) => 0` today —
     DSv4 never participated in the generic page-radix reuse path (Route B
     used its own bespoke position-0 store instead, now deleted). Making
     this arm real — a per-block residency predicate taking the MIN across
     every buffer class a block depends on (KV pages, compress-state page,
     ring-materialize page), same pattern Qwen's arm already uses
     (`pages_only_reusable_prefix_blocks(blocks, |key|
     self.tier.contains(key))`, `executor.rs:946-948`) — plugs DSv4 into
     the SAME generic `attach_prefix_to_request`/`clamp_prefix_to_backend`
     flow every other backend already uses
     (`infer-core/src/prefix.rs:35-146`). This is the actual min-available-
     length restore-time query the plan called for — simpler than
     rebuilding anything shaped like Route B's deleted `restore_cached_prefix`.
   - **Cross-rank sync for this namespace's writes — likely a non-issue,
     contingent on step 7's open precondition.** No one-to-all broadcast
     primitive exists in `tp.rs` today (`all_reduce_sum`/
     `all_reduce_min_scalar_i32`/`all_gather_bytes`/`all_gather_bf16_raw`
     are the full set; `all_gather_bytes` could serve as broadcast-by-
     convention but wastefully moves every rank's shard for a 1-rank-real
     payload). Per step 7's finding, if the compressor's input hidden state
     is already rank-replicated before `compressor_forward` (unverified,
     needs the pod check flagged there), every rank already independently
     computes the identical value — no sync needed at all, ckl's
     "one rank computes, syncs to others" directive becomes moot for THIS
     namespace specifically (each rank IS correctly the writer, redundantly).
     If that precondition fails, a real broadcast primitive needs adding to
     `tp.rs` first.
   - **Not resolvable from source alone**: whether the ring's own
     materialize buffer needs the identical separate-instance pattern
     (near-certain yes, same byte-accounting argument, not independently
     re-verified against the ring's own page size); step 7's replication
     precondition (blocks the cross-rank-sync question definitively either
     way).
7. Multi-rank lockstep design and build, **directly for Route A's
   page-granular events** — since Route B is already deleted (step 3), this
   is not "extend Route B's relay," it's a fresh design against real
   measured page-granular event rates from steps 4-6.

   **CLOSED 2026-07-09 — no build needed.** The compressor pool's
   no-relay-required finding (rank-diff MD5, bit-identical across 4 ranks —
   see the design pass above) is confirmed by source to extend to the ring
   and `dsa_official` pools: neither `update_bf16_sw_window`
   (`attention.rs:2034`, writes the ring) nor `csa_select_official`
   (`attention.rs:8972`, writes `dsa_official`) takes a `tp_world`/rank
   parameter — same architectural shape as `compressor_forward`, same shared
   replicated input (`k_prepared`/`hidden`). No new broadcast primitive
   built; `tp.rs` unchanged. This is source-level analogy, not a separate
   rank-diff measurement for these two pools specifically — flag as the one
   residual gap if a future correctness issue ever looks rank-dependent.
8. **Verify, then fix what verification finds** (final phase, not an
   afterthought):
   - *Correctness*: `needle_gate.py` across multiple context lengths,
     covering both the compressor and ring/`dsa_official` paths, GLM-5.2
     included.
   - *Performance*: a direct high-concurrency benchmark measuring **TTFT,
     TPOT (ITL), and throughput** — `scripts/bench_throughput.py` (the
     canonical tool, `AGENTS.md` §Benchmarks), swept across concurrency
     levels (c=1/4/8/16) and at least one long-context production shape, not
     a single smoke shape (distilled lesson: SLO verdict needs the SLO
     workload). Compare against the last Route-B baseline snapshot taken
     before step 3's deletion.
   - Any regression found here gets fixed in this phase — this is the
     explicit "verify + fix" tail of the rewrite, not a separate follow-up
     project.

KILL criteria: if the correctness gate ever fails, **stop and fix before
proceeding** — there is no Route B fallback once step 3 deletes it; the
only safety net is the min-available-length rule's reject-then-full-reprefill
degradation (correctness-safe, matches #152's existing posture). Also KILL
if the min-available-length restore rule is ever bypassed for any buffer (a
restore uses a buffer's stale/absent content without the cross-buffer length
check) — that reintroduces exactly the class of staleness bug (#151/#152)
this design exists to eliminate.
