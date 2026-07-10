# DSv4 KV reuse — abstraction/physical seam refactor (deletion-first)

> Status: Shipped — Phases 0-2a landed earlier; 2b (park deletion) + 3b
> (demand-paged bands) pod-gated 2026-07-10
> ([2b wins](../experience/wins/2026-07-10-dsv4-park-deletion-phase2b.md),
> [3b wins](../experience/wins/2026-07-10-dsv4-band-demand-paging-phase3b.md);
> E6 c=4 +3.8% wall miss documented there). Remaining 3c/3d items fold into
> follow-up issues.

## Verdict

DSv4 KV reuse has three addressing domains. Two of their three seams are
contractual; the third is a pile of conventions, and today's warm-cache
needle regression (solo miss 98.3% at HEAD, `job150b` runs) is that seam
failing. The refactor gives the seam one contract (Phase 0), deletes the
churn and the parallel direct-reuse machinery the contract obsoletes
(Phases 1–2), and makes the L2/L3 story honest — the tier plumbing shipped
with Route A is currently write-side dead code (Phase 3).

| Domain | What | Where |
|---|---|---|
| A — logical | radix host pages, matched_len, min-rule | `infer-core/src/prefix.rs`, `executor.rs:2498` `reusable_prefix_blocks` |
| B — physical | per-layer `TokenKVPool` band + 4 Route A pools | `kv_layout.rs` (`flashmla_kv_pool`, `Dsv4CompressStatePool`×2, `Dsv4DsaKeyCachePool`, `Dsv4SwRingSnapshotPool`) |
| C — kernel view | per-(slot,layer) persistent `device_page_table` (CUDA-graph-referenced) | `flashmla.rs:264` `refresh_device_page_table` |

Seam A→B (min-available-length + attach) is contractual — keep as is.
Seam B→C is convention: synced "exactly once per slot lifetime"
(`dsv4.rs:1165-1169`), plus per-prefill-row (`executor.rs:3166`), plus
post-attach (`executor.rs:2650`), and `e05a467e6` padded the host table with
repeated-last-page aliases to silence the one size guard that caught drift.
Result: decode grows the host band (`prepare_kv_batch` → `mirror_band` per
tick), the device table keeps prefill-time alias entries, the FlashMLA
kernel reads/writes band pages ≥ n through stale aliases onto the last real
page — clobbering prompt-tail KV, which is exactly where the needle lives.

## Phase 0 — B→C dirty-bit contract (fixes the regression class)

1. **Delete the padding at all three mirror call sites**
   (`kv_layout.rs:1568-1585, 1663-1674, 1732-1770`). Host `page_indices`
   holds ONLY real pages — the source of truth carries no fabricated
   entries. Padding to the kernel's fixed lsp-length format happens solely
   at the device-format boundary (`flashmla_page_table_padded_i32`,
   `kv_layout.rs:1005`, already exists for exactly this).
2. **`mirror_band`/attach set a per-(slot,layer) `device_table_dirty` bit**
   whenever the page list actually changes.
3. **Forward paths consume the dirty bit before kernel launch** (H2D of
   lsp i32s, outside graph capture — replay reads refreshed contents through
   the same persistent buffer pointer, the #8 design's intended use).
   Delete the three ad-hoc sync sites and the "exactly once per lifetime"
   contract comments — subsumed.

Padding was never the bug; staleness was. With tick-accurate sync the
pad-with-last-real device-format convention is safe: a page is real in the
host table (and the refreshed device table) before any kernel launch that
can touch it.

**Phase 0b — identity contract (restore-path review findings, 2026-07-09,
full table in #154).** The dirty-bit fixes coherence; the review found the
pool/mapping DESIGN also violates identity:

- **D1 (top)**: `Dsv4CompressStatePool` keys by absolute block index with
  NO content identity, and the live kernel routes `prev_overlap` through
  the shared pool whenever it exists (`attention.rs:7698-7714`) — two
  concurrent different-content requests clobber each other. SGLang keys by
  swa-page identity (radix-allocated = content-bearing); the port dropped
  that half. Immediate fix: live path back to per-slot registers (stride 0),
  pool becomes restore-only storage; durable fix: key the pool by the KV
  page id the block belongs to (content identity via the radix), not by
  position.
- **D2**: `host_to_flashmla` entries never die without radix eviction —
  recycled host page ids resolve to dead requests' physical bands.
  Lifecycle: remove on host-page free, not only on radix evict; assert
  owner-slot on resolve.
- **D3**: `claim_mirrored_page` ignores `page_ref_count` — a published
  prefix's physical pages can be zeroed/overwritten by a fresh slot while
  resident bits still advertise the boundary. Claim must respect published
  (`ref_count>0`) pages.
- **D4**: engine pop-trim (`prefix.rs:72-75`) runs AFTER
  `clamp_prefix_to_backend`, un-aligning `matched_len` → SW-ring restore
  silently skipped. Trim before clamp, and make the executor's alignment
  predicate a hard error, not a silent skip.
- **Restore completeness** (A2/A10+B1-B3/A14/A15/A16/A17): every ③ item in
  the #154 table gets disposition ① or ② with evidence — the same
  full-enumeration bar as the DSv4 EAGLE rollback anchor.

Gate: pod needle sweep — pt=462 solo ×15 must return to the `943bacda`
baseline (0/15 miss, byte-identical outputs), cache-on AND
`ARLE_DISABLE_PREFIX_CACHE=1`, plus a concurrent n≥2 unique-content sweep
clean of both signature classes (D1 kill-test arm included).

## Phase 1 — `mirror_band` churn deletion

`mirror_band` (`paged_kv.rs:847`) does mem::take + release-all + claim-all
refcount churn per layer per tick even when nothing changed. `mirror_slot`
(`paged_kv.rs:834`) already has the superset fast path — extend-only when
the new list is a superset. Port it; full release/claim only on real band
replacement (attach/free). Pure deletion of per-tick work; A/B-guard the
baseline tok/s.

## Phase 2 — implementation design (2026-07-10, supersedes the sketch below)

**One pool, content-keyed, copy-based, HOST-resident.** Per layer, a
`Dsv4PrefixStatePool` keyed by **host page id** (content identity: the
radix dedupes prefix chains, so matching prefixes share host page ids;
non-matching content never collides — the D1 flaw is unrepresentable).
Entries are written ONCE (first producer wins; content-identical by keying)
and live in host pinned DRAM — **the pool IS the L2 tier**; zero HBM
footprint, zero slot-capacity cost (the V5 class of regression cannot
recur). L3 = LRU spill of the same entries through `CudaKvTierStore`'s
existing mmap substrate under the `--kv-dram` budget.

Entry contents per (layer, host page k) — everything a restore at boundary
64·(k+1) needs, D2H'd at page completion from the single executor
choke point:
- 16 compressed FP8 band rows (contiguous 16×584 B slice via
  `Dsv4BlockMap::comp_row` — the ONE addressing source, B1-B3 lesson);
- 16 bf16 staging rows (`compressed.data`, A2 consumers);
- 16 dsa key-cache rows (paged data+scale copied via the same map);
- overlap kv/score registers at page end (`ratio×head_dim×2` bf16);
- at 128-aligned pages only: the full bf16 ring (ring restore source;
  FP8 ring region is NOT stored — restore sets
  `fp8_kv_sw_bootstrapped=false` and the existing bootstrap repacks from
  the restored bf16 ring, so the bf16 ring stays the single source, A14
  resolved by design).

**Write hook**: one executor post-forward choke point — for each slot,
for each host page completed this tick, D2H the entry (async, off the
critical path). No attention.rs changes.

**Restore** (`restore_prefix_sidecar` reinstated): for matched host pages
(present in DRAM or promoted from L3; per-page presence = the min-rule),
H2D pool→slot: band rows + staging + dsa rows + boundary overlap + ring;
set `compressed.seq_len`/`packed_rows`/`fp8_kv_comp_packed_rows` =
matched/4, `fp8_kv_sw_bootstrapped=false`, rearm the decode graph (A16),
reset `spec_slots` (A17), `seq_len=matched`. Tail re-prefills. Every ③
item in #154's table now has disposition ① or ② — the full-enumeration
bar, met by construction.

**Alignment (D4 fixed at the engine)**: `prefill_restore_boundary_alignment`
reinstated = `sliding_window` (128); the full-match pop-trim in
`infer-core/src/prefix.rs` must run BEFORE the backend clamp so the
clamped length stays ring-aligned. `reusable_prefix_blocks` returns
consecutive pool-present pages floored to the alignment.

**Gap audit (2026-07-10) — folded into the design:**
- **G13 (design-critical)**: overlap/ring are boundary-TRANSIENT (register
  overwritten every ratio tokens; ring wraps) — a post-forward hook can only
  capture the LAST boundary per forward call. Entries therefore split:
  *per-page section* (FP8 band rows / bf16 staging rows / dsa rows —
  positionally stored, capturable for every completed page) and
  *per-boundary section* (overlap + ring — captured only at prefill chunk
  ends + decode 128-crossings; `f317a7e27` already forces chunks onto
  128-alignment so every capture point aligns). Restore floor = highest
  128-boundary with a per-boundary entry whose lower pages are all present.
  v1 ships with chunk-granular prefill reuse; v2 (page-granular) = kernel
  per-block overlap output into a per-slot capture buffer (the kernel's
  `overlap_page_stride` param survives; no sharing, no D1 recurrence) —
  deferred until reuse-hit telemetry justifies it.
- **G1/G2**: host page ids RECYCLE — entries are provisional at capture,
  confirmed at radix publish, dropped unconditionally on page free/abort
  (the D2 lesson, now at L2). Whether generated-token pages enter the radix
  (multi-turn reuse) must be read from `prefix.rs`'s publish path, not
  assumed.
- **G6**: entry schema is per-layer-ratio-shaped (`rows_per_page = 64/ratio`;
  cr=128 rows span pages, stored at completion page; SWA layers ring-only;
  GLM dsa ratio=1) — no hardcoded cr=4 shapes.
- **G8**: capture sits AFTER the spec-decode commit point — rejected drafts
  never enter the pool. **G11**: pool + L3 entries carry the weight-epoch
  tag (#9 `recall_tier` pattern); weight hot-swap flushes. **G3**: TP ranks
  each hold a replicated copy (rank-identical writes proven) — 4× DRAM,
  named and accepted.
- **Publish is LAST-producer-wins, not first-wins as planned** (deviation,
  benign — `prefix_state.rs` `Dsv4PrefixStatePool::publish`): host page ids
  recycle when freed, so a
  republish under a recycled id MUST overwrite; a page a slot completes is
  either radix-shared (content-identical) or slot-exclusive, so the newest
  content is always correct.
- **Exhaustion cascade (2b/3b)**: HBM page exhaustion → preempt = flush
  in-flight D2H + free pages + requeue (completed pages are ALREADY in L2 —
  publish-is-the-demotion, zero-copy preemption); L2 over budget → LRU to
  NVMe; NVMe over soft cap → self-evict oldest (only here is reuse history
  lost). "Complete exhaustion" is not a correctness event anywhere on the
  cascade; admission watermark tunes preemption frequency. Evidence gate
  gains a lane: deliberately breach the watermark, verify preempt-resume
  correctness.

**Sequencing**: 2a pools + cross-request reuse + L2/L3 (this design) →
pod evidence gate → 2b park replacement (delete `Dsv4LayerImage`/swap,
promote = attach + tail re-prefill; preemption rides the cascade above) →
3b on-demand band paging (admission watermark; enablers already landed:
dirty-bit contract, `flashmla_alloc_append`, page-granular admission
machinery) — each with its own pod gate, one variable per gate.

## Phase 2 — direct reuse = cross-request reuse (delete Route B leftovers)

The whole-slot tier (`demote_slot`/`promote_slot`, `executor.rs:2672/2706`)
is the last consumer of the Route B serialization machinery
(`Dsv4LayerImage`, `swap_out_image`/`swap_in_image`, `Dsv4SlotSnapshot`,
`mirror_restore_pages`) — a second, position-exact capture path parallel to
Route A's boundary pools. The pools are written during ANY forward pass
including decode (verified: ring `attention.rs:2121`, compress
`attention.rs:7615/7701`, dsa shadow `attention.rs:9169`), so a preempted
slot's state IS already durably captured at its last
`sliding_window`-aligned boundary.

- `demote_slot` → release the slot's pages; nothing else to serialize.
- `promote_slot` → floor the resume position to the boundary alignment,
  run the SAME `restore_prefix_reuse_state` path (`executor.rs:2549`),
  re-prefill the tail (prompt remainder + already-generated tokens are all
  known token ids).
- Delete: `Dsv4LayerImage`, `swap_out_image`/`swap_in_image`,
  `Dsv4SlotSnapshot`, `mirror_restore_pages`, the `slot_tier` 16-MiB-blob
  namespace and its `BLOB_CHUNK_BYTES` sizing, plus their tests.

Cost: each promote pays ≤`sliding_window`(=128)-token re-prefill plus pool
promotes. Preemption is a capacity-pressure event, not steady state — the
#152 "reject/floor rather than patch" posture, extended to its logical end.
One mechanism for 直接复用 and 跨请求复用; two state machines become one.

Open item (pod verify before wide ship): the ring snapshot's D2D during
CUDA-graph capture caveat (`attention.rs:2099-2102`) — same caveat exists
today, Phase 2 widens its blast radius to preemption recovery.

## Phase 3 — L2/L3 wired and tested, converged on existing machinery

Owner mandate (ckl, 2026-07-09): L2/L3 must be actually wired in and
verified with evidence — not left as dormant plumbing — and the refactor
must NOT miss any existing tier implementation. Hard-to-discover parallel
implementations are themselves the defect; converge by deletion.

Facts from code: all three Route A pool `demote()`s are
`#[allow(dead_code)]` — nothing ever evicts, the four dedicated
`CudaKvTierStore` instances (`compress_tier`, `indexer_compress_tier`,
`dsa_official_tier`, `sw_ring_tier`) never receive a write, and every
`contains()` in the `*_boundary_available` checks is decorative. The pools
are flat, always-resident, HBM = f(max_seq_len) summed over layers.

### Inventory verdicts (Explore sweep, 2026-07-09 — full table in the
### session record; substrates S1=`CudaKvTierStore`, S2=`KvMmapStore`,
### S3=Metal sharded files)

| Mechanism | Liveness | Verdict |
|---|---|---|
| #1 Qwen dense page tier (`executor.rs:764`; demote `prefix.rs:410`, promote `executor.rs:1036`) | LIVE, default-on | **Keep — the convergence template for page-granular demote/promote** |
| #2 DSv4 whole-slot park `slot_tier` (`executor.rs:2672/2706`) | LIVE, opt-in, proven (21-slot disk round-trip, wins 2026-07-02) | Keep until Phase 2 replaces it with boundary restore; then delete |
| #3/#4 compress + indexer-compress tiers (`executor.rs:1843/1846`) | Write side DEAD (zero `demote()` callers) | **Delete both instances** — pools stay always-resident (eviction is kernel-gated, `kv_layout.rs:171-175`); no decorative plumbing |
| #5/#6 dsa-shadow + sw-ring tiers (`executor.rs:1849/1852`) | Write side DEAD | **Wire for real (3b below)** — these two are host-managed, pageable without kernel change |
| #7/#8 Qwen3.5 slot park + sidecar | LIVE | Keep, out of DSv4 scope |
| #9 Qwen3.6 `recall_tier` (`executor.rs:3479`) | LIVE opt-in, only durable instance (`set_disk_durable`+manifest/epoch) | Keep — the durability template |
| #10 Metal S3 tier (`kv_ssd.rs`) | LIVE, tested | Separate backend, separate convergence item (still on the sharded-file substrate CUDA abandoned, `kv_tier.rs:14-16`) — flagged, not this refactor |

Cross-cutting facts driving the design:
- Calling the existing `demote()` on any Route A pool frees ZERO HBM — the
  flat device buffer stays allocated; `resident=false` only round-trips
  bytes. True eviction requires a paged physical store + block→page table.
- `NS_SIDECAR=3` (`executor.rs:1859`) collides in value with
  `NS_COMPRESS_STATE=3` (`kv_tier.rs:175`) — benign today (separate store
  instances), exactly the hard-to-discover drift class. One namespace
  registry in `kv_tier.rs`.
- The seam ALREADY supports demoted-page attach (`PrefixBlock::DemotedKey`,
  `infer-seam/src/lib.rs:61-66`); DSv4's `reusable_prefix_blocks` breaks on
  the first non-ResidentPage (`executor.rs:2513`) — discarding capability
  the seam already ships.
- The 2026-07-09 L2-vs-L2+L3 guidellm bench recorded ZERO tier activity in
  BOTH arms (wins doc) — no existing evidence that DSv4 tiering fires under
  load; the evidence gate below is the first real exercise.
- Dead code to delete on touch: `DiskTier::mmap_path` (`kv_tier.rs:280`),
  `KvMmapStore::flush` (`kv-native-sys/lib.rs:329`).

Steps:

- **3a — pure deletion (immediate)**: delete tier instances #3/#4 + their
  `contains()` checks + the placeholder 64-MiB budget consts; unify the
  namespace constants into one registry; delete the dead substrate fns.
  Compress pools stay always-resident — an explicit, stated boundary
  (kernel-baked addressing; pod nvcc makes lifting it possible later, as
  its own planned change).
- **3b — ring + dsa-shadow pools become genuinely evictable**: flat buffer →
  `TokenKVPool`-backed physical pages (`KVFormat::PackedBytes`, the same
  primitive `flashmla_kv_pool` uses) + block→page table; budget term in
  `kv_budget_plan` (summed-per-layer pattern); LRU demote into their tier
  instances (#5/#6) under budget pressure — `demote()` finally has a
  caller; L3 spill via `set_disk` on those instances (ephemeral first;
  durable inherits #9's manifest pattern only if/when DSv4 restart-survival
  is scoped).
- **3c — DSv4 plain-KV page demote/promote**: adopt #1's proven path
  (`try_demote_pages`/`promote_prefix_pages` + `DemotedKey` handling in
  `reusable_prefix_blocks` and the attach flow through `host_to_flashmla`).
  Without this the min-rule caps end-to-end reuse at whole-slot-or-nothing
  — this is the piece that actually moves prefix-reuse capacity.
- **3d — evidence gate (pod, mandatory, all four)**:
  - *Eviction fires*: boot with a deliberately small tier budget, drive
    enough distinct prefixes to exceed it — demote counters > 0
    (`kv_tier_demoted_pages` metrics, `infer-server/src/metrics.rs:118-157`)
    and pool HBM residency dropping.
  - *L2 restore correct*: needle sweep where the matched boundary state
    must return from host DRAM — exact retrieval, no `738.`-class signature.
  - *L3 restore correct*: same, demoted through to disk (S2 mmap path).
  - *Perf*: guidellm c-sweep tiers on vs off — Δ% table in the wins entry.

## Sequencing

0. #150b bisect convicts (running) → Phase 0 lands, pod needle gate.
1. Phase 1 same pass (small, A/B-guarded).
2. Phase 2 after Phase 0's gate is green (it reuses the same restore path).
3. Phase 3: measurement table first, then wire-or-delete decision.

KILL: any phase whose pod needle gate fails stops the sequence — full
re-prefill is always the correct fallback (same posture as #152).
