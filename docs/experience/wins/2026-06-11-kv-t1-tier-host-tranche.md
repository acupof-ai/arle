# KV T1 DRAM Tier — Host-Side Tranche (#82 A+B+C)

## Goal

Land the backend-neutral half of the T1 host-tier re-port (#82): radix
demoted-node representation, the executor tier seam, and engine
demote-on-evict / promote-on-match orchestration — fully unit-tested on CPU
mocks, inert for every backend that reports no tier store.

## Hypothesis

Prefix pages evicted under page pressure can demote into a backend host
store and re-enter via promotion on the next prefix match, with the engine
expressed entirely in host page ids + opaque `u64` tier keys (no device
types crossing the seam), and the non-tier path byte-for-byte unchanged.

## Params

Design (per the [#82 approach](https://github.com/cklxx/arle/issues/82#issuecomment-4678992591)):

- `RadixCache`: nodes gain `tier_key`; demoted nodes stay linked/matchable.
  `tiered_longest_prefix_match` walks through demoted blocks;
  `demote_block`/`promote_block`/`drop_demoted`/`evict_page` with cascade
  sever; invalidated keys land in a drained accumulator so no path leaks a
  store entry. Invariant: demoted subtrees are entirely non-resident
  (insert-revive restores residency before children can appear).
- Seam (`BackendExecutor`, default no-op / capacity 0):
  `kv_tier_capacity_pages`, `demote_prefix_pages` (synchronous-copy
  contract, prefix-accept count), `promote_prefix_pages`,
  `drop_kv_tier_entries`. `KvAllocator::free_detached_pages` added as the
  inverse of `alloc_detached_pages`.
- Engine: `evict_prefix_cache_for_pages` demotes-else-severs per LRU page,
  rotating the coldest demoted entry out when the store is full;
  `lookup_prefix_for_attach` promotes demoted match blocks into freshly
  allocated pages so the existing resident-only attach path applies
  unchanged; promote failure truncates the match and re-prefills the tail.
  `KvTierStats` on `/v1/stats` (`kv_tier` block) + `/metrics`
  (`arle_kv_tier_*`, 17-series exposition).

## Env

- Local Apple Silicon (M4 Pro); host-only code, CPU mocks
  (`TierMockExecutor` with capacity-capped `BTreeMap` store).

## Results

- `cargo test -p infer-core --release` — **45 passed** (6 new radix tier
  unit tests + 3 new engine tier tests + throughput-counter test).
- `cargo test -p infer-server --release` — 23 passed (17-metric exposition).
- `cargo test -p infer-seam --release` — 8 passed.
- clippy `-D warnings` clean on all three crates; CI-mirrored
  `cpu,no-cuda,cli` lane green.
- Engine scenarios proven on mocks: evict→demote→store, prefix-hit→promote→
  attach (counts as prefix hit; store entry dropped via key drain),
  promote-failure→sever+re-prefill (request still completes), capacity-1
  store rotates the coldest entry instead of refusing.

## Problems

- **Test-caught engine bug**: tier-LRU rotation deferred the dropped key to
  the post-batch drain, so the executor store was still full when the next
  demote in the same eviction batch ran — rotation never freed a slot. Fix:
  drain immediately after `drop_demoted` inside `try_demote_page`. This is
  exactly the class of ordering bug the mock-store tests exist for.
- No device backend implements the store yet — wall-clock claims are
  **deliberately absent**. The CUDA pinned-store tranche (default-on per
  ckl 2026-06-11, `--kv-t1-budget-bytes 0` to opt out) carries the pod
  needle-gate + matched A/B; until then this tranche is observability +
  semantics only.

## Learnings

- Keeping the promotion step *before* `attach_prefix_to_request` means the
  attach/clamp/refcount path needed zero changes — demoted blocks
  materialize into ordinary cache-owned resident pages. One canonical flow,
  no parallel attach path.
- The dropped-key accumulator (drain after every radix mutation batch)
  centralizes store-entry lifetime; per-call-site forwarding would have
  leaked on the insert-revive path, which is easy to forget because it runs
  inside publish, not eviction.
