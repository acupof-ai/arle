# Qwen3.6 slot-tier and sidecar lifecycle moves to infer-kvspace

Date: 2026-09-10 · Lane: lane/d3 · Step 3a slice 1 of the architecture refactor

## Context

Block 4 of the qwen35 executor split (~535 lines of KV/tier/sidecar
lifecycle in `executor/qwen35.rs`, measured at `f8dac665a`) is host logic
over the tier store: whole-slot demote/promote chunking, the recurrent
sidecar's boundary math, its radix-riding eviction map, and the
off-thread serialization. The device contact points are exactly four
(snapshot D2H, restore H2D, page mirror, whole-slot swap) and stay in the
executor. The cross-rank two-phase commit already converged in the seam
(PR #260), so the block had no remaining device dependency.

## What worked

- **One generic tier, one model-specific trait.** `KvSlotTier<S>` owns
  the `KvTierStore`, the sidecar eviction map, and the serialization
  thread; `S: SidecarSnapshot` (`serialize` / `deserialize -> Option`)
  is implemented for `Qwen35RecurrentSnapshot` in infer-cuda, so the
  tier never names the model's type and the restore probe's
  corrupt-payload-skip falls through the trait's `Option` instead of a
  model-specific path.
- **Boundary math as pure functions.** `sidecar_lstar`, `sidecar_mat_len`,
  `sidecar_periodic_savable`, `sidecar_candidates`, and the FNV prefix
  hash are free functions; 12 unit tests cover the alignment, the probe
  order, the tail-page eviction ride, the corrupt-blob fall-through, the
  async drain, and the whole-slot image round trip. `cargo tree -p
  infer-kvspace` still contains zero `cuda-kernels`.
- **The debug build caught a latent key-space bug.** `tier_key` packs the
  namespace into the top 8 bits and `chunk_sub` the chunk index into the
  low 16, so a key must fit 40 bits — but the sidecar key was a full
  64-bit FNV hash. In release the overflow is silent and save/probe
  rendezvous by symmetry (both sides mangle the same way); in debug
  `tier_key`'s `debug_assert!` panics. The old path ran release-only on
  GPU, so it never fired; the new host tests run debug and tripped it on
  the first run. `hash_prefix_tokens` now folds to 40 bits.
- **DSv4 shares the pure mapping.** Its `kv_tier_io_stats` aggregation
  maps the store's stats through the same `infer_kvspace::tier_io_stats`
  as the qwen35 arm; its inline 2PC demote/promote stays untouched
  (separate slice).

## Rule

A full-width hash handed to a packed key space works only by release-mode
symmetry — debug assertions exist to make that kind of contract visible,
and code that only ever runs release-only on GPU keeps the contract
invisible. Moving the logic to a host-tested crate is what surfaced it:
the test lane that matters is the one that runs debug.

## Net

`executor/qwen35.rs` loses ~250 lines (sidecar structs, the thread spawn,
the chunking, the eviction map, the probe loop, the tier passthroughs);
the crate gains `tier.rs` (490 lines including the 12 tests). Diff stat
vs `f8dac665a`: +543/-258 across 8 files.
