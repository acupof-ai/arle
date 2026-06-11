# DSv4 Whole-Slot KV Swap (Route B) — #84/#85

Owner directive (ckl, 2026-06-11): DSv4 must support the KV tier. Its KV is
not page-addressable (prefix cache disabled at
`crates/infer-api/src/loaded.rs` carve-out), so the page-granular T1/T2 tier
(#82/#83) cannot apply. Route B gives DSv4 multi-turn "KV stays hot"
semantics by demoting/promoting the **entire slot state at its exact
position** — no `start_pos > 0` attach, no page mapping. Route A
(page+sidecar boundary snapshots → cross-request prefix sharing) remains
#85's end state.

## What already exists (verified in source, 2026-06-11)

The EAGLE-rollback work (2026-06-06, `truncate_decode_len` enumeration)
already built the per-layer snapshot set Route B needs — as **device-side
D2D** copies for spec-decode rollback:

- `Dsv4CompressorStateSnapshot` (`crates/infer-cuda/src/attention.rs:136`):
  `pending_kv`, `pending_score`, `prev_overlap_kv`, `prev_overlap_score`,
  `compressed_seq_len`, with `capture_from`/`restore_to`.
- `Dsv4SlotState.spec_rollback: Option<Vec<Dsv4LayerAttentionSnapshot>>`
  (`crates/infer-cuda/src/dsv4.rs:239`) — the per-layer snapshot vector
  covering compressor + sliding-window + FlashMLA FP8 pool slot state.
- The MLA latent arena is slot-ranged inside `flashmla_fp8_kv_pool`
  (`attention.rs:279,547-555` — per-slot exclusive range, validated length).

Route B = the same enumeration, serialized **D2H** into a host store instead
of D2D into a rollback buffer.

## Buffer enumeration (§0.1 — per-buffer verdict)

Per slot, per CSA/HCA layer (`Dsv4LayerAttentionState`):

| Buffer | Verdict | Notes |
| --- | --- | --- |
| FlashMLA FP8 pool slot range (`fp8_kv_pool_len` bytes) | snapshot `[0, seq_len*584B)` | only the written prefix, not the max range |
| compressor `pending_kv` / `pending_score` | snapshot (full, ratio×width) | tiny |
| compressor `prev_overlap_kv` / `prev_overlap_score` | snapshot (full) | tiny |
| compressor `compressed` data + `seq_len` | snapshot `[0, compressed_rows_written)` | |
| indexer compressed cache (CSA layers, official DSA) | snapshot written prefix | same shape discipline as compressor |
| sliding-window ring slots | snapshot **whole ring + write cursor** when `seq_len ≥ sliding_window`; else written prefix | the EAGLE lesson: ring self-heal only below window |
| `start_pos_device`, `seq_len`, host-side positions | snapshot scalars | |
| `moe_decode_scratch`, `deepep_ll_scratch`, decode-graph scratch | **no snapshot** — overwritten per step from inputs | prove: zero reads of prior-step contents (grep each consumer) |
| MTP draft state (when `--spec-type mtp`) | reset on restore (`rearm_for_new_request` pattern) | draft re-warms; acceptance restarts |

Restore = exact inverse at the same `seq_len`/`start_pos`; the next decode
step continues as if never swapped. Gate: needle ladder ×3 same-config +
same-config-twice envelope with a forced swap mid-generation (NOT
byte-identity — MoE non-determinism).

## TP=8/EP=8 lockstep (the real design problem)

Slot state is sharded across 8 ranks. The seam hooks run on rank 0
(coordinator); demote/promote must execute on **every rank in lockstep**, or
the deterministic planner diverges and NCCL deadlocks:

- Ride the existing multiproc relay (`infer-server/src/multiproc_relay.rs`,
  `broadcast_tick`): add `SwapOut{slot, key}` / `SwapIn{slot, key}` control
  envelopes broadcast exactly like plan ticks. Each rank serializes its own
  shard into its own local host store (no cross-rank gather — the store is
  per-rank, keyed identically).
- Engine integration is the #84 preemption hook, NOT the radix: DSv4 has no
  radix pages, so swap keys come from the request handle. New seam surface
  (whole-slot flavor): `demote_slot(slot, key) -> bool` /
  `promote_slot(slot, key) -> Result<()>` / `drop_slot_entries(keys)`,
  default no-op, used by `requeue_preempted_decode` when the page path
  reports no tier. Re-admission promotes by key when the SAME request
  re-enters (handle match), else falls back to recompute.

## Size & store

~17 MB device bytes per 30K-token slot for the MLA arena (584 B/token) +
small sidecars, ×(layers on this rank). Reuse `CudaKvTierStore` with a
second key namespace (slot keys vs page keys — disjoint u64 ranges: top bit
set for slot keys).

## Sequencing

1. Host-side seam + engine hook + mock tests (CPU, same pattern as #82's
   tranche) — no GPU needed.
2. Single-rank DSv4 swap (world_size=1 lane) — snapshot serializer reusing
   the rollback enumeration; needle gate on the pod (or single-GPU H20
   slice).
3. Multi-rank relay envelopes + 8×H20 lockstep verify (the pod session this
   shares with #82/#83's pending gates — one pod window covers all three).
4. KILL criteria: if swap restore ever fails the needle gate, fall back to
   recompute permanently for DSv4 and keep Route A as the only path.

Route A (#85) builds on the same serializer: boundary snapshots at radix
page edges + `reusable_prefix_pages` clamp (Metal GDR precedent) — sequenced
after Route B proves the enumeration is complete.
