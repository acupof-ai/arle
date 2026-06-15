# Lever 2 — batch the per-slot DRAFT + COMMIT (batched MTP residual ~30%)

Batched MTP fold WIN (+81% @c=12,
[wins](../experience/wins/2026-06-15-dsv4-batched-mtp-fold-win.md)) batched the VERIFY
but left DRAFT + COMMIT per-slot serial. Phase profile (`ARLE_DSV4_MTP_STEP_PROFILE`,
n=4): draft+cap 15.5ms + commit 18.5ms = **~30% of the 110ms wave**; verify 76ms (70%).
Goal: batch these two, mirroring the proven `forward_decode_batch_verify` per-slot
pattern (per-slot attention loop + batched point-wise). Gated `ARLE_DSV4_BATCHED_MTP`.

## Current state (the serial residual)
`spec_step_batched` (executor/spec_decode.rs):
- Phase 1 (draft): `for s in 0..n { capture_spec_rings(s); draft_chain(s) }` — each
  `draft_chain` loops `depth` levels × `mtp_forward_level(1 row, slot s)`. Fully serial.
- Phase 3 (commit): `for s in 0..n { truncate; restore_spec_ring_tail; commit_accepted_fold(s) }`
  — each `commit_accepted_fold` loops 60 layers × `commit_layer_fold(slot s)`. Fully serial.

## Design A — batched DRAFT (depth-sequential, batched over slots per level)
The chain draft is depth-sequential (level i+1 chains from level i's stream), but
WITHIN a level the N slots are independent → batchable. New `mtp_forward_level_batched`
(or extend `mtp_forward_level` to N slots): per level, ONE batched MTP-head forward over
N rows (one current draft token per slot) + N h_prevs:
- Head math (embed / enorm / e_proj / hnorm / h_proj / stream-combine) already batches
  over `m` rows (`dsv4.rs:3220-3251` `mtp_forward_level`) — feed N rows.
- The MTP head's ATTENTION is per-slot (each slot's KV ring at the draft position) →
  **per-slot loop** like the verify (tree-attn isn't needed; it's a 1-row decode per
  slot at `start_pos+level`). Reuse the per-slot decode-attention (device start_pos) into
  a combined `[N,hidden]` buffer.
- Returns N (next_token, stream) — one per slot. `spec_step_batched` phase 1 becomes:
  capture all slots (loop, cheap) → `for level: mtp_forward_level_batched(N)`.

Saving: the head-math point-wise (embed/proj/norm) amortizes over N (was N× serial);
the per-slot attention stays per-slot but the launches batch. Est: draft 15.5 → ~6-8ms.

## Design B — batched COMMIT fold (one 60-layer pass over all slots' accepted rows)
`commit_accepted_fold` (dsv4.rs:1305) loops 60 layers × `commit_layer_fold(slot)`. New
`commit_accepted_fold_batched(&slot_ids, &accepted_per_slot, &start_positions)`: ONE
60-layer pass; per layer, gather EACH slot's accepted rows from its `spec_normed[layer]`
into a combined `[Σaccepted, hidden]`, then per-slot `commit_layer_fold` writing each
slot's own ring (per-slot dispatch within the shared layer loop). Saving: the 60-layer
host loop runs once (was N×); the fold compute stays per-slot but launches batch +
`truncate_slot`/`restore_spec_ring_tail` still per-slot (cheap host ops). Est: commit
18.5 → ~8-10ms.

## §0.1 buffers (per-slot, no cross-slot aliasing — inherits the verify's discipline)
- **draft**: each slot's MTP-head ring write at `start_pos+level` (the draft layer-0
  ring) — already captured by `capture_spec_rings` pre-draft, restored by
  `restore_spec_ring_tail` (rejected tail). Batched draft writes the SAME per-slot rings;
  the combined `[N,hidden]` head buffer is SCRATCH. h_prev streams per slot.
- **commit**: `commit_layer_fold` writes each slot's `attention[layer]` ring + advances
  `slot.seq_len` (per-slot). The gather scratch `[Σaccepted,hidden]` is SCRATCH. Each
  slot reads its OWN `spec_normed[layer]` (persisted by the batched verify) — no cross-slot.

## Open questions for codex (discuss the details)
1. **Batched draft attention**: is a per-slot decode-attention loop (device start_pos)
   into a combined buffer correct for the MTP head, OR does the head attention need the
   tree-meta path? (The draft is 1 row/slot/level at `start_pos+level`, attending the
   committed KV + the prior draft levels' rings — is that a plain decode-attention or a
   chain?) — verify against `mtp_forward_level`'s current single-slot attention.
2. **Commit fold per-slot ring dispatch**: can `commit_layer_fold` be called per-slot
   inside a shared layer loop without re-resolving per-slot page tables each layer (the
   `flash.slot_idx` re-resolve cost)? Or batch the gather + N fold calls/layer?
3. **Variable accepted count per slot**: Σaccepted varies; the gather + fold must handle
   ragged per-slot accepted lengths. Cumsum offsets (like the verify's per-slot blocks).
4. **Is the saving worth it?** Both savings are mostly host-loop + launch amortization
   (the per-slot fold/attention COMPUTE stays per-slot). Net est draft+commit 34 → ~16ms
   → wave 110 → ~92 → ~+19% throughput. Confirm the compute-stays-per-slot assumption
   doesn't make this a wash (the verify won big because its MoE COMPUTE batched; here the
   draft/commit compute is small + per-slot — is the launch/loop amortization enough?).

## Code-read resolution (2026-06-15, before codex)
- **DRAFT amortizes a real MoE → GOOD ROI.** `mtp_forward_level` (dsv4.rs) is a FULL
  transformer layer: `mla_attention` + `dsv4_moe_forward` (weight-read-bound) + shared
  expert + all-reduce. So batching the draft over N slots amortizes the MoE — the SAME
  reason the verify won. Draft 15.5ms (n=4, fully serial today) → expect a verify-class
  amortization. **This is the bulk of lever 2's win.**
- **COMMIT is attention/KV-only → MODEST ROI.** `commit_layer_fold` (attention.rs:3670)
  = compressor/indexer ingestion + ring-K writes, **no MoE**. Batching only amortizes
  launches + the 60-layer host loop; the per-slot compressor compute stays per-slot.
  Commit 18.5ms → smaller gain. **Do the draft first (MoE win); commit is the follow-on.**
- Revised priority: **2a = batched draft (MoE-amortizing, verify-class win); 2b =
  batched commit (launch/loop amortization, modest).** Measure 2a alone before 2b.

## Verification
decode-read coherence c≥4 + needle + matched A/B (lever-2 vs lever-1-only, same binary,
`ARLE_DSV4_BATCHED_MTP_LEVER2` sub-gate) @c=8/12. Phase profile re-run to confirm
draft+commit shrank. License on net wall-clock + coherent.
