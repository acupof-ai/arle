# DSv4 batched MTP decode — the production concurrency win (理想态)

Target chosen by ckl (2026-06-15): **batched MTP** = batched amortization × MTP
~1.98× acceptance. Production DSv4 serves `--spec-type mtp --mtp-draft-tokens 2` at
**steady c≥4**. The N=4 dense flip (`e638bbad`) is inert under MTP (the `!spec`
guard); it only helps no-MTP serves. This is the real production lever.

## One sentence (understand-until-simple gate — PASSED)

Generalize the Phase A/B batched decode lane (per-slot attention + grouped MoE over
N slots) from **1-row-per-slot** to **(depth+1) chain-rows-per-slot**, and drive the
per-row `spec_step`'s draft + verify across all N slots at once, then per-slot
accept/commit/ring-restore.

## One measured number + hypothesis

- Production MTP @c=8 = **46.38 tok/s** (per-row spec, batched lane disabled).
- Batched-dense floor @c=8 = **73.65 tok/s** (+58.8%, [c-sweep]
  (../experience/wins/2026-06-14-dsv4-batched-decode-csweep-threshold-n4.md)).
- Batched-MTP **hypothesis**: batched amortization (MoE is 60.8% of GPU work, grouped,
  −66%/row) × MTP ~1.98 acceptance ⇒ **> 73.65** (best of both). Magnitude TBD by
  matched A/B — this is the license-or-kill number.

## The wall (the one hard sub-piece)

**Per-slot block-diagonal attention over N×(depth+1) rows** (draft + verify). Each
slot attends its OWN committed KV ring + its OWN chain prefix (tree attention); the
MoE / proj / norm / MTP-head math all batch trivially across rows (they already do:
`mtp_forward_level` batches embedding/rms_norm/e_proj/h_proj over `m` rows,
`dsv4.rs:3220-3244`). Attention is the only thing keyed per-slot.

## Current state — what already exists (substrate, NOT new infra)

| Piece | Where | Status |
|---|---|---|
| Batched decode lane (per-slot attn + grouped MoE over N) | `dsv4.rs::forward_decode_batch_stream_impl` (Phase A/B) | ✅ built, decode-read coherent c=4 |
| Single-slot batched verify (chain rows, tree attn) | `dsv4.rs:1529` `dsv4_mtp_batched_verify_enabled` → `forward_tokens_stream_impl(..,Some(sched))` | ✅ built (+63% per chain) |
| Tree-attention meta | `attention.rs:3870` `Dsv4TreeAttnMeta`; `dsv4.rs:289` `SpecVerifySchedule` | ✅ built |
| MTP head level (N-row head math, 1 slot) | `dsv4.rs:3182` `mtp_forward_level` | ✅ math batches; ⚠ 1-slot KV |
| Per-slot ring snapshot/restore | `attention.rs:3378/3418` capture/restore; `dsv4.rs:694/720` wrappers | ✅ built (proven rollback) |
| **Cross-slot batched spec_step** | — | ❌ **the gap** |

At c=8 MTP today: `forward_decode_batch` (executor.rs:1544) loops `forward_decode_row`
→ `forward_decode_tokens` → `spec_step` **8× serially** (one slot each). That serial
per-row loop is the whole inefficiency.

## Decomposition (file:line)

**Entry** — `executor.rs::forward_decode_batch` (≈1589 gate): add a spec-batched
branch. When `spec on && rows >= N`, call a new `spec_step_batched(&slot_ids,
&start_positions, &positions)` instead of the per-row `forward_decode_row` loop.
(Keep the per-row `spec_step` as the c<N / N=1 reference + the B=1 identity gate.)

**`spec_step_batched`** (new, `executor/spec_decode.rs`) — orchestrates over N slots:
1. **Batched capture** — loop `capture_spec_rings(slot_s)` for s in 0..N (per-slot
   rings independent; §buffers).
2. **Batched draft** — depth-sequential; at each level call `mtp_forward_level` with
   N rows (one per slot) + N `h_prev`. Head math already batches; the **draft-layer
   attention must go per-slot** (the wall — Stage 1 loops it; Stage 2 batches it).
3. **Batched verify** — ONE forward over all N chains (N×(depth+1) rows) via a
   batched `forward_tokens_stream_impl` with a per-slot `SpecVerifySchedule`. MoE
   grouped over all rows; attention per-slot block-diagonal.
4. **Per-slot accept/commit/restore** — loop s: `longest_accepted_prefix` + bonus →
   `truncate_slot` → `restore_spec_ring_tail` → commit (fold/re-forward) → set
   `spec_slots[s].pending/hidden`. Host-side accept + per-slot ring restore are cheap.

## Staged build (reuse proven infra first)

- **Stage 1 — batched MoE, per-slot looped attention** (= the Phase A substrate
  applied to draft+verify rows). Attention per slot (the existing single-slot verify
  attention, looped); MoE/proj/norm grouped over ALL N×(depth+1) rows. Captures the
  dominant 60.8% MoE amortization with **no new attention kernel**. This is the bulk
  of the win and reuses the proven Phase A path.
- **Stage 2 — batched attention** (block-diagonal tree FlashMLA over N×(depth+1) rows,
  extending the Phase A/B batched indices builder + sched_meta to include each slot's
  chain prefix + committed KV). The +3%-class refinement (cf. Phase B over Phase A).

Re-baseline after each stage; do not bundle.

## §0.1 mutated-buffer enumeration (per slot, the rollback focus)

Each slot's draft writes the frozen target layer's rings; the verify is pure (reads
frozen KV). Per slot s, looped — **no cross-slot aliasing** (each `Dsv4SlotState` owns
its rings), so correctness is inherited from the proven per-row path:

| Buffer | Disposition | Precondition |
|---|---|---|
| `sw_window_cache` ring slots `[start_pos..=start_pos+depth]` | snapshot pre-draft (`capture_sw_slot`), restore rejected tail `(accepted+1..=depth)` | per slot; restore window must match capture (`captured_start_pos/depth`) |
| `flashmla.fp8_kv_comp_*` ring slots | snapshot (`capture_fp8_slot`), restore rejected tail (`restore_fp8_slot`) | per slot; FP8 page table keyed by `flash.slot_idx` |
| `flashmla.fp8_kv_comp_packed_rows` | restore to `fp8_packed_rows_before` | per slot |
| `flashmla.fp8_kv_sw_bootstrapped` | restore to `fp8_bootstrapped_before` (P1-B) | per slot; false⇒force re-bootstrap |
| accepted slots `[0..=accepted]` | left to the commit re-forward / fold (overwrites) | — |

The DSv4-EAGLE-rollback anchor (2026-06-06: missed `pending_kv`/`prev_overlap` +
`sw_window`+`fp8_kv_pool`) is already covered by the per-row capture/restore; batched
MTP must loop the SAME calls per slot (do NOT re-derive a "batched snapshot" — loop the
proven per-slot one).

## Verification gates

- **decode-read coherence at c≥4** — THE batched-kernel correctness gate (the Phase B
  KILL/fix showed needle alone is insufficient): read the actual N-way generation,
  require coherent continuation per slot
  ([[feedback_spec_decode_gate_needs_multi_prompt]], needle ≥2 prompts).
- **B=1 identity** — `spec_step_batched` with N=1 must equal per-row `spec_step`
  (42.2 ms/step, needle exact).
- **self-consistency, NOT byte-identity** — MoE non-determinism
  ([[feedback_correct_inference_not_baseline_identity]]).
- **Perf** — matched same-binary A/B: per-row MTP vs batched MTP @c=4/8, ~2300-tok
  prod prompt. License only if batched MTP > batched-dense (73.65) AND coherent.
- Acceptance rate per row must match per-row MTP (~1.98) — a drop = a batched-draft bug.

## Implementation spec (impl-level, from reading the verify path)

The single-slot verify (`forward_tokens_stream_impl`, `dsv4.rs:2537`,
`verify=Some(sched)`) runs attention in 3 sub-modes (`dsv4.rs:2650-2810`):
1. **tree-attn batched lane** (`tree_meta` Some): ONE `mla_attention` over the chunk
   with `Dsv4TreeAttnMeta` (per-row positions + branch indices), host start_pos, no
   ring writes (`:2699-2736`).
2. **per-row ring-replay** (`:2737-2791`): loop rows, device start_pos, each attends
   its ancestor path's just-written KV (the needle-validated reference).
3. decode (seq_len==1).

MoE / HC / norm / head run over the whole `seq_len` chunk (token-independent).

**The new function** `forward_decode_batch_verify` = generalize
`forward_decode_batch_stream_impl` (`dsv4.rs:1717`) to **M = Σ_s (depth_s+1) rows
grouped by slot**, layer-major:
- per layer: **loop slots**, run that slot's verify attention (sub-mode 1 or 2, the
  PROVEN single-slot path) writing into the combined `[M, hidden]` `attn_out` at slot
  s's row block; then **batched MoE / HC / norm over all M rows** (the existing
  batched-lane MoE, `:2200-2316`).
- **Stage 1** = this restructure with per-slot attention (sub-mode 2 looped, or
  sub-mode 1 per slot) — NO new attention kernel → no Phase-B-class risk; the win is
  the MoE amortization over M rows.
- **Stage 2** = make sub-mode 1 span N slots (one FlashMLA over M rows, a
  `Dsv4TreeAttnMeta` with per-(slot,row) block-diagonal branch indices into each
  slot's KV — extends the Phase A/B batched indices builder).

`spec_step_batched(&slot_ids, &start_positions, &positions)`:
1. loop s: `capture_spec_rings` (§buffers).
2. batched draft: depth-sequential; per level call `mtp_forward_level` with N rows
   (head math batches; Stage 1 loops the draft-layer attention per slot, Stage 2
   batches it).
3. `forward_decode_batch_verify` over the N chains → per-slot (argmax, hiddens).
4. loop s: `longest_accepted_prefix` + bonus → `truncate_slot` →
   `restore_spec_ring_tail` → commit (fold/re-forward) → set `spec_slots[s]`.

Gated `ARLE_DSV4_BATCHED_MTP` (default OFF until pod-licensed); the executor gate
(`executor.rs:1589`) routes spec + rows≥N to `spec_step_batched`, else the per-row
loop (the B=1 / c<N reference).

## Out of scope

- EPLB (separate track; decode is weight-read-bound → likely a prefill lever,
  [scope](../research/2026-06-15-dsv4-moe-batching-eplb-scope.md)).
- deepep_ll transport for MTP (`mtp_forward_level` asserts allreduce, `dsv4.rs:3196`);
  batched MTP Stage 1/2 stay on the allreduce transport (production default).
- Width/tree drafting (chain-only per `spec_decode.rs:12`; deleted `94d91948`).
