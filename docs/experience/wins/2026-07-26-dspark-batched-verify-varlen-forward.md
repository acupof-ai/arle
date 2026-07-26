# DSpark verify batches across slots — one ragged `B×T` forward replaces N serial steps

## Context

DSpark spec decode was a 2.24× win at c=1 and a 39% loss at c=16
([serializes across slots](../errors/2026-07-26-dspark-spec-decode-serializes-and-loses-above-c1.md)).
The phase timer (`ARLE_DSPARK_PHASE=1`) located the whole cost, and it was not
contention:

| | steps | draft | snap | verify | accept_commit | append | **per slot** |
|---|---|---|---|---|---|---|---|
| c=1 | 49 | 4.5 | 0.4 | **24.6** | 2.2 | 0.6 | **32.4 ms** |
| c=16 | 845 | 4.4 | 0.5 | **24.7** | 2.2 | 0.5 | **32.3 ms** |

Identical. `dispatch_decode_rows`' `DecodeRoute::Dspark` arm ran
`for row in decode_rows { self.dspark_decode_row(row)? }` — N complete B=1 spec
steps per tick, so spec throughput was flat in concurrency (80.7 → 64.7 tok/s)
while the plain batched path scaled 36 → 106.

Verify is 76% of the step, and it is worth batching: a prefill-latency roofline
probe on the same serve shows a 246-row forward costs the same as a 54-row one
(600 → 641 ms, +7% for 4.5× the rows); cost turns linear only past ~500 rows.
c=16 × chain 17 = 272 rows sits inside that flat region.

## What Worked

The paged attention kernel ABI was already ragged — `(bsz, total_q, max_q)` with
`q_indptr` — and prefill (`1,T,T`) and batched decode (`B,B,1`) were two callers
of it that had drifted into two functions. No new CUDA kernel was needed.

- `PageMeta::for_rows` (`loader.rs`) is now the single page-table builder over
  `(slot, start_pos, len)` rows; `for_slot` and `for_decode_batch` are thin
  callers. `full_attention_paged_batch` is deleted (~183 lines) — one paged
  full-attention path dispatches all three shapes.
- The multi-token prep kernel reads one scalar `start_pos` off a table based at
  element 0, so a ragged batch launches it once per row at that row's column and
  page-list offsets (host mirrors `q_offsets` / `page_offsets`).
- Linear (gated-delta) layers: weight-heavy stages run once over all
  `Σ len_i` columns; only conv1d + the recurrent scan loop per slot
  (`LinearCore::{Rows,Tables}`), replacing `linear_attention_batch`.
- `forward_hidden_staged` takes a `&mut [LinearRow]`; `dspark_verify_logits`
  verifies every chain in ONE forward, chain `i` owning logits rows
  `[Σ_{j<i} len_j, ..)`.
- Draft `logits` / `q_probs` moved from the shared `DsparkScratch` into
  `Qwen35DsparkSlotState`: a batched tick drafts every row before verifying any,
  so the shared buffer would have handed row `i`'s accept row `B-1`'s
  distributions. `slot_state_bytes` reserves them out of the KV budget.
- Quant-KV pools cannot build a multi-row page table, so the spec gate clamps to
  1 there; those serves keep the c=1 win and fall to the plain path above it.

Net −7 lines across `infer-cuda` for a strictly larger capability.

## Rule

Before building a batched path, check whether the kernel ABI is already ragged —
`(bsz, total_q, max_q)` with an indptr is a varlen contract, and two callers of
it that grew into two functions are a merge, not a rewrite. And measure the
phase split before attributing a concurrency loss to contention: a per-slot cost
that is *identical* at c=1 and c=16 is not contention, it is a serial loop.

## Status

Runtime perf **pending-remote**: the A/B (c=1/8/16, `--spec-max-batch 16`, vs the
80.7 / 69.4 / 64.7 spec baseline and the 36.0 / 96.2 / 106.0 no-spec control) runs
on the 8×H20 box. Target from the phase split is ~3.5× on the spec path at c=16.
Correctness parity (needle ladder ×3) runs in the same pass.
