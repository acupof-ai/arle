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

## Measured

8×H20 GPU 0, ThinkingCap-Qwen3.6-27B-FP8 + Qwen3.6-27B-DFlash, block 16,
`--spec-max-batch 16` (forcing the spec route past the c=1 gate),
8 requests per concurrency level × 64 max_tokens, temperature 0:

| aggregate tok/s | c=1 | c=8 | c=16 |
|---|---|---|---|
| DSpark before | 80.7 | 69.4 | 64.7 |
| DSpark batched | **80.2** | **91.8** | **88.1** |
| Δ | −0.6% | **+32%** | **+36%** |
| no-spec control | 36.0 | 96.2 | 106.0 |

`accept_rate` is unchanged (0.2426 at c=16), so the gain is entirely step cost.
Greedy output at c=8 is token-identical to the no-spec reference on three
prompts, i.e. the batched verify commits the same tokens the trunk would.

Per-slot step cost by batch width (`ARLE_DSPARK_PHASE=1`, medians, ms):

| rows | draft | snap | verify | commit | total | per slot |
|---|---|---|---|---|---|---|
| 1 | 4.4 | 0.3 | 24.6 | 2.9 | 32.3 | **32.3** |
| 8 | 34.1 | 2.8 | 72.4 | 22.0 | 131.2 | **16.4** |
| 15 | 62.1 | 5.6 | 134.7 | 40.5 | 242.7 | **16.2** |

Per-slot cost halves and then plateaus. The verify amortizes (16 → 240 chain
rows for 5.5× the time), but 45% of the plateau — draft 26%, commit 17%, snap
2% — is still strictly linear in B, and the verify keeps its own O(B) residue:
the multi-token prep is launched per row, and the linear layers' conv1d +
recurrent scan loop per slot (`LinearCore::Rows`) while only `LinearCore::Tables`
(T=1) collapses to one launch.

So DSpark is now 0.95× the plain path at c=8 and 0.83× at c=16 — up from 0.61×,
but still not a win above c=1. **`--spec-max-batch` keeps its default of 1.**
The next levers are the two per-slot phases (batch the draft across slots) and a
ragged `B×T` pointer-table variant of the conv/GDR kernels.

## Rule

Before building a batched path, check whether the kernel ABI is already ragged —
`(bsz, total_q, max_q)` with an indptr is a varlen contract, and two callers of
it that grew into two functions are a merge, not a rewrite. And measure the
phase split before attributing a concurrency loss to contention: a per-slot cost
that is *identical* at c=1 and c=16 is not contention, it is a serial loop.

And a flat roofline probe does not license a flat prediction for a *ragged*
batch: the 246-row prefill that cost the same as 54 rows was one contiguous
sequence. The B×T verify pays per-row prep launches and per-slot recurrent
scans that the probe never exercised — which is why 15× the rows cost 5.5×,
not 1.07×.
