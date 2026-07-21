# DSv4 batched FlashMLA decode Phase B — correctness KILL (garbled decode at c≥4, faster-nonsense)

## Context
Phase B (commit `fd4d240f`) replaced
the per-row FlashMLA attention kernel in the batched decode lane with ONE
`sparse_decode_fwd(b=N)` (SW+HCA; CSA kept per-row). Gated
`INFER_DSV4_BATCHED_DECODE=1` + `ARLE_DSV4_FLASHMLA_DECODE_BATCHED=1`, default OFF.
Pod-verified on `fd4d240f` (8×H20). Hypothesis: batching the kernel reclaims the
per-row 25% launch-gap + gridX=1 occupancy → c=8 aggregate rises above the Phase-A
(per-row-attention) batched baseline of 73.65 tok/s.

## What happened (KILL)
- **B=1 identity PASS** — the `mla_attention` prepare/fwd split did NOT break the
  single-row path: needle 512/6000 exact ×3, 42.49 ms/step (≈ pre-split 42.7). The
  structural split is sound.
- **c≥4 batched lane produces GARBLED decode** (silent — 0 IMA/CUDA-error/panic):
  Phase-B `Rome. 意大利的首都 意大利的首都。`, `Ottawa { { { {. {.`,
  `Berlin. The capital of Germany of capital The.]] is Berlin`. Phase-A control
  (same binary, flag-OFF, per-row attention) on the SAME prompts is coherent:
  `Rome. The capital of Italy is Rome.`, `Ottawa. The largest city is Toronto.`.
- **First answer token correct, continuation drifts to garbage** → immediate-context
  attention is partially intact but the accumulating batched SW/HCA attention is
  wrong across decode steps.
- **The perf "win" is void:** Phase-B c=8 73.53 vs Phase-A 71.44 (+2.9%) ≈ the 73.65
  reference (−0.2%) — **the +20-40% launch-gap reclaim did NOT materialize even
  before correctness.** Faster nonsense.

| ctx | c | Phase-A tok/s (coherent) | Phase-B tok/s (GARBLED) | Δ% |
|-----|---|---|---|---|
| short | 4 | 61.92 | 58.27 | −6% |
| short | 8 | 71.44 | 73.53 | +2.9% (void) |

## Root Cause — TBD (prime suspect: shared sched_meta / batched-kernel math)
Not yet root-caused (needs the next iteration). Suspects, in order:
1. **`sched_meta` aliasing** — the Phase-A wins entry flagged that
   `sched_meta_for_batch(b=N)` overwrites the shared `self.sched_meta` scratch. The
   Phase-B agent claimed it was "resolved (no per-row reader in the batched lane)",
   but the corruption says otherwise — verify the b=N sched_meta / num_splits /
   split-KV accum (`lse_accum`/`o_accum` `[num_sm_parts+b,…]`) are correct PER ROW,
   not just for row 0.
2. **gather/scatter strides** — `q_batched`/`out_batched` row stride vs the kernel's
   `stride_q/stride_o = global_heads*head_dim`; `slice_out_row` offset per rank.
3. **per-row indices / block_table** — each row's sparse `indices` must reference its
   OWN slot's pool blocks (`slot_block_offsets[r]`); a shared/row-0 index = wrong KV.
The split (B=1 path) is correct; the bug is in the b=N kernel wiring.

### Fix attempt 1 (2026-06-14, commit `33aa8b0f`) — NECESSARY but INSUFFICIENT
Defect #3 (the offset bound) was real + fixed: `build_indices_batched` passed the
per-slot `total_blocks` as the kernel's pool-absolute `block_offset` bound, masking
every index of rows r≥1 to -1. Fixed to `total_blocks * max_batch` (whole-pool).
**But re-verify (decode-read c=2/c=4) still GARBLED.** New signature: the garbled
row now **shifts with batch composition** (c=2: Italy garbled / France ok; c=4:
France+Canada+Egypt garbled / Italy ok) — disproving "row 0 always survives". A
SECOND defect remains in the b=N split-KV / gather path: batch-composition-dependent
+ deterministic = **per-row data mis-attribution** (rows reading each other's
KV/Q/accum). Prime suspects now: `sched_meta_for_batch(b=N)` tile-scheduler metadata
+ the shared `[num_sm_parts+b]` split-KV accum's per-row attribution, or the shared
`tp_gathered_q` staging in `gather_q_row`. Code-read can't pin it (the bound was
also code-read-airtight yet missed this) — needs controlled isolation: compare each
row's batched output to its per-row-kernel reference for the SAME inputs.

### Numerical pinpoint (2026-06-14, `INFER_DSV4_BATCHED_NUMDIFF` infra `1c4414dd`)
Built a reusable numerical-diff harness (the confounded row-vs-row BATCH_PROBE is
useless — concurrent HTTP isn't lockstep). It diffs the RAW batched kernel output
vs a b=1 reference reading the SAME packed KV + SAME gathered `q_batched[r]`
(pre-all-reduce). Result, DETERMINISTIC:
```
[numdiff] FIRST-EXCEEDANCE L0 row1 slot1 maxdiff=5.75 (thresh 0.05, ~115×)
```
**Invariant: row0 (canonical first position) ALWAYS correct; every row≥1 corrupt,
position-dependent NOT slot-dependent** (c=4: row0 only corrupts once a higher slot
slides into position 0 after a batch shrink). So the b=N kernel mis-attributes the
per-row split-KV accumulator — only position 0 reads the right slice. Since the
numdiff shares `q_batched[r]`+KV, a gather/pack bug is RULED OUT (would be diff≈0).
And the per-row q/o/lse/indices strides + the accum split-strides are all correct
(verified: stride_q=h_q*head_dim advances per row; accum is `[num_sm_parts+b]` with
b folded via `num_splits`, not a stride). **⇒ Defect-2 is `num_splits(b=N)` /
`sched_meta(b=N)` / the official FlashMLA b=N split-KV combine attribution — deep
vendored-kernel, FIXABLE (SGLang runs b>1), not the Rust wiring.** Next: study the
official FlashMLA b=N decode convention (arle-upstream-runtime-scan) — compare our
`sched_meta_for_batch` + the shim's `[num_sm_parts+b]` combine to the official b=N
usage; the bug is a deviation there. The numdiff infra will confirm the fix
(FIRST-EXCEEDANCE must vanish + decode-read coherent at c≥2).

## Fix
Stays gated OFF (main default byte-identical, safe). Next: root-cause the b=N
corruption (decode greedy tokens at c=8, compare per-row output to the per-row-kernel
reference for the SAME inputs — the new batched kernel's autoregressive output is the
thing under test). The mla_attention split is kept (sound). Do NOT default-flip.

## Rule
- **needle-keyword presence ≠ correct decode.** The batched kernel got the first ~5
  tokens right (keyword) so `needle_gate.py` PASSED, but the continuation was garbage.
  A batched-kernel correctness gate MUST require **coherent multi-token continuation
  + self-consistency** (decode-and-read the actual generation), not just "needle in
  output". Extends [[feedback_correct_inference_not_baseline_identity]] — needle
  retrieval is necessary, not sufficient, for a new attention kernel.
- **Faster-nonsense: a perf number on corrupted output is void.** Gate correctness
  (coherent continuation) BEFORE reading the perf delta. The +2.9% here was meaningless.
- **A claimed "aliasing resolved" is a hypothesis until the output is coherent.** The
  Phase-B agent asserted the sched_meta aliasing was fixed; the garbled decode
  falsifies it. Don't trust a structural claim over the autoregressive output.
- Decode-greedy-read the generation when verifying a new GPU decode path
  ([[reference_dsv4_b1_tps_msstep_vs_tokstep_diagnostic]] family) — keyword/metric
  gates miss continuation corruption.
