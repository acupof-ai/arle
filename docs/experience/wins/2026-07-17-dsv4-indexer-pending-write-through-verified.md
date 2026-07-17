# DSv4 CSA indexer pending in finish write-through — pod-verified (#165)

## Context

#165 (found in the f59dd79af carry audit): the finish write-through
captured/restored only the MAIN compressor's bf16 pending; the CSA indexer's
pending was never captured, so an off-ratio `finish_len` restore left the
indexer's bf16 pending holding the prior occupant's rows. Fix bf336d0a8 adds
`idx_pending_kv/score` sections mirroring the main pending at
`dsa_index_ratio` (GLM SparseIndexed ratio=1 → no-op by construction);
ENTRY_MAGIC DSPP→DSP2 fail-closes stale pool entries. Refactor 7f24937ca
collapses the 4 main/indexer clones into `capture_pending_tail` /
`restore_pending_tail` (behavior + error strings unchanged).

## What Worked

Pod 8×H20 @ bf336d0a8, DSv4-Flash-FP8 TP=4, decode-reuse ON, chunk 2048:

- Binary check: `strings arle` → DSP2=2, DSPP=0.
- Codec: `entry_codec_round_trips_through_pool` (incl. idx_pending) 1/1 ok.
- Restore parity: reuse-hit generations byte-identical to cold at targets
  2000/2003 (needle Y across cold + both hits); 4003 hit1 == cold, hit2
  within the known MoE non-determinism envelope — no reuse-path divergence.
- Needle backstop: 27/27 exact (115–8000 ×3), serve log clean.

Verification protocol note: prefix-EXTENSION prompts never hit this engine's
cache (only exact-repeat/containment), so the off-ratio path was licensed via
exact-repeat restore parity at arbitrary finish positions — the extension-miss
gap and the `prefix_reuse` gate protocol are tracked in #166. Perf: no bench
delta claimed (correctness fix on the restore path; no default flip).

## Rule

Every state-capture image enumerates BOTH compressor instances — the CSA
indexer is a second `Dsv4CompressorState` and every carry the main compressor
persists (overlap, pending) needs its `idx_*` twin, or restore hands the next
occupant the prior request's rows.
