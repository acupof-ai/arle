# Batched rubric writeback: fused indexed CE kills the dense logits tile — parity GREEN, phase-C +0.13 GB

> Status: Shipped (10389a340 code, this entry closes its `pending-remote` VRAM
> measurement). Pod: 8×H20, GPU 0 only, sm_90, pod binary sha-identical to HEAD
> for the two changed files (`opd.rs`, `qwen35.rs`; remaining tree delta
> functionally no-op for round-0 rubric-opd).

## Context

`rubric_writeback_ce_step_batched` materialized `[B, max_len, vocab]` f32 logits
via `forward_batch_tokens` then sliced per row. 10389a340 routes it through
`forward_batch_hidden` + per-row `fused_linear_ce_loss_indexed` (chunked, only
masked completion positions projected through `lm_head`) and enables
grad-checkpoint host offload past `writeback_offload_for_seq(b*max_len)`.
Loss reduction (mean-of-row-means) preserved exactly; CPU equivalence gate
`rubric_batched_writeback_matches_per_row_masked_writeback` pins it (≤1e-5,
unequal completion lengths). This entry is the pod measurement.

## What Worked

**Correctness parity — PASS (RUN_EXIT=0).** rubric-opd round 0, prompts=4:
`accepted=13 distinct=4 parse_err=0 trained=13 corrected=0 mean_loss=0.1323`
vs pre-change `mean_loss=0.1318` — Δ+0.4%, every count identical; within the
sampling/MoE non-determinism floor.

**VRAM (GPU 0, clean 0 MiB baseline, 0.4 s sampler).** Peak 64,013 MiB
(64.0 GB) vs the ~84 GB pre-change anchor (~−20 GB). Timeline: ~63.9 GB
plateau through rollout phases A/B (engine + student resident); phase-C
writeback adds only **+128 MiB** — the dense tile is gone. Offload correctly
stayed OFF (`b*max_len ≈ 600 < 4096`).

**Attribution, stated honestly.** On this short-completion config the removed
tile is small (B=4 × short max_len × 151936 × 4 B < 1.5 GB), so the full 20 GB
gap vs the anchor is not cleanly attributable to this change alone (no
same-box pre-change A/B — see below). The SOLID measured claims are: parity
holds, and phase-C writeback now costs ~0 VRAM over the resident baseline.
The win scales linearly with completion length: at B=4, max_len=3072 the old
path materializes 4·3072·151936·4 ≈ **7.5 GB** logits (~15 GB with gradient);
the new path holds hidden `[B·max_len, 1024]` ≈ 50 MB + chunk temporaries.

**Long-completion demonstration — blocked, not failed.** Two attempts
(`rubricwblong`, `rubricwblong2`) died at pre-exec integrity guards ("binary
SHA mismatch" / "source changed since build") — a concurrent `busytimer` lane
was continuously rebuilding the shared pod tree. GPU 0 was free both times;
no code fault. Re-run the long config when the shared tree is quiet to put a
measured number on the tile removal.

## Rule

A VRAM win whose removed buffer scales with a config dimension must be
measured at a config where that dimension is large — a short-completion run
proves the mechanism (phase Δ ≈ 0) but not the magnitude. When the shared pod
tree is contended, per-file sha-verified push of only the changed files beats
syncing a dirty tree: one variable, no confounders.
