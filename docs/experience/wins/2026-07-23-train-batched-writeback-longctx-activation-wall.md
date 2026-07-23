# Long-completion batched writeback: tile removal real, activation wall now binds — B=4×3K OOMs, B=1 works at 59.3 GB

> Status: Shipped (measurement; closes the long-config follow-up in
> [2026-07-23-train-batched-writeback-fused-ce-vram.md](2026-07-23-train-batched-writeback-fused-ce-vram.md)).
> Pod: 8×H20, GPU 0 only, snapshot binary sha-verified == local `9a307f2e8`
> (+ `0a42841ad` args fix). Qwen3.5-0.8B rubric-opd, completions truncated at
> the 3072-token cap (max_len≈3150, genuinely long).

## Context

The short-completion run proved the fused-CE mechanism (phase-C +0.13 GB) but
not the magnitude. This run measures the long-completion regime the dense-tile
removal targets — and refutes the "phase-C ≈ 0 at any length" expectation.

## What Worked (and what was refuted)

| Config | Plateau | Phase-C | Peak | Verdict |
|---|---|---|---|---|
| B=4, max_len≈3150 | 20.9 GB | stepped ramp to 97.5 GB | 97.5 GB | **OOM** (`cuda alloc_zeros`, micro-batch 1/4, RUN_EXIT=1) |
| B=1 control, same 16 rollouts | 20.9 GB | one step to 59.3 GB | **59.3 GB** | RUN_EXIT=0, mean_loss 0.0419 (sane; not comparable to short-config 0.13 — different data) |

**Attribution (case-as-fact, not hand-waved):** the OOM is NOT the removed
logits tile (that would be one ~7.6 GB alloc). The observed ramp is per-layer
steps of ~3.4–4.1 GB over ~2 min — the batched forward's **per-layer activation
footprint**, B-scaled. Only grad-checkpoints offload to host; per-layer
internals stay resident. The offload gate DID engage at B=4
(`b*max_len ≈ 12.6 K > 4096`) and did not save it; at B=1 (3150 < 4096) it
stayed off, as expected.

**The fix still strictly helps:** the old path adds the `[4,3150,151936]` f32
tile (~7.6 GB, ~15 GB with grad) on top of the SAME activations → earlier OOM;
even B=1 would carry ~2–4 GB extra.

**Method wins:** ① snapshot-binary strategy (build once → copy `arle` to a
private dir → run the snapshot) fully defeated the shared-tree rebuild
contention that killed both prior attempts. ② The build surfaced a cuda-lane
compile bug in the simplify pass (`writeback_window` missing on
`TrainRubricOpdArgs`, fixed in `0a42841ad`) — the Mac no-cuda check never
compiles `cli`'s `#[cfg(feature = "cuda")]` lanes, and `cargo check -p cli
--features cuda,no-cuda` can't run on Mac (cudarc build.rs needs nvcc), so the
pod build is that lane's first compiler.

**Harness caveat:** self-consistency accepted all 16 truncated rollouts
(majority-vote-on-`\boxed` under truncation, incl. a degenerate enumeration) —
fine for a VRAM probe, not for capability claims.

## Rule

- Long-completion rubric writeback: `--writeback-batch 1` (59.3 GB, works).
  B=4 is short-completion-only on 96 GB H20 — the binding wall is now batched
  per-layer activations, not the logits tile; next lever is activation-side
  (offload/recompute per-layer internals), not CE-side.
- A "removed buffer" win claim must name the NEXT binding wall at the target
  config — removing tile X proves little if activation wall Y OOMs first.
- Edits inside `cli`/`train` cuda-gated lanes get no Mac typecheck: grep-verify
  every field/symbol fed into the lane, and budget one pod-build fix loop.
