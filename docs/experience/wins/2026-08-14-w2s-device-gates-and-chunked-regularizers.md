# w2s device-side gates and chunked KL regularizers — CUDA, 2026-08-14

> Status: Confirmed

## Goal

Reduce w2s steady-state s/step (baseline 4.105 s on 27B-FP8 student + four 0.8B
aux, GSM8K, one H20). Targets from the
[step budget](2026-08-13-w2s-step-budget-kl-terms-dominate.md): consistency
0.652 s, confidence 0.302 s, local_kl + global_kl 1.396 s.

## Hypothesis

Three treatments in `crates/train/src/w2s.rs`, one commit:

1. `confidence`: slice the last position of `z_s` before the softmax; take the
   max prob via the backend `argmax_last_dim` reduction plus a one-element
   `gather_last_dim`. Removes the full `[seq, vocab]` softmax and its host
   copy. Expected: 0.302 s → near zero.
2. `consistency`: cosine via device reductions — `sum(mul(ΔT₁, ΔT₂))` for the
   dot product, backend `sum_squares` for each norm; three scalars cross to
   host. Removes both `[seq, vocab]` host copies. Expected: 0.652 s → near
   zero. NaN aux logits still yield NaN cosine, so the NaN skip is preserved.
3. `local_kl` / `global_kl`: `kl_distill_loss` → `kl_distill_loss_chunked`
   (`DEFAULT_KL_CHUNK_SIZE = 32`), matching the kd_loss path (0.010 s control
   in the budget entry). Regularizer stages retain their 27B forward
   (~0.57 s); the expected saving is the KL portion beyond it.

## Parameters

Matched A/B, identical flags, same GPU, `--confidence-threshold 0.99` so no
step skips and every stage runs:

- Baseline arm: parent commit of the treatment commit.
- Treatment arm: the treatment commit.
- Run: the 2026-08-13 budget-run config (27B-FP8 base/student, four 0.8B aux,
  GSM8K `--train-data`, 6–8 steps).
- Compare: per-stage seconds from the driver's stage line, and total s/step
  over the steady-state steps. Loss trajectory must be unchanged within noise
  for stages 1–2 (gate math is exact); the chunked regularizers must match the
  unchunked loss within MoE-nondeterminism noise.

## Environment

- Host / GPU: H20 pod, one GPU (serial).
- Model: `/host/nvme0/ThinkingCap-Qwen3.6-27B-FP8` + four 0.8B aux.

## Results

Run 2026-08-14, one H20 (GPU 0), 8 steps per arm, `--confidence-threshold
0.99`, GSM8K `/host/gsm8k-train-wrong.jsonl`. Before arm build `w2s-before2`
at `cb63dbe34`, after arm build `w2s-after-7b9b133` at `7b9b13393`; both
`RUN_EXIT=0`. Steady-state means over steps 1–7:

| Stage | Before s | After s | Delta |
|-------|----------|---------|-------|
| confidence | 0.284 | 0.000 | −100% |
| consistency | 0.583 | 0.006 | −99.0% |
| local_kl | 0.534 | 0.620 | +16% (one 0.988 outlier; 0.558 without it, wash) |
| global_kl | 0.620 | 0.566 | −9% (one 1.175 outlier in before; wash without it) |
| student_fwd | 0.528 | 0.533 | wash |
| backward | 0.774 | 0.776 | wash |
| aux_delta | 0.152 | 0.146 | wash |
| kd_loss | 0.008 | 0.008 | wash |
| cleanup | 0.110 | 0.070 | −36% |
| total | 3.614 | 2.742 | −24.1% |

Correctness:

- Loss per step (before vs after): 25.158342/25.158342, 23.383511/23.387928,
  20.914721/20.906494, 22.125345/22.108746, 19.475698/19.484320,
  22.873857/22.870569, 22.877493/22.870110, 18.022076/18.020435 — max
  divergence 0.017 (0.08%), within MoE-nondeterminism noise.
- `max_prob` matches per step within 4e-4.
- Skip-parity spot run at `--confidence-threshold 0.9` (8 steps, both arms):
  identical trained set {0, 3} and skipped set {1, 2, 4, 5, 6, 7}.

## Problems

- The reported `consistency` gate value shifted (step 0: 0.7372 before, 0.6442
  after). The before path accumulated dot and norms in serial naive f32 over
  the ~20M-element `[seq, vocab]` tensors; the after path uses device tree
  reductions with an f64 host combine, which is the more accurate value. With
  the default `consistency_threshold 0.0` no skip decision changes; a run
  using a nonzero consistency threshold recalibrates against the new values.
- The chunked regularizers did not beat the unchunked path: local_kl +
  global_kl was 1.154 s before and 1.186 s after (each stage is dominated by
  its 27B forward, ~0.53 s). The budget entry's 1.396 s combined figure was
  not reproduced in the before arm either; the reachable KL-beyond-forward
  saving at seq ~140 is within noise.
- One mid-run failure: the shared `/host/arle-build` tree was re-synced and
  rebuilt by a concurrent session between build and run, tripping the run
  helper's binary-SHA check. Re-ran from a dedicated tree
  (`POD_TREE=/host/arle-w2sab`).

## Learnings

The win came from the two host-round-trip gates (0.87 s/step, −24.1% total),
not from chunking the regularizers. Chunked KL only pays where the full-tensor
KL materialization dominates the stage; here each regularizer stage is a 27B
forward plus a small KL, so both KL paths are equivalent. kd_loss reached
0.010 s in the budget entry because it has no forward, not because chunking is
intrinsically ~60× cheaper.
