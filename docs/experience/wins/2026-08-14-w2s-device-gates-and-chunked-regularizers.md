# w2s device-side gates and chunked KL regularizers — CUDA, 2026-08-14

> Status: pending-remote

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

Pending-remote: CUDA unavailable locally; the Run phase executes the A/B on
the pod. Both lanes typecheck clean
(`cargo check -p train --release --no-default-features --features cuda,no-cuda`
and `--features metal,no-cuda`).

## Problems

None yet.

## Learnings

pending-remote.
