# OPD `kl_chunk_size` sweep — loss DRIFTS with chunk size → correctness gate FAIL, default unchanged

## Context

Goal: kill the O(n²) prefix recompute in `backward_chunked_kl_rollout`
(`crates/train/src/opd.rs:833`). Attribution (prior tick) showed backward
≈ 63 s ≈ 79 % of an OPD step at `kl_chunk_size=16`, dominated by re-forwarding
+ backpropping the entire prefix from token 0 for every chunk. The hypothesis:
`kl_chunk_size = n_completion` collapses to a single full-sequence
forward+backward with zero redundant recompute, so a larger default would be a
free win.

Sweep run (sm89, RTX, 16 GiB; `--release`; build env
`CUDA_HOME=/opt/cuda NVCC_CCBIN=g++-14
INFER_TILELANG_PYTHON=…/.venv/bin/python TORCH_CUDA_ARCH_LIST=8.9
ARLE_CUDA_DISABLE_FLASHMLA=1`):

```
opd_step_cuda_infer_teacher_train --steps 2 --rollout-len 128 \
  --lora-rank 8 --lora-alpha 16 --lora-target-set attention-qv \
  --kl-chunk-size {16,32,64,128}
```

Qwen3.5-0.8B self-distill (teacher = student = `Qwen3___5-0___8B-Base`),
infer rollout ON, `mem_fraction_static = 0.05`, one GPU job at a time.
teacher_seq_len=128, vocab=248320, rollout_len=129, prompt_index=0 (identical
deterministic prompt + greedy rollout across all points).

## Sweep table (step-2 warm numbers; loss is the step-2 reported KL)

| chunk | step-1 loss   | step-2 loss   | backward s | step s  | peak VRAM |
|-------|---------------|---------------|------------|---------|-----------|
| 16    | 9.6031e-5     | 9.6565e-5     | 61.77      | 77.34   | 8618 MiB  |
| 32    | 1.02302e-4    | 1.02553e-4    | 34.37      | 43.88   | 8714 MiB  |
| 64    | 1.09738e-4    | 1.04287e-4    | 21.36      | 32.63   | 8970 MiB  |
| 128   | 7.6636e-5     | 7.6620e-5     | 13.82      | 22.12   | 9450 MiB  |

Performance side (had the gate passed): backward 61.77 s → 13.82 s = **4.5×**;
step 77.34 s → 22.12 s = **3.5×**; VRAM rises only 8618 → 9450 MiB (+10 %),
single-pass fits with ~7 GiB headroom. The perf hypothesis is fully confirmed.

## Root Cause — correctness gate FAILED (this is the blocker)

The chunked KL is supposed to be a pure memory split of the *same*
mathematical KL sum, so the loss VALUE must be invariant across chunk sizes
(within ~1e-3 relative / float noise). It is **not**:

- step-1 loss (same initial params, deterministic rollout) spans
  **7.66e-5 … 1.10e-4 — a ~43 % spread**, far above the 1e-3 gate.
- The variation is systematic, not noise: within a single chunk size the
  step-1 loss reproduces to ~5 significant figures across reruns
  (chunk=16: 9.6031e-5 vs 9.6036e-5 on two runs).
- It is non-monotonic (16 < 32 < 64, then 128 drops below all of them).

The arithmetic of the per-chunk weighting is provably invariant: each chunk
computes `kl_distill_loss(student_chunk, teacher_chunk, chunk_len)` =
`sum_chunk / (chunk_len · vocab)`, then multiplies by
`chunk_weight = chunk_len / kl_range.len()`, giving
`sum_chunk / (kl_range.len() · vocab)`; summed over disjoint chunks this is
`total_sum / (kl_range.len() · vocab)` — `chunk_len` cancels. So the weighting
is not the bug.

The actual difference between chunk sizes is the **forward**:
`backward_chunked_kl_rollout` does a *separate*
`student.forward(prefix[..seq_end])` and
`teacher.forward_logits_device(prefix[..seq_end])` per chunk
(`opd.rs:859, 877`), then slices positions `[seq_start..seq_end)` out of that
forward. chunk=128 is one forward over the full sequence (the reference);
smaller chunks re-forward growing sub-prefixes. The measured drift proves the
re-forwarded sub-prefix logits at a given absolute position are **not** equal
to the full-sequence logits at that position. Under pure causal attention they
should be; the train-crate Qwen3.5 forward has both Full and **Linear**
attention layers (`qwen35.rs:362`, LinearAttention was 29 % of backward in the
attribution), so the prefix-length sensitivity is *hypothesis*: a
linear-attention recurrence/normalization that is not prefix-length-invariant
in this from-scratch autograd path. Exact mechanism is **not yet evidenced**
(would need a per-position logit diff between a full forward and a sub-prefix
forward) and is left for a dedicated investigation.

The single-pass path (chunk=128) is the only one that forwards the exact
sequence once and is therefore the trustworthy value (7.66e-5). The chunked
loop is producing a *different, drifting* objective — a faster-but-wrong
default would silently change what OPD optimizes.

## Fix

**None landed.** Per the sweep brief's correctness gate: loss drift with
chunk_size is a pre-existing chunking correctness bug → STOP, do not change
`DEFAULT_KL_CHUNK_SIZE` (still 32). A faster-but-wrong default is worse than a
slow correct one.

Follow-up to license a single-pass default later:
1. Add a parity test: full forward over `seq` vs sub-prefix forward over
   `seq[..k]`, assert per-position logits match at shared positions (CPU,
   train-crate `Qwen35Model`, both Full and Linear attention). This isolates
   whether the drift is in Linear attention.
2. If Linear attention is prefix-length-sensitive by design, the chunked
   rollout KL must reuse ONE full-sequence forward and slice it (like the
   already-correct `loss::kl_distill_loss_chunked`), instead of re-forwarding
   sub-prefixes — that removes both the O(n²) recompute AND the drift.

## Rule

A chunked loss that re-forwards a different-length prefix per chunk is NOT a
pure memory split of one objective when the model forward is not
prefix-length-invariant (linear/recurrent attention). Before flipping a chunk
default for speed, gate on **step-1 loss invariance across chunk sizes** (same
params, deterministic rollout) — step-2 drift can be confounded by the
optimizer, but step-1 drift is the objective itself. Self-distill near-zero KL
(~1e-4) has tiny absolute values but the within-chunk reproducibility (5 sig
figs) vs cross-chunk spread (43 %) cleanly separates float noise from a real
bug.
