# T5a Chunked-Logits KL Code Patch

## Context

T5 split into:

- T5a: code-only chunked KL entrypoint and CPU parity tests.
- T5b: real-corpus GKD `prompt_max_tokens=512` acceptance run after P5
  PID 28950 releases the GPU.

This patch implements mitigation 2 from
`docs/research/2026-05-24-bf16-frozen-base-impl-path.md`: reduce the
`[B, S, V]` KL intermediate peak by computing sequence chunks independently
and accumulating an equivalent scalar loss. It does not switch the OPD or eval
callers; the existing `kl_distill_loss` remains the default baseline.

## Audit

Grep scope:

```bash
rg -n "kl_distill_loss\\(|expected_teacher_shape|teacher_logits\\.shape|student\\.forward\\(|prompt\\.len\\(\\)" \
  crates/train/src/loss.rs \
  crates/train/src/opd.rs \
  crates/train/examples/opd_step_cuda_infer_teacher_train.rs
```

Verdict table:

| Surface | Current shape/allocation behavior | T5a verdict |
| --- | --- | --- |
| `crates/train/src/loss.rs:33-52` | Baseline `kl_distill_loss` validates `num_positions == logits.numel() / vocab`, then materializes full-logits `softmax(teacher)`, `log_softmax(student)`, `mul`, and `mean` over the whole `[positions, vocab]` surface. | Keep as parity baseline; add sibling entrypoint. |
| `crates/train/src/opd.rs:1061-1100` | OPD teacher scores the exact rollout and validates `teacher_logits.shape == [1, rollout.len(), vocab]`; student forward returns logits for the same rollout; KL caller passes `rollout.len()`. | Do not switch in T5a; T5b decides callsite. |
| `crates/train/examples/opd_step_cuda_infer_teacher_train.rs:982-991` | Eval step 0 scores each prompt with teacher/student logits for `prompt.len()` positions and calls baseline KL. Long real-corpus prompts therefore allocate near-512-token logits before the first train step. | Do not switch in T5a; T5b owns real-corpus validation. |
| Autograd layout ops | `slice` records backward scatter into the original tensor; `mean` has the tested KL scale path used by baseline. | Implement chunks with `slice` + per-chunk `mean` + weighted scalar accumulation. |

## What Worked

- Added `kl_distill_loss_chunked(student_logits, teacher_logits,
  num_positions, chunk_size, ...)` as a sibling entrypoint; the existing
  `kl_distill_loss` signature and both OPD/eval callsites remain unchanged.
- Reused `slice` so backward scatters each chunk gradient into the original
  student logits tensor through the existing autograd layout op.
- Preserved baseline scale exactly: each chunk computes the same
  mean-over-chunk-positions-and-vocab value, then weights by
  `chunk_positions / num_positions` before the final negative scalar.
- Avoided `sum_backward`; the chunked path stays on the same `mean` backward
  family that the baseline KL and CE loss already exercise.

## Memory Sanity

Pending tests. The intended synthetic calculation is for one f32
`[1, 512, 248320]` KL intermediate:

- Full: `1 * 512 * 248320 * 4 = 508,559,360` bytes, about 485 MiB.
- Chunk size 64: `1 * 64 * 248320 * 4 = 63,569,920` bytes, about 60.6 MiB.
- Theoretical reduction: 8x per KL intermediate.

The unit test locks the exact byte math. This is synthetic, not measured on a
real GPU run. It covers KL intermediates; end-to-end peak drops only after T5b
switches callers so full forward logits do not remain the dominant allocation.

## Verification

```bash
cargo check -p train --no-default-features --lib
cargo test -p train --lib chunked_kl
cargo test -p train --lib
```

- Exit 0 for all commands.
- `cargo test -p train --lib chunked_kl`: 4 passed.
- `cargo test -p train --lib`: 85 passed, 0 failed.
- T5a does not run real-corpus 512-token GKD; T5b owns that GPU acceptance
  after P5 PID 28950 finishes.

## Rule

Keep a baseline loss entrypoint while adding memory-saving loss variants. Loss
chunking must prove forward and student-logit gradient parity before any OPD or
eval caller switches to it.
