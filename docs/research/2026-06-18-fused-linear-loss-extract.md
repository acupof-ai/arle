# Fused linear distillation loss extract

Date: 2026-06-18

Scope: P2 research extraction only. No code edits, no H20/pod/bench, no tmux2.
Source read in place from `/tmp/liger`. This adopts the algorithmic structure,
not Triton kernels.

## Short verdict

Use Liger's chunked final-linear loss pattern, not a Triton drop-in:

1. Iterate token chunks.
2. For each chunk only, compute `hidden_chunk @ lm_head.T`.
3. Compute loss and `dlogits` immediately.
4. Accumulate `dhidden = dlogits @ lm_head` and `d_lm_head += dlogits.T @ hidden_chunk`.
5. Save only gradients for ARLE backward, not full `[T, vocab]` logits.

Important caveat: Liger's generic chunked distillation base matches this. The
specialized `ops/fused_linear_jsd.py` has the same final-linear gradient idea,
but still allocates `loss_1d = torch.zeros((BT, V))`; do not copy that allocation.

## Upstream algorithm, with file:line

| Stage | Upstream step | Peak memory avoided / kept | ARLE adoption note |
|---|---|---|---|
| Inputs and target shapes | Liger defines distillation inputs as flattened token rows: student/teacher input `(batch_size * seq_len, hidden)`, weights `(vocab_size, hidden)`, target `(batch_size * seq_len)` in `/tmp/liger/src/liger_kernel/chunked_loss/fused_linear_distillation.py:176-183`. | Avoids requiring a caller-owned `[T, vocab]` logits tensor. | ARLE window hidden is already a token range; flatten `[1, window, hidden]` to `[T, H]` inside the new op. |
| Chunk-local final linear | `chunk_forward` computes `student_logits_chunk = student_input_chunk @ student_weight.t()` and optional bias, then `student_log_probs_chunk = F.log_softmax(...)`; teacher logits are computed under `torch.no_grad()` as `teacher_input_chunk @ teacher_weight.t()` in `/tmp/liger/src/liger_kernel/chunked_loss/fused_linear_distillation.py:42-53`. | Keeps only per-chunk `chunk_size x vocab` student and teacher logits. | Adopt the loop and math; in ARLE, teacher logits may come from dense teacher window first, then later from sparse top-k targets. |
| Optional hard CE | Hard loss uses `F.nll_loss(..., reduction="sum", ignore_index=...)` in `/tmp/liger/src/liger_kernel/chunked_loss/fused_linear_distillation.py:54-64`. | No full logits beyond the current chunk. | OPD pure KL can set CE weight/compute off; mixed SFT+distill can reuse this shape. |
| Temperature and vocab padding | `_compute_loss` divides student and teacher logits by temperature and pads student logits if teacher vocab is larger in `/tmp/liger/src/liger_kernel/chunked_loss/fused_linear_distillation.py:119-136`. | Still chunk-local. | ARLE should keep vocab equality as the first version; padding is a later compatibility branch. |
| Valid-token normalization | It computes `num_valid_tokens = (full_target != ignore_index).sum().clamp_min(1)`, divides hard and soft loss by that count in `/tmp/liger/src/liger_kernel/chunked_loss/fused_linear_distillation.py:137-147`. | Normalization state is scalar. | ARLE current KL uses batchmean scale; keep one explicit normalization contract in the new op. |
| Chunk loss + gradient capture | `accumulate_chunk` calls `torch.func.grad_and_value(..., argnums=(0, 1[, 5]))`, then adds `chunk_grad_weight`, `chunk_loss`, optional soft/hard losses, and returns `chunk_grad_input` in `/tmp/liger/src/liger_kernel/chunked_loss/fused_linear_distillation.py:217-266`. | Stores accumulated `grad_weight` and per-token `grad_input`, not full logits. | In ARLE autograd, compute `dhidden` and `d_lm_head` inside forward-like loss evaluation and save those tensors for backward. |
| Chunk iteration | Liger chooses `num_chunks = max(1, student_input.shape[0] // CHUNK_SIZE)`, chunks student input, teacher input, and targets, then loops and appends returned grad inputs in `/tmp/liger/src/liger_kernel/chunked_loss/fused_linear_distillation.py:268-280`. | Peak logits are `ceil(T / chunks) x vocab`, not `T x vocab`. | ARLE should use an explicit max rows per chunk, not `torch.chunk` semantics. |
| Backward contract | Liger saves `cat(grad_inputs)`, `grad_weight`, `grad_bias`; backward only multiplies by outer `grad_output` and returns those grads in `/tmp/liger/src/liger_kernel/chunked_loss/fused_linear_distillation.py:282-299`. | No recompute or saved logits in backward. | ARLE op can be a custom autograd op whose backward returns precomputed `dhidden` and `d_lm_head`. |

## JSD / KL formulas to adopt

| Loss piece | Upstream file:line | Extracted behavior |
|---|---|---|
| JSD definition and `dX` | `/tmp/liger/src/liger_kernel/ops/jsd.py:29-32`, `:67-85` | For beta in `(0,1)`, compute `Q=exp(X)`, `P=exp(Y)`, `M=beta*P + (1-beta)*Q`, loss `beta*P*Y + (1-beta)*Q*X - M*log(M)`, and `dX=(1-beta)*Q*(X-log_M)`. |
| Forward KL beta=0 | `/tmp/liger/src/liger_kernel/ops/jsd.py:54-60` | Loss `P * (Y - X)`, `dX = -P`. |
| Reverse KL beta=1 | `/tmp/liger/src/liger_kernel/ops/jsd.py:60-65` | Loss `Q * (X - Y)`, `dX = loss + Q`. |
| Mask and scale | `/tmp/liger/src/liger_kernel/ops/jsd.py:40-46`, `:87-93` | Ignore-label rows get zero gradients; otherwise loss and gradient scale by `1 / n_non_ignore`. |
| Standalone KL | `/tmp/liger/src/liger_kernel/ops/kl_div.py:74-79`, `:114-119`, `:246-251` | KL uses `target * (log(target)-input)` or `exp(target)*(target-input)` and scales derivative by reduction mode. |

## Fused CE / fused JSD implementation details

- CE fused linear computes dynamic chunk size from `BT,H,V` and comments that
  materialized activations are `BT x V`; chunking reduces that to chunk-local
  `chunk x V`: `/tmp/liger/src/liger_kernel/ops/fused_linear_cross_entropy.py:45-58`.
- CE computes `logits_chunk = input_chunk @ weight.t()`, calls the CE kernel
  in-place so `logits_chunk` becomes `grad_logits_chunk`, then computes
  `grad_input = grad_logits_chunk @ weight` and
  `grad_weight += grad_logits_chunk.t() @ input_chunk`:
  `/tmp/liger/src/liger_kernel/ops/fused_linear_cross_entropy.py:96-103`,
  `:146-153`, `:200-212`.
- CE saves only `grad_input`, `grad_weight`, `grad_bias`, and backward returns
  those after optional outer-grad scaling:
  `/tmp/liger/src/liger_kernel/ops/fused_linear_cross_entropy.py:341-383`.
- JSD fused linear has the same final-linear gradient flow:
  `student_logits_chunk = student_input_chunk @ student_weight.t()`,
  `teacher_logits_chunk = teacher_input_chunk @ teacher_weight.t()`, in-place
  JSD gradient in `student_prob_chunk`, then
  `grad_input[start:end] = student_logits_chunk @ student_weight` and
  `grad_weight.add_(student_logits_chunk.t() @ student_input_chunk)`:
  `/tmp/liger/src/liger_kernel/ops/fused_linear_jsd.py:62-79`,
  `:85-119`.
- Do not copy JSD's full loss buffer: it allocates
  `loss_1d = torch.zeros((BT, V))` at
  `/tmp/liger/src/liger_kernel/ops/fused_linear_jsd.py:48-52`, and the JSD
  kernel stores per-vocab loss values in `/tmp/liger/src/liger_kernel/ops/jsd.py:92-93`.

## Current ARLE dense path

| Site | Current materialization | Replacement target |
|---|---|---|
| `crates/train/src/opd.rs:2000-2018` | Teacher cached pure-KL path requires full `[1, rollout.len(), vocab]` teacher logits. | Later sparse top-k teacher targets can remove this; fused student loss alone does not. |
| `crates/train/src/opd.rs:2095-2113` | Each window slices dense teacher logits into `[1, window.len(), vocab]`. | First fused op can consume this dense teacher window; second pass should consume sparse teacher targets. |
| `crates/train/src/opd.rs:2048-2067` | Student hidden is computed once and cached. | Keep this; fused loss should consume `student_hidden` directly. |
| `crates/train/src/opd.rs:2124-2155` | `student.logits_from_hidden_window` materializes `[1, window.len(), vocab]`, then `kl_distill_loss_for_config` consumes it. | Replace both calls with `fused_lm_head_distill_loss(student_hidden, lm_head, teacher_target, window, config)`. |
| `crates/train/src/opd.rs:1446-1476` | Non-cached path avoids full scored-prefix student logits, but still materializes per-KL-window dense student logits and calls chunked KL. | Same replacement, scoped to `kl_window`. |
| `crates/train/src/loss.rs:96-103` | Current comment says chunked KL chunks the loss graph only; callers must stop materializing full forward logits separately. | This is exactly the gap the fused op closes. |
| `crates/train/src/loss.rs:155-194` | Current KL slices logits chunks, then creates dense `softmax` / `log_softmax` / elementwise tensors over `[chunk, vocab]`. | Fused op keeps `[chunk, vocab]` as a transient local buffer and saves only gradients/loss scalar. |

## ARLE op map

Proposal only:

1. Add one autograd op, not a new training framework:
   `fused_lm_head_distill_loss(hidden, lm_head_weight, teacher_logits_or_target, window, config)`.
2. Forward-like execution loops token chunks inside the window. For each chunk:
   compute student logits, compute loss and `dlogits` using the cited JSD/KL
   equations, accumulate scalar loss, `dhidden`, and `d_lm_head`.
3. Backward returns saved `dhidden` to `student_hidden` and `d_lm_head` to the
   lm head weight, matching Liger's saved-gradient contract.
4. CPU backend can implement this with host `Vec<f32>` first as the exactness
   oracle. CUDA/Metal can then replace the per-chunk inner loop without changing
   the autograd surface.
5. Plug point: replace `logits_from_hidden_window` plus
   `kl_distill_loss_for_config` in `crates/train/src/opd.rs:2124-2155`; then
   replace the non-cached `forward_logits_window` plus `kl_distill_loss_chunked`
   path in `crates/train/src/opd.rs:1446-1476`.

## Sparse teacher top-k note

Sparse teacher targets compose with the fused loss but are a separate lever.
Current local extraction only covered Liger. The Track 1 document records the
intended top-k teacher target direction at
`docs/research/2026-06-18-opd-memory-infra-gap.md:26-29`,
`:37-38`, `:68-72`; this pass does not spec that algorithm without a local
upstream `file:line`.

## Open questions for Claude

1. Should the first op support dense teacher logits only, then add sparse top-k
   teacher targets as v2? Lazy answer: yes, because it replaces the student
   `[window, vocab]` allocation immediately.
2. Which normalization is canonical for ARLE OPD fused JSD/KL: current
   `batchmean` in `crates/train/src/loss.rs:40-51`, or Liger's
   `1 / n_non_ignore` scaling in `/tmp/liger/src/liger_kernel/ops/jsd.py:87-90`?
3. Should CPU-first exactness be mandatory before CUDA? Lazy answer: yes; ARLE's
   autograd is host-authoritative and this is a correctness-sensitive loss.
4. When sparse top-k teacher targets land, should dense teacher logits stay as a
   debug fallback? Lazy answer: yes, until quality/equivalence gates prove top-k.

Stop condition: extraction complete; await implementation approval.
