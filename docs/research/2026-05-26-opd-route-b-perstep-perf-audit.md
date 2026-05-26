# OPD Route B per-step performance audit

**Date**: 2026-05-26
**Scope**: static audit only. No V100 GPU time was used.
**Workload reference**: `wD-windowed-train-1step` from
[`docs/experience/wins/2026-05-26-opd-chunked-kl-route-b-bench.md`](../experience/wins/2026-05-26-opd-chunked-kl-route-b-bench.md).

## Evidence Baseline

The clean V100 Route B run now completes one train step, but the step is
not yet productive:

| Phase | Wall-clock |
|---|---:|
| Total train step | 897.4 s |
| Rollout | 111.6 s |
| Teacher forward | 168.2 s |
| Student forward | 78.1 s |
| Backward | 538.2 s |
| Peak GPU | 25,440 MiB |

The forward and backward numbers are coarse `OpdStepProfile` phase
timers. `crates/autograd/src/tape.rs` already has `BackwardProfile`
with per-op totals, merge-grad time, prelude time, and total time, but
`crates/train/src/opd.rs` calls `tape.backward(...)` directly in
`backward_weighted_window_loss`. There is no per-op backward evidence for
the 538.2 s yet. Any optimization below is therefore a hypothesis until
Phase 2 runs the profiled 1-step bench.

## Static Callgraph

Route B train scoring goes through:

1. `opd_step_with_teacher_forward_profiled_gkd_anchor`.
2. `backward_windowed_gkd_loss` when `logits_window_size` is set.
3. For KL windows:
   - teacher `forward_logits_window_device`, tape disabled
   - student `forward_logits_window`, tape enabled
   - `kl_distill_loss_chunked`
   - `backward_weighted_window_loss`
   - `cleanup_after_backward`
4. For corpus-truth SFT windows:
   - student `forward_logits_window`, tape enabled
   - `next_token_sft_loss_from_logits`
   - `backward_weighted_window_loss`
   - `cleanup_after_backward`

`forward_logits_window` is correct for memory but not for compute reuse:
it runs the transformer over `input_ids[..window.end]`, slices hidden
`[1, window.start..window.end, hidden]`, then applies `lm_head` only to
the sliced hidden. It does not use the rollout KV cache, and it does not
reuse prefix hidden/KV across windows.

`OpdKlMask::CompletionOnly` means the KL part only scores completion
positions (`prompt_len - 1 .. rollout_len - 1`). With `rollout_len=8`,
the KL loss itself is not the source of eight 64-token windows. The
multi-window train work comes from corpus-truth SFT when the corpus
completion spans many windows; eval also scores full prompts window by
window.

## Findings

### F1. The largest static suspect is train-side linear attention, not KL reduce/scale

The V100 student is Qwen3.5-0.8B-Base. The local matching config has 24
layers: 18 `linear_attention`, 6 `full_attention`, hidden size 1024, and
linear-attention state dimensions:

| Item | Value |
|---|---:|
| `linear_num_value_heads` | 16 |
| `linear_key_head_dim` | 128 |
| `linear_value_head_dim` | 128 |
| `state_history` per token | 262,144 f32 = 1 MiB |
| `state_history` at seq 512 | 512 MiB per linear-attention layer |

`crates/autograd/src/ops/linear_attention.rs` is entirely host-side:

- forward calls `store.ensure_host(...)` for qkv, z, b, a, conv,
  dt_bias, a_log, and norm, then runs `linear_attention_forward` over
  host `Vec<f32>`;
- backward calls `store.ensure_host(...)` again, reads the upstream grad
  to host, recomputes `linear_attention_forward`, allocates host
  `dqkv/dz/db/da/...`, and walks the scan in Rust loops.

This affects both forward and backward:

- Teacher forward is tape-disabled, but still uses the train-side
  `InProcessTeacher::forward_logits_window_device`, which calls the same
  `Qwen35Model::forward_logits_window` and therefore the host
  linear-attention body.
- Student forward does the same host linear-attention body.
- Backward has to propagate through linear-attention layers that sit
  after the first trainable full-attention LoRA layer. With the
  3-linear + 1-full pattern, that is up to 15 host linear-attention
  backward ops on a long prefix.

This is a stronger hypothesis than "reduce-mean plus scale is too
chatty". KL/SFT loss ops are visible and should be profiled, but they
operate on logits windows. The linear-attention host scan operates on
long prefixes and has a per-layer `state_history` footprint of about
`seq_len MiB`.

### F2. Route B forward avoids full logits but repeats full-prefix compute

The Route B design deliberately runs full context `0..window.end` for
causal correctness, then slices hidden before `lm_head`. This killed
`[B,S,V]` peak memory, but it means a later window re-runs all earlier
tokens through every layer.

The wC eval symptom matches this exactly: early windows are much faster,
and later windows grow as `window.end` grows. For train, KL completion
with `rollout_len=8` is small, but corpus-truth SFT can still walk many
windows over the corpus completion, each with a longer prefix.

The existing rollout KV path (`forward_batch_indices_with_kv_cache`) is
not a drop-in scoring path:

- it is built for rollout/decode;
- it returns only the final token logits when `seq_len > 1`;
- it is currently used with tape disabled for sampling;
- using it for student scoring with gradients would require a
  training-safe cache contract, not a detached inference cache.

Teacher scoring is easier because teacher tape is disabled and teacher
gradients are forbidden.

### F3. Backward allocates fresh same-shape buffers per window

Each window calls `Tape::backward` independently. Inside
`backward_impl`, the tape creates a fresh loss grad, walks the relevant
post-order graph, and every device backward returns fresh output handles.
`cleanup_after_backward` retains params and persistent grads, but it does
not provide a reusable workspace for repeated window shapes.

Examples:

- `gather_last_dim_backward` allocates a zero-filled `[positions, vocab]`
  gradient on device.
- `mean_backward_device`, `mul_scalar_backward_device`, and
  `add_into_device` allocate fresh device buffers.
- `slice_backward_device` allocates a zero-filled full input shape and
  scatters the upstream window grad into it.

The hidden slice scatter is real, but it is probably not the primary
cost: `[1, 512, 1024]` is about 2 MiB. The much larger repeated buffers
are logits-sized `[window, vocab]` and the host linear-attention scan
intermediates.

### F4. The loss graph is small-op heavy, but it is not yet licensed as the bottleneck

`kl_distill_loss_chunked` creates, per KL chunk:

- `slice(teacher_logits)`
- `slice(student_logits)`
- `softmax(teacher_chunk)`
- `log_softmax(student_chunk)`
- `mul`
- `mean`
- `mul_scalar`
- `add` into the accumulated chunk total

SFT does `log_softmax -> gather_last_dim -> mean -> mul_scalar`.
`backward_weighted_window_loss` then adds one more `mul_scalar` for the
global GKD weight and calls `store.to_host(weighted_loss)` to log the
scalar before backward.

This graph is a legitimate optimization target, especially because the
KL and SFT gradients with respect to student logits have closed forms.
But without `BackwardProfile`, we cannot claim it explains the 538.2 s.

### F5. CUDA Graph is not first-line until allocation and host sync are controlled

Autograd backward currently has several graph-capture blockers:

- fresh TensorStore ids and fresh CUDA allocations per backward op;
- scalar `store.to_host(weighted_loss)` before every backward;
- cleanup/retain between windows;
- host-only `linear_attention` forward/backward;
- dynamic window lengths for tail windows.

Inference has CUDA Graph infrastructure for decode/prefill, but the
autograd tape does not yet expose a stable graph-capture boundary.
Backward op order is likely stable for fixed window size, so CUDA Graph
is plausible later, but only after the hot path is device-resident and
workspace addresses are stable.

## Optimization Candidates

| Rank | Candidate | Expected wall-clock impact | Effort | Risk | First proof |
|---:|---|---:|---|---|---|
| 0 | Wire `BackwardProfile` into OPD Route B behind an env flag and log per-window op totals | 0 s direct | S | Low | 1-step wD profile shows op totals, merge time, and host demoters |
| 1 | Move train-side Qwen3.5 linear-attention forward/backward off host, or reuse a CUDA GDR training op | 200-500 s possible | L | Medium-high | Backward profile shows `LinearAttention` dominates; then port one long-prefix layer/path and rebench |
| 2 | Teacher-only scoring KV/recurrent cache for `forward_logits_window_device` | 80-160 s possible | M | Medium | Tape-disabled teacher path produces same logits as full window; teacher_forward drops materially |
| 3 | Analytic logits-gradient loss path for KL + SFT, bypassing loss-op tape | 50-200 s possible | M | Medium | Add `backward_from_grad(logits, dlogits)` or equivalent; small-shape parity with current loss |
| 4 | Reuse per-window gradient workspaces for repeated shapes | 20-100 s possible if alloc dominates | M | Medium | Backward profile plus CUDA API trace shows alloc/add_into/slice/gather buffers hot |
| 5 | Save linear-attention forward intermediates for backward in Route B | 50-250 s possible | M | High memory | Host RSS stays under budget while linear backward recompute time falls |
| 6 | CUDA Graph capture for fixed-size backward windows | 10-80 s possible | L | High | Only after candidates 1/4 remove host sync and stabilize allocations |
| 7 | Student prefix KV/hidden reuse across windows | Large theoretical | L/XL | High correctness | Requires gradient-safe cache, or proves a stop-gradient approximation is acceptable |

## Recommended Phase 2 Order

1. **Profile first.** Add an opt-in `ARLE_OPD_BACKWARD_PROFILE=1` path
   that calls `tape.backward_profiled` inside
   `backward_weighted_window_loss` and prints per-window totals. This is
   required to license any root-cause claim.
2. **If `LinearAttention` dominates, stop chasing KL small ops.** The
   highest-ROI path is a train-side CUDA implementation or a
   memory-bounded saved-intermediate experiment. Use the 0.8B config math
   above to budget host/GPU memory before saving intermediates.
3. **If teacher forward is still a top wall-clock share, add a
   teacher-only recurrent/KV scoring path.** This can be no-grad and
   avoids the hardest student-gradient cache problem.
4. **If loss ops dominate after linear-attention is addressed, implement
   analytic logits gradients for KL/SFT.** That removes softmax/gather/
   mean/mul backward from the tape for the loss head.
5. **Only then consider CUDA Graph.** Capturing the current eager tape
   would mostly capture allocation churn and host boundaries; it is not
   the first experiment.

## SOLID Gaps

- This audit is static. The ranked savings are hypotheses, not evidence.
- The exact wD prompt/completion lengths are taken from the existing
  bench entry and user-provided phase summary, not from a fresh V100 run.
- The Qwen3.5-0.8B config was read locally to count layer types and
  estimate state size. The V100 ModelScope snapshot should be checked in
  Phase 2 before final attribution, though it should match the same model
  family and tensor shapes.
- `BackwardProfile` is the next license-or-kill gate. Without it, the
  correct action is not CUDA Graph or loss fusion yet; it is measuring
  where the 538.2 s actually lands.
