# Sparse top-k teacher targets — the real remaining dense-logits cost (OPD)

**Date**: 2026-06-18  Source: adversarial-review workflow `wf_6a493801` research agent
(read-only, file:line verified). Complements P2 (fused-linear-distill, student-side).

## Why this, not "more offload"

P2 already removed the dense **student** `[win,vocab]` logits (fused-linear-distill,
`fused_linear_distill_loss` at opd.rs:2127). The dense cost that **survives** is the
**teacher target tensor** `[1, seq, vocab]` f32, born at `teacher_infer.rs:790`
(`InferTeacher.forward_logits_device`: bf16 `[seq,vocab]` → D2D bridge
`import_bf16_device_ptr_as_f32` → `store.alloc_device_tensor [1,seq,vocab]`), held
across the whole window loop in `backward_windowed_pure_kl_cached_student_hidden`
(opd.rs:1951, asserted `[1,seq,vocab]` at :2010-2018, freed :2217). At R1
(vocab≈152k, seq 1536) that is **~932 MiB f32** — the `[seq,524288]` H2D OOM class.

## Design (3 pieces — verl async-OPD sparse-KL form)

**A. Top-k at the teacher boundary** — store `[seq,k]` not `[seq,vocab]`.
Add `forward_topk_targets_device(input_ids, positions, k, store) -> {topk_logprobs
[1,seq,k], topk_indices Vec<i32> seq*k}` to the `TeacherForward` trait
(`teacher_infer.rs:55-99`, defaulted `Unsupported` so in-process/API teachers stay
valid). InferTeacher impl (`:717`): take top-k + **full-vocab log-softmax on the bf16
`[seq,vocab]` BEFORE** the D2D f32 bridge (`:775`); bridge only the `[seq,k]` f32
logprobs (k/vocab ≈ 0.04% of current D2D traffic). Store **true global log-probs**
(log-softmax over full vocab) — not re-softmaxed top-k — to avoid the normalizer error
(see caveat). Needs an engine-side topk-logprob reduction over raw bf16 logits
(`infer-api/src/types.rs` RawLogits + `loaded.rs:366` forward_token_logits) — **this
piece is CUDA, builds on H20**.

**B. Sparse fused loss op** — `fused_linear_distill_loss_sparse` in
`autograd/src/ops/fused_linear_distill.rs`, sibling of P2's op, same
`SavedContext::FusedLinearDistillCtx` + backward. Student side **unchanged** (per-row
lm_head matmul `:64-69`, full `student_log_probs` `:75` — already dense-free). Only the
teacher term changes: Forward-KL `row_ce = -Σ_{i∈topk} exp(t_logprob_i)·student_log_probs[idx_i]`;
`dlogits[j] = grad_scale·(student_prob[j] − t_prob_on_j)` where `t_prob_on_j` is the
stored top-k prob if j∈topk else 0 (student term stays dense & free; teacher term sparse-scatter).
**This piece is autograd CPU — locally implementable + testable with synthetic top-k.**

**C. Rewire windowed path** — `backward_windowed_pure_kl_cached_student_hidden`
(opd.rs:1951): swap `forward_logits_device`→`forward_topk_targets_device` (:2000),
shape assert →`[1,seq,k]` (:2010), per-window slice (:2095) → `[seq,k]` slice + index
sub-slice, fused call (:2127) → sparse op. Gate behind `GkdLossConfig.teacher_topk:
Option<usize>` (opd.rs:216, default `None` ⇒ current dense path byte-identical) wired
from `--teacher-topk` + a `TEACHER_TOPK` env in the example script.

## Composes with P2
Strictly additive, orthogonal axis. P2 = no dense **student** logits; this = no dense
**teacher** logits. Together the hot loss path holds neither dense `[win,vocab]` —
only `[win,k]` teacher logprobs + P2's per-row `[vocab]` student scratch. (= gap-doc
adopt-list items 2 & 3.)

## Memory effect (HYPOTHESIS — needs same-binary A/B)
Teacher target `[1,seq,vocab]` f32 → `[1,seq,k]` f32 + `[seq,k]` i32. R1 k=64:
~932 MiB → ~0.4 MiB device; D2D bridge ~2400× less. NOT benched (read-only).

## Caveat (load-bearing — license-or-kill, not source-survey)
Top-k truncation drops tail mass `m = 1 − Σ_{topk} t_p`. Two distinct errors: (1)
**normalizer** — avoided by storing full-vocab log-softmax (not re-softmaxed top-k);
(2) **missing-tail** — the dropped `−Σ_tail t_p·log s_p` means the student isn't
penalized for leaking mass onto teacher-tail tokens; **position-dependent, concentrates
where the teacher is uncertain** (early-rollout/ties), which may be where distillation
matters most. Peaky reasoning teacher (R1, T=1.0, fwd-KL): top-64 usually >0.99 mass
(m<0.01), but not uniform. **Reverse-KL is worse**: needs teacher log-probs at the
student's high-prob tokens, which may not be in the teacher's top-k → floor to k-th
value or gate sparse-reverse off until separately validated. Forward-KL is the default
(opd.rs:234) → primary target. Ship only after the correct-inference gate (needle ×3
same-config vs baseline envelope, NOT byte-identity) + a k-sweep (16/32/64/128)
measuring captured-mass and downstream eval.
