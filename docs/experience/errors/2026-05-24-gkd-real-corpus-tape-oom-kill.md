# GKD Real-Corpus 2k Killed by Autograd Tape OOM Before Step 0

Related: `docs/projects/2026-05-24-opd-mainline-task-backlog.md`,
`docs/research/2026-05-24-bf16-frozen-base-impl-path.md`, and
`docs/projects/2026-05-22-eod-opd-cycle-wrap.md`.

## Context

P4 KILL (`2026-05-22-p4-gkd-corpus-anchor-kill.md`) licensed two next branches:
1. longer-horizon pure-OPD 5k run, or
2. real SFT anchor corpus matching eval distribution before retesting GKD λ mixing.

Branch 2 was attempted today on 2026-05-24:

- Codex built `examples/opd/sft-anchor-mmlu-gsm8k.jsonl`: 56 rows
  (26 MMLU diverse-subject + 30 GSM8K, train splits only, leak-checked,
  tokenizer-validated, max_tokens = actual tokenized prompt length capped 512,
  4 MMLU >512 rows dropped).
- MMLU max_tokens distribution: median 381, p95 501, max 503.
- GSM8K max_tokens: median 44, p95 56.

Pre-flight: a 50-step pure-OPD smoke on the synthetic 16-tok corpus PASSED clean
(mean 5.25s/step, GPU peak 11.5/16 GB). Pipeline + GPU budget known good for the
small-prompt shape.

## Run

```bash
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
NVCC_CCBIN=g++-14 \
cargo run -p train --example opd_step_cuda_infer_teacher_train --release --features cuda -- \
  --teacher-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B \
  --student-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base \
  --prompts-file examples/opd/sft-anchor-mmlu-gsm8k.jsonl \
  --steps 20 --rollout-len 8 --lr 2e-5 \
  --eval-steps 0 --prompt-max-tokens 512 --max-step-seconds 120 \
  --gkd-lambda 0.3 --sft-anchor corpus-truth
```

This was the 20-step feasibility smoke (NOT yet the full 2k). Hardware: single
RTX 4070 Ti SUPER 16 GB.

Artifact: `bench-output/2026-05-24-smoke-gkd-real-corpus/run.txt`.

## Evidence

The run loaded teacher + student (`student_load_seconds=8.20`, `teacher_load_seconds=2.08`),
emitted all `prompt split=train/heldout index=...` lines for the full 56-row corpus,
then failed on the autograd tape allocation BEFORE the first train step:

```
model_summary teacher_source=infer student_hidden=1024 student_layers=24 ...
Error: TapeInvariant("cuda alloc_zeros failed")
```

No `train_step step=0` line ever emitted. Failure was deterministic — preallocation
of the autograd tape sized for `prompt_max_tokens=512` + `rollout_len=8` worst-case
(seq_len ≈ 520) overflowed remaining VRAM after model + optimizer state were loaded.

For comparison, the prior smoke at `prompt_max_tokens=16` peaked at 11.5/16 GB —
leaving ~5 GB headroom for the tape. Scaling tape ~22× by seq_len (and non-linearly
for any O(N²) attention terms) easily breaks 16 GB.

GPU state after crash: 1156 MB used (only Edge browser) → process fully released.

## Root Cause

**Initial hypothesis (`prompt_max_tokens × rollout_len` tape preallocation) was WRONG.**
Codex deep-audit (`docs/research/2026-05-24-bf16-frozen-base-impl-path.md`)
ground-truthed the allocator and disproved it. Corrected root cause below.

The real OOM driver is **`[S, V]` logits/loss tape memory**, where
`S = prompt_max_tokens + rollout_len` (additive per `crates/train/src/opd.rs:462-469`)
and `V = vocab_size = 248320` (Qwen3.5 vocab).

Per `crates/autograd/src/backend_cuda.rs:1140-1148`, the largest documented train
allocation shape is logits/loss over sequence × vocab:
`[B, S, V] = 2 × 512 × 248070 × 4 B ≈ 1 GB` for one logits tensor.
Multiple such tensors stack at each step: teacher logits, student logits, KL
loss intermediate, gradient. ~4 × 1 GB ≈ 4 GB of f32 logits memory at
`prompt_max_tokens=512`.

Eval at step 0 already triggers this — `maybe_eval(0, ...)` runs before the
first train step and computes per-prompt logits at `[1, prompt.len(), vocab]`
(`crates/train/examples/opd_step_cuda_infer_teacher_train.rs:570-579, 967-995`).
With MMLU rows at max_tokens=503, eval allocates ~500 MB per prompt — and the
allocator already has model + optimizer at ~10 GB resident → 16 GB OOMs before
step 0 ever runs.

**Model-base residency is NOT the issue.** BF16 frozen-base path IS already
shipped for rank-2 tensors (`crates/train/src/qwen35_loader.rs:822-881`).
Verified: local Qwen3.5-0.8B-Base safetensors are BF16 — `embed_tokens.weight`
and per-layer norms are BF16 on disk, so the BF16 frozen-base storage path
activates. Linear-attn `A_log` is f32 but only 16 elements/layer (negligible).

The corpus file itself is fine; the harness is fine; the autograd tape
allocator is sized correctly for its workload. The mismatch is between the
real corpus's seq_len distribution (~500 tok MMLU) and a 16 GB GPU's logits
budget at f32.

**Mitigations** (ranked by code cost / effectiveness):

1. **Lower prompt_max_tokens to 256** — immediate, no code change. Forces
   corpus to ≤256 tok; halves logits memory; should fit. Cost: MMLU coverage
   drops because many MMLU prompts are 256-503 tokens. Tradeoff:
   capability-eval-matching vs runnable.
2. **Chunked-logits / streaming KL** (medium impl): compute logits + KL over
   chunks of seq_len, accumulate loss without materializing full `[B, S, V]`.
   Biggest structural win. Affects `crates/train/src/loss.rs:89-115` and the
   eval path.
3. **BF16 logits + BF16 KL** (medium impl, depends on kernel support): halves
   each `[B, S, V]` tensor. Easier than (2) but smaller win.
4. **Activation checkpointing in student forward** (per codex tranche table
   Commit 3): orthogonal to logits memory; helps overall activations, not
   primarily the `[S, V]` peak.
5. **BF16 rank-1 norm weights** (per codex Commit 3a): negligible savings on
   this shape — won't move the needle.

## Rule

Do not run GKD λ-mixing with the real MMLU+GSM8K corpus on this 16 GB GPU at
`prompt_max_tokens=512` without one of mitigations 1-3 above. The fail mode is
deterministic OOM during eval-step-0 logits allocation, not a marginal
stochastic failure.

**Pre-flight requirement** going forward: any new `--prompt-max-tokens` must
be smoke-tested with `--eval-steps 0` only against a fresh GPU before
licensing a multi-thousand-step run. Eval at step 0 is where big-seq logits
get allocated; if it fits there, it fits during training too.

**Diagnostic mistake to avoid (meta-rule)**: my first-pass root cause
(`prompt_max_tokens × rollout_len` tape preallocation) was an unverified
hypothesis from naming conventions, not from reading the allocator. Codex's
counter-hypothesis was grounded in `backend_cuda.rs:1140-1148` and proved
correct. SOLID §0 reminder: 推断 ≠ SOLID. Evidence = source-grounded
allocator quote, not a plausible-sounding formula. Always cite file:line for
allocation-rule claims.

**Next licensed actions (parallel)**:

- P5 pure OPD 5k at the known-good 16-tok shape (already launched — runs).
- After P5 lands or fails: retry GKD-real-corpus with `--prompt-max-tokens 256`
  (mitigation 1) — drops MMLU coverage to ~6 rows but tests the GKD-with-
  realistic-completion hypothesis cheaply.
- For a clean full-MMLU retest: chunked-logits KL implementation
  (mitigation 2) — license once P5 closes.
