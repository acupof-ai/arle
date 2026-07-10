# Gated-delta kernel + kernel-org gap — world best-practice vs ours

**Date**: 2026-06-18  **Track 3** of
[`docs/plans/2026-06-18-research-first-best-practice.md`](../plans/2026-06-18-research-first-best-practice.md).
Author: Claude (ckl pulled this track off the session-2 codex — "tmux2 你不要用,
你自己来看看"). Read-only; **no code this pass**.

## ⚠️ CORRECTION (2026-06-18, adversarial review wf_6a493801)

**The "correctness bug" claim below is WRONG — retracted.** It described the
*uncommitted WIP working-tree* version of `linear_attention.cu` (424 lines, with
`s_decay/exp_g` reverse-division + `exp_g<1e-8→0` guard + `recover_previous_state`).
That version was real when read but **was never committed and is now deleted**. The
**committed** kernel (HEAD/`5b26db30`, 436 lines) uses a stored `state_history` buffer
+ multiply-by-`exp_g` (lines 87/196/201/341) — the textbook-correct adjoint, **no
division** (`git log --all -S 's_decay / exp_g_value'` = empty). So **there is no
shipping correctness bug**. The line numbers `:320/:321/:416` cited below do not match
HEAD. The real distinction: committed = correct **but memory-heavy** (`state_history`
is `[B,T,H,Dk,Dv]`); the FLA chunked backward is correct **and** memory-light
(chunk-boundary checkpoints + recompute) **and** faster (tensor cores). So P0 is a
**memory+perf** play that must clear the normal **license-or-kill perf/memory A/B bar
on H20** — NOT a correctness emergency, NOT a blocker. Keep the gradient-check as a
standard gate for any NEW chunked backward. Lesson: pin the exact version/SHA before
quoting source (§0: code is truth, and *which* code matters).

## Verdict (one line)

The gated-delta 闭门造车 is **exactly one decision**: the OPD **training
backward** was hand-rolled as a numerically-unstable scalar reverse-division
kernel in a scattered dir, while the repo *already* carries a **licensed,
canonical TileLang-AOT FlashQLA/FLA gated-delta forward** that is **forward-only
by an explicit TODO** — and both FLA and FlashQLA ship the matching **chunked
backward**. Fix = extend the existing AOT path with the chunked backward; delete
the hand-roll. Inference forward needs no change (already adopted).

## Evidence map (current source)

| Lane | File:line | State | Evidence/Hypothesis |
|---|---|---|---|
| Inference **fwd** (chunked) | `cuda-kernels/tools/tilelang/flashqla_gdr.py` (`fq_kkt`, `fq_fwd`, chunk=64); `gated_delta_rule.py` | **Adopted** — 1:1 TileLang AOT port of FlashQLA Hopper + FLA `chunk.py`; landed `91421983`, env-gated | EVIDENCE (read source + git log) |
| Inference **fwd** (fallback/decode) | `cuda-kernels/csrc/misc/gated_delta_rule.cu:209,249` `gated_delta_rule_prefill_recurrent_kernel` — serial `for token_idx in 0..seq_len`, 1 block/value-head, scalar | Serial recurrent; flagged **28.0% of prefill** at `qwen35.rs:206` | EVIDENCE |
| Inference fwd **backward** | `gated_delta_rule.py:44` | **"No backward, no varlen surface (single-sequence prefill only)"** — explicit gap | EVIDENCE |
| **Training bwd (the WIP)** | `autograd/src/backend_cuda/kernels/linear_attention.cu` (424 L) | Hand-rolled scalar; header (`:1-8`) self-labels a **"spike"** that "only replaces the `scan_state_history` host loop" | EVIDENCE |

### The training-backward defect (correctness, not just perf)

`linear_attention.cu` walks time in **reverse** (`:124`) and reconstructs the
previous recurrent state by **dividing by the forget gate**:

```
:320  float s_decay = state[idx] - k_vec[k]*delta[v];      // = exp_g · state[t-1]
:321  float prev = (seq_idx==0 || fabsf(exp_g_value)<1e-8f) ? 0.0f : s_decay/exp_g_value;
:416  state[idx] = (...exp_g<1e-8...) ? 0.0f : s_decay/exp_g_value;   // state[t-1]
```

`exp_g = exp(−exp(a_log)·softplus(a+dt_bias)) ∈ (0,1]` is **small precisely when
the gate forgets hard** — the defining regime of a *gated* delta net. So the
backward (a) divides by a near-zero gate → fp32 error blow-up, and (b) the
`exp_g < 1e-8 → 0` guard **silently zeros the gradient** in that regime. The WIP
diff confirms the trade: it *deletes* the exact `state_history` buffer
(`ops/linear_attention.rs` forward) and its CPU twin `recover_previous_state`
(`:965`, same `value/exp_g` + zero-guard) — **memory bought with wrong
gradients.** EVIDENCE (read .cu + git diff). The magnitude of the gradient error
is HYPOTHESIS until a gradient-check A/B (see Verification gate).

Perf, secondary: scalar, 1 block per (batch, value_head), serial `seq_len`-step
time-walk, **no tensor cores, no chunking** — far from FLA/FlashQLA.

## World best-practice (grounded)

| Source | What it is | Relevance |
|---|---|---|
| **FLA** `fla-org/flash-linear-attention`, `fla/ops/gated_delta_rule/chunk.py` | Canonical chunkwise GDN, **fwd+bwd**, chunk=64, fused Triton, **WY-form** (UT-transform) recurrence — backward via chunk-boundary states + intra-chunk recompute, **no reverse-division** | arXiv 2406.06484 (Yang et al., "Parallelizing Linear Transformers with the Delta Rule"). Adopted by **vLLM** and **SGLang** (V2 mamba scheduler *requires* the FLA kernel backend) — nobody hand-rolls |
| **FlashQLA** `QwenLM/FlashQLA` | Qwen-team GDN kernels **built on TileLang**; GDN **Chunked Prefill fwd AND bwd**; **2–3× fwd / 2× bwd vs FLA Triton**; **SM90+ (= our H20 sm_90)**, CUDA 12.8+ (pod 12.9), MIT; "algebraic reformulation **without losing numerical precision**" | The repo's `flashqla_gdr.py` is already its 1:1 port. Same library has the backward we're missing. The model authors' own kernels |
| Qwen3.5 = Gated DeltaNet 3:1 | 75% gated-delta + 25% full-attn | matches our 30 linear / 10 full (`full_attention_interval=4`) → backward is on the **35B-A3B OPD-student critical path** (30 of 40 layers) |

## Adopt list (ranked, proposal — await approval, needs H20-on to build/test)

1. **Extend the existing TileLang AOT GDR with the chunked backward.** Add bwd
   stages to `tools/tilelang/gated_delta_rule.py` / `flashqla_gdr.py` by porting
   FLA `fla/ops/gated_delta_rule/chunk.py` backward (or FlashQLA's Hopper bwd) —
   chunk=64, WY-form, **reuses the already-licensed forward recompute** →
   numerically stable, no `exp_g` division. This *removes* the "No backward"
   TODO at `gated_delta_rule.py:44`, it doesn't add a new dependency.
2. **Wire the AOT backward into autograd** — `backend_cuda.rs::cuda_linear_attention_scan_backward`
   calls the cuda-kernels GDR bwd instead of launching the hand-rolled kernel.
3. **Delete the hand-roll** — `autograd/src/backend_cuda/kernels/linear_attention.cu`
   + the WIP `recover_previous_state` reverse-division in `ops/linear_attention.rs`.
   (Resolves the 4-file WIP; restore the exact `state_history` path on the CPU
   reference branch only, as the gradient-check oracle.)
4. **Org / unification** — one GDR kernel family lives in **`cuda-kernels/`**
   (csrc + tools/tilelang), serving **both** inference (fwd, done) and OPD
   training (fwd+bwd, new). Autograd FFIs into it; nothing under
   `autograd/src/backend_cuda/kernels/`. Open design Q (confirm before code):
   the autograd-crate → cuda-kernels build dependency direction.

## Verification gate (before any default flip)

- **Gradient-check** the new chunked backward vs the *exact* `state_history`
  reference (the deleted path, kept on a test branch) — relative error per
  output (dq/dk/dv/dnorm/da_log/ddt/dbeta). **Must include the hard-forget
  regime** (small `exp_g`) where the reverse-division silently zeroed.
- This is a CUDA build → needs the pod sm_90a TileLang AOT toolchain; defer to
  H20-on + approval. Research/plan only this pass.

## Sources

- FLA — https://github.com/fla-org/flash-linear-attention (`fla/ops/gated_delta_rule/chunk.py`)
- Delta-rule chunk algorithm — https://arxiv.org/pdf/2406.06484
- FlashQLA — https://github.com/QwenLM/FlashQLA ; https://www.alibabacloud.com/blog/603084
- vLLM Qwen3-Next (FLA Triton) — https://blog.vllm.ai/2025/09/11/qwen3-next.html
- SGLang Qwen3-Next (FLA backend) — https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3-Next
- Qwen3.5 gated-deltanet analysis — https://gist.github.com/justinchuby/0213aa253664fb72e9adb0089816de15
