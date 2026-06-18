# Research-first: world best-practice vs ours (no code, no H20)

**Date**: 2026-06-18  **Driver**: ckl — "闭门造车" critique. Codex agents kept
reinventing infra (engine-offload ≈ vLLM sleep-mode + verl HybridFlow; grad-ckpt
≈ standard FSDP; `linear_attention.cu` hand-rolled vs FLA/SGLang) **despite the
repo already holding the best-practice mappings**. Stop coding, stop H20,
research first, then adopt.

## Shared rules (ALL tracks — non-negotiable)

1. **No implementation code this pass.** No `.cu`/`.rs` edits. Do **not** touch
   the 4 WIP files (`crates/autograd/src/{backend.rs, backend_cuda.rs,
   backend_cuda/kernels/linear_attention.cu, ops/linear_attention.rs}`) — they
   are the scatter to be superseded; leave them for the adopt plan.
2. **No H20, no pod, no training, no bench.** Pure literature + local source
   reading. If you are mid-pod-command, stop it now (Esc).
3. **Read the repo's existing research FIRST** (listed per track). 闭门造车
   happened *with these already on disk* — your job is not to re-discover them,
   it is to find where **current source** diverges from what they + world
   best-practice prescribe.
4. **Gap = current source vs best-practice, with `file:line` evidence.** Label
   evidence vs hypothesis (CLAUDE.md §0). A source-survey is hypothesis until a
   citation or a second read confirms it.
5. **Web research via `/browse`** (gstack skill) — never `mcp__claude-in-chrome`.
   Cite every external claim (paper / repo path / commit).
6. **Output** one doc `docs/research/2026-06-18-<track-slug>.md` with a table:
   `世界最佳做法 (引用来源) | 我们当前做法 (file:line) | gap | adopt 清单 (删什么/换什么/留什么)`.
   End with a ranked adopt list (highest leverage first) — **proposal only,
   await approval before any code.**

---

## Track 1 (session 0) — OPD 训练显存 / 吞吐 infra

**Existing docs to re-read first**:
`docs/research/2026-05-29-opd-memory-best-practice.md` (already maps verl
HybridFlow / vLLM sleep L1+L2 / TRL co-located / Tinker / Unsloth FP8-RL),
`2026-05-28-opd-rollout-perf-208s-bottleneck.md`,
`2026-05-26-opd-route-b-perstep-perf-audit.md`.

**Benchmark against**: verl HybridFlow (offload gen-weights during backward),
vLLM sleep mode (L1 weights→CPU/discard-KV, L2 discard both), TRL co-located
GKD, OpenRLHF `--vllm_enable_sleep`, FSDP activation-checkpointing + CPU
activation-offload, **Liger-Kernel fused-linear-cross-entropy / fused-linear-JSD**
(the industry trick for the `[seq, vocab]` logits-memory problem).

**Current source**: `crates/train/src/opd.rs` — `EngineOffloadMode` /
`ARLE_OPD_ENGINE_OFFLOAD`; grad-ckpt (commit `5c7fa6f1`); windowed-KL
`backward_windowed_pure_kl_cached_student_hidden` (opd.rs:1951); the two fixed
leaks (`f3a690dc` device-resident frozen-base input-grad, `5c7fa6f1` grad-ckpt).

**Key questions**:
- For the real target (**35B-A3B-FP8 teacher + student colocated, H20×8**), is
  our `engine_offload` reinventing sleep-mode, or is it the correct Tier-2? Is
  there a simpler **Tier-1 (W4/FP8-resident teacher, no offload)** as the
  2026-05-29 doc found for the 4B teacher — does it still hold at 35B?
- Is our hand-rolled grad-ckpt equivalent to FSDP/PyTorch standard, or weaker?
- **windowed-KL vs Liger fused-linear-JSD/CE** — is our window scheme reinventing
  chunked-logits distillation? Which is the canonical memory-optimal form?

---

## Track 2 (session 1) — on-policy 蒸馏配方 (recipe)

**Existing docs to re-read first**:
`docs/research/2026-05-25-opd-methodology-audit.md` (lists 4 missing GKD knobs),
`2026-06-14-rubric-opd.md`, `2026-06-14-self-training-lora-options-survey.md`,
`2026-05-28-opd-effect-axis-next.md`.

**Benchmark against**: GKD (Agarwal et al., ICLR 2024), TRL `GKDTrainer`,
MiniLLM (reverse-KL), Thinking Machines "On-Policy Distillation" / Tinker,
DeepSeek-R1-Distill-Qwen, Qwen distillation reports.

**Current source**: `crates/train/src/opd.rs`, `crates/train/src/loss.rs`,
`crates/cli/src/args.rs` (TrainOpdArgs).

**Key questions** — the 2026-05-25 audit found 4 gaps; **verify which are NOW
fixed in current source, file:line each**:
1. distillation temperature (γ) — present?
2. stochastic-sampling rollout — `b092f4aa` landed temperature-sampled rollout
   **opt-in, greedy default**; why is on-policy stochastic not the default per
   GKD's `γ=1` recommendation?
3. completion-only token masking — do we still compute KL over prompt tokens?
4. LR schedule / warmup wired into the OPD path?
- Canonical recipe convergence: loss = forward-KL / reverse-KL / JSD? on-policy
  data fraction? rollout length (default vs the 2048 we now run)? Which knobs
  must flip default / wire into CLI?

---

## Track 3 (session 2) — gated-delta 核 + 核统一 (kernel org)

**FIRST: Esc to stop the DSv4 pod build — no H20 this pass.**

**Existing docs to re-read first**:
`docs/research/2026-05-28-arle-kernel-vs-sota-audit.md` (scores every kernel
PASS/GAP vs SOTA), `2026-05-29-oplib-sota-kernel-gap.md`,
`2026-06-04-sglang-operator-selection-dsv4-qwen3moe.md`.

**Focus A — gated-delta backward kernel**: the WIP
`crates/autograd/src/backend_cuda/kernels/linear_attention.cu` is a hand-rolled
gated-delta backward scan (one block per (batch, value_head), reverse walk).
Benchmark against **Flash-Linear-Attention (FLA)** chunked gated-delta-rule
kernels (fwd **and** bwd) and SGLang gated-delta. Can FLA's bwd be adopted
directly (Triton AOT / port)? Is the hand-roll even correct, and why not FLA?

**Focus B — kernel unification**: scattered `crates/autograd/src/backend_cuda/
kernels/` vs canonical `crates/cuda-kernels/csrc/{attention,gemm,kv,quant,misc}/`.
Audit every `.cu` under `crates/autograd/` → table of "which `csrc/` subdir it
belongs in" + the build wiring (`cuda-kernels` build.rs) the move needs.
