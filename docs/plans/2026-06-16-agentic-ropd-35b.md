# Plan — Agentic ROPD on Qwen3.6-35B-A3B (LoRA, 8×H20)

**Date**: 2026-06-16 · **Driver**: ckl ("直接用 35B A3B 做好 agentic rl") · **Status**: plan → needs decisions (§Open)

**Goal.** Make Qwen3.6-35B-A3B good at agentic tasks via **ROPD** (rubric-graded
on-policy distillation) trained as a LoRA on the 8×H20 pod — the ARLE-differentiated
form of "agentic RL" (ckl decided **never GRPO**, issue #98 / `docs/research/2026-06-14-rubric-opd.md`,
Path A). Headline output = an agent-benchmark capability curve (base 35B vs ROPD-LoRA).

---

## Load-bearing feasibility verdict (VERIFIED in code)

**We cannot train a 35B-A3B LoRA via the autograd path today.** Three independent,
each-sufficient blockers:

1. **Autograd training is single-device.** No NCCL / TP / all-reduce anywhere in
   `crates/autograd` (`backend_cuda.rs` binds one device/one stream). TP/EP/NCCL exists
   only on the **inference** side (`infer-cuda/src/tp.rs`). No tensor-parallel *training*.
2. **The training model is dense-only.** `train/qwen35.rs` builds a dense SwiGLU MLP per
   layer with `num_experts: 0`; there is no router / no routed-expert forward or backward
   in the autograd path. The MoE (E=256, top-8) exists only in the inference executor.
3. **Memory + frozen-base list.** `estimate-memory` is fp32 single-device; the bf16-frozen-base
   LoRA optimization's allowed-tensor list (`qwen35_loader.rs`) is **dense names only** — no
   expert tensor — so even frozen-base can't host the 35B MoE weights.

→ "Train 35B-A3B LoRA via autograd" is a **from-scratch capability build** (multi-GPU TP/EP
training + MoE training layer + sharded backward), **not** a tuning task. This is the
dominant fact. See Open Decision #1.

---

## Foundation prereqs (gate everything; in flight)

| # | Prereq | Status |
|---|---|---|
| **F1** | cuBLASLt logits-GEMM SIGFPE (large-vocab `matmul_bt` OP_T) | Fix in working tree (vocab-chunk ≤65536, `backend_cuda.rs`); **needs commit + on-GPU verify** (tmux0 verifying) |
| **F2** | **35B-A3B inference emits degenerate output** (rewrite CUDA path, TP=1 & TP=2; `errors/2026-06-11-...degenerate-output.md`, root cause **OPEN**) | **HARD GATE** — on-policy distillation is meaningless if the generator emits garbage. May have improved post-FP8-win; **must re-confirm live on the pod.** |
| **F3** | 35B-A3B load single-threaded slow (#101) | Partial (`26d7d4e5` shard-cache); iteration-speed prereq |

---

## What exists vs gap (the ROPD axis)

- **Agentic rollout** — `crates/agent` is strong: `run_turn` multi-turn loop, real `ToolExecutor`,
  `TokensRecord{prompt_ids,response_ids,response_mask}` (verl-compatible mask). **Gap**: adapter
  feeding `run_turn` output (against the LoRA-synced student) into `opd_step`'s `forced_rollout` + mask.
- **Reward/rubric** — seam exists (`trajectory_scorer.rs`: `TrajectoryScorer` + `ExactMatchScorer` +
  `select_best`); `xgrammar-sys` present for constrained decode. **Gap (net-new)**: Rubricator
  (judge-forward auto-rubrics), Verifier (blind per-criterion), `RubricScorer`. Trait is
  token-vector-shaped; agentic scoring is text/trajectory-shaped → minor trait extension.
- **ROPD loss** — **minimal change**: `GkdSftAnchor::StudentRollout` + `mix_gkd_losses` already do
  `CE(student‖rollout)` blended with KL→EMA. Path A = "sample N → `RubricScorer` → `select_best` →
  τ* as `forced_rollout`". **No new loss kernel, no GRPO, no negative gradient.**
- **Eval** — **does not exist.** `agent-bench` is throughput-only (Echo/latency). No task benchmark,
  capability scorer, or curve harness. Largest net-new product surface.

---

## Dependency DAG / critical path

```
F2 (fix 35B inference coherence) → F1 (SIGFPE) → T0 (train-path decision)
  → T1/T2 (infer-engine-as-student + LoRA-on-MoE) → T3 (LoRA sync) 
  → R1→R2 (agent rollout into opd_step) → S2/S3/S4 (rubric machinery)
  → L1→L2 (best-of-N ROPD step) → E1→E2→E3 (capability curve, ≥5 seed + Wilson CI)
```
Critical path dominated by **F2 + the train-path build (T0–T2)** and the **net-new eval surface
(E1–E3)**. Rubric machinery (S2–S4) parallelizes once the scorer trait shape is fixed.

---

## Open decisions (need ckl)

1. **Train-path fork** (everything hangs on it; autograd can't train 35B today):
   - A. Build TP-EP training in autograd — months; duplicates inference TP/EP. **Not recommended.**
   - **B (recommended). Infer-engine-as-student + restricted LoRA backward** — use the proven
     inference MoE+TP=8 forward via `InferStudent` (`infer_student.rs`, `sync_lora_from_store` exist),
     LoRA backward only over a small trainable set (e.g. attn q/v). **Un-derisked at 35B** (only the
     4B/0.8B dense bring-up has run).
   - C. External trainer (verl) for LoRA, ARLE as rollout+serve substrate — fastest curve, off-thesis.
2. **Agent benchmark + reward source** — none in-repo. Pick benchmark (tau-bench / SWE-bench-lite /
   custom #97 suite) + reward (EMA-judge teacher-free default, vs black-box frontier API teacher = A4).
3. **Scope vs "direct to 35B"** — going direct loses the cheap 4B loop that would de-risk the
   InferStudent LoRA-sync + rubric machinery on a *coherent* model. Recommendation: build R/S/L
   machinery on the 4B (cheap, coherent) **in parallel** while F2 + train-path are de-risked, then port.

---

## Scope / biggest risk

**Weeks, not days; contingent on Decision #1.** The ROPD loop itself is small (reuses
`mix_gkd_losses` / `StudentRollout` / `select_best` / `xgrammar-sys`), but sits on **two un-derisked
multi-week prereqs** — **F2 (35B inference coherence, root cause OPEN)** and the **35B train path
(T0–T2)** — plus a fully net-new eval surface.

**Single biggest risk = F2.** On-policy distillation needs a coherent on-policy generator; per
`errors/2026-06-11` the 35B-A3B rewrite path emits deterministic garbage (resolution "None yet").
**Re-confirm 35B inference coherence live on the pod before any ROPD build starts.**
