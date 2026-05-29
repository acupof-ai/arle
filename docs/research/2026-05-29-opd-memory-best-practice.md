# OPD single-GPU memory: industry best-practice → ARLE mapping

**Date**: 2026-05-29
**Source**: deep-research workflow (verl, TRL GKD, OpenRLHF, Thinking Machines
On-Policy Distillation/Tinker, vLLM sleep mode, Unsloth FP8-RL — adversarially
verified, cited in the run transcript).
**Trigger**: user — "训练部分对比业界太差", "用这么显存合理吗", "调研最佳实践".

## What the industry actually does (grounded)

1. **Never co-reside two full models on one GPU.** Two topologies:
   - **Disaggregated**: separate GPU pools (train Actor vs gen Rollout), weights
     synced over NCCL/ZMQ (verl async/OPD).
   - **Colocated + time-share**: one GPU set, only one role hot at a time, via
     **vLLM sleep mode** — Level 1 (weights→CPU, discard KV), Level 2 (discard
     both). `--vllm_enable_sleep` + `--deepspeed_enable_sleep` (OpenRLHF),
     co-located vLLM (TRL), HybridFlow offload-gen-weights-during-training
     (verl). The generation engine is **offloaded during backward**.
2. **Teacher = scoring-only.** A single forward (`max_tokens=1` +
   `prompt_logprobs`, top-k) grades the student's already-generated tokens;
   no sampling, no generation engine, replaces the reward model
   (Thinking Machines OPD, verl OPD). **ARLE's teacher already does this**
   (`forward_logits_device` over the prefix).
3. **Activation memory**: gradient/activation checkpointing + BF16
   saved-activations are standard; practitioners cut in the order
   **activations → optimizer state → weights**.
4. **ARLE's exact case is unaddressed by OSS.** No surveyed framework does a
   quantized-resident teacher, cached teacher logits, or self-distill teacher
   hot-swap. The one weight-buffer-sharing trick (Unsloth FP8-RL: gen+train
   share one FP8 buffer, transient BF16 dequant in backward) is for the
   generate==train *policy* case, **not** teacher+student coexistence.

## Mapping to ARLE (no PyTorch, own autograd + own infer engine, 16GB)

ARLE today holds **3 copies hot**: infer-student (BF16 ~1.6GB) + train-student
(BF16 ~1.6GB + F32 tape) + teacher (BF16 ~8GB) — the anti-pattern the research
says to avoid. The OPD step has two **memory-disjoint phases**:

- **Rollout**: only the infer-student is needed.
- **Score + backward**: teacher scores → `teacher_logits` cached as a tensor →
  then train-student backprops. The infer-student is idle here; and once the
  teacher has emitted logits, the teacher weights are idle during the backward.

### The key insight for ARLE's scale
The sleep/offload pattern exists because for **large** models quant isn't
enough. For a **4B teacher, W4-resident (~2GB) IS enough** and avoids the
per-step offload cost (offloading 8GB teacher ⇄ CPU ≈ 1.4s PCIe round-trip per
step, heavy at ~13s/step). ARLE's Marlin/GPTQ path keeps weights **packed-
resident** (`weight_loader.rs:622/776`), unlike its FP8 path which dequants to
BF16 (`:1001`, why the FP8 teacher attempt freed 0GB). So:

- **Tier 1 (chosen, in progress): W4-resident teacher.** 8GB → ~2GB, frees
  ~6GB → rollout-128/256 fit. Teacher precision is irrelevant (KL target), so
  naive RTN W4 — no calibration. This is *ahead* of OSS (no framework does a
  quantized-resident teacher) and right-sized for a 4B teacher. Avoids offload
  latency entirely.
- **Tier 2 (fallback if W4 insufficient or rollout ≫256): phase-based
  offload/sleep** — the vLLM-sleep-mode equivalent ARLE lacks: offload teacher
  after scoring (logits already cached), sleep infer-student during backward.
  Bigger change (needs a weights→CPU offload/reload API on
  `LoadedInferenceEngine`); only worth it if Tier 1 doesn't fit the target.

## Verdict

For ARLE's 4B-teacher + 0.8B-student-LoRA on 16GB, **W4-resident teacher
(Tier 1)** is both the simplest and the most appropriate fix — and the research
confirms no OSS does better for this specific co-residency case. Pursue Tier 2
(engine sleep/offload) only if rollout length must scale past what W4 frees.
The student-side activation tape (gradient checkpointing) remains the lever for
*very* long rollouts, per the standard cut order.
