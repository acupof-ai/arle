# RL Training Acceleration — Systematic State + ARLE OPD Grounding

**Date**: 2026-06-22
**Track**: research / OPD
**Type**: survey (source-reported numbers = hypothesis) + code-grounding (file:line = evidence)

Two-part note. Part A surveys the systematic state of RL-training acceleration.
Part B grounds the two ARLE-specific questions it raised against the actual
`train` / `infer-*` source, and corrects one survey-level claim with code.

**SOLID labelling (§0)**: every magnitude in Part A is **self-reported by the
cited source**, not a local measurement → hypothesis-grade. Every claim in
Part B carries a `file:line` and is **code-grounded** → evidence. The one
deferred item (MoE-scale mismatch behaviour) is called out explicitly, not
silently passed.

---

## Part A — The field

### A1. One axis explains the whole field: generation is the bottleneck

Autoregressive rollout generation is **70–81 %** of an RLHF/RLVR iteration
(NeMo-Aligner: 81.2 %; FP8-rollout papers cite >70 %). Every systems lever is
therefore an *inference-side* lever: the question is always "stop the training
GPUs idling while the generator runs." This is the structural reason an
inference-runtime-led project (ARLE) is positioned on the RL/post-training
leverage point rather than beside it.

### A2. Design axes (HF "16 libraries" taxonomy + SemiAnalysis "mind the gap")

- **Sync vs async**: sync alternates gen/train (generation waits for the
  *longest* sample in the batch → idle); async decouples them under a staleness
  budget η (η=0 ≡ sync). Async is the 2025 mainline (AReaL, ROLL Flash, slime,
  Laminar, PipelineRL).
- **Colocated vs disaggregated**: colocated time-shares one GPU pool (verl
  HybridEngine — serial within a PPO step); disaggregated runs separate
  inference/training clusters (concurrent, but cross-cluster weight sync +
  resharding cost). Field is trending disaggregated+async for long reasoning.
- **Throughput-matching is a queue problem**: trainer-consume rate must ≈
  generator-produce rate, else starvation or staleness. Measured pathologies:
  Qwen3-235B long-output → trainer idle 30 %, MFU 10.5 %; GLM-5 high-tool-use →
  trainer idle 74 %.
- **Long-tail straggler**: "the longest rollout sets the group's completion
  time." Levers: oversampling (wastes ~60 % compute), early-pruning, **partial
  rollout** (APRIL / RollPacker / SkyRL prefix-resume), PD disaggregation, or
  async (root cure).
- **Weight sync** (the disaggregated core cost): per-param NCCL → **bucketed
  NCCL** (~1 GB uint8 buffers; vLLM `NCCLWeightTransferEngine`, slime) →
  **RDMA P2P** (LMSYS: Kimi-K2 1T FP8 / 32 nodes, **53.3 s → 7.2 s, 7.37×**;
  DSv3.2 744B 6.88×; Qwen3-235B 3.40×) → Awex / SparseRL-Sync (~100×
  compression). vLLM shipped native `update_weights` RL APIs (2026-05).
- **Quantized rollout / spec-decode**: 8-bit rollout = +20–80 % throughput, but
  **BF16-train + FP8-rollout collapses** under long rollouts → Jet-RL unifies
  train+rollout precision; QuRL INT8 +1.7 %. Spec-decode for rollout scaled to
  2048× GB200.

### A3. Training–inference mismatch — a *correctness* axis, not just speed

Even with bit-identical weights, the inference engine (vLLM/SGLang, FlashInfer/
FA3/DeepGEMM kernels) and the training engine (FSDP/Megatron) compute
**different token logprobs**: float-add non-associativity, batch-size-dependent
reduction order/tiling, atomic-add, and — the amplifier — **MoE routing flips**
(a logit nudge selects a different expert). Mean |δ| is small but max δ can hit
**1.0** → argmax flips that silently corrupt the gradient (`πvllm=1, πfsdp=0`).

Measured magnitude (slime/Miles, K3-KL):
- **Dense**: 1e-5 … 1e-3.
- **MoE**: 1e-3 … **1e-1** (two orders larger; expert-selection amplification).
- Reproducible **MoE collapse at step ~320**; precursor = grad-norm drop
  0.07 → 0.02.

Corrections, by loss form:
- **Policy-gradient / RL (PPO, reverse-KL-as-reward)**: needs **TIS** (truncated
  importance sampling `min(ρ,C)`) — bound-sensitive (`[0.5,1.5]` survives,
  `[0.5,2.0]` collapses); **MIS** (TIS + reject); batch-norm IS to mean 1.
  Recomputation and bypass both *fail*.
- **Distribution-matching / GKD (forward-KL direct backprop)**: **no importance
  ratio exists**, so the catastrophic ratio-explosion mode does not apply. verl
  ships GKD (`recipe/gkd/megatron_kl_loss.py`, grad `student_probs −
  teacher_sparse_probs`) with **no IS even in the async one/two-step-off
  recipe** — it only "trades strict on-policy guarantees." The "OPD needs TIS"
  claim in general write-ups refers to *PG-flavoured* OPD, not GKD.
- **Truly-on-policy (terminal fix)**: batch-invariant kernels (RMSNorm/MatMul/
  log_softmax) + FA3 + DeepGEMM → **strict 0-KL**. Requires kernel ownership.

### A4. slime = the DSv4/GLM-5.2-architecture reference

slime (Megatron-train + SGLang-rollout, Ray) is the RL framework behind
GLM-4.5/5/5.2 and supports DeepSeek-V3 + Qwen3MoE — i.e. the DSv4
family (GLM-5.2 = the same MLA+DSA+FP8-MoE arch). It deliberately binds one inference engine
(SGLang), mirroring ARLE's single-engine bind. Weight sync:
`UpdateWeightFromTensor` (colocated, **CUDA IPC** — Megatron gathers the
distributed tensor, exports an IPC handle, SGLang `update_weights_from_tensor`
maps+reloads, zero CPU↔GPU copy) / `UpdateWeightFromDistributed`
(disaggregated, NCCL) / DCS async streaming + delta sync. Public docs do **not**
disclose the exact Megatron-EP → SGLang-EP MoE reshard flow or FP8-rollout
support — read source if ARLE needs them.

---

## Part B — ARLE grounding (code = evidence)

### B1. ARLE already implements "train-inference unified" rollout

OPD routes the **student rollout through the in-process infer engine**
(`InferStudent` = CUDA graph + paged KV), **default since P4**
(`crates/train/src/opd.rs:73-110`, `infer_rollout_flag_enabled()` opt-out via
`ARLE_OPD_INFER_ROLLOUT=0`; win `2026-05-29-opd-infer-rollout-default-p4.md`,
**5.0× step / 60.9× rollout**). Flow: student samples greedily on the infer
engine → teacher re-scores the same tokens → train autograd forward recomputes
student logits → forward-KL distill loss backward (`opd.rs:6-7`).

**This is exactly the field's "training-inference unified rollout"** (the
vLLM-rollout + FSDP-train structure of A3): two engines, same weights, different
kernels.

### B2. The train→infer weight path exists — but LoRA-adapter-only (corrects A-level (A))

A general `update_weights_from_tensor` equivalent does **not** exist in the
seam. `infer_seam::BackendExecutor` exposes only `offload_weights` /
`reload_weights` (`infer-seam/src/lib.rs:306-320`), whose doc-comments say
"move device weights to host RAM" / "restore from the host snapshot" — pure VRAM
time-share, **weights unchanged**. CUDA-IPC plumbing in `infer-cuda/src/deepep.rs`
serves DeepEP all-to-all, not weights.

**But** OPD has a per-step **LoRA-adapter-only sync**:
`InferStudent::sync_lora_from_store` (`infer_student.rs:269-280`, called every
step at `opd.rs:3088-3092`) D2H-copies the LoRA matrices from the train store →
`remerge_student_lora` / `StudentLoraUpdate`
(`infer-api/src/serve_engine.rs:297`, `loaded.rs:605`) re-merges into the
resident base. This is precisely the HF survey's recommended "adapter-only
sync" — ARLE already has the cheap path.

**Gap**: full-weight SOPD (no LoRA) would need a full-tensor reload path = the
slime `UpdateWeightFromTensor` / CUDA-IPC reference (A4). DSv4 EP=8 resharding
(train-EP → infer-EP layout) is the hard part. Not built.

### B3. ARLE's OPD loss is forward-KL direct-backprop, no IS — the robust regime

`kl_distill_loss` (`crates/train/src/loss.rs:33-41`) = forward KL
`KL(teacher‖student)`, minimising soft cross-entropy `−Σ t_p log s_p`,
backprop **through student only**; default `KlDirection::Forward`
(`opd.rs:238`); sparse top-k is forward-only (`loss.rs:378`). A `Reverse`
variant exists but is still direct-backprop, **not** a policy gradient — **no
importance ratio anywhere** in the loss. → Same family as verl GKD (A3), the
regime where the "secretly off-policy" ratio-explosion **cannot** occur.

### B4. ARLE already ships a coarse mismatch gate — and it passes bit-exact (dense)

The P2 **cross-path canary** checks the infer-rollout step-1 argmax against the
train-crate forward argmax (plan `2026-05-29-opd-student-rollout-via-infer.md:131-149`,
PASS ≥90 %, KILL <60 %). Bring-up result: **100 % / bit-exact** on Qwen3.5 dense
(P3 doc line 11). So ARLE is *not* naïve about the mismatch — it has a bit-exact
argmax-agreement gate, and at dense scale the two engines genuinely agree.

### B5. Verdict + where it breaks (dig to 95 %, defer the rest explicitly)

**SOLID now (dense bring-up)**: 训练推理一体 is real (B1) + the loss is ratio-free
forward-KL (B3) + the cross-path canary is bit-exact (B4). ARLE is **not** exposed
to the RL "secretly off-policy" collapse (slime MoE step-320). At Qwen3.5 dense
this is fully covered.

**Deferred / load-bearing at the real target** (35B-A3B / DSv4 MoE — the real
OPD target: DSv4-Flash teacher → Qwen3.6-35B-A3B student) — **not yet measured**:
1. **MoE expert-flip**: A3 shows MoE K3-KL jumps to 1e-1; the argmax canary will
   drop below 100 %. Upgrade the gate from argmax-agreement to a **K3-KL
   tolerance** on the full distribution, and re-confirm forward-KL absorbs it.
   Tripwire to watch: grad-norm 0.07→0.02 precursor.
2. **FP8 teacher fidelity**: when the teacher routes through the infer runtime in
   FP8 vs in-process autograd, the *teacher target itself* shifts. Forward-KL is
   only as good as the target → add a teacher-logit-fidelity check
   (KL(infer-teacher ‖ reference-forward) on the same tokens). Jet-RL lesson:
   never BF16-train + FP8-rollout without unifying precision.
3. **If SOPD turns RL/PG-flavoured** (reverse-KL-as-reward, not direct
   backprop): an importance ratio appears → **TIS/MIS becomes mandatory**. Until
   then, building TIS is over-engineering (B3).
4. **Full-weight SOPD**: needs the slime-style `update_weights_from_tensor` /
   CUDA-IPC reload (B2 gap) + EP=8 reshard.

**ARLE's structural edge**: it owns its kernels (DeepGEMM/FlashMLA already in
hand), so the *truly-on-policy* 0-KL route (A3 terminal fix) is reachable —
batch-invariant infer-rollout kernels matching the train forward — rather than
papering over with IS. That is the lazy *and* correct fix when (1) bites.

---

## Sources

Field: [HF — 16 RL libraries](https://huggingface.co/blog/async-rl-training-landscape) ·
[SemiAnalysis — mind the gap](https://newsletter.semianalysis.com/p/rl-systems-mind-the-gap-matching) ·
[AReaL](https://arxiv.org/abs/2505.24298) ·
[LMSYS P2P 1T-in-seconds](https://www.lmsys.org/blog/2026-04-29-p2p-update/) ·
[APRIL](https://arxiv.org/pdf/2509.18521) · [Jet-RL FP8](https://arxiv.org/html/2601.14243v1).
Mismatch: [Rollout-Training Mismatch](https://arxiv.org/html/2605.14220) ·
[slime/Miles mismatch blog](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/mismatch/blog-en.md) ·
[TRL #4159 vLLM temperature logprob](https://github.com/huggingface/trl/issues/4159).
OPD: [verl OPD](https://verl.readthedocs.io/en/latest/algo/opd.html) ·
[verl async OPD recipe](https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html) ·
[Thinking Machines OPD](https://thinkingmachines.ai/blog/on-policy-distillation/).
slime: [repo](https://github.com/THUDM/slime) · [DeepWiki](https://deepwiki.com/THUDM/slime).
