# Plan — 训推一体: Train vs Inference Operators, Kernel-Level Diff + Native-Training Path (A)

**Date**: 2026-06-16 · **Driver**: ckl · **Status**: review **CORRECTED to kernel level** (supersedes the earlier "Path B = share inference forward" framing, which was wrong at the kernel level)

## 1. Verified facts (grep)

Train(autograd) and inference(infer-cuda) share **~0% operator/kernel code**. `grep cuda_kernels
crates/autograd/src/` = 0 (autograd uses cudarc only); `grep infer_seam crates/train/src/` = 0.
autograd has its own 28-file NVRTC `.cu` tree (+ `*_backward.cu`). Shared = `qwen35-spec` (contract)
+ `infer-api` `LoadedInferenceEngine` (a runtime bridge). Symptoms: the OPD logits cuBLASLt SIGFPE
(autograd's own cuBLASLt, separate from inference's crash-fixed `gemm_cuda`; N-padded 6996575f),
dense-only (`num_experts:0`, qwen35_loader.rs:322), no-TP.

## 2. THE KERNEL-LEVEL DIFF — why the two are antithetical (the load-bearing correction)

Read at code level: inference kernels are built on the **opposite** principle from what backward needs.

| | Training kernel (autograd) | Inference kernel (cuda-kernels) |
|---|---|---|
| Intermediates | **materialized + saved**: `silu_backward(x)` reads saved input; `softmax_backward(output)` reads saved output; scores live in smem/HBM | **fused + discarded**: silu folded into final `act` (register-only, dsv4_fp8_grouped_swiglu_decode); `fused_attention` **online-softmax never materializes S×S scores** |
| Precision | **f32** accumulate + grads | **bf16 / FP8** (w8a16 on-the-fly dequant) |
| Fusion | **decomposed** (separate silu/matmul/softmax) so the tape inserts save+backward per boundary | **fused across boundaries** (gate+up+swiglu+down; online-tiled attention) *precisely to avoid materializing* |
| Backward | paired `*_backward.cu` read the saved tensors | **none** (forward-only) |
| Layout/shape | contiguous, full-batch, save-all | paged-KV, grouped-by-expert, decode-band B=1, online-tiled |

**Consequence:** you **cannot "add a backward" to the inference fused/online kernel** — the
intermediates the backward needs (`gate·x`, `up·x`, the full attention scores) are **never written**;
fusion + online-softmax + quant exist *to not write them*. The duplication is **mostly justified**
(two antithetical requirements), **not waste**. autograd's "slow" GEMM+materialized-softmax attention
is the *correct* training design — that is why the 4B-dense bring-up works.

## 3. Corrected verdict

- **WRONG (earlier framing):** "share inference forward, autograd adds backward, delete the autograd dup."
  At kernel level this is impossible — inference forward kernels don't materialize what backward reads.
- **Shareable = math/numerics + contracts (`qwen35-spec`), NOT kernels.** Keep two kernel sets.
- **Move-#1 / forward-through-inference is right ONLY for the ROLLOUT** (forward-only: rollout
  generation + teacher scoring + logits value). It does **not** serve the training forward (which
  needs save-for-backward). It kills the rollout-path cuBLASLt SIGFPE and gives the rollout MoE+TP —
  do it, but it is **not** "35B training unlocked".

## 4. Path A — native training runtime (the thesis; C/external-trainer rejected: it concedes ARLE can't train)

**Precision = FP8 LoRA (QLoRA-style):** frozen base stored FP8 (35B ≈ 35GB vs 70GB bf16 — fits H20
training) + f32 LoRA adapters trainable. QAT-consistent (train in the served precision = the OPD+QAT
convergence target). **Orthogonal to the kernel work:** FP8 changes weight storage + adds a forward
dequant; it does **not** provide the backward — the LoRA backward still needs the frozen base's VJPs
(MoE/attn/TP), and the inference FP8 forward discards intermediates. So Path A still writes
training-grade (save-for-backward) FP8-base fwd/bwd kernels; FP8 just makes 35B fit + matches deploy.

35B-A3B = MoE+TP. Training a LoRA needs a **training-grade** (save-for-backward / recompute, f32-accumulate
of the FP8-dequant'd base, grad-checkpointed) forward+backward through the frozen base — **distinct kernels**
from the inference fused/quant/online set (borrow the math, not the code), **extending** autograd's existing
dense training set.

What autograd HAS (dense, save-for-backward + backward): silu, matmul (cudarc), rms_norm, rope,
softmax, embedding, materialized-scores attention. **This is why 4B-dense trains.**

The 35B gap — NEW training-grade kernels to write:
- **MoE forward (save-for-backward) + backward**: router (top-k gate; backward = gate-softmax grad through
  selected experts) + grouped-expert GEMM (fwd materializing per-expert intermediates; bwd). **Riskiest new piece.**
- **Attention forward+backward at scale**: flash-style forward instrumented to **recompute scores in the
  backward** (training-flash ≠ inference-flash, which discards scores). f32 accumulate.
- **TP all-reduce in the backward** (NCCL exists on the inference side; wire into the autograd backward).
- **Gradient checkpointing** (afford 35B activations / recompute).
- Reuse: cuBLAS GEMM backward (∂x=Wᵀ∂y, frozen W), adamw, LoRA-grad — already present.

**License-or-kill de-risk (do first, cheapest, riskiest piece):** write the **MoE-layer training kernel
(save-for-backward fwd + bwd)** + one LoRA grad through it on a **small MoE**, **finite-difference-checked**.
Pass → MoE-training core proven, scale to attention-bwd → TP-bwd → grad-ckpt → 35B. Fail → the top-k /
grouped-GEMM / EP backward is the real wall; reassess before burning 35B.

## 5. Tracks

```
轨0 配方验证(now, autograd dense已能): 4B corrected arm → GSM8K 方向 (independent of A)
轨1 rollout 走推理(forward-only): InferStudent/InferTeacher for rollout+teacher; kills rollout-path SIGFPE; gets MoE+TP rollout
轨A 原生训练算子(thesis, 35B唯一入口):
  A0 de-risk: MoE-layer training kernel (save-for-bwd fwd + bwd) + LoRA-grad finite-diff (small MoE) — license-or-kill
  A1 attention fwd+bwd at scale (flash + recompute-in-backward, f32 accum)
  A2 TP all-reduce in autograd backward (reuse infer NCCL)
  A3 gradient checkpointing
轨3 agentic ROPD(gated 轨0配方正 + 轨A 跑通 + F2绿)
```

## 6. Thesis

ARLE = native Rust/CUDA inference **and** training runtime. The two kernel sets are antithetical by
requirement (forward-only-fused-quant vs save-for-backward-f32) — keeping both is correct. Path A
(write the training-grade MoE+TP kernels) is the native-training work that *is* ARLE's reason to exist;
it duplicates neither the inference kernels (antithetical design) nor verl (PyTorch-autograd-over-PyTorch;
ARLE = own tape over own CUDA ops). The forward already exists to inform the math; only the
training-grade fwd/bwd kernels are the build.
