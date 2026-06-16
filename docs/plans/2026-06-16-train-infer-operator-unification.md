# Plan — 训推一体: Train/Inference Operator Unification (Path B finish)

**Date**: 2026-06-16 · **Driver**: ckl ("训推一体算子架构好好梳理") · **Status**: review → moves (接 #102 Path B)

## Verdict (verified by grep, not inference)

Train (autograd) and inference (infer-cuda) share **~0% of operator/kernel code** — two
disjoint CUDA stacks. `grep cuda_kernels crates/autograd/src/` = 0 (autograd uses cudarc
only); `grep infer_seam crates/train/src/` = 0. autograd carries its own 28-file NVRTC `.cu`
tree (+ `*_backward.cu`) reimplementing every op. **cuda-kernels is NOT the seam.** Only
shared: `qwen35-spec` (model contract) + `infer-api`'s `LoadedInferenceEngine` (Path B bridge).

| Operator | Train (autograd) | Inference (infer-cuda) | Shared? |
|---|---|---|---|
| Attention | GEMM+softmax+host S×S mask (no flash) | nonpaged_prefill / FA3 / batched / FlashMLA | duplicated-divergent |
| GEMM/linear | cudarc cuBLASLt (own handle/algo-cache) | cuda-kernels FFI + TileLang/DeepGEMM/quant | duplicated-divergent (two cuBLASLt) |
| **MoE** | **ABSENT** (`num_experts:0` hardcoded, qwen35_loader.rs:322) | router+grouped GEMM+EP | **train-missing** |
| logits/norm/rope/embed | own NVRTC + f32; logits=cudarc (the SIGFPE path) | cuda-kernels FFI, fused/quant | duplicated-divergent |

## One root cause behind every recurring training bug

- **cuBLASLt logits SIGFPE** = autograd's own cuBLASLt (separate from inference's crash-fixed
  `gemm_cuda`); the N-pad fix (6996575f) is a band-aid on the *duplicated* path, not a root fix.
- **dense-only / no-MoE** = training cannot build the production MoE graph (35B-A3B, DSv4-Flash).
- **no-TP/FA3/FlashMLA/quant** = all behind cuda-kernels FFI, structurally unreachable from the autograd tape.

All one cause: **training reimplements instead of sharing inference's operators.**

## Unification verdict — Path B confirmed ("forward is forward")

Route training's FORWARD through the inference operators (one impl, MoE+TP-capable, crash-fixed);
autograd owns ONLY the backward tape + LoRA grads (OPD/ROPD needs LoRA-adapter grads, frozen base).
This is the InferStudent shape already half-built — finishing the seam, not a redesign.

## Moves (highest-leverage first; verified file:line)

1. **`InferStudent.forward_token_logits` = the sole student forward in OPD/ROPD; delete the
   autograd dense student forward from the gradient path** (infer_student.rs:115 exists). One move:
   kills the SIGFPE class + unlocks MoE (35B student) + TP/FA3/FlashMLA. Independently shippable.
2. **Shrink autograd's CUDA surface to backward + LoRA-grad + optimizer**; quarantine the dead
   base-forward kernels behind a "training-only bring-up" feature so they can't re-diverge.
3. **Add a LoRA-grad activation-capture seam on `LoadedInferenceEngine`**
   (`forward_with_lora_activations`) so autograd backward attaches to the inference forward's
   activations — the only net-new piece; spec to implementation level first.

**Strategic:** the N-pad fix only unblocks the 4B bring-up (use it to get the OPD recipe verdict
now), but move #1 is the real architecture — it makes the band-aid moot AND opens the 35B student
(the autograd-can't-train-35B wall is bypassed: 35B forward runs on the inference engine). Turns
#102 Path B from "un-derisked" into "confirmed correct + half-built".

Refs: `infer_student.rs:115`, `backend_cuda.rs` SIGFPE workaround :50/:304/:540, `qwen35_loader.rs:322`
(`num_experts:0`), `train/src/qwen35.rs` (dense forward to retire), `autograd/src/backend_cuda/kernels/`
(28-file `.cu` to scope down), `infer-api/src/serve_engine.rs` (`forward_token_logits`).
