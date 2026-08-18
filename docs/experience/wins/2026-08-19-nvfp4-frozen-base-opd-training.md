# NVFP4 as an OPD frozen base — CUDA, 2026-08-19

> Status: Shipped

## Goal

Train against a 4-bit checkpoint: load `unsloth/Qwen3.8-27B-NVFP4` as the frozen
base of an OPD run, with the trainable parameters in BF16 LoRA. The point is
VRAM — a 27B base is what forces the teacher, the student, activations, LoRA
grads and AdamW state to compete for one H20.

## What 4-bit means on this hardware

sm_90 has no FP4 tensor cores. Keeping the base at 4 bits is a memory decision,
not a compute one: every projection dequantizes to BF16 scratch and the GEMM
rides cuBLAS. Two designs were possible —

- dequantize once at upload and keep BF16: no new code, 54 GB resident base
- keep FP4 resident and dequantize per projection: 15.2 GB resident base

The first spends 38.8 GB to avoid this diff, which removes the reason for the
feature. The second is also what the existing FP8 lane already does
(`DeviceHandle::CudaFp8BlockScaled`), so the 4-bit path is its twin rather than
a second mechanism.

Gradients flow to the LoRA adapters only. The 4-bit base is read and never
written, so no quantized-update numerics are involved.

## Parameters

```bash
arle train opd --steps 1 \
  --student-model /data00/Qwen3.8-27B-NVFP4 \
  --teacher-model /data00/Qwen3.8-27B-NVFP4 \
  --lora-rank 8 --rollout-len 8
```

- GPU: 1×H20 (sm_90, 96 GB), TP=1, nothing else resident
- Runtime `d3ea2f267`

## Results

A step completes: `step 1/1 loss 12.422474 rollout_len 11`, on
`Qwen3.x (vocab=248320, hidden=5120, layers=64, full_attn_gated=true)`. Two
consecutive steps reproduce the same loss under a fixed rollout.

VRAM after both engines load, from the `opd-vram-plan` line:

| checkpoint | free after load | resident |
|---|---:|---:|
| Qwen3.6-27B-FP8 | 40891 MiB | 55.3 GiB |
| Qwen3.8-27B-NVFP4 | 55803 MiB | 40.7 GiB |

14.6 GiB less, 26% of the FP8 footprint.

**This is indicative, not a controlled A/B.** The two checkpoints are different
models — different layer counts and a different vocab — so the delta mixes the
quantization difference with the model difference. A controlled figure needs
one model exported in both formats.

The `student_engine` / `teacher_engine` numbers on that line are a planned
reservation (`rollout_mem_fraction=0.5` of free VRAM at plan time), not measured
weight bytes; the NVFP4 run reports a larger reservation only because more VRAM
was free when it planned. The post-load free column is the meaningful one.

## What was changed

- `crates/autograd/src/backend.rs` — `CudaFp4E2M1GroupStorage` +
  `DeviceHandle::CudaFp4E2M1Group`, mirroring the FP8 storage including its
  borrowed-view `Drop` for `--share-frozen-base`.
- `crates/autograd/src/backend_cuda/kernels/fp8_block_scaled.cu` —
  `fp4_e2m1_group_to_bf16`, the FP4 form of the kernel beside it.
- `crates/autograd/src/backend/cpu_math.rs` —
  `dequantize_fp4_e2m1_group_host`, the host twin.
- `crates/autograd/src/backend_cuda/{matmul,matmul_backward,embedding,handle}.rs`
  — dispatch, upload and readback arms at every site the FP8 lane occupies.
- `crates/train/src/qwen35_loader.rs` — `weight_packed` name resolution, the
  packed `[rows, cols/2]` shape, `group_size` derived from the scale shape, and
  `PlannedFrozenBase` collapsing what had been two parallel `Option` fields.

## Problems

Three defects surfaced, none of them in the new 4-bit code:

- The checkpoint's attention and linear-attn projections are **per-channel** FP8
  (`.weight_scale`, `[N, 1]`), not block-scaled (`.weight_scale_inv`). Serving
  has treated that as block-scaled with block `[1, K]` since `33f4863c7`; the
  trainer had never needed to.
- A missing sidecar raised `LoaderError::MissingTensor`, which the candidate
  loop reads as "this HF name does not exist, try the next one". The real cause
  was discarded and the weight itself was reported absent, for a tensor plainly
  present in the checkpoint.
- The layer named in that error changed between runs, which looked like
  nondeterminism in the new code. It was an inference server still holding
  52 GB on the same GPU: the OOM surfaced at whichever tensor exhausted the
  allocation. A control run against `Qwen3.6-27B-FP8` reported
  `CUDA_ERROR_OUT_OF_MEMORY` directly, which separated the two.

`compressed-tensors` also quantizes `lm_head`, which the FP8 checkpoints leave
BF16, so the frozen-base whitelist had never covered it.

## Learnings

An error variant that a caller uses for control flow must not be reused for a
different failure. `MissingTensor` meant "try the next candidate name" to the
loop and "the sidecar is absent" to the planner, and the second meaning was
silently consumed by the first.

When a failure's identity varies between runs, suspect the environment before
the diff. The varying layer number read as a bug in new code and was contention
for VRAM; one control run against a known-good checkpoint settled it.
