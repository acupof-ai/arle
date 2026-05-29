# OPD FP8 teacher to free VRAM — KILL (infer Qwen35 FP8 path is dequant-to-BF16-resident)

> Memory-fit investigation (not a guidellm sweep). Target metric: teacher
> engine resident VRAM on RTX 4070 Ti SUPER (16 GiB, sm_89). Goal: route the
> OPD teacher through ARLE's infer FP8 loader to keep weights FP8-resident
> (~4 GB) instead of BF16 (~8 GB), freeing ~4 GB so `--rollout-len 128` fits.
> **Verdict: KILL — the infer Qwen35 `weight_scale_inv` FP8 path dequantizes
> FP8→BF16 at load and uploads BF16 to VRAM, so resident footprint is
> BF16-sized regardless of the on-disk FP8 format. No VRAM is freed. The FP8
> saving is disk-only (5.5 GB file vs ~9.3 GB).**

## Setup

- Teacher (treatment): `lovedheart/Qwen3___5-4B-FP8` — block-FP8
  (`quant_method=fp8`, `weight_block_size=[128,128]`, `weight_scale_inv` side
  tensors; `modules_to_not_convert` keeps linear-attn + norms + embed in BF16).
  Disk: 5.65 GB (2 safetensors). arch `Qwen3_5ForConditionalGeneration`,
  vocab 248320.
- Teacher (baseline, prior wins): `Qwen3___5-4B` BF16, ~9.3 GB file.
- Student: `Qwen3___5-0___8B-Base` (BF16 hybrid linear-attn), LoRA r=16
  `attention-qv`, infer rollout default.
- Build env: `CUDA_HOME=/opt/cuda NVCC_CCBIN=g++-14
  INFER_TILELANG_PYTHON=.venv/bin/python TORCH_CUDA_ARCH_LIST=8.9
  ARLE_CUDA_DISABLE_FLASHMLA=1`, `--release`. GPU free before run: 14.6 GB.
- No code change required: the existing default infer-teacher path
  (`opd_step_cuda_infer_teacher_train` → `load_infer_engine(&teacher_model)` →
  `InferTeacher`) already routes `--teacher-model` through the infer Qwen35
  loader, which auto-detects FP8 via `QuantLoadConfig::from_model_path`. So the
  treatment is simply `--teacher-model <FP8 dir>`.

## Result — teacher resident VRAM (driver `mem_get_info`, MiB)

| Phase | used | Δ | Notes |
|---|---:|---:|---|
| backend init | 1302 | — | CUDA ctx + driver + msedge |
| + train student base (0.8B) | 2742 | +1440 | BF16 frozen base |
| + infer student engine (0.8B) | 4358 | +1616 | 2nd student copy |
| **+ teacher infer engine (4B FP8 ckpt)** | **12524** | **+8166** | **BF16-resident — identical to the BF16 teacher's +8166 in [rollout128 wins](2026-05-29-opd-rollout128-vram-fit.md)** |

The teacher delta is **+8166 MiB**, bit-identical to the BF16 teacher's
+8166 MiB recorded in the rollout-128 attribution doc. A genuinely FP8-resident
4B would be ~4 GB. The match is the decisive evidence: all 4B weights are
BF16-resident.

## Root cause (source + empirical)

`infer/src/weight_loader.rs:990-1021`: the `config.fp8_weight_scale_inv` branch
calls `dequantize_fp8_e4m3_weight_scale_inv_to_bf16_host(...)` then
`DeviceMatrix::from_host(ctx, &host, ...)`. `DeviceMatrix::from_host`
(`crates/cuda-kernels/src/tensor.rs`) stores **bf16** and yields
`WeightFormat::DenseBf16`. So FP8 checkpoints are de-quantized to BF16 at load;
the FP8 e4m3 bytes never reach VRAM. Only the DeepSeek-V4
`WeightFormat::Dsv4Fp8BlockScaled` path (`infer/src/ops/linear.rs`) is
FP8-native-resident, and it is not wired for the Qwen35 `weight_scale_inv`
layout.

## KL sanity / step time (FP8 teacher, rollout-8, 1 step)

- `loss=7.306107727345e-5` — finite, ~1e-4 order (sane KL target; the
  FP8→BF16-dequantized teacher is a valid distillation target).
- `teacher_seq_len=8 teacher_vocab=248320` — correct full-vocab logits.
- step_seconds=2.92 (rollout-8 smoke), teacher_forward_total=21 ms.
- Teacher load 53.5 s (host FP8 dequant of all MLP/full-attn linear weights).
- FP8 loader handles this hybrid checkpoint cleanly — no error on the
  linear-attn / `modules_to_not_convert` / `weight_scale_inv` layout.

## rollout-128 verdict

Not attempted at 128: the teacher resident floor is unchanged (12524 MiB), so
the rollout-128 backward transient (~4.85 GB est. per the prior attribution)
still overflows 16 GB exactly as before. Routing the teacher through the FP8
loader cannot unblock rollout-128.

## Fastest alternative (proposed, not landed)

The licensed heavy change from the prior attribution stands: **gradient /
activation checkpointing on the student's 24 layers** (the per-token activation
tape, ~30 MiB/token, is the confirmed binding constraint — not the teacher
weights). To actually free teacher VRAM via FP8 would require a **new
FP8-native-resident Qwen35 weight path** (mirror `Dsv4Fp8BlockScaled`: keep
e4m3 bytes + `weight_scale_inv` on device, add a block-scaled GEMV/GEMM kernel
for the Qwen35 full-attn + MLP linears) — a CUDA-kernel project, not a wiring
flag.

## Rule

- "FP8 checkpoint → less VRAM" is **false for any loader that dequantizes to
  BF16 at load**. Resident VRAM is set by the in-VRAM dtype/format, never the
  on-disk format. Verify resident footprint with driver `mem_get_info`, not
  disk size, before claiming a memory win.
- ARLE's infer Qwen35 FP8 (`weight_scale_inv`) path is dequant-to-BF16-resident;
  only DeepSeek-V4 `Dsv4Fp8BlockScaled` is FP8-native-resident. Don't assume
  "infer supports FP8" means "FP8-resident".
