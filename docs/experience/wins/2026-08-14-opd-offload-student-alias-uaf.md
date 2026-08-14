# OPD offload=student frozen-base alias UAF fix — CUDA, 2026-08-14

> Status: Shipped

## Goal

`train opd --engine-offload student` completes 3 steps with finite losses on
the H20 pod (0.8B GDN student `qwen35-08b-clean`, teacher
`qwen35-08b-w8a16b-bf16`). Pre-fix behavior: step-1 backward aborts with
"global gradient norm became non-finite (NaN)" with 72 GB free.

## Hypothesis

Use-after-free of non-owning frozen-base aliases. `sync_lora_from_store`
re-pointed the autograd student's frozen-base tensors at the engine's merged
BF16 buffers (`replace_device_handle` dropped the trainer's own copies); the
same step's `offload_engine_weights` then freed those buffers. The student
forward read freed device memory: nondeterministic partial NaN in normal runs,
illegal-address READ in `cublasGemmEx` (LinearWithLora::forward) under
compute-sanitizer. Fix: skip the re-point when the mode offloads the student
(commit `a1a3fda92`), plus a fail-fast `ensure!` in `offload_engine_weights`
when pointers are exported, load-time bail for share-frozen-base + student
offload, and deletion of the detached KV-pool trim thread (the one free no
foreground fence ordered).

## Parameters

```bash
train opd --backend auto \
  --teacher-model /host/nvme0/qwen35-08b-w8a16b-bf16 --teacher-runtime infer \
  --student-model /host/nvme0/qwen35-08b-clean \
  --steps 3 --rollout-len 2048 --rollout-temperature 1.0 --rollout-top-p 1.0 \
  --rollout-top-k 0 --rollout-seed 0 --kl-direction forward --kl-temperature 1.0 \
  --kl-mask completion --gkd-lambda 0.0 --lr 2e-5 --lr-schedule cosine \
  --lr-warmup-steps 1 --grad-clip 1.0 --lora-rank 16 --lora-alpha 32 \
  --lora-target-set attention-qv --prompts-file examples/opd/gsm8k-train.jsonl \
  --prompt-max-tokens 2048 --prompt-seed 0 --gate-every-n 0 \
  --engine-offload student --rollout-mem-fraction 0.25
```

- Baseline: pre-fix tree (fail side established on the pod 2026-08-13: step-1
  NaN, sanitizer illegal-address READ in cublasGemmEx)
- Treatment: `4b8b02f9f` (includes `a1a3fda92` fix, `ef486bd86` merge
  idempotency, `4b8b02f9f` VRAM plan)
- Trials: 1 repro run + `--engine-offload off` control

## Environment

- Host / GPU: H20 pod, CUDA_VISIBLE_DEVICES=0
- Model / dtype: 0.8B GDN student (bf16), W8A16 teacher
- Diagnostics: `ARLE_OPD_STEP_TRACE=1`

## Results

| arm | steps | losses | NaN/illegal lines | outcome |
|---|---:|---|---|---|
| pre-fix, offload=student | 0 (NaN at step 1) | non-finite | yes (sanitizer: illegal READ) | established 2026-08-13 |
| post-fix, offload=student | 3/3 | 26.849 / 27.136 / 14.958 | 0 | PASS |
| post-fix, offload=off control | 3/3 | 25.917 / 27.033 / 14.919 | 0 | PASS |
| post-fix, offload=student, compute-sanitizer memcheck (1 step) | 1/1 | 25.277 | ERROR SUMMARY: 0 errors | PASS |

`[probe-forward]` hidden_sum_sq at after_autograd_load / after_engine_load /
after_init_offload: `2263590.243214886` — byte-identical to the established
pre-fix probes.
`[opd-vram-plan]` (H20, 95387 MiB free): student_engine 23846 MiB,
teacher_engine 23846 MiB, autograd_reserve 47693 MiB.

Raw artifacts: pod logs `/host/wf-uaf1-repro.log`, `/host/wf-uaf1-control.log`,
`/host/wf-uaf1-sanitizer.log`; build receipt `build:wf-uaf1` (BUILD_EXIT=0 at
`4b8b02f9f`).

## Problems

The offload=off regime carries a separate residual defect (trainer aliases
MERGED base+delta while LinearWithLora re-adds A·B from step 2) — filed as
#201, not fixed here, so the off-control gate is "3 finite steps", not loss
parity with pre-fix curves. `ef486bd86` additionally changes offload=off loss
curves from step 2 (pristine-base restore); pre/post curves are not
comparable.

## Learnings

Pending-remote. Mechanism class: three allocators on one GPU with no memory
contract; the write-only `frozen_base_ptrs_exported` flag is now load-bearing
in both directions (skip the alias when offloading, refuse the offload when
aliased). Startup-fixed VRAM grants (`OpdVramPlan` →
`EngineLoadConfig.memory_budget_bytes`) delete the last
instantaneous-free-VRAM decision from the OPD path.
