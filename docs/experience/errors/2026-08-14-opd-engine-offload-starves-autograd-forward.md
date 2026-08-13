# 2026-08-14 — `--engine-offload student` starves the autograd forward into UAF

## Context

First runtime exercise of single-GPU `train opd` with the 0.8B GDN student
(`qwen35-08b-clean`, H=16/Hg=16, 18 linear-attention + 6 full-attention layers).
Step 1 failed three different ways depending on sequence length and memory
state, all on the same config:

- seq 2171: `global gradient norm became non-finite (NaN)`, loss finite
- seq 635 / 340: `cuda synchronize failed` in the student windowed-KL forward
- compute-sanitizer: `Warp illegal address` inside `cublasGemmEx`, stack
  `LinearWithLora::forward → matmul_bt_device_f32_bf16` at GDN layer 6

Signature that unlocked it: every KL window reported the identical loss
increment (0.19410133362 × 65 windows = ln 248320 = ln vocab), i.e. the student
logits were a constant vector — the hidden state was zeros at the scored
positions and NaN elsewhere. Layer trace showed layers 0–8 healthy, NaN from
layer 9 on (long seq); short seq faulted at layer 6.

## Root cause

`--engine-offload student` is the trigger; `--engine-offload off` runs the
identical config clean (all 24 layers finite, per-window loss varies, step
completes, loss 22.56).

Mechanism: in offload mode the free-VRAM trace collapses to **2.8 GB** right
before the student autograd forward (vs **53 GB** with offload off). The
offload→reload cycle re-profiles the engine KV pool at momentarily-high free
(`profile_kv_pool_tokens` = free − total×(1−frac)), ratcheting pool size up
each cycle until the co-resident autograd allocator is starved; under that
pressure the forward's cuBLAS matmul reads freed/unmapped memory — garbage
(→ NaN) when the pages were reused, illegal address when unmapped.

Ruled out along the way, with evidence:

- Not the 08-13 refactors: NaN reproduces on `1c5847839`; the v0.5.5 binary
  cannot run this model at all (H=16/Hg=16 AOT geometry only added 08-13), so
  this path had never executed anywhere.
- Not the flashqla `fq_bwd` kernel: `--gdr-chunkwise-prefill false` still NaNs.
- Not the bf16 teacher-logits bridge: reproduces before and after `196eb2bb1`
  + `4f37b60ff`.
- Not FP8 teacher numerics: bf16 0.8B teacher reproduces.
- Not zero-copy base sharing: plain `train opd` uploads an owned bf16 base
  (`upload_bf16_bits`); the FP8 `--share-frozen-base` table is agent/rubric-OPD
  only.

Related fix landed en route: the teacher engine ignored
`--rollout-mem-fraction` (bare `EngineLoadConfig::single_sequence` left it at
the 0.9 default, 62 GB pool on one GPU) — `f1f568d1a`.

## Fix

Interim: run single-GPU OPD with `--engine-offload off` (verified clean).
Open: pool re-profile on reload must be capped by the original grant (or the
offload bracket must release + re-acquire at a fixed token count), and an
autograd allocation failure under pressure must fail the step instead of
letting a kernel read stale memory.

## Rule

A co-resident engine's KV pool must never be re-profiled from instantaneous
free VRAM while another allocator on the same device holds live tensors —
profile once at startup and carry the token count through offload/reload.
Diagnostics that made this tractable: per-param grad-norm dump on non-finite
global norm, per-layer hidden sum-squares (`ARLE_OPD_LAYER_TRACE`), and the
constant-per-window loss = ln(vocab) signature for "uniform logits".
