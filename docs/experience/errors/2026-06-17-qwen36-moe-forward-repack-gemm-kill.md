# Qwen3.6 MoE forward f32 repack grouped-GEMM killed

## Context

Path A's current Qwen3.6 FP8 LoRA wall is no longer rollout generation; OPD
rollout-256 is on infer-core KV decode. The remaining train-side path still has
host-heavy sparse MoE autograd work. A tempting narrow tranche was to replace
`moe_grouped_linear`'s active-expert scalar row loop with the existing batched
grouped matmul helper.

## Evidence

Local CPU finite diff first caught that using the batched helper on CPU changed
the oracle precision: the helper's f32 matmul accumulation produced
`max_rel=1.318222e-2`, above the 1e-2 threshold, while the old scalar f64-dot
path is `max_rel=3.407372e-3`.

After gating the batched path to CUDA/device only, the real checkpoint gate on
`.62` GPU0 was correct but slower:

```text
model=/data01/models/Qwen3.6-35B-A3B-FP8
mode=mlp-layer layer=0 target=auto:routed-up eps=1e-3
qwen36_fp8_lora_fd_gate PASS
analytic_seconds=0.892429 plus_seconds=0.641784 minus_seconds=0.657702
backward_profile total_seconds=0.251270 MoeGroupedLinear=0.198370
rel_err=9.863e-5
```

The A14 baseline on the same gate was:

```text
analytic_seconds=0.361241 plus_seconds=0.093031 minus_seconds=0.092101
backward_profile total_seconds=0.266602 MoeGroupedLinear=0.212961
rel_err=3.170e-3
```

## Root Cause

The proposed forward path paid per-call f32 packing and H2D upload of active
expert weights before every grouped matmul. That dominates the small active
expert/token shape and erases the GEMM benefit. It also changes CPU reference
numerics unless CPU stays on the old f64 scalar path.

This is not evidence against a resident grouped MoE CUDA path. It is evidence
against a non-resident "pack f32 weights every forward" implementation.

## Fix

Do not land the repack grouped-GEMM forward path. The local source was restored
to the A14 implementation; only this error entry records the killed attempt.

Next valid implementation route:

1. Keep FP8/BF16 expert weights resident and grouped, not repacked to f32 per
   forward call.
2. Fuse pack/gather of active token rows with the grouped kernel or cache the
   active expert weight layout across the step.
3. Re-run the same `mlp-layer` FD/profile gate before touching full-model OPD
   timing.

## Rule

For tiny active MoE train shapes, a grouped GEMM is only a win if weights are
resident in the grouped layout. Per-call f32 repack/upload is a measured
regression even when the finite-diff check passes.
