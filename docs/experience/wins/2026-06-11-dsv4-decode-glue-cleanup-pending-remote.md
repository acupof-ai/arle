# DSv4 Decode Glue Cleanup Pending Remote

## Context

Goal: remove confirmed per-layer decode glue overhead from the DSv4 H20 B=1
path without changing model math.

Scope:
- CSA official selector fallback no longer allocates `selected` before the
  official path can return its own output.
- Official DSA decode fills `context_lens` and `positions` from device
  `start_pos`, removing two host-to-device copies per CSA tile on decode.
- B=1 shared expert reuses per-slot/per-layer output scratch, including the
  `ARLE_DSV4_COMM_OVERLAP=1` path, instead of allocating a fresh output per
  layer.

## Results

Local Mac verification only:

| Check | Result |
| --- | --- |
| `CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib` | PASS |

H20 runtime benchmark is pending because this host has no CUDA toolkit/GPU.

## Pending Remote

Run on the DSv4 H20 pod:

```bash
CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib
ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1 bash scripts/dsv4_fast_build.sh
```

Then collect same-binary A/B:

| Arm | Env |
| --- | --- |
| baseline | current default |
| overlap | `ARLE_DSV4_COMM_OVERLAP=1` |

Required checks:
- needle gate unchanged
- `dsv4/stage/shared_expert` allocation disappears from nsys allocation trace
- CSA official path has no pre-official `selected` allocation
- DSA decode tile no longer shows context/position H2D copies
- report tok/s and ms/token delta against latest DSv4 H20 baseline

## Rule

Small decode glue fixes can land behind unchanged math if the local typecheck
passes, but perf claims stay pending until a same-binary H20 A/B validates the
wall-clock effect.
