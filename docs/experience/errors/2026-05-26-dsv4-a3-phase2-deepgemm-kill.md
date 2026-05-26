# DSv4 A3 Phase 2 DeepGEMM required KILL

## Context

A3 Phase 2 next tried the native DeepGEMM grouped expert backend after the
route-grouped GEMV prototype was KILLed. The goal was to replace route-wise
local expert GEMV with a true grouped FP8 GEMM path for decode while preserving
greedy output.

Model path and host identifiers are intentionally omitted.

## Root Cause

Two toolchain/runtime bugs blocked first contact with real decode:

- The native JIT was using NVRTC-only compilation assumptions. The target
  DeepGEMM source expects external NVCC-style options, so the bridge now writes
  generated CUDA to disk and compiles cubins through `${CUDA_HOME}/bin/nvcc`.
- The in-process runtime cache keyed only on kernel code and arch. In the
  8-rank server, that reused a `CUfunction` loaded under one CUDA context on a
  different rank's stream, causing `CUDA_ERROR_INVALID_HANDLE`. The runtime
  cache now keys module/function handles by current CUDA context while keeping
  the cubin cache shared by code+arch.

After those were fixed, the next failure exposed an ARLE scratch-reuse mismatch:
DeepGEMM's upstream masked grouped path assumes per-call compact scale tensors,
where scale stride equals `align(m)`. ARLE reuses a larger per-layer scratch
capacity, so a later decode step can run with `m=3` and `scale_stride_m=8`.
The bridge now accepts `scale_stride_m >= align(m)` and uses the actual stride
in the SFA tensor map.

These fixes made the required DeepGEMM path run a real `max_tokens=32` decode,
but the optimization itself failed the license gate.

## Fix

- Build-time native bridge no longer links or calls NVRTC.
- Runtime JIT uses NVCC and `cuobjdump`, matching the DeepGEMM source layout.
- Loaded DeepGEMM modules/functions are cached per CUDA context.
- Required DeepGEMM backend is decode-only unless MoE runtime scratch exists;
  multi-token prefill stays on the native path instead of failing before decode.
- `scripts/dsv4_toolchain.sh` now provides `env-check`, `build`, `smoke`, and
  `nsys` entrypoints with CUDA/NCCL/DeepGEMM/model validation and
  `max_tokens>=32` decode guardrails.

## Results

### Bring-up Gates

| Gate | Result |
|---|---|
| `env-check` | PASS |
| `cargo build --release -p infer --features cuda,nccl --bin infer` via helper | PASS |
| DeepGEMM required smoke, `max_tokens=32` | PASS, 32 completion tokens |

### A/B Wall-Clock

One warmup, then 3 measured non-streaming requests, `temperature=0`,
`ignore_eos=true`, `max_tokens=32`.

| Backend | Measured seconds | Mean |
|---|---:|---:|
| `native` | 3.7817, 3.7539, 3.7541 | 3.7632 s |
| `deepgemm` | 7.2149, 8.0272, 7.3619 | 7.5347 s |

DeepGEMM is +100.2% slower than native on this decode smoke.

### Correctness

The first measured native and DeepGEMM outputs were not byte-identical. Native
was stable across all three measured requests; DeepGEMM produced three different
strings across the three measured requests.

## Artifacts

- Toolchain/context-cache build:
  `dsv4-toolchain-context-cache-build-20260526-062011`
- Context-cache smoke before SFA stride fix:
  `dsv4-toolchain-deepgemm-smoke-max32-20260526-062111`
- Shape-detail smoke:
  `dsv4-deepgemm-shape-detail-smoke-20260526-063124`
- SFA stride build:
  `dsv4-deepgemm-sfa-stride-build-20260526-063442`
- Required DeepGEMM smoke after SFA stride fix:
  `dsv4-deepgemm-sfa-stride-smoke-max32-20260526-064221`
- Native vs DeepGEMM A/B:
  `dsv4-deepgemm-native-ab-max32-20260526-064326`

## Rule

Keep `ARLE_DSV4_EXPERT_BACKEND=deepgemm` required mode for diagnosis only and
do not make it the default. A3 Phase 2 still needs a different grouped-GEMM
route: byte-identical output first, then wall-clock win on the DSv4 SLO
framing. Toolchain smoke is not performance evidence unless it uses
`max_tokens>=32`.
