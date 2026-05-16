# P3.6 DSv4 FP8 batch tiled ncu profiling permission blocker

## Context

Phase 1 for the P3.6 follow-up was scoped to profile
`dsv4_fp8_gemv_batch_tiled_kernel` with Nsight Compute before choosing any
further optimization axis. The required deliverables were a raw `.ncu-rep`, a
binding-constraint table, and a research note at
`docs/research/2026-05-17-p3-6-ncu-binding.md`.

No kernel or runtime code was changed.

## Root Cause

The local NVIDIA driver blocks non-admin access to GPU performance counters.
This prevents Nsight Compute from collecting the runtime counters required for
the binding verdict.

The static fallback is insufficient: `cuobjdump --dump-resource-usage` reports
`REG:64` and `SHARED:2560` for
`dsv4_fp8_gemv_batch_tiled_kernel`, but that does not provide HBM throughput,
cache hit rates, SM busy, achieved occupancy, or warp stall breakdown. A
binding verdict from static resource usage would be a hypothesis, not SOLID
evidence.

## Evidence

Release bench build succeeded:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo build --release --features cuda -p infer --bench ops_bench
```

Full ncu attempt:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
ncu --set full \
  --launch-skip 100 \
  --launch-count 20 \
  --kernel-name regex:dsv4_fp8_gemv_batch_tiled_kernel \
  --target-processes all \
  --export /tmp/p3_6_ncu_baseline \
  --csv \
  --print-summary per-gpu \
  cargo bench -p infer --bench ops_bench --features cuda -- \
    --profile-time 5 \
    'ops_cuda/dsv4_fp8_gemv_batch/'
```

Reduced LaunchStats probe:

```bash
CUDARC_CUDA_VERSION=13010 \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
ncu --section LaunchStats \
  --launch-skip 100 \
  --launch-count 1 \
  --kernel-name regex:dsv4_fp8_gemv_batch_tiled_kernel \
  --target-processes all \
  --export /tmp/p3_6_ncu_launchstats_probe \
  --csv \
  --print-summary per-gpu \
  cargo bench -p infer --bench ops_bench --features cuda -- \
    --profile-time 1 \
    'ops_cuda/dsv4_fp8_gemv_batch/'
```

Both ncu attempts reached the benchmark process and failed with:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
Performance Counters on the target device 0.
```

Driver and sudo checks:

```text
/proc/driver/nvidia/params: RmProfilingAdminOnly: 1
sudo -n true: a password is required
```

No report artifact was produced:

```bash
find /tmp docs/trace-artifacts -maxdepth 4 -type f \
  \( -name '*p3_6*ncu*' -o -name '*.ncu-rep' \)
```

## Fix

No local code fix exists. Enable NVIDIA profiling counters for this user or
run the exact ncu command with admin permissions, then resume Phase 1.

Example root-side recovery path:

```bash
sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
```

Alternatively, place a valid report at `/tmp/p3_6_ncu_baseline.ncu-rep` and
resume with metric extraction.

## Rule

Do not start Phase 2 P3.6 optimization axes until the required ncu runtime
counter evidence exists. Static resource usage and prior nsys traces are not
sufficient to choose `MEM_BANDWIDTH_BOUND`, `MEM_LATENCY_BOUND`,
`COMPUTE_BOUND_DECODE`, `OCCUPANCY_BOUND`, or `DIVERGENCE_BOUND`.
