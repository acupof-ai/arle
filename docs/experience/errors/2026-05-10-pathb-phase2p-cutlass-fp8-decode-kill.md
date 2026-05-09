# Path B-Phase2' Cutlass FP8 Smoke Killed For Decode ITL

## Context

Path B-Phase2' tested whether an Ada-native FP8 GEMM path could support the
user's requested `-20%..-40%` decode ITL target before committing to the
larger W4+FP8 substrate.

The prior cuBLASLt smoke had already failed because heuristic dispatch only
reached about `1.8x`. This run intentionally did not reuse cuBLASLt. It used
CUTLASS directly with the Ada-specific FP8 template:

- `cutlass::gemm::device::GemmUniversalWithAbsMax`
- `cutlass::arch::Sm89`
- `cutlass::float_e4m3_t` inputs
- `cutlass::arch::OpMultiplyAdd` and `OpMultiplyAddFastAccum`
- `M=1,N=4096,K=2560` decode shape
- `M=2048,N=4096,K=2560` prefill shape

The smoke source stayed outside the repo at `/tmp/cutlass_fp8_smoke.cu`.

Command:

```bash
CUT=/home/ckl/.cache/uv/archive-v0/RgIy_TdYM0SbMj4Y/flashinfer/data/cutlass
/opt/cuda/bin/nvcc -arch=sm_89 -O3 -std=c++17 -ccbin /usr/bin/g++-14 \
  -I "$CUT/include" -I "$CUT/tools/util/include" \
  /tmp/cutlass_fp8_smoke.cu -o /tmp/cutlass_fp8_smoke -lcudart
/tmp/cutlass_fp8_smoke
```

Environment:

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| SM | `sm_89` |
| CUDA | 13.2 stack via `/opt/cuda` |
| CUTLASS | FlashInfer vendored CUTLASS headers |
| Iterations | 20 warmup + 100 timed |

## Results

Raw output:

```text
GPU: NVIDIA GeForce RTX 4070 Ti SUPER sm_89
BF16-CUTLASS can_implement=Success
BF16-CUTLASS workspace=0 bytes threads=128 smem=73728 bytes
BF16-CUTLASS M=1 N=4096 K=2560 mean=0.426588 ms std=0.001367 ms TFLOPS=0.05
FP8-CUTLASS-staged can_implement=Success
FP8-CUTLASS-staged workspace=0 bytes threads=128 smem=73728 bytes
FP8-CUTLASS-staged M=1 N=4096 K=2560 mean=0.245514 ms std=0.001255 ms TFLOPS=0.09
FP8-CUTLASS-fast can_implement=Success
FP8-CUTLASS-fast workspace=0 bytes threads=128 smem=73728 bytes
FP8-CUTLASS-fast M=1 N=4096 K=2560 mean=0.229284 ms std=0.001263 ms TFLOPS=0.09
FP8 staged speedup vs BF16 M=1 N=4096 K=2560: 1.74x
FP8 fast speedup vs BF16 M=1 N=4096 K=2560: 1.86x
BF16-CUTLASS can_implement=Success
BF16-CUTLASS workspace=0 bytes threads=128 smem=73728 bytes
BF16-CUTLASS M=2048 N=4096 K=2560 mean=1.399307 ms std=2.092525 ms TFLOPS=30.69
FP8-CUTLASS-staged can_implement=Success
FP8-CUTLASS-staged workspace=0 bytes threads=128 smem=73728 bytes
FP8-CUTLASS-staged M=2048 N=4096 K=2560 mean=0.311385 ms std=0.000485 ms TFLOPS=137.93
FP8-CUTLASS-fast can_implement=Success
FP8-CUTLASS-fast workspace=0 bytes threads=128 smem=73728 bytes
FP8-CUTLASS-fast M=2048 N=4096 K=2560 mean=0.268771 ms std=0.000746 ms TFLOPS=159.80
FP8 staged speedup vs BF16 M=2048 N=4096 K=2560: 4.49x
FP8 fast speedup vs BF16 M=2048 N=4096 K=2560: 5.21x
```

Summary:

| Shape | BF16 mean | Best FP8 mean | Best speedup | Decision |
|---|---:|---:|---:|---|
| Decode `M=1,N=4096,K=2560` | 0.426588 ms | 0.229284 ms | 1.86x | Kill for ITL target |
| Prefill `M=2048,N=4096,K=2560` | 1.399307 ms | 0.268771 ms | 5.21x | Separate prefill signal only |

The prefill result proves the Ada FP8 path can produce a real CUTLASS speedup
when the GEMM is large enough. It does not license the decode ITL target:

- The requested Path B-Phase2' mechanism was decode ITL reduction.
- The decode shape is only `1.86x`, which falls in the `<=2x` kill bucket from
  the phase brief.
- The absolute prefill FP8 throughput is `159.80 TFLOPS`, still below the
  `>=350 TFLOPS` target in the phase brief.
- The BF16 prefill control has high variance, so the prefill speedup should be
  treated as a separate lead, not a production license for this path.

## Root Cause

Ada FP8 tensor cores are available and the CUTLASS Ada template works, but the
single-token decode GEMM shape is too small to expose the theoretical FP8
throughput. Launch and occupancy effects dominate `M=1`, so the FP8 path only
cuts per-call latency from `0.426588 ms` to `0.229284 ms`.

That is useful, but it is not enough to justify the 900-1700 LOC Phase 2'
runtime substrate for the stated `-20%..-40%` ITL objective.

## Fix

Do not start Path B-Phase2' W4+FP8 decode substrate from this smoke.

Per `docs/research/2026-05-10-p0b-ppl-eval-infra-inventory.md`, P0.B PPL is
skipped when P0.A lands in the `<=2x` kill bucket. Accuracy evidence would not
rescue a decode path that already misses the performance gate.

Recommended next options:

| Option | Rationale |
|---|---|
| Path B Phase 1 `dequant.h` port | Conservative, low-risk `-3%..-8%` ITL path. |
| Phase 2 multi-shape spec | If FP8 is revisited, benchmark a broader shape table before runtime work. |
| Separate FP8 prefill investigation | The `M=2048` signal is real but belongs to a prefill-compute axis, not this decode ITL axis. |

## Rule

For quantized GEMM changes, the license shape must match the claimed metric.
Large-batch prefill speedups do not license a decode-ITL optimization when the
single-token GEMM shape falls below the kill threshold.
