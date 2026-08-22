# W4AFP8 custom M=1 decode kernel — CUDA, 2026-08-22

> Status: Shipped (correct, perf-neutral at c=1)

## Goal

Decode throughput for DSv4-Flash-0731 (NVFP4→W4AFP8) at c=1 on H20, TP=4.
The W4AFP8 decode path used the generic W4A16 paired GEMV kernel (64 threads/
row, 2 warps, shared-mem reduction, half2 accumulation). The FP8 decode path
has a custom M=1 kernel (1 warp/row, shuffle-only, float accumulation, fused
gate+up+SwiGLU). Port the FP8 structure to W4AFP8 to close the gap.

## Implementation

`w4afp8_grouped_swiglu_decode_kernel` in `quantized_gemv.cu`:
- 1 warp per row (W4D_WARPS=8, W4D_THREADS=256), warp-shuffle reduce, no
  shared memory
- Fused gate+up+clamped-SwiGLU in one kernel (writes `act` directly)
- Float accumulation (matches FP8 decode kernel accuracy; old kernel used
  half2 FMA)
- W4AFP8 dequant: two's-complement nibble → (n-8) via the 0x6400 half trick,
  × BF16 per-128-group scale (identical to the existing W4A16 dequant)
- Grid: `(N/8, max_count/8, num_experts)` — compact, work scales with real
  routed rows

Dispatch: `dsv4_moe_forward_w4a16` gains `use_custom_decode: bool`; the W4AFP8
M=1 path passes `true`, the W4A16 path `false`. The w2 down projection stays
on the existing `moe_w4a16_grouped_gemv_batch` kernel.

## Parameters

```bash
python3 scripts/bench_throughput.py \
  --url http://localhost:8000 \
  --concurrency-grid 1 \
  --requests-per-concurrency 16 \
  --max-tokens 128 \
  --synthetic-prompts 8
```

- Baseline: `994163a98` (fused-SwiGLU W4A16 paired GEMV, 2026-08-22)
- Treatment: `w4afp8-decode8` (custom W4AFP8 decode kernel)
- Prompt tokens: 8 (synthetic)
- Completion tokens: 128
- Trials: 16

## Environment

- Host / GPU: H20 96GB ×8
- Driver / CUDA: sm_90, CUDA 12.x
- Model / dtype: DeepSeek-V4-Flash-0731, NVFP4→W4AFP8 (INT4+BF16)
- TP=4: `--tensor-parallel-size 4 --max-running-requests 16 --max-total-tokens 131072 --spec-type none`

## Results

| concurrency | arm | decode tok/s | delta |
|---:|---|---:|---:|
| 1 | baseline (2026-08-22) | 41.1 | — |
| 1 | treatment (custom decode) | 41.2 | +0.2% (wash) |

Coherence: PASS (17×23=391, correct reasoning). Lever gate: PASS
(`correctness PASS: summaries=5`, exit 0).

## Problems

1. **`--spec-type none` required for 0731 checkpoint** — the checkpoint's MTP
   head uses `main_norm` layout; the loader's strict MTP gating expects
   DSv3-style names. Separate fix needed.
2. **Pod sync digest non-deterministic on macOS** — `source_digest` returns
   different values across runs on the Mac (filesystem/Python behavior
   difference), so `pod.sh sync` always fails the digest guard. Workaround:
   `tn push` individual files + manual receipt update. Root cause TBD.
3. **`autograd` compile error at HEAD** — `DequantCacheEntry.bf16` field was
   `CudaSlice<u16>` but `cached_dequant` stores `Arc<CudaSlice<u16>>`. Hidden
   locally by `--features cuda,no-cuda` (the struct is `#[cfg(not(feature =
   "no-cuda"))]`); exposed on the pod's `--features cuda,nccl` build. Fixed
   by changing the field to `Arc<CudaSlice<u16>>`.

## Learnings

Wash at c=1. The custom kernel's structural improvements (warp-per-row, no
shared-mem reduction, float accumulation, fused SwiGLU) are real but do not
move c=1 throughput because the MoE GEMV is **memory-bound** at M=1: both
kernels load the same INT4 weight bytes (~36 MB/layer across gate+up+down),
and H20's bandwidth floors the load time regardless of the compute structure.
The c=1 W4AFP8 decode floor is set by weight loading + attention + launch
overhead, not by the GEMV compute path. The kernel is correct and more
accurate (float vs half2 accumulation), so it stays as the structural
baseline — but further c=1 gains need a different axis (weight compression,
attention, or launch reduction), not GEMV restructuring.
