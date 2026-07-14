# DSv4 c=1 host/GPU timestamp model — 79% of launch time overlaps GPU

## Goal

**Diagnosis.** Prove whether host CUDA launches serialize DSv4 decode and establish a timestamp-based token/layer critical-path model.

## Hypothesis

Most launch API time overlaps queued GPU work, so CUDA API share overstates non-overlapped wall time.

## Command

```bash
nsys export --type sqlite --output /tmp/kern141.sqlite \
  /host/kern141_decode2.nsys-rep
```

The analysis joined `CUPTI_ACTIVITY_KIND_RUNTIME` to
`CUPTI_ACTIVITY_KIND_KERNEL` by process and `correlationId`. Interval unions,
not summed event durations, measured CPU/GPU overlap. Rank-0
`embedding_batched_native_kernel` timestamps delimited 689 steady decode steps
from 5–25 seconds.

## Environment

- Report: `/host/kern141_decode2.nsys-rep`, 1.1 GB, captured 2026-07-03
- Backend/model: CUDA, DeepSeek-V4-Flash FP8, no speculative decoding
- Hardware: 4x H20, TP=4, GPUs 0–3
- Workload: concurrency 1, about 512 input + 256 output tokens
- Capture: CUDA events, 30-second steady window
- Binary: `/host/arle-kern141-bin`; this predates the 2026-07-14 DSpark concurrency binary

## Results

### Host/GPU overlap

Rank 0, 20-second steady window:

| Metric | Measured |
|---|---:|
| GPU kernel busy union | 14,249.40 ms |
| Host launch API union | 1,041.39 ms |
| Launch/GPU intersection | 818.21 ms |
| Launch time covered by GPU work | 78.57% |
| Launch outside GPU busy intervals | 223.19 ms / 20 s = 1.12% |

Per-token medians:

| Metric | p50 |
|---|---:|
| Token wall | 26.263 ms |
| GPU busy union | 18.660 ms |
| GPU idle | 7.589 ms |
| Host launch API | 1.434 ms |
| Launch/GPU intersection | 1.121 ms |
| Launch outside GPU busy | **0.313 ms = 1.19% token wall** |

The correlation join matched 1,771,123 launches. `kernel_start - launch_end`
was positive for 99.9967%; queue delay was 2.112 / 4.051 / 7.512 ms at
p50/p90/p99. Median launch API and kernel execution were 3.455 and 4.032 us.
The host therefore queued work ahead of GPU execution. This proves temporal
overlap, not whether an overlapped launch gates a dependent kernel.

### Token/layer sequence

Counts per token recover the 43-layer loop:

| Anchor | Count/token |
|---|---:|
| embedding / argmax | 1.01 / 1.01 |
| mHC params / all-reduce | 86.93 / 86.94 |
| all-gather | 43.45 |
| FlashMLA split/combine | 43.09 |
| grouped SwiGLU/down | 43.28 |

GPU timestamps establish the repeated dependency order:

```text
embedding
  -> 43 x (mHC-attention -> all-gather -> FlashMLA -> attention all-reduce
           -> mHC-FFN -> grouped SwiGLU/down -> MoE all-reduce)
  -> lm_head -> argmax -> host/tick boundary
```

Kernel-sum medians were about 9.25 ms for GEMV/DeepGEMM plus quantization,
3.76 ms for MoE routing/grouped compute, 2.32 ms mHC, 2.35 ms collectives,
and 2.1–2.3 ms attention/CSA/KV. These are work sums, not additive critical
wall; stream overlap makes their sum exceed the 18.66 ms GPU busy union.

Cross-rank collective start spread was 11.7/53.0/79.8 us for all-reduce and
5.8/8.6/30.0 us for all-gather at p50/p90/p99. Maxima were 1.61 and 3.13 ms.

## Problems

- The capture has CUDA timestamps but no NVTX. Generic `sm90_fp8_gemm` kernels
  cannot be assigned exactly to attention projection versus shared expert.
- This is c=1 target decode, not the current c=8/c=16 layer-major batch path.
- Nsight observer overhead is present; matched wall-clock A/B remains the speed license.

## Learnings

- CUDA API share is not non-overlapped wall. Use interval intersection and
  report both; here 39.8% API share became 1.19% outside GPU-busy intervals.
- Overlap is not a causal speedup ceiling. An overlapped launch may still gate
  its dependent kernel; license graph/preallocation with an A/B.
- `correlationId` queue delay proves whether the host feeds ahead of the GPU.
- Model repeated layers from GPU timestamps and collective ordinals; host NVTX
  enqueue ranges cannot prove cross-rank GPU order.
- Add NVTX stage/layer/batch metadata before attributing generic GEMMs.

## Delta vs baseline

First timestamp-model record. The existing whole-step graph A/B moved c=1
throughput by -1.5%. That A/B, not the 1.19% overlap metric, is the speed
verdict for this historical c=1 binary.
