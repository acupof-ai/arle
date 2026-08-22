# W4A16 paired GEMV fuses clamped SwiGLU — CUDA, 2026-08-22

> Status: Shipped (perf-neutral at c=1)

## Goal

Decode throughput for DSv4-Flash-0731 (NVFP4→W4AFP8) at c=1 on H20, TP=4.
The W4A16/W4AFP8 MoE decode path ran a paired gate+up GEMV then a separate
clamped SwiGLU kernel, round-tripping `gate_out`/`up_out` through VRAM.

## Hypothesis

Fusing the SwiGLU into the GEMV write site saves one kernel launch and two
VRAM round-trips (write gate_out + up_out, read both back) per MoE layer.
For 42 MoE layers at c=1 decode, that is ~42 fewer launches per step.
Expected delta: +2-4% at c=1 (launch-bound), wash at c>1 (compute-bound).

## Parameters

```bash
python3 scripts/bench_throughput.py \
  --url http://localhost:8000 \
  --concurrency-grid 1 \
  --requests-per-concurrency 16 \
  --max-tokens 128 \
  --synthetic-prompts 8
```

- Baseline: `993c3e49b` (pre-fusion: paired GEMV + separate SwiGLU)
- Treatment: `8804072ef` (fused GEMV+SwiGLU)
- Prompt tokens: 8 (synthetic)
- Completion tokens: 128
- Trials: 16

## Environment

- Host / GPU: H20 96GB ×8
- Driver / CUDA: sm_90, CUDA 12.x
- Model / dtype: DeepSeek-V4-Flash-0731, NVFP4→W4AFP8 (INT4+BF16)
- TP=4: `--tensor-parallel-size 4 --max-running-requests 16 --max-total-tokens 131072`

## Implementation

`w4a16_grouped_gemv_pair_batch_kernel` gains `fuse_swiglu` + `swiglu_limit`
params. When fused, the kernel accumulates gate and up in float, applies
clamped SwiGLU (`gate=min(gate,limit); up=clamp(up,±limit); out=silu(gate)*up`)
at the write site, and writes only `act` — no `gate_out`/`up_out` buffers,
no separate SwiGLU launch.

A `fuse_swiglu` param keeps the Qwen path on the separate unclamped
`silu_mul` (different semantics — Qwen does not clamp). The now-dead
`dsv4_swiglu_clamped_batch` Rust wrapper is removed; the FFI + kernel stay
(HIP backend uses them).

Numeric note: the fused path keeps gate/up in float through SwiGLU; the
separate path rounded to BF16 first. The fused path is more accurate (no
BF16 round-trip on the intermediates) but not bit-identical.

## Results

| concurrency | arm | decode tok/s | delta |
|---:|---|---:|---:|
| 1 | baseline (2026-08-21) | 41.1 | — |
| 1 | treatment (`8804072ef`) | 41.2 | +0.2% (wash) |

Coherence: PASS (17x23=391, correct reasoning). Lever gate: PASS
(`correctness PASS: summaries=5`, exit 0). Needle scores all-miss on the
seed-baseline pass (no pre-fusion baseline envelope to compare against).

## Problems

1. **`--spec-type none` required for 0731 checkpoint** — the checkpoint's MTP
   head uses `main_norm` layout (no `enorm`/`hnorm`/`eh_proj` tensors), but the
   loader's strict MTP gating (2026-08-20) expects DSv3-style names. Any serve
   of this checkpoint needs `--spec-type none`. Loader name mapping is a
   separate fix.
2. **Transient OOM during gate load** — ranks 0/1 hit OOM late in weight
   upload while ranks 2/3 loaded clean. Retry loaded all 4 ranks at 42 GB/GPU.
   Foreign memory on the shared box during the load window, not a build defect.

## Learnings

Wash at c=1. The fused SwiGLU removes 42 kernel launches per decode step
(one per MoE layer) plus the gate_out/up_out VRAM round-trips, but the SwiGLU
kernel is a simple elementwise op — 42 launches is ~0.4 ms on a 24 ms step
(~1.7%), inside bench noise. The fusion is a structural improvement (fewer
kernels, less VRAM traffic, dead wrapper removed) with no measurable c=1
delta. The c=1 W4AFP8 GEMV decode floor is not launch-bound on the SwiGLU
elementwise; weight-loading and GEMV compute dominate.
