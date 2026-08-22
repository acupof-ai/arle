# W4A16 paired GEMV fuses clamped SwiGLU — CUDA, 2026-08-22

> Status: pending-remote

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
| 1 | baseline | | — |
| 1 | treatment | | pending-remote |

## Problems

None yet.

## Learnings

pending-remote. The fusion is the same optimization the FP8 decode path
already ships (`dsv4_fp8_grouped_swiglu_decode`); this brings the W4A16/
W4AFP8 4-bit path to parity. Correctness gate (needle/lever ×3) and the
pod bench are the pending-remote work.
