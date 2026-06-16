# Qwen3.6 FP8 Decode Fused Root Cause

## Goal

Root-cause the Qwen3.6 FP8 decode regression observed after the prefill warmup
fix: FP8 4K/256 c=1 had ITL 71.4 ms vs BF16 24 ms and 12.83 tok/s vs 32.61
tok/s. The required gate here was not another long serve sweep; it was a cheap
isolated decode probe that breaks down the FP8 decode wall.

## Hypothesis

The Qwen FP8 routed-MoE lane was not using the decode-fused small-kernel path.
`moe.rs` gated decode kernels behind `!routed_quant`, so FP8 fell through to
the generic grouped FP8 GEMV batch path:

- gate/up pair batch GEMV
- separate `silu_mul`
- down batch GEMV
- scatter/combine

That path launches across `num_experts * max_count`, which is pathological for
the real Qwen3.6 decode shape (`256 experts * topk 8`) even though only 8 routes
are live.

## Params

- Harness: `cargo build -p infer-cuda --release --features cuda --example fp8_decode_probe`
- Binary: `/data01/arle-qwenfp8-smoke/target/release/examples/fp8_decode_probe`
- GPU: H20, compute capability 9.0
- Shape: `experts=256`, `topk=8`, `hidden=2048`, `intermediate=512`, `iters=200`
- Controls:
  - `experts=8`, `topk=8`
  - `experts=256`, `topk=1`
- Scope: isolated kernels only. HTTP, scheduler, tokenizer, checkpoint loader,
  and sampling were intentionally bypassed.

## Results

Real Qwen decode shape (`experts=256`, `topk=8`):

| Path | CUDA ms/layer | Delta vs BF16 fused |
| --- | ---: | ---: |
| BF16 fused MoE decode | 0.0666 | baseline |
| FP8 legacy grouped batch | 1.1578 | +1639.3% |
| FP8 decode-fused | 0.0754 | +13.3% |

Legacy FP8 breakdown:

| Stage | CUDA ms/layer |
| --- | ---: |
| FP8 legacy gate/up pair batch | 0.2515 |
| FP8 legacy `silu_mul` | 0.0031 |
| FP8 legacy down batch | 0.8970 |
| Scatter/combine | 0.0062 |

Decode-fused FP8 breakdown:

| Stage | CUDA ms/layer |
| --- | ---: |
| FP8 decode-fused SwiGLU | 0.0320 |
| FP8 decode-fused down | 0.0376 |
| Scatter/combine | 0.0059 |

Controls:

| Shape | BF16 fused | FP8 legacy batch | FP8 decode-fused |
| --- | ---: | ---: | ---: |
| experts=8, topk=8 | 0.0312 | 0.1104 | 0.0350 |
| experts=256, topk=1 | 0.0521 | 0.1570 | 0.0567 |

The controls license the root cause: the bad path grows with the
`num_experts * max_count` launch grid, and the existing FP8 decode-fused kernel
removes that cost.

Dense per-layer B=1 probe was not the dominant wall:

| Dense path | CUDA ms |
| --- | ---: |
| BF16 GEMM batch B=1 | 0.0068 |
| FP8 block-scaled GEMV batch B=1 | 0.0169 |

## Fix

`crates/infer-cuda/src/moe.rs` now routes Qwen FP8 block-scaled decode through
the existing FP8 decode-fused ABI when all of these are true:

- `WeightFormat::Fp8BlockScaled`
- decode-band route count is within `QWEN35_MOE_DECODE_MAX_ROUTES`
- hidden and intermediate dimensions satisfy the fused-kernel alignment
- gate/up and down scale signatures are exactly 128x128 blocks
- scale rows and cols match the real Qwen tensor dimensions

The fused path uses the resident FP8 expert pointer tables and f32 block scales.
Qwen has no SwiGLU clamp, so the DSv4-origin wrapper is called with
`limit = f32::INFINITY`. Non-matching quant formats keep the generic fallback.
BF16 decode-fused behavior is unchanged.

## Problems

The post-fix 4K/256 e2e serve verification did not produce a decode verdict in
this pass. The FP8 service process on GPU2 did not bind HTTP within 6 minutes;
logs stopped after tokenizer initialization, while the process stayed at 100%
CPU and grew to roughly 37 GB RSS / 24.5 GB GPU memory. That is a checkpoint
loader/startup issue, not a decode measurement. The process was killed to avoid
burning GPU time.

Therefore this entry claims the isolated decode-kernel root cause and dispatch
fix only. It does not claim an end-to-end FP8 throughput/default win. The next
gate remains one 4K/256 FP8-vs-BF16 e2e run on a cleanly started server.

## Learnings

The FP8 decode regression was our dispatch misuse, not hardware and not the FP8
method. Qwen FP8 had inherited a correctness-oriented grouped GEMV batch path
for decode; it should have used the same compact decode-fused ABI already
available in the CUDA kernel layer.

The dominant cost was not dequant, scatter, or dense projection overhead. It was
the legacy FP8 grouped batch MoE down GEMV, followed by the paired gate/up GEMV.
