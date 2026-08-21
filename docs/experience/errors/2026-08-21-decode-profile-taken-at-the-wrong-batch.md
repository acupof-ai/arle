# The decode profile was taken at the wrong batch, and it inverted — CUDA, 2026-08-21

> Status: Measured. Reorders
> [the NVFP4 decode backlog](../../plans/2026-08-21-nvfp4-decode-lever-backlog.md)
> from the top down.

## Context

Every lever in that backlog was ranked off one `nsys` profile that read **Marlin
68.3%, `paged_attention_quantized_fa3_partial` 1.6%**. The arithmetic did not
support it. The model has 16 full-attention layers of 64, `num_key_value_heads=4`,
`head_dim=256`, FP8 KV — **32 KB of KV per token of context**. At 32K that is
1.07 GB per sequence per decode step, so at c=32 the step reads 34.4 GB of KV
against 20.0 GB of weights. A 1.6% attention share cannot describe that.

## What the profile at serving concurrency says

`nsys stats --report cuda_gpu_kern_sum`, window opened mid-decode with
`nsys start`/`stop`. **Zero prefill and zero mixed steps in either window**; the
observed batch is exactly nominal and the step time reproduces the sweep:

| run | decode steps | tokens/step | ms/step | sweep ms/step |
|---|---:|---:|---:|---:|
| c=16 | 129 | 16.00 | 99.13 | 99.16 |
| c=32 | 84 | 32.00 | 188.23 | 187.36 |

| group | c=16 | c=32 |
|---|---:|---:|
| **paged attention** | **80.64%** (78.6 ms/step) | **82.72%** (153.9 ms/step) |
| Marlin, four instances | 13.90% (13.6) | 12.74% (23.8) |
| `gdr_decode_batch_kernel` | 3.46% (3.4) | 3.27% (6.1) |
| everything else | 1.42% | 0.93% |
| `rms_norm_batched_offset` | 0.58% | 0.34% |

At c=32 the groups sum to 186.2 ms against a measured 188.2 — the GPU is ~99%
busy inside a decode step. `cuStreamSynchronize` is 96.3% of CUDA API time at
c=16 and 98.0% at c=32, so there is no launch-overhead story at serving
concurrency either.

The split does not shade toward attention with batch — **it inverts**. Attention
is a per-layer kernel with exactly 16 launches per step, 4.90 ms each at c=16 and
9.61 ms at c=32: it scales with batch x context, i.e. with KV bytes, while
Marlin's weight read is batch-invariant. That is the whole 9.1x step-time growth
from c=1 to c=32.

## The number worth acting on

Per launch the kernel must move `batch * ctx * 2 KB`: 1.06 GB at c=16 in 4.897 ms
and 2.18 GB at c=32 in 9.61 ms.

**217 GB/s and 226 GB/s, against this card's 4.0 TB/s — 5.5% of peak.**

The dominant kernel is 18x off its bandwidth roofline. If it reached it, the
attention share of a c=16 step would fall from 78.6 ms to 4.25 ms and the step
from 99 ms to about 25 ms. No other entry in the backlog is within an order of
magnitude of that.

Two candidate causes, being separated by `ncu` rather than by argument:

- **GQA read amplification.** The grid is `(num_q_heads * num_splits, batch)` and
  each CTA takes one q-head (`kv_head = q_head / gqa_ratio`). At 24 q-heads over
  4 kv-heads, **six CTAs read the same KV bytes**. Whether that costs 6x DRAM or
  is absorbed by L2 is a measurement, not a deduction: H20 L2 is ~60 MB and one
  kv-head's KV at 32K is ~16.8 MB per batch row.
- **Access pattern.** The per-lane element map is `d = lane_id * EPT + i` with
  `EPT = 8`, so lanes sit 8 bytes apart and one `i` iteration spans 256 B while
  using 32. The loop is `#pragma unroll`, so L1 may serve the other seven eighths
  — or may not.

## Rule

This is the third time in one day that a profile taken at the wrong operating
point produced a confident wrong ranking, after `gdr_decode_batch_kernel` read at
B=2 under ncu replay and the Marlin-versus-collective comparison read at a single
M. The pattern is the same each time: **a kernel whose cost scales with a
parameter cannot be ranked at one value of that parameter.**

Attention scales with batch x context and the weight GEMM does not, so their
shares cross. Record the batch and the context the capture actually saw next to
every share, and treat a profile without them as unranked.
