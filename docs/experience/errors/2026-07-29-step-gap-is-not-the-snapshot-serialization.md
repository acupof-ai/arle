# The step-boundary gap is not the recurrent-snapshot serialization

## Context

At c=16 the GPU idles 40.7% (dense) / 18.9% (MoE) of the wall in 50 ms+ gaps
between `argmax_batch_kernel` and the next `embedding_batched_native_kernel`,
with only 6.9% of that time holding any CUDA API call
([entry](../wins/2026-07-29-decode-tail-is-host-side-step-boundary-stall.md)).
Pure host CPU work, per request finish, scaling with the model.

## Root Cause — the hypothesis, and why it is wrong

`Qwen35RecurrentSnapshot::to_bytes` serialized element-wise:

```rust
for &x in gdr { buf.extend_from_slice(&x.to_le_bytes()); }
```

A dense finish carries 48 linear layers × 48 value heads × 128 × 128 f32 =
151 MB = **37.7M** of those calls; MoE carries 15.7M. Four things lined up:

- the dense/MoE element ratio is **2.4×** against a measured 2.7× on the
  largest gaps;
- ~47 gaps against ~64 request finishes — about one each;
- 93% of the gap time has no CUDA activity, and this is pure CPU;
- CPU sampling put 44% of in-gap samples in one ~270-byte function in the main
  binary, the shape of a tight serialization loop.

Replaced both directions with bulk copies (`push_le_slice` / `read_le_vec`) and
measured, matched, 48 req/point:

| | before | after | |
|---|---:|---:|---|
| dense TPOT c=16 | 99.29 ms | 97.64 ms | −1.7% |
| dense ITL p50 c=16 | 63.76 ms | 63.89 ms | wash |
| dense tail share | 45.9% | 44.5% | wash |
| MoE TPOT c=16 | 61.02 ms | 61.63 ms | +1% |

Everything inside the documented ±3% drift band. **Wash.**

The estimate that made it plausible — "37.7M × ~12 ns ≈ 450 ms" — was
ungrounded. LLVM compiles a fixed 4-byte push into a `Vec` with reserved
capacity into something far cheaper than a call, so the loop was never the
hundreds of milliseconds the arithmetic implied.

## Fix

Reverted. The change was strictly fewer operations, but it bought nothing
measurable and paid two `unsafe` blocks for it.

## Rule

**A per-element loop is not automatically the hot loop — price it before you
rewrite it.** Ratio, frequency, and a CPU-sample cluster all agreed, and the
hypothesis was still wrong, because the one number nobody measured was the cost
of a single iteration. The gaps remain unattributed: next is resolving that
sample cluster's symbol, not another candidate that merely fits the shape.
