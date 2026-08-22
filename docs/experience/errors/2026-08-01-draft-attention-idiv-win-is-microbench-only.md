# The draft attention IS ALU-bound — but removing the IDIV only pays in a microbench

## Context

Second attempt at `nonpaged_prefill_attention_kernel`, after
[the reduction-axis revert](2026-08-01-draft-attention-reduction-axis-was-not-the-cost.md).
That entry's leftover hypothesis — GQA re-read of the K/V window — is also dead;
this run killed it with counters.

The serve-side profile is confounded (`ncu` serializes the launches it inspects,
the batch drains, the grid shrinks out of the costly regime), so the kernel moved
into `crates/cuda-kernels/tools/nonpaged_attn_bench.cu` — body verbatim, shape
pinned from `dspark-fr-native/config.json`: 32 q heads, 8 kv heads, head_dim 128,
ring cap = 2048 window + 16 block.

## What the counters said

One `ncu --set full` at 3072 blocks:

| | value |
|---|---|
| Compute (SM) throughput | **80.15%** |
| ALU pipeline (integer/logic) | **61.9%** — highest |
| FP32 of peak | 11% |
| L2 hit rate | **99.58%** |
| L2 cache throughput | 7.29% |
| DRAM throughput | 0.06% |

L2 absorbs the GQA re-read entirely and sits 93% idle, so that hypothesis is
gone. The kernel is issue-bound on *integer* work, and the integer work is one
expression evaluated per key, per thread, twice per key:

```cuda
int row = ring_modulus > 0 ? ((ring_base + abs_pos) % ring_modulus) : abs_pos;
```

`ring_modulus` is a runtime value, so `%` is a ~20-instruction IDIV emulation.
Every caller bounds the walk by `kv_len <= ring_modulus`, so it wraps at most
once: normalize `ring_base` once at kernel entry, and the loop needs only a
conditional subtract.

## Root Cause of the failure

**The microbench regime is not the serve regime.** Pinned at a full 2048-key
window the rewrite is a large, clean win — 20 iterations after 3 warmups, ring
bases deliberately unnormalized so the A/B covers the precondition:

| draft rows | IDIV | conditional subtract | Δ | output |
|---|---:|---:|---:|---|
| 12 | 0.571 ms | 0.503 | −12.0% | bit-identical |
| 48 | 1.970 | 1.364 | −30.8% | bit-identical |
| 96 | 3.794 | 2.535 | **−33.2%** | bit-identical |

`ncu` after: duration 4.12 → 2.74 ms, memory throughput 37.78 → 56.83%.

In the serve it is reproducibly **slower**. Matched A/B, both binaries built from
the same HEAD with only the `.cu` differing, ThinkingCap-27B-FP8 +
`dspark-fr-native`, block 16, `--spec-max-batch 8`, c=8, short prompts ×
3000 tokens out so the ring fills:

| | baseline | conditional subtract | Δ |
|---|---:|---:|---:|
| draft attn, Q4 (ring full) | 7.46 ms | 7.66 | +2.7% |
| draft attn, overall median | 5.70 ms | 5.83 | +2.3% |
| accept | 5407/11525 | 5407/11525 | identical |

GPUs swapped between arms and rerun: 7.46 → 7.66 and 5.70 → 5.81 again. Same
ordering, same magnitude — device asymmetry is excluded.

The reason the microbench win does not transfer: at Q4 the serve spends 7.46 ms
on 5 layers × 128 rows, i.e. ~1.5 ms per launch, where the microbench's 128-row
full-window launch costs ~5 ms. **The serve's `kv_len` is roughly 600, not
2048** — the draft's sliding window never fills in this workload, so the per-key
IDIV is a much smaller share of the loop while the entry-normalization IDIV is
paid unconditionally by every thread.

Also measured, both wash: 32K agent prompts at c=1/c=8 (133.1 vs 133.4 tok/s,
per-line draft phases agreeing to 0.5%), and short prompts at c=1/c=16 (80.8 vs
80.7 and 108.7 vs 108.1 tok/s).

## Fix

Reverted. The harness stays at `crates/cuda-kernels/tools/nonpaged_attn_bench.cu`
— it is the only clean way to profile this kernel, and its `FastRing` template
parameter gives a bit-identity gate for free.

The characterization stands and is the real result: this kernel is **ALU-bound,
not bandwidth-bound**, at 11% of FP32 peak with L2 93% idle. Whatever pays next
has to remove integer/issue work at the `kv_len ≈ 600` operating point, not at
the full-window one.

## Rule

**A pinned-shape microbench proves the mechanism, not the win.** Pin the shape
from the model config and you still have to check that the *serving* workload
reaches it — here the config says the window is 2048 and the serve runs at ~600,
and the entire −33% lived in the gap.

**One arm faster on GPU 0 is not a result until the arms swap GPUs.** The swap
costs one rerun and converts "2.7%, probably noise" into a verdict.
