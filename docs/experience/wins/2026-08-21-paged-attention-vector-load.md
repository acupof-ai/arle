# One vector load for the paged-attention KV row — CUDA, 2026-08-21

> Status: Shipped, `185319d40`. Qwen3.8-27B-NVFP4, 1xH20, FP8 KV.

## Context

At serving concurrency `paged_attention_quantized_fa3_partial` is **80.6% of
decode GPU time at c=16 and 82.7% at c=32**
([errors/2026-08-21](../errors/2026-08-21-decode-profile-taken-at-the-wrong-batch.md)).
`ncu` says it is not bandwidth-bound — DRAM sits at 5.87% of peak while the SM
issue pipe is 68% busy — and that it needs **2.03 billion instructions per launch
to consume 1.07 GB of KV**, about 163 per (token, q-head) for 512 bytes.

## What worked

The inner loop read KV one byte at a time: `d = lane_id * EPT + i` with `EPT = 8`
gave 48 `LDG.E.U8` per unrolled body, each with its own address. But a lane's
`EPT` bytes are **contiguous**, so they are one aligned load. The KV scale is
also constant over `d`, so it leaves the loop.

| SASS, `<256,false>` | base | shipped |
|---|---:|---:|
| `LDG.E.U8.CONSTANT` | 48 | **0** |
| `LDG.E.64.CONSTANT` | 0 | **4** |
| `FMUL` | 300 | **260** |
| `LDS.128` / `STS` / `STS.128` | 32 / 8 / 6 | 32 / 8 / 6 |

`cuobjdump --dump-sass` was the working gate here, not the GPU. It is free, takes
the length of a build, and it caught two regressions I wrote before either
reached a bench:

- **A `union` cannot hold the row.** nvcc round-tripped it through memory: 48
  `LDG.E.U8` became a store plus **96 `LDS` and 32 `STS`**. Holding the row as
  `uint32_t w[EPT/4]` and shifting the byte out of the register word fixes it.
- **Restaging `o_reg` to kill the shared-memory bank conflict lost the merge.**
  `[i*32 + lane]` is conflict-free but scatters a lane's floats 32 apart, so
  `STS.128` became 8 scalar `STS`. The conflict `ncu` flags at 47% is 430k
  wavefronts against a 9.6M-cycle kernel; the merge is worth more. Reverted.

## Result

c=32 on the 32K agent chain, 32 requests per point, 3 repeats per arm, both arms
one GPU back to back, 32/32 complete and 0 errors everywhere. No decode-only
metric (ITL) was recorded for this A/B, so the serving delta is unreported.
(`rep1` is a cold-start artifact in both arms and is excluded.)

The gain is smaller than the instruction count predicts: ~25% fewer inner-loop
instructions against a small serving delta. The kernel is latency-bound as much as
issue-bound — `long_scoreboard` is 56% of a 13.96-cycle issue gap — so removing
instructions does not shorten the critical path proportionally.

## Correctness

Needle ladder 512 / 4096 / 16384 / 32768, ×3 same-config, `RAW=1
TEMPLATE=qwen3_nonthink`, FP8 KV: **`exact=3 partial=0 miss=0 DET` at every
length on both arms.**

The vector load is bit-identical by construction — each lane still owns the same
`EPT` elements, so the warp reduction is unchanged. The scale hoist is not:
`Σ(q·k·s)` becomes `Σ(q·k)·s`, which rounds once instead of `EPT` times. It is
therefore gated on the ladder rather than on byte equality, and it is the more
accurate of the two forms.

## Tradeoff

None material, which is the tell that the baseline was suboptimal rather than
that the win is noise: +20 lines, one struct, no change to registers, shared
memory or occupancy. The `uint2` load needs `EPT` alignment, which `row_off`
being a multiple of `HEAD_DIM` guarantees at both 128 and 256.

Below the project's 10% generic-kernel-retune license bar — but that bar assumes
σ ≈ 5%, and this is 4.25% at σ = 0.2%. Kept as a strict improvement, not claimed
as a licensed default flip.

## Rule

Disassemble before you bench. Three of this kernel's four iterations were
decided by `cuobjdump` in the time a build takes: the first proved the compiler
was *not* already vectorising (48 real `LDG.E.U8`), the second and third caught
regressions I had written into the fix. Only the fourth was worth a GPU.
