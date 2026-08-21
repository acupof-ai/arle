# `gdr_decode_batch_kernel`: state slice staged in registers — 1.2–2.2× at the kernel, wash end-to-end

> Status: Landed. Closes backlog item 2 of
> [the NVFP4 decode plan](../../plans/2026-08-21-nvfp4-decode-lever-backlog.md);
> follows [the latency-bound finding](2026-08-21-gdr-decode-batch-is-latency-bound.md).

## Context

`csrc/recurrent/gdr_decode_batch.cu` ran two passes over each thread's 32-row
state slice: pass 1 loaded, decayed, stored; pass 2 re-loaded, applied the
rank-1 update, stored. Two dependent global round trips per element, and each
loop iteration's load waited on the previous one — the 61.6 % long-scoreboard
stall ncu measured.

## What Worked

The 32 values stay in registers between the passes (`float s_reg[32]`, both
loops `#pragma unroll`): one load and one store per element, all 32 loads
issued independently. The stored decayed value was a bit-exact f32 round trip,
so the arithmetic is unchanged and the state and output are bit-identical.

Standalone microbench, H20 (sm_90), Qwen3.6-27B shapes (48 value heads, 16 key
heads, 128×128), 200 launches after warm-up, state and output compared bitwise
after 3 steps against the old kernel:

| B | old µs | new µs | speedup | bit-exact |
|---|---:|---:|---:|---|
| 1 | 11.2 | 5.0 | 2.24× | state + out |
| 4 | 16.0 | 11.9 | 1.34× | state + out |
| 8 | 29.5–31.3 | 18.9 | 1.57–1.66× | state + out |
| 16 | 70.4–71.3 | 55.1–55.6 | 1.28× | state + out |
| 32 | 131.3–131.8 | 109.0–109.2 | 1.20× | state + out |

ncu at B=16 (`--launch-skip 669 -c 1` on the bench): warp cycles per issued
instruction 69.0 → 17.5; DRAM throughput 33 % → 43 % (1.33 → 1.75 TB/s);
registers 32 → 128 per thread (the unroll), achieved occupancy 88 % → 25 %.
Above B≈16 the state (3 MB per request) leaves L2 and the kernel is now
DRAM-bound on its own traffic.

Rejected on measurement, same bench: float4 loads with 16 row slices (no
faster, and the 16-way partial sum breaks bit-exactness at 1e-7);
`__launch_bounds__(512, 2)` to recover occupancy (364 B spill, 0.6–0.8×).

End-to-end, Qwen3.8-27B-NVFP4, 1×H20 (GPU 4), fp8 KV, 32 K agent prompts
×32, 214 output tokens, two interleaved trials per arm, same HEAD otherwise:

| arm | c=1 out tok/s | c=1 decode tok/s | c=16 out tok/s | c=16 decode tok/s |
|---|---:|---:|---:|---:|
| base t1 / t2 | 24.8 / 24.7 | 83.8 / 82.9 | 109.9 / 107.1 | 7.8 / 7.7 |
| new t1 / t2 | 24.8 / 24.9 | 83.0 / 83.4 | 109.1 / 109.6 | 7.8 / 7.9 |

Wash: the kernel saves ≈0.2 ms per token at c=1 (36 linear-attention layers ×
6 µs) against ≈12 ms per token, ≈2 %; less at c=16. The "13 % of decode GPU
time" in the backlog came from a different capture and does not describe this
workload. Needle ladder ×3 at 512/4096/16384/32768 on the new binary: 12/12
exact, DET.

## Rule

A kernel that is bit-exact and faster at every batch size ships on the kernel
receipt even when the end-to-end A/B is a wash; the entry says which it is.
Stage a thread's private slice in registers before reaching for wider loads or
occupancy knobs — on a latency-bound loop the first removes a dependent round
trip, the other two only move the stall.
