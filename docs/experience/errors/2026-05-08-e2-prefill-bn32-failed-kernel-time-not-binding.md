# E2 prefill BLOCK_N=32 failed — kernel time isn't the binding constraint at sm_89 4k longctx

## Context

E2 prefill HD128 sm_89 re-tune attempt. Hypothesis: Hopper-default
`BLOCK_M=64, BLOCK_N=64, NUM_STAGES=2, NUM_THREADS=128` puts ~96 KB/CTA
smem usage which exhausts Ada's 100 KB/SM cap at occupancy ≤ 1 block/SM.
Halving `BLOCK_N` to 32 reduces (k_tile + v_tile) per stage from 32 KB
to 16 KB → 48 KB/CTA total → predicted occupancy 2 blocks/SM gain.

Plan scaffold: [`docs/plans/E2-tilelang-prefill-hd128-sm89-retune.md`](../../plans/E2-tilelang-prefill-hd128-sm89-retune.md).
Master strategy §3.3 + §7.3.

E1 (decode `BLOCK_M` re-tune) was tried first but hit TileLang 0.1.9
codegen rigidity for HD128 decode (see §Lessons below). Pivoted to E2
prefill which has more sweep space.

## Setup

ARLE built with prefill `BLOCK_N=32` (single-variable A/B vs Hopper-default
`BLOCK_N=64` baseline kept by codex's prior bench). Same `--kv-cache-dtype
bf16` as codex's Phase 0 baseline so the comparison is matched (BF16 KV
on both arms — production auto-FP8 was deliberately avoided to isolate
kernel-tile effect from KV format).

```bash
CUDA_HOME=/opt/cuda TORCH_CUDA_ARCH_LIST=8.9 ./target/release/infer \
    --model-path infer/models/Qwen3-4B --port 8000 --num-slots 8 \
    --max-seq-len 5120 --kv-cache-dtype bf16

PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/bench_guidellm.sh e2-prefill-bn32-c4 \
  --concurrencies 4 --max-seconds 120 --warmup 10 \
  --data 'prompt_tokens=4096,prompt_tokens_stdev=1,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256'
```

Raw artifacts:
`bench-output/2026-05-08-e2-prefill-bn32-c4/`.

## Results

| Engine | TTFT mean | TTFT p50 | TTFT p99 | ITL p50 | tok/s | Verdict |
|---|---:|---:|---:|---:|---:|---|
| ARLE pre-Phase 0 (FP8 KV, `786a20a`) | n/a | 1976.4 | n/a | 19.27 | 153.83 | control |
| ARLE Phase 0 graph BF16 KV (KILLED, `8b4a03b`) | 1956.8 | 1961.2 | 1997.3 | 25.58 | 122.95 | killed |
| **ARLE E2 prefill BN=32, BF16 KV (this run)** | **2111.1** | **2010.4** | 9215.6 | 25.43 | 120.81 | **killed** |
| SGLang 0.5.11 (#2) | 1117 | 972.9 | n/a | 19.44 | 164.3 | reference |

Matched A/B vs Phase 0 (both BF16-forced):

| Comparison | TTFT p50 Δ | ITL p50 Δ | tok/s Δ |
|---|---:|---:|---:|
| E2 BN=32 vs Phase 0 BF16 baseline | **+2.5% slower** | -0.6% (flat) | -1.7% |

## Root Cause

`BLOCK_N=32` doubles the outer KV-pipelined loop trips (4096-token
prefill: 64 iters at BN=64 → 128 iters at BN=32). Per-iter sync +
issue cost compounded through the pipelined loop equals or exceeds the
smem-savings → occupancy-gain benefit. **At sm_89 + Qwen3-4B + 4k
prefill, kernel time is not the binding constraint** — dispatch /
scheduling / launch overhead dominate, per master strategy §3.3 R1
finding (`docs/research/2026-05-07-sglang-prefill-stack-survey.md`).
This is the third independent confirmation:

1. master §3.3 graph capture math: 3.8 ms / 1003 ms gap = 0.38%
2. Phase 0 graph capture KILL: -0.8% TTFT (`8b4a03b`)
3. **E2 prefill kernel re-tune: +2.5% (regression)** ← this entry

## Fix

Revert prefill `BLOCK_N` from 32 back to 64. Decode kernels also reverted
to `BLOCK_M=64` after the BLOCK_M=16/32 attempts hit TileLang 0.1.9
codegen rigidity (see §Lessons).

## Lessons

### TileLang 0.1.9 HD128 codegen is rigid

E1 attempt: BLOCK_M 64 → 16 in BF16 + FP8 decode kernels.
Failure: `AssertionError: warp_col_tiles must be greater than 8, got 4`.
TileLang's `T.gemm(... policy=GemmWarpPolicy.FullRow)` requires
`HEAD_DIM / warp_tile_n > 8`. With BLOCK_M=16 + NUM_THREADS=128 (4 warps),
the row distribution forces warp_col_tiles=4.

E1 retry: BLOCK_M=32. Failure: fragment layout error
(`Fragment([32, 16] -> [4], replicate: 1, thread: 128, ...)` —
TileLang's LayoutInferencer can't map the score fragment to 128 threads
when (BLOCK_M, BLOCK_N) = (32, 16).

The HD128 prefill+decode kernels were tuned in lockstep with TileLang
0.1.9 codegen; the only valid `BLOCK_M` value here is 64. Sweep space
narrower than predicted in
[`E1 plan`](../../plans/E1-tilelang-decode-hd128-blockm-retune.md) +
[`E2 plan`](../../plans/E2-tilelang-prefill-hd128-sm89-retune.md).

### ncu wrapper out of date

`scripts/profile_ncu_guidellm.sh` uses ncu's obsolete `--attach-pid`
flag. ncu 2026.1.1.0 only supports `--mode=attach --hostname` to a
later-attach socket. The wrapper's attach mode no longer works on this
machine. E5 ncu profile run was skipped; pivoted to first-principles
E1/E2 sweeps. Wrapper repair / migration is out of scope for this entry
but should be filed under a tooling task.

### Kernel re-tune ROI is bounded by binding constraint

Three pieces of evidence (this entry's §Root Cause) now support the
thesis: at ARLE's current state on sm_89 + 4k longctx, kernel time is
not the binding constraint. The closure path to SGLang's TTFT is on
the **dispatch / scheduling / launch overhead** axis, which means
graph capture (Phase 0v2 with codex's fix list) is the high-ROI
investment, not kernel tile sweeps.

Per master strategy §6.2 moat list, TileLang custom kernel remains a
defensible capability (✓), but **further tuning on HD128 BF16 has
diminishing returns at sm_89**. FP8 decode (M_b.2 A1) and DSV4 HD64
(E4 substrate, just landed) are kernel-side investments with clearer
ROI signals (memory and new SKU coverage respectively).

## Rule

For sm_89 + agent shape (4k+ longctx), don't license kernel re-tune
plans on smem-budget arithmetic alone. Need either:

1. **nsys evidence that kernel-time is the binding fraction of TTFT**
   (e.g. >40% of step time in attention kernels) before investing
   in tile sweeps; OR
2. **Gross algorithmic restructure** (split-KV decode, persistent
   kernel, fundamentally different memory access pattern) — not just
   tile parameter changes.

Tile parameter sweeps without that grounding repeat the same +/- 2%
noise band Phase 0 graph + E2 BN=32 just demonstrated.

## Cross-references

- Master strategy §3.3 + §6.2 + §10:
  [`docs/projects/2026-05-07-arle-master-strategy.md`](../../projects/2026-05-07-arle-master-strategy.md)
- R1 SGLang prefill stack survey:
  [`docs/research/2026-05-07-sglang-prefill-stack-survey.md`](../../research/2026-05-07-sglang-prefill-stack-survey.md)
- Phase 0 graph KILL:
  [`docs/experience/errors/2026-05-08-m_pgc-phase0-killed-ttft-under-threshold.md`](2026-05-08-m_pgc-phase0-killed-ttft-under-threshold.md)
- E1 plan: [`docs/plans/E1-tilelang-decode-hd128-blockm-retune.md`](../../plans/E1-tilelang-decode-hd128-blockm-retune.md)
- E2 plan: [`docs/plans/E2-tilelang-prefill-hd128-sm89-retune.md`](../../plans/E2-tilelang-prefill-hd128-sm89-retune.md)
- Bench artifacts: `bench-output/2026-05-08-e2-prefill-bn32-c4/`
