# NVFP4: the last four load sites onto Marlin — +51.7% at c=16, CUDA, 2026-08-19

> Status: Shipped

## Context

[`2026-08-19-marlin-fp8-per-channel.md`](2026-08-19-marlin-fp8-per-channel.md)
routed per-channel FP8 to Marlin and closed most of the concurrency gap, but
left its own ceiling on record: `cuda.qwen.fp8_gemv` still took 80507 of 238235
FP8 calls. Four load sites called `load_matrix_quant_aware`, which does no
Marlin repack, so their weights had no Marlin layout to route to — lm_head,
`linear_attn.out_proj` x48, the TP=1 fused qkv, and the MTP fc.

## What worked

Four call sites switched to `load_dense_matrix_quant_aware`, which is the same
load followed by `marlin_repack_dense` (`loader.rs:5675`). Both repacks inside
it are format-gated no-ops, so one call covers NVFP4 and per-channel FP8, and
every other format keeps the scalar GEMV.

- `qwen35_load.rs:346` lm_head
- `qwen35_load.rs:437` `linear_attn.out_proj`
- `qwen35_load.rs:1006` TP=1 fused qkv
- `qwen35_load.rs:1294` MTP fc

Engagement checked before the numbers were read. `cuda.qwen.fp8_gemv` 80507 ->
**513** (load-time and warm-up only), `cuda.qwen.fp8_marlin_tensorcore`
1153107, `cuda.fp4.marlin_tensorcore` 891072. `SERVER_ERRORS=0`.

## Result

1xH20 (sm_90), TP=1, `--kv-cache-dtype fp8`, no spec, decode graph on.
Synthetic prompt (mean 8 tokens), `--seconds-per-concurrency 30 --max-tokens
128 --temperature 0 --seed 42`. Aggregate tok/s = `c * 1000 / itl_mean_ms`.

| c | before | after | delta | FP8 | vs FP8 |
|---:|---:|---:|---:|---:|---:|
| 1 | 82.1 | 82.5 | +0.5% | 61.5 | **+34.1%** |
| 2 | 128.5 | 140.4 | +9.3% | 99.8 | **+40.7%** |
| 4 | 233.8 | 272.9 | +16.7% | 195.9 | **+39.3%** |
| 8 | 367.1 | 489.2 | +33.3% | 358.0 | **+36.6%** |
| 16 | 472.4 | 716.6 | **+51.7%** | 632.5 | +13.3% |

The gain grows with concurrency, which is the mechanism confirming itself: the
batched GEMV's tile is the batch, so it re-reads the weight as the batch grows,
and Marlin's per-call cost does not.

The FP8 column is carried from the previous entry's run rather than
re-measured. It is a control that this change cannot reach: the Qwen3.6-27B-FP8
checkpoint is 128x128 block-scaled, which fails `quant_block_m == 1` in
`repack_for_marlin_fp8`, and that run recorded `fp8_marlin_tensorcore` ABSENT
on the FP8 server.

## Against the +30%-over-FP8 goal

Met at c=1 through c=8 (+34.1% / +40.7% / +39.3% / +36.6%). **Not met at c=16
(+13.3%)**, and c=16 is a cliff rather than a decay — c=8 is still +36.6%.

The c=16 residue is not routing; every quantised GEMM is on Marlin there. A
per-op profile (`ARLE_CUDA_PROFILE=1`, c=1 vs c=16, both checkpoints) puts it in
one op:

| op | bytes | NVFP4 ms | FP8 ms | NVFP4 TB/s |
|---|---|---:|---:|---:|
| linear/in_proj | same | 2.874 | 3.680 | 1.88 |
| linear/out_proj | same | 1.405 | 1.935 | 2.03 |
| full_paged/qkv_gemm | same | 0.723 | 0.976 | 1.99 |
| full_paged/o_proj | same | 0.447 | 0.642 | ~2.0 |
| lm_head | same | 0.453 | 0.606 | ~2.1 |
| **dense_ffn** | 10.56 vs 17.11 GB | **9.201** | 11.635 | **1.15** |
| GEMM subtotal | | 15.103 | 19.474 | |
| non-GEMM (shared) | | 7.351 | 6.963 | |
| leaf total | | 22.454 | 26.437 | |

On the five ops where both checkpoints read the same bytes, Marlin is 22-30%
faster than the FP8 checkpoint's path. `dense_ffn` is the exception: 38% fewer
bytes for 21% less time. Splitting it (8.42 GB NVFP4 + 2.14 GB per-channel FP8
in layers 56-63) puts Marlin's NVFP4 arm at ~1.04 TB/s against its FP8 arm's
~1.95 TB/s on the same card.

## Rule

A format-gated repack helper is only as wired as its call sites. After adding
one, grep every `load_*_quant_aware` caller rather than the ones the current
model exercises — `5499e20a7` fixed two sites for FP4 and named these four,
and they still shipped one release later carrying 34% of the FP8 calls.
