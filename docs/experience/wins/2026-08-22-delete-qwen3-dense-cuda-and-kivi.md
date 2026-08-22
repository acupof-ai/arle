# CUDA drops Qwen3 dense and KIVI per-channel K — one quantized-KV scheme, one decode kernel

> Status: Landed. Phase 2 of
> [the unification plan](../../plans/2026-08-22-quantized-kv-attention-unification.md),
> resolved by deleting the family instead of porting it.

## Context

The only consumer of KIVI per-channel K quantisation (`decode_attention_quantized.cu`,
970 lines, plus calibration, pool scale tables and a CSR decode metadata path)
was the CUDA `model_type=qwen3` dense executor. No Qwen3 dense checkpoint is
served on CUDA; the evidence that per-channel K was needed
(`errors/2026-05-26-*`) was retracted as a test artifact, and the Qwen3.5
family runs per-(token, head) K through the tensor-core kernel with needle and
eval gates passing.

## What Worked

Delete the family and the scheme: `executor/qwen.rs`, `model.rs`,
`decode_graph*.rs`, the dense branches of `attention.rs`, `decode_prep_paged.cu`,
the HD128 TileLang rows, KIVI kernels and pool fields, `KVFormat::INT4`, the
`quant_decode_meta` CSR metadata, and the `--no-cuda-graph` /
`enable_cuda_graph` chain whose only sink was the dense decode graph
(Qwen3.5's graph is `--qwen35-decode-graph`). `model_type=qwen3` now fails at
load with "Qwen3 dense is not supported on CUDA; use a Qwen3.5-family
checkpoint". 57 files, +223 / −7340.

Every quantized KV pool on CUDA now stores per-(token, head) scales for K and
V and decodes through `paged_attention_quantized_fa3.cu`.

Gate on the resulting binary (`0d46ac4a8`), Qwen3.8-27B-NVFP4, fp8 KV: needle
ladder ×3 at 512/4096/16384/32768 12/12 exact, DET; 200-item GSM8K-train greedy
eval 177/200 (177 on the previous binary; same-base FP8 176).

## Rule

A quantisation scheme with one consumer is that consumer's cost. When the
consumer is unused, delete both; do not port a scheme to keep a dead family
alive.
