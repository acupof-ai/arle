# Qwen3.5 BF16 HD256 FlashInfer Plan Gate

## Context

Qwen3.5 BF16 KV failed during server warmup/probe after `tilelang-attn` was
restored to the default `cuda` feature by `47bad713`. The failure reproduced
with and without CUDA graph capture.

Observed logs:

- `bench-output/2026-04-29-arle-qwen35-bf16kv-fixed-server/server.log`
- `bench-output/2026-04-29-arle-qwen35-bf16kv-fixed-nograph-server/server.log`

Both failed in FlashInfer HD256 decode with CUDA error 9
`invalid configuration argument`.

## Root Cause

The Qwen3.5 decode context planned BF16 HD256 attention using
`cfg(not(feature = "tilelang-attn"))`, but the runtime dispatch selected the
TileLang HD256 decode kernel only under `cfg(feature =
"tilelang-decode-hd256")`.

Default CUDA after `47bad713` is:

- `tilelang-attn = ON`
- `tilelang-decode-hd256 = OFF`

That combination skipped `plan_hd256` but still dispatched BF16 HD256 decode
to FlashInfer. FlashInfer then read zeroed plan metadata, including
`plan_info.padded_batch_size = 0`, and launched with an invalid grid in the
non-partition-KV branch.

FP8 did not hit this path because Qwen3.5 FP8/INT8 KV decode dispatches to the
quantized kernels instead of FlashInfer HD256.

## Fix

`infer/src/model/qwen35/batch_decode.rs` now gates `plan_hd256` on the actual
HD256 decode-kernel feature:

- `cfg(not(feature = "tilelang-decode-hd256"))` plans FlashInfer HD256.
- `cfg(feature = "tilelang-decode-hd256")` skips the FlashInfer plan because
  the TileLang HD256 decode kernel is plan-less.

Verification:

- `cargo check -p infer --features cuda`
- Qwen3.5 BF16 probe generated 32 tokens without CUDA error 9.
- Qwen3.5 BF16 c=16 / 4096 / 256 / 120s completed successfully:
  `bench-output/2026-04-29-arle-qwen35-bf16kv-planfix/`.

## Rule

Gate planning on the same feature that gates the kernel dispatch. The broad
`tilelang-attn` feature is not a proxy for `tilelang-decode-hd256`.
