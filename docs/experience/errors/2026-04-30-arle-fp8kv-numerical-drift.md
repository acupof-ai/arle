# ARLE Qwen3.5 FP8 KV Numerical Drift

## Context

The 2026-04-29 fp8 KV ablation found ARLE Qwen3.5 fp8 vs bf16 spot-check
badly diverged: `0/16` exact pairs, `2.79%` average generated-token match,
and earliest divergence at generated token `0`. SGLang's same fp8-vs-bf16
spot-check was much closer (`8/16` exact, `77.54%` token match), so ARLE's
fp8 KV path had a runtime bug rather than ordinary fp8 precision loss.

## Root Cause

Two fp8 readback bugs were present in the ARLE Qwen3.5 durable-KV path:

1. The Rust FFI for `dequantize_paged_kv_fp8_to_hnd_cuda` passed a `scales`
   pointer, but the CUDA C function signature did not accept it. That shifted
   every following ABI argument and made the refill kernel read/write through
   the wrong pointers.
2. The FP8 durable-to-BF16 HND refill kernel cast fp8 bytes directly to float
   and did not multiply by the per-token/per-head K/V scales written by the
   quantize kernels.

The failure hit Qwen3.5 full-attention decode because that path dequantizes
durable fp8 KV back into the BF16 HND work buffer before using FlashInfer
HD256. Short one-token smoke checks could miss it because the first sampled
token mostly depends on prefill logits; the corrupted durable readback shows
up as soon as decode consumes cached KV.

## Fix

- Keep the FP8 scale buffers allocated for `KVFormat::FP8E4M3`.
- Quantize FP8 KV with per-token/per-head scales for K and V.
- Pass scale pointers through the Rust and CUDA FFI consistently.
- Multiply FP8 values by their stored scale during Qwen3.5 durable-to-BF16 HND
  refill.
- Keep the qlen=1 fp8 decode kernel scale-aware for Qwen3 HD128 paths and
  keep Qwen3.5 fp8 decode on the fused split-KV path instead of full-cache
  BF16 refill.
- Bump the FP8 stable tag from `4` to `5` so pre-scale persisted KV payloads
  become clean misses instead of incompatible restores.

## Verification

Commands:

```bash
cargo fmt --all --check
CUDA_HOME=/usr/local/cuda CARGO_TARGET_DIR=/tmp/arle-target CARGO_HOME=/tmp/arle-cargo-home ZIG=/tmp/zig-0.15.2/zig-x86_64-linux-0.15.2/zig cargo check -p infer --release --no-default-features --features cuda
```

Spot-check:

- Model: `infer/models/Qwen3.5-4B`
- Server flags: `--num-slots 16 --max-seq-len 8192 --chunked-prefill-size 2048 --max-prefill-tokens 16384 --max-num-batched-tokens 16384 --mem-fraction-static 0.85`
- Pair: ARLE fp8 KV vs ARLE bf16 KV
- Prompts: same 16 fixed prompts from the 2026-04-29 spot-check
- Sampling: `temperature=0`, `seed=20260429`, `max_tokens=64`
- Tokenizer: `infer/models/Qwen3.5-4B/tokenizer.json`

| metric | before | after |
|---|---:|---:|
| exact pairs | 0 / 16 | 3 / 16 |
| avg common-token match | 2.79% | 39.08% |
| earliest divergence | generated token 0 | generated token 1 |

Artefacts:

- `bench-output/2026-04-30-arle-qwen35-spotcheck-fp8-scale-fused.json`
- `bench-output/2026-04-30-arle-qwen35-spotcheck-bf16-scale-fix.json`
- `bench-output/2026-04-30-arle-qwen35-spotcheck-compare-scale-fused.json`
- `bench-output/2026-04-30-arle-qwen35-spotcheck-bf16-scale-fix-server/server.log`

## Rule

For quantized KV, treat scale buffers as part of the ABI and layout contract.
Every write path, read path, refill path, and FFI declaration must be checked
together. A first-token-only check is insufficient; verify at least 16 prompts
with 64 generated tokens before claiming fp8 KV correctness.

## Follow-Up

The two concrete readback bugs are fixed, but fp8 KV is still not ready to flip
default for Qwen3.5: ARLE remains below the SGLang reference spot-check
(`77.54%` average token match). Keep `auto`/default policy unchanged until the
remaining divergence is understood.
