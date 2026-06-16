# Qwen3.6 FA3 Split Decode Attention

## Goal

Optimization: replace the Qwen3.6 HD256 GQA c=1 decode full-attention path that
used the in-tree prefill-style kernel as decode attention.

## Hypothesis

The old decode attention launched only one CTA per query head and serially
walked the 4K KV prefix. Routing decode to an upstream split-KV flash-decoding
kernel should parallelize each KV sweep and move the c=1 ITL wall from
attention/KV to the remaining decode stack.

## Command

Build:

```bash
cd /data01/arle-qwenfp8-smoke
CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda CUDARC_CUDA_VERSION=12090 \
  NVCC_CCBIN=/usr/bin/g++ RUSTFLAGS="-C link-arg=/tmp/ssl_peer_cert_compat.o" \
  ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1 ARLE_CUDA_ENABLE_FA3=1 \
  ARLE_CUDA_KERNEL_SET=dsv4_flash ARLE_CUDA_DISABLE_FLASHMLA=1 \
  cargo build --release --features cuda --bin arle
```

Serve A/B used the same binary and same prompt:

```bash
CUDA_VISIBLE_DEVICES=0 INFER_CUDA_DEVICES=0 INFER_TP_SIZE=1 \
  ARLE_QWEN35_FA3_DECODE={0,1} ARLE_QWEN35_FA3_DECODE_SPLITS=8 \
  ARLE_CUDA_DISABLE_FLASHMLA=1 RUST_LOG=info \
  ./target/release/arle serve --backend cuda \
  --model-path /data01/models/Qwen3.6-35B-A3B-FP8 \
  --port <port> --bind 127.0.0.1 \
  --num-slots 1 --total-pages 272 --page-size 16 \
  --max-prompt-tokens 4096 --max-total-tokens 4352
```

Client: OpenAI chat completions, `temperature=0`, prompt_tokens=4015,
`max_tokens=1` and `max_tokens=257`; ITL is the wall-clock slope.

## Environment

- Backend: CUDA, single H20, TP=1
- Model: `/data01/models/Qwen3.6-35B-A3B-FP8`
- Feature set: `--features cuda`
- Non-default build env: `ARLE_CUDA_ENABLE_FA3=1`,
  `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1`, `ARLE_CUDA_KERNEL_SET=dsv4_flash`
- Runtime gate: `ARLE_QWEN35_FA3_DECODE=1`
- Split count: `ARLE_QWEN35_FA3_DECODE_SPLITS=8`

## Results

Correctness smoke passed before the perf claim. The 2824-token needle prompt
generated text containing the exact needle `BLUE-73-MANGO` under
`ARLE_QWEN35_FA3_DECODE=1`.

Same-binary c=1 A/B:

| Mode | Prompt / output tokens | max1 wall | max257 wall | Slope ITL |
| --- | ---: | ---: | ---: | ---: |
| Baseline devpos decode attention | 4015 / 257 | 0.913 s | 6.749 s | 22.80 ms |
| FA3 split decode attention | 4015 / 257 | 0.984 s | 3.323 s | 9.14 ms |
| Delta | same | +7.8% TTFT-like max1 | -50.8% total decode wall | -59.9% ITL |

This hits the tranche target: the Qwen FP8 c=1 decode wall is now near
9-10 ms instead of the previous 22 ms band.

## Problems

- Nsight Systems process-tree capture produced a SQLite export without CUDA
  kernel tables for the launched serve child, so it is not used as evidence.
- Nsight Compute could connect to the serve process, but the vendored
  FA3/CUTLASS kernels show up as the generic `device_kernel`. A regex on
  `FlashAttn` matched no kernels; a `device_kernel` capture made the serve exit
  during startup. The reliable measured evidence for this entry is the
  correctness smoke plus the same-binary e2e A/B above, anchored by the prior
  NCU RCA of the old kernel.
- The path is opt-in only. It uses host `seqlen_k`, so it is deliberately
  disabled when `ARLE_QWEN35_DECODE_GRAPH=1` until a graph-safe dynamic length
  design lands.

## Learnings

- Adopt-first worked: SGLang uses FlashInfer
  `BatchDecodeWithPagedKVCacheWrapper` for decode, and ARLE already vendors FA3
  HD256 split + PackGQA + combine instantiations. Extending the torch-free FA3
  shim was the smallest mature-kernel port; no hand-written attention kernel was
  needed.
- For Qwen3.6 HD256 GQA decode, the old in-tree prefill kernel was the wrong
  algorithmic shape. KV-split flash decoding is the load-bearing lever; CUDA
  graph and GEMV roofline work were secondary for this c=1 wall.

## Delta vs baseline

Baseline: [`2026-06-16-qwen36-fp8-decode-vectorized-dequant.md`](2026-06-16-qwen36-fp8-decode-vectorized-dequant.md)
recorded the post-dequant-fix c=1 FP8 ITL at 22.38 ms and the old attention/KV
trace at 13.10 ms/token.

| Metric | Old path | FA3 split decode | Delta |
| --- | ---: | ---: | ---: |
| 4015/257 slope ITL | 22.80 ms | 9.14 ms | -59.9% |
| Prior c=1 ITL band | 22.38 ms | 9.14 ms | -59.2% |

## Verification

```bash
# Local checkout check was blocked by another agent's dirty
# crates/infer-cuda/src/{attention.rs,dsv4.rs} half-state, so CUDA validation
# used the clean smoke tree with only this tranche's files synced.
CUDARC_CUDA_VERSION=12090 cargo check -p infer-api \
  --release --no-default-features --features cuda,no-cuda --lib
CUDARC_CUDA_VERSION=12090 cargo check -p cuda-kernels \
  --release --no-default-features --features cuda,no-cuda --tests
CUDARC_CUDA_VERSION=12090 cargo clippy -p infer-api \
  --release --no-default-features --features cuda,no-cuda --lib -- -D warnings
```
