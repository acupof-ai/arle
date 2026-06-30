# DSv4-Flash-FP8 compressor/indexer BF16→FP8 host quantize — pending-remote

## SLO-shape probed? N — pending-remote (Mac, no CUDA)

## Roofline check

Deferred pending pod bench. The change eliminates the BF16 GEMV kernel (53.4% of
decode GPU time per nsys at B=4) and routes compressor/indexer matrices through
`dsv4_fp8_gemv_batch_tiled_kernel` instead of `gemv_handwritten_kernel`. Expected
improvement: ~2× decode throughput at B=4 TP=4 (hypothesis only, not measured).

## Goal

Eliminate `gemv_handwritten_kernel` (53.4% decode GPU time, 973 K calls in 10s
nsys trace) by quantizing DSv4-Flash-FP8 compressor/indexer BF16 weights to FP8
E4M3FN block-scaled at load time.

## Hypothesis

Compressor (`wkv`, `wgate`, `ape`) and indexer (`wq_b`, `weights_proj`) matrices
ship as BF16 in the `deepseek-ai/DeepSeek-V4-Flash-FP8` checkpoint. Quantizing
them to 128×128-block E8M0-scaled FP8 at load time routes every `dsv4_linear`
call through `mla_linear` → `dsv4_fp8_gemv_batch_cuda` instead of
`ops::gemm_batch` → `gemv_handwritten_kernel`.

## Change

Single file: `crates/infer-cuda/src/loader.rs`

- `encode_f8_e4m3fn_sat(f32) -> u8`: saturating E4M3FN encoder (sign + 4-bit exp +
  3-bit mant, bias=7, max=448, NaN→0).
- `SafetensorLoader::quantize_to_dsv4_fp8_host(...)`: BF16/F32 → (FP8 bytes, E8M0
  scale bytes, scale_rows, scale_cols) using 128×128 blocks. Scale byte =
  ceil(log2(max_abs/448) + 127), clamped [1,254].
- `load_dsv4_global_matrix`: BF16/F32 branch now calls `quantize_to_dsv4_fp8_host`
  + `DeviceMatrix::from_dsv4_fp8_block_scaled` instead of `load_dsv4_bf16_matrix`.
  FP8/I8 branch unchanged.

## Verification

`CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib` — clean.

Pod bench pending: `scripts/bench_guidellm.sh dsv4-fp8-compressor-fp8 --backend cuda`
on 8×H20, DSv4-Flash-FP8 TP=4, B=1/4/8/16, compare vs baseline snapshot.

## Problems

- Checkpoint dtype for compressor/indexer confirmed BF16 indirectly via nsys
  (53.4% in `gemv_handwritten_kernel`); direct `safetensors` dtype probe on pod
  not run before landing (pending).
- Host quantization introduces a small approximation error vs native FP8 weights.
  For attention compressor matrices this is expected to be below the MoE
  non-determinism floor.

## Learnings

`load_dsv4_global_matrix` was the single dtype dispatch point for all non-attention
matrices (compressor/indexer/hyper-connection). Routing BF16 through host
quantization at load time requires zero changes to the runtime dispatch (`dsv4_linear`
already handles `Dsv4Fp8BlockScaled`). The `DeepGEMM` cache is built automatically
by `from_dsv4_fp8_block_scaled` when the env gate is on; it is unused by
`mla_linear`/`dsv4_linear` for these matrices (those use the scalar GEMV path).
