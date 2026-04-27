# Metal GGUF Q4/Q6 Affine Repack on Qwen3.5 0.8B

## Goal

- Type: optimization / regression check.
- Move the Qwen3.5 GGUF Q4_K_M decode hot path from custom scalar GGUF
  matmul toward MLX/oMLX-style repacked tiled quantized matmul, with Q4_K and
  Q6_K covered because Q4_K_M stores `lm_head` as Q6_K.

## Hypothesis

- Loading Q4_K/Q6_K GGUF rows into MLX affine quantized layout should let
  `quantized_matmul` dispatch tiled qmv/qmm kernels and improve decode
  throughput substantially.
- Q6_K requires affine group size 16, so the Metal qmv kernel table must expose
  `gs_16_b_6`; qmm remains unsupported for group 16 and is routed through qmv.

## Command

```bash
./target/release/metal_bench \
  --model /path/to/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt-tokens 512 \
  --generation-tokens 1024 \
  --warmup 0 \
  --runs 1 \
  --ignore-eos \
  --json

AGENT_INFER_QWEN35_GENERATE_PROFILE=1 \
./target/release/metal_bench \
  --model /path/to/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt-tokens 32 \
  --generation-tokens 256 \
  --warmup 1 \
  --runs 1 \
  --ignore-eos \
  --json
```

Correctness / verification:

```bash
cargo test -p infer --release --no-default-features --features metal,no-cuda gguf_q -- --nocapture

QWEN35_MODEL_PATH=/path/to/code/agent-infer/models/Qwen3.5-0.8B \
QWEN35_GGUF_PATH=/path/to/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
cargo test -p infer --release --no-default-features --features metal,no-cuda \
  compare_qwen35_0p8b_gguf_cpp_generate_vs_rust_replay -- --ignored --nocapture

QWEN35_MODEL_PATH=/path/to/code/agent-infer/models/Qwen3.5-0.8B \
QWEN35_GGUF_PATH=/path/to/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
cargo test -p infer --release --no-default-features --features metal,no-cuda \
  compare_qwen35_0p8b_gguf_cpp_gdr_kernel_vs_fallback_prefill -- --ignored --nocapture

cargo check -p infer --release --no-default-features --features cuda,no-cuda
cargo clippy -p infer --release --no-default-features --features metal,no-cuda -- -D warnings
cargo fmt --check
```

## Environment

- Backend: Metal
- Hardware: Apple M4 Pro, unified memory
- OS / Metal: macOS 26.3.1 (25D771280a), Metal 4
- Model: Qwen3.5-0.8B GGUF Q4_K_M
- Feature set: `--release --no-default-features --features metal,no-cuda`
- Non-default flags: `--ignore-eos`; profile run used
  `AGENT_INFER_QWEN35_GENERATE_PROFILE=1`
- Commit: working tree before commit

## Results

Raw JSON, 512 prompt / 1024 decode:

```json
{"avg_tokens":1024,"generation_tokens_requested":1024,"generation_tps":{"mean":82.52107442882134,"p50":82.52107442882134,"p99":82.52107442882134},"load_ms":1945.3749999999998,"model":"/path/to/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf","peak_rss_mb":1411.90625,"prompt_tokens":512,"prompt_tokens_requested":512,"prompt_tps":{"mean":483.8475138494854,"p50":483.8475138494854,"p99":483.8475138494854},"quantization":"gguf-q4_k","repo_e2e_tps":{"mean":76.03695395962438,"p50":76.03695395962438,"p99":76.03695395962438},"timed_runs":1,"total_time_ms":{"mean":13467.136,"p50":13467.136,"p99":13467.136},"ttft_ms":{"mean":1058.184625,"p50":1058.184625,"p99":1058.184625},"warmup_runs":0}
```

Profile run, 32 prompt / 256 decode:

```text
[qwen35-profile] decode tokens=256 total=2923.087ms avg=11.418ms forward_build_avg=1.550ms sample_build_avg=0.003ms async_avg=9.659ms eval_wait_avg=0.000ms item_avg=0.001ms bookkeep_avg=0.171ms clear_cache_total=2.353ms last_intermediates=582
{"generation_tps":{"mean":87.57865794630825},"repo_e2e_tps":{"mean":85.39434047340654},"ttft_ms":{"mean":74.770167}}
```

## Delta vs Baseline

Baseline entry:
[`2026-04-27-bench-metal-qwen35-0p8b-gguf-quant-coverage.md`](2026-04-27-bench-metal-qwen35-0p8b-gguf-quant-coverage.md).

| Workload | Metric | Prior Q4 | Current Q4 | Delta |
|---|---|---:|---:|---:|
| 512/1024 | Peak RSS | 905.8 MB | 1411.9 MB | +55.9% |
| 512/1024 | Prompt throughput | 186.53 tok/s | 483.85 tok/s | +159.4% |
| 512/1024 | Decode throughput | 38.57 tok/s | 82.52 tok/s | +114.0% |
| 512/1024 | Repo E2E throughput | 34.95 tok/s | 76.04 tok/s | +117.6% |
| 512/1024 | TTFT | 2744.8 ms | 1058.2 ms | -61.4% |
| 512/1024 | Total time | 29296.8 ms | 13467.1 ms | -54.0% |

## Correctness

| Check | Result |
|---|---|
| GGUF packed matmul Q8_0/Q3_K/Q4_K/Q5_K/Q6_K vs CPU dequant reference | passed |
| Q4_K/Q6_K affine repack vs CPU dequant reference | passed |
| Qwen3.5 GGUF C++ generate vs Rust replay | passed; replay tokens `\n\n`, `Hello`, ` world`, ` from` |
| GGUF C++ custom GDR prefill vs fallback | passed; max logits 0.032017, max KV 0.062500, max GDR 0.036978 |
| CUDA typecheck on macOS | passed |
| Metal clippy | passed |

## Problems

- This is not yet the requested 200 tok/s target. Long decode is 82.5 tok/s;
  short profile decode is 87.6 tok/s.
- RSS regressed versus raw packed GGUF because Q4_K/Q6_K now keep MLX affine
  weights plus scales/biases instead of only raw GGUF bytes.
- The remaining decode cost is layer execution, not `lm_head`: a local
  diagnostic that skipped `lm_head` only moved short decode from the high 80s
  tok/s to roughly the same band.

## Learnings

- Q4_K_M must optimize Q6_K too; otherwise `lm_head` stays on the slow path.
- MLX affine qmv can carry exact GGUF Q6_K if group size 16 is exposed for
  qmv/qmv_fast. MLX qmm templates still assume group size >= 32, so group16 Q6
  must route through qmv for now.
- The next 200 tok/s step is not more quant type coverage; it needs fewer layer
  launches or fused tiled kernels for Qwen3.5 GDR/MLP/full-attention projections.

