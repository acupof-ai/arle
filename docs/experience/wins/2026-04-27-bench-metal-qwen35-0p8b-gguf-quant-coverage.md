# Metal GGUF Quant Coverage on Qwen3.5 0.8B

## Goal

- Type: regression check.
- Verify the real packed GGUF quant path after adding Metal Q3_K support,
  keeping grouped Qwen3.5 V-row projections packed, and unifying CUDA
  safetensors quant loader metadata.

## Hypothesis

- Q4_K_M should keep the memory win and decode-speed advantage against BF16.
- Preserving more true packed projections may trade some prompt speed for
  correctness of the quantized-weight execution path.

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

./target/release/metal_bench \
  --model /path/to/code/agent-infer/models/Qwen3.5-0.8B \
  --prompt-tokens 512 \
  --generation-tokens 1024 \
  --warmup 0 \
  --runs 1 \
  --ignore-eos \
  --json
```

Correctness checks:

```bash
cargo test -p infer --release --no-default-features --features metal,no-cuda \
  gguf_quantized_matmul_matches_cpu_reference_for_all_metal_packed_formats -- --nocapture

QWEN35_MODEL_PATH=/path/to/code/agent-infer/models/Qwen3.5-0.8B \
QWEN35_GGUF_PATH=/path/to/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
cargo test -p infer --release --no-default-features --features metal,no-cuda \
  compare_qwen35_0p8b_gguf_cpp_gdr_kernel_vs_fallback_prefill -- --ignored --nocapture

QWEN35_MODEL_PATH=/path/to/code/agent-infer/models/Qwen3.5-0.8B \
QWEN35_GGUF_PATH=/path/to/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
cargo test -p infer --release --no-default-features --features metal,no-cuda \
  compare_qwen35_0p8b_gguf_cpp_generate_vs_rust_replay -- --ignored --nocapture
```

## Environment

- Backend: Metal
- Hardware: Apple M4 Pro, 20-core GPU, unified memory
- OS / Metal: macOS 26.3.1 (25D771280a), Metal 4
- Model: Qwen3.5-0.8B BF16 safetensors vs Qwen3.5-0.8B GGUF Q4_K_M
- Feature set: `--release --no-default-features --features metal,no-cuda`
- Non-default flags: `--ignore-eos`
- Commit: working tree based on `d08e1f2`; this entry is committed with the
  code change

## Results

Raw JSON:

```json
{"avg_tokens":1024,"generation_tokens_requested":1024,"generation_tps":{"mean":30.01767110837447,"p50":30.01767110837447,"p99":30.01767110837447},"load_ms":676.851459,"model":"models/Qwen3.5-0.8B","peak_rss_mb":1605.28125,"prompt_tokens":512,"prompt_tokens_requested":512,"prompt_tps":{"mean":1483.3347571801862,"p50":1483.3347571801862,"p99":1483.3347571801862},"quantization":"bf16","repo_e2e_tps":{"mean":29.716985543614868,"p50":29.716985543614868,"p99":29.716985543614868},"timed_runs":1,"total_time_ms":{"mean":34458.407583,"p50":34458.407583,"p99":34458.407583},"ttft_ms":{"mean":345.168208,"p50":345.168208,"p99":345.168208},"warmup_runs":0}
{"avg_tokens":1024,"generation_tokens_requested":1024,"generation_tps":{"mean":38.56588337389695,"p50":38.56588337389695,"p99":38.56588337389695},"load_ms":1262.322792,"model":"models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf","peak_rss_mb":905.84375,"prompt_tokens":512,"prompt_tokens_requested":512,"prompt_tps":{"mean":186.53252761885582,"p50":186.53252761885582,"p99":186.53252761885582},"quantization":"gguf-q4_k","repo_e2e_tps":{"mean":34.95262882741987,"p50":34.95262882741987,"p99":34.95262882741987},"timed_runs":1,"total_time_ms":{"mean":29296.794958,"p50":29296.794958,"p99":29296.794958},"ttft_ms":{"mean":2744.829583,"p50":2744.829583,"p99":2744.829583},"warmup_runs":0}
```

512 prompt / 1024 decode:

| Metric | BF16 safetensors | GGUF Q4_K_M | Delta |
|---|---:|---:|---:|
| Peak RSS | 1605.3 MB | 905.8 MB | -43.6% |
| Prompt throughput mean | 1483.33 tok/s | 186.53 tok/s | -87.4% |
| Decode throughput mean | 30.02 tok/s | 38.57 tok/s | +28.5% |
| Repo E2E throughput mean | 29.72 tok/s | 34.95 tok/s | +17.6% |
| TTFT mean | 345.2 ms | 2744.8 ms | +695.2% |
| Total time mean | 34458.4 ms | 29296.8 ms | -15.0% |

## Localization

`AGENT_INFER_QWEN35_GENERATE_PROFILE=1` on 32 prompt / 32 decode showed:

| Run | Decode tok/s | Avg decode step | Main finding |
|---|---:|---:|---|
| BF16 true lm_head | 28.7 tok/s | 34.8 ms | MLX execution dominates; graph build is ~1.6 ms/token |
| BF16 fake lm_head diagnostic | 40.1 tok/s | 24.9 ms | lm_head costs ~10 ms/token, but layers remain the larger issue |
| Q4 before packed out_proj fix | 29.7 tok/s | 33.7 ms | GGUF out_proj BF16 fallback erased much of Q4's decode win |
| Q4 after packed out_proj fix | 39.6 tok/s | 25.3 ms | Q4 is now faster, but still far below optimized MLX runtimes |

Correctness:

| Check | Result |
|---|---|
| Metal GGUF packed matmul Q8_0/Q3_K/Q4_K/Q5_K/Q6_K vs CPU dequant reference | passed |
| Qwen3.5 grouped V packed row reorder fixture | passed |
| GGUF C++ custom GDR prefill vs C++ fallback | passed; max logits 0.034760, max KV 0.085938, max GDR 0.040545 |
| GGUF C++ generate vs Rust replay | passed multi-token prompt and 4 decode tokens |

## Problems

- Q4_K_M remains far below the public oMLX expectation for this model class
  (~305 tok/s TG at 1k context on M4 Pro 16-core GPU, 2026-03-25 benchmark).
- The remaining decode gap is in Qwen3.5's Metal GDR/MLP/lm_head execution:
  `async_eval(next_y)` is ~21-22 ms/token on Q4 after the out_proj fix, while
  graph construction is only ~3 ms/token.
- The current Metal GGUF matmul still performs scalar unpack-and-dot work
  instead of a repacked/tiled kernel aligned with Apple GPU SIMD groups.
- CUDA runtime performance was not measured on this Apple machine. The CUDA
  path was checked with `cargo check -p infer --release --no-default-features
  --features cuda,no-cuda`.
- Metal does not publish quantized KV cache today; the inspected KV quant
  paths are CUDA-only and remain orthogonal to weight quantization.

## Learnings

- GGUF support should be explicit per executable format: parse-only types
  such as Q2_K/Q8_K now fail in the dequant path instead of silently looking
  runnable.
- Weight quant metadata must be loaded once and propagated uniformly; hard
  coding TurboQuant as TQ3 makes TQ2/TQ4 checkpoints unsafe.
- Reordering the small activation vector is the right fix for Qwen3.5 grouped
  value-head column order: it preserves true packed GGUF weights and avoids a
  BF16 dense out_proj fallback.
- For long decode workloads, Q4_K_M remains useful: it used 43.6% less RSS and
  delivered 17.6% higher end-to-end throughput than BF16 in this run.

## Delta vs Baseline

Baseline entry:
[`2026-04-27-bench-metal-qwen35-0p8b-gguf-gdr-m4.md`](2026-04-27-bench-metal-qwen35-0p8b-gguf-gdr-m4.md).

| Workload | Metric | Prior Q4 | Current Q4 | Delta |
|---|---|---:|---:|---:|
| 512/1024 | Peak RSS | 907.8 MB | 905.8 MB | -0.2% |
| 512/1024 | Prompt throughput | 186.43 tok/s | 186.53 tok/s | +0.1% |
| 512/1024 | Decode throughput | 38.78 tok/s | 38.57 tok/s | -0.5% |
| 512/1024 | Repo E2E throughput | 35.13 tok/s | 34.95 tok/s | -0.5% |
| 512/1024 | TTFT | 2746.4 ms | 2744.8 ms | -0.1% |
| 512/1024 | Total time | 29149.2 ms | 29296.8 ms | +0.5% |

## Artefacts

- Raw: inline JSON above
- External reference: https://omlx.ai/benchmarks/ti2qtqq6
- CSV / HTML: n/a for local helper benchmark
- Service trace: n/a; no HTTP server
