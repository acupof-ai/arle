# Metal Qwen3.5 0.8B GGUF GDR Alignment + M4 Matmul

## Goal

- Type: optimization.
- Verify true GGUF Q4_K_M weight inference after re-enabling the Qwen3.5
  custom GDR Metal kernel and adding a prompt-batch GGUF matmul tile.

## Hypothesis

- Aligning C++ and Rust GDR custom-kernel inputs to compact bf16 tensors should
  restore GGUF replay correctness without falling back to the per-op GDR path.
- Reusing GGUF Q4 unpack work across 4 prompt rows should materially reduce
  long-prompt TTFT while preserving the existing faster Q4 decode.

## Command

```bash
./target/release/metal_bench \
  --model /path/to/code/agent-infer/models/Qwen3.5-0.8B \
  --prompt-tokens 2048 \
  --generation-tokens 256 \
  --warmup 0 \
  --runs 1 \
  --use-step-driver \
  --ignore-eos \
  --json

./target/release/metal_bench \
  --model /path/to/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt-tokens 2048 \
  --generation-tokens 256 \
  --warmup 0 \
  --runs 1 \
  --use-step-driver \
  --ignore-eos \
  --json

./target/release/metal_bench \
  --model /path/to/code/agent-infer/models/Qwen3.5-0.8B \
  --prompt-tokens 512 \
  --generation-tokens 1024 \
  --warmup 0 \
  --runs 1 \
  --use-step-driver \
  --ignore-eos \
  --json

./target/release/metal_bench \
  --model /path/to/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt-tokens 512 \
  --generation-tokens 1024 \
  --warmup 0 \
  --runs 1 \
  --use-step-driver \
  --ignore-eos \
  --json
```

Correctness checks:

```bash
QWEN35_MODEL_PATH=/path/to/code/agent-infer/models/Qwen3.5-0.8B \
QWEN35_GGUF_PATH=/path/to/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
cargo test --release -p infer --no-default-features --features metal,no-cuda \
  compare_qwen35_0p8b_gguf_cpp_generate_vs_rust_replay -- --ignored --nocapture

METAL_NO_CPP=1 \
QWEN35_MODEL_PATH=/path/to/code/agent-infer/models/Qwen3.5-0.8B \
QWEN35_GGUF_PATH=/path/to/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
cargo test --release -p infer --no-default-features --features metal,no-cuda \
  compare_qwen35_0p8b_gguf_incremental_decode_vs_full_replay -- --ignored --nocapture
```

## Environment

- Backend: Metal
- Hardware: Apple M4 Pro, 20-core GPU, 48 GB unified memory
- OS / Metal: macOS 26.3.1 (25D771280a), Metal 4
- Model: Qwen3.5-0.8B safetensors vs Qwen3.5-0.8B GGUF Q4_K_M
- Feature set: `--release --no-default-features --features metal,no-cuda`
- Non-default flags: none for primary bench runs; `METAL_NO_CPP=1` for the
  Rust-only correctness check
- Commit: working tree based on `7483a3d`; this entry is committed with the
  code change

## Results

Raw JSON:

```json
{"avg_tokens":256,"generation_tokens_requested":256,"generation_tps":{"mean":29.956755201683627,"p50":29.956755201683627,"p99":29.956755201683627},"load_ms":675.577625,"mode":"step-driver","model":"/path/to/code/agent-infer/models/Qwen3.5-0.8B","peak_rss_mb":1594.015625,"prompt_tokens":2048,"prompt_tokens_requested":2048,"prompt_tps":{"mean":2657.3621518393816,"p50":2657.3621518393816,"p99":2657.3621518393816},"quantization":"bf16","repo_e2e_tps":{"mean":27.47859989985875,"p50":27.47859989985875,"p99":27.47859989985875},"timed_runs":1,"total_time_ms":{"mean":9316.34075,"p50":9316.34075,"p99":9316.34075},"ttft_ms":{"mean":770.6890830000001,"p50":770.6890830000001,"p99":770.6890830000001},"warmup_runs":0}
{"avg_tokens":256,"generation_tokens_requested":256,"generation_tps":{"mean":38.5268375730726,"p50":38.5268375730726,"p99":38.5268375730726},"load_ms":1194.461708,"mode":"step-driver","model":"/path/to/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf","peak_rss_mb":903.40625,"prompt_tokens":2048,"prompt_tokens_requested":2048,"prompt_tps":{"mean":184.14617636773127,"p50":184.14617636773127,"p99":184.14617636773127},"quantization":"gguf-q4_k","repo_e2e_tps":{"mean":14.40928830604287,"p50":14.40928830604287,"p99":14.40928830604287},"timed_runs":1,"total_time_ms":{"mean":17766.318125,"p50":17766.318125,"p99":17766.318125},"ttft_ms":{"mean":11121.599375,"p50":11121.599375,"p99":11121.599375},"warmup_runs":0}
{"avg_tokens":1024,"generation_tokens_requested":1024,"generation_tps":{"mean":30.60291493094057,"p50":30.60291493094057,"p99":30.60291493094057},"load_ms":381.397166,"mode":"step-driver","model":"/path/to/code/agent-infer/models/Qwen3.5-0.8B","peak_rss_mb":1594.0625,"prompt_tokens":512,"prompt_tokens_requested":512,"prompt_tps":{"mean":2947.674460444598,"p50":2947.674460444598,"p99":2947.674460444598},"quantization":"bf16","repo_e2e_tps":{"mean":30.44487525625064,"p50":30.44487525625064,"p99":30.44487525625064},"timed_runs":1,"total_time_ms":{"mean":33634.560542,"p50":33634.560542,"p99":33634.560542},"ttft_ms":{"mean":173.69625,"p50":173.69625,"p99":173.69625},"warmup_runs":0}
{"avg_tokens":1024,"generation_tokens_requested":1024,"generation_tps":{"mean":38.78378157810975,"p50":38.78378157810975,"p99":38.78378157810975},"load_ms":1177.64175,"mode":"step-driver","model":"/path/to/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf","peak_rss_mb":907.796875,"prompt_tokens":512,"prompt_tokens_requested":512,"prompt_tps":{"mean":186.42746463565848,"p50":186.42746463565848,"p99":186.42746463565848},"quantization":"gguf-q4_k","repo_e2e_tps":{"mean":35.129650674749,"p50":35.129650674749,"p99":35.129650674749},"timed_runs":1,"total_time_ms":{"mean":29149.165458,"p50":29149.165458,"p99":29149.165458},"ttft_ms":{"mean":2746.3764579999997,"p50":2746.3764579999997,"p99":2746.3764579999997},"warmup_runs":0}
```

Long prompt, 2048/256:

| Metric | BF16 safetensors | GGUF Q4_K_M | Delta |
|---|---:|---:|---:|
| Peak RSS | 1594.0 MB | 903.4 MB | -43.3% |
| Prompt throughput mean | 2657.36 tok/s | 184.15 tok/s | -93.1% |
| Decode throughput mean | 29.96 tok/s | 38.53 tok/s | +28.6% |
| Repo E2E throughput mean | 27.48 tok/s | 14.41 tok/s | -47.6% |
| TTFT mean | 770.7 ms | 11121.6 ms | +1343.1% |
| Total time mean | 9316.3 ms | 17766.3 ms | +90.7% |

Long output, 512/1024:

| Metric | BF16 safetensors | GGUF Q4_K_M | Delta |
|---|---:|---:|---:|
| Peak RSS | 1594.1 MB | 907.8 MB | -43.1% |
| Prompt throughput mean | 2947.67 tok/s | 186.43 tok/s | -93.7% |
| Decode throughput mean | 30.60 tok/s | 38.78 tok/s | +26.7% |
| Repo E2E throughput mean | 30.44 tok/s | 35.13 tok/s | +15.4% |
| TTFT mean | 173.7 ms | 2746.4 ms | +1481.1% |
| Total time mean | 33634.6 ms | 29149.2 ms | -13.3% |

Correctness:

| Check | Result |
|---|---|
| GGUF C++ custom GDR prefill vs C++ fallback | passed; max logits 0.034760, max KV 0.085938, max GDR 0.040545 |
| GGUF C++ generate vs Rust replay | passed multi-token prompt and 4 decode tokens |
| GGUF Rust-only replay with `METAL_NO_CPP=1` | passed 4 incremental-vs-full replay decode steps |
| Grouped GDR kernel vs CPU reference, `Hk=2`, `Hv=4`, `Dk=128`, `Dv=128` | passed |

## Problems

- Long-prompt Q4 is still prefill-bound and slower than BF16 because GGUF Q4
  unpacked matmul is not Tensor-Core/tile-MMA aligned. The M4 kernel reuses
  unpacked weights across four prompt rows but still computes scalar dot
  reductions.
- The local helper benchmark is single-run with no HTTP service stats; it is
  appropriate for this Metal kernel diagnosis but not a replacement for a
  canonical GuideLLM service sweep.
- CUDA runtime performance was not measured locally on this Apple machine;
  CUDA compile/typecheck was run with `cuda,no-cuda`.

## Learnings

- Raw Metal kernels must receive compact bf16 inputs when they do manual
  pointer arithmetic. Lazy GGUF projection views and f32 gate tensors can pass
  shape checks while producing incorrect recurrent state.
- GGUF Q4 can be faster end-to-end when decode dominates: on 512/1024, Q4 is
  +15.4% E2E and -43.1% RSS vs BF16.
- For prompt-heavy workloads, Q4 speed depends on a batch-aware dequant/matmul
  kernel. Reusing one Q4 unpack across four prompt rows lifted Q4 prompt
  throughput from roughly 53 tok/s to 184-186 tok/s, but BF16 dense matmul is
  still much faster.

## Delta vs Baseline

Baseline long-sequence entry:
[`2026-04-26-bench-metal-qwen35-0p8b-long-gguf-q4.md`](2026-04-26-bench-metal-qwen35-0p8b-long-gguf-q4.md).

| Workload | Metric | Prior Q4 | Current Q4 | Delta |
|---|---|---:|---:|---:|
| 2048/256 | Prompt throughput | 53.67 tok/s | 184.15 tok/s | +243.1% |
| 2048/256 | TTFT | 38160.7 ms | 11121.6 ms | -70.9% |
| 2048/256 | Decode throughput | 37.58 tok/s | 38.53 tok/s | +2.5% |
| 2048/256 | Repo E2E throughput | 5.69 tok/s | 14.41 tok/s | +153.1% |
| 2048/256 | Peak RSS | 1072.2 MB | 903.4 MB | -15.7% |
| 512/1024 | Prompt throughput | 53.19 tok/s | 186.43 tok/s | +250.5% |
| 512/1024 | TTFT | 9626.5 ms | 2746.4 ms | -71.5% |
| 512/1024 | Decode throughput | 37.78 tok/s | 38.78 tok/s | +2.7% |
| 512/1024 | Repo E2E throughput | 27.88 tok/s | 35.13 tok/s | +26.0% |
| 512/1024 | Peak RSS | 1074.5 MB | 907.8 MB | -15.5% |

## Artefacts

- Raw: inline JSON above
- CSV / HTML: n/a for local helper benchmark
- Service trace: n/a; no HTTP server

