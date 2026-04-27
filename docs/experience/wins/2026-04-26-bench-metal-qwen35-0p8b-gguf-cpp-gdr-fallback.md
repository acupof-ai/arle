# Metal Qwen3.5 0.8B GGUF C++ GDR Fallback

## Goal

- Type: regression.
- Verify the Qwen3.5-0.8B GGUF/Q4_K_M path after reconnecting packed GGUF
  weights to the Metal C++ compiled model while explicitly disabling the
  unstable custom GDR recurrent Metal kernel for GGUF.

## Hypothesis

- Packed Q4 should remain faster than BF16 in memory-bound decode and much
  smaller in resident memory.
- Prompt/TTFT will still trail BF16 until the custom GDR recurrent kernel is
  corrected for GGUF and multi-token prefill.

## Params

```bash
./target/release/metal_bench \
  --model models/Qwen3.5-0.8B \
  --prompt-tokens 32 \
  --generation-tokens 32 \
  --warmup 0 \
  --runs 1 \
  --use-step-driver \
  --ignore-eos \
  --json

./target/release/metal_bench \
  --model models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt-tokens 32 \
  --generation-tokens 32 \
  --warmup 0 \
  --runs 1 \
  --use-step-driver \
  --ignore-eos \
  --json
```

Secondary C++ generate path check:

```bash
AGENT_INFER_GDR_METAL_KERNEL=0 ./target/release/metal_bench \
  --model models/Qwen3.5-0.8B \
  --prompt-tokens 32 --generation-tokens 32 --warmup 0 --runs 1 \
  --ignore-eos --json

./target/release/metal_bench \
  --model models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt-tokens 32 --generation-tokens 32 --warmup 0 --runs 1 \
  --ignore-eos --json
```

## Env

- Backend: Metal
- Hardware: Apple M4 Pro, 20-core GPU, 48 GB unified memory
- OS / Metal: macOS 26.3.1 (25D771280a), Metal 4
- Model: Qwen3.5-0.8B safetensors vs Qwen3.5-0.8B GGUF Q4_K_M
- Feature set: `--release --no-default-features --features metal,no-cuda`
- Non-default flags: none for primary step-driver runs
- Commit: `a0f6548`

## Results

Primary raw JSON:

```json
{"avg_tokens":32,"generation_tokens_requested":32,"generation_tps":{"mean":29.976971578020123,"p50":29.976971578020123,"p99":29.976971578020123},"load_ms":374.22054199999997,"mode":"step-driver","model":"models/Qwen3.5-0.8B","peak_rss_mb":1601.21875,"prompt_tokens":32,"prompt_tokens_requested":32,"prompt_tps":{"mean":353.21517925013586,"p50":353.21517925013586,"p99":353.21517925013586},"quantization":"bf16","repo_e2e_tps":{"mean":27.631883877477748,"p50":27.631883877477748,"p99":27.631883877477748},"timed_runs":1,"total_time_ms":{"mean":1158.082458,"p50":1158.082458,"p99":1158.082458},"ttft_ms":{"mean":90.596333,"p50":90.596333,"p99":90.596333},"warmup_runs":0}
{"avg_tokens":32,"generation_tokens_requested":32,"generation_tps":{"mean":37.229470092301455,"p50":37.229470092301455,"p99":37.229470092301455},"load_ms":1807.72,"mode":"step-driver","model":"models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf","peak_rss_mb":912.171875,"prompt_tokens":32,"prompt_tokens_requested":32,"prompt_tps":{"mean":35.91127166442151,"p50":35.91127166442151,"p99":35.91127166442151},"quantization":"gguf-q4_k","repo_e2e_tps":{"mean":18.279246043935597,"p50":18.279246043935597,"p99":18.279246043935597},"timed_runs":1,"total_time_ms":{"mean":1750.61925,"p50":1750.61925,"p99":1750.61925},"ttft_ms":{"mean":891.0851250000001,"p50":891.0851250000001,"p99":891.0851250000001},"warmup_runs":0}
```

| Metric | BF16 safetensors | GGUF Q4_K_M | Delta |
|---|---:|---:|---:|
| Load time | 374.2 ms | 1807.7 ms | +383.1% |
| Peak RSS | 1601.2 MB | 912.2 MB | -43.0% |
| Prompt throughput mean | 353.22 tok/s | 35.91 tok/s | -89.8% |
| Decode throughput mean | 29.98 tok/s | 37.23 tok/s | +24.2% |
| Repo E2E throughput mean | 27.63 tok/s | 18.28 tok/s | -33.8% |
| TTFT mean | 90.6 ms | 891.1 ms | +883.6% |
| Total time mean | 1158.1 ms | 1750.6 ms | +51.2% |

Secondary C++ generate raw JSON:

```json
{"avg_tokens":32,"generation_tokens_requested":32,"generation_tps":{"mean":22.701834596052056,"p50":22.701834596052056,"p99":22.701834596052056},"load_ms":973.2075,"model":"models/Qwen3.5-0.8B","peak_rss_mb":1603.90625,"prompt_tokens":32,"prompt_tokens_requested":32,"prompt_tps":{"mean":91.94539227533315,"p50":91.94539227533315,"p99":91.94539227533315},"quantization":"bf16","repo_e2e_tps":{"mean":18.206537953555262,"p50":18.206537953555262,"p99":18.206537953555262},"timed_runs":1,"total_time_ms":{"mean":1757.610375,"p50":1757.610375,"p99":1757.610375},"ttft_ms":{"mean":348.032666,"p50":348.032666,"p99":348.032666},"warmup_runs":0}
{"avg_tokens":32,"generation_tokens_requested":32,"generation_tps":{"mean":37.823967437573366,"p50":37.823967437573366,"p99":37.823967437573366},"load_ms":2140.3842910000003,"model":"models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf","peak_rss_mb":911.734375,"prompt_tokens":32,"prompt_tokens_requested":32,"prompt_tps":{"mean":25.071455273255957,"p50":25.071455273255957,"p99":25.071455273255957},"quantization":"gguf-q4_k","repo_e2e_tps":{"mean":15.07743913620175,"p50":15.07743913620175,"p99":15.07743913620175},"timed_runs":1,"total_time_ms":{"mean":2122.376334,"p50":2122.376334,"p99":2122.376334},"ttft_ms":{"mean":1276.351917,"p50":1276.351917,"p99":1276.351917},"warmup_runs":0}
```

Correctness checks paired with this run:

| Check | Result |
|---|---|
| `compare_qwen35_0p8b_gguf_cpp_generate_vs_rust_replay` | passed multi-token prompt and 4 decode tokens |
| `compare_qwen35_0p8b_gguf_incremental_decode_vs_full_replay` | passed 4 decode steps |
| `cargo test --release -p infer --no-default-features --features no-cuda` | 423 passed, 6 ignored |

## Problems

- GGUF Q4 is faster than BF16 for decode only. Prompt and TTFT remain slower
  because GGUF deliberately uses the safe C++ ops fallback for GDR recurrent
  state instead of the custom recurrent Metal kernel.
- The C++ ops fallback was decode-only before this change; multi-token prefill
  now runs a sequential state update loop.
- C++ generate with GGUF is still slower end-to-end than the step-driver
  helper on this 32/32 profile.

## Learnings

- Kernel alignment has to be correctness-gated. The custom GDR recurrent Metal
  kernel can be faster, but GGUF now opts into C++ compiled dispatch while
  explicitly disabling that kernel until it passes replay correctness.
- Q4 speed depends on workload shape. For Qwen3.5-0.8B on M4 Pro, decode is
  memory-bound enough to show a +24.2% throughput gain, while prompt remains
  dominated by recurrent fallback and packed GGUF GEMM overhead.
- A multi-token replay test is required; single-token prompts would not catch
  the C++ fallback reshape bug.

## Delta vs Prior Packed GGUF

Prior local packed GGUF entry:
[`2026-04-26-bench-metal-qwen35-0p8b-packed-gguf-local.md`](2026-04-26-bench-metal-qwen35-0p8b-packed-gguf-local.md).

| Metric | Prior packed GGUF | Current GGUF C++ fallback | Delta |
|---|---:|---:|---:|
| Decode throughput mean | 37.51 tok/s | 37.23 tok/s | -0.7% |
| Prompt throughput mean | 36.35 tok/s | 35.91 tok/s | -1.2% |
| Peak RSS | 896.5 MB | 912.2 MB | +1.8% |
| TTFT mean | 880.4 ms | 891.1 ms | +1.2% |

## Artefacts

- Raw: inline JSON above
- CSV / HTML: n/a for local helper benchmark
- Service trace: n/a; no HTTP server
