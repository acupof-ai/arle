# Metal GGUF native-q4 mode cleanup

## Goal

- Type: deletion-style refactor / regression check.
- Remove the unused Qwen3.5 Metal GGUF `higher` / `q5q6` native-q4 mode after
  local benchmarking showed it was not the winning path.
- Keep the runtime surface to two states only: exact GGUF by default, or
  explicit `AGENT_INFER_METAL_GGUF_NATIVE_Q4=all` for the lossy speed path.

## Hypothesis

Deleting the half-mode should not move decode throughput or correctness. It
should reduce the GGUF loader state surface and make future exact-kernel work
less ambiguous.

## Params

```bash
AGENT_INFER_METAL_GGUF_NATIVE_Q4=all ./target/release/metal_bench \
  --model models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt-tokens 128 \
  --generation-tokens 32 \
  --warmup 1 \
  --runs 3 \
  --json
```

## Env

- Backend: Metal / MLX bridge
- Model: `models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf`
- Hardware: Apple M4 Pro, 20 GPU cores, 48 GB unified memory, Metal 4
- Feature set: `--release --no-default-features --features metal,no-cuda`
- Non-default env vars: `AGENT_INFER_METAL_GGUF_NATIVE_Q4=all`

## Results

Raw JSON:

```json
{"avg_tokens":32,"generation_tokens_requested":32,"generation_tps":{"mean":235.87430868130141,"p50":235.0619352205448,"p99":238.3040200771137},"load_ms":2259.8826249999997,"model":"models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf","peak_rss_mb":1685.234375,"prompt_tokens":128,"prompt_tokens_requested":128,"prompt_tps":{"mean":3330.7791276555417,"p50":3376.3844330213033,"p99":3394.8614846121804},"quantization":"gguf-q4_k","repo_e2e_tps":{"mean":183.79312397301942,"p50":183.36795358498676,"p99":186.06134028402684},"timed_runs":3,"total_time_ms":{"mean":174.1237363333333,"p50":174.5125,"p99":175.87241699999998},"ttft_ms":{"mean":38.450833333333335,"p50":37.910375,"p99":39.738083},"warmup_runs":1}
```

Reference full profile:
[`2026-04-28-bench-metal-qwen35-0p8b-gguf-native-q4.md`](2026-04-28-bench-metal-qwen35-0p8b-gguf-native-q4.md)
recorded the same mode at 236.740 tok/s direct for `1024/256`.

## Problems

- This is not a new SOTA claim. It is a cleanup checkpoint that verifies the
  remaining `all` mode still lands in the same throughput band.
- The exact GGUF path remains the default because default-on native-q4 already
  failed the C++ generate vs Rust replay gate.

## Learnings

- Keep the Metal GGUF native-q4 surface binary. The intermediate `higher`
  mode added operator and code complexity without becoming a credible
  production path.

## Verification

```bash
cargo fmt --check
cargo check -p infer --release --no-default-features --features metal,no-cuda
cargo clippy -p infer --release --no-default-features --features metal,no-cuda -- -D warnings
cargo test -p infer --release --no-default-features --features metal,no-cuda \
  gguf_native_q4_requantizes_to_mlx_group64 -- --nocapture
QWEN35_MODEL_PATH=/path/to/code/agent-infer/models/Qwen3.5-0.8B \
QWEN35_GGUF_PATH=/path/to/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
  cargo test -p infer --release --no-default-features --features metal,no-cuda \
  compare_qwen35_0p8b_gguf_cpp_generate_vs_rust_replay -- --ignored --nocapture
```
