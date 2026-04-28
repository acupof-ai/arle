# Metal Qwen3.5 0.8B GGUF native-q4 opt-in load path

## Goal

- Type: optimization / format-gap reduction.
- Add an opt-in Qwen3.5-0.8B GGUF Q4_K_M load path that moves mixed
  K-quant affine layouts into MLX native affine q4 group64 weights.
- Keep exact GGUF affine/packed behavior as the default correctness path.

## Hypothesis

- The matched GGUF `1024/256` profile is slower than MLX SafeTensors 4bit
  because Q4_K_M mixes Q4_K/Q5_K/Q6_K/Q8_0 and especially Q6_K group16,
  which misses the fastest MLX 4bit group64 qmv path. Requantizing packed
  GGUF tensors to MLX native q4 group64 at load time should improve decode,
  at the cost of lossy double-quantization.

## Command

```bash
cargo build -p infer --release --no-default-features --features metal,no-cuda --bin metal_bench

AGENT_INFER_METAL_GGUF_NATIVE_Q4=all ./target/release/metal_bench \
  --model models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt-tokens 1024 \
  --generation-tokens 256 \
  --warmup 2 \
  --runs 5 \
  --json

AGENT_INFER_METAL_GGUF_NATIVE_Q4=all ./target/release/metal_bench \
  --model models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt-tokens 1024 \
  --generation-tokens 256 \
  --use-step-driver \
  --ignore-eos \
  --warmup 2 \
  --runs 5 \
  --json

./target/release/metal_bench \
  --model models/Qwen3.5-0.8B-MLX-4bit \
  --prompt-tokens 1024 \
  --generation-tokens 256 \
  --warmup 2 \
  --runs 5 \
  --json
```

Invoked via: local `metal_bench`, not `scripts/bench_guidellm.sh`, because this
isolates single-request direct/step-driver performance before HTTP serving.

## Environment

- Backend: Metal / MLX bridge
- Model: `models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf`
- Comparison model: `models/Qwen3.5-0.8B-MLX-4bit`
- Hardware: Apple M4 Pro, 20 GPU cores, 48 GB unified memory, Metal 4
- Feature set: `--release --no-default-features --features metal,no-cuda`
- Commit: this commit
- Non-default env vars for GGUF runs: `AGENT_INFER_METAL_GGUF_NATIVE_Q4=all`.
  Unset/default remains exact GGUF affine/packed behavior.

## Results

Raw JSON, GGUF direct with opt-in native-q4 load:

```json
{"avg_tokens":256,"generation_tokens_requested":256,"generation_tps":{"mean":236.74030821058454,"p50":237.291823639804,"p99":237.4637847437171},"load_ms":1904.017791,"model":"models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf","peak_rss_mb":1680.9375,"prompt_tokens":1024,"prompt_tokens_requested":1024,"prompt_tps":{"mean":4095.493458485819,"p50":4092.421864606275,"p99":4103.331290710511},"quantization":"gguf-q4_k","repo_e2e_tps":{"mean":192.2805794971719,"p50":192.61699436792802,"p99":192.77470118558338},"timed_runs":5,"total_time_ms":{"mean":1331.3968418,"p50":1329.062375,"p99":1335.798333},"ttft_ms":{"mean":250.03120840000003,"p50":250.218583,"p99":250.246167},"warmup_runs":2}
```

Raw JSON, GGUF step-driver with opt-in native-q4 load:

```json
{"avg_tokens":256,"generation_tokens_requested":256,"generation_tps":{"mean":239.80775791707157,"p50":240.55094623557153,"p99":242.06179912052085},"load_ms":1815.1651659999998,"mode":"step-driver","model":"models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf","peak_rss_mb":1678.546875,"prompt_tokens":1024,"prompt_tokens_requested":1024,"prompt_tps":{"mean":4282.407214337674,"p50":4280.547232849855,"p99":4298.449834588599},"quantization":"gguf-q4_k","repo_e2e_tps":{"mean":195.92025629504627,"p50":196.28880618580374,"p99":197.40863885203572},"timed_runs":5,"total_time_ms":{"mean":1306.7304084,"p50":1304.200708,"p99":1325.356459},"ttft_ms":{"mean":239.11955820000003,"p50":239.22175000000001,"p99":239.977417},"warmup_runs":2}
```

Raw JSON, MLX SafeTensors 4bit direct comparison on the same binary:

```json
{"avg_tokens":256,"generation_tokens_requested":256,"generation_tps":{"mean":300.0442667903134,"p50":299.58797864242445,"p99":301.5877980253511},"load_ms":286.791792,"model":"models/Qwen3.5-0.8B-MLX-4bit","peak_rss_mb":662.640625,"prompt_tokens":1024,"prompt_tokens_requested":1024,"prompt_tps":{"mean":4804.582403605626,"p50":4808.224686198424,"p99":4814.30657706446},"quantization":"4-bit","repo_e2e_tps":{"mean":240.0732056429434,"p50":239.82952430258848,"p99":241.1590519281409},"timed_runs":5,"total_time_ms":{"mean":1066.3515750000001,"p50":1067.4248750000002,"p99":1070.712125},"ttft_ms":{"mean":213.13073319999998,"p50":212.968417,"p99":213.948833},"warmup_runs":2}
```

## Delta vs baseline

Baseline:
[`2026-04-28-bench-metal-qwen35-0p8b-mlx4bit-qknorm-default.md`](2026-04-28-bench-metal-qwen35-0p8b-mlx4bit-qknorm-default.md)
recorded GGUF Q4_K_M direct `1024/256` at 202.076 tok/s mean / 202.620 p50.

| profile | baseline tok/s | now tok/s | delta |
|---|---:|---:|---:|
| GGUF Q4_K_M direct 1024/256, mean | 202.076 | 236.740 | +17.2% |
| GGUF Q4_K_M direct 1024/256, p50 | 202.620 | 237.292 | +17.1% |
| GGUF Q4_K_M step-driver 1024/256, mean | ~207.1 | 239.808 | +15.8% |
| MLX SafeTensors 4bit direct 1024/256, mean | 300.044 | 300.044 | comparison |

## Correctness / verification

```bash
cargo fmt --check
cargo check -p infer --release --no-default-features --features metal,no-cuda
cargo clippy -p infer --release --no-default-features --features metal,no-cuda -- -D warnings
cargo check -p infer --no-default-features --features cuda,no-cuda
cargo test -p infer --release --no-default-features --features metal,no-cuda
cargo test -p infer --release --no-default-features --features metal,no-cuda \
  gguf_native_q4_requantizes_to_mlx_group64 -- --nocapture
QWEN35_MODEL_PATH=/path/to/code/agent-infer/models/Qwen3.5-0.8B \
QWEN35_GGUF_PATH=/path/to/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
  cargo test -p infer --release --no-default-features --features metal,no-cuda \
  compare_qwen35_0p8b_gguf_cpp_generate_vs_rust_replay -- --ignored --nocapture
AGENT_INFER_METAL_GGUF_NATIVE_Q4=all ./target/release/metal_bench \
  --model models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt-tokens 16 \
  --generation-tokens 4 \
  --warmup 0 \
  --runs 1 \
  --json
```

## Problems

- This is a lossy speed path. It preserves the GGUF model architecture and
  tensor routing, but it requantizes already-quantized K-quant tensors to
  MLX q4 group64. Exact GGUF affine/packed behavior is the default; use
  `AGENT_INFER_METAL_GGUF_NATIVE_Q4=all` only for this speed/quality tradeoff.
- Default-on native-q4 was rejected after the GGUF C++ generate vs Rust replay
  gate diverged. The default path stays exact, and this load path remains an
  explicit optimization knob until it has a separate quality acceptance gate.
- It narrows the GGUF gap but does not close SOTA: current GGUF direct is
  236.7 tok/s versus 300.0 tok/s for MLX SafeTensors 4bit direct and 305.5
  tok/s for the latest MLX SafeTensors 4bit step-driver run.
- Tried but did not keep: value-column affine repack for grouped out-proj
  regressed to 197 tok/s, dense GGUF embedding did not improve decode, forced
  merged GDR projections regressed, and explicit `contiguous()` on quantize
  outputs did not beat the default.

## Learnings

- On Metal, exact K-quant preservation is not the same as fastest inference.
  The MLX native q4 group64 layout is a better decode target than exact
  Q6_K group16 / Q5_K group32 affine for this single-request workload.
- GGUF Q4_K_M remains a separate format-quality/speed tradeoff from MLX
  SafeTensors 4bit. Further SOTA work needs a weight-cache format that stores
  the native-q4 conversion once, or a dedicated exact K-quant qmv kernel that
  can match MLX q4 group64 throughput without lossy requantization.

## Artefacts

- `bench-output/2026-04-28-metal-gguf-native-q4/direct-1024-256-current.json`
- `bench-output/2026-04-28-metal-gguf-native-q4/step-driver-1024-256-current.json`
- `bench-output/2026-04-28-metal-gguf-native-q4/mlx4-direct-1024-256.json`
