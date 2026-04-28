# Metal Qwen3.5 0.8B MLX 4bit Q/K norm default

## Goal

- Type: optimization / SOTA calibration.
- Bring the serving-equivalent Metal Qwen3.5-0.8B MLX 4bit single-request path
  to the Apple-native public baseline for M4 Pro.

## Hypothesis

- The C++ Q/K norm helper reduces graph nodes, but it may block MLX from using
  the faster native `fast::rms_norm(...) * scale` lowering. Making the helper
  opt-in should improve decode without affecting correctness.

## Command

```bash
cargo build -p infer --release --no-default-features --features metal,no-cuda --bin metal_bench

./target/release/metal_bench \
  --model /path/to/code/agent-infer/models/Qwen3.5-0.8B-MLX-4bit \
  --prompt-tokens 1024 \
  --generation-tokens 256 \
  --warmup 1 \
  --runs 5 \
  --ignore-eos \
  --use-step-driver \
  --json

./target/release/metal_bench \
  --model /path/to/code/agent-infer/models/Qwen3.5-0.8B-MLX-4bit \
  --prompt-tokens 1024 \
  --generation-tokens 256 \
  --warmup 1 \
  --runs 5 \
  --ignore-eos \
  --json

./target/release/metal_bench \
  --model /path/to/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt-tokens 1024 \
  --generation-tokens 256 \
  --warmup 1 \
  --runs 5 \
  --ignore-eos \
  --json
```

Invoked via: local `metal_bench`, not `scripts/bench_guidellm.sh`, because this
isolates single-request direct/step-driver performance before HTTP serving.

## Environment

- Backend: Metal / MLX bridge
- Model: `mlx-community/Qwen3.5-0.8B-MLX-4bit` downloaded to
  `models/Qwen3.5-0.8B-MLX-4bit`
- Regression model: `models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf`
- Hardware: Apple M4 Pro, 20 GPU cores, 48 GB unified memory, Metal 4
- OS: macOS 26.3.1 `(25D771280a)`, Darwin 25.3.0
- Commit before change: `a3ea6f3`
- Feature set: `--release --no-default-features --features metal,no-cuda`
- Non-default bench flags: `--ignore-eos`; step-driver run used
  `--use-step-driver`

## Results

Raw JSON, serving-equivalent step-driver after the default change:

```json
{"avg_tokens":256,"generation_tokens_requested":256,"generation_tps":{"mean":305.49496285896436,"p50":304.6576621532344,"p99":314.9274858181695},"load_ms":365.120375,"mode":"step-driver","model":"/path/to/code/agent-infer/models/Qwen3.5-0.8B-MLX-4bit","peak_rss_mb":652.34375,"prompt_tokens":1024,"prompt_tokens_requested":1024,"prompt_tps":{"mean":4965.36072702682,"p50":4989.111702206081,"p99":5025.537706646221},"quantization":"4-bit","repo_e2e_tps":{"mean":245.1379456929116,"p50":243.82100847565448,"p99":251.44079183270568},"timed_runs":5,"total_time_ms":{"mean":1044.5540334000002,"p50":1049.950542,"p99":1059.853959},"ttft_ms":{"mean":206.25005819999996,"p50":205.246958,"p99":209.66341699999998},"warmup_runs":1}
```

Raw JSON, direct generate after the default change:

```json
{"avg_tokens":256,"generation_tokens_requested":256,"generation_tps":{"mean":289.59411310871366,"p50":292.493277376216,"p99":299.0685359744571},"load_ms":336.164792,"model":"/path/to/code/agent-infer/models/Qwen3.5-0.8B-MLX-4bit","peak_rss_mb":662.640625,"prompt_tokens":1024,"prompt_tokens_requested":1024,"prompt_tps":{"mean":4559.867884744633,"p50":4570.524852159701,"p99":4655.725257650518},"quantization":"4-bit","repo_e2e_tps":{"mean":230.8936910510632,"p50":233.28149314327467,"p99":237.93251522487182},"timed_runs":5,"total_time_ms":{"mean":1109.4389328,"p50":1097.3866659999999,"p99":1158.619791},"ttft_ms":{"mean":224.6196246,"p50":224.044291,"p99":228.548791},"warmup_runs":1}
```

Raw JSON, GGUF regression check after the default change:

```json
{"avg_tokens":256,"generation_tokens_requested":256,"generation_tps":{"mean":202.07614806500368,"p50":202.62043900768498,"p99":204.2996361954978},"load_ms":2124.631875,"model":"/path/to/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf","peak_rss_mb":1428.71875,"prompt_tokens":1024,"prompt_tokens_requested":1024,"prompt_tps":{"mean":4247.145286353959,"p50":4262.697030057426,"p99":4275.162072792792},"quantization":"gguf-q4_k","repo_e2e_tps":{"mean":169.7634234338299,"p50":170.2819854713215,"p99":171.26461331536424},"timed_runs":5,"total_time_ms":{"mean":1508.0780247999999,"p50":1503.388625,"p99":1525.627958},"ttft_ms":{"mean":241.11505000000002,"p50":240.2235,"p99":244.184625},"warmup_runs":1}
```

## Delta vs baseline

Baselines:

- Prior MLX 4bit default step-driver run before changing the default:
  `bench-output/2026-04-28-metal-single-sota-diagnosis/mlx4bit-step-driver-1024p-256d.jsonl`
- Public M4 Pro references:
  - oMLX 20 GPU cores, 1k context: `299.5` tok/s in
    <https://omlx.ai/benchmarks/d85rndz7>, and `307.1` tok/s in
    <https://omlx.ai/benchmarks/cl2qtokd>
  - oMLX 16 GPU cores, 1k context: `320.7` tok/s in
    <https://omlx.ai/benchmarks/gfh40662>

| profile | baseline tok/s | now tok/s | delta |
|---|---:|---:|---:|
| MLX 4bit step-driver 1024/256, mean | 258.182 | 305.495 | +18.3% |
| MLX 4bit step-driver 1024/256, p50 | 291.049 | 304.658 | +4.7% |
| MLX 4bit direct 1024/256, mean | 254.457 | 289.594 | +13.8% |
| GGUF Q4_K_M direct 1024/256, mean | 200.798 | 202.076 | +0.6% |

## Correctness / verification

```bash
cargo check -p infer --release --no-default-features --features metal,no-cuda

QWEN35_MODEL_PATH=/path/to/code/agent-infer/models/Qwen3.5-0.8B \
QWEN35_GGUF_PATH=/path/to/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
cargo test -p infer --release --no-default-features --features metal,no-cuda \
  compare_qwen35_0p8b_gguf_cpp_generate_vs_rust_replay -- --ignored --nocapture

cargo clippy -p infer --release --no-default-features --features metal,no-cuda -- -D warnings
cargo fmt --check
```

| Check | Result |
|---|---|
| Metal release check | passed |
| GGUF C++ generate vs Rust replay | passed; replay tokens `!`, `!`, `!`, `!` |
| Metal release clippy | passed |
| fmt | passed |

## Problems

- This closes the single-request MLX 4bit SOTA gap, not the GGUF gap.
  GGUF Q4_K_M remains around 202 tok/s on the same 1024/256 profile.
- This is a local helper benchmark. A canonical GuideLLM HTTP sweep is still
  required before claiming serving-concurrency SOTA.
- The improvement is shape/model-specific evidence for Qwen3.5 MLX 4bit. Do
  not assume it improves every Qwen3.5 GGUF or Qwen3.6 MoE profile.

## Learnings

- Fewer graph nodes are not automatically faster on MLX/Metal. The custom Q/K
  norm helper looked like a fusion but was slower than native MLX lowering for
  this Qwen3.5 single-request decode path.
- Use MLX SafeTensors 4bit when comparing ARLE to oMLX public SOTA numbers.
  GGUF Q4_K_M is a separate optimization target with different weight format
  and memory behavior.
