# Metal Qwen3.5 0.8B GGUF vs safetensors local A/B

## Goal

- Type: diagnosis.
- Compare local Metal Qwen3.5 0.8B GGUF Q4_K_M performance against the local
  safetensors checkpoint on the same step-driver path.

## Hypothesis

- Metal GGUF should have similar steady-state runtime throughput to safetensors
  because the current Metal GGUF path dequantizes to BF16 at load time; GGUF may
  load slower and use more resident memory because the packed GGUF source and
  BF16 materialized tensors coexist during/after load.

## Command

```bash
./target/release/metal_bench \
  --model models/Qwen3.5-0.8B \
  --prompt-tokens 128 \
  --generation-tokens 64 \
  --warmup 1 \
  --runs 3 \
  --use-step-driver \
  --ignore-eos \
  --json

./target/release/metal_bench \
  --model models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt-tokens 128 \
  --generation-tokens 64 \
  --warmup 1 \
  --runs 3 \
  --use-step-driver \
  --ignore-eos \
  --json
```

## Environment

- Backend: Metal step-driver path (`--use-step-driver`)
- Hardware: Apple M4 Pro, 20-core GPU, 48 GB memory
- OS / Metal: macOS 26.3.1 (25D771280a), Metal 4
- Commit: `b64192a`
- Working tree: dirty with unrelated local files, so this is a local A/B
  measurement and not a canonical guidellm entry.
- Feature set: release Metal binary, `--no-default-features --features metal,no-cuda`
- Models:
  - `models/Qwen3.5-0.8B`
  - `models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf`

## Results

Raw JSON:

```json
{"avg_tokens":64,"generation_tokens_requested":64,"generation_tps":{"mean":30.195357780540093,"p50":30.412402879660522,"p99":30.48761729322363},"load_ms":585.750417,"mode":"step-driver","model":"models/Qwen3.5-0.8B","peak_rss_mb":1587.96875,"prompt_tokens":128,"prompt_tokens_requested":128,"prompt_tps":{"mean":29.797082818171912,"p50":29.76488069631218,"p99":29.929353286094607},"quantization":"bf16","repo_e2e_tps":{"mean":9.97601683359519,"p50":9.992545534488572,"p99":10.037718491675173},"timed_runs":3,"total_time_ms":{"mean":6415.605833333332,"p50":6404.774417,"p99":6466.092208},"ttft_ms":{"mean":4295.768499666668,"p50":4300.37,"p99":4310.197583},"warmup_runs":1}
{"avg_tokens":64,"generation_tokens_requested":64,"generation_tps":{"mean":29.934840726470526,"p50":29.896566633536384,"p99":30.05025141554585},"load_ms":2006.658542,"mode":"step-driver","model":"models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf","peak_rss_mb":2727.53125,"prompt_tokens":128,"prompt_tokens_requested":128,"prompt_tps":{"mean":29.7574985053171,"p50":29.77470463798306,"p99":29.814407643995974},"quantization":"bf16","repo_e2e_tps":{"mean":9.938787469169531,"p50":9.938404795187141,"p99":9.964203081978052},"timed_runs":3,"total_time_ms":{"mean":6439.4449583333335,"p50":6439.66525,"p99":6455.677291999999},"ttft_ms":{"mean":4301.451458333334,"p50":4298.9511250000005,"p99":4312.176917000001},"warmup_runs":1}
```

Headline:

| Metric | safetensors BF16 | GGUF Q4_K_M via load-time BF16 | Delta |
|---|---:|---:|---:|
| Load time | 585.8 ms | 2006.7 ms | +242.6% |
| Peak RSS | 1588.0 MB | 2727.5 MB | +71.8% |
| Prompt throughput mean | 29.80 tok/s | 29.76 tok/s | -0.13% |
| Decode throughput mean | 30.20 tok/s | 29.93 tok/s | -0.86% |
| Repo E2E throughput mean | 9.98 tok/s | 9.94 tok/s | -0.37% |
| TTFT mean | 4295.8 ms | 4301.5 ms | +0.13% |
| Total time mean | 6415.6 ms | 6439.4 ms | +0.37% |

Single-token prompt smoke check:

| Model | Prompt tokens | Output tokens | TTFT | Gen TPS |
|---|---:|---:|---:|---:|
| safetensors BF16 | 1 | 8 | 98.9 ms | 33.2 tok/s |
| GGUF Q4_K_M | 1 | 8 | 99.8 ms | 33.2 tok/s |

## Problems

- A larger `512` prompt / `128` generation `metal_bench --use-step-driver`
  run against `models/Qwen3.5-0.8B` hit a Metal command buffer GPU hang:
  `kIOGPUCommandBufferCallbackErrorHang`. This run was discarded and the
  comparison was reduced to `128/64`.
- The working tree was dirty before this measurement; the dirty paths appear
  unrelated to the Metal Qwen3.5 backend, but this is not clean enough to use
  as a canonical regression gate.
- This is a helper benchmark, not the canonical `scripts/bench_guidellm.sh`
  HTTP sweep.

## Learnings

- Current Metal GGUF Qwen3.5 is a compatibility path, not a quantized runtime
  speed path: it reports `quantization=bf16` after loading because GGUF tensors
  are materialized into BF16 MLX arrays.
- Runtime throughput is effectively identical to safetensors at this shape;
  the observable GGUF cost is load time and RSS, not decode speed.
- To get Metal GGUF memory or speed wins, the Metal backend needs native
  packed/k-quant execution instead of load-time BF16 dequantization.

## Delta vs baseline

- Baseline: local `models/Qwen3.5-0.8B` safetensors BF16 under the same command.

| Metric | Baseline | GGUF | Delta |
|---|---:|---:|---:|
| Decode throughput mean | 30.20 tok/s | 29.93 tok/s | -0.86% |
| Prompt throughput mean | 29.80 tok/s | 29.76 tok/s | -0.13% |
| Peak RSS | 1588.0 MB | 2727.5 MB | +71.8% |
| Load time | 585.8 ms | 2006.7 ms | +242.6% |
