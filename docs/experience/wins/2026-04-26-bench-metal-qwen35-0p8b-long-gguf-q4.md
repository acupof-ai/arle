# Metal Qwen3.5 0.8B Long GGUF Q4 Comparison

## Goal

- Type: diagnosis.
- Compare long-sequence Qwen3.5-0.8B BF16 safetensors against true GGUF
  Q4_K_M weights on the Metal step-driver path.

## Hypothesis

- Q4 should be faster on decode and use less resident memory.
- Long-prompt TTFT may remain slower until the GGUF linear-attention GDR state
  update is aligned with the custom Metal recurrent kernel.

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

Diagnostic-only check:

```bash
METAL_NO_CPP=1 ./target/release/metal_bench \
  --model /path/to/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt-tokens 512 \
  --generation-tokens 64 \
  --warmup 0 \
  --runs 1 \
  --use-step-driver \
  --ignore-eos \
  --json
```

## Environment

- Backend: Metal
- Hardware: Apple M4 Pro, 20-core GPU, 48 GB unified memory
- OS / Metal: macOS 26.3.1 (25D771280a), Metal 4
- Model: Qwen3.5-0.8B safetensors vs Qwen3.5-0.8B GGUF Q4_K_M
- Feature set: `--release --no-default-features --features metal,no-cuda`
- Non-default flags: none for primary runs
- Commit: `a0f6548`

## Results

Raw JSON:

```json
{"avg_tokens":256,"generation_tokens_requested":256,"generation_tps":{"mean":29.41539967063871,"p50":29.41539967063871,"p99":29.41539967063871},"load_ms":662.1587079999999,"mode":"step-driver","model":"/path/to/code/agent-infer/models/Qwen3.5-0.8B","peak_rss_mb":1602.28125,"prompt_tokens":2048,"prompt_tokens_requested":2048,"prompt_tps":{"mean":2512.967072677112,"p50":2512.967072677112,"p99":2512.967072677112},"quantization":"bf16","repo_e2e_tps":{"mean":26.89669869582948,"p50":26.89669869582948,"p99":26.89669869582948},"timed_runs":1,"total_time_ms":{"mean":9517.896709,"p50":9517.896709,"p99":9517.896709},"ttft_ms":{"mean":814.972875,"p50":814.972875,"p99":814.972875},"warmup_runs":0}
{"avg_tokens":256,"generation_tokens_requested":256,"generation_tps":{"mean":37.584898188473005,"p50":37.584898188473005,"p99":37.584898188473005},"load_ms":1752.73575,"mode":"step-driver","model":"/path/to/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf","peak_rss_mb":1072.171875,"prompt_tokens":2048,"prompt_tokens_requested":2048,"prompt_tps":{"mean":53.66781560559791,"p50":53.66781560559791,"p99":53.66781560559791},"quantization":"gguf-q4_k","repo_e2e_tps":{"mean":5.69244095091698,"p50":5.69244095091698,"p99":5.69244095091698},"timed_runs":1,"total_time_ms":{"mean":44971.920167000004,"p50":44971.920167000004,"p99":44971.920167000004},"ttft_ms":{"mean":38160.673709,"p50":38160.673709,"p99":38160.673709},"warmup_runs":0}
{"avg_tokens":1024,"generation_tokens_requested":1024,"generation_tps":{"mean":30.070981649172438,"p50":30.070981649172438,"p99":30.070981649172438},"load_ms":644.31675,"mode":"step-driver","model":"/path/to/code/agent-infer/models/Qwen3.5-0.8B","peak_rss_mb":1594.671875,"prompt_tokens":512,"prompt_tokens_requested":512,"prompt_tps":{"mean":3217.8945102531106,"p50":3217.8945102531106,"p99":3217.8945102531106},"quantization":"bf16","repo_e2e_tps":{"mean":29.931130230424362,"p50":29.931130230424362,"p99":29.931130230424362},"timed_runs":1,"total_time_ms":{"mean":34211.872125,"p50":34211.872125,"p99":34211.872125},"ttft_ms":{"mean":159.11025,"p50":159.11025,"p99":159.11025},"warmup_runs":0}
{"avg_tokens":1024,"generation_tokens_requested":1024,"generation_tps":{"mean":37.77996449114118,"p50":37.77996449114118,"p99":37.77996449114118},"load_ms":1506.9372919999998,"mode":"step-driver","model":"/path/to/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf","peak_rss_mb":1074.5,"prompt_tokens":512,"prompt_tokens_requested":512,"prompt_tps":{"mean":53.18629723288152,"p50":53.18629723288152,"p99":53.18629723288152},"quantization":"gguf-q4_k","repo_e2e_tps":{"mean":27.878470450985713,"p50":27.878470450985713,"p99":27.878470450985713},"timed_runs":1,"total_time_ms":{"mean":36730.853,"p50":36730.853,"p99":36730.853},"ttft_ms":{"mean":9626.539665999999,"p50":9626.539665999999,"p99":9626.539665999999},"warmup_runs":0}
```

Long prompt, 2048/256:

| Metric | BF16 safetensors | GGUF Q4_K_M | Delta |
|---|---:|---:|---:|
| Peak RSS | 1602.3 MB | 1072.2 MB | -33.1% |
| Prompt throughput mean | 2512.97 tok/s | 53.67 tok/s | -97.9% |
| Decode throughput mean | 29.42 tok/s | 37.58 tok/s | +27.8% |
| Repo E2E throughput mean | 26.90 tok/s | 5.69 tok/s | -78.8% |
| TTFT mean | 815.0 ms | 38160.7 ms | +4582.4% |
| Total time mean | 9517.9 ms | 44971.9 ms | +372.5% |

Long output, 512/1024:

| Metric | BF16 safetensors | GGUF Q4_K_M | Delta |
|---|---:|---:|---:|
| Peak RSS | 1594.7 MB | 1074.5 MB | -32.6% |
| Prompt throughput mean | 3217.89 tok/s | 53.19 tok/s | -98.3% |
| Decode throughput mean | 30.07 tok/s | 37.78 tok/s | +25.6% |
| Repo E2E throughput mean | 29.93 tok/s | 27.88 tok/s | -6.9% |
| TTFT mean | 159.1 ms | 9626.5 ms | +5950.2% |
| Total time mean | 34211.9 ms | 36730.9 ms | +7.4% |

Diagnostic-only Rust fallback:

```json
{"avg_tokens":64,"generation_tokens_requested":64,"generation_tps":{"mean":37.53627162184553,"p50":37.53627162184553,"p99":37.53627162184553},"load_ms":1397.151875,"mode":"step-driver","model":"/path/to/code/agent-infer/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf","peak_rss_mb":898.515625,"prompt_tokens":512,"prompt_tokens_requested":512,"prompt_tps":{"mean":38.4375590205622,"p50":38.4375590205622,"p99":38.4375590205622},"quantization":"gguf-q4_k","repo_e2e_tps":{"mean":4.259475996767091,"p50":4.259475996767091,"p99":4.259475996767091},"timed_runs":1,"total_time_ms":{"mean":15025.322375,"p50":15025.322375,"p99":15025.322375},"ttft_ms":{"mean":13320.30475,"p50":13320.30475,"p99":13320.30475},"warmup_runs":0}
```

## Problems

- GGUF Q4 decode is faster, but long prefill is not aligned: the safe C++ ops
  fallback for Qwen3.5 GDR state update is sequential over prompt tokens.
- Re-enabling the custom GDR Metal recurrent kernel for GGUF was tested and
  rejected by replay correctness: generated tokens became `[271, 13, 198, 198]`
  instead of `[271, 9419, 1814, 494]`.
- A background `kv_tier::` cargo test from another local process repeatedly
  respawned and was killed before each primary bench run to avoid CPU noise.
- `cargo test --release -p infer --no-default-features --features no-cuda`
  still hangs at `kv_tier::coordinator::tests::coordinator_receives_commands`;
  this is outside the Metal/GGUF path touched here.

## Learnings

- Q4 is already aligned enough for decode on M4 Pro: 25.6-27.8% higher decode
  throughput and about 33% lower RSS on these runs.
- Q4 is not end-to-end faster on long prompts until GGUF packed/reordered
  linear-attention state can use a correctness-validated recurrent Metal
  kernel.
- Long-output workloads hide the prefill gap better, but they do not remove
  the TTFT regression; kernel alignment has to be fixed before claiming Q4 as
  generally faster.

## Delta vs Baseline

Baseline entry:
[`2026-04-26-bench-metal-qwen35-0p8b-gguf-cpp-gdr-fallback.md`](2026-04-26-bench-metal-qwen35-0p8b-gguf-cpp-gdr-fallback.md).

| Metric | 32/32 Q4 baseline | 512/1024 Q4 | Delta |
|---|---:|---:|---:|
| Decode throughput mean | 37.23 tok/s | 37.78 tok/s | +1.5% |
| Prompt throughput mean | 35.91 tok/s | 53.19 tok/s | +48.1% |
| Peak RSS | 912.2 MB | 1074.5 MB | +17.8% |

## Artefacts

- Raw: inline JSON above
- CSV / HTML: n/a for local helper benchmark
- Service trace: n/a; no HTTP server
