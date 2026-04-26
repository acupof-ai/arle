# CUDA Weight Quant Kernel Plan — guidellm sweep, CUDA, 2026-04-26

## Goal

- Regression check (pending remote): validate that explicit weight formats and
  linear kernel-plan dispatch do not regress CUDA quantized serving throughput or
  TTFT.

## Hypothesis

- Dispatch should be performance-neutral for existing BF16, W2/W4/W8, GGUF
  Q3_K/Q4_K/Q6_K, Marlin W4, and TurboQuant paths because the launched kernels
  and batch thresholds are unchanged; only Rust-side selection moved from
  `quant_bits` sentinels to `WeightFormat` and `LinearKernelPlan`.

## Command

```bash
scripts/bench_guidellm.sh cuda-weight-quant-kernel-plan \
  --target http://localhost:8000 \
  --model Qwen/Qwen3-4B \
  --processor models/Qwen3-4B \
  --trace-interval-ms 1000
```

Invoked via: pending remote CUDA machine.

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B, plus GGUF Q4_K smoke if available on the remote
- **Hardware:** pending remote NVIDIA GPU
- **Commit:** `9420311` plus working diff for weight quant format/plan refactor
- **Feature set:** `CUDA_HOME=/usr/local/cuda cargo build --release --features cuda`
- **Non-default flags / env vars:** none expected
- **Server launch:** pending remote CUDA launch for the quantized checkpoint

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh cuda-weight-quant-kernel-plan`

## Results — sweep headline table

| rate (req/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---:|---:|---:|---:|---:|---:|
| pending-remote | pending | pending | pending | pending | pending | pending |

## Results — service-side KV / scheduler metrics

| metric | value |
|---|---:|
| peak active | pending |
| peak waiting | pending |
| peak prefill_queue | pending |
| peak kv_util | pending |
| `prefix_hit_rate` | pending |
| `prefix_skip_rate` | pending |
| `kv_fetch_q` | pending |
| `kv_fetch_waiters` | pending |
| `kv_store_q` | pending |
| `kv_store` | pending |
| `kv_bp` | pending |
| `tier_recall` | n/a |
| `tier_src` | n/a |
| `tier_promoted` | n/a |
| `tier_fallback` | n/a |

## Results — request accounting

| metric | value |
|---|---:|
| completed input tokens | pending |
| incomplete input tokens | pending |
| completed output tokens | pending |
| incomplete output tokens | pending |

## Problems

- Local machine is Apple Silicon without CUDA runtime; `cargo check` can
  validate Rust/CUDA types under `cuda,no-cuda`, but GuideLLM CUDA serving and
  CUDA kernel tests require a remote NVIDIA host.

## Learnings

- Kernel alignment should be represented as data (`WeightFormat` +
  `LinearKernelPlan`) before adding new quant kernels; otherwise loader format,
  scale layout, and dispatch drift into duplicated sentinel checks.

## Δ vs baseline

- **Baseline:** pending selection from latest CUDA quantized GuideLLM entry on
  the remote host.

| metric | baseline | now | Δ% |
|---|---:|---:|---:|
| TTFT p50 @ synchronous | pending | pending | pending |
| out tok/s @ saturation | pending | pending | pending |

## Artefacts

- Raw: pending `bench-output/2026-04-26-cuda-weight-quant-kernel-plan/benchmarks.json`
- CSV: pending `bench-output/2026-04-26-cuda-weight-quant-kernel-plan/benchmarks.csv`
- HTML: pending `bench-output/2026-04-26-cuda-weight-quant-kernel-plan/benchmarks.html`
- Service trace (before): pending
- Service trace (during): pending
- Service trace (after): pending
- Service trace (summary): pending

## Notes

- What changed in the code since baseline: explicit `WeightFormat`, explicit
  linear kernel plan dispatch, dense-BF16 graph-safe predicate cleanup.
- Suspected cause of any regression: Rust dispatch bug or changed TurboQuant
  `is_quantized()` branch behavior; launched CUDA kernels should otherwise be
  the same.
- Follow-ups: run remote CUDA GuideLLM and replace this stub with a new dated
  completed entry rather than editing this file.
