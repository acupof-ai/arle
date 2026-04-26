# CUDA Qwen3.5 GGUF packed V-row reorder — guidellm sweep, cuda-qwen35-gguf, 2026-04-26

## Goal

- Regression-check CUDA Qwen3.5 GGUF after restoring packed Q3_K/Q4_K/Q6_K row-reorder loading for linear-attention V-head projections.

## Hypothesis

- Correctness should be unchanged; CUDA GGUF memory residency should improve for row-reordered Q_K tensors such as `linear_attn.in_proj_z.weight` because they remain packed instead of becoming BF16 host-dequantized matrices.

## Command

```bash
CUDA_HOME=/usr/local/cuda cargo build --release
./target/release/infer_serve --model <qwen35-gguf-path> --port 8010
scripts/bench_guidellm.sh cuda-qwen35-gguf-packed-v-row-reorder \
  --target http://127.0.0.1:8010 \
  --model <qwen35-gguf-path> \
  --processor <qwen35-tokenizer-dir>
```

Invoked via: pending remote CUDA runner.

## Environment

- **Backend:** cuda
- **Model:** Qwen3.5 GGUF, preferably the same Q4_K_M family used by the GGUF loader smoke tests
- **Hardware:** pending remote NVIDIA GPU
- **Commit:** pending; working tree after `6020668`
- **Feature set:** `CUDA_HOME=/usr/local/cuda cargo build --release`
- **Non-default flags / env vars:** none; rerun once with `INFER_FORCE_BF16_QUANT=1` only if packed-vs-BF16 bisection is needed
- **Server launch:** pending remote

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh cuda-qwen35-gguf-packed-v-row-reorder`

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

- Local verification ran on Apple M4 Pro / Metal 4 only; CUDA runtime benchmark is pending remote NVIDIA access.

## Learnings

- Qwen3.5 GGUF row reorder can preserve packed K-quant superblocks when the permutation moves complete rows; column reorder and QKV V-slice transforms still require BF16 host materialization unless a dedicated packed transform is added.

## Delta vs baseline

- **Baseline:** prior CUDA Qwen3.5 GGUF packed-loader state before `load_qwen35_linear_attention_host` unified the CUDA linear-attention load path.
- **Delta table:** pending remote.

| metric | baseline | now | Δ% |
|---|---:|---:|---:|
| TTFT p50 @ synchronous | pending | pending | pending |
| out tok/s @ saturation | pending | pending | pending |

## Artefacts

- Raw: pending remote `bench-output/<date>-cuda-qwen35-gguf-packed-v-row-reorder/benchmarks.json`
- CSV: pending remote `bench-output/<date>-cuda-qwen35-gguf-packed-v-row-reorder/benchmarks.csv`
- HTML: pending remote `bench-output/<date>-cuda-qwen35-gguf-packed-v-row-reorder/benchmarks.html`
- Service trace: pending remote

## Notes

- What changed in the code since baseline: CUDA Qwen3.5 GGUF linear-attention row-reordered matrices now call `load_tensor_2d_gguf_v_reorder_rows`, which keeps Q3_K/Q4_K/Q6_K packed when `cols % 256 == 0`; `INFER_FORCE_BF16_QUANT=1` still forces the fallback path for bisection.
- Suspected cause of any regression: packed row permutation byte-stride mismatch; local typecheck covers this path but cannot execute CUDA GEMV on this machine.
- Follow-ups: run CUDA remote smoke on Qwen3.5 GGUF and fill this entry with guidellm results.
