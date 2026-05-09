# Hybrid W4 Phase 2 Dispatch — W4A8 Prefill Gate, CUDA, 2026-05-09

## Goal

- **License / kill gate:** route `marlin_w4_hybrid` prefill GEMM dispatch to W4A8 side tensors under `INFER_HYBRID_W4A8_PREFILL=1`, while keeping decode on the known-good W4A16 Marlin tensors and preserving the default-off guard.

## Hypothesis

- Since the `aaf0b55` nsys decomposition showed long-context serving is prefill-dominant, hybrid prefill should recover most of the earlier W4A8 TTFT gain without making W4A8 the default decode path.
- Kill condition: TTFT regression versus the W4A16 Marlin baseline.

## Command

W4A16 baseline:

```bash
CUDA_HOME=/opt/cuda \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
./target/release/infer \
  --model-path infer/models/Qwen3-4B-W4A16-sym-g128-marlin \
  --port 8000 --num-slots 8 --max-seq-len 5120

PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
scripts/bench_guidellm.sh hybrid-phase2-w4a16-baseline \
  --model Qwen3-4B-W4A16-sym-g128-marlin \
  --concurrencies 4 --max-seconds 120 --warmup 10 \
  --data 'prompt_tokens=4096,prompt_tokens_stdev=1,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256'
```

Hybrid W4A8 prefill:

```bash
INFER_HYBRID_W4A8_PREFILL=1 \
CUDA_HOME=/opt/cuda \
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
./target/release/infer \
  --model-path infer/models/Qwen3-4B-W4-hybrid-zpfix \
  --port 8000 --num-slots 8 --max-seq-len 5120

PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
scripts/bench_guidellm.sh hybrid-phase2-w4a8-prefill-fixed \
  --model Qwen3-4B-W4-hybrid-zpfix \
  --concurrencies 4 --max-seconds 120 --warmup 10 \
  --data 'prompt_tokens=4096,prompt_tokens_stdev=1,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256'
```

## Environment

- **Backend:** CUDA
- **Model:** Qwen3-4B W4A16 Marlin baseline vs Qwen3-4B hybrid W4A16/W4A8 zero-point-fixed checkpoint
- **Hardware:** NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB
- **CUDA / driver:** CUDA 13.2.78, driver 595.71.05
- **Commit:** measured on `ff3e8c1` plus the P1.0 dispatch diff
- **Feature set:** `--features cuda`
- **Non-default flags / env vars:** `INFER_HYBRID_W4A8_PREFILL=1` only for the hybrid run; `NVCC_CCBIN=/usr/bin/g++-14`, `INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python`, `TORCH_CUDA_ARCH_LIST=8.9`

## Implementation

- `DeviceMatrix::is_hybrid_w4_marlin()` now selects a phase-aware linear plan.
- `CudaOpsBackend::new()` is decode-phase dispatch; `CudaOpsBackend::prefill()` is prefill-phase dispatch.
- Decode and mixed-batch decode rows keep using W4A16 Marlin tensors through the existing Marlin path.
- Explicit prefill batches with `seq_len > 1` may route to W4A8 side tensors only when `INFER_HYBRID_W4A8_PREFILL=1`.
- With the env var unset, multi-row hybrid prefill still fail-fasts instead of silently serving on an unlicensed path.
- Hybrid W4 is opted out of the experimental mixed decode+prefill launch until that path can split decode and prefill projection phases.
- Unphased batched `gemm` / `gemm_into` now rejects hybrid W4 tensors so legacy callers cannot bypass the explicit phase gate.

## Results — GuideLLM 4k/c=4

| run | checkpoint | gate | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E mean | out tok/s | req/s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | `Qwen3-4B-W4A16-sym-g128-marlin` | n/a | 2384.3 ms | 2571.5 ms | 11.76 ms | 11.91 ms | 5.46 s | 191.49 | 0.727 |
| hybrid pre-review | `Qwen3-4B-W4-hybrid-zpfix` | `INFER_HYBRID_W4A8_PREFILL=1` | 1633.5 ms | 1811.7 ms | 19.09 ms | 19.95 ms | 6.64 s | 156.35 | 0.618 |
| hybrid fixed | `Qwen3-4B-W4-hybrid-zpfix` | `INFER_HYBRID_W4A8_PREFILL=1` | 1632.1 ms | 1810.8 ms | 13.11 ms | 13.29 ms | 5.05 s | 205.64 | 0.800 |

The pre-review hybrid run is retained only as diagnostic evidence. `codex review` caught that it accidentally routed decode-batched rows through W4A8, so it is not a license measurement.

## Delta vs W4A16 Baseline

| metric | W4A16 baseline | hybrid W4A8 prefill | delta |
|---|---:|---:|---:|
| TTFT p50 | 2384.3 ms | 1632.1 ms | -31.5% |
| TTFT p99 | 2571.5 ms | 1810.8 ms | -29.6% |
| ITL p50 | 11.76 ms | 13.11 ms | +11.5% |
| out tok/s | 191.49 | 205.64 | +7.4% |
| E2E mean | 5.46 s | 5.05 s | -7.5% |

## Service Metrics

| metric | W4A16 baseline | hybrid W4A8 prefill |
|---|---:|---:|
| peak active | 4 | 4 |
| peak waiting | 0 | 0 |
| peak running_batch | 4 | 4 |
| peak prefill_queue | 3 | 3 |
| plan labels | `idle=11148`, `decode=5617`, `prefill=80`, `split=0`, `mixed=0` | `idle=8353`, `decode=6127`, `prefill=87`, `split=0`, `mixed=0` |
| peak kv_util | 83.0% | 82.9% |
| completed input tokens | 344148 | 376924 |
| incomplete input tokens | 0 | 0 |
| completed output tokens | 21504 | 23552 |
| incomplete output tokens | 0 | 0 |

## License-Or-Kill

| criterion | result | verdict |
|---|---|---|
| TTFT regression vs W4A16 baseline | -31.5% TTFT p50 | pass |
| E2E / out tok/s regression vs W4A16 baseline | -7.5% E2E mean, +7.4% output tok/s | pass |
| Match earlier W4A8 prefill -36% claim | -31.5%, short by 4.5 percentage points | partial |
| Production default flip | Still blocked by W4A8-only accuracy and graph/scratch-hoist work | keep env-gated |

The Phase 2 dispatch path passes the no-regression wall-clock gate as an opt-in experiment. It does not justify a default flip: the strict historical -36% TTFT target was not reached, W4A8-only accuracy remains gated, and hybrid decode intentionally keeps CUDA graph disabled until Marlin scratch is hoisted.

## Verification

- `cargo fmt --all --check` PASS
- `git diff --check` PASS
- `cargo check --release -p infer --features cuda` PASS
- `cargo check -p infer --no-default-features --features cuda,no-cuda` PASS
- `cargo clippy --release -p infer --features cuda --lib -- -D warnings` PASS
- `cargo test --release -p infer --features cuda marlin_w4_hybrid -- --nocapture` PASS
- `cargo test --release -p infer --features cuda load_hybrid_w4_marlin_dispatches_to_w4a8_prefill -- --nocapture` PASS
- `cargo test --release -p infer --features cuda --test e2e -- --test-threads=1` PASS
- `scripts/bench_guidellm.sh hybrid-phase2-w4a8-prefill-fixed ...` PASS
- `codex review --uncommitted -c sandbox.timeouts.exec_seconds=900` found two P2 phase-boundary issues; both fixed by disabling hybrid mixed-batch and rejecting unphased hybrid GEMM.
- Final `codex review --uncommitted -c sandbox.timeouts.exec_seconds=900` PASS with no actionable correctness issues.

## Problems

- The first hybrid run is invalidated by review: B=2..8 decode rows were routed through W4A8, causing ITL and E2E regression. The implementation now uses explicit decode/prefill phase dispatch and keeps mixed-batch decode rows on W4A16.
- The fixed run recovers the whole-request wall-clock regression, but ITL remains +11.5% versus W4A16 because decode is still eager with hybrid Marlin graph capture disabled.
- Hybrid mixed-batch is deliberately disabled. The current mixed implementation uses one combined backend for decode and prefill rows, so routing prefill rows to W4A8 without contaminating decode rows needs a separate design.
- Full default-on remains blocked until W4A8-only accuracy and graph/scratch-hoist work land.

## Learnings

- Hybrid W4 needs phase-aware dispatch, not a single checkpoint-level quantization decision. W4A16 remains the safer decode path; W4A8 is useful for high-row prefill.
- Env-gated dispatch lets the prefill experiment land without weakening the Phase 1b fail-fast default.
- TTFT and ITL must be judged together under wall-clock framing. A prefill win is not licensed if decode pollution or eager overhead regresses E2E throughput.

## Artefacts

- Baseline raw: `bench-output/2026-05-09-hybrid-phase2-w4a16-baseline-run2/benchmarks.json`
- Baseline CSV: `bench-output/2026-05-09-hybrid-phase2-w4a16-baseline-run2/benchmarks.csv`
- Baseline service trace: `bench-output/2026-05-09-hybrid-phase2-w4a16-baseline-run2/service_stats_trace_summary.md`
- Invalid pre-review hybrid raw: `bench-output/2026-05-09-hybrid-phase2-w4a8-prefill/benchmarks.json`
- Invalid pre-review hybrid CSV: `bench-output/2026-05-09-hybrid-phase2-w4a8-prefill/benchmarks.csv`
- Invalid pre-review hybrid service trace: `bench-output/2026-05-09-hybrid-phase2-w4a8-prefill/service_stats_trace_summary.md`
- Fixed hybrid raw: `bench-output/2026-05-09-hybrid-phase2-w4a8-prefill-fixed/benchmarks.json`
- Fixed hybrid CSV: `bench-output/2026-05-09-hybrid-phase2-w4a8-prefill-fixed/benchmarks.csv`
- Fixed hybrid service trace: `bench-output/2026-05-09-hybrid-phase2-w4a8-prefill-fixed/service_stats_trace_summary.md`
