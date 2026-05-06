# 2026-05-05 · TileLang Phase 2b (GDR AOT swap + Triton deletion) — superseded pending-remote stub

> **Superseded on 2026-05-06.** The attempted Phase 2b build swap was reverted
> before GPU validation because production Qwen3.5 GDR must stay on the
> validated Triton AOT path until a CUDA parity run passes. Treat this file as
> historical trace only; the active follow-up obligation is
> `2026-05-06-bench-gdr-triton-restore-pending-remote.md`.

## Goal

- **Type:** regression.
- Historical goal: validate whether replacing the Qwen3.5 chunk-wise GDR Triton AOT pipeline with TileLang AOT preserves correctness and throughput. This was not shipped as the active runtime path after the 2026-05-06 correction.

## Hypothesis

- Superseded hypothesis: the Phase 2b TileLang code path might be numerically equivalent to the historical seven-stage Triton pipeline. The correction keeps Triton as production GDR until that hypothesis is measured on CUDA.

## Command

Pending remote CUDA validation:

```bash
source /tmp/arle-env.sh
CUDA_HOME=/usr/local/cuda cargo test --release -p infer --features cuda --test e2e_qwen35
CUDA_HOME=/usr/local/cuda cargo build --release -p infer --features cuda --bin infer
./target/release/infer \
  --model-path infer/models/Qwen3.5-4B \
  --port 8000 \
  --num-slots 16 \
  --max-seq-len 8192
scripts/bench_guidellm.sh tilelang-phase2b-qwen35 \
  --target http://127.0.0.1:8000 \
  --model Qwen/Qwen3.5-4B \
  --processor infer/models/Qwen3.5-4B \
  --concurrencies 16 \
  --max-seconds 120
```

Local non-kernel validation run in this workspace:

```bash
cargo check -p infer --no-default-features --features cuda,no-cuda
python3 -m py_compile crates/cuda-kernels/tools/tilelang/gen_tilelang_aot.py
```

## Environment

- **Backend:** CUDA runtime path changed, but local validation used `no-cuda` build-script skip.
- **Model:** Qwen/Qwen3.5-4B pending remote.
- **Hardware:** local Linux x86_64 container without CUDA validation GPU; remote target should be NVIDIA L4 sm_89 or equivalent.
- **Commit:** pending final commit for this change.
- **Feature set:** local `cargo check -p infer --no-default-features --features cuda,no-cuda`; remote `cargo test/build --release -p infer --features cuda`.
- **Non-default flags / env vars:** none locally.

## Results — sweep headline table

Pending remote. The local workspace can only validate non-kernel Rust/Python code paths.

| rate (req/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---:|---:|---:|---:|---:|---:|
| pending remote | ... | ... | ... | ... | ... | ... |

## Results — service-side KV / scheduler metrics

| metric | value |
|---|---:|
| peak active | pending remote |
| peak waiting | pending remote |
| peak prefill_queue | pending remote |
| peak kv_util | pending remote |
| `prefix_hit_rate` | pending remote |
| `prefix_skip_rate` | pending remote |
| `kv_fetch_q` | pending remote |
| `kv_fetch_waiters` | pending remote |
| `kv_store_q` | pending remote |
| `kv_store` | pending remote |
| `kv_bp` | pending remote |
| `tier_recall` | n/a |
| `tier_src` | n/a |
| `tier_promoted` | n/a |
| `tier_fallback` | n/a |

## Results — request accounting

| metric | value |
|---|---:|
| completed input tokens | pending remote |
| incomplete input tokens | pending remote |
| completed output tokens | pending remote |
| incomplete output tokens | pending remote |

## Problems

- GPU correctness/perf validation cannot run in this local container. The remote run must compare against the latest Qwen3.5 TileLang Phase 2 baseline and fail the change if JSON substring or Δ ≤ 5% out tok/s does not hold.

## Learnings

- Preserve stable FFI names for compiler-backend swaps when possible: Rust call sites stay type-checkable on Mac/non-CUDA while the build script changes the AOT producer behind the same ABI.

## Δ vs baseline

- **Baseline:** `docs/experience/wins/2026-05-05-bench-tilelang-phase2-pending-remote.md`.

| metric | baseline | now | Δ% |
|---|---|---|---|
| Qwen3.5 e2e correctness | pending remote | pending remote | n/a |
| out tok/s @ c=16 | pending remote | pending remote | n/a |

## Artefacts

- Raw: pending `bench-output/2026-05-05-tilelang-phase2b-qwen35/benchmarks.json`.
- CSV: pending `bench-output/2026-05-05-tilelang-phase2b-qwen35/benchmarks.csv`.
- HTML: pending `bench-output/2026-05-05-tilelang-phase2b-qwen35/benchmarks.html`.

## Notes

- What changed in the code since baseline: `build.rs` now compiles GDR through TileLang AOT; `gen_tilelang_aot.py` supports `attention` and `gdr` wrapper families; `tools/triton/` and the `triton` build dependency were removed.
- Suspected cause of any regression: TileLang codegen differences in stage 4 triangular solve or launch-grid mismatch; validate first with `e2e_qwen35` before throughput.
- Follow-ups: run the pending remote CUDA gate and replace this stub with a dated after-snapshot entry.
