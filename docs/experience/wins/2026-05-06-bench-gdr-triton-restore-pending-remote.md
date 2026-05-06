# 2026-05-06 · GDR Triton AOT restore — pending-remote regression check

## Goal

- **Type:** regression.
- Validate that restoring the production Qwen3.5 chunk-wise GDR Triton AOT path preserves the last known CUDA behavior after the unvalidated TileLang Phase 2b swap attempt was backed out.

## Hypothesis

- Restoring `tools/triton/` and the `build.rs` Triton GDR generator should return the runtime GDR symbols to the validated seven-stage AOT path.
- The TileLang GDR source remains scaffold-only and should not affect CUDA runtime output until a separate GPU-validated swap lands.

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
scripts/bench_guidellm.sh gdr-triton-restore-qwen35 \
  --target http://127.0.0.1:8000 \
  --model Qwen/Qwen3.5-4B \
  --processor infer/models/Qwen3.5-4B
```

Local non-kernel validation run in this workspace:

```bash
cargo fmt --all -- --check
python3 -m py_compile crates/cuda-kernels/tools/tilelang/gen_tilelang_aot.py crates/cuda-kernels/tools/tilelang/gated_delta_rule.py crates/cuda-kernels/tools/triton/gen_triton_aot.py crates/cuda-kernels/tools/triton/gated_delta_rule_chunkwise_kernels.py
cargo check -p infer --no-default-features --features cuda,no-cuda
cargo clippy --workspace --release --all-targets --no-default-features --features no-cuda -- -D warnings
cargo test --workspace --release --no-default-features --features no-cuda
```

## Environment

- **Backend:** CUDA runtime path restored; local validation uses `no-cuda` build-script skip where required.
- **Model:** Qwen/Qwen3.5-4B pending remote.
- **Hardware:** local Linux x86_64 container without `nvcc`, TileLang, Triton, or CUDA validation GPU; remote target should be NVIDIA L4 sm_89 or equivalent.
- **Commit:** pending final commit for this correction.
- **Feature set:** local `no-cuda` checks plus `cargo check -p infer --no-default-features --features cuda,no-cuda`; remote `cargo test/build --release -p infer --features cuda`.
- **Non-default flags / env vars:** none locally.

## Results — sweep headline table

Pending remote. The local workspace can only validate Rust/Python syntax and non-kernel paths.

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

- This container lacks `nvcc`, TileLang, Triton, and a CUDA GPU, so the production CUDA build and Qwen3.5 numerical gate must run remotely.

## Learnings

- Do not delete or replace the production GDR AOT owner in the same change that only adds a source-level TileLang scaffold. The runtime ABI swap is Phase 2b and needs the CUDA e2e + guidellm gate before Triton removal.

## Δ vs baseline

- **Baseline:** `docs/experience/wins/2026-05-05-bench-tilelang-phase2-pending-remote.md`.

| metric | baseline | now | Δ% |
|---|---|---|---|
| Qwen3.5 e2e correctness | pending remote | pending remote | n/a |
| out tok/s @ c=16 | pending remote | pending remote | n/a |

## Artefacts

- Raw: pending `bench-output/2026-05-06-gdr-triton-restore-qwen35/benchmarks.json`.
- CSV: pending `bench-output/2026-05-06-gdr-triton-restore-qwen35/benchmarks.csv`.
- HTML: pending `bench-output/2026-05-06-gdr-triton-restore-qwen35/benchmarks.html`.

## Notes

- What changed in the code since baseline: restored Triton GDR AOT build files and dependency, removed the unvalidated TileLang GDR runtime build swap, and reverted unrelated cleanup churn from the previous change.
- Suspected cause of any regression: environment or dependency drift from restoring the Triton generator; fail the change if `e2e_qwen35` diverges.
- Follow-ups: run the pending remote CUDA gate and replace this stub with a dated after-snapshot entry.
