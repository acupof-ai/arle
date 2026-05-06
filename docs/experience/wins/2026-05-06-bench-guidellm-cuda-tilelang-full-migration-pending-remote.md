# TileLang CUDA Full Migration — guidellm sweep, cuda-l4, 2026-05-06

Status: `pending-remote`

## Goal

- Complete the CUDA backend migration so `--features cuda` uses TileLang AOT for BF16 paged attention and AOT-compatible GDR stages, with no runtime FlashInfer or Triton AOT fallback. The GDR triangular solve and legacy non-paged direct-model prefill fallback are native CUDA C because TileLang 0.1.9 cannot lower those layouts on sm_89.

## Hypothesis

- Removing the old feature gates and wrapper paths should make CUDA builds simpler without changing the scheduler contract; numerical and throughput validation still needs a full model-weight run.

## Command

```bash
scripts/bench_guidellm.sh cuda-tilelang-full-migration
```

Invoked via: not run locally; no local Qwen model weights were available for a server run.

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B intended baseline; local model weights absent.
- **Hardware:** NVIDIA L4, 23034 MiB VRAM, compute capability 8.9, driver 580.82.07, CUDA 12.8 / nvcc 12.8.93.
- **Commit:** working tree based on `b9fe80ba`; final migration commit records this entry.
- **Feature set:** `cargo build --release --features cuda`
- **Non-default flags / env vars:** `CUDA_HOME=/usr/local/cuda`, `TORCH_CUDA_ARCH_LIST=8.9`, `CMAKE_CUDA_ARCHITECTURES=89`, `INFER_TILELANG_PYTHON=/usr/bin/python3`, `CARGO_TARGET_DIR=/tmp/arle-target-release`.
- **Server launch:** not attempted; model weights absent.

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh cuda-tilelang-full-migration`

## Results — sweep headline table

| rate (req/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| pending-remote | n/a | n/a | n/a | n/a | n/a | n/a |

## Results — service-side KV / scheduler metrics

| metric | value |
|---|---:|
| peak active | n/a |
| peak waiting | n/a |
| peak prefill_queue | n/a |
| peak kv_util | n/a |
| `prefix_hit_rate` | n/a |
| `prefix_skip_rate` | n/a |
| `kv_fetch_q` | n/a |
| `kv_fetch_waiters` | n/a |
| `kv_store_q` | n/a |
| `kv_store` | n/a |
| `kv_bp` | n/a |
| `tier_recall` | n/a |
| `tier_src` | n/a |
| `tier_promoted` | n/a |
| `tier_fallback` | n/a |

## Results — request accounting

| metric | value |
|---|---:|
| completed input tokens | n/a |
| incomplete input tokens | n/a |
| completed output tokens | n/a |
| incomplete output tokens | n/a |

## Problems

- Initial cold-cache `cargo check` attempts timed out while compiling dependencies; reruns with warmed target dirs completed.
- `cargo fmt --all -- --check` passed.
- `git diff --check` passed.
- TileLang Python generators passed `python3 -m py_compile`.
- `cargo check -p infer --no-default-features --features no-cuda` passed.
- `cargo check -p infer --no-default-features --features cuda,no-cuda` passed.
- `cargo check -p infer --no-default-features --features cuda` passed and ran TileLang AOT for sm_89.
- `cargo check -p infer --benches --no-default-features --features cuda` passed and compiled the renamed CUDA ops Criterion bench.
- `cargo clippy -p infer --no-default-features --features no-cuda -- -D warnings` passed.
- `cargo clippy -p infer --no-default-features --features cuda -- -D warnings` passed and ran TileLang AOT for sm_89.
- `cargo test -p infer --no-default-features --features no-cuda --lib` passed (`539 passed; 10 ignored`).
- `cargo test -p cuda-kernels --features cuda` ran in the L4 environment; CUDA context-dependent tensor tests passed, but `paged_kv::tests::retain_release_without_free_slot_does_not_move_pages` failed on the pre-existing retain/release invariant. This migration only changed comments/type names in `paged_kv.rs`, so the failure remains a residual paged-KV unit-test issue rather than a TileLang migration blocker.
- Active source/script/test/bench scans found no current FlashInfer/Triton AOT feature gates, tool paths, call sites, or benchmark entry names.
- No local Qwen model weights were found, so `scripts/bench_guidellm.sh` could not be run.

## Learnings

- Current L4/SM89 can compile the migrated CUDA feature set end to end.
- Static checks validate the removal scope: current source, Cargo manifests, build.rs, setup docs, scripts, tests, and bench entrypoints no longer contain runtime FlashInfer/Triton AOT gates or call sites.

## Δ vs baseline

- **Baseline:** `docs/experience/errors/2026-05-06-qwen3-bf16-baseline-degenerate-post-00def315.md`
- **Delta table:** pending remote run.

| metric | baseline | now | Δ% |
|---|---|---|---|
| TTFT p50 @ synchronous | pending | pending | n/a |
| out tok/s @ saturation | pending | pending | n/a |

## Artefacts

- Raw: pending remote
- CSV: pending remote
- HTML: pending remote
- Service trace (before): pending remote
- Service trace (during): pending remote
- Service trace (after): pending remote
- Service trace (summary): pending remote

## Notes

- What changed in the code since baseline: CUDA BF16 paged attention is unified on TileLang AOT; direct non-paged prefill uses a native CUDA C fallback; GDR uses TileLang AOT for compatible stages plus native CUDA C for the triangular solve; FlashInfer wrappers, Triton AOT tools, optional TileLang feature aliases, and build-time FlashInfer/Triton Python deps were removed.
- Suspected cause of any regression: n/a until remote numerical and throughput runs complete.
- Follow-ups: run `cargo build --release --features cuda`, `cargo test --release --test e2e`, `cargo test --release --test e2e_qwen35`, and the canonical guidellm sweep on the remote L4/H100 validation host with model weights.
