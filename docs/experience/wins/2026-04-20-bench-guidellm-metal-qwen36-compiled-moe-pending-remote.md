# Qwen3.6 compiled MoE bridge — guidellm pending-remote, metal, 2026-04-20

**Status:** `pending-remote`
**Commissioned by:** [`docs/plans/2026-04-15-metal-backend-acceptance-plan.md`](../../plans/2026-04-15-metal-backend-acceptance-plan.md)

## Goal

- Wire Qwen3.6 MoE checkpoints onto the compiled Metal C++ path and record the
  pending baseline sweep that should replace the Rust scalar fallback.

## Hypothesis

- Once the MoE layer is registered through the compiled C++ model, Qwen3.6
  should stop routing through the Rust fallback and recover the fast prefill /
  decode path already used by the compiled step model.

## Command

```bash
scripts/bench_guidellm.sh metal-qwen36-compiled-moe
```

- Canonical params remain the locked `scripts/bench_guidellm.sh` defaults.
- This command was **not** executed locally in this turn; this file exists to
  satisfy the required `pending-remote` stub flow for a runtime-affecting
  Metal diff where the canonical `4k in / 256 out` sweep is still queued for a
  dedicated Apple Silicon bench host.

## Environment

- **Backend:** metal
- **Model:** `mlx-community/Qwen3.6-35B-A3B-4bit`
- **Hardware:** Apple M4 Pro, 48 GB unified memory for local smoke; canonical sweep pending remote Metal host
- **Commit:** `80b25b2` + local uncommitted diff
- **Feature set:** `cargo build --release --no-default-features --features metal,no-cuda`
- **Non-default flags / env vars:** none
- **Server launch:** local `target/release/metal_serve --model-path '<snapshot>' --bind 127.0.0.1 --port 8011 --warmup 0`

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json,csv,html`
- Wrapper: `scripts/bench_guidellm.sh metal-qwen36-compiled-moe`

## Results — sweep headline table

| rate (req/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| pending-remote | pending-remote | pending-remote | pending-remote | pending-remote | pending-remote | pending-remote |

### Local quick bench (`--quick`, non-canonical)

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| conc1 | 1365.9 | 10922.0 | 16.36 | 37.66 | 42.21 | 0.073 |
| conc2 | 1533.4 | 5210.4 | 29.43 | 30.12 | 54.86 | 0.109 |
| conc4 | 5165.1 | 12581.4 | 53.52 | 61.64 | 41.35 | 0.073 |
| conc8 | 8791.5 | 33763.3 | 57.05 | 59.27 | 41.16 | 0.073 |

## Problems

- The canonical `guidellm` sweep has not been run yet.
- Local verification is limited to `metal_bench` and HTTP smoke because the
  canonical `4096 / 256` sweep is still pending a remote Apple Silicon host.
- The local canonical `4096 / 256` sweep was started, but it took too long to
  finish on this workstation for an interactive turn; I replaced it with a
  completed `--quick` exploration run for local no-OOM validation.
- During the quick run, the server logged a few
  `Metal batched decode post-process failed ... stream consumer dropped`
  lines at phase boundaries. `guidellm` still reported `Err Tot = 0` for all
  four concurrency levels, so these look like client-side stream teardown
  noise rather than request failures.

## Learnings

- Re-attaching Qwen3.6 MoE to `cpp_model` materially improves prefill/TTFT
  before any DFlash work: the step-driver prefill path switched from
  `rust_scalar_prefill` to `cpp_batch_prefill`.
- The next gap to close against public MLX/oMLX numbers is still prefill
  efficiency; compiled MoE improved local TTFT substantially, but not yet to
  the ~2.6 s public reference level.
- The local quick run completed at concurrency `1,2,4,8` without killing
  `metal_serve`; on this host the practical saturation point for the
  `512 / 128` profile is around `~55 out tok/s` at `conc2`.

## Δ vs baseline

- **Baseline:** [metal qwen3.6 rust runtime fix — guidellm pending-remote, 2026-04-20](./2026-04-20-bench-guidellm-metal-qwen36-rust-runtime-fix-pending-remote.md)
- Local `metal_bench` delta at `prompt=1024`, `gen=16`, `--use-step-driver`:

| metric | baseline | now | Δ% |
|---|---|---|---|
| Prompt speed | 56.4 tok/s | 125.0 tok/s | +121.6% |
| TTFT | 18159 ms | 8193 ms | -54.9% |

- Canonical guidellm delta table: pending remote.

## Artefacts

- Local verification:
  - `cargo check --release --manifest-path infer/Cargo.toml --no-default-features --features metal,no-cuda`
  - `cargo test --release --manifest-path infer/Cargo.toml --no-default-features --features metal,no-cuda loads_qwen36_config_with_nested_moe_block -- --nocapture`
  - `AGENT_INFER_METAL_QWEN35_TRACE=1 cargo run --release --manifest-path infer/Cargo.toml --no-default-features --features metal,no-cuda --bin metal_bench -- --model '<snapshot>' --prompt-tokens 1024 --generation-tokens 16 --warmup 0 --runs 1 --use-step-driver`
  - `cargo run --release --manifest-path infer/Cargo.toml --no-default-features --features metal,no-cuda --bin metal_bench -- --model '<snapshot>' --prompt-tokens 1024 --generation-tokens 16 --warmup 0 --runs 1`
  - `target/release/metal_serve --model-path '<snapshot>' --bind 127.0.0.1 --port 8011 --warmup 0`
  - `curl -s http://127.0.0.1:8011/v1/chat/completions ...`
  - `scripts/bench_guidellm.sh metal-qwen36-compiled-moe-quick --quick --target http://127.0.0.1:8011 --model mlx-community/Qwen3.6-35B-A3B-4bit --processor '<snapshot>'`
- Local observations:
  - `metal_bench --use-step-driver`: `cpp_batch_prefill`, prompt `125.0 tok/s`, generation `38.8 tok/s`, TTFT `8192.6 ms`
  - `metal_bench` direct path: prompt `205.3 tok/s`, generation `8.4 tok/s`, TTFT `4987.6 ms`
  - HTTP `/v1/chat/completions`: succeeded; startup log reported `C++ forward model ready (all 40 layers wired through one step call)` and `Metal live prefix cache enabled`
  - `guidellm --quick`: completed all four concurrency levels and wrote:
    - `bench-output/2026-04-20-metal-qwen36-compiled-moe-quick/benchmarks.json`
    - `bench-output/2026-04-20-metal-qwen36-compiled-moe-quick/benchmarks.csv`
    - `bench-output/2026-04-20-metal-qwen36-compiled-moe-quick/benchmarks.html`
- Remote canonical artefacts: pending.

## Notes

- What changed in the code since baseline:
  - `infer/src/backend/metal/qwen35.rs`: compiled builder now keeps MoE layers on the C++ path via `qwen35_compiled_set_last_moe_mlp`
  - `crates/mlx-sys/src/mlx_qwen35_model.cpp`: compiled model stores MoE layer state and runs the sparse-MoE block in `forward_impl`
  - `crates/mlx-sys/src/mlx_qwen35_moe_block.cpp`: array-native helper split from the C ABI wrapper
- Suspected cause of any remaining regression vs oMLX: compiled MoE removes the Rust fallback, but the packed serving path still needs more prefill optimization to match MLX-native baselines.
- Follow-ups:
  - Run the canonical `guidellm` sweep on a dedicated Apple Silicon host
  - Compare compiled-MoE prefill against the public oMLX `M4 Pro 48 GB` reference
  - Re-evaluate DFlash now that Qwen3.6 has a compiled-model target again
