# Qwen3-4B L4 c16 on `4eddda8` with unified CUDA scheduler budget

## Goal

- Regression + cleanup: collapse CUDA scheduler budget math into one shared module without materially moving long-context `Qwen3-4B` `c16` throughput or active-set behavior.

## Hypothesis

- The shared budget module should preserve the current `c16` behavior within noise because it rewires budgeting logic, not policy.
- A clean refactor should keep `peak active=10`, `peak waiting≈8`, and hold throughput within roughly `±3%` of the latest waiting-aware reclaim baseline.

## Command

```bash
export ZIG=/content/workspace/agent-infer/.toolchains/zig/zig-x86_64-linux-0.16.0/zig
cargo fmt --all
cargo test --release -p infer --lib scheduler::cuda:: -- --nocapture
cargo build --release -p infer --bin infer
cargo clippy --release -p infer --lib --bin infer -- -D warnings

MODEL_PATH=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

# run 1
 target/release/infer \
  --model-path "$MODEL_PATH" \
  --port 8031 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384 \
  --trace-output-path bench-output/infer-qwen3-4b-l4-c16-4eddda8-unified-budget-server/traces

GUIDELLM__MP_CONTEXT_TYPE=forkserver ./scripts/bench_guidellm.sh infer-qwen3-4b-l4-c16-4eddda8-unified-budget \
  --target http://127.0.0.1:8031 \
  --model Qwen/Qwen3-4B \
  --processor "$MODEL_PATH" \
  --concurrencies 16 \
  --max-seconds 60 \
  --warmup 5 \
  --trace-interval-ms 200

# run 2
 target/release/infer \
  --model-path "$MODEL_PATH" \
  --port 8032 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384 \
  --trace-output-path bench-output/infer-qwen3-4b-l4-c16-4eddda8-unified-budget-rerun-server/traces

GUIDELLM__MP_CONTEXT_TYPE=forkserver ./scripts/bench_guidellm.sh infer-qwen3-4b-l4-c16-4eddda8-unified-budget-rerun \
  --target http://127.0.0.1:8032 \
  --model Qwen/Qwen3-4B \
  --processor "$MODEL_PATH" \
  --concurrencies 16 \
  --max-seconds 60 \
  --warmup 5 \
  --trace-interval-ms 200
```

## Environment

- **Backend:** `cuda`
- **Model:** `Qwen/Qwen3-4B` bf16
- **Hardware:** `NVIDIA L4` `23034 MiB`, driver `580.82.07`, runtime CUDA `13.0`
- **Code state:** `4eddda8` plus local unified-budget refactor in `infer/src/scheduler/cuda/{budget,core,execution,runtime}.rs`
- **Feature set:** `cargo build --release -p infer --bin infer`
- **Non-default flags:** `--num-slots 16`, `--max-seq-len 4608`, `--mem-fraction-static 0.94`, `--chunked-prefill-size 4096`, `--max-prefill-tokens 16384`, `GUIDELLM__MP_CONTEXT_TYPE=forkserver`

## Results

| run | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual | completed output tok | incomplete input tok | incomplete output tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `run1` | `7107.4` | `49291.2` | `65.43` | `73.58` | `105.56` | `0.364` | `6630` | `40960` | `762` |
| `run2` | `7260.1` | `49791.5` | `65.59` | `73.92` | `104.75` | `0.364` | `6630` | `40960` | `761` |
| `mean` | `7183.8` | `49541.4` | `65.51` | `73.75` | `105.16` | `0.364` | `6630` | `40960` | `761.5` |

### Run-to-run stability

- Output-throughput spread across the two repeats: `0.77%`
- The repeated run is therefore inside the `<2%` stability target from `docs/bench-and-trace-spec.md`

### Service trace summary

| run | peak active | peak waiting | peak prefill queue | peak kv util |
|---|---:|---:|---:|---:|
| `run1` | `10` | `8` | `8` | `99.4%` |
| `run2` | `10` | `8` | `8` | `99.4%` |

## Problems

- The first attempt to start the repeat server failed with `CUDA_ERROR_OUT_OF_MEMORY` because the prior `infer` process had not exited cleanly yet and was still holding `19.8 GiB` on the L4. After killing that stale process, the repeat run was clean.
- `cargo test --release -p infer --lib` still has unrelated environment / fixture failures outside `scheduler::cuda` (CUDA init, model fixtures, and existing GPU test failures).
- `cargo clippy --release -p infer --lib --bin infer -- -D warnings` passes for `infer`, but package-wide `cargo clippy --release -p infer --all-targets -- -D warnings` is still blocked by a pre-existing lint in `crates/cuda-kernels/src/paged_kv.rs:841`.

## Learnings

- Unifying budget arithmetic does not by itself move the `c16` ceiling; the remaining throughput gap still lives in refill / host-tier pressure, not in divergent page math between admission and execution.
- The right refactor boundary is one shared budget module plus policy kept in the caller; that keeps formulas single-sourced without hiding scheduler intent.
- For local CUDA repeats, kill the prior `infer` explicitly before starting the next run; otherwise startup OOM can masquerade as a model-memory regression.

## Δ vs baseline

- **Prior local baseline:** `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-l4-c8-c16-39152ac-waiting-reclaim.md`
- **Reference trace diagnosis:** `docs/experience/wins/2026-04-22-profile-cuda-qwen3-4b-c16-end-to-end-bottleneck-39152ac-localfix.md`
- **Reference `sglang` baseline:** `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-infer-vs-sglang-c1-c16.md`

| metric | prior | now | Δ% |
|---|---:|---:|---:|
| `c16` out tok/s vs prior infer | `107.76` | `105.16` | `-2.4%` |
| `c16` TTFT p50 vs prior infer | `7120.8` | `7183.8` | `+0.9%` |
| `c16` ITL p50 vs prior infer | `62.74` | `65.51` | `+4.4%` |
| `c16` incomplete input tok vs prior infer | `40960` | `40960` | `+0.0%` |
| `c16` out tok/s vs `sglang` | `137.07` | `105.16` | `-23.3%` |

## Artefacts

- `run1` raw bench: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-4eddda8-unified-budget/benchmarks.json`
- `run1` service trace summary: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-4eddda8-unified-budget/service_stats_trace_summary.md`
- `run1` server traces: `bench-output/infer-qwen3-4b-l4-c16-4eddda8-unified-budget-server/traces`
- `run2` raw bench: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-4eddda8-unified-budget-rerun/benchmarks.json`
- `run2` service trace summary: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-4eddda8-unified-budget-rerun/service_stats_trace_summary.md`
- `run2` server traces: `bench-output/infer-qwen3-4b-l4-c16-4eddda8-unified-budget-rerun-server/traces`
