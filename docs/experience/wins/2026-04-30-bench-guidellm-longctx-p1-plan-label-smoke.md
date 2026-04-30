# Plan-label stats smoke — guidellm smoke, longctx-p1-plan-label-smoke, 2026-04-30

## Goal

- Add machine-checkable scheduler plan counters so Phase 1 S5 can prove
  `Mixed > 0` and `Split = 0` from `/v1/stats` instead of relying on log grep.

## Hypothesis

- Lifetime plan counters can be exposed with one atomic increment per planned
  scheduler tick and no change to request behavior.

## Command

```bash
scripts/bench_guidellm.sh longctx-p1-plan-label-smoke \
  --workload longctx-32k \
  --smoke \
  --target http://127.0.0.1:8000 \
  --model Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

## Environment

- **Backend:** CUDA
- **Model:** Qwen3-4B
- **Hardware:** NVIDIA L4, 23034 MiB VRAM, driver 580.82.07, CUDA 13.0 host,
  nvcc from `/usr/local/cuda/bin/nvcc`
- **Commit:** local uncommitted diff after `bce0754a`
- **Feature set:** `cargo build -p infer --release --no-default-features --features cuda --bin infer`
- **Non-default flags / env vars:** `CUDA_HOME=/usr/local/cuda`,
  `PATH=/usr/local/cuda/bin:$PATH`, `TORCH_CUDA_ARCH_LIST=8.9`,
  `INFER_TRITON_PYTHON=/usr/bin/python3`,
  `INFER_TILELANG_PYTHON=/usr/bin/python3`,
  `CARGO_HOME=/tmp/arle-cargo-home`, `CARGO_TARGET_DIR=/tmp/arle-target`,
  `ZIG=/content/workspace/agent-infer/.toolchains/zig/zig-x86_64-linux-0.16.0/zig`
- **Server launch:** `/tmp/arle-target/release/infer --model-path infer/models/Qwen3-4B --port 8000 --kv-cache-dtype fp8 --num-slots 16 --max-seq-len 131072 --mem-fraction-static 0.85 --max-num-batched-tokens 16384 --max-prefill-tokens 16384 --schedule-policy fcfs`

## Canonical params

- Smoke mode: `prompt_tokens=512`, `output_tokens=16`, `rate=1`,
  `max_seconds=5`
- Workload wrapper: `--workload longctx-32k --smoke`
- Service trace interval: `1000ms`

## Results

| rate | out tok/s | total tok/s | req/s | TTFT p50 | ITL p50 |
|---|---:|---:|---:|---:|---:|
| conc1 | 26.69 | 882.52 | 1.60 | 100.9 ms | 34.0 ms |

## Service Trace Peaks

- Samples: `35` (ok: `35`, failed: `0`)
- Peak active: `1`
- Peak running_batch: `1`
- Plan labels: `idle=300996`, `decode=128`, `prefill=10`, `split=0`, `mixed=0`
- Peak kv_util: `3.4%`

## Problems

- `idle` is intentionally noisy: while a request is active but a tick has no
  launchable GPU work, the scheduler can choose `Idle` quickly. The useful S5
  gate is still the monotonic `split` and `mixed` totals.
- Smoke has no mixed batches by shape, so `mixed=0` here is expected. The S5
  rerun is the acceptance point for `Mixed > 0`.

## Learnings

- `/v1/stats` now includes
  `plan_label=idle:N,decode:N,prefill:N,split:N,mixed:N`.
- `/metrics` now includes `infer_scheduler_plan_total{plan="..."}` counters.
- `scripts/bench_guidellm.sh` trace summaries now lift these counters into a
  `Plan labels:` row.

## Verification

```bash
cargo fmt --check
ZIG=/content/workspace/agent-infer/.toolchains/zig/zig-x86_64-linux-0.16.0/zig \
  cargo test -p infer metrics::tests::server_metrics_prometheus_render --release --no-default-features --features no-cuda
ZIG=/content/workspace/agent-infer/.toolchains/zig/zig-x86_64-linux-0.16.0/zig \
  cargo test -p infer metrics::tests::server_metrics_render_summary --release --no-default-features --features no-cuda
ZIG=/content/workspace/agent-infer/.toolchains/zig/zig-x86_64-linux-0.16.0/zig \
  cargo test -p infer endpoint_returns --release --no-default-features --features no-cuda
ZIG=/content/workspace/agent-infer/.toolchains/zig/zig-x86_64-linux-0.16.0/zig \
  cargo clippy -p infer --release --no-default-features --features no-cuda -- -D warnings
```

Initial attempts without `ZIG=...` failed in `kv-native-sys` build script; reruns
with the local Zig toolchain passed.

## Artefacts

- Raw: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-p1-plan-label-smoke/benchmarks.json`
- CSV: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-p1-plan-label-smoke/benchmarks.csv`
- HTML: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-p1-plan-label-smoke/benchmarks.html`
- Service trace summary: `/content/workspace/agent-infer/bench-output/2026-04-30-longctx-p1-plan-label-smoke/service_stats_trace_summary.md`
