# Metal Qwen3-0.6B Phase Metrics And Unified Prefill Policy

## Goal

Optimization. Measure where long-prompt Metal serving time is actually spent, then remove one scheduler policy layer if the data says queueing dominates.

## Hypothesis

`4096 / 256` throughput on Metal is mostly paying for scheduler-side queueing rather than single-request active service time. If that is true, the extra `decode_active_prefill_cap` shrink path is the wrong complexity: keeping `decode-first` but deleting the smaller decode-era prefill chunk should reduce TTFT and capacity warnings without hurting single-request decode materially.

## Command

Server:

```bash
cargo build -p infer --release --no-default-features --features metal,no-cuda --bin metal_serve
target/release/metal_serve --model-path mlx-community/Qwen3-0.6B-4bit --port 8019 --warmup 1
```

Before:

```bash
scripts/bench_guidellm.sh metal-m4pro-qwen3-0p6b-phase-metrics \
  --target http://127.0.0.1:8019 \
  --model mlx-community/Qwen3-0.6B-4bit \
  --processor /path/to/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/73e3e38d981303bc594367cd910ea6eb48349da8 \
  --profile sweep \
  --max-seconds 60
```

After:

```bash
scripts/bench_guidellm.sh metal-m4pro-qwen3-0p6b-phase-metrics-after \
  --target http://127.0.0.1:8019 \
  --model mlx-community/Qwen3-0.6B-4bit \
  --processor /path/to/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/73e3e38d981303bc594367cd910ea6eb48349da8 \
  --profile sweep \
  --max-seconds 60
```

Service-side snapshots:

```bash
curl -sSf http://127.0.0.1:8019/metrics > /tmp/metal_qwen3_phase_metrics.prom
curl -sSf http://127.0.0.1:8019/v1/stats
rg -c 'Scheduler at capacity: request submission failed' /tmp/metal_qwen3_phase_metrics*.log
```

Note: `--profile sweep --max-seconds 60` forces GuideLLM exploration mode, but the workload itself stayed at the canonical `4096 / 256` sweep parameters.

## Environment

- Host: `Apple M4 Pro`
- RAM: `48 GB`
- OS kernel: `Darwin 25.3.0`
- Repo base during run: `06e7e17`
- Build: `cargo build -p infer --release --no-default-features --features metal,no-cuda --bin metal_serve`
- Model: `mlx-community/Qwen3-0.6B-4bit`
- Backend path: Metal unified scheduler runtime

## Results

Raw artefacts:

- Before: [bench-output/2026-04-21-metal-m4pro-qwen3-0p6b-phase-metrics](/path/to/code/agent-infer/bench-output/2026-04-21-metal-m4pro-qwen3-0p6b-phase-metrics)
- After: [bench-output/2026-04-21-metal-m4pro-qwen3-0p6b-phase-metrics-after](/path/to/code/agent-infer/bench-output/2026-04-21-metal-m4pro-qwen3-0p6b-phase-metrics-after)

GuideLLM headline deltas:

| Workload | Metric | Before | After | Delta |
|---|---:|---:|---:|---:|
| `sync` | TTFT p50 | `956.0 ms` | `899.5 ms` | `-5.9%` |
| `sync` | out tok/s | `103.04` | `109.57` | `+6.3%` |
| `throughput` | TTFT p50 | `25496.6 ms` | `23829.9 ms` | `-6.5%` |
| `throughput` | ITL p50 | `32.26 ms` | `32.26 ms` | `flat` |
| `throughput` | out tok/s | `98.03` | `99.37` | `+1.4%` |

Service-side phase metrics after the full run:

| Metric | Before | After |
|---|---:|---:|
| mean `queue_wait` | `2.12 s` | `2.36 s` |
| mean `active_ttft` | `0.99 s` | `0.97 s` |
| mean `service` | `2.13 s` | `2.81 s` |
| mean `e2e` | `5.24 s` | `6.14 s` |
| capacity warnings | `4909` | `4680` |

Observed `/v1/stats` during the after throughput window:

```text
requests=69 active=4 waiting=1 tokens_out=17409 ... queue_p50=10.0ms active_ttft_p50=2000.0ms ttft_p50=2000.0ms ttft_p99=60000.0ms service_p50=5000.0ms tpot_p50=15.0ms
```

Interpretation:

- Single-request active service is still around `~1s` to first token for long prompts.
- Throughput-mode TTFT is still an order of magnitude larger than that, so queueing remains the dominant contributor under load.
- Removing the smaller decode-era prefill chunk did reduce synchronous TTFT and reduced capacity warnings modestly, but it did not eliminate queue dominance.

## Problems

- The new phase metrics exposed a pre-existing bug in [`infer/src/metrics.rs`](/path/to/code/agent-infer/infer/src/metrics.rs): histogram buckets were being accumulated twice. That had to be fixed before `/metrics` percentiles were trustworthy.
- `cargo clippy -p infer --release --no-default-features --features metal,no-cuda -- -D warnings` was also blocked by unrelated small API/lint gaps in `kv-native-sys` and a few Metal CLI bins; those were fixed as part of restoring a clean verification baseline.
- The after sweep showed a stability warning on one constant-rate point (`ITL p99 > 2x p50`). The main sync/throughput comparisons remained directionally stable.

## Learnings

- On this workload, Metal throughput TTFT is not primarily a GPU-single-step problem. `active_ttft` stays around `~1s`; the rest is scheduler queueing and request backlog.
- Deleting a policy layer can help even when the gain is modest. Removing the decode-specific prefill shrink path improved the two top-line benchmarks that matter here (`sync` and `throughput`) while simplifying the scheduler.
- Phase metrics are now mandatory before changing Metal scheduling again. Without `queue_wait / active_ttft / service`, it is too easy to misdiagnose queueing as kernel latency.
- Capacity warnings are now a useful pressure signal. They dropped slightly here, which means the unified policy moved in the right direction, but not enough to call long-prompt throughput solved.

## Δ vs baseline

Compared against [2026-04-21-bench-guidellm-metal-m4pro-qwen3-0p6b-unified-bounded.md](/path/to/code/agent-infer/docs/experience/wins/2026-04-21-bench-guidellm-metal-m4pro-qwen3-0p6b-unified-bounded.md) for the latest committed long-prompt Metal baseline, plus the direct before/after exploration pair above.
