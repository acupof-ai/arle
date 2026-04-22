# Qwen3-4B L4 c4/c8/c16 on `39152ac` (`guidellm`, local compile fix)

## Goal

- Regression-check the latest pulled `main` on the same L4 long-context `Qwen3-4B` `c4/c8/c16` workload.

## Hypothesis

- The new observability / emit-gate tranche should be throughput-neutral on `Qwen3-4B`; if anything moves, it should stay within noise of the `4848cd1-localfix` run.

## Command

```bash
export PATH="/root/.local/bin:$PATH"
export ZIG=/content/workspace/agent-infer/.toolchains/zig/zig-x86_64-linux-0.16.0/zig
cargo build --release -p infer --bin infer

MODEL_PATH=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

CUDA_HOME=/usr/local/cuda ./target/release/infer \
  --model-path "$MODEL_PATH" \
  --port 8064 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94

./scripts/bench_guidellm.sh 2026-04-22-infer-qwen3-4b-l4-c4-39152ac \
  --target http://127.0.0.1:8064 \
  --model Qwen/Qwen3-4B \
  --processor "$MODEL_PATH" \
  --concurrencies 4 \
  --max-seconds 60

# repeat the same server shape for c8 on :8068 and c16 on :8076
```

## Environment

- **Backend:** `cuda`
- **Model:** `Qwen/Qwen3-4B` bf16
- **Weights path:** `/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c`
- **Hardware:** NVIDIA L4 24GB, CUDA 12.8
- **Commit:** `39152ac`
- **Local code delta:** one-line compile fix in `infer/src/scheduler/cuda/core.rs`
- **Feature set:** `cargo build --release -p infer --bin infer`
- **Non-default flags:** `--num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94`

## Results

| leg | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | out tok/s | req/s actual | completed out tok | incomplete in tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `c4` | `2837.4` | `14620.9` | `43.43` | `43.58` | `70.86` | `0.233` | `4590` | `0` |
| `c8` | `5813.9` | `29754.6` | `46.09` | `66.36` | `85.11` | `0.250` | `5100` | `12288` |
| `c16` | `7536.0` | `42797.8` | `46.12` | `77.60` | `95.02` | `0.333` | `6375` | `45056` |

Service-trace headline:

| leg | peak active | peak waiting | peak prefill queue | peak kv util |
|---|---:|---:|---:|---:|
| `c4` | `4` | `2` | `2` | `93.6%` |
| `c8` | `8` | `6` | `7` | `99.0%` |
| `c16` | `10` | `11` | `9` | `99.0%` |

Artefacts:

- `bench-output/2026-04-22-2026-04-22-infer-qwen3-4b-l4-c4-39152ac/benchmarks.json`
- `bench-output/2026-04-22-2026-04-22-infer-qwen3-4b-l4-c8-39152ac/benchmarks.json`
- `bench-output/2026-04-22-2026-04-22-infer-qwen3-4b-l4-c16-39152ac/benchmarks.json`
- `bench-output/2026-04-22-2026-04-22-infer-qwen3-4b-l4-c4-39152ac/service_stats_trace_summary.md`
- `bench-output/2026-04-22-2026-04-22-infer-qwen3-4b-l4-c8-39152ac/service_stats_trace_summary.md`
- `bench-output/2026-04-22-2026-04-22-infer-qwen3-4b-l4-c16-39152ac/service_stats_trace_summary.md`

## Problems

- `39152ac` did not compile as-is on this host. `queue_emit_finish()` in `infer/src/scheduler/cuda/core.rs` destructured a 6-tuple from a 5-element map result; a one-line local fix was required before benching.
- The `c8/c16` tail is still not clean. Incomplete input tokens remain non-zero (`12288`, `45056`), so the scheduler still leaves backlog in the long-context shape.
- `c16` still underfills the active set despite `16` slots: service trace peaks at only `10` active rows while waiting climbs to `11`.

## Learnings

- The latest emit/observability tranche is effectively throughput-neutral on `Qwen3-4B` long-context `c4/c8/c16`; the numbers stay within noise of the previous `4848cd1-localfix` run.
- The limiting factor is still not single-token decode. `ITL p50` stays flat around `46 ms`; the remaining gap is the same active-set / prefill-queue / tail-backlog problem at `c8+`.
- On this workload, `peak kv_util ≈ 99%` plus `peak active < configured slots` is still the signature for scheduler underfill rather than a kernel ceiling.

## Δ vs baseline

- **Baseline:** `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-l4-c4-c8-c16-4848cd1-localfix.md`
- **SGLang reference:** `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-infer-vs-sglang-c1-c16.md`

| metric | baseline | now | Δ% |
|---|---:|---:|---:|
| `c4` out tok/s vs `4848cd1-localfix` | `70.77` | `70.86` | `+0.1%` |
| `c8` out tok/s vs `4848cd1-localfix` | `86.51` | `85.11` | `-1.6%` |
| `c16` out tok/s vs `4848cd1-localfix` | `95.39` | `95.02` | `-0.4%` |
| `c4` out tok/s vs `sglang` | `74.05` | `70.86` | `-4.3%` |
| `c8` out tok/s vs `sglang` | `107.79` | `85.11` | `-21.0%` |
| `c16` out tok/s vs `sglang` | `137.07` | `95.02` | `-30.7%` |
