# Qwen3-4B L4 c8/c16 on `39152ac` with waiting-aware prefix-cache reclaim

## Goal

- Optimization + regression check: raise long-context `Qwen3-4B` CUDA throughput at `c8/c16` by reclaiming more prefix-cache pages when queued backlog is the real pressure, while keeping the eviction path single and uniform.

## Hypothesis

- The scheduler should recover a materially larger active set at `c16` by reclaiming beyond the fixed low-water mark when queued requests still need GPU page headroom.
- `c8` should stay at least neutral, and ideally move closer to the `sglang` baseline.
- The remaining bottleneck should shift from prefix-cache retained-page underfill toward host-tier saturation / fallback.

## Command

```bash
export ZIG=/content/workspace/agent-infer/.toolchains/zig/zig-x86_64-linux-0.16.0/zig
cargo fmt --all
cargo test --release -p infer --lib waiting_admission_shortage -- --nocapture
cargo test --release -p infer --lib reclaim_goal_can_trigger_below_watermark_when_waiting_needs_headroom -- --nocapture
cargo build --release -p infer --bin infer

MODEL_PATH=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

target/release/infer \
  --model-path "$MODEL_PATH" \
  --port 8027 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384

GUIDELLM__MP_CONTEXT_TYPE=forkserver ./scripts/bench_guidellm.sh infer-qwen3-4b-l4-c16-39152ac-waiting-reclaim \
  --target http://127.0.0.1:8027 \
  --model Qwen/Qwen3-4B \
  --processor "$MODEL_PATH" \
  --concurrencies 16 \
  --max-seconds 60 \
  --warmup 5 \
  --trace-interval-ms 200

target/release/infer \
  --model-path "$MODEL_PATH" \
  --port 8028 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384

GUIDELLM__MP_CONTEXT_TYPE=forkserver ./scripts/bench_guidellm.sh infer-qwen3-4b-l4-c8-39152ac-waiting-reclaim \
  --target http://127.0.0.1:8028 \
  --model Qwen/Qwen3-4B \
  --processor "$MODEL_PATH" \
  --concurrencies 8 \
  --max-seconds 60 \
  --warmup 5 \
  --trace-interval-ms 200
```

## Environment

- **Backend:** `cuda`
- **Model:** `Qwen/Qwen3-4B` bf16
- **Hardware:** `NVIDIA L4` `23034 MiB`, driver `580.82.07`, runtime CUDA `13.0`
- **Code state:** `39152ac` plus local waiting-aware reclaim patch in `infer/src/scheduler/cuda/core.rs`
- **Feature set:** `cargo build --release -p infer --bin infer`
- **Non-default flags:** `--num-slots 16`, `--max-seq-len 4608`, `--mem-fraction-static 0.94`, `--chunked-prefill-size 4096`, `--max-prefill-tokens 16384`, `GUIDELLM__MP_CONTEXT_TYPE=forkserver`

## Results

| concurrency | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual | completed output tok | incomplete input tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `c8` | `5328.7` | `20819.8` | `54.85` | `65.97` | `103.22` | `0.382` | `5610` | `28672` |
| `c16` | `7120.8` | `49566.8` | `62.74` | `73.59` | `107.76` | `0.364` | `6630` | `40960` |

### Service trace summary

| concurrency | peak active | peak waiting | peak prefill queue | peak kv util |
|---|---:|---:|---:|---:|
| `c8` | `8` | `2` | `7` | `91.6%` |
| `c16` | `10` | `8` | `8` | `98.2%` |

## Problems

- The reclaim patch is intentionally throughput-biased: ITL regressed versus the earlier `39152ac-localfix` baseline (`c8` `46.09 → 54.85 ms`, `c16` `46.12 → 62.74 ms`).
- `c8` completed-throughput improved, but unfinished input tokens also rose (`12288 → 28672`), so this is not a free win.
- The host pinned tier still saturates under long-context backlog, so the scheduler repeatedly falls back to dropping GPU radix blocks after partial demotion. Example: `c16` reclaimed `1280` pages and drove `retained now 0`, but only after host-tier-full fallback, see `bench-output/infer-qwen3-4b-l4-c16-39152ac-waiting-reclaim-server/infer.log`.
- `c16` still trails `sglang` materially on throughput.

## Learnings

- The important lever was the **reclaim target**, not a second eviction policy: keeping one `SessionBiasedLru` path but widening the page budget on backlog is enough to recover throughput.
- Under this workload, fixed low-water cleanup is too conservative; it parks enough GPU pages in the prefix cache to starve the next admission wave.
- After backlog-aware reclaim, the next limiting factor is host-tier capacity, not retained-page underfill.

## Δ vs baseline

- **Prior local baseline:** `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-l4-c4-c8-c16-39152ac-localfix.md`
- **Reference trace diagnosis:** `docs/experience/wins/2026-04-22-profile-cuda-qwen3-4b-c16-end-to-end-bottleneck-39152ac-localfix.md`
- **Reference `sglang` baseline:** `docs/experience/wins/2026-04-22-bench-guidellm-qwen3-4b-infer-vs-sglang-c1-c16.md`

| metric | prior infer | now | Δ% |
|---|---:|---:|---:|
| `c8` out tok/s | `85.11` | `103.22` | `+21.3%` |
| `c16` out tok/s | `95.02` | `107.76` | `+13.4%` |
| `c8` vs `sglang` out tok/s | `107.79` | `103.22` | `-4.2%` |
| `c16` vs `sglang` out tok/s | `137.07` | `107.76` | `-21.4%` |
| `c8` incomplete input tok | `12288` | `28672` | `+133.3%` |
| `c16` incomplete input tok | `45056` | `40960` | `-9.1%` |

## Artefacts

- `c8` raw bench: `bench-output/2026-04-22-infer-qwen3-4b-l4-c8-39152ac-waiting-reclaim/benchmarks.json`
- `c8` service trace summary: `bench-output/2026-04-22-infer-qwen3-4b-l4-c8-39152ac-waiting-reclaim/service_stats_trace_summary.md`
- `c8` server log: `bench-output/infer-qwen3-4b-l4-c8-39152ac-waiting-reclaim-server/infer.log`
- `c16` raw bench: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-39152ac-waiting-reclaim/benchmarks.json`
- `c16` service trace summary: `bench-output/2026-04-22-infer-qwen3-4b-l4-c16-39152ac-waiting-reclaim/service_stats_trace_summary.md`
- `c16` server log: `bench-output/infer-qwen3-4b-l4-c16-39152ac-waiting-reclaim-server/infer.log`
