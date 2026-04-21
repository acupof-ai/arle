# guidellm sweep metal-m4pro-qwen3-0p6b-unified-bounded — bounded single-gate Metal scheduler, 2026-04-21

## Goal

- Canonical long-prompt regression-check for the simplified Metal Qwen3 scheduler/runtime after removing duplicate admission, moving Qwen3/Qwen3.5 prefill onto the session-owned C++ path, and pruning the legacy concurrent-DFlash downgrade branch.

## Hypothesis

- The unified scheduler should keep the improved low-pressure path from the earlier prefill-batching fix.
- A bounded single waiting gate should replace allocator/session failures and runaway queue drain with explicit HTTP backpressure during the `throughput` leg.

## Command

```bash
scripts/bench_guidellm.sh metal-m4pro-qwen3-0p6b-unified-bounded \
  --target http://127.0.0.1:8019 \
  --model mlx-community/Qwen3-0.6B-4bit \
  --processor models/Qwen3-0.6B
```

Server launch:

```bash
target/release/metal_serve \
  --model-path mlx-community/Qwen3-0.6B-4bit \
  --port 8019 \
  --warmup 1
```

## Environment

- **Backend:** metal
- **Model:** `mlx-community/Qwen3-0.6B-4bit`
- **Hardware:** Apple M4 Pro, 48 GB unified memory, macOS 26.3.1
- **Commit:** `ef4ec4a`
- **Feature set:** `cargo build --release --no-default-features --features metal,no-cuda`
- **Non-default flags / env vars:** none
- **Server launch:** `target/release/metal_serve --model-path mlx-community/Qwen3-0.6B-4bit --port 8019 --warmup 1`

## Canonical params

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json,csv,html`
- Wrapper: `scripts/bench_guidellm.sh <backend-label>`

## Results — sweep headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| sync | 941.9 | 1122.5 | 5.97 | 6.87 | 104.76 | 0.4 |
| throughput | 28559.9 | 50588.8 | 33.48 | 39.45 | 91.42 | 0.333 |
| 0.39166666666666666r/s | 961.1 | 1019 | 6.08 | 6.16 | 102.11 | 0.383 |
| 0.38333333333333336r/s | 947.6 | 1041.2 | 5.99 | 6.29 | 99.94 | 0.383 |
| 0.375r/s | 946.6 | 1040 | 6.06 | 6.23 | 97.88 | 0.367 |
| 0.3666666666666667r/s | 954 | 1026.4 | 5.89 | 7.72 | 95.49 | 0.367 |
| 0.35833333333333334r/s | 970.8 | 1031.7 | 6.07 | 6.27 | 93.64 | 0.35 |
| 0.35r/s | 994.6 | 1115 | 6.2 | 7.72 | 91.08 | 0.35 |
| 0.3416666666666667r/s | 970 | 1029.7 | 6.07 | 6.43 | 89.41 | 0.333 |
| 0.3333333333333333r/s | 1005.4 | 1056.7 | 5.88 | 6.12 | 87.42 | 0.333 |

## Problems

- The `throughput` leg still is not a user-facing latency number:
  - `TTFT p50 28.6 s`
  - `TTFT p99 50.6 s`
  - `ITL p50 33.5 ms`
- The bounded queue now sheds load explicitly during that leg instead of trying to absorb the whole burst. Server logs show repeated:
  - `Scheduler at capacity: request submission failed`
- The throughput summary confirms that behavior:
  - completed input tokens: `81,940`
  - incomplete input tokens: `1,064,960`
  - errored input tokens: `20,676,608`
- `cargo clippy -p infer --release --no-default-features --features metal,no-cuda -- -D warnings` still fails on pre-existing unrelated lints in `infer/src/backend/metal/dflash.rs`, `request_state.rs`, `runtime.rs`, `kv_pool.rs`, `mlx.rs`, and `metal.rs`. This diff did not attempt a repository-wide lint cleanup.

## Learnings

- The simplified single-gate scheduler preserved and improved the low-pressure path:
  - canonical `sync` moved from `TTFT p50 1135.4 ms / 90.42 tok/s` to `941.9 ms / 104.76 tok/s`
  - the constant-rate legs now stay in the `87-102 tok/s` band instead of the earlier `71-85 tok/s` band
- The new Qwen3/Qwen3.5 session-owned prefill path plus the scheduler/runtime cleanup removed the old hard failures from the canonical run:
  - no `metal::malloc Resource limit (499000) exceeded`
  - no `session_begin` failure
  - no runtime panic containment event was needed
- Raising the single waiting cap too high is not a real optimization. A local `max_waiting=1024` sweep removed rejections but stretched the run into a very long post-window drain. Keeping one explicit backlog cap and letting throughput shed load is both simpler and healthier.
- The remaining weakness is now obvious and narrower: long-prompt throughput is backpressure-limited, not allocator-limited. The next optimization should target how much useful work the server can admit under pressure, not another round of allocator panic triage.

## Δ vs baseline

- **Baseline:** [2026-04-21-bench-guidellm-metal-m4pro-qwen3-0p6b-prefill-batch-fix.md](./2026-04-21-bench-guidellm-metal-m4pro-qwen3-0p6b-prefill-batch-fix.md)

| metric | baseline | now | Δ% |
|---|---|---|---|
| TTFT p50 @ synchronous | 1135.4 ms | 941.9 ms | -17.0% |
| out tok/s @ synchronous | 90.42 | 104.76 | +15.9% |
| TTFT p50 @ throughput | 27342.6 ms | 28559.9 ms | +4.5% |
| out tok/s @ throughput | 79.68 | 91.42 | +14.7% |

## Artefacts

- Raw: `/path/to/code/agent-infer/bench-output/2026-04-21-metal-m4pro-qwen3-0p6b-unified-bounded/benchmarks.json`
- CSV:  `/path/to/code/agent-infer/bench-output/2026-04-21-metal-m4pro-qwen3-0p6b-unified-bounded/benchmarks.csv`
- HTML: `/path/to/code/agent-infer/bench-output/2026-04-21-metal-m4pro-qwen3-0p6b-unified-bounded/benchmarks.html`

## Notes

- What changed in the code since baseline:
  - `crates/mlx-sys/src/mlx_qwen35_model.cpp`, `crates/mlx-sys/src/lib.rs`, `infer/src/backend/metal/qwen35.rs`, and `infer/src/backend/metal/request_state.rs` now run multi-token prefill through a session-owned compiled C++ path instead of round-tripping full KV handles on every chunk.
  - `infer/src/backend/metal/runtime.rs` and `infer/src/backend/metal/scheduler.rs` now use one decode-first `MetalScheduleStep` path with one waiting gate instead of split admission logic and the legacy concurrent-DFlash downgrade switch.
  - `infer/src/bin/metal_serve.rs` keeps a bounded default `--max-waiting=256`, and `infer/src/http_server.rs` now reports submit-full as capacity backpressure rather than an internal error.
- Suspected cause of the remaining `throughput` regression:
  - open-loop long-prompt traffic still outruns what the Metal backend can complete inside the 60-second measurement window, so the bounded queue starts rejecting aggressively;
  - completed-request throughput improved, but tail latency stays dominated by the requests that do make it into the queue.
- Follow-ups:
  - replace per-request `Scheduler at capacity` warning spam with a rate-limited counter/log path if we keep using HTTP 503 backpressure in throughput benches;
  - quantify whether a smaller or model-aware single waiting cap improves completed-request throughput without regressing the healthy sync/constant legs;
  - only after the queueing policy is settled, revisit whether promoting more DFlash behavior into the main decode path is still worth the complexity on Metal.
