# guidellm sweep metal-m4pro-qwen3-0p6b-prefill-batch-fix — canonical local sweep, 2026-04-21

## Goal

- Canonical long-prompt regression-check for the Metal Qwen3 scheduler/runtime fixes that batched prompt prefill and contained runtime panics, using `Qwen3-0.6B-4bit` on the full `4096 / 256` guidellm sweep.

## Hypothesis

- The prefill batching fix should collapse the catastrophic long-prompt TTFT seen before the change and make the full sweep complete locally.
- The runtime panic containment should keep allocator/resource-limit failures from killing `metal_serve`, but it will not by itself fix throughput-leg admission pressure.

## Command

```bash
scripts/bench_guidellm.sh metal-m4pro-qwen3-0p6b-prefill-batch-fix \
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
- **Commit:** `0079d62`
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
| sync | 1135.4 | 1743.9 | 6.07 | 11.39 | 90.42 | 0.333 |
| throughput | 27342.6 | 50493.2 | 37.14 | 38.55 | 79.68 | 0.267 |
| 0.32499999999999996r/s | 1195.3 | 1442.7 | 6.01 | 10.49 | 85.23 | 0.317 |
| 0.31666666666666665r/s | 1196.1 | 1221.9 | 6.11 | 6.16 | 83.27 | 0.317 |
| 0.30833333333333335r/s | 1183.6 | 1247.6 | 6.05 | 6.11 | 81.13 | 0.300 |
| 0.3r/s | 1140.3 | 1203.2 | 5.81 | 5.95 | 79.17 | 0.300 |
| 0.29166666666666663r/s | 1174.9 | 1260.5 | 6.05 | 6.29 | 76.96 | 0.283 |
| 0.2833333333333333r/s | 1135.1 | 1155.0 | 5.63 | 5.68 | 75.12 | 0.283 |
| 0.275r/s | 1135.3 | 1149.8 | 5.62 | 5.74 | 73.03 | 0.267 |
| 0.26666666666666666r/s | 1135.7 | 1151.5 | 5.64 | 5.70 | 70.99 | 0.267 |

## Problems

- The `throughput` leg still drives the runtime into pathological queueing: `TTFT p50 27.3 s`, `TTFT p99 50.5 s`, `ITL p50 37.1 ms`.
- Server logs showed a contained Metal allocator failure during the run:
  - `mlx_array_from_data returned a null MLX handle: [metal::malloc] Resource limit (499000) exceeded.`
- The new runtime panic containment worked as intended: the affected request/batch was aborted, the server stayed alive, and the sweep completed.
- This is the first canonical sweep after the fix, so there is no like-for-like prior `sweep` wins snapshot to diff against.

## Learnings

- The prefill batching fix solved the primary long-prompt regression. On the canonical `4096 / 256` shape, low-pressure legs now cluster around `TTFT ~1.14-1.20 s` and `70.99-90.42 out tok/s` instead of being effectively unusable.
- Panic containment is now sufficient for survivability but not for throughput quality. The server can finish the sweep under allocator pressure, but admission still allows the `throughput` profile to create a massive queue and terrible user latency.
- For this Metal path, the next bottleneck is not prompt graph materialization anymore. It is admission/backpressure under long prompts, followed by resource-count pressure inside MLX/Metal allocation.

## Δ vs baseline

- **Baseline:** first canonical `sweep` snapshot for this exact post-fix label; no prior like-for-like wins entry exists.
- **Nearest pre-fix long-prompt reference:** `bench-output/2026-04-21-metal-m4pro-qwen3-0p6b-local-c12/`
  - That earlier local `concurrent` run had `TTFT p50 33131.8 ms @ conc1` and `conc2` failed to complete.
  - It is useful as a qualitative before/after for survivability and TTFT shape, but not a strict same-profile delta.

## Artefacts

- Raw: `/path/to/code/agent-infer/bench-output/2026-04-21-metal-m4pro-qwen3-0p6b-prefill-batch-fix/benchmarks.json`
- CSV:  `/path/to/code/agent-infer/bench-output/2026-04-21-metal-m4pro-qwen3-0p6b-prefill-batch-fix/benchmarks.csv`
- HTML: `/path/to/code/agent-infer/bench-output/2026-04-21-metal-m4pro-qwen3-0p6b-prefill-batch-fix/benchmarks.html`

## Notes

- What changed in the code since the failing pre-fix runs:
  - [`infer/src/backend/metal/request_state.rs`](../../../infer/src/backend/metal/request_state.rs) now batches Qwen3 prompt chunks through `run_tokens(...)` instead of building one-token prefill graphs.
  - [`infer/src/backend/metal/runtime.rs`](../../../infer/src/backend/metal/runtime.rs) now catches panics around prefill/decode execution and aborts only the affected work.
  - [`docs/experience/wins/2026-04-21-bench-guidellm-metal-qwen3-0p6b-prefill-batch-fix-local.md`](./2026-04-21-bench-guidellm-metal-qwen3-0p6b-prefill-batch-fix-local.md) captured the quick local regression-check before this canonical sweep.
- Suspected cause of the remaining bad `throughput` leg:
  - long-prompt admission is still too permissive, so `guidellm` can drive huge queued concurrency before the runtime pushes back;
  - once the queue is large enough, MLX allocation count pressure shows up as `metal::malloc` resource-limit failures.
- Follow-ups:
  - add admission/backpressure limits for long-prompt Metal serving before trusting `throughput` as a user-facing number;
  - investigate how many live arrays/buffers each queued prefill/decode request holds and where they can be collapsed or reused;
  - only after that, judge whether promoting DFlash into the main path is the next highest-leverage optimization.
