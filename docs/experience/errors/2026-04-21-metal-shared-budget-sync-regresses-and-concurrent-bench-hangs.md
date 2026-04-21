# Metal Shared Budget Sync Regresses And Concurrent Bench Hangs

## Goal

Regression check for the new Metal shared-budget mixed batching policy on
`mlx-community/Qwen3-0.6B-4bit` after unblocking the local `kv-native-sys` Zig
build.

## Hypothesis

Replacing the runtime-only mixed cap with a single shared scheduler budget
would keep low-pressure `sync` roughly flat and make the concurrent path simpler
without reintroducing the earlier mixed-batch collapse.

## Command

```bash
cargo check -p infer --release --no-default-features --features metal,no-cuda
cargo test -p infer --release --no-default-features --features metal,no-cuda backend::metal::scheduler::tests:: -- --nocapture
cargo test -p infer --release --no-default-features --features metal,no-cuda request_state -- --nocapture

target/release/metal_serve \
  --model-path mlx-community/Qwen3-0.6B-4bit \
  --port 8019 \
  --warmup 1

scripts/bench_guidellm.sh metal-m4pro-qwen3-0p6b-shared-budget-sync \
  --target http://127.0.0.1:8019 \
  --model mlx-community/Qwen3-0.6B-4bit \
  --processor /path/to/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/73e3e38d981303bc594367cd910ea6eb48349da8 \
  --profile synchronous \
  --max-seconds 60

scripts/bench_guidellm.sh metal-m4pro-qwen3-0p6b-shared-budget-quick \
  --target http://127.0.0.1:8019 \
  --model mlx-community/Qwen3-0.6B-4bit \
  --processor /path/to/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/73e3e38d981303bc594367cd910ea6eb48349da8 \
  --quick
```

## Environment

- Host: Apple M4 Pro / macOS 26.3.1
- Backend: Metal (`--no-default-features --features metal,no-cuda`)
- Model: `mlx-community/Qwen3-0.6B-4bit`
- Baseline for comparison:
  [2026-04-21-bench-guidellm-metal-m4pro-qwen3-0p6b-unified-bounded.md](../wins/2026-04-21-bench-guidellm-metal-m4pro-qwen3-0p6b-unified-bounded.md)
- Raw artifacts:
  - [shared-budget-sync](../../../../bench-output/2026-04-21-metal-m4pro-qwen3-0p6b-shared-budget-sync)
  - [shared-budget-quick](../../../../bench-output/2026-04-21-metal-m4pro-qwen3-0p6b-shared-budget-quick)

## Results

### Sync (`4096 in / 256 out`)

| profile | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---:|---:|---:|---:|---:|---:|
| shared-budget sync | 1032.8 | 1396.4 | 5.92 | 6.35 | 101.0 | 0.383 |
| prior unified-bounded | 941.9 | n/a | 5.97 | n/a | 104.76 | n/a |
| delta | +9.7% | n/a | -0.8% | n/a | -3.6% | n/a |

### Concurrent quick (`512 in / 128 out`)

- The run never produced `benchmarks.json` / `headline_table.md`.
- `guidellm` remained live for more than 3 minutes despite `--quick` using a
  60-second window and had to be killed manually.

## Problems

- The shared-budget `sync` path regressed relative to the committed rollback
  line: TTFT p50 moved from `941.9 ms` to `1032.8 ms` and output throughput fell
  from `104.76 tok/s` to `101 tok/s`.
- The concurrent quick benchmark did not terminate on its own.
- During the hanging concurrent run, the server log showed:

```text
ERROR infer::backend::metal::runtime: runtime.rs:984
Metal mixed decode post-process failed for RequestId(160): stream consumer dropped
```

## Root Cause

- The new scheduler policy is simpler, but it does **not** actually constrain
  mixed prefill very much: `max_batch_tokens - decode_count` still leaves almost
  the full `512`-token budget available to prefill once the decode side is only
  a handful of rows. In practice this means the refactor removed the runtime cap
  without replacing it with a decode-protecting budget rule.
- The concurrent hang appears tied to request teardown under mixed batching:
  the benchmark client dropped a stream consumer, the runtime logged the failed
  mixed decode post-process, and at least one request remained live long enough
  for `guidellm` to never emit final output. The exact cleanup bug is still not
  isolated from this run alone.

## Fix

- Keep the `kv-native-sys` Zig compatibility fix so local Metal verification can
  run on current Zig.
- Do **not** treat `decode row count` as a sufficient mixed-batch budget guard
  when `max_batch_tokens` remains large.
- Next iteration should preserve the single scheduler policy but reserve an
  explicit decode floor for mixed ticks, or lower the scheduler-level batch
  budget itself, instead of reintroducing a runtime-only cap.
- Investigate mixed cancellation/teardown around
  `Metal mixed decode post-process failed ... stream consumer dropped`.

## Rule

“Single policy” is not enough by itself. If a shared mixed-batch budget still
lets prefill consume almost the whole tick, the policy is simple but not yet
correct for latency protection.
