# Metal resource guard: fail closed before macOS swap/SSD pressure

## Context

During a local Qwen3.6 Metal performance probe, starting the server without a
startup resource guard pushed the Mac into severe unified-memory pressure. The
failure mode is not a clean device OOM: macOS compresses memory and spills to
SSD-backed swap, which can stall the whole system.

## What Worked

- Added a Metal resource plan before weight load:
  - reads model weight bytes from local weight files;
  - reads physical memory via `sysctl hw.memsize`;
  - reads current free/inactive/speculative pages via `vm_stat`;
  - reads current swap usage via `sysctl vm.swapusage`;
  - samples `vm_stat` for 1 second and rejects startup when pageout/swapout
    deltas are actively high;
  - treats residual `swap_used` as a warning/status field, not a hard failure;
  - rejects startup when current available memory is below the anti-swap
    reserve;
  - sets MLX memory/cache/wired limits before loading weights;
  - clamps scheduler-visible KV pages and max token admission to the memory
    budget.
- Added CLI knobs:
  - `--memory-budget-bytes BYTES`;
  - `--system-reserve-bytes BYTES`;
  - `--allow-swap` as an explicit danger escape hatch.
- Kept this below the unified service/scheduler layer: `infer-api` carries
  neutral budget fields; `infer-metal` owns macOS/MLX-specific policy.

## Verification

Large-model startup is now gated by active paging and budget, not residual swap.

| Check | Result |
| --- | --- |
| `cargo test -p infer-metal --release --no-default-features --features metal -- --nocapture` | PASS, 17 passed |
| `cargo check -p infer-api --release --no-default-features --features metal,no-cuda --lib` | PASS |
| `cargo test -p cli --release --no-default-features --features metal,no-cuda serve::tests -- --nocapture` | PASS, 21 passed |
| `./target/release/arle serve --backend metal --model-path mlx-community/Qwen3.6-35B-A3B-4bit --port 8137 --num-slots 1` with residual swap and quiet paging delta | FAIL-CLOSED before weight load because budget < fixed requirement |
| Same command with `--allow-swap --memory-budget-bytes 1073741824` | FAIL-CLOSED before weight load: budget below fixed requirement |

Follow-up CLI status check:

```text
sysctl vm.swapusage: used = 785.06M
vm_stat sampled free/inactive/speculative pages
```

Qwen3.6 trial now reports the system state directly in the CLI error. Residual
swap no longer blocks it; quiet paging deltas show the Mac is not actively
swapping. The actual blocker is budget:

```text
Metal resource guard rejected startup: system total=48.0GiB available=23.4GiB swap_used=577MiB pageouts_delta=0MiB swapouts_delta=0MiB sample=1000ms; memory budget 17 GiB is below fixed requirement 25 GiB (weights 19 GiB + runtime headroom 6 GiB + static state 61 MiB).
```

Budget-only trial with `--allow-swap --memory-budget-bytes 17179869184` still
fails before weight load:

```text
Metal resource guard rejected startup: system total=48.0GiB available=22.4GiB swap_used=785MiB; memory budget 16 GiB is below fixed requirement 25 GiB (weights 19 GiB + runtime headroom 6 GiB + static state 61 MiB).
```

Qwen3.5-9B now starts under the same residual-swap condition because active
paging is quiet and the fixed requirement fits:

```text
[infer-metal] resource guard: system total=48.0GiB available=20.7GiB swap_used=777MiB pageouts_delta=0MiB swapouts_delta=0MiB sample=1000ms memory_limit=14GiB wired=6GiB cache=1024MiB weights=5GiB runtime_headroom=6GiB static_state=49MiB kv_budget=3GiB kv_capacity_tokens=131072 pages=8192
```

9B short code prompt:

```json
{
  "elapsedMs": 792.803542,
  "usage": { "prompt_tokens": 47, "completion_tokens": 16, "total_tokens": 63 }
}
```

Process RSS was about 5.0 GiB and swap did not increase (`577 MiB` after the
request). The service was stopped immediately after the sanity check.

## Prompt Performance Sanity

Because the host already had active swap (`vm.swapusage used=817 MiB`), the
Qwen3.6 path was intentionally not benchmarked. For a low-risk final path check,
the small Metal model was run inside an explicit 8 GiB process budget:

```bash
./target/release/arle serve --backend metal \
  --model-path mlx-community/Qwen3.5-0.8B-MLX-4bit \
  --port 8138 --num-slots 1 \
  --memory-budget-bytes 8589934592 --allow-swap
```

Resource guard log:

```text
memory_limit=8GiB wired=1GiB cache=1024MiB weights=0GiB runtime_headroom=6GiB static_state=18MiB kv_budget=1GiB kv_capacity_tokens=131072 pages=8192
```

Non-streaming API caveat: `/v1/chat/completions` has no streaming first-token
event, so TTFT below is `max_tokens=1` wall latency. TPOT is estimated from
`max_tokens=64` median wall latency minus token-1 median latency.

| Scenario | Prompt tokens | token-1 wall median | 64-token wall median | Estimated TPOT | Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Agent tool-call prompt | 90 | 11.47 ms | 229.05 ms | 3.45 ms | 289.5 |
| Code patch prompt | 166 | 8.51 ms | 228.19 ms | 3.49 ms | 286.8 |

Service stats after the run:

```json
{
  "requests_completed": 14,
  "prefill_tokens": 352,
  "generated_tokens": 406,
  "prefix_cache": {
    "lookups": 14,
    "hits": 12,
    "hit_rate": 0.8571428571428571,
    "hit_tokens": 1440,
    "hit_pages": 90,
    "published_pages": 15,
    "cached_pages": 15
  },
  "ssd_recall": { "available": false }
}
```

The small-model tool-call output was not quality-clean (0.8B generated malformed
tool JSON), so this is a path/perf sanity check, not an agent-quality claim.
Process RSS was ~649 MiB during the run, and swap did not increase
(`817 MiB -> 809 MiB`).

## Rule

On Apple unified memory, model serving must reserve system headroom and reject
active pageout/swapout pressure before loading weights. Residual `swap_used` is
normal macOS behavior and should be a warning/status field, not a standalone
hard fail. A process memory limit without an available-memory gate is not
enough: if macOS is actively paging, the next allocation can spill to
SSD-backed swap and freeze the foreground system.
