# Metal Recommended Working Set Guard

## Context

Qwen3.6-35B-A3B-4bit on a 48 GiB Apple Silicon Mac was easy to over-admit by
looking only at total unified memory or a fixed 75% rule. macOS exposes
`MTLDevice.recommendedMaxWorkingSetSize`, which is the GPU working-set size Apple
expects to stay below for normal runtime performance.

## What Worked

ARLE now queries `recommendedMaxWorkingSetSize` through the Metal bridge and
feeds it into the Metal resource guard as an automatic memory-budget cap. The
guard output now reports `gpu_working_set`, and `arle --doctor` reports both the
working set and the smaller current effective backend memory after anti-swap
headroom.

On this M4 Pro 48 GiB host the API returned `37.4 GiB`. With only about
`23.0 GiB` currently free, doctor reports `15.0 GiB` effective backend memory and
does not recommend the 35B model. A dry 35B serve attempt failed before loading
weights:

```text
system total=48.0GiB available=23.9GiB gpu_working_set=37.4GiB swap_used=553MiB
memory budget 15 GiB is below fixed requirement 27 GiB
```

## Verification

| Command | Result |
| --- | --- |
| `cargo test -p infer-metal --release --no-default-features --features metal resource::tests` | PASS, 9 tests |
| `cargo test -p cli --release --no-default-features --features metal,no-cuda` | PASS, 141 tests |
| `cargo test -p cli --release --no-default-features --features cpu,no-cuda hardware::tests::metal_effective_memory_accounts_for_current_available_headroom` | PASS |
| `CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib` | PASS |
| `cargo build --release --no-default-features --features metal,no-cuda,cli -p agent-infer --bin arle` | PASS |

No throughput benchmark was run because this change only affects startup
budgeting, system reporting, and recommendation filtering; it does not enter the
model forward hot path.

## Rule

On Apple Silicon, model recommendation and startup admission should use the
smaller of Apple-reported GPU working set, physical-memory reserve budget, and
current available-memory reserve budget. The fixed 75% heuristic is only a
fallback when the Metal API is unavailable.
