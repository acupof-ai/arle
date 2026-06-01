# Metal Auto Wired Limit Opt-In Memory Fix

## Context

`docs/experience/wins/2026-06-01-bench-arle-vs-mlxlm-ttft-tpot-mem.md`
showed ARLE Metal Qwen3.6 peak RSS at 20.2 GB while `mlx-lm` stayed around
10.6 GB. The first hypothesis was ordinary KV or prefix-cache growth, but the
~9 GB gap was too large for those paths.

The root cause is the ARLE default `auto_wired_limit`: `metal_serve` computed
model weight bytes plus 1 GiB, set MLX `set_wired_limit`, then startup warmup
touched enough Qwen3.6 MoE weights to keep Metal buffers resident. `mlx-lm`
does not auto-enable this residency policy, so process RSS is much lower.

## What Changed

- `metal_serve` no longer auto-computes a wired limit by default.
- Added explicit `--auto-wired-limit` for users who want the previous p99
  pageout-protection behavior.
- In-process Metal engine load no longer auto-pins weights by default.
- `--wired-limit-bytes` remains available for an explicit fixed limit.

## Evidence

Built:

```bash
cargo build --release -p infer --bin metal_serve --no-default-features --features metal
```

Memory A/B used the rebuilt binary, same model, same startup warmup, same
`max-running-requests=1`, same `max-batch-tokens=4096`.

| case | ready RSS | delta vs opt-in |
|---|---:|---:|
| default, no auto wired limit | 5.71 GiB | -69.3% |
| explicit `--auto-wired-limit` | 18.61 GiB | baseline |

The opt-in run logged:

```text
auto wired_limit = 20 GiB (21475946095 bytes; model dir ...)
Metal runtime wired limit set to 21475946095 bytes (previous 0)
```

The default run did not set a wired limit and still completed the same startup
warmup before serving `/v1/models`.

Raw results:
`docs/experience/wins/assets/2026-06-01-metal-auto-wired-opt-in-memory.json`

## Problems

- This is a deliberate latency/memory tradeoff change. The previous default was
  added to reduce cold expert/pageout p99 on Qwen3.6. That behavior is now
  explicit rather than default.
- This entry proves startup RSS reduction, not full TTFT/TPOT parity. Run the
  full ARLE-vs-mlx-lm memory bench again before claiming the final user-facing
  benchmark is closed.

## Learnings

- `set_wired_limit` is not an allocation to free after warmup; it configures
  Metal residency capacity. Once warmup touches model buffers, ARLE can look as
  if it loaded much more of the model than `mlx-lm` because those buffers are
  intentionally kept resident.
- Memory default must match the competing runtime first. Pageout protection is
  useful, but it belongs behind an explicit latency-first flag.
