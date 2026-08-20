# Wired limit clamp: Qwen3.8-27B-2bit loads without MLX scheduler deadlock — Metal, 2026-08-20

> Status: Shipped

## Context

`set_wired_limit(weight_bytes + 1 GiB)` could exceed available GPU memory.
When the wired limit exceeds what the Metal residency set can pin, the MLX
scheduler deadlocks on first `eval` — the `Load::eval_cpu` file-read enqueues
a wait on a residency commit that can never complete. Qwen3.8-27B-MLX-2bit
(11 GB weights, wired=12 GB) hung at `qwen35_norm_needs_offset_correction`
when available memory was below 12 GB.

## What Worked

`resolve_wired_limit_bytes()` clamps the wired limit to
`available_memory_bytes` (free + inactive + speculative pages from `vm_stat`).
Partial pinning (wired < weights) is safe — MLX leaves the remainder
unpinned. Also removed the redundant `memory_limit > wired_limit` guard:
the fixed-bytes guard (`memory_limit > weight_bytes + runtime_headroom +
static_state`) is always stricter and already covers the rejection case.

## Result

M4 Pro 48GB, Qwen3.8-27B-MLX-2bit, `--max-running-requests 1`:

| Metric | Value |
|---|---:|
| Weight bytes per decode step | 10.09 GB |
| Theoretical limit (273 GB/s ÷ 10.09 GB) | 27.1 tok/s |
| ARLE steady-state | 19.2 tok/s (71% of theoretical) |
| mlx-lm steady-state | 21.0 tok/s (78% of theoretical) |
| ARLE/mlx-lm ratio | 91% |
| TTFT | 0.74s |

Weight breakdown: 6.72 GB U32 packed + 3.36 GB BF16 scales/biases + 0.01 GB
norms. Vision weights (0.92 GB) not read during text-only decode.

Cannot reach 32 tok/s — the theoretical ceiling is 27.1 tok/s at 273 GB/s.

## Environment

- Host: M4 Pro 48GB, macOS
- Model: majentik/Qwen3.8-27B-MLX-2bit (64 layers, 48 GDR + 16 full attn)
- Flags: `--backend metal --max-running-requests 1`
- Commit: `455ad36c0`
