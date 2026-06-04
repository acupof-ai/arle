# Bench: first real e2e generation on the new engine (Metal, Qwen3.5-0.8B)

## Goal

The rewrite's first **real end-to-end** bench: drive a real request through the new
clean stack (`infer-core` device-neutral scheduler → host-only seam →
`infer-metal::MetalExecutor` real MLX forward) on Apple Silicon and measure real
throughput / TTFT. Validates that the new engine actually serves tokens, and that
the `agent-bench` harness drives a real backend (not just the mock).

## Params / Env

- `cargo test --release -p agent-bench --features metal bench_agent_workflow_metal_qwen35_08b -- --ignored --nocapture`
- Model: `mlx-community/Qwen3.5-0.8B-MLX-4bit` (real MLX forward, ported behind the seam).
- Host: Apple Silicon dev Mac, release build, `MetalExecutor` + `MetalKvPool`.
- Shape: single request (R3a/b is single-row; multi-turn needs R3c), 192-token prompt
  (128 system + 64 user, one chunk), greedy, 128 generated tokens.

## Results (raw)

```
[agent-workflow METAL Qwen3.5-0.8B] turns=1 total_gen=128 total_wall=680.846ms tok_per_s=188.0 os_impact=OsImpactReport { samples: 1, peak_rss_bytes: 0 }
  turn 0 prompt_len=192 gen=128 ttft_ticks=2 wall=680.822ms
```

| metric | value |
|---|--:|
| decode throughput (new engine) | **188.0 tok/s** |
| TTFT | **2 scheduler ticks** (prefill chunk + first decode) |
| total wall (192 prompt + 128 gen) | 680.8 ms |

## Learnings

- The new clean engine produces real tokens end-to-end on Metal at **188 tok/s** for
  Qwen3.5-0.8B-4bit — a real serving number on the rewritten architecture, on top of
  the bit-identical correctness already verified (R3a/R3b parity).
- `os_impact.peak_rss_bytes=0` is the `PeakMemProbe` stub; the macOS peak-RSS syscall
  + foreground-responsiveness proxy are the remaining wiring for the full G3
  OS-impact gate.

## Status / next (honest scope)

This is the **single-request** number. The headline **multi-turn agent-workflow**
bench (the AI-PC north-star) is **blocked on R3c**: the harness with growing context
hits `MetalExecutor does not support prefix reuse or chunked prefill yet` — R3c
(MetalPageStore + materialize KV from page ids on prefix attach + multi-chunk
prefill + GDR-state restore) is therefore **critical-path**, not a perf refinement
(task #8). Also pending: a same-shape **legacy-engine baseline** for a Δ% (correctness
vs legacy is already verified; perf Δ is the next quick run), the canonical
**Qwen3.6 MoE** number (R3e, in flight), and the **CUDA** legs on V100/H20. The
consolidated rewrite bench report (goal deliverable, task #7) folds these in.

Real measurement of the new crate's GPU path. The runtime default is unchanged
(new engine runs beside the old), so bench-exempt for the default-flip rule; this is
additive evidence.
