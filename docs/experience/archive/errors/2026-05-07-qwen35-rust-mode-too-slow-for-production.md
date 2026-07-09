# Qwen3.5 Rust step path is unviable for production benchmarking — 2026-05-07

## Context

M_e.1 §7.5 "Alternative path" proposed switching Qwen3.5 to
`Qwen35StepMode::Rust` so the Rust side has direct K/V access (instead
of the C++ session-owned KV per the §7.4 errata). This would avoid C++
FFI work for P2.1 and downstream paged-KV.

To test this hypothesis, I added a diagnostic env var
`AGENT_INFER_QWEN35_FORCE_RUST` (commit follows) that bypasses CPP
mode even when `weights.cpp_model` is loaded. Then ran the same
bench cells used for the c=4 / c=1 isolation work earlier today.

## Result — Rust mode is structurally too slow for any production
## bench cell

c=4 long (4096-in / 256-out / 30s):
- 0 requests completed within 30s
- benchmarks.csv reports all-zero successful counts
- the bench produced 16384 in / 0 out tokens — request received but
  no completion finished

c=1 short (128-in / 2048-out / 30s):
- Server smoke OK (single chat completion of 5 tokens succeeds)
- Bench scrolling through requests: 6144 in / 14321 total tokens
  observed but 0 successful completions
- A 2048-token decode took longer than 30s — indicates ITL of at
  least ~15ms per token, likely 30+ ms. CPP mode at the same cell
  ran 246 tok/s (4 ms ITL).

Smoke tests succeed at small workloads (5 tokens) so correctness
isn't the issue. **The Rust path is functional but too slow** — at
least 4-10× behind the C++ session step.

## Root cause (assessed, not measured)

The C++ session in `crates/mlx-sys/src/mlx_qwen35_model.cpp` holds:
- A pre-compiled MLX graph (one-step closure)
- KV cache state owned by C++
- Possibly per-step fused operations (RMSNorm + RoPE + matmul)

The Rust path (`qwen35_forward_step` in `metal/qwen35.rs`) builds the
forward graph per call — no compiled session, no fusion across the
24-layer step. Per-step Rust-side overhead (FFI calls into MLX, eager
boundaries, MlxArray construction) compounds with no compilation
benefit.

## Implications for M_e.1

The "switch to Rust mode" alternative path in M_e.1 §7.5 is
**eliminated empirically**. Closing the c=4 ITL gap requires the
production CPP path; therefore P2.1 needs the C++ FFI readback work
(or a deeper C++ pool integration) — no shortcut exists.

## Rule

When evaluating a "switch to slower-but-more-flexible path" trade-off
on Metal/MLX hot paths, **bench it before relying on it**. The Rust
path's general behavior (functional under smoke) does not imply
production viability under c=4 long-prompt load. Smoke success ≠
bench success.

Stack with `feedback_ffi_session_owns_data.md`: the C++ session is
load-bearing for performance, not just for ownership semantics.
Removing it without replacing the compilation/fusion benefits puts
us back at "Rust eager step" speeds, which this exercise empirically
classifies as 4-10× too slow for the 4 ms ITL anchor.

## Reproduce

```bash
AGENT_INFER_QWEN35_FORCE_RUST=1 \
  target/release/metal_serve \
  --model-path models/Qwen3.5-0.8B-MLX-4bit \
  --port 8000

PATH=$HOME/.local/bin:$PATH ./scripts/bench_guidellm.sh metal-c1-rustmode \
  --concurrencies 1 --max-seconds 30 \
  --data 'prompt_tokens=128,output_tokens=2048' \
  --target http://localhost:8000 \
  --model Qwen3.5-0.8B-MLX-4bit \
  --processor models/Qwen3.5-0.8B-MLX-4bit
```

Raw artefacts: `bench-output/2026-05-07-metal-c{1,4}-rustmode/`.

## Cross-references

- M_e.1 plan §7.5 alternative path:
  [`docs/plans/M_e1-metal-paged-kv-hot-path.md`](../../plans/M_e1-metal-paged-kv-hot-path.md)
- Master analysis decomposition:
  [`docs/projects/2026-05-07-metal-optimization-master-analysis.md`](../../projects/2026-05-07-metal-optimization-master-analysis.md)
- CPP-mode anchor numbers (which Rust mode fails to approach):
  [`docs/experience/wins/2026-05-07-bench-guidellm-metal-c1-isolation-decomposition.md`](../wins/2026-05-07-bench-guidellm-metal-c1-isolation-decomposition.md)
