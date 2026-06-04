# R3a/R3b: new engine real Metal forward — bit-identical parity vs legacy

## Goal

Correctness verification for the `infer/` rewrite (branch `arch/ideal-inference-engine`):
prove the new clean engine (device-neutral `infer-core` → host-only seam →
`infer-metal::MetalExecutor` running the ported MLX forward, **zero legacy `infer`
dependency**) produces output identical to the proven legacy `MetalBackend`. This is
the gate that turns "the architecture should work" into measured fact, and unblocks
the e2e bench.

## What Worked

Ported the existing tested MLX Qwen3.5 stack (config / loader / mlx wrapper / Qwen35
builder / `qwen35_compiled_*` session calls) into `crates/infer-metal` behind the
`BackendExecutor` + `KvPool` seam — no re-derivation of numerics, no dependency on
the legacy `infer` crate (grep-clean).

**Independently re-run locally (not just the agent's report):**

- **R3a** (single greedy first token), `mlx-community/Qwen3.5-0.8B-MLX-4bit`:
  `r3a_single_row_prefill_first_token_matches_legacy_metal_backend` →
  `legacy=11751 new=11751`, 1 passed.
- **R3b** (full greedy sequence through `Engine<MetalExecutor, MetalKvPool>::run_to_idle`):
  `r3b_clean_engine_full_sequence_matches_legacy_metal_backend` →
  `legacy=[11751, 11, 321, 279, 6511, 314, 279, 3516, 4042, 369, 6312, 11, 414, 707, 13, 198]`
  `new  =[11751, 11, 321, 279, 6511, 314, 279, 3516, 4042, 369, 6312, 11, 414, 707, 13, 198]`,
  1 passed.

Both runs: `cargo test -p infer --lib --no-default-features --features metal,no-cuda <test> -- --ignored --nocapture`.

Commits: R3a `7f75b3af`, R3b `8951493c`. Plan: `docs/projects/2026-06-03-r3-metal-port-plan.md`.

## Rule

A real backend wired behind the host-only seam produced **bit-identical** output to
the legacy path with **zero changes to engine-core or the scheduler**. That is the
evidence behind the rewrite thesis: new backends (HIP, ggml/llama.cpp, …), CUDA-graph
(`GraphRunner` seam), and per-executor perf work are all localized plug-ins, not
scheduler rewrites. Correctness for a new executor = a parity test against the legacy
path on its matching hardware (per `2026-06-03-rewrite-verification-targets.md` G2),
run independently — do not trust the agent's report alone.

## Status

Metal G2 parity (single + full sequence) on Qwen3.5-0.8B: **passing, independently
verified**. Remaining for the full goal: R3c prefix-reuse · R3d packed/mixed ·
R3e Qwen3.6 MoE + Metal e2e bench · CUDA legs on V100/H20 · cutover · the e2e bench
report. This is a correctness artifact (parity), not a perf bench — no runtime-path
default change, bench-exempt for the runtime rule.
