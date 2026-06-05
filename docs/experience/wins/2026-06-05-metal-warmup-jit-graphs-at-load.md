# Metal: warmup() JITs prefill+decode graphs at load (the turn-0 cold-prefill residual) + TtftObserver transparency fix

**Date:** 2026-06-05. **Backend:** Metal (MLX), Apple Silicon. **Status:** landed,
default-on (opt-out `INFER_METAL_WARMUP=0`), **correctness-verified**; turn-0 perf
magnitude on canonical Qwen3.6 is a pending A/B (see Honest read). Task #19.

## Context

The steady-decode −17% regression was already recovered by the cross-step
pipeline ([[2026-06-04-metal-rewrite-decode-pipeline-recovery]]). The verification
report's documented **residual** was the turn-0 turn-wall gap: "graph build +
first MoE encode" landing lazily on the **first real request** because
`MetalExecutor` never overrode `BackendExecutor::warmup()` — it used the seam's
no-op default, so the MLX graph JIT happened on turn-0 instead of at load.

## What Worked

- **`MetalExecutor::warmup()`** (`crates/infer-metal/src/executor.rs`): runs a tiny
  throwaway forward at load — a 2-token prefill + 1 decode step on a reserved
  warmup slot (`usize::MAX`, 8-token cache, never inserted into `self.slots` or
  published to the kv pool) — to **JIT-compile the prefill + decode-step MLX
  graphs before the first request**. The standard vLLM/SGLang warm-at-load pattern.
- **`TtftObserver` transparency fix** (`crates/agent-bench/src/lib.rs`): the bench
  wrapper forwarded `submit`/`poll` but **not** `warmup`/`model_stop_token_ids`, so
  the seam default (no-op) shadowed the inner executor — warmup never ran under any
  wrapped engine. Now forwards all four seam methods. (Production uses the *bare*
  `MetalExecutor` — `ServeInferenceEngine<MetalExecutor, …>`, loaded.rs:149 — so
  production warmup was unaffected; the gap was wrapper-only, but a wrapper must be
  transparent.)

## Verified (0.8B, serial — one Metal load)

`metal_qwen35_greedy_parity` (warmup default-on, now reaching the executor via the
fixed wrapper): `[infer-metal] warmup = true` fires, and the greedy continuation is
**bit-exact gold** (`0xf005cfaa7dc1793e`, 32/32) — the warmup forward JITs the
graph **without corrupting** the real request. No crash, no drift.

## Honest read

- **Correctness + wiring verified; the turn-0 perf magnitude is NOT yet measured.**
  The benefit is mechanically sound for the **decode-step graph** (every step is
  the same 1-token shape, warmed once). For **prefill** the win is partial: MLX may
  recompile per distinct shape, so a 2-token warmup primes the prefill *kernels*
  but a 4096-token real prefill is a different shape. The precise turn-0 TTFT delta
  belongs on the **canonical Qwen3.6-35B-A3B MoE** (where "first MoE encode over 3
  turns" is the documented cold-start) via a same-load `INFER_METAL_WARMUP=1` vs `0`
  TTFT A/B — pending (a 2× 19 GB serial load). Default-on because warm-at-load is
  the industry norm + verified safe; flip off if that A/B shows net-negative.

## Rule

A backend executor must override `warmup()` to pre-JIT its graphs at load —
otherwise the first request pays the compile (the documented turn-0 residual).
And an executor **wrapper must be transparent**: forward *every* seam method
(`warmup`, `model_stop_token_ids`, …), or the seam's no-op defaults silently
shadow the real backend — a whole-method behavior drop that a `submit`/`poll`-only
forward hides.
