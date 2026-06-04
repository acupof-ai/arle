# Metal eval-count fix KILLED; the real ~−17% lever is cross-step pipelining

## Context

Continuing the rewrite `infer-metal` ~−17% pure-decode regression hunt (after the
publish/drain cadence + wired-limit kills and the turn-wall de-confound). Ranked
next lever: the `step()` "two-phase submit→poll eval round-trip" — `submit_decode`
does `async_eval(logits)` → argmax → `async_eval(sampled)`, then `poll` does
`eval(sampled)`. Hypothesis: collapsing to a single eval recovers the gap.

## Root Cause (of the FALSE hypothesis)

Matched same-binary env-gated A/B (`INFER_METAL_SINGLE_EVAL`), c=1, canonical
Qwen3.6-35B-A3B-4bit + Qwen3.5-0.8B, 4 runs each, runtime probe live:

- Qwen3.6: A(two-phase) 65.1 vs B(single-eval) 65.5 tok/s = **+0.6%**.
- Qwen3.5-0.8B: 233.6 vs 237.5 = **+1.7%**.

Both an order of magnitude below the recovery needed and inside the ~8-15%
run-to-run noise floor (A's 70.5 outlier exceeded all B runs) → **no reproducible
recovery**. Greedy tokens bit-identical A vs B (FNV-1a64 per-turn fingerprints
matched across all 8 runs); TTFT radix reuse held.

The localization was wrong: **HEAD already has only ONE blocking sync per token**
(the `eval` in `poll`). The second `async_eval(sampled)` is a *non-blocking*
host-side kickoff that MLX deduplicates into the same graph as the argmax — so
single-eval saves only the host enqueue call (microseconds), not a device
round-trip. (Also ruled out: the per-token `begin_session`/`end_session` churn is
pure host-side ref-counted `std::vector<array>` moves — no eval/sync.)

## Fix

Reverted the env-gated change; tree clean. The real structural gap vs legacy is
**cross-step pipelining, not eval count**:

- The Engine loop (`infer-core/src/lib.rs:367-400`) is designed to poll plan N then
  build+submit plan N+1 — but the rewrite's `poll` does an unconditional **blocking**
  `eval(sampled)`, so step N's GPU work fully drains before step N+1 is even built.
- Legacy's c=1 hot path (`infer/src/backend/metal/request_state.rs:6100-6149`) keeps
  the GPU one step deep via `pending_sampled` double-buffering: it submits step N+1
  *before* resolving step N's token, overlapping host readback with the next forward.

Recovering this needs `submit` to be truly async and `poll` to return `NotReady`
(or a deferred cross-step handle) instead of blocking — a **seam/Engine-level
change** (the `BackendExecutor` submit/poll contract), larger than a backend-local
flag, and it must preserve the c≥2 `NotReady` path (cannot be a c=1-only hack). This
is the leading candidate for the remaining ~−17%.

## Rule

- An "extra eval / extra round-trip" decode-regression hypothesis must be checked by
  counting the per-token **blocking syncs**, not the `async_eval` kickoffs — MLX
  dedups non-blocking kickoffs into one graph, so kickoff count ≠ device round-trips.
- A decode-throughput gap vs a double-buffering baseline is a **pipelining (GPU-idle)
  gap, not an op-count gap**: measure where the GPU stalls (here: a blocking `poll`
  drains each step) rather than how many ops the step encodes.
