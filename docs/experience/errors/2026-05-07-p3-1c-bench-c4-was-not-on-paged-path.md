# M_e.1 P3.1c bench c=4 was on the wrong code path — 2026-05-07

## Context

P2.0 → P3.1c.2 wired Qwen3.5 paged-KV through `Qwen35StepDriver` —
P2.0 alloc'd the pool, P2.2 added per-step dual-write after
`run_step`, P3.1c.1 added prefill→pool batch write, P3.1c.2 had Rust
gather K/V from pool and pass to `step_session_paged`. Each bench
showed c=4 4096-in/256-out within ±5% of the no-pool baseline,
which I read as "dual-write fires without regression".

A grep audit during P3.1c.3 prep revealed the c=4 bench did NOT
exercise any of that code:

- `Qwen35StepDriver::run_step` is the **single-request** path (c=1).
- The c=4 path goes through `decode_qwen35_batch`
  (`request_state.rs:2976`), which is dispatched by
  `try_homogeneous_decode_batch` at line 1285 when ≥2 Qwen3.5 states
  are present and ready.
- `decode_qwen35_batch` builds `flat_kv` from each state's
  `Qwen35CppState.kv_flat` after `ensure_cpp_session_drained`, then
  calls `cpp_model.step_batch` directly. It does NOT touch
  `Qwen35StepDriver::run_step` and therefore does NOT touch
  `dual_write_pool_after_step`, `dual_write_pool_after_prefill`, or
  `run_cpp_step_paged`.

Net: every "c=4 ±5% noise" reading from P2.0/P2.2/P3.1b/P3.1c.1/
P3.1c.2 wins entries was structurally noise — the pool path was
**dead code on the c=4 path**.

## Verification — c=1 path actually works

Re-ran the bench at c=1 long-context (4096-in/256-out, 30s) which
DOES go through `Qwen35StepDriver::run_step`:

| Cell | TTFT p50 | ITL p50 | Output tok/s |
|---|---:|---:|---:|
| --kv-pool OFF (c=1 long, 2026-05-07-bench-…-c1-isolation…) | 920 ms | 4.37 ms | 129.5 |
| **--kv-pool ON (this audit)** | **887 ms** | **4.15 ms** | **133.6** |

All within ±5% noise. Single-stream dual-write fires correctly with
zero perceptible overhead (lazy MLX ops + cheap clone semantics).
Logits behavior unchanged (smoke "Say hi" still returns sensible
output).

So the work is **correct and validated for c=1**. It's just not the
c≥2 hot path the c=4 bench was supposed to measure.

## Implications for P3.1c.3

The kernel cutover (the actual unlock for the c=4 batching gap) needs
to land on **`decode_qwen35_batch`** + the C++ `step_batch_packed`
entry point, NOT only on `step_session_paged`. Specifically:

- The Rust side of `decode_qwen35_batch` must, when any state has a
  pool, replace `flat_kv` (built from `cpp.kv_flat`) with
  pool-gathered K/V per state.
- The C++ side `step_batch_packed` (`mlx_qwen35_model.cpp:2489`)
  needs a paged variant that reads SDPA inputs from the gathered
  K/V instead of slice_update'ing a per-state cache.
- The dual-write hook also needs to land on the c≥2 batched decode
  path (after `step_batch` returns, write each state's new K/V row
  to its pool).

The `step_session_paged` work that already landed (P3.1a/b/c.1/c.2)
remains useful — it covers the c=1 path that ARLE genuinely wins on
(per the c-sweep wins entry). It is the **minor** path; closing the
c=4 gap requires `step_batch` paged work too.

## Recommended re-scoping

**Updated M_e.1 commit ladder for the batched path:**

- **P3.1c.3a** — In `decode_qwen35_batch`, when all states have a
  pool, gather K/V per state from pools, build `flat_kv` from those
  gathers (instead of `cpp.kv_flat`). Add per-state dual-write after
  `step_batch` returns. C++ side unchanged — `flat_kv` content
  changes but layout matches what `step_batch` already expects.
  Effort: M.
- **P3.1c.3b** — Add `qwen35_compiled_step_batch_packed_paged` C
  entry point taking external K/V per state per layer. P3.1a-style
  wired-but-ignored at first.
  Effort: M.
- **P3.1c.3c** — C++ `step_batch_packed_paged` flips SDPA read source
  to external K/V. THE batched-path unlock; acceptance c=4 ITL p50
  ≤ 9.3 ms, c=16 ≥ 350 tok/s.
  Effort: L.

**P3.1c.3 (single-stream)** — finish the StepDriver-side cutover that
P3.1a/b/c.1/c.2 prepared, but acknowledge it only helps c=1
benchmarks. Acceptance: c=1 long ITL p50 ≤ 4.20 ms (matches today's
4.15 ms; verify no regression). May become a small win or no-op.

## Lesson

Never bench at concurrency N to validate a code path without first
confirming that path actually fires at concurrency N. The dispatch
logic in
`MetalRequestState::try_homogeneous_decode_batch` at line 1255-1287
splits c=1 (Qwen35StepDriver::run_step) from c≥2 (decode_qwen35_batch)
along a hidden seam. Future plans must cite the dispatch site for
each bench cell they claim to measure. Logged as a feedback memory
update under `feedback_substrate_audit_grep_full_tree.md` extension.

## Cross-references

- Plan: [`docs/plans/M_e1-metal-paged-kv-hot-path.md`](../../plans/M_e1-metal-paged-kv-hot-path.md)
- P3.1c design (now with §"Errata for batched path"):
  [`docs/plans/M_e1-p3-1-kernel-cutover-design.md`](../../plans/M_e1-p3-1-kernel-cutover-design.md)
- Single-stream c=1 anchor (the path that DID get exercised):
  [`docs/experience/wins/2026-05-07-bench-guidellm-metal-c1-isolation-decomposition.md`](../wins/2026-05-07-bench-guidellm-metal-c1-isolation-decomposition.md)
- Master analysis decomposition:
  [`docs/projects/2026-05-07-metal-optimization-master-analysis.md`](../../projects/2026-05-07-metal-optimization-master-analysis.md)
