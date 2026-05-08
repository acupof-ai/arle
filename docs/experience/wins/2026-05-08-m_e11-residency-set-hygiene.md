# Bench — M_e.11 residency-set hygiene shipped + probe fires — 2026-05-08

## Goal

Ship the omlx-derived periodic `mx::clear_cache` on ARLE's Metal
scheduler hot path so long-generation workloads (>4096 tokens) don't
abort inside `-[IOGPUMetalResidencySet addAllocation:]`. Cover all
three scheduler paths (c=1 step_session, c=1 step_session_paged, c≥2
step_batch_packed) per the codex review on M_e.11 plan.

## Hypothesis

- **Functional**: `M_E11_RESIDENCY_CLEAR_FIRED` path probe fires on
  every workload that crosses the configured threshold.
- **Perf**: at default 1024-token threshold, short workloads
  (eli e2e, max_tokens=64) don't trigger the clear path; ITL p50
  unchanged within ±2%.
- **Stability**: long-generation (max_tokens > 4096) bench survives
  without IOGPU abort. (Stability bench is a future tick — needs
  long-context workload setup.)

## Implementation

`infer/src/backend/metal/ops.rs` — new
`track_generated_token_for_residency_clear(n: u64)` helper. Uses a
single `static AtomicU64 COUNTER`, env-tunable threshold via
`INFER_METAL_RESIDENCY_CLEAR_TOKENS` (default 1024; 0 disables). On
crossing threshold:
1. Reset counter (with `next % threshold` to absorb any overshoot)
2. `std::sync::Once` log probe `M_E11_RESIDENCY_CLEAR_FIRED` on first
   fire
3. `clear_cache()`

`infer/src/backend/metal/request_state.rs` — single hook in
`ResumableRequestState::record_sampled_token` (line 244). Centralizes
across all three scheduler paths because every successful sample
commits via this method:
- c=1 step_session: `decode_step` → `record_sampled_token`
- c=1 step_session_paged: same path
- c≥2 step_batch_packed (decode_qwen35_packed_batch + the oMLX-C v3
  pipelined helper): both call `state.record_sampled_token(token)`
  per row inside the iter loop

Counter is global atomic, summed across the active batch — matches
omlx's `len(responses)`-per-step semantic exactly.

**Note vs omlx**: omlx calls `mx.synchronize(generation_stream)`
BEFORE `mx.clear_cache()`. ARLE doesn't have a `synchronize` FFI
today; instead, the per-step `eval(&[&sampled])` chain already drains
the prev async_eval before record_sampled_token fires. If a future
multi-stream design re-introduces overlap (M_e.5 v2), we'd add the
synchronize FFI then.

## Command

```bash
INFER_METAL_RESIDENCY_CLEAR_TOKENS=64 \
INFER_PHASE_TIMING=1 \
RUST_LOG=info \
./target/release/metal_serve \
  --model-path mlx-community/Qwen3.6-35B-A3B-4bit \
  --port 8765 \
  --max-running-requests 16

/tmp/cN_smoke_q36.sh 4   # 4 concurrent /v1/chat/completions
```

(Threshold lowered to 64 to verify probe fire on a short workload —
`/tmp/cN_smoke_q36.sh 4` with max_tokens=64 produces 4 × 64 = 256
total tokens, comfortably above 64.)

## Environment

- **Backend:** Metal (Apple Silicon)
- **Model:** `mlx-community/Qwen3.6-35B-A3B-4bit` (canonical Metal
  per AGENTS.md)
- **Commit:** following commit `cbd57f24` (M_e.10 probes); this
  commit adds M_e.11.
- **Feature set:** `cargo build --release --no-default-features
  --features metal -p infer --bin metal_serve`
- **Non-default flags / env vars:**
  `INFER_METAL_RESIDENCY_CLEAR_TOKENS=64` (probe-firing threshold
  override; default is 1024 in production), `INFER_PHASE_TIMING=1`.
  Default Metal stack: oMLX-C v3 default-on, auto-wired-limit
  auto-detected, M_e.4 SwiGLU compile-fusion default-on.
- **Workload:** 4 concurrent /v1/chat/completions requests,
  max_tokens=64, temperature=0.0.

## Results

```
$ grep "M_E11_RESIDENCY_CLEAR_FIRED" /tmp/metal_qwen36_residency_smoke.log
2026-05-08T00:28:38.372390+08:00  INFO infer::backend::metal::ops:
  ops.rs:299 metal_path_probe: M_E11_RESIDENCY_CLEAR_FIRED
  (threshold=64 tokens; first fire after 64 accumulated)
```

→ **Probe fires at exactly the configured threshold.** Confirms the
counter increments per record_sampled_token, the threshold check
fires correctly, and clear_cache() is invoked.

The c=4 smoke completed normally (server processed all 4 requests
without error); no functional regression.

## Δ vs baseline

Baseline = pre-M_e.11 commit `cbd57f24` (M_e.10 probes only).

| Aspect | Pre-M_e.11 | Post-M_e.11 (default 1024) |
|---|---|---|
| `M_E11_RESIDENCY_CLEAR_FIRED` log | absent | fires at 1024-token threshold |
| Long-generation IOGPU abort | possible at >4096 tokens | preempted by clear |
| Steady-state c=4 smoke (max_tokens=64, ~256 total tokens) | unchanged | unchanged (256 < 1024 default; clear never fires) |
| Code surface | n/a | +57 LoC ops.rs helper, +5 LoC request_state hook |

For the production default (1024), no clear ever fires on the eli
e2e workload (4 sessions × 2-3 turns × 64 tokens ~= 700 tokens).
The clear cadence kicks in only on long-generation workloads (chat
with max_tokens=2048+ or longctx benches) — exactly where the
IOGPU abort risk lives.

## Problems / observations

1. **No long-generation bench yet.** This commit confirms the
   FUNCTIONAL path (probe fires on threshold cross) and zero-perf
   on short workloads (clear never fires at default). The actual
   stability win (no IOGPU abort) needs a long-context bench
   (`scripts/bench_guidellm.sh longctx-32k-c4 --max-tokens 8192` or
   similar) to reproduce-then-verify. Filed as next-tick work.
2. **omlx's `mx.synchronize` is intentionally omitted** — see §
   Implementation. May need to add later if ARLE adopts dedicated
   encode threads (M_e.5 v2 territory).
3. **Counter is process-global**, summed across active batch. Matches
   omlx exactly. The c≥2 path increments faster (more
   record_sampled_token calls per scheduler tick) — correct, since
   c=8 generates 8× more tokens per tick than c=1 and the
   residency-set fills 8× faster.

## Codex review (this commit)

Pass 1 (this commit, run after the build is green): see
`/tmp/codex_review_m_e11.log`. Result: clean — no actionable
regressions identified.

## What worked

- **Centralized hook in record_sampled_token** beats per-scheduler-
  path counters for code clarity. One line of integration covers
  all three paths.
- **Env-tunable threshold** lets the bench drop it to 64 to verify
  the fire path without waiting hours for 1024 tokens — exactly
  what the canonical `INFER_PHASE_TIMING` and `INFER_M_E10_TRACE`
  knobs already established. Keeps the discipline pattern uniform.
- **Reset via `next % threshold`** absorbs the overshoot from
  multi-row bumps without dropping any tokens — defensive against
  small races.

## Rule

When porting an upstream cadence-based optimization, run a smoke
with the threshold dropped low enough that the probe fires within
seconds. Confirms the wiring works before benching at the default
threshold (which may not fire on short workloads at all).

## Next

- **Long-context stability bench** (`bench_guidellm.sh longctx-32k-c4
  --max-tokens 8192` or similar) — reproduce the >4096-token IOGPU
  abort pre-fix, verify M_e.11 prevents it. Should be
  reproducible-and-verifiable in one bench cycle on the canonical
  Qwen3.6 model.
- **Synchronize FFI** if any future multi-stream work needs an
  explicit drain. Today's single-stream path doesn't need it.
- **dflash-mlx Prometheus `/metrics` port** still on the deck (S
  effort).
- **M_e.8 Tier-2 quality eval** still on the deck for INFER_MOE_TOP_K
  default flip.

## References

- omlx commit `6bda6781` (the source pattern):
  https://github.com/jundot/omlx/commit/6bda6781
- ARLE existing C++ clear (256-token cadence inside
  qwen35_compiled_generate, NOT covering the scheduler step paths):
  [`crates/mlx-sys/src/mlx_qwen35_model.cpp:3399-3406`](../../../crates/mlx-sys/src/mlx_qwen35_model.cpp)
- M_e.11 plan (codex-reviewed, widened):
  [`docs/plans/M_e11-omlx-residency-set-hygiene.md`](../../plans/M_e11-omlx-residency-set-hygiene.md)
- Predecessor (parallel research surveying omlx commits):
  [`2026-05-08-m_e10-try-import-probes-clean.md`](2026-05-08-m_e10-try-import-probes-clean.md)
  §"omlx ALL-MLX-on-one-thread invariant"
