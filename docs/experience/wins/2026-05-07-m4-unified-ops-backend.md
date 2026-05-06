# M4 — Unified Op Trait + Metal `crate::ops::*` Implementor

## Context

M1 unified backend telemetry, M2 wired the KV-tier adapter across CUDA and
Metal, and M3 introduced the shared scheduler decision IR. M4 moved the next
layer up: the hot op families now have a shared `OpsBackend` trait surface so
CUDA and Metal can converge at model-forward call sites without exposing
backend-specific tensor handles across the contract.

This deliberately stayed below attention/recurrent/KV ops. Those paths still
carry backend-specific scheduling and kernel contracts and are left for M5+.

## What Worked

The milestone landed as small, reversible slices:

- `8ce20e0 feat(ops): scaffold unified backend trait`
- `961ac13 refactor(qwen3): route norm ops through backend trait`
- `477b761 refactor(qwen3): route linear ops through backend trait`
- `c70ad34 refactor(qwen3): route elementwise and embedding ops through trait`
- `5f209d2 refactor(qwen3): route sampling ops through backend trait`
- `84caef0 feat(metal): implement unified ops backend`
- `5e5784f fix(cuda): guard batched rmsnorm vector alignment`

Review and follow-up fixes:

- `c40ec21 fix(ops): cover mixed backend tensor accessors`
- `20ee677 test(spec): exercise sparse radix pollution path`
- `7a2baae test(spec): disable vanilla request speculation`

The CUDA implementation is a thin trait wrapper over the existing free
functions, preserving old callsite compatibility and the CUDA graph/raw-pointer
paths. The Metal implementation routes through MLX and keeps the eager boundary
inside the backend: most ops stay lazy, while sampling and host readback force
materialization where the caller needs CPU-visible values.

Feature coverage was expanded beyond the single-backend happy paths:

- `cargo check -p infer --no-default-features --features cuda,no-cuda`
- `cargo check -p infer --no-default-features --features metal,no-cuda`
- `cargo check -p infer --no-default-features --features cuda,metal,no-cuda`

The mixed CUDA+Metal feature check caught the missing `TensorRepr::Metal`
accessor arms and prevented a cfg-combination regression from shipping.

## Bench Status

Canonical GuideLLM remains pending, not because of the op-trait work itself but
because the run exposed a scheduler KV-pressure deadlock:

- Error record:
  [`2026-05-07-m4-guidellm-canonical-stuck.md`](../errors/2026-05-07-m4-guidellm-canonical-stuck.md)
- Follow-up plan:
  [`docs/plans/M4.5-kv-preemption-on-regular-decode-path.md`](../../plans/M4.5-kv-preemption-on-regular-decode-path.md)
- Gating test:
  `d7bb023 test(scheduler): M4.5 P2 — KV pressure drain test (gating)`

The observed canonical stall was `active=12 waiting=560 scheduled=0
decode_rows=0 prefill_rows=0` with `kv_util` peaking at `99.8%`. No valid
`benchmarks.json` was produced, so no perf number is published for M4. The
bench gate transfers to M4.5 regular-decode KV preemption.

## Rule

Unifying an op family should stay one-family-per-commit so each slice is easy
to revert and review.

Trait wrappers must preserve the existing backend ABI first. CUDA wrappers
forward to free functions; Metal owns MLX laziness/eval boundaries; raw graph
and cached-pointer paths can remain on free functions until their ABI is
explicitly migrated.

Alignment assumptions belong in the kernel boundary, not in the trait layer.
The batched RMSNorm misalignment fix is the reminder: wrapping an op is also a
chance to re-run odd-shape tests and guard vectorized loads.

GuideLLM runs must satisfy the documented preconditions before counting as
performance evidence. The `60fd7d3` lesson stands: server `--max-seq-len` must
leave room for the canonical 4096-token prompt after tokenizer expansion, and a
stuck-active trace is an error artifact, not a benchmark result.
