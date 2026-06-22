# Metal resource guard: right-size runtime_headroom (false-rejected loads that fit)

## Context

The Metal resource guard (`infer-metal/src/resource.rs`) repeatedly rejected
loading `Qwen3.6-27B-OptiQ-4bit` on a 48 GiB Mac at ~29 GiB available, with
"memory budget 23 GiB is below fixed requirement 25 GiB (weights 18 + runtime
headroom 6 + static 0.587)". But the model's measured peak was ~14.6 GiB RSS and a
prior successful load (at 31.8 GiB available) used only ~3–5 GiB of runtime+KV above
the 18 GiB weights — the guard over-reserved ~2× and blocked a load that fits.

## Root cause

The guard stacks **two independent ~6 GiB buffers** above the weights:
- `DEFAULT_RUNTIME_HEADROOM_BYTES = 6 GiB`, folded into `fixed` (weights + runtime +
  static), the reject threshold `memory_limit > fixed`.
- `DEFAULT_AVAILABLE_RESERVE_BYTES = 6 GiB`, subtracted from available to form
  `memory_limit = available − reserve` (anti-swap, kept free for the OS).

Net requirement: `available ≥ weights + runtime_headroom(6) + available_reserve(6) +
static ≈ weights + 12`. At available 29.4: `memory_limit = 23.4`, `fixed = 24.6` →
rejected by a **0.6 GiB margin**. Crucially **KV is budgeted separately**
(`kv_budget = memory_limit − fixed`, clamped to fit), so `runtime_headroom` only
needs to cover **non-KV** transients (activations, MLX command buffers) — measured
< 3 GiB for a c=1 27B. 6 GiB on top of the 6 GiB anti-swap reserve was the over-count.
`weight_bytes` (18 GiB, summed weight files) is correct — weights are wired/resident.

## Fix

`DEFAULT_RUNTIME_HEADROOM_BYTES 6→4 GiB`, `LOW_IMPACT 8→6 GiB`
(`resource.rs:19-20`). The anti-swap `AVAILABLE_RESERVE` (6/8 GiB) is untouched and
remains the swap backstop. Now `fixed = 18 + 4 + 0.587 ≈ 22.6 < memory_limit 23.4` →
loads at available ≥ ~29 GiB, leaving ~7 GiB free (> the 6 GiB anti-swap reserve).

## Validation

- 10/10 `resource::tests` pass (incl. `default_reserve_leaves_headroom`,
  `recommended_working_set_caps_auto_budget`, `explicit_memory_budget_wins`).
- At 29.76 GiB available the 27B **loaded** (was rejected) and an ARLE agent task
  (count .rs files via the shell tool) ran clean, exit 0, no swap explosion —
  total 314 correct.

## Rule

A memory guard is a hypothesis about footprint — verify it against measured RSS
before trusting it. When KV (or any sub-budget) is reserved separately, a flat
"runtime headroom" must NOT also size for it, and must not stack redundantly with an
anti-swap reserve. Follow-up: scale runtime_headroom by num_slots/context instead of
a flat constant.
