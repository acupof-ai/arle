# arle --doctor prints the resource-guard solve — Metal, 2026-09-09

> Status: Shipped. CLI/diagnostics change, no hot-path code touched.

## Context

The Metal resource guard (`plan_resource_budget`,
`crates/infer-metal/src/resource.rs`) computes a full solve at startup —
weights, runtime headroom, static state, anti-swap reserve, memory limit, KV
budget, planned pages — and `arle serve` prints one condensed line of it.
`arle --doctor` printed only the verdict: which model it resolved and whether
the backend was available. On a machine under memory pressure the doctor
could not answer the question the guard actually asks: does the fixed set
fit, and what is the KV pool worth if it does? The 35B rejection printed
"below fixed requirement" with no itemization anywhere the user could see
before attempting a serve.

## What worked

Surface the existing solve instead of recomputing anything.

- `doctor.rs` calls `plan_resource_budget` with the same `EngineLoadConfig`
  defaults `serve` uses (num slots resolved the same way, same KV dtype
  resolution) and prints every term: weights, runtime headroom, static
  state, anti-swap reserve, system reserve, memory limit, wired limit, cache
  limit, KV budget in tokens and pages, and whether the page count was
  clamped.
- The rejection path prints the itemized fixed requirement (weights +
  headroom + static state) from the planner's error, not a restatement.
- JSON mode gains a `resource` object with the same fields; inspection
  schema version 3 → 4.
- The two new reserve fields on `MetalResourcePlan`
  (`anti_swap_reserve_bytes`, `system_reserve_bytes`) are populated in both
  the full and the weight-only planners, so the printed solve and the
  serve-time solve are the same struct.

## Result

Qwen3.5-0.8B-MLX-4bit, M4 Pro 48 GB, `arle --doctor` (Metal):

```
Resource solve (Metal)
weights 0.6 GiB
runtime headroom 4.0 GiB
static state 4770 MiB
anti-swap reserve 6.0 GiB
system reserve 14.0 GiB
memory limit 13.2 GiB
KV budget 2.7 GiB (131072 tokens, 8192 pages)
```

35B under memory pressure prints the rejection with the itemized fixed set:
weights 19 GiB + runtime headroom 4 GiB + static state 15,720 MiB = 38 GiB
against a 13 GiB budget. The printed fixed-plus-KV sum matches measured
resident bytes after load within the tolerance stated in
[design note 4](../../design/memory-is-the-product.md).

No baseline: this is a CLI/diagnostics change; the GiB/MiB figures are the
printed solve output, not a performance measurement, so there is no baseline
to compare against.

## Rule

A resource guard that only prints its verdict makes the operator re-derive
the solve from the source. The solve already exists to keep the process
below the limit; printing it is a formatting change, and it turns the doctor
from a model resolver into the machine it was meant to inspect.
