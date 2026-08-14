# Seam redefined as a cost contract: 49 trait methods to 15

## Context

Issue #204: `BackendExecutor` had 49 methods, 46 defaulted, many to `{}` — a
new feature added a method, the other backend compiled and silently lacked the
behavior (the sampling-penalty drop was this failure mode one layer up). The
"device-neutral abstraction" framing had also stopped being true: MLA, hybrid
linear attention, and MTP each punched a model-specific hole through it.

## What worked

The seam is now the engine's cost contract: submit/poll + `step_limits()`
(6 cost/limit methods collapsed into one struct, including `spec_row_tokens`)
+ `stats()` (4 telemetry methods) + 6 capability accessors defaulting to
`None`. Capability traits (`PrefixReuse` 9, `KvPageTier` 11, `KvSlotTier` 3,
`DeviceKvFit` 1, `WeightResidency` 5, `MultimodalGenerate` 2) carry zero
default bodies — a backend that claims a capability is compiler-forced to
implement all of it; one that lacks it returns `None`, and a runtime flag
requesting an absent capability fails at load with the flag named
(`--kv-oversubscription`, `--dsv4-decode-reuse`). `kv_slot_tier_enabled` and
`kv_device_gate_active` deleted — accessor presence replaces both booleans.

Fork criterion recorded in the seam module doc: costs are parameterized,
capabilities are explicit; a family whose decode stops being submit/poll
shaped forks the loop (precedent: `diffusion_executor.rs`) instead of bending
the trait.

## Measured

Zero-behavior-change refactor by construction: every engine call site's
`None` arm reproduces the deleted default verbatim (audited per site). All
lanes green (CUDA host, Metal, core/seam/server/metal tests, clippy -D,
workspace check). Pod smoke on the refactored binary (build fix205b,
381b681c…, HEAD incl. the #205 fix): ENGINE_READY, sampling gate PASS on all
7 arms, spec window delta drafted=640 accepted=387, binary sha byte-match.

Follow-up /simplify tranche (net −37 lines): one `backend_stats()`
destructure per tick, unreachable `None` arms bound at their gates,
weight-residency dedup via `run_on_engine`, `kv_page_tier_view(&self)`
reverting the `&mut` ripple on stats paths, `step_limits` hoisted out of
per-row loops. One intended behavior change: `--kv-oversubscription` on a
backend without a slot tier now fails at engine construction instead of
silently no-oping (the check moved from the CUDA-only path to `Engine`).

## Rule

A trait method with a silent `{}` default is an unverifiable promise. Costs
go in a struct the scheduler reads; behaviors go in capability traits with no
defaults; absence is a written `None` the load path can check against the
flags that need it.
