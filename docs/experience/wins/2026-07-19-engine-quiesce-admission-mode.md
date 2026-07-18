# Engine admission mode fix — OPD multi-round writeback KV-pool race

> Status: pending-remote — GPU gate: f6 multi-round run survives round ≥2
> (no `qwen35.rs:1366 full_attn_kv` panic) + F.6 numbers unchanged vs f6d
> single-round. Fixes [#51]; unblocks
> [2026-07-16-agent-rl-unified-infra](../../plans/2026-07-16-agent-rl-unified-infra.md) F.4/staleness.

## Context

f6d (`--rounds 2`) completed round 1 (F.6 logged) then crashed in round 2:
`thread 'infer-engine' panicked at qwen35.rs:1366 full_attn_kv present` — the
paged KV pool was `None` when a prefill ran. Wrapped downstream as
"rollout-logprob student hidden ... cuda synchronize failed" (the training thread
syncing a CUDA context the engine-thread panic had poisoned — not the recompute's
own forward, which is a fresh autograd `forward_hidden_states` that never touches
the engine KV).

## Root Cause

`quiesce_serve` cancelled all requests then polled `active_requests==0` before
`release_kv_pool` nulled `full_attn_kv`. But `Engine::cancel_all_requests` only
snapshots waiting+active at call time — a submit-channel-backlog request (cc
900s-timeout orphan) admitted a step LATER got prefilled after the pool was
released → `full_attn_kv=None` → panic. Only bit round-2 long-context: cc
timeouts leave orphaned in-flight requests, and 20K+ prompts chunk prefill across
many steps, widening the release↔late-admit window. The engine had no explicit
"not serving" state — quiesce was faked via cancel+poll, which cannot fence a
late admission.

## Fix

Give the engine an explicit `EngineMode { Serving, Quiesced }` (infer-core
`Engine`), gating the single admission entry `admit_waiting` (first statement:
`Quiesced → return Ok(())`). `Engine::quiesce()` sets `Quiesced` BEFORE cancelling
— atomically on the engine thread, so no step admits between mode-set and cancel.
`resume_serving()` re-arms after the KV pool is re-acquired. Plumbed as
`ServeHandle::quiesce_admissions`/`resume_admissions` (renamed from
`cancel_all_requests`) through `LoadedInferenceEngine` / `ServeInferenceEngine` /
`InferStudent`; `train_cli` quiesces in `quiesce_serve` and resumes right after
each round's `ensure_kv_pool`. Default serving path byte-identical (`mode` is only
ever `Quiesced` under the OPD writeback bracket). `qwen35.rs:1366 .expect()` kept
as the now-unreachable invariant assertion — not softened (softening would hide
the invariant break).

## Rule

A quiesce that only cancels + polls in-flight count cannot fence a LATE arrival;
give the reactive component an explicit paused state that gates its single
admission entry, and set the state BEFORE the cancel so the two are atomic. The
`.expect()` that fired is the invariant assertion — fix the scheduler so it's
unreachable, never soften it to hide the break.
