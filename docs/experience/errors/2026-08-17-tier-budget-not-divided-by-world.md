# 2026-08-17 — T1/L2 tier budget not divided by world size at executor construction

## Context

The 2026-08-16 engine architecture audit flagged that the L2 host-DRAM tier
budget was the deployment-total, not the per-rank share, at executor
construction. Each CUDA arm built its `KvTierStore` with
`default_t1_budget_bytes(DEFAULT_DRAM_FRACTION)` — the full 0.5 × MemAvailable —
undivided by the TP world size. The engine builder re-budgeted pre-serve via
`set_kv_tier_budget_bytes(resolve_dram_budget_bytes(config.kv_dram, world))`,
but direct-construction paths (agent-bench) that skip the setter kept the
undivided budget. Under TP>1 each rank then claimed the full deployment budget,
oversubscribing host DRAM by the world factor. Tracked as issue #213.

The symptom hides: it is tier eviction/disk-underflow pressure computed against
a wrong capacity, not an immediate OOM (the store is pageable host memory, not a
reservation).

## Root cause

`default_t1_budget_bytes(DEFAULT_DRAM_FRACTION)` returns the deployment-total
budget. The three construction sites (`executor/qwen.rs:191`,
`executor/qwen35.rs:757`, `executor/dsv4/build.rs:204,221`) called it
undivided. The world-size division lived only in the post-hoc setter
(`loaded.rs:2237`), which direct-construction paths never reach.

## Fix

Added `default_t1_budget_per_rank()` in `executor.rs`: resolves the TP world
size from env (`resolve_tp_config_from_env`, which sees the builder's
`TpEnvGuard` override of `INFER_TP_SIZE` set before construction) and divides
the default budget by it, defaulting to world=1 when unset or unreadable. The
four construction sites now call it. The loaded-path setter is unchanged — it
still re-budgets for non-default `--kv-dram` configs (Off/Bytes/Fraction); for
the default fraction it now sets the same value the constructor already
computed.

Single-rank: `world=1`, so the budget is unchanged. No behavior change off the
TP>1 direct-construction path.

## Gate

Construction-time unit test (`executor::tests::default_t1_budget_per_rank_divides_by_world`):
sets `INFER_TP_SIZE=1` then `=2`, asserts the world=2 budget is exactly half the
world=1 budget. The `executor` module is `#[cfg(feature = "cuda")]`-gated, so
the test compiles under `--features cuda,no-cuda` (Mac typecheck passes) and
runs on the H20 pod — **pending-remote**. The issue's heavier integration gate
(TP=2 direct construction, agent-bench shape, assert eviction pressure against
the correct capacity) is also pod-side.

## Rule

A per-rank resource budget is divided by world at the constructor, not in a
post-hoc setter the caller might skip. The constructor is the one path every
deployment — engine builder and direct construction alike — must traverse.
