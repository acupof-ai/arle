# Omit unmeasured metrics from /metrics and /v1/stats — 2026-09-13

> Status: Shipped, CPU-only unit gates. No device behavior involved; no pod
> build. Closes findings #1, #2, #5 from the false-zero measurement audit.
>
> Rule in one sentence: a surface publishes a number only when something
> measured it.

## Context

A follow-up audit found three places where an unmeasured quantity was rendered
as a confident number:

1. `spec_accept_rate` was emitted as an integer-percent u64 gauge. Before any
   draft it read `0`, byte-identical to a measured all-rejected rate, and a real
   sub-1% rate (e.g. 1/200) truncated to `0`.
2. The multiproc `/metrics` handler aggregated the empty group set into the
   default all-zero `CounterSnapshot` and rendered it. `kv_free_pages 0` reads
   as a *full* pool. `/v1/stats` fails with a 500 on the identical condition,
   so the two surfaces disagreed.
3. The five copy/wait fields from
   `2026-09-12-omit-impossible-page-tier-copy-metrics` were omitted from
   Prometheus but still serialized as `0` in the `/v1/stats` JSON. (The audit
   also named the observe JSONL surface; that was wrong — `StoredSample` never
   carried the five fields, so the JSON path alone closed #5.)

## What changed

- `spec_accept_rate` is omitted until `spec_drafted_total > 0` (the sibling
  counter is the presence signal), and rendered as an f64 ratio when present.
  The Prometheus emitter became a type-generic macro since one closure cannot
  borrow `out` for both u64 and f64.
- `query_stats_all` returns `Option`, `None` on an empty responding-group set.
  `/metrics` returns the same 500 as `/v1/stats`; the observer keeps the last
  good snapshot instead of caching zeros.
- The five `/v1/stats` fields are `Option<u64>` with `skip_serializing_if`,
  emitted `None` today. Wire DTOs and `infer-core` fields stay u64 — a future
  copying page tier still populates them; only the rendered JSON omits.

Red-first: the #2 gate was run against the pre-fix handler with one configured,
silent worker group; it served the full-pool zero body and failed. After the
fix it asserts the handler errors, for both zero configured groups and one
silent group. #1 asserts the series is *absent* (not present-and-zero) at
drafted==0 and renders `0.5` at 1/200. #3 asserts the JSON keys are absent
while sibling batch counts stay present. 32 infer-server tests pass; clippy
`-D warnings` and fmt clean.

## Rule — the honest measurement set

Every exported metric must distinguish "zero events" from "not measured". The
codebase's convention, which these fixes converge on rather than introduce:

- **Monotonic `_total` counters**: a literal `0` is honest — the event has
  happened zero times since start. Always emit.
- **Ratios / rates**: compute from a total+count pair and return `None`
  (`ratio()` → JSON `null`) or omit the series when the denominator is zero;
  never default a ratio to 0. Render ratios in wide enough a type (f64) that a
  true sub-unit value cannot truncate to 0.
- **Presence booleans / enums**: `available` and `io_mode` say whether a
  subsystem reported at all; clients gate on them before reading the values.
- **Totals for client-side division**: export `*_total` plus a `*_count` and
  let the client divide under its own denominator guard, instead of shipping a
  pre-divided gauge that needs a sentinel for "no samples".
- **Handler-level emptiness**: when no backend/group reported, fail the request
  or hold the last good sample — never aggregate the empty set into a default
  struct and serve it, because a default gauge like `free_pages = 0` asserts an
  extreme and false state (full pool).

## Remaining audit findings

| # | Finding | Status |
|---|---|---|
| 3 | `reasoning_tokens` scan seeds `in_thinking` from the request intent flag, so marker-less output counts the whole completion as reasoning — a false non-zero, worse than a false zero. Details-object presence is a separate bug. | Written up separately; fix gated on a concrete-input demonstration. |
| 4 | `ssd_recall.lookups`/`hits` are never written yet render under `available:true`. | Open product decision: wire or suppress. |
| 6 | Prometheus `kv_system_host_demoted_pages` / `_disk_pages` hard-zero on backends with no page-tier view (HIP, Vulkan, OCR, CUDA placeholder) where CUDA carries live samples. | Open; needs a presence gate. |
| 7 | Prometheus `kv_tier_resident_blocks` structurally pinned 0 on CUDA (capacity 0; CUDA parks whole slots) while slots are actively demoted. | Open; needs a presence gate. |
| 8 | `kv_system_resident_pages` / `_resident_evictable_pages` default 0 in pools that do not override the trait methods (`infer-seam/src/kv_query.rs:18-29`); `VulkanKvPool` allocates pages but leaves the defaults. | Recorded, latent: Vulkan numeric forward fails loud today, no serving deployment. |
| 9 | Observe JSONL `cpu_pct` / `disk_used_pct` fall back to 0.0 on a sampling failure (`observe.rs:26-33`), indistinguishable from idle; no availability flag. | Recorded; never triggers on the Linux hosts (CPUs and `/` always present). |
| 10 | `arle doctor` turns an unparseable nvidia-smi memory row into `Cuda{0.0GB}` instead of `None` (`cli/hardware.rs:170`), so `min_memory_gb` checks pass. | Recorded; CLI status only, not serve HTTP. |
