# DSv4 content-keyed prefix-state pool (Phase 2a reland) — pending-remote

> Status: **pending-remote** — code series landed locally (`7298f3338`,
> `f6d32f795`, `5dc5ef79b`, `754648421`, `14b264ea7`); the evidence gate
> needs the 8×H20 pod. No perf or correctness claim until it runs.

## Context

Reland of DSv4 cross-request prefix reuse after the #154 Route A deletion
(`bbaaea93b`): a host-resident `Dsv4PrefixStatePool`, content-keyed by host
page id, written once per completed page from the executor choke point,
restored on radix prefix match, spilled to L3 mmap under the `--kv-dram`
share. Plan: `docs/plans/2026-07-09-dsv4-kv-reuse-seam-refactor.md` Phase 2.

## Evidence gate (pod, mandatory before any claim)

- Correctness: pt=462 needle solo ×15 byte-identical to the `943bacda`
  envelope, cache-on AND `ARLE_DISABLE_PREFIX_CACHE=1`; concurrent n≥2
  unique-content sweep clean of the `738.`/digit-substitution signatures.
- Reuse fires: prefix-cache hit on an identical resend restores at a
  128-aligned boundary (serve log `prefill_start_pos > 0`, no restore error
  fallback), and re-prefill drops accordingly (TTFT Δ% vs baseline).
- Tiering fires: deliberately small `--kv-dram` drives
  `kv_tier_demoted_pages`/disk counters > 0; L3 read-back restore correct.
- Perf: `scripts/bench_guidellm.sh` c-sweep vs the latest DSv4 baseline,
  Δ% table (TTFT / ITL / output tok/s), pool on vs `--kv-dram 0`.

KILL rule: any gate failure stops the series — full re-prefill is always
the correct fallback (#152 posture).
