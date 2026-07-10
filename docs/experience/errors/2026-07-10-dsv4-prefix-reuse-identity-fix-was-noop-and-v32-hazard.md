# DSv4 "identity formula fix" (f7891c3f0) — no-op on Flash, latent break on V32/GLM

## Context

`f7891c3f0` ("use engine page IDs instead of identity formula") claimed to fix
DSv4 prefix reuse by feeding engine host-pool page ids into the FlashMLA
physical band tables (`prepare_kv_batch` identity branch + `mirror_full_band`
`prefix_pages` head). Its pod verification (needles, 4/8/16-concurrent storms,
89.7% hit rate — `wins/2026-07-10-dsv4-prefix-reuse-identity-fix.md`) passed.

## Root Cause

Codex review + code trace showed the fix never executes on the model it was
verified on, and breaks the only model it does execute on:

- **No-op on DSv4-Flash**: `dsv4_flashmla_demand_paged()` is `head_dim != 576`
  (`attention.rs:1765`) — on Flash (512) every FlashMLA layer takes the
  demand-paged branch, which `continue`s before the new mapping code in both
  touched functions. The passing pod numbers actually validate the existing
  L2 copy-restore path (`restore_prefix_state` copies entry content into the
  slot's OWN band; identity mapping is then correct by construction).
- **Latent break on V32/GLM (head_dim=576)**: the identity branch is kept
  precisely because the V32 pack lane has no device-page-table routing —
  `flashmla_pages_byte_range` requires a contiguous band. Mapping foreign
  engine ids there violates contiguity → pack errors on the next forward.
- **Domain confusion (D2 class)**: engine host-pool ids are the LOGICAL
  domain; `mirror_band` tables are the PHYSICAL domain. The Route A deletion
  (#154) made this translation structurally absent; the commit reintroduced
  it by pattern-matching Qwen's unified-namespace semantics onto DSv4.

## Fix

Revert the mapping change: identity branch restored to `slot * lsp + i`,
`mirror_full_band` drops `prefix_pages`. Cross-request reuse stays on the
copy-restore path (which the commit's own pod numbers already licensed).

## Rule

- **A fix must be path-probed on the config it claims to fix** — one
  `log::info!` in the new branch during the verify run would have shown zero
  hits (`feedback_path_probe_before_perf_claim` applies to correctness too).
- **Engine logical page ids never enter physical band tables.** The DSv4
  adapter owns physical addressing; content crosses slots only via the
  prefix-state pool (copy), never via table aliasing.
