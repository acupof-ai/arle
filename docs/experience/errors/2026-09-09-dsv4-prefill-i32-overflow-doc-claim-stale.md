# DSv4 prefill i32-overflow doc claim stale — not reproduced, fixed since 2026-06-05

**Date:** 2026-09-09. **Backend:** CUDA, DSv4-Flash-0731 4-bit TP=2, 2×H20.

## Context

`docs/architecture.md` stated "Prefill at production shapes is in repair (a MoE
padded-layout i32 work-size overflow at >~1560 tokens)", and the 2026-08-24
roadmap carried it as Goal 1 item 3. The task was verify-or-delete: reproduce
the crash on H20 with 4K/16K-token prompts, or delete the claim.

## Root Cause

The bug was real, and it was fixed three months ago. `be2bc5960` (2026-06-05,
"fix(cuda): DSv4 prefill i32-overflow crash — contiguous MoE prefill layout +
i64 guards", with its own errors entry
`errors/2026-06-05-dsv4-prefill-moe-i32-overflow-crash.md`) established: the
padded masked layout materialized `num_groups × max_m` rows and overflowed the
unpad kernel's i32 work size at ~1,560 prompt tokens; the fix switched DSv4
prefill to the contiguous active-row layout unconditionally and added i64
guards. The docs were written before the fix and never updated.

The masked layout survives only on the Qwen path, gated by
`use_masked = total_routes <= 128` (`crates/infer-cuda/src/moe/qwen.rs:819`,
introduced in `3c0a6eb67`), where the unpad work `32*T*topk*H` cannot reach
the i32 range. DSv4 prefill takes the contiguous path
(`crates/infer-cuda/src/moe/dsv4.rs:1803` comment records the history).

## Fix

Reproduction attempt on today's tree (DSv4-Flash-0731 4-bit, TP=2 on H20 cards
5+7, `--spec-type none`, release binary):

- 4,108-token prompt: prefill completed in 6.9 s, coherent output.
- 16,355-token prompt: prefill completed in 11.7 s, coherent output — 10× the
  claimed threshold.
- Serve log grep for `panic|overflow|i32|error|fatal|abort`: zero hits; server
  alive throughout.

Not reproduced → deleted the in-repair paragraph from `architecture.md` and
Goal 1 item 3 from the roadmap (items renumbered).

Same PR also closed roadmap Goal 2 items 1–2: item 1 (the cold page-read
measurement) shipped in
`wins/2026-09-09-kv-tier-cold-page-read-microbench.md`; item 2 (per-page paging
of active sequences) was gated on that measurement, and the gate answer is
negative for L3 (cold page fault p50 20.8 ms vs the 15.25 ms decode step
budget), so the item is deleted rather than left open.

## Rule

A doc claim describing a crash as "in repair" must name a bug ID or a fix
commit; when the fix lands, the doc updates in the same change. Here the fix
entry existed in `errors/` since June — the architecture doc and roadmap were
left describing a tree that no longer existed. Verify-or-delete means reproduce
first: the 2026-06-05 entry already recorded the fix, and the 2026-09-09 probe
extended the verified envelope to 16K tokens.
