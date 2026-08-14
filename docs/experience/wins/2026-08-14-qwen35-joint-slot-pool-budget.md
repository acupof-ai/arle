# Qwen3.6 joint (slots, pool) budget replaces slots-only clamp

## Context

Issue #182: `kv_budget_num_slots` solved for slot count alone with the fixed
term zero, so on a 32 GB V100 the recurrent reservation ate the headroom —
86 slots x 47 KV tokens each; the engine advertised 86-way concurrency it could
not serve one full request with. DSv4 already solves the pair jointly
(`Dsv4Model::kv_budget_plan`); this was a uniformity gap, not new design.

## What worked

`Qwen35Model::kv_budget_plan` scans slot count down from the old clamp until
`profile_kv_pool_tokens(free - per_slot*n, ...) >= max_seq_len` — first
feasible n wins (feasibility is monotone in decreasing n). The scan starts at
exactly the old answer, so plentiful-VRAM cards take the first probe and are
byte-identical to the previous behavior. Also folded in: the 91-line
`build_full_attn_kv_pool` deleted (pool allocated from the plan), pool pages
NCCL min-reduced under TP, and the OPD `memory_budget_bytes` grant now caps
the pool profile too.

Issue-number walk (V100 32 GB, free 14092 MB, per_slot 146.8 MB):
old = 86 slots / 4096 pool tokens; new = 68 slots / ~36.7K pool tokens —
18 slots shed buy a 9x larger pool, every admitted request can run full length.

## Measured

- H20 (97 GB, plentiful regime): boot capacity line identical before/after —
  `free 64731MB / total 97508MB, recurrent reservation 3127MB (16 slots x
  195MB) -> max_total_tokens 829654 (51853 pages)` on both the pre-change
  binary (run sgserve2, build sgate2) and the fixed one (run sgserve7, build
  sgate7 at 8ad726e1c). Same serve then passed the full sampling gate
  (7 arms, window delta drafted=640 accepted=385, binary sha byte-match).
- V100 32 GB (starved regime): `pending-remote` — re-run the #178 serve on the
  V100 box, expect ~68 slots / ~36K-token pool (issue #182 tracks it).

## Rule

When two capacity quantities trade against the same free-VRAM pool, solve them
jointly; a per-quantity maximizer starves whichever is profiled second. The
in-repo pattern to mirror is `kv_budget_plan`, and the degeneracy proof is
"the scan starts at the old answer".
