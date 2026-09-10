# spec draft plan and accepted-token constructor land in infer-plan

Date: 2026-09-10 · Lane: lane/d9 · Step 3a slice 5 (blocks 1+2 → `infer-plan`)

## Context

Slices 3-4 moved the dispatch ladder, prefill geometry, accept scan, and page
math. The two batch functions still open-coded the draft-phase host
arithmetic: `dspark_decode_batch` computed the batched-draft gate (≥2 seeded
rows, all greedy), the slot-sorted row indices, and the anchor/start vectors
inline, then scattered the drafted chains back by hand; `mtp_decode_batch`
had its own copy of the seeded-greedy index filter. Both accept loops also
hand-built the same `SlotToken` shape (no logprob, no finish — spec is
vetoed for logprobs requests) as a 6-field struct literal each.

## What worked

- **`dspark_draft_plan(seeded, rows) -> Option<DsparkDraftPlan>` is the
  batched-draft decision.** The gate (≥2 seeded, all greedy), the slot sort,
  and the anchor/start collection are all host facts about `DecodeRow`s;
  the plan value carries `idx`/`anchors`/`starts` to the device half, which
  now only gathers per-slot state (`pick`/`dfs`) and calls
  `dspark_draft_blocks`. `scatter_into` keeps the idx→row mapping with the
  plan that owns the idx.
- **`greedy_seeded_indices` is mtp's filter.** Same shape (seeded flags +
  rows, slot-sorted), different predicate (per-row greedy, no ≥2 gate) — a
  separate named function, not a parameterized one.
- **`SlotToken::spec_accepted(slot, token, logprob)` names the output
  shape.** Both accept loops mapped tokens to the same six fields with the
  same "spec is vetoed for logprobs" comment; the constructor carries the
  comment once, at the type.

## Rule

The draft phase of a spec batch is host arithmetic over the decode rows:
which rows draft together, what each drafts from, and where the results
land. Return that as a plan value; the caller gathers device state and runs
the draft kernel. A struct literal duplicated across call sites with the
same invariant comment is a constructor waiting to happen — put the
invariant on the type, once.

## Net

No baseline: refactor, no measurement; test counts and diff stat are
mechanical facts.

`git diff --stat` vs the slice-4 commit: +142/−42 = +100 net across 4
files (infer-plan +130 with 2 new tests, qwen35.rs −30 — the draft section
and both SlotToken maps shrink). `cargo test -p infer-plan` 14/14 green;
Mac CUDA clippy gate exit 0; `check_repo_hygiene.py` green; `cargo
fmt --check` clean.
