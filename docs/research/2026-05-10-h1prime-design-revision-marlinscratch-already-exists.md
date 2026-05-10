---
title: H1' design REVISION — MarlinScratch pattern ALREADY EXISTS in linear.rs; PF8 path is the only missing variant + Task #43 linked to same root cause
date: 2026-05-10
type: research
status: open (CRITICAL — supersedes H1' design 05e2135 §3-§4 design choices)
related_tasks: [#43 (W4A16 stack overflow under sustained load — likely SAME root cause), #44 (PF8 chain), #47 (H1' refactor — design needs revision)]
---

# H1' design REVISION — MarlinScratch pattern ALREADY EXISTS; PF8 path is missing variant; Task #43 may share root cause

> **Purpose**: surface a SOLID gap discovered via Claude CPU-bound parallel
> work this tick (per cron-loop directive "Working + GPU 空 → 读源码找
> anti-pattern"). The audit found the static-scratch pattern this design
> proposes ALREADY EXISTS in `infer/src/ops/linear.rs` for W4 and W4A8
> variants; the PF8 variant is simply the only one missing. This both
> SHRINKS the H1' implementation cost AND provides a concrete root-cause
> link to Task #43 W4A16 stack overflow.

## §1 The finding (raw evidence per SKILL v1.12.0 #31)

`grep -nE "^fn run_marlin|^fn run_w4|^fn run_hybrid" infer/src/ops/linear.rs`:

```
1158:fn run_marlin_w4_gemm(
1167:fn run_marlin_w4_linear(                ← per-call alloc variant
1256:fn run_marlin_w4_linear_with_scratch(   ← static-scratch variant (EXISTS)
1363:fn run_marlin_w4a8_linear(              ← per-call alloc variant
1484:fn run_marlin_w4a8_linear_with_scratch( ← static-scratch variant (EXISTS)
1637:fn run_marlin_w4_fp8_prefill(           ← per-call alloc (PF8.3 KILL victim)
                                              ↑ NO _with_scratch variant exists
```

`grep -nE "alloc_zeros" infer/src/ops/linear.rs` (excerpts):

```
317-323: MarlinScratch struct fields:
   w4_x_fp16: Option<CudaSlice<u16>>,
   w4_y_fp16: Option<CudaSlice<u16>>,
   w4_workspace: Option<CudaSlice<i32>>,
   w4a8_x_int8: Option<CudaSlice<i8>>,
   w4a8_activation_scales: Option<CudaSlice<f32>>,
   w4a8_y_fp16: Option<CudaSlice<u16>>,
   w4a8_reduce: Option<CudaSlice<i32>>,
   w4a8_workspace: Option<CudaSlice<i32>>,

352-414: scratch alloc block (one-time allocation at scratch init):
   .alloc_zeros(max_rows * max_k)        // w4_x_fp16
   .alloc_zeros(max_rows * max_n)        // w4_y_fp16
   .alloc_zeros(w4_workspace_elems)      // w4_workspace
   .alloc_zeros(max_rows * max_k)        // w4a8_x_int8
   .alloc_zeros(max_rows)                // w4a8_activation_scales
   .alloc_zeros(max_rows * max_n)        // w4a8_y_fp16
   .alloc_zeros(MARLIN_MAX_PAR * 64 * max_n)  // w4a8_reduce
   .alloc_zeros(w4a8_workspace_elems)    // w4a8_workspace
```

The `MarlinScratch` struct is the **exact pattern** my H1' design proposed
to invent from scratch as `PF8Scratch`. It's right there in the same
file, ~1300 lines above where `run_marlin_w4_fp8_prefill` lives.

## §2 Implications

### §2.1 H1' implementation is much smaller than 110 LOC

Original `M_pf83_h1prime_static_scratch.md` (commit `05e2135`) proposed:
- New `PF8Scratch` struct (~50 LOC)
- Plumbing through call chain (~30 LOC)
- 3× State impl changes (~30 LOC)
- = ~110 LOC total

**Revised plan**:
- Extend existing `MarlinScratch` struct with 5 new PF8 fields (~10 LOC)
- Extend existing alloc block at 352-414 with 5 new allocations (~25 LOC)
- Add `run_marlin_w4_fp8_prefill_with_scratch` variant (~30 LOC, mirrors
  `run_marlin_w4a8_linear_with_scratch`)
- Update sole caller at `linear.rs:2094` to use `_with_scratch` variant
  with State's existing `MarlinScratch` (~5 LOC)
- = **~70 LOC total**, ~36% smaller than original estimate

The State trait integration (§5 of original plan) is also moot —
`MarlinScratch` already lives in State, just needs new fields.

### §2.2 Task #43 may share the same root cause

Task #43 "Server stack overflow under sustained W4A16 4k-token bench
load" (pending, observed under sustained load, no resolution).

Hypothesis: dispatch may be routing through `run_marlin_w4_linear`
(line 1167, per-call alloc variant) instead of
`run_marlin_w4_linear_with_scratch` (line 1256, static-scratch
variant). If so, sustained W4A16 load would trigger the same cudarc
fragmentation pattern that killed PF8.3.

**Cheap experiment** to test:
- `grep -n "run_marlin_w4_linear\b" infer/src/ops/linear.rs
  infer/src/scheduler/`
- If callers exist that don't use `_with_scratch`, route them through
  the scratch variant
- Run sustained W4A16 4k-token bench again

If this hypothesis holds, **fixing Task #47 H1' AND Task #43 with the
same dispatch routing change** is high-ROI — one PR resolves both
tasks.

### §2.3 SOLID gap in original H1' design (self-audit)

Per SKILL `kernel-optimization` v1.12.0 #29 "default test fixtures may
be known-broken" + #31 "ANY ARLE surface claim needs raw evidence in
same response":

The original H1' design (`05e2135`) was based on reading
`run_marlin_w4_fp8_prefill` in isolation, NOT the broader `linear.rs`
file. If I had grepped `^fn run_marlin` first to enumerate the marlin
variant family, the existing `_with_scratch` pattern would have
surfaced immediately. This is a recurrence of skill anti-pattern #29
applied to architecture decisions, not just test results.

**Skill candidate v1.13.0+ #36** (single evidence point, not
sedimenting yet per skill accumulation policy): "Before designing a
new substrate pattern, grep the file for existing variants of the same
shape. The pattern may already exist; designing from scratch
duplicates effort and risks divergence."

## §3 Action

**This tick** (Claude, CPU-bound parallel work while codex Task #35
benches):
- (this doc) Document the finding + revision direction

**Next codex pickup** (when Task #35 lands and bench v11 license decision
clarifies PF8 fate):
- If PF8 LICENSES: codex picks up REVISED H1' plan (~70 LOC, extend
  MarlinScratch + add `run_marlin_w4_fp8_prefill_with_scratch`)
- Bonus: same PR routes any non-`_with_scratch` W4A16 callers through
  the scratch variant, testing the Task #43 hypothesis

**Task #47 description** should be updated to reference this revision
(reduced LOC estimate from 110 → 70).

**Task #43 description** should be updated with this hypothesis: "may
share root cause with PF8.3 KILL — check if W4A16 dispatch routes
through `run_marlin_w4_linear` (per-call alloc) vs
`run_marlin_w4_linear_with_scratch` (static-scratch)".

## §4 Cross-references

- `M_pf83_h1prime_static_scratch.md` (commit `05e2135`) — ORIGINAL design,
  needs revision per §2.1
- `infer/src/ops/linear.rs:1158-1693` — the marlin variant family
- `infer/src/ops/linear.rs:317-323` — existing `MarlinScratch` fields
- `infer/src/ops/linear.rs:352-414` — existing alloc block
- Task #43 W4A16 stack overflow (pending, hypothesis link in §2.2)
- Task #47 H1' static-scratch refactor (pending, scope reduction in §2.1)
- SKILL v1.12.0 anti-pattern #29 (test fixtures broken) + #31 (raw evidence
  in same response) + candidate #36 (grep for variants before designing
  from scratch)

## §5 Status

**Open + URGENT** — supersedes design choices in `05e2135` §3-§4.
Codex pickup of Task #47 should read THIS doc first, then `05e2135`
for context, then implement the revised plan. Saves ~40 LOC + ensures
consistency with existing scratch pattern.

Surface this finding via PushNotification this tick — it materially
changes the H1' implementation plan AND opens a free Task #43
hypothesis test.
