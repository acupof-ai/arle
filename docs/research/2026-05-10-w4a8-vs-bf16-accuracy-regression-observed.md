---
title: W4A8-vs-BF16 token diff regressed to 84.4% (>>> 25% test threshold) — surfaced during Task #35 codex work
date: 2026-05-10
type: research
status: open (needs Task #48 audit, NOT blocking Task #35)
related_tasks: [#25 (W4A8 accuracy fix, completed but root cause TBD), #35 (cap=8 prefill warmup, in_progress codex)]
---

# W4A8-vs-BF16 token diff regressed to 84.4% — surfaced during Task #35 codex work

> **Source**: codex tmux 0:0 transcript captured 2026-05-10 ~09:00 during
> Task #35 cap=8 prefill warmup verification. Full greedy_consistency
> binary failed on the existing `test_w4a8_vs_bf16_token_diff` accuracy
> gate. Codex correctly identified this as **unrelated to Pass 3 warmup
> change** and proceeded with targeted `test_greedy_solo_vs_concurrent`
> (PASS in 19.26s). The 84.4% number comes from codex's tmux output;
> I have not independently re-run the test this tick.

## §1 The finding

`test_w4a8_vs_bf16_token_diff` at `infer/tests/greedy_consistency.rs:310`
asserts:

```rust
// line 365 (verified via Read this tick)
assert!(
    diff_pct <= 25.0,
    "W4A8 token diff {:.1}% exceeds 25% threshold — quantization\
     accuracy unacceptable for default-on flip.\n  BF16: {:?}\n  W4A8: {:?}",
    ...
);
```

Test docstring at line 304 cites "skill v1.3.0 rule: W4A8 default-on
flip is gated on token-level diff < 1% vs BF16". Actual assertion is
the lenient 25% gate.

**Codex's report**: 84.4% diff. **>3.4× over the lenient 25% threshold.**

## §2 Why this matters

### Task #25 was closed with "root cause TBD"

Per current task list: Task #25 "W4A8 accuracy fix (codex own, root
cause TBD)" status `completed`. The "root cause TBD" annotation
suggests the original fix was a workaround that didn't address the
underlying accuracy mechanism. The 84.4% regression is consistent with
"workaround has degraded over time" — possibly due to:

- Other W4A8 path changes between #25 close and now (many commits
  modified W4A8 marlin path: #24 graph capture hoist `35fc3cf`, #29
  W3+W4 admission, the recent PF8 chain that may have rebalanced the
  hybrid W4 paths, etc.)
- Test data drift (model checkpoint changed)
- Lenient 25% gate let real accuracy decay through unnoticed

### Task #35 is NOT blocked

Codex's targeted test_greedy_solo_vs_concurrent PASSED, and the W4A8
regression predates the Pass 3 warmup change. Codex's discipline:

1. Spotted the failure
2. Verified it was unrelated to current change
3. Used targeted test as substitute for full binary
4. Documented the known blocker for the wins entry

This is correct per SKILL #34 (matched-control A/B) — codex isn't
falsely attributing the W4A8 regression to Pass 3.

### But it IS a real ARLE accuracy issue

W4A8 path is a production code path. 84.4% token diff vs BF16 means
W4A8 outputs are essentially uncorrelated with BF16 outputs at greedy
decode. If W4A8 is default-on for any model, output quality is
catastrophically degraded vs BF16 baseline.

Per `infer/src/ops/linear.rs:2094` (run_marlin_w4_fp8_prefill caller)
and the broader hybrid W4 dispatch logic, W4A8 path is opt-in via
`INFER_HYBRID_W4A8_PREFILL=1` env var (verified earlier this session).
**So default behavior is unaffected** — the regression only bites if
someone deliberately enables W4A8 prefill, e.g. for benching or because
they expect quant inference quality.

## §3 What to do

**This tick (no action required)**:
- Codex continues Task #35 with targeted test (correct)
- Bench v11 user-blocked stays user-blocked (PF8 license decision)

**Follow-up Task #48 (proposed)**:
- Re-run `test_w4a8_vs_bf16_token_diff` to confirm 84.4% number
  (codex's tmux output is reliable but not yet independently verified
  by Claude this tick)
- If confirmed: bisect when the regression entered. Suspects:
  - `35fc3cf` #24 W4A8 prefill graph capture hoist (significant
    refactor)
  - `c44788f` #40 Path B.2 bucketing fix (touched device metadata
    capture)
  - `09ae5a5` Path B Phase 1 (`marlin_dequant.cuh` 651 LOC, hybrid
    strategy single-file) — most likely candidate, broadest change
- Test threshold gap also worth resolving: docstring says <1% but
  assert is 25%. Either tighten the assert (likely will fail until
  bisect closes), or reconcile the docstring to current production
  expectations.

**Skill candidate (v1.13.0+ #35 candidate)**:
"When closing a Task with `root cause TBD` annotation, mark a follow-up
test or assertion that will catch regressions. Otherwise the workaround
quietly decays and lenient gates pass it through."

This pattern showed up here (Task #25 closed `root cause TBD` →
84.4% regression goes unnoticed for unknown duration → only surfaced
incidentally during Task #35 verification). Recurring risk in any
codebase with workaround-style fixes.

## §4 Cross-references

- `infer/tests/greedy_consistency.rs:310` `test_w4a8_vs_bf16_token_diff`
- `infer/tests/greedy_consistency.rs:365` 25% assert threshold (lenient)
- `infer/tests/greedy_consistency.rs:304` docstring claims <1% rule
  (skill v1.3.0)
- Task #25 W4A8 accuracy fix (completed `root cause TBD`)
- Task #35 cap=8 prefill warmup (in_progress codex, where this surfaced)
- `e61d26e` W4A8 substrate-LAND wins entry §Phase 7 referenced in
  test docstring
- Recent W4A8 path changes (bisect candidates):
  `35fc3cf` `c44788f` `09ae5a5`

## §5 Status

Open. Not blocking Task #35. Noted via PushNotification this tick.
Awaiting user decision on Task #48 priority vs other pending work
(bench v11 license, Task #28 Medusa, Task #47 H1' refactor).
