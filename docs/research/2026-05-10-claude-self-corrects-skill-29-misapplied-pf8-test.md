---
title: Claude self-corrects — applied SKILL #29 anti-pattern by substituting model into test designed for different fixture class
date: 2026-05-10
type: research
status: closed (Claude self-correction sediment, also reveals codex 8d1caad TIGHTENED gate to 1%)
related_tasks: [#48 (codex 8d1caad fix actually tightened gate to 1%, not just changed default)]
---

# Claude self-corrects — applied SKILL #29 anti-pattern by misapplying test fixture

> **Purpose**: Claude attempted to validate `57c37b5` H8 DISPROVEN
> finding (PF8 kernel works at conc=1) by running greedy_consistency
> against W4-hybrid-zpfix checkpoint with PF8 env vars. Result was
> 100% diff — but the diff is a TEST/FIXTURE MISMATCH, NOT a real
> PF8 correctness bug. Claude committed the same SKILL #29 anti-
> pattern (broken fixture defaults) by substituting a model into a
> test designed for a different fixture class.

## §1 The bench attempt

```bash
INFER_TEST_W4A8_MODEL_PATH=/home/ckl/projects/arle/infer/models/Qwen3-4B-W4-hybrid-zpfix \
INFER_HYBRID_W4A8_PREFILL=1 \
INFER_MARLIN_W4_FP8_PREFILL=1 \
cargo test --release -p infer --features cuda --test greedy_consistency \
    test_w4a8_vs_bf16_token_diff -- --test-threads=1 --nocapture
```

## §2 The result

```text
W4A8 (32 toks): " is器 is a type of instrument, but it's not clear
                 what it refers to. Maybe it's a typo or a specific
                 term in a certain context."
W4A8 vs BF16: matched first 0/32 tokens, diff 100.0%
First W4A8/BF16 divergence: idx=0 bf16=Some(12095) w4a8=Some(374)

thread panicked: W4A8 token diff 100.0% exceeds 1% threshold
test result: FAILED in 72.33s
```

## §3 Two findings

### §3.1 Codex 8d1caad TIGHTENED the gate from 25% to 1%

I had previously thought codex's fix only changed the default fixture
path. But the panic message reveals the assert threshold is now **1%**
(was 25% per pre-fix greedy_consistency.rs:365). Codex's fix is
stronger than I credited:
- Old: lenient 25% gate (allowed 84.4% to slip past unnoticed before
  test broke)
- New: tight 1% gate (would catch any real regression immediately)

This is a substantive part of the Task #48 fix beyond the fixture
update. **Updates Claude's understanding of codex 8d1caad scope.**

### §3.2 Claude committed SKILL #29 anti-pattern

The 100% diff is NOT a real PF8 correctness bug. It's the SAME pattern
SKILL #29 codifies (default broken fixtures): I substituted a model
(W4-hybrid-zpfix, the PF8 hybrid) into a test (test_w4a8_vs_bf16) that
was DESIGNED for a different fixture class (Qwen3-4B-W4A8-marlin
variants). The test's BF16 reference output is for the W4A8-marlin
test path, NOT for the W4-hybrid-zpfix model.

When I substituted the model:
- The test still loads BF16 from `models/Qwen3-4B` (the bf16 path,
  unchanged)
- The "W4A8" comparison loaded W4-hybrid-zpfix instead of expected
  W4A8-marlin variant
- 100% diff is expected when comparing 2 different model
  CHECKPOINTS (not just quantizations)

**Same anti-pattern Claude has been documenting in others' work** —
broken fixture defaults applied to my own action.

## §4 Honest assessment of PF8 correctness

This bench did NOT validate PF8 path correctness. Per `57c37b5` H8
DISPROVEN: codex's verify-script approach (server start + curl
single request) IS the right method for PF8 verification. There's
no automated test that:
- Loads W4-hybrid-zpfix
- Routes through PF8 path (`INFER_MARLIN_W4_FP8_PREFILL=1`)
- Compares against BF16 baseline

The 57c37b5 verification was MANUAL via curl. To get an automated
test, the test framework would need:
- W4-hybrid-zpfix-specific BF16 reference output (or generate
  on-the-fly)
- PF8 path-specific test setup (env vars + model + scheduler config)

This is a gap in test coverage for PF8 substrate that Task #47 H1'
refactor could address (when implemented).

## §5 SKILL implications

### §5.1 #29 (default broken fixtures) — n=4 evidence (was n=3 from Task #48)

n=4 evidence point added: Claude itself committed the anti-pattern by
substituting a model into a test not designed for it. **#29 is
universal — applies to test users (Claude this case), not just test
fixtures.**

Strengthened wording proposal for #29 v1.x.0:
> "Default test fixtures may be known-broken AND test-fixture
> compatibility check is the responsibility of whoever invokes the
> test. Substituting a non-default fixture into a test designed for
> a different fixture class is the SAME pattern as using a broken
> default — both produce false-failure or false-pass signals."

### §5.2 #34 + #34b reinforced

- #34 (greedy single-request not sufficient): the 100% diff WAS a
  PASS in greedy_consistency sense (it ran, produced output) but the
  diff metric is wrong because it's comparing apples to oranges. Need
  to verify FIXTURE COMPATIBILITY before trusting diff metrics.
- #34b (server log first): the test output had no kernel failure
  logs — confirming this is fixture mismatch, not substrate KILL.
  Following #34b would have caught the framing error earlier.

## §6 Action

**This bench attempt is documented but does NOT add to the
"3 Claude-run benches PASS" tally.** Counting honestly:
- 3 successful Claude-run benches (Task #48 verification + true A/B)
- 1 misapplied Claude-run bench (this — 100% diff but test/fixture
  mismatch, not real signal)

§4 gap (no automated PF8 correctness test) is documented for
future codex Task #47 H1' refactor scope consideration.

## §7 Cross-references

- `8d1caad` codex Task #48 fix — TIGHTENED gate from 25% to 1%
  (richer fix than Claude's earlier framing)
- `57c37b5` H8 DISPROVEN via manual curl test (correct method for
  PF8 verification)
- `infer/tests/greedy_consistency.rs:365` (now 1% threshold per
  codex 8d1caad)
- `infer/models/Qwen3-4B-W4-hybrid-zpfix` (PF8 hybrid checkpoint)
- SKILL `kernel-optimization` v1.11.0+ #29 (default broken fixtures)
  — strengthened to n=4 with this evidence point
- SKILL `kernel-optimization` v1.12.0 #34 + #34b
- Previous Claude-run bench docs:
  `2026-05-10-claude-independent-verify-task48-fix-0pct-diff.md`
  (3 successful benches)

## §8 Status

**Closed — Claude self-correction sediment.** No PF8 correctness
claim made or claimed-disproven. Bench attempt revealed:
1. Codex 8d1caad fix scope is wider than Claude knew (1% gate)
2. Claude itself committed SKILL #29 anti-pattern (test/fixture
   mismatch)
3. Gap in PF8 test coverage documented for future H1' refactor
   consideration

Per §0 SOLID rule 1 ("推断 ≠ SOLID"): the 100% diff metric is
NOT evidence of PF8 correctness — it's evidence of fixture
mismatch. Distinguish raw measurement from valid measurement.
