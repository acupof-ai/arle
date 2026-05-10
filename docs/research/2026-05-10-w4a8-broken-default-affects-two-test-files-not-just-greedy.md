---
title: W4A8 broken default fixture affects 2 test files (e2e.rs + greedy_consistency.rs) — broader than Task #48 scope
date: 2026-05-10
type: research
status: open (audit finding, codex pickup recommendation)
related_tasks: [#48 (W4A8 regression bisect, in_progress codex)]
---

# W4A8 broken default fixture affects 2 test files

> **Purpose**: while codex is wrapping Task #48 (W4A8-vs-BF16 84.4%
> regression in `greedy_consistency.rs`), Claude audit per directive
> "读源码找 anti-pattern" surfaced that the SAME broken default
> fixture path is hardcoded in 2 test files, not just one. Codex's
> Task #48 fix should address both.

## §1 The audit

`grep -rn "W4A8_MODEL_PATH\|GPTQ-W4A8" infer/tests/` this tick:

```
infer/tests/e2e.rs:21:
  const W4A8_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"),
                                        "/models/Qwen3-4B-W4A8-marlin");
infer/tests/e2e.rs:31:
  std::env::var("INFER_TEST_W4A8_MODEL_PATH")
      .unwrap_or_else(|_| W4A8_MODEL_PATH.to_string())

infer/tests/greedy_consistency.rs:30:
  const W4A8_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"),
                                        "/models/Qwen3-4B-W4A8-marlin");
infer/tests/greedy_consistency.rs:36:
  fn get_w4a8_model_path() -> String {
      std::env::var("INFER_TEST_W4A8_MODEL_PATH")
          .unwrap_or_else(|_| W4A8_MODEL_PATH.to_string())
  }
```

Same default path, same env-var override convention. Per `eb2b4b6`
research entry, the default `Qwen3-4B-W4A8-marlin` is the **naive
checkpoint that produces 100% garbage output**; the recommended
override is `INFER_TEST_W4A8_MODEL_PATH=Qwen3-4B-GPTQ-W4A8-marlin`
(calibrated).

## §2 Implications

### §2.1 e2e.rs may be silently broken too

If `e2e.rs` runs without `INFER_TEST_W4A8_MODEL_PATH` env var set
(default), it loads the naive checkpoint and either:
- (a) Fails with hard assert (caught loudly)
- (b) Passes due to lenient assert (silent regression — same pattern
  as greedy_consistency.rs's 25% lenient gate at line 365)

Need to inspect e2e.rs's W4A8-related test bodies + assert thresholds
to determine which case applies.

### §2.2 Codex Task #48 fix should address BOTH

When codex fixes greedy_consistency.rs default (likely by either:
making `INFER_TEST_W4A8_MODEL_PATH=Qwen3-4B-GPTQ-W4A8-marlin` the
NEW default, OR adding a hard "fixture is naive checkpoint, skip
test or fail loudly" guard), the SAME fix should be applied to
e2e.rs:21+31.

Otherwise: greedy_consistency.rs gets fixed, e2e.rs continues to use
broken default → SKILL #29 pattern repeats one test file later.

### §2.3 SKILL #29 evidence strengthens

This audit adds another data point reinforcing skill #29 ("default
test fixtures may be known-broken"):
- Original n=1 (eb2b4b6 + e3e1ab5): greedy_consistency.rs default is
  naive checkpoint
- This audit: SAME broken default also in e2e.rs (cargo test --test
  e2e + cargo test --test greedy_consistency both affected)

Pattern: shared `concat!(env!("CARGO_MANIFEST_DIR"), "/models/...")`
defaults across multiple tests = single point of fix-or-rot. When
the canonical model checkpoint changes (e.g. naive → calibrated),
the default needs updating in EVERY test, OR the test framework needs
a single source-of-truth model registry.

Skill candidate v1.15.0+ enhancement to #29: "default broken fixtures
may be DUPLICATED across test files via copy-paste constants. When
fixing one, grep for other test files using the same path constant."

## §3 Cross-references

- `eb2b4b6` original research entry documenting naive W4A8 checkpoint
  is known-broken
- `e3e1ab5` Claude's Task #48 surfacing of 84.4% regression via codex
- `81b6481` errors entry "W4A8 substrate produces 100% garbage output"
- Task #48 codex investigation (in_progress)
- `infer/tests/e2e.rs:21+31` identical broken default
- `infer/tests/greedy_consistency.rs:30+36+365` original site of
  broken default + lenient assert
- SKILL `kernel-optimization` v1.11.0+ #29 (default broken fixtures)

## §4 Recommended action for codex Task #48 wrap

When codex finalizes Task #48:
1. Apply same default-fix to BOTH `e2e.rs:21` and
   `greedy_consistency.rs:30` (or extract to shared constant in a
   module both can import)
2. Either: change default to `Qwen3-4B-GPTQ-W4A8-marlin` (calibrated)
   if available locally OR keep current default + add explicit hard
   skip when default fixture detected
3. If using shared constant module: future-proof against this exact
   pattern recurring in new test files

## §5 Status

**Open audit finding**, codex-pickup-ready. Recommend including this
fix in Task #48 wrap (small additional scope, high consistency value).
