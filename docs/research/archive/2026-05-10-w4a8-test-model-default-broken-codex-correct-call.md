---
title: greedy_consistency W4A8 default test model is the known-broken naive checkpoint — codex's INFER_TEST_W4A8_MODEL_PATH override correct
date: 2026-05-10
type: research
status: codex-call-validated
---

# greedy_consistency W4A8 default test model is the known-broken naive checkpoint — codex's INFER_TEST_W4A8_MODEL_PATH override correct

> Codex (Working 10m+) caught a potential false-negative source mid-
> greedy_consistency for Phase 1.1 dequant.h port: the default W4A8
> test model was a known-broken naive/max-scale checkpoint. Codex
> switching to the GPTQ-calibrated variant via `INFER_TEST_W4A8_MODEL_PATH`
> env override. Claude verified the call against test source + model
> directory inventory.

## §0 Direct evidence (raw `grep` + `ls` this tick, NOT memory recall)

### Test source: `infer/tests/greedy_consistency.rs`

```rust
29: const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");
30: const W4A8_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B-W4A8-marlin");
32: fn get_model_path() -> String {
33:     std::env::var("INFER_TEST_MODEL_PATH").unwrap_or_else(|_| MODEL_PATH.to_string())
34: }
36: fn get_w4a8_model_path() -> String {
37:     std::env::var("INFER_TEST_W4A8_MODEL_PATH").unwrap_or_else(|_| W4A8_MODEL_PATH.to_string())
38: }
```

Default W4A8 path: `infer/models/Qwen3-4B-W4A8-marlin` — line 30 const.

### Model directory inventory: `ls -d infer/models/*W4A8*`

```
infer/models/Qwen3-4B-GPTQ-W4A8-marlin    ← codex's choice (calibrated)
infer/models/Qwen3-4B-GPTQ-W4A8-zpfix      ← third variant (fix for prior #25 W4A8 accuracy issue)
infer/models/Qwen3-4B-W4A8-marlin          ← test default = known-broken naive
```

Three W4A8 variants exist. Test source defaults to the naive one.

### Other tests with same pattern

```bash
$ grep -rln "INFER_TEST_W4A8_MODEL_PATH" infer/tests/
infer/tests/greedy_consistency.rs
infer/tests/spec_decode_correctness.rs
infer/tests/spec_decode_radix_pollution.rs
infer/tests/e2e.rs
```

4 tests use this env override pattern. All would default to the
broken naive checkpoint without explicit override.

## §1 Why default points at the broken checkpoint

`Qwen3-4B-W4A8-marlin` was the original W4A8 quant (naive max-scale
per-channel) shipped before the `#25 W4A8 accuracy fix` task added
the GPTQ calibration variant. The default constant in
greedy_consistency.rs was set then and never updated to
`Qwen3-4B-GPTQ-W4A8-marlin` after #25 closed.

Without the override, greedy_consistency runs against the broken
checkpoint and may report failures unrelated to the change under
test (e.g., a Phase 1.1 dequant.h port that's perfectly correct
would still fail greedy on the naive checkpoint).

## §2 Codex's call: correct

Setting `INFER_TEST_W4A8_MODEL_PATH=infer/models/Qwen3-4B-GPTQ-W4A8-marlin`
makes greedy_consistency test against the calibrated checkpoint —
isolates the dequant.h port verification to the actual W4A8 path
ARLE production uses.

This is a **good cooperative judgment** under skill v1.10.0 #28
discipline: codex investigated the test surface BEFORE running, found
the false-negative source, and corrected without needing Claude
intervention.

## §3 Sediment — anti-pattern candidate for skill v1.11.0

### Anti-pattern #29 candidate: "Default test fixtures may be known-broken; verify before relying on test PASS/FAIL"

**Trigger**: when running existing tests as a license/kill gate for
a substrate change, the test's default fixture (model, dataset,
config) may be a known-broken artifact retained for historical
reasons. Test PASS doesn't necessarily mean substrate works; FAIL
doesn't necessarily mean substrate broke.

**Mitigation**: before relying on a test's verdict, grep the test
source for fixture defaults + cross-reference against project
status (e.g., recent errors entries about that fixture). When in
doubt, override via env var to use the production-canonical fixture.

**Example caught this session**: greedy_consistency W4A8 default =
naive checkpoint (broken since #25). Codex caught + override via
`INFER_TEST_W4A8_MODEL_PATH=Qwen3-4B-GPTQ-W4A8-marlin` before relying
on the gate.

**Companion to**: anti-pattern #28 (verify raw output not memory
recall). Both are about VERIFYING the substrate of a claim before
trusting it.

## §4 Recommended one-line PR for greedy_consistency.rs (Claude won't ship this tick)

If a future tick wants to fix the default once-and-for-all:

```rust
// greedy_consistency.rs:30
- const W4A8_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B-W4A8-marlin");
+ const W4A8_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B-GPTQ-W4A8-marlin");
```

Same change in `spec_decode_correctness.rs`, `spec_decode_radix_pollution.rs`, `e2e.rs` if their defaults match.

NOT shipped this tick because:
- Codex's env override works for current Phase 1.1 license
- One-line PR is small but unrelated to active Phase 1 cooperative work
- Better as a separate hygiene PR after Phase 1 lands

## §5 Cross-references

- Phase 1 Substep 1.1 audit (CLEAN): `docs/research/2026-05-10-phase1-substep1.1-codex-impl-audit-clean.md` (70b4d7b)
- Phase 1 wins skeleton: `docs/experience/wins/SKELETON-2026-05-10-path-b-phase1-substep1.1-dequant-port.md` (48c6e49)
- #25 W4A8 accuracy fix (closed): task #25 in TaskList — root cause was the naive checkpoint, fix introduced GPTQ-calibrated variant
- Skill v1.10.0 anti-pattern #28 (verify raw output): `.claude/skills/kernel-optimization/SKILL.md`
- 3 W4A8 model dirs (verified raw `ls`): see §0

## §6 Status

Codex's INFER_TEST_W4A8_MODEL_PATH override correct per direct
evidence verify. greedy_consistency runs in flight (10m+ tick at
capture). Anti-pattern #29 candidate logged for next skill update.
One-line greedy_consistency.rs hygiene PR deferred to post-Phase 1.
