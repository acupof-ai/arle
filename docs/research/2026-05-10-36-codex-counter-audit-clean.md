---
title: #36 codex counter instrumentation audit — CLEAN, ship-ready
date: 2026-05-10
type: research
status: audit-pass-pre-bench-clearance
---

# #36 codex counter instrumentation audit — CLEAN, ship-ready

> Codex picked up the counter sub-task from Claude's brief
> (`/tmp/codex-brief-36-bench.txt` this tick) and produced a 42 LOC
> delta across 3 files. Pre-build audit (cargo check still running on
> codex side) clears the diff for landing + #36 bench A/B run.

## Audit scope

3 files, +42/-1 LOC:

```
infer/src/metrics.rs                          | 27 +++++++++++++++++++++++++++
infer/src/metrics/render.rs                   | 15 ++++++++++++++-
infer/src/scheduler/cuda/runtime/admission.rs |  1 +
```

## Per-file findings

### `infer/src/metrics.rs` — counter substrate

Adds `prefix_aware_admit_deferrals_total: AtomicU64`:

- Doc comment table row added (matches existing counter doc style)
- `MetricsInner` struct field
- Constructor init `AtomicU64::new(0)`
- `pub fn record_prefix_aware_admit_deferral(&self)` setter using
  `fetch_add(1, Relaxed)` — matches every other counter in the file
- `pub fn prefix_aware_admit_deferrals_total(&self) -> u64` getter
- Unit-test invocation added in existing test fn

**PASS** — matches existing counter discipline byte-for-byte.

### `infer/src/metrics/render.rs` — three serialization paths

1. **Prometheus** (`# HELP` + `# TYPE counter` + value line) — slotted
   between existing prefix-cache metrics, format-string labels
   correctly threaded
2. **JSON** (`/v1/stats` agent-cache section) — `"prefix_aware_admit_deferrals":`
   key added in canonical position
3. **Text-summary suffix** — format-string arg alignment matches the
   added `prefix_aware_admit_deferrals={}` placeholder in correct
   position vs. the other agent-cache fields

**PASS** — three serialization surfaces all consistently updated.

### `infer/src/scheduler/cuda/runtime/admission.rs` — call site

```rust
if !self.prefix_aware_admission_allows_plan(&candidate.plan, scan_len) {
+   self.metrics.record_prefix_aware_admit_deferral();
    policy_deferred.push_back(candidate);
    continue;
}
```

Single call site, placed exactly where `policy_deferred.push_back`
fires. **PASS** for placement.

## Semantic note (NOT a blocker)

The counter name says "deferrals" but fires on every gate-trigger,
including candidates that `prefix_aware_fail_open_candidate`
(`admission.rs:437-458`) later promotes back into `candidates`. So:

- True "gate fired" count: this counter (matches what we want for
  bench evidence)
- True "request ultimately rejected" count: would require subtracting
  the fail-open recoveries

For the #36 bench A/B purpose ("did the workload actually exercise
PrefixAware logic?"), the gate-fired semantics is the more useful
single number. If the bench shows zero gate fires, no further
analysis is needed — the workload didn't pressure the gate. If the
bench shows N gate fires, the next-layer question (how many were
recovered?) becomes interesting.

For now the name is OK for purpose. If a follow-up bench needs
rejection-vs-fired distinction, add a paired `prefix_aware_admit_fail_open_recoveries_total`
counter rather than renaming. Renaming risks merge churn for marginal
clarity gain.

## Build status (codex's terminal at audit time)

```
NVCC_CCBIN=/usr/bin/g++-14 CUDA_HOME=/opt/cuda
TORCH_CUDA_ARCH_LIST=8.9 INFER_TILELANG_PYTHON=...
cargo check --release -p infer --features cuda

Working (6m 07s • esc to interrupt) · 1 background terminal running
```

Long incremental CUDA compile in progress, no failure output. Codex
self-reports waiting on same process, not parallelizing.

## Audit verdict

**CLEAN, ship-ready.** When codex's `cargo check` clears, the diff
should:

1. Commit + push (codex owns)
2. Run baseline bench (`36-bench-A-queuebound`) — verify counter
   stays at 0 throughout
3. Run treatment bench (`36-bench-B-prefixaware --max-waiting-requests 4`)
   — verify counter increments (proves gate fires)
4. If counter increments well: license A/B comparison meaningful
5. If counter stays at 0: workload doesn't exercise gate, brief fails
   gate-trigger requirement, propose alternative workload (longer
   prompts, higher concurrency, mixed warm/cold sessions)

Per kernel-optimization skill v1.9.0 anti-pattern #6 ("license on
capture exists not capture reused"), the counter check at step 3
acts as the gate-firing equivalent of capture-reuse evidence.
Without it, a "PrefixAware looks the same as QueueBound" bench
result is unattributable.

## Cross-references

- #36 substrate survey: `docs/research/2026-05-10-36-prefix-aware-admission-substrate-complete-bench-pending.md`
- M_b3 plan (now SUPERSEDED): `docs/plans/M_b3-prefix-aware-admission-step1-directive.md`
- Prefix-aware gate site: `infer/src/scheduler/cuda/runtime/admission.rs:409-458`
- Counter substrate convention: `infer/src/metrics.rs` (every other
  `*_total: AtomicU64` follows the same setter+getter pattern)
- Brief that triggered this work: `/tmp/codex-brief-36-bench.txt`
  (sent via paste-buffer this loop tick)
- Skill v1.9.0 anti-pattern catalog: `.claude/skills/kernel-optimization/SKILL.md`

## 状态

42 LOC counter instrumentation by codex passes pre-build audit.
Awaits codex's `cargo check` to clear, then commit + push + bench.
Audit catches the gate-fired-vs-rejected naming nuance — not a
blocker, but worth noting for bench result interpretation.
