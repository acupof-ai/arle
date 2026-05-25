# Stale Doc And Dead Code Cleanup

Related: `docs/projects/2026-05-24-opd-mainline-task-backlog.md` T19 and
`docs/experience/wins/2026-05-25-stale-doc-quick-cleanup-pass.md`.

## Context

T19 continues the 2026-05-25 cleanup directive: delete stale docs and unused
code when safe; edit stale claims in otherwise useful docs; list ambiguous
items for ckl instead of over-deleting.

This pass deliberately stayed conservative. It shipped only items with clear
grep/compiler evidence and left broad archived-plan cleanup as pending review.

## What Worked

Two cleanup clusters landed:

| Class | Item | Evidence | Action |
| --- | --- | --- | --- |
| SAFE_DELETE | `tests/cli_tiny_fixture_live.rs` | Only tested retired `arle train test`; file was ignored and current refs were limited to `CONTRIBUTING.md` plus historical review text | Deleted the test file |
| EDIT_NOT_DELETE | `CONTRIBUTING.md` test command | Pointed at deleted `cli_tiny_fixture_live` | Replaced with current `cli_smoke` command |
| EDIT_NOT_DELETE | `crates/train/tests/test_metrics.rs` fixture labels | `"train_sft"` was only a metrics-fixture job tag, not behavior | Renamed fixture job string to `opd` |
| EDIT_NOT_DELETE | Retired train-surface docs | Existing banners said retired, but body/final status still said current or ready | Updated stale "current/ready" claims in 7 docs |

Edited docs:

- `docs/plans/2026-05-10-rope-yarn-phase3b-ppl-eval-plan.md`
- `docs/plans/M_medusa-phase1a-dataset-directive.md`
- `docs/projects/agent-rl-self-evolving.md`
- `docs/plans/rust-agent-rl-single-node.md`
- `docs/plans/train-runtime-architecture-v1.md`
- `docs/plans/train-eval-infer-dx-v1.md`
- `docs/plans/train-observability-v1.md`

## Kept

| Item | Verdict | Reason |
| --- | --- | --- |
| `docs/experience/wins/` and `docs/experience/errors/` | KEEP | Historical SOLID record; T19 explicitly forbids deletion |
| `bench-output/2026-05-19+` | KEEP | Outside the requested old-artifact cutoff and actively used by OPD/CUDA evidence |
| `bench-output/2026-04-*` and `bench-output/2026-05-01..2026-05-17` | KEEP / no candidate | No top-level matching dirs or files exist in this checkout |
| Broad `ready-for-codex-pickup` perf plans | NEEDS_CKL_REVIEW | Many are archived strategy notes; date/status alone is not deletion evidence |
| Production TODO comments | KEEP | Remaining TODOs are actionable phase notes, vendor code, or explicit implementation gaps |

## Pending Ckl Review

| Candidate | Why not shipped here | Suggested next question |
| --- | --- | --- |
| CUDA-feature strict clippy debt in DeepSeek V4 scaffold | `cargo clippy -p infer --features cuda --no-default-features -- -D warnings -W dead_code -W unused` reports 57 errors, mostly unused DSv4 MoE scratch/scaffold plus a few mechanical clippy findings. DSv4 is the #1 next-model path, so deleting it under T19 would be unsafe. | Should DSv4 scaffold get a separate cleanup/hardening tranche, or should intentional scaffold be annotated? |
| Markdown broken-link sweep | A quick local-link scan found 61 missing links, many in archived plans/research or placeholder prose. Fixing them safely needs a separate link-repair tranche with archive policy. | Should archived plan/research broken links be fixed in place, annotated as historical, or left unchanged unless active docs point at them? |
| `docs/plans/M_pf83_h1prime_v2_redesign_brief.md`, `M_37-pathB-device-mem-startpos.md`, `2026-05-10-post-37-license-decision-tree.md` | Still say ready for pickup/execution, but they encode older perf-axis decision logic. No current evidence in this pass proves deletion or exact replacement. | Are these still useful as historical perf playbooks, or should they receive retired/superseded banners? |

## Verification

```bash
cargo clippy --workspace --no-deps -- -D warnings -W dead_code -W unused
cargo test -p train --test test_metrics
cargo test -p agent-infer --no-default-features --features no-cuda,cli --test cli_smoke
```

Results:

- Workspace clippy default feature set: PASS.
- `test_metrics`: PASS, 16 tests.
- `cli_smoke`: PASS, 5 tests.
- CUDA-feature strict clippy: FAIL on pre-existing `infer` warning debt, not
  caused by T19 edits. This was recorded as pending instead of broad-deleting
  DSv4 scaffold.

## Line Stats

- `ccc9a86`: 3 files, `+5/-99`.
- `196919d`: 7 files, `+51/-53`.

Total T19 shipped delta before this entry: 10 files, `+56/-152`.

## Rule

For cleanup, "old" is not enough. Delete only when current refs are gone and
the artifact no longer preserves useful rationale. Otherwise edit the stale
claim or list the item for ckl review.

## Follow-up Pass

User licensed pending items (2) and (3) from the table above, while keeping
item (1) DSv4 scaffold strict-clippy debt out of scope as architectural.

| Item | Commit | Verdict | Action |
| --- | --- | --- | --- |
| `docs/plans/M_pf83_h1prime_v2_redesign_brief.md` | `5329498` | BLOCKED, not ready for pickup | Status banner now says PF8.3 remains opt-in/default-off and runtime-KILLed; fresh license is required before any PF8 scratch/kernel work. |
| `docs/plans/M_37-pathB-device-mem-startpos.md` | `5329498` | SUPERSEDED | Path B v1 preserved as historical context; banner points to Tier 4 KILL and Path B.2 Tier 1 outcome. |
| `docs/plans/2026-05-10-post-37-license-decision-tree.md` | `5329498` | EXECUTED | Decision tree preserved as record; banner says do not run it as current instructions. |
| Editable local Markdown links | `7066bb9` + post-rebase follow-up | CLEAN | 61 initial editable broken local links reduced to 0; rebase on remote prune exposed 2 more pre-May links, also annotated before push. |
| `docs/experience/wins/` and `docs/experience/errors/` broken links | not touched | KEEP | 39 immutable historical broken links remain by policy; wins/errors entries were not edited. |

Broken-link classification:

| Class | Count | Action |
| --- | ---: | --- |
| Existing-but-moved targets | 14 | Repointed to current files, mostly consolidated Metal strategy and renamed W4/W3 evidence entries. |
| Truly missing historical targets | 47 | Converted to non-link text with `(historical reference, file removed)` annotation. |
| Malformed placeholder/prose links | 2 | Normalized to plain code/prose (`<date>-bench...`, `signed nibble`). |

Verification for the follow-up:

```bash
git diff --check
python - <<'PY'
# inline local-link audit snippet, not committed
PY
```

Results:

- Editable docs (`docs/plans`, `docs/projects`, `docs/research`, `README.md`,
  `CONTRIBUTING.md`): missing local links `61 + 2 post-rebase -> 0`.
- Immutable docs (`docs/experience/wins`, `docs/experience/errors`): missing
  local links still `39`, intentionally untouched.
- No Rust/code files changed; cargo/bench exempt.

Follow-up line stats:

- `5329498`: 3 files, `+29/-3`.
- `7066bb9`: 36 files, `+78/-64`.
