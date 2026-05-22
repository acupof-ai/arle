# CI keeps failing — three independent root causes

## Context

Last 4+ pushes to `main` show CI failures or cancellations. User asked:
"为什么会一直出问题呢 不能自动化的搞好吗" — why does this keep
breaking, can't we automate this properly?

Investigation found **three independent root causes**, all stacked:

## Root Cause 1: dev-dep feature leakage pulls cudarc in CPU-only build

`infer/Cargo.toml` had:

```toml
[dev-dependencies]
autograd = { path = "../crates/autograd", default-features = false, features = ["cuda", "safetensors"] }
```

The `cuda` feature on autograd activates `cudarc`. `cargo test --lib`
unifies dev-dep features into the dep graph even when `--lib` would not
compile the test binaries. Result: every CPU-only CI run tried to
compile `cudarc v0.19.7` and failed at `nvcc --version` (no CUDA
toolchain on the standard `ubuntu-latest` runner).

Locally `cargo test --no-default-features --features no-cuda --lib`
succeeded because the local `target/` had cudarc already built. CI
starts fresh → fails.

**Single usage** of the dev-dep was `infer/examples/qwen35_train_vs_infer_parity.rs`,
a cuda-only example that wasn't even declared with `required-features =
["cuda"]`.

### Fix

1. Moved `autograd` from `[dev-dependencies]` to `[dependencies]` with
   `optional = true` and only the `safetensors` feature by default.
2. Tied autograd activation to infer's own `cuda` feature:
   `cuda = [..., "dep:autograd", "autograd/cuda", ...]`.
3. Added `[[example]] qwen35_train_vs_infer_parity` with
   `required-features = ["cuda"]` so the example fails cleanly on
   CPU-only builds instead of trying to compile.

Verified locally:
```
cargo test --no-default-features --features no-cuda --lib --no-run
  Finished `test` profile [unoptimized + debuginfo] target(s) in 2.06s
  (no cudarc compile, no warnings)
```

## Root Cause 2: hygiene script + README diverged

`scripts/check_repo_hygiene.py` required a `## 📰 Latest Updates`
section in both READMEs. Commit `5654142 docs(readme): trim to 166
lines, drop verbose Latest Updates tables` intentionally removed the
section, but didn't update the hygiene script. Every commit since has
failed the Repo Hygiene job.

### Fix

Commented out the two `check_latest_updates` invocations in
`main()`. Function preserved for any future opt-in.

## Root Cause 3: docs-only commits triggered the full ~15 min CI for no value

Every wins/errors/research/plan/memory commit triggered the full CI
matrix: macOS Metal check, Linux CPU test, clippy, Cargo Deny, Repo
Hygiene, Python tests. None of this is affected by a docs-only diff.

When the workspace has dirty Rust files from parallel codex work
(which is common per `feedback_git_status_before_commit_in_cooperative.md`),
**fmt check** picks up the un-formatted dirty file and fails for every
push, even docs-only. This made docs-only commits appear to have broken
CI when the actual cause was parallel-track work-in-progress.

### Fix

Added `paths-ignore` to `.github/workflows/ci.yml`:

```yaml
on:
  push:
    paths-ignore:
      - 'docs/**'
      - 'bench-output/**'
      - 'memory/**'
      - '**/*.md'
      - 'CLAUDE.md'
      - 'AGENTS.md'
      - 'ROADMAP.md'
      - 'CHANGELOG.md'
      - 'README.md'
      - 'README.zh-CN.md'
```

Now docs commits skip CI; only commits touching code (Rust / CUDA /
build config / workflows) trigger it.

## Why does this keep happening (the user's real question)

Three structural reasons:

1. **No reproducible CI image locally**. The dev-dep feature unification
   problem only surfaces on a fresh `target/`. A local pre-push smoke
   that mirrors the CI commands (`cargo test --no-default-features
   --features no-cuda --lib` from `infer/` with empty `target/`) would
   have caught Root Cause 1 immediately. Suggested follow-up: add a
   `scripts/ci-smoke.sh` that runs the exact CI commands locally with
   `CARGO_TARGET_DIR=/tmp/arle-ci-smoke`.
2. **Hygiene checks weren't tested in PRs**. The hygiene script was
   updated independently of README changes. Suggested follow-up: meta-test
   that runs `check_repo_hygiene.py` on the canonical fixture before any
   commit modifies the script or the files it checks.
3. **Parallel-track dirty files leak into push CI**. When user/codex
   are working in parallel and one tranche pushes while the other has
   un-formatted dirty files, CI fmt-check trips on the un-formatted
   file. `paths-ignore` for docs is half the answer; the other half is
   commit hygiene — only stage files that are actually in the tranche.
   The existing pre-commit hook covers this for staged files; the
   parallel-track scenario is when codex's dirty files are NOT staged
   but cargo fmt --check still scans them. Suggested follow-up: scope
   the CI fmt step to only changed files via `git diff` so it doesn't
   penalize commits that don't touch the un-formatted files.

## Fix

Three commits land separately:

1. `fix(ci): scope autograd cuda feature to infer's cuda` — Root Cause 1
2. `fix(ci): drop stale README Latest Updates hygiene check` — Root Cause 2
3. `ci: skip docs-only and bench-output commits via paths-ignore` — Root Cause 3

## Rule

When a feature flag chain `A → B → C → cudarc/nvcc` exists, **dev-deps
unify features into the lib build**. A dev-dep with `features = ["cuda"]`
will pull cudarc even for `cargo test --lib --no-default-features
--features no-cuda` on the parent crate. The fix is to move the dep to
the regular `[dependencies]` section with `optional = true`, and only
activate it through the parent's `cuda` feature.

Docs-only / bench-output / memory commits don't affect compile/test
outcomes and must not trigger the full CI matrix. Always add a
`paths-ignore` for these paths in the CI trigger config.
