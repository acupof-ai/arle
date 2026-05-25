# T3 Non-Mainline Prune — `arle train test` Stub

Related: `docs/projects/2026-05-24-opd-mainline-task-backlog.md` T3 and
`docs/projects/2026-05-18-opd-only-pivot.md`.

## Context

- T3 in `docs/projects/2026-05-24-opd-mainline-task-backlog.md` asks for
  non-mainline/dead-code deletion, one cluster per commit, with zero-usage
  evidence.
- The OPD-only pivot says scratch pretrain, SFT, GRPO, and multi-turn RL are
  retired, and `docs/support-matrix.md` already records `arle train test` as
  retired with the legacy `convert->pretrain->sft->eval` fixture.
- The CLI still exposed `arle train test` as a pending OPD smoke stub. That
  kept a retired user-facing command alive after `arle train opd --smoke`
  became the active smoke path.

## What Worked

- Removed the `TrainCommand::Test` variant, `TrainTestArgs`, and
  `run_train_test()` dispatch.
- Removed `train test` from CLI help and `train env` command listing.
- Added a parser negative test so `arle train test` stays retired.

## Zero-Usage Evidence

```bash
rg -n "TrainCommand::Test|TrainTestArgs|run_train_test|accepts_train_test_stub_args|OPD smoke fixture pending|train test" \
  crates/cli crates/train README.md docs/support-matrix.md \
  docs/projects/2026-05-18-opd-only-pivot.md \
  docs/projects/2026-05-24-opd-mainline-task-backlog.md
```

- After deletion, the only hit is `docs/support-matrix.md`, which is the
  retirement notice rather than a code path.

## Verification

```bash
cargo fmt -p cli
cargo check -p cli --no-default-features
cargo test -p cli --no-default-features rejects_retired_train_test_stub
cargo clippy -p cli --no-default-features --no-deps -- -D warnings
```

- All commands exited 0.
- This tranche was CPU-only and did not touch P5 PID 28950, GPU runtime,
  `infer/src/kv_tier/`, or `infer/src/scheduler/`.

## Rule

- Retired CLI stubs are still product surface. If the active replacement
  exists, delete the stub and pin the rejection path in parser tests instead
  of keeping a no-op command around.
