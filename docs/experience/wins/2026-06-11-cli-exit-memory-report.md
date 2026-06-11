# CLI exit memory report for local agent runs

## Context

Local Metal agent runs can load large unified-memory models. The operator needs
a simple answer after exiting: how high did this process's resident memory go,
what is it using now, and how much system memory is still available.

## What Worked

Added a CLI-only `ExitResourceReport` guard for the local agent path
(`arle` and `arle run`). It prints one stderr line at process exit:

```text
[ARLE] exit memory: peak_rss=... current_rss=... system_available=...
```

The report uses `getrusage(RUSAGE_SELF).ru_maxrss` for peak RSS and `sysinfo`
for current RSS plus system available memory. Units are normalized across
macOS (`ru_maxrss` bytes) and Linux (`ru_maxrss` KiB). `serve`, `doctor`,
`list-models`, and train/model management commands do not print the agent-run
exit line.

## Evidence

- `cargo test -p cli --release --no-default-features --features metal,no-cuda exit_report -- --nocapture`
  passed: 5 tests.
- `cargo test -p cli --release --no-default-features --features metal,no-cuda -- --nocapture`
  passed: 140 tests.
- `cargo clippy -p cli --release --no-default-features --features metal,no-cuda -- -D warnings`
  passed.
- Lightweight behavior smoke:
  `cargo run --release --no-default-features --features metal -- --non-interactive --model-path /tmp/arle-missing-model-for-exit-report`
  prints the error first and the final line:
  `[ARLE] exit memory: peak_rss=0.01 GiB current_rss=0.01 GiB system_available=21.39 GiB`.

## Rule

Operator-facing memory diagnostics should live in the CLI layer and print to
stderr so JSON/stdout one-shot outputs stay machine-readable.
