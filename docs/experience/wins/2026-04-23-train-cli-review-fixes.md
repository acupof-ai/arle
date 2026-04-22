# Train CLI Review Fixes

## Context

- Goal: close the remaining review gaps in the unified `agent-infer train ...`
  / `agent-infer data ...` surface after the first DX pass landed.
- Concrete problems found during review:
  - `agent-infer train test --json` mixed child-command logs with the wrapper's
    final JSON, so stdout was not machine-readable.
  - the wrapper always injected scratch shape flags for `train pretrain`, which
    duplicated the train binary defaults and created a future drift hazard.
  - failed `train test` runs could leave temporary smoke directories behind.

## What Worked

- Reworked `train test` to execute the real top-level `agent-infer` subcommands
  as child processes instead of calling the train binaries in-process. That
  made the smoke path match what users actually run and let the wrapper keep
  stdout clean in `--json` mode.
- Captured child stdout/stderr per smoke step and surfaced them only on
  failure, which keeps the success path concise while preserving debuggability.
- Parsed the `train eval` JSON output and attached it to the final
  `train test --json` report as `eval_summary`.
- Stopped forwarding scratch shape defaults unless the user explicitly selected
  a preset or shape override. The underlying `pretrain` binary is once again
  the single source of truth for its default model shape.
- Added unit coverage for the pretrain-defaults behavior and verified the
  wrapper behavior end-to-end:
  - `cargo test -p cli --release --no-default-features --features no-cuda`
  - `cargo clippy -p cli --release --no-default-features --features no-cuda --no-deps -- -D warnings`
  - `cargo build -p agent-infer --release --no-default-features --features cli,no-cuda`
  - `cargo build -p agent-infer --release --no-default-features --features cli,metal,no-cuda`
  - `target/release/agent-infer train test --backend metal --json`
  - `target/release/agent-infer train pretrain --corpus README.md --tokenizer models/Qwen3-0.6B --dry-run --json`
  - `target/release/agent-infer data convert`
  - `target/release/agent-infer data convert --input /tmp/does-not-exist.jsonl --format chat --output /tmp/out.chat.jsonl`

## Rule

- For script-facing smoke commands, emit exactly one JSON document on stdout in
  `--json` mode. If nested tools are noisy, isolate them behind child-process
  capture instead of letting logs leak into the machine-readable channel.
- When a wrapper exists only to improve DX, avoid re-encoding lower-level
  defaults unless the wrapper is intentionally overriding them. Let the owning
  binary stay the source of truth.
- CLI / operator-DX only; no runtime hot path changed, so no benchmark entry
  was required.
