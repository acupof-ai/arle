# M_xgrammar Phase 1 — FFI Scaffold

## Context

- Task #26 starts the structured-output axis from
  `docs/plans/M_xgrammar-ffi-scaffold.md`.
- This tranche intentionally stops at the FFI substrate: no HTTP,
  scheduler, sampler, or GPU hot-path integration yet. That keeps #37
  throughput bench isolated while giving the next tranche a typed Rust
  wrapper over upstream `mlc-ai/xgrammar`.

## What Worked

- Added `crates/xgrammar-sys` as a workspace crate.
- Default build is a stub that compiles without native sources or network.
- `--features real` builds a C++ shim against a pinned upstream checkout
  (`mlc-ai/xgrammar` `v0.1.34`) via `cc`.
- Rust wrapper surface:
  - `GrammarCompiler`
  - `CompiledGrammar`
  - `GrammarMatcher`
  - `bitmask_size`
  - JSON schema / EBNF compile entry points
  - per-step bitmask fill and token accept APIs
- The C ABI keeps xgrammar classes opaque and converts C++ exceptions into
  Rust errors.
- `codex review --uncommitted` caught two scaffold issues before commit:
  per-call bitmask shape storage must not be static, and C++ stdlib linkage
  must remain target-neutral. Both were fixed before landing.

## Verification

```bash
cargo test -p xgrammar-sys

XGRAMMAR_SOURCE_DIR=/tmp/xgrammar-v0.1.34 \
  cargo test -p xgrammar-sys --features real

cargo clippy -p xgrammar-sys --all-targets -- -D warnings

XGRAMMAR_SOURCE_DIR=/tmp/xgrammar-v0.1.34 \
  cargo clippy -p xgrammar-sys --features real --all-targets -- -D warnings
```

Results:

| Gate | Result |
|---|---|
| default stub crate tests | pass, 2/2 |
| real xgrammar C++ FFI tests | pass, 2/2 |
| real EBNF compile + matcher bitmask smoke | pass |
| default + real clippy | pass |

## Bench Status

No runtime hot path is wired in this tranche, so decode-overhead and JSON
validity license remain pending. The license gate stays:

| Gate | Target |
|---|---:|
| JSON validity | 100% |
| decode overhead | <= 10% |

## Rule

Keep xgrammar as a thin FFI substrate. Do not rewrite the FSM in Rust. The
next tranche should attach the wrapper to `response_format=json_schema`
request metadata and apply the mask in sampling, then run the license bench.
