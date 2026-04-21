# Metal Shared Budget Verify Blocked By `kv-native-sys`

## Context

I refactored the Metal scheduler/runtime toward a simpler unified policy:

- `MetalSchedulerConfig` now uses a shared per-tick token budget
  (`max_batch_tokens`) instead of a decode-agnostic prefill chunk cap.
- The scheduler subtracts active decode rows from that budget before emitting
  any prefill work.
- The temporary runtime-only `Qwen3` mixed-prefill cap was removed so mixed
  batching follows the same scheduler budget as the normal path.

The intended verification plan was:

1. `cargo check -p infer --release --no-default-features --features metal,no-cuda`
2. Targeted Metal scheduler/request-state tests
3. `metal_serve` benchmark sweep for `Qwen3-0.6B-4bit`

## Root Cause

Workspace-local verification is currently blocked by an unrelated native build
failure in `crates/kv-native-sys`:

```text
zig/src/kv_native.zig:571:23: error: missing struct field: items
        .free_list = .{},
                     ~^~
zig/src/kv_native.zig:571:23: note: missing struct field: capacity
```

This surfaces during both `cargo check` and `cargo test` for `infer`, because
`infer/Cargo.toml` has an unconditional path dependency on `kv-native-sys`.

The failing Zig source is outside the Metal diff and was already dirty in the
worktree when this refactor started.

## Fix

Use the narrowest verification that does not mutate unrelated user changes:

- Keep the Metal diff scoped to `infer/src/backend/metal/*` plus
  `infer/src/bin/metal_bench.rs`.
- Verify `scheduler.rs` behavior with a standalone `rustc --test` harness that
  stubs `crate::events` and `crate::types`.
- Do not touch `crates/kv-native-sys/*` just to unblock a Metal refactor unless
  the user explicitly expands scope to include that native layer.

For this attempt, the standalone harness passed all 12 scheduler tests after
moving the `chunk_cap == 0` check ahead of request admission.

## Rule

When a runtime refactor is blocked by an unrelated dirty native dependency,
record the blocker explicitly and keep the verification scope narrow rather than
"fixing" someone else's in-progress substrate work implicitly.
