# CPU backend now preserves completion-token budget accounting

## Context

The CPU backend is a dev-only smoke path, but it still needs to respect the
same `max_new_tokens` contract as the serving backends. Its synthetic text path
could already clip generated token ids to the requested budget, yet
`GenerateResult.completion_tokens` was recomputed from the decoded text.

That became inaccurate when decoding introduced extra `[UNK]`-style tokens and
the decoded string re-encoded to a longer token sequence than the original
clipped ids.

## What Worked

- `infer/src/backend/cpu.rs` now threads the actual generated-token count
  through `generate_text(...)` instead of recomputing it from decoded text.
- The tokenizer-backed clip path returns the clipped-id length directly.
- The fallback word-based clip path also returns the exact completion-token
  count it enforced.
- Added a regression test that uses a tokenizer where decoded `[UNK]` text
  re-encodes longer than the clipped generation budget, and verified that the
  backend still reports the original budgeted completion length.

## Verification

- `cargo test -p infer --release --no-default-features --features cpu,no-cuda backend::cpu -- --nocapture`
- `cargo check -p infer --release --no-default-features --features cpu,no-cuda`
- `cargo clippy -p infer --release --no-default-features --features cpu,no-cuda -- -D warnings`

## Rule

Backend-facing `completion_tokens` accounting must follow the generated-token
budget that the runtime actually enforced, not a second pass over decoded text.
