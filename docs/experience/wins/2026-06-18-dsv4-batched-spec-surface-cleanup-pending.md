# DSv4 batched spec surface cleanup - pending remote bench

## Context

DSv4 batched decode and batched MTP are now the B>1 default paths. Keeping
`--dsv4-batched-decode`, `INFER_DSV4_BATCHED_DECODE`, `ARLE_DSV4_BATCHED_MTP`,
`ARLE_DSV4_BATCHED_MTP_DRAFT`, and `ARLE_DSV4_BATCHED_MTP_COMMIT` implied old
per-row or re-forward fallback lanes that the current code should not expose.

## What Worked

- Removed the `--dsv4-batched-decode` CLI/env bridge.
- Made DSv4 decode dispatch simple: B=1 uses single-row decode; B>1 batches.
- Made B>1 greedy MTP always use `spec_step_batched`.
- Removed the unused MTP batch/draft/commit env gates and the dead batched commit
  helper.
- Narrowed `spec_step_batched` and `mtp_forward_level_batched` so spec code no
  longer accepts an arbitrary `positions` vector. MTP draft positions are derived
  from `start_positions[s] + draft_level`; `positions` remains only on normal
  batched decode sampling.
- Removed the batched verify `fold` parameter; it now always persists
  `spec_normed` for commit fold.

## Verification

Local no-CUDA gates:

```text
rustfmt --edition 2024 --check crates/infer-cuda/src/dsv4.rs crates/infer-cuda/src/executor/spec_decode.rs crates/infer-cuda/src/executor.rs crates/cli/src/args.rs crates/cli/src/serve.rs
PASS

git diff --check -- crates/infer-cuda/src/dsv4.rs crates/infer-cuda/src/executor/spec_decode.rs crates/infer-cuda/src/executor.rs crates/cli/src/args.rs crates/cli/src/serve.rs
PASS

CUDARC_CUDA_VERSION=12090 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib
PASS

CUDARC_CUDA_VERSION=12090 cargo test -p infer-cuda --release --no-default-features --features cuda,no-cuda spec_decode --lib
PASS: 6 passed

CUDARC_CUDA_VERSION=12090 cargo check -p cli --release --no-default-features --features cuda,no-cuda --lib
PASS

CUDARC_CUDA_VERSION=12090 cargo clippy -p infer-api --release --no-default-features --features cuda,no-cuda --lib -- -D warnings
PASS

CUDARC_CUDA_VERSION=12090 cargo clippy -p cli --release --no-default-features --features cuda,no-cuda --lib -- -D warnings
PASS
```

## Bench Status

Pending remote CUDA bench. This turn did not touch H20/pod; local host only ran
no-CUDA compile/test/clippy gates.

## Rule

Once a fallback is deleted, delete the public knob and function parameter that
suggests it still exists.
