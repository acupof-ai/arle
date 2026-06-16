# DSv4 #60 R4 batched DSA gather compile closure

## Context

The working tree had an unfinished DSv4 #60 batched-gather tranche in
`attention.rs` and `dsv4.rs`: the new batched DSA gather contract was threaded
through most of the path, but the tree had previously failed `cuda,no-cuda`
typecheck while the call sites were half-updated.

## What Worked

The diff was coherent DSv4 #60 R4 work, not an alien change:

- `Dsv4DsaBatchedGather` now threads per-row `q_i` / weights staging and
  `key_count` capture into `csa_select`.
- The batched decode path constructs the gather sink per row, skips per-row
  CSA read/select, and runs one `csa_select_official_batched` after the prepare
  loop.
- Single-row and prefill call sites pass `None`, preserving the byte-identical
  path outside the batched lane.
- Batched DSA scratch is charged as a per-slot budget term because it scales
  with `num_slots`.

## Verification

Local host verification:

```bash
CUDARC_CUDA_VERSION=12060 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib
CUDARC_CUDA_VERSION=12060 cargo clippy -p infer-api --release --no-default-features --features cuda,no-cuda --lib -- -D warnings
cargo test -p infer-cuda --release --no-default-features --features no-cuda --lib
```

Results: check passed, clippy passed, and `infer-cuda` no-cuda tests passed
87/87.

## Verdict

This closes the half-state and restores a compiling CUDA/no-cuda tree for the
DSv4 #60 R4 tranche. It is not a DSv4 performance license; the batched DSA lane
still needs its own remote correctness/perf run before any default or throughput
claim.

## Rule

When a batched-lane refactor adds an optional fast-path context, every caller
must be closed in the same tranche: fast path passes the real gather sink,
baseline paths pass `None`, and the commit lands only after the neutral
`cuda,no-cuda` typecheck proves all signatures are synchronized.
