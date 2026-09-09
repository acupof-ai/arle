# Prefix reuse-license policy moves to infer-kvspace behind a trait

Date: 2026-09-10 · Lane: lane/d · Step 2 of the architecture refactor

## Context

`reusable_prefix_blocks` / `reusable_prefix_blocks_for_prompt` — the DSv4
prefix reuse-license policy (which leading pages a radix match may attach,
and when a frontier tail must be token-verified) — lived on
`Dsv4CudaExecutor` in `executor/dsv4/prefix_pool.rs`, so the policy's
branches (demoted key, missing meta, boundary gating, tail verify) could
only be exercised through the executor. This is the second half of the
infer-kvspace batch: the policy is host logic over the content index, with
no device contact.

## What worked

- **One trait, two adapters.** `infer_kvspace::PrefixPoolIndex`
  (`page_block_size` / `page_boundary` / `frontier_tail_tokens`) is the
  policy's only view of the pool. The real adapter
  (`PrefixStateIndex` in `executor/dsv4/prefix_pool.rs`) is three one-line
  delegations over `Dsv4PrefixStatePool`. The fake is in-memory.
- **Contract tests on both sides of the adapter.** The fake-side tests in
  infer-kvspace cover every policy branch (fail-closed on demoted keys,
  missing-meta break, non-boundary scan continuation, tail match/mismatch/
  short-prompt). The real-side test in infer-cuda publishes entries into a
  real `Dsv4PrefixStatePool` (host-only — `new(budget, entry_bytes)` needs
  no GPU) and runs the policy through the real adapter, so a delegation
  typo is caught, not just the policy's logic.
- **The real-adapter test is GPU-host-only.** Under `--no-default-features
  --features no-cuda` the executor module is `#[cfg(feature = "cuda")]`, so
  that lane compiles the test out and runs 0 tests; under `cuda,no-cuda` on
  Mac it compiles but fails to link (`_cublas_init`, `_gemm_cuda`,
  `_add_scaled_row_cuda` from the cuda_kernels FFI are not stubbed). The test
  stays in the tree with a GPU-only comment; the fake-side tests carry Mac
  and CI coverage of every policy branch.
- **Mutation negative control.** Flipping the policy's verdict branch and the
  tail-continuation check each turned the suite red; reverts used
  line-addressed sed — a pattern shared by the mutated site and an unrelated
  guard clobbered the guard on the first pass.

## Rule

A policy extracted behind a trait is only as tested as its adapters:
contract tests run against the fake for every branch (cheap, Mac/CI) and
against the real adapter for delegation correctness. Here the real-adapter
test links only on a CUDA host — the executor module is cuda-gated — so the
fake side is what gates Mac and CI. The crate boundary is the pipeline
stage; the DSv4 policy sits in the crate's `dsv4` module alongside the
codec from the first batch.

## Net

`executor/dsv4/prefix_pool.rs` loses the two policy bodies (~60 lines); the
crate gains the trait, the policy functions, and the fake contract tests.
