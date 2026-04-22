# nixl-sys macOS tests link stdc++ and fail

## Context

While validating the split `rdma-nixl` / `rdma-nixl-real` feature surface for
`infer/src/kv_tier/transport/nixl.rs`, local `cargo check` passed for both
feature sets on macOS, but `cargo test` for the `rdma-nixl` stub lane failed at
link time.

## Root Cause

The external `nixl-sys` crate injects `-lstdc++` into the macOS test link line.
Apple toolchains ship `libc++`, not `libstdc++`, so the final test binary link
fails before any repo-local code runs.

## Fix

No repo-local logic fix was required for the feature split itself. The current
workaround is to treat `cargo check --features no-cuda,rdma-nixl` and
`cargo check --features no-cuda,rdma-nixl-real` as the local verification bar,
and defer stub-lane test binaries until `nixl-sys` stops hard-linking
`libstdc++` on macOS or the dependency is patched.

## Rule

For `nixl-sys`-gated work on macOS, `cargo check` is currently the local proof.
If a test binary links `-lstdc++`, treat it as an external dependency blocker,
not as a scheduler/kv-tier regression.
