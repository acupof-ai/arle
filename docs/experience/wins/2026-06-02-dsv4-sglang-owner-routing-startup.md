# DSv4 SGLang Owner Routing Startup Gate

## Context

SGLang DeepSeek V4 does not treat the replicated-token debug lane as the
high-performance path. Its controller routes work to an attention-DP owner
slice, then broadcasts only inside the selected ATTN_TP/ATTN_CP slice. ARLE
already had token-owned relay mechanics, but DSv4 startup still selected
`replicated-token` for both in-process and multiproc serving.

## What Worked

- Added `build_attn_owner_groups` from the SGLang request-routing shape. It is
  separate from `build_attn_dp_groups`: owner groups are per-DP compute slices;
  DP groups are cross-DP gather/scatter groups.
- Switched DSv4 serving startup to use token-owned owner groups only when
  `ARLE_DSV4_PERFORMANCE_PROFILE=sglang` is selected. Debug fallback remains
  explicitly `replicated-token`.
- Stopped treating `native-deepep` as universally unavailable. It remains
  forbidden on the debug fallback lane, but can proceed under the SGLang
  best-practice contract.
- Removed stale SGLang-path blockers for request ownership from config/runtime
  validation. The remaining fail-closed blocker is CUDA graph capability, which
  still reports unsupported for DSv4 TP/EP decode.

Verification:

```text
cargo test -p infer --no-default-features --features no-cuda tensor_parallel -- --nocapture
cargo test -p infer --no-default-features --features no-cuda request_handle -- --nocapture
cargo check -p infer --no-default-features --features no-cuda
CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda
cargo fmt --check && git diff --check
```

No performance claim is made in this entry. The graph-safe DSv4 decode path is
still the next SGLang-best-practice blocker.

## Rule

Do not let a SGLang/high-performance profile silently select the replicated
debug lane. Route to the SGLang owner topology, or fail closed with the specific
missing contract.
