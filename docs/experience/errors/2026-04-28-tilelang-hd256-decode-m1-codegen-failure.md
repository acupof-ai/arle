# TileLang HD256 decode AOT codegen fails with M=1

## Context

`cargo build --release -p infer --features cuda,tilelang-attn --bin infer`
on the L4 box (TileLang 0.1.9, CUDA 13.0, sm_89) crashes during the
build script's TileLang AOT generation step with:

```
TileLang AOT generator failed for tilelang_batch_decode_paged_hd256_q8_kv2_run.
[04:45:32] : Fatal: InternalError:
  Check failed: (M % kMPerWarp == 0) is false: M must be divisible by 16, but got 1
  in tvm::tl::GemmWarpPolicyNode::computeWarpPartition(...)
```

Affects all three head configs in `TILELANG_DECODE_HD256_HEAD_CONFIGS`
(`(8,2)`, `(16,2)`, `(16,4)` — the Qwen3.5 full-attn layers). The
HD128 path is unaffected; only HD256 decode trips this.

## Root Cause

`tools/tilelang/batch_decode_paged_hd256.py` declares `BLOCK_M = 1`
(one Q row per tile, the natural choice for pure-decode batches where
each request has a single Q token). The two `T.gemm` calls inside the
inner loop (Q*K and S*V at lines 186 and 229) are emitted by TileLang's
codegen using `GemmWarpPolicy.FullRow`. Under TileLang 0.1.9, every
GEMM warp policy enforces an invariant that the M dimension must be
divisible by `kMPerWarp = 16` so the warp partition is well-formed.
M=1 violates this. There is no `policy=GemmWarpPolicy.NoPartition` (or
equivalent GEMV-style policy) in TileLang 0.1.9 to fall back to; the
available policies are `FullRow`, `FullCol`, `Square`, all of which
require M ≥ 16.

This means the entire HD256 decode tranche has been broken at HEAD —
the wins entry
[`2026-04-27-bench-guidellm-cuda-tilelang-decode-hd256-pending-remote.md`](../wins/2026-04-27-bench-guidellm-cuda-tilelang-decode-hd256-pending-remote.md)
was authored against a build that never actually completed on a real
L4. The Qwen3-4B HD128 path (`tilelang-attn` Tranche 4) was not
exercised on hardware until 2026-04-28 because the build was failing
upstream at the HD256 codegen step.

## Fix

Two-part fix in commit `<this commit>`:

1. **Gate HD256 decode behind a new opt-in feature
   `tilelang-decode-hd256`** (default-off) so the canonical
   `--features cuda,tilelang-attn` build for Qwen3-4B / HD128 succeeds
   without involving the broken kernel. Touched files:
   - `crates/cuda-kernels/Cargo.toml` — new feature
   - `crates/cuda-kernels/build.rs` — HD256 decode codegen gated by
     `CARGO_FEATURE_TILELANG_DECODE_HD256`
   - `crates/cuda-kernels/src/ffi/attention.rs` — extern decls + macro
     gated on the new feature
   - `infer/Cargo.toml` — re-export of the feature
   - `infer/src/ops/attention.rs` — dispatch arm in
     `flashinfer_run_layer_hd256` switched from `tilelang-attn` to
     `tilelang-decode-hd256`. When `tilelang-attn` is on but the
     decode-hd256 feature is off, HD256 decode falls through to
     FlashInfer (the existing default path).

2. **Architectural fix is deferred to a separate ticket** — rewrite
   `batch_decode_paged_hd256.py` to avoid M=1 GEMMs. Two viable
   approaches:
   - Pad Q to 16 rows with masked-out lanes; do a regular M=16 GEMM
     and ignore lanes 1..15. Wastes shared memory but keeps the
     existing kernel structure.
   - Implement the dot-product as a hand-rolled
     `T.alloc_fragment` + reduction (i.e., emit GEMV explicitly
     rather than calling `T.gemm`). Avoids the warp-partition policy
     entirely.

   This work is on the operator-optimization roadmap and is not on the
   critical path for Qwen3-4B serving at c=16 (which uses HD128 only).

## Rule

**A `pending-remote` wins entry is not a green build.** Wins entries
authored without an actual matched-bench run on the target hardware
should be flagged in the title. When `cargo build --features X` has
NEVER successfully completed on the target, the feature is not
"shipped, not yet validated" — it is "shipped broken." Distinguish
the two states by:

- titling the wins entry `*-pending-remote` only when the diff is
  expected to compile (or has compiled in CI) but hasn't been
  performance-measured;
- opening an `errors/` entry like this one whenever a build failure
  is discovered after the fact, so the false-positive wins doc
  doesn't keep mis-leading future readers.

Also: **A new TileLang kernel needs at least one local AOT-codegen
smoke run before a wins entry is filed.** The error above is
deterministic; one `python -m gen_tilelang_aot --kernel
batch_decode_paged_hd256` would have surfaced it before any commit.
