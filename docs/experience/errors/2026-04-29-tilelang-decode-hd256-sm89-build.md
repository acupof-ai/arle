# TileLang decode HD256 sm89 AOT build failure

## Context

`tilelang-decode-hd256` was rechecked after the fp8 KV and TileLang ON/OFF
bench tranche.

Command:

```bash
CARGO_TARGET_DIR=/tmp/arle-target-hd256 \
CARGO_HOME=/tmp/arle-cargo-home \
CUDA_HOME=/usr/local/cuda \
ZIG=/tmp/zig-0.15.2/zig-x86_64-linux-0.15.2/zig \
cargo build -p infer --features cuda,tilelang-decode-hd256 --release
```

## Root Cause

The build still fails during TileLang AOT for the HD256 Qwen3.5 full-attention
decode kernel on L4 / sm89:

```text
TileLang AOT failed to compile tilelang_batch_decode_paged_hd256_q8_kv2_run for sm_89
Fatal: InternalError: Check failed: (M % kMPerWarp == 0) is false:
M must be divisible by 16, but got 1
```

This is a compile-time TileLang layout inference failure for the decode shape,
not a Rust typecheck failure and not a runtime scheduler issue.

## Fix

Not fixed in this tranche. Keep `tilelang-decode-hd256` opt-in and treat the
Qwen3.5 HD256 decode path as FlashInfer-backed until the TileLang kernel shape
is redesigned or the TileLang pin is bumped to a version that accepts this
decode layout.

## Rule

Do not enable HD256 TileLang decode by default on sm89 while the AOT generator
requires an `M` dimension divisible by 16 for a single-token decode shape.
