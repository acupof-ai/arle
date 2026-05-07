# 2026-05-07 · M_b.1 Phase A — TileLang HD128 paged decode kernel codegen

## Context

Phase 1 nsys trace (`fdb531b`) shows
`decode_attention_varlen_quantized_partial_kernel<128, 1, 0>` =
**41.6% of all GPU time** at avg 4.55 ms × 4,104 calls. The HD128
decode path is the largest single kernel-axis target.

`crates/cuda-kernels/tools/tilelang/` covered HD128 prefill, HD256
prefill, and HD256 decode — but **not HD128 decode**. The existing
HD128 BF16 decode path in
`infer/src/ops/attention.rs::tilelang_tc_run_layer` reused the HD128
**prefill** kernel as a "TC decode alias", paying for unnecessary
causal-mask logic and Q_indptr indirection. The HD128 FP8/INT8 path
(the production hot path, the 41.6% above) goes through hand-CUDA in
`crates/cuda-kernels/csrc/attention/decode_attention_*.cu`.

Plan reference: [`M_b-tilelang-hd128-decode.md`](../../plans/M_b-tilelang-hd128-decode.md).

## What Worked

Phase A scope: **add the kernel + FFI without changing dispatch**. The
goal is to validate that TileLang's HD128 decode IR codegens cleanly
before any runtime path switches to it.

Three files:

- **NEW** `crates/cuda-kernels/tools/tilelang/batch_decode_paged_hd128.py`
  — direct port of `batch_decode_paged_hd256.py` with
  `HEAD_DIM=128`, `SM_SCALE=1/√128`, and
  `SUPPORTED_HEADS = [(16,8),(32,8),(40,8),(64,8)]`. Tile/pipeline
  tunables (`BLOCK_M=64`, `BLOCK_N=PAGE_SIZE=16`, `NUM_STAGES=2`,
  `NUM_THREADS=128`) unchanged from the HD256 decode template — they
  are HEAD_DIM-independent at this scale and shared-memory budget
  (~32 KB double-buffered) is well under every SM cap.
- **EDIT** `crates/cuda-kernels/build.rs` — added
  `TILELANG_DECODE_HD128_HEAD_CONFIGS = &[(16,8),(32,8),(40,8),(64,8)]`
  + the codegen iteration loop modeled on the HD256 decode block. Also
  updated the `cargo:warning=TileLang AOT...` summary string to
  mention HD128 decode.
- **EDIT** `crates/cuda-kernels/src/ffi/attention.rs` — added
  `tilelang_decode_hd128_decl!` macro mirroring `tilelang_decode_hd256_decl!`,
  with four declarations:
  - `tilelang_batch_decode_paged_hd128_q16_kv8_run_cuda`
  - `tilelang_batch_decode_paged_hd128_q32_kv8_run_cuda`
  - `tilelang_batch_decode_paged_hd128_q40_kv8_run_cuda`
  - `tilelang_batch_decode_paged_hd128_q64_kv8_run_cuda`

## Verification

Codegen artifacts produced under `target/release/build/cuda-kernels-*/out/`:

- `tilelang_aot/batch_decode_paged_hd128_q{16,32,40,64}_kv8_sm89/` — per-SM cubin dirs ✅
- `tilelang_aot/batch_decode_paged_hd128_q{16,32,40,64}_kv8_dispatch/` — dispatch wrappers ✅
- `tilelang_batch_decode_paged_hd128_q{16,32,40,64}_kv8_*.o` — host objects linked into `tilelang_kernels_aot` static lib ✅

Build/typecheck gates:

- `cargo check --release -p infer --features cuda` ✅ 3m03s (full TileLang regen, all 4 kernels lowered + nvcc-compiled successfully)
- (gauntlet pending: `cargo clippy --features cuda -- -D warnings`, `cargo check --features cuda,no-cuda`, `cargo check --features metal,no-cuda`, `cargo fmt --all --check`)

## Bench Status

**`pending-phase-B`** — Phase A is purely additive (new kernels available, no
runtime dispatch wired). No runtime path changes, so no bench delta is
expected vs `2a534c4`. Phase B will switch
`tilelang_tc_run_layer` to dispatch the new HD128 decode kernel when
`max_qlen == 1` (pure decode, no varlen Q), then run:

```
scripts/bench_guidellm.sh m_b1-arle-s48-bf16
```

against the BF16 KV path (Phase 1 baseline used FP8 KV; the hand-CUDA
FP8 hot path will only move with M_b.2).

## Rule

When porting a new TileLang specialization, **land the kernel + FFI as
a separate commit before changing any dispatch arm**. This isolates
the IR-validation step from the routing change: if codegen breaks, the
diff is small and obvious; if dispatch breaks, the codegen is already
proven.

When the existing path "reuses" a kernel from a different shape (here:
HD128 prefill kernel running decode workloads), the *correct* kernel
absence is invisible from runtime metrics — only a kernel-coverage
audit (`ls tools/tilelang/`) surfaces it. Run that audit any time
nsys says "kernel X dominates" and X has no matching `tools/tilelang/`
entry.
