# CUDA registry-driven TileLang FFI codegen (tranche 1) — pod-verified 2026-06-24

## Context

Tranche 1 of the kernel-pipeline redesign
([plan](../../plans/2026-06-23-cuda-kernel-pipeline-redesign.md)). `kernels.toml`
(34 rows, 25 ffi) became the single source of truth for the TileLang AOT matrix;
`build.rs` now GENERATES `OUT_DIR/ffi_tilelang_generated.rs` (25 externs across 4
ABIs + `resolve_*()` dispatch), `include!`-ed by `ffi/attention.rs`. Commits
`d638c134` (tranche 0: rm dead vendored tilekernels) + `64dc0b13` (tranche 1).
The old 4-place lockstep (Python `SUPPORTED_HEADS` + `build.rs` consts + ffi
macros + consumer match arms) collapses to one toml row.

CUDA can't build on the Mac dev box → perf/correctness gate is **pending-remote**
on the H20 pod.

## What Worked (Mac-verified)

- **FFI parity proven byte-equal** (adversarial workflow verify): 25 old externs
  == 25 generated, zero missing/extra, all 4 ABI signatures byte-match
  (`paged_attn_v1` 18-arg; `fp8` 20-arg w/ k/v_scales + `*const u8`;
  `split_partial` 21-arg `*mut f32`; `split_merge` 15-arg `*const f32`) → **no
  silent GPU UB**. 93 generated dirs all covered; allow_sm70 per-row verified.
- `cargo check -p infer-api --release --features cuda,no-cuda --lib` **green**.
- `cuda-kernels` clippy-clean; tranche-1 diff adds **0** new warnings (the 7 in
  `infer-cuda` are pre-existing in ckl's `attention.rs`/`dsv4.rs`, not my hunk).
- Net **−905 lines** in build.rs; consumer 171-line 8-arm match → 44-line
  `resolve_paged_attn_v1()`, gates + `(1,1,1)`/`(1,seq,seq)` scalar choice kept.
- Caught pre-merge: 3rd enumerator `compile_tilelang_stub_kernels` (dsv4_flash
  branch) migrated to registry; `emit_ffi_generated` made unconditional-early;
  `gdr_only` filter fixed to keep flashqla (else OpdGdr → undefined `gdr_fq_*`).

## Pod-verified (8×H20 sm_90, 2026-06-24)

Built on the H20 pod from a fresh bundle clone (`/host/arle-c2`), `cargo build
--release --features cuda,nccl` + DeepGEMM-native, `BUILD_EXIT=0`:

1. **Registry regen on the real target**: `build.rs` regenerated the per-SM
   cubins from `kernels.toml` for sm_90 (the committed `generated/*.c` cover only
   sm100/sm120 Blackwell; sm_90 is TileLang-regenerated) and **linked all
   externs** — the binary built and served real models (linker parity confirmed
   end-to-end, not just symbol-count).
2. **Runtime correctness**: `needle_gate.py` (RAW + `qwen3_nonthink`) on
   **Qwen3-4B TP=1** through the regenerated `resolve_paged_attn_v1` hd128 path:
   **exact ×3 DET at all 9 lengths 115→8000** — the codegen→link→execute chain is
   correct, behavior-identical to the hand-written FFI (matches the byte-equal ABI
   proof). c=1: prefill 69 ms, decode 81.7 tok/s.
3. The 4 KernelSet enumerators all compile (full build is `KernelSet::Full`);
   `dsv4_flash`/`opd_gdr` stub branches unchanged by tranche 1.

`bench_guidellm` full sweep still owed (only c=1 curl-probe run); behavior is
codegen-identical so Δ≈0 expected.

## Rule

Generated FFI from a single registry is only safe to land once an adversarial
pass proves the emitted symbol set **and every ABI signature** are byte-equal to
the hand-written externs — a wrong `*mut`/`*const` or int width is silent GPU UB,
not a compile error. Map the FULL surface first: a third stub enumerator
(`compile_tilelang_stub_kernels`) and the unconditional-emit requirement were
invisible until the map/verify caught them.
