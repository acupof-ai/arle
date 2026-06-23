# CUDA registry-driven TileLang FFI codegen (tranche 1) — pending-remote

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

## Pod-pending (bench/correctness gate — H20)

1. `cargo build --release --features cuda` green for **all 3 KernelSet branches**:
   `full`, `opd_gdr` (GDR+FlashQLA stub link), `dsv4_flash` (all-stub link).
2. **Linker parity**: the 25 generated externs (`kernel_name + "_cuda"`) resolve
   1:1 against `libtilelang_kernels_aot.a` symbols from `format_dispatch_wrapper`.
3. `scripts/needle_gate.py` ×3 same-config vs the **bf16 envelope** (tranche 1 is
   behavior-identical — must match pre-redesign).
4. `scripts/bench_guidellm.sh` vs latest Qwen3.5 CUDA baseline, Δ% row (expect
   ~0 — codegen change, not a perf change).
5. **Offline feed**: the new `toml` build-dep tree (`serde_spanned, toml,
   toml_datetime, toml_edit, toml_write, winnow`) must be in the pod's cargo
   registry cache before the offline build.

## Rule

Generated FFI from a single registry is only safe to land once an adversarial
pass proves the emitted symbol set **and every ABI signature** are byte-equal to
the hand-written externs — a wrong `*mut`/`*const` or int width is silent GPU UB,
not a compile error. Map the FULL surface first: a third stub enumerator
(`compile_tilelang_stub_kernels`) and the unconditional-emit requirement were
invisible until the map/verify caught them.
