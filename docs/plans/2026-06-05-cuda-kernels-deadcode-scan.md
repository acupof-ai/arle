# cuda-kernels dead-code scan + #18 cleanup execution plan

**Date:** 2026-06-05. **Method:** FFI-symbol caller scan — 232 `*_cuda` symbols
declared in `crates/cuda-kernels/src/ffi/`, counted callers across all crates
(excluding the `ffi/` decls). **127 of 232 (55%) have zero callers anywhere.**
This is the `necessity-not-callers` trap at scale: zero-caller ≠ dead. Classified
below before anything is deleted.

## Execution gate (why not now)

The kernel-lib code cleanup is **blocked on Codex's live DSv4 kernel work**
(DeepGEMM dense FP8 GEMM + FlashMLA attention). Concretely: Codex has
`csrc/gemm/deepgemm_native.cu`, `src/ffi/gemm.rs`, `src/moe.rs`, `src/tensor.rs`
dirty — and the biggest dead chunk (Marlin) has its FFI decls **in `ffi/gemm.rs`**.
Plus `.cu` removal can't be compile-verified on the Mac (no nvcc) — it needs a pod
build. **Execute post-DSv4-kernel-arc, one category per commit, each gated on a
pod `cargo build -p cuda-kernels` + the relevant parity test.**

## Confident-dead (rewrite dropped the path) — ~5.3k+ LOC

- **Marlin** (`csrc/gemm/marlin_kernel.cu` 869, `marlin_w4a8_kernel.cu` 1086,
  `marlin_dequant.cuh` 651, `marlin_pf8/{dequant.h 611, marlin_template.h 2081}`
  ≈ **5298 LOC**, + `w2a16/w4a16/w8a16_gemv_*_cuda`, `gemm_w4_fp8_marlin_cuda`,
  `gemm_w4a8_marlin_cuda`, `marlin_gemm_cuda` Rust ffi + wrappers). **Zero
  consumers** in infer-cuda/infer-api — the rewrite uses DeepGEMM/native FP8, never
  Marlin. Highest-ROI deletion.
- **GGUF k-quant** (`q3k/q4k/q5k/q6k/q8_{gemv,embedding,dequant}_*_cuda` ≈ 22
  symbols + their `.cu`). Zero exact-symbol callers; the rewrite loads safetensors
  (FP8/BF16), no GGUF loader. **Verify** the broad-grep hits in `qwen35.rs`/
  `attention.rs`/`moe.rs` are substring false-positives (not real k-quant uses)
  before deleting.

## KEEP — necessary-but-unwired or live (do NOT delete on caller-count)

- **turboquant** — LIVE: the KV TQ path (`kv_turboquant.rs`, `paged_kv.rs`,
  `turboquant_state.rs`, `tensor.rs`). Per-variant zero-caller symbols
  (`turboquant_dequantize_paged_cuda` etc.) may be prunable, but the system stays.
- **gated_delta_rule_prefill_chunk_*** (8) — Qwen3-Next gated-delta-net, referenced
  in `qwen35.rs`. Necessity-unwired (future model). KEEP.
- **dsv4_fp4_*** (FP4 experts) — FP8/FP4 is config-gated (`SGLANG_DSV4_FP4_EXPERTS=0`
  in prod); supported-but-off. KEEP until the FP4 path is decided.

## AVOID (Codex-live) — re-scan after the DSv4 kernel arc lands

- `dsv4_*route*/*pack*/*scatter*/*dispatch*` (DeepEP/route helpers) — Codex is in
  `moe/dsv4_route.cu` + `src/moe.rs`. Several show zero-caller mid-refactor; do not
  judge until that work commits.
- `dsv4_fp8_gemv_*` family — the active FP8-GEMV→DeepGEMM target.

## Other clarity (non-deletion)

- `docs/reviews/kernel-registry.md` is **stale** — references deleted
  `batch_decode.rs`/`prefill.rs`/`forward.rs` (the KVFormat→kernel `match`
  collapsed to `infer-cuda/src/executor.rs` in the rewrite). Refresh post-cleanup.
- `docs/reviews/2026-05-30-cuda-kernel-systematic-audit.md` headline ("kernels are
  NOT the easy win") is **superseded** by the 2026-06-05 ncu finding: the FP8 GEMV
  is a scalar kernel (tensor pipe <1%) and the SGLang H20 A/B shows a real 2.5×
  kernel gap. Re-rank after the DeepGEMM dense lands.

**Estimated removable (confident): ~5.3k LOC (Marlin) + ~GGUF k-quant once
verified.** Net of the necessity-keepers, this is the bulk of #18.
