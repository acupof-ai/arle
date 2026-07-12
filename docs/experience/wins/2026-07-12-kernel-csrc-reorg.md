# CUDA kernel `csrc/` reorg — `misc/` junk drawer exploded, dead code deleted

**Date:** 2026-07-12. **Backend:** CUDA (`crates/cuda-kernels/csrc/`).
**Commits:** `a07a48d90` (delete dead kv+marlin GEMM), `9fc53e7e4` (explode
`misc/`), `051edb29b` (relocate last file + delete `misc/` dir). **Bench-exempt:**
pure directory reorg + dead-code deletion — no runtime code path changed
(byte-identical kernels, `build.rs` walks `csrc/` recursively so nvcc picks up
the new dirs automatically). No FFI symbol with a live caller was touched.

## Context

The `kernel-registry.md` audit (`5ea39b464`, 2026-07-12) flagged `misc/` as a
19-file **MISPLACED dumping ground**: it held core live ops (`norm.cu`,
`sampling.cu` incl. DSpark chain-rejection, `elementwise_basic.cu`, the entire
live DSv4 attention family, Qwen3.5 linear-attn, the FlashMLA/FA3 shims) with no
family boundary and no alignment to the `src/ffi/*.rs` split that already
existed. The same audit listed 3 Marlin W4/W4A8 GEMM `.cu` + 2 `kv/` kernels
with **0 callers**.

## What Worked

**Deleted dead code (`a07a48d90`, 15 files, −6545 LOC):** Marlin GEMM
(`marlin_kernel.cu` 869, `marlin_w4a8_kernel.cu` 1086, `marlin_w4_fp8_kernel.cu`
308, `marlin_dequant.cuh` 651, `marlin_pf8/` 7 headers ~3393) — the repack path
(`gptq_marlin_repack`, `marlin_int4_fp8_preprocess`) stays live, only the
never-called GEMM kernels went; `kv/paged_kv_append.cu` (89), `kv/scatter_kv.cu`
(63) — superseded by the TileLang kv8 path; plus their 5 `extern "C"` decls in
`src/ffi/{gemm,kv}.rs`.

**Exploded `misc/` 19→0 (`9fc53e7e4` + `051edb29b`):** new `sampling/`·`norm/`·
`recurrent/`·`elementwise/` dirs; DSv4 MLA/DSA/MHC + TP-repack + FlashMLA/FA3
shims → `attention/`; `fused_mlp.cu` → `gemm/`, `split_qkv.cu` → `attention/`;
`kvcacheio/transfer.cu` merged into `kv/`. `misc/` directory removed. Every
family now aligns 1:1 with `src/ffi/{sampling,norm,recurrent,elementwise,
attention,...}.rs`. Result: `csrc/` = 56 `.cu` across 10 kernel dirs +
`deepep_sidecar/`.

## Rule

**判死代码要 grep 全树,包含 `cuda-kernels/src` 自身,不只 `infer-cuda`.** During
this pass `quant/turboquant*.cu` was momentarily misjudged dead — a grep scoped
only to `infer-cuda/src` callers showed 0 hits. But `KVFormat::TurboQuant`
(`kv_types.rs`) + `paged_kv.rs` reference it *inside* `cuda-kernels/src`: it is
the deferred TQ4 feature, not dead. `quant/` was correctly kept. Deleting a
kernel requires grepping the **whole workspace tree including the defining crate
itself**, not just the primary consumer crate — a half-tree grep produces a
false-dead verdict. (Reorg/deletion of `.cu` files with 0 whole-tree callers is
byte-identical to callers and needs no bench; a domain-dir move is picked up by
the recursive `build.rs` walk with zero code change.)
