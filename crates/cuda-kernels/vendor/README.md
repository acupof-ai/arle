Vendored upstream kernels (adopt-official-first; see each entry's pin).

- `deepgemm/`: copied from `deepseek-ai/DeepGEMM` at
  `714dd1a4a980f7937a74343d19a8eba4fe321480`, with the SM90 FP8 MegaMoE
  dependency closure overlaid from PR #323 head
  `9e3afe91cb145ddfa0b18ae874a11dbb449e16a9`:
  `csrc/jit_kernels/heuristics/{config,mega_moe,runtime,sm90_mega_moe}.hpp`,
  `csrc/jit_kernels/impls/{runtime_utils,sm90_fp8_mega_moe}.hpp`,
  `csrc/utils/{compatibility,layout,math}.hpp`, and the transitive
  `deep_gemm/include/deep_gemm/{comm/barrier,common/{math,types,utils},impls/sm90_fp8_mega_moe,layout/{mega_moe,sym_buffer},ptx/{ld_st,utils},scheduler/mega_moe}.cuh`.
  No FP8xFP4 files or Python/TVM API surface are included.
- `flashmla/`: `deepseek-ai/FlashMLA` csrc (cutlass submodule snapshot at
  NVIDIA tag `147f5673`). Linked via `arle_flashmla_shim.cu` whenever the
  vendored tree is present.
- `flash-attention/`: `Dao-AILab/flash-attention` at
  `fc8cbad6b6b90220cf6ef8121c29e299a3ba7d9a` — `hopper/` headers + the
  hdim256/bf16/fwd/sm90 instantiation set only (5 units + combine +
  prepare_scheduler; full matrix is regenerable via the vendored
  `generate_kernels.py`). `csrc/cutlass/include` is the FA3-pinned cutlass
  `7127592069c2fe01b041e174ba4345ef9b279671` — deliberately NOT shared with
  flashmla's older pin. `flash_api.cpp` is kept as the heuristics reference
  (num_splits, pagedkv_tma) and is never compiled.

DeepGEMM is not linked into the default CUDA build; the ARLE path ports raw
kernels behind C ABI entry points first, then can replace selected kernels
with direct integrations. FlashMLA and flash-attention link through ARLE
shims, env-gated, sm_90a only.
