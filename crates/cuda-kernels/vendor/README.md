Vendored upstream kernels (adopt-official-first; see each entry's pin).

- `deepgemm/`: copied from `deepseek-ai/DeepGEMM` at
  `714dd1a4a980f7937a74343d19a8eba4fe321480`.
- `flashmla/`: `deepseek-ai/FlashMLA` csrc (cutlass submodule snapshot at
  NVIDIA tag `147f5673`). Linked via `arle_flashmla_shim.cu` behind
  `ARLE_CUDA_ENABLE_FLASHMLA`.
- `flash-attention/`: `Dao-AILab/flash-attention` at
  `fc8cbad6b6b90220cf6ef8121c29e299a3ba7d9a` — `hopper/` headers + the
  hdim256/bf16/fwd/sm90 instantiation set only (5 units + combine +
  prepare_scheduler; full matrix is regenerable via the vendored
  `generate_kernels.py`). `csrc/cutlass/include` is the FA3-pinned cutlass
  `7127592069c2fe01b041e174ba4345ef9b279671` — deliberately NOT shared with
  flashmla's older pin. `flash_api.cpp` is kept as the heuristics reference
  (num_splits, pagedkv_tma) and is never compiled. Consumer plan:
  `docs/plans/2026-06-11-qwen35-fa3-hd256-adoption.md`.

DeepGEMM is not linked into the default CUDA build; the ARLE path ports raw
kernels behind C ABI entry points first, then can replace selected kernels
with direct integrations. FlashMLA and flash-attention link through ARLE
shims, env-gated, sm_90a only.
