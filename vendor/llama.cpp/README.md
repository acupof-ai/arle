# llama.cpp reference kernels (pinned copy)

Operator sources copied from
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) at commit
`d2462f8f7ac6d80070a587ffebf6cd73730f4280` (2026-06-10 sparse checkout),
MIT — upstream `LICENSE` preserved in this directory. Commissioned by the
AIPC track (#71, #76/#77): serve **DSv4-Flash at 2-bit** on Ryzen AI Max+
395 (gfx1151).

## Rules

- **This directory stays pristine upstream.** Adapted kernels live in
  `crates/` (HIP lane: hipcc-compiled, CUDA syntax via
  `ggml-cuda/vendors/hip.h`; Vulkan lane: GLSL→SPIR-V at build). Never
  edit in place — copy out, adapt to ARLE's paged-KV layout, attribute.
- `template-instances/` carries only the iq2/q2_k MMQ instantiations we
  vendored; other quant types regenerate from the `mmq.cuh` macros.
- `ggml-quants.{c,h}` + `ggml-common.h` are the **offline quantizer
  reference** (block layouts + CPU quantize/dequantize) for the
  FP8-safetensors → IQ2-class converter.

## Op map (what each file is for)

| Need | HIP lane (`ggml-cuda/`) | Vulkan lane (`vulkan-shaders/`) |
| --- | --- | --- |
| 2-bit matmul (IQ2_XXS/XS/S, Q2_K) | `vecdotq.cuh`, `mmvq.*`, `mmq.*` + `template-instances/`, `dequantize.cuh`, `convert.*` | `mul_mat_vec_iq2_*.comp`, `mul_mat_vec_q2_k.comp`, `mul_mmq*`, `dequant_iq2_*.comp`, `dequant_q2_k.comp` |
| MoE (per-expert indirect matmul + router) | `mmid.*`, `topk-moe.*` | `mul_mm_id_funcs.glsl` |
| Flash attention (decode vec / prefill WMMA·coopmat) | `fattn-vec.cuh`, `fattn-tile.*`, `fattn-wmma-f16.*`, `fattn-mma-f16.cuh`, `fattn.cu` (dispatch), `fattn-common.cuh`, `mma.cuh` | `flash_attn*.comp/.glsl` (`_cm1` = KHR coopmat, works on RDNA3.5) |
| Elementwise / norm / rope / embedding / sampling | `norm.*`, `rope.*`, `softmax.*`, `unary.*`, `binbcast.*`, `getrows.*`, `argmax.*`, `quantize.*` | `rms_norm*.comp`, `rope_*`, `soft_max*`, `silu/swiglu/glu_*`, `add.comp`, `get_rows*.comp`, `argmax.comp`, `copy*.comp` |
| Qwen3.5 hybrid tier-2 (gated delta net, conv ring) | `gated_delta_net.*`, `ssm-conv.*`, `ssm-scan.*` | — (port from CUDA when scheduled) |
| Arch plumbing | `common.cuh` (RDNA3_5/gfx1151 macros), `vendors/hip.h` (CUDA→HIP shim) | `types.glsl`, `generic_*_head.glsl`, `dequant_head/funcs*.glsl` |

DSv4-Flash's own DSA/CSA/compressed-KV operators are NOT here — they are
ARLE kernels (`crates/cuda-kernels/csrc/`) and port to HIP via the
`vendors/hip.h` shim pattern. FlashMLA / DeepGEMM / DeepEP stay
datacenter-only (SM90), excluded from the AIPC lane.
