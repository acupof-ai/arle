# TileKernels Integration Decision — Don't Submodule, Port Selectively

Date: 2026-05-02
Status: Decision recorded after async subagent investigation
Scope: DeepSeek V4 kernel path (DS2-DS6 per `2026-05-01-deepseek-v4-readiness.md`)

## TL;DR

`cklxx/TileKernels` is a fork of `deepseek-ai/TileKernels` (DeepSeek-authored,
MIT-licensed). It does **not** contain the three load-bearing DSv4 kernels
(MLA decode, grouped expert GEMM, block-FP8 GEMM). Direct integration as a
git submodule is **not warranted**. Instead, treat the repo as a **TileLang
reference implementation** for the routing/cast primitives it does ship, and
port them into ARLE's existing AOT track at
`crates/cuda-kernels/tools/tilelang/`.

## What TileKernels has vs DSv4 needs

| DSv4 need | TileKernels coverage |
|---|---|
| MLA decode (paged, compressed `c_kv` + `k_rope`) | None |
| MLA prefill | None |
| MoE top-k routing | Yes — `topk_gate_kernel.py`, `top2_sum_gate_kernel.py`, `get_fused_mapping_kernel.py`, `mask_indices_by_tp_kernel.py` |
| MoE expand/reduce (combine/scatter) | Yes — `expand_to_fused_kernel.py`, `reduce_fused_kernel.py` |
| Grouped expert GEMM (sgmm) | None — only index plumbing, no matmul |
| Block-FP8 weight cast | Yes — `per_block_cast_kernel.py` (configurable 32 / 128 block) |
| Block-FP8 GEMM | None |
| MTP (multi-token prediction) | None |
| Attention kernels (any) | None — zero overlap with ARLE's `crates/cuda-kernels/csrc/attention/` |

The three highest-cost DSv4 kernels (MLA decode, grouped GEMM, block-FP8
GEMM) are exactly the ones TileKernels does not ship. The repo is a kernel
pack of routing / cast / gating *plumbing*, not the dense compute path.

## Compatibility blockers (would apply to any submodule plan)

1. **No C ABI**: every TileKernels kernel is a Python callable returned by
   `@tilelang.jit`; modeling layers are `torch.autograd.Function`. ARLE's
   FFI contract is `extern "C" fn xxx_cuda(...)` linked into
   `libcuda-kernels.a`. There is no `.so` artifact, no header, no
   extern-C symbol to call from Rust. Production runtime contains no
   Python — matches ARLE's existing posture.
2. **SM90/SM100 only**: L4 (sm_89) — ARLE's primary CI/bench box — is
   unsupported. ARLE's existing TileLang prefill kernels already had to
   halve `BLOCK_N` for L4 fit (codex T3 review finding); TileKernels has
   no L4 retuning.
3. **TileLang 0.1.9+ pin**: matches ARLE's pin per
   `project_tilelang_0p1p9_aot_blocker`, but any ABI shift in upstream
   TileLang affects both code paths.
4. **Training-oriented modeling**: the `modeling/` autograd wrappers
   compose forward + backward; ARLE serving uses forward only. Reusing
   the wrappers as-is would pull autograd state through the inference
   hot path.
5. **Alpha maturity**: 3 commits, 1 day of upstream activity, no L4
   testing.

## What we will do instead

1. **DS3 MLA decode + prefill**: implement ARLE-side per
   [`2026-05-01-mla-kernel-design.md`](../plans/2026-05-01-mla-kernel-design.md).
   Reference FlashMLA / FlashInfer MLA. TileKernels offers nothing here.
2. **DS4 MoE top-k routing + expand/reduce**: read TileKernels' Python
   sources as a reference implementation (`tile_kernels/moe/`); port the
   logic into `crates/cuda-kernels/tools/tilelang/` with SM89 support and
   ARLE's AOT codegen path. Attribute under MIT.
3. **DS4 grouped expert GEMM**: ARLE-side implementation required
   (Marlin / Cutlass grouped / hand-rolled). Not in TileKernels.
4. **DS2 block-FP8 weight quantization**: port
   `per_block_cast_kernel.py` (block = 128) into ARLE's TileLang AOT
   path for the cast side. The GEMM side needs ARLE-side authoring.
5. **Watch upstream**: once `deepseek-ai/TileKernels` adds MLA, grouped
   GEMM, or block-FP8 GEMM, and once SM89 support lands, re-evaluate.

## Files cited

- ARLE DSv4 readiness: `docs/projects/2026-05-01-deepseek-v4-readiness.md`
- ARLE MLA design: `docs/plans/2026-05-01-mla-kernel-design.md`
- ARLE TileLang AOT: `docs/plans/tilelang-integration.md`,
  `crates/cuda-kernels/tools/tilelang/`
- ARLE existing kernel layout: `crates/cuda-kernels/csrc/{attention,gemm,kv,quant,misc}/`
- TileKernels upstream: https://github.com/deepseek-ai/TileKernels
  (fork: https://github.com/cklxx/TileKernels)

## Re-open trigger

If any of the following land upstream in `deepseek-ai/TileKernels`,
revisit this decision:

- MLA decode kernel with paged-KV layout
- Grouped expert GEMM (sgmm)
- Block-FP8 GEMM
- SM89 (L4 / RTX 4090) tuning

Until then: port what's useful, build what's missing, no submodule.
