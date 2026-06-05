# DSv4-Flash FP8 attention-side kernel — upstream alignment scan

**Date:** 2026-06-05. **Scope:** inform the DSv4 FP8 kernel-floor optimization
(#28/#16) — the attention-side FP8 block-scaled linear (`dsv4_fp8_gemv_batch_cuda`:
MLA wq/wkv/wo + compressor + HC) and hybrid MLA attention, the shared
prefill+decode floor (roadmap §9). **Hypothesis-grade** (source survey); each item
is license-or-kill on a matched local nsys/ncu A/B + KV-parity before any flip.

**Sources (read read-only in `/tmp`, not vendored):** DeepGEMM
`88965b07` · FlashMLA `9241ae3e` · SGLang `631db6c7`.

## Q1 — Dense FP8 block-scaled GEMM (prefill, M>1) — KEEP (primary lever)

DeepGEMM exposes a dense (non-grouped) entrypoint **`deep_gemm.fp8_gemm_nt`**
(`csrc/apis/gemm.hpp:50`) → sm90 `sm90_fp8_gemm_1d2d` (1×128 act-scale +
128×128 weight block-scale, **BF16 out**) — WGMMA + TMA, warp-specialized
producer/consumer, `BLOCK_K=128`, arbitrary M/N/K (K 128-aligned). **SGLang calls
exactly this for all non-MoE projections incl. attention QKV/O**
(`deep_gemm_wrapper/entrypoint.py:170`, priority #1 backend on Hopper). This is
the documented ~10× lever vs ARLE's scalar kernel.

**Verify-local (gates):**
- **Scale layout mismatch:** sm90 dense wants **FP32, MN-major** scales
  (`make_tma_sf_desc`); ARLE's MoE DeepGEMM path uses **E8M0/int32** (the
  sm100/Blackwell form). Don't feed MoE-form scales to the sm90 dense kernel —
  reconfirm the attention weights' scale layout first.
- **No Python on hot path:** DeepGEMM is JIT/Python-driven; ARLE must reuse its
  existing **native** DeepGEMM bridge (`ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE`, the MoE
  mechanism), not the Python API.
- A/B vs the scalar kernel on the real `[2048,7168]×[7168,N]` prefill shape under
  ARLE's sync framing.

## Q2 — Decode M=1 GEMV — KILL "write a dedicated FP8 GEMV"

**Neither DeepGEMM nor SGLang ships a single-row FP8 block-scaled matvec on
NVIDIA** (grep: zero hits). M=1 routes through the **same** WGMMA GEMM
(`fp8_gemm_nt` / Triton `_w8a8_block_fp8_matmul`, `BLOCK_M=64`) — tensor-core rows
wasted but it's **HBM-BW-bound on the one weight pass**, so underutilization is
moot. The only M-aware specialization is AMD/HIP-only. This corrects the earlier
"write a memory-bound GEMV → 187µs→17µs" plan: **route decode through `fp8_gemm_nt`
too**, ncu the achieved GB/s; only hand-write (FlashMLA dequant recipe below) if
DeepGEMM's M=1 launch/tile overhead measures as the bottleneck. Hand-writing
without that measurement is the `errors/2026-05-12-fp8-kv-pair-quantize-fusion-no-license`
trap. (ARLE's own ncu shows the current scalar decode kernel at ~283 GB/s ≈ 7% of
H20's ~4 TB/s — BW-underused because it's scalar/dequant-bound, not because M=1.)

FlashMLA's FP8-KV dequant recipe (if a hand GEMV is later licensed):
`ld.global.nc.L1::evict_first/L2::prefetch.v4` 128-bit loads +
`__nv_fp8x4_e4m3 → float4 → bf16x2 × scale` (`sparse_fp8/components/dequant.h`).

## Q3 — Hybrid MLA attention (#2 floor) — KEEP dense skeleton, SW/CSA/HCA is ours

FlashMLA dense MLA decode (`csrc/sm90/decode/dense/`) is directly reusable:
**HEAD_DIM_K=576 = 512 NoPE-latent + 64 RoPE**, the **MQA Q-absorb** trick (fold
all 128 q-heads into the seq dim so the 576-latent KV is read from HBM once →
memory-bound), split-KV + a `combine` kernel for long context, WGMMA QK + PV split
over `HEAD_DIM_V/2` across two warpgroups. On H-series the FP8 sparse path is
**dequant-bound, not MMA-bound** (~50cyc dequant vs ~34cyc MMA); headline trick =
**"crossover"** (two CTAs each dequant half the KV, exchange via distributed smem
+ cluster barrier → 250→410 TFLOPS). Matmul stays BF16 (dequant FP8→BF16, FP32
accum).

**Reusable:** dense skeleton + FP8-KV dequant recipe + MQA-absorb + split-KV/combine.
**ARLE-original (not upstream):** the sliding-window / compressed-sparse /
hyper-compressed (SW/CSA/HCA) layers sit on top of MLA — FlashMLA gives only the
dense+FP8 base. **Verify-local:** FlashMLA is `sm90a` + `page_block_size==64` +
head 576/512 **hard-asserted** — match or generalize; confirm ARLE's paged-KV
block size + the NoPE/RoPE byte layout (`arle_dsv4_output_inverse_rope_cuda`,
`project_dsv4_compressed_attention_longctx_bug`) before reusing the dequant; ncu
whether ARLE decode is dequant-bound (licenses crossover) or latency-bound (not).

## Net (kept / killed)

1. **KEEP (top ROI):** prefill/M>1 attention FP8 projections → DeepGEMM
   `fp8_gemm_nt` sm90 WGMMA path (resolve the FP32-vs-E8M0 scale layout + native
   bridge first).
2. **KILL** the dedicated-FP8-GEMV framing for decode — reuse `fp8_gemm_nt`
   (BW-bound at M=1), gate any hand kernel on a local ncu GB/s measurement.
3. **KEEP** FlashMLA's dense MLA decode skeleton + FP8-KV dequant for the hybrid
   attention floor; SW/CSA/HCA is ARLE-original.
