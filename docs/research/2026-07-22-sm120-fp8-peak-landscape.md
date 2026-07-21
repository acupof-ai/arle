# sm_120 (Blackwell RTX PRO 6000) FP8 peak-perf landscape

> Status: Active — 4-stream deep research, 2026-07-22. Grounds the sm_120 peak-perf
> plan ([2026-07-21-sm120-tc-kernel-compat.md](../plans/2026-07-21-sm120-tc-kernel-compat.md) §6).

**Bottom line.** The vendored peak FP8 GEMM path on sm_120 is **CUTLASS sm_120
block-scaled collective** (`mma.sync…block_scale`, the vLLM PR #22131 / CUTLASS
example-79 family) — **not DeepGEMM** (no sm_120 kernels, ever — it needs tcgen05
that GB202 lacks) and **not cuBLASLt as the fast path** (per-tensor only on sm_120,
runs at the *throttled* legacy-MMA rate). Consumer Blackwell **halves FP32-accumulate
legacy warp-MMA but does NOT throttle block-scaled MMA**, so per-tensor FP8 gets ~2×
BF16 while block-scaled FP8/MXFP8 gets ~4× — the throughput case for block-scaled is
also the correctness-of-format case. Attention has **no Blackwell-native FMHA for
hd256**; the peak is FA2 (Ampere-class MMA, forward-compiled to sm_120). All four
frameworks already run these exact models on RTX PRO 6000, so the target is proven.

## 1. The governing hardware fact — sm_120 ≠ sm_100

[EVIDENCE, cross-validated by all 4 streams] GB202 (sm_120: RTX PRO 6000 / RTX 5090)
is architecturally NOT B200 (sm_100). It has **no `tcgen05` instruction and no TMEM**
(tensor memory); it uses the Ampere-lineage `mma.sync.aligned…block_scale` warp-MMA
family, with **99 KiB shared mem/SM** (vs sm_100's 256 KB TMEM). Every sm_90/sm_100
FP8 kernel written against tcgen05 (DeepGEMM SM100, FlashMLA, trtllm-gen FMHA, CUTLASS
sm_100 collectives) **must be rewritten kernel-by-kernel** for sm_120 — it does not
recompile. `tcgen05.fence not supported on sm_120f` ([vLLM #41063](https://github.com/vllm-project/vllm/issues/41063)).

**The throughput fact that drives the format choice** [EVIDENCE, SASS-verified
microbench, [flashinfer #3628](https://github.com/flashinfer-ai/flashinfer/issues/3628)]:
consumer Blackwell halves FP32-accumulate throughput for *legacy* warp-MMA, but the
*block-scaled* tensor instruction escapes the throttle:

| path | SASS | TFLOP/s | vs BF16 |
|---|---|---:|---:|
| BF16, FP32 acc | `HMMA.16816.F32` | ~51 | 1× |
| plain FP8 (per-tensor), FP32 acc | `QMMA.16832.F32.E4M3` | ~102 | **2×** |
| **MXFP8 block-scaled, FP32 acc** | `QMMA.SF.16832.F32.E4M3.E8` | **~202** | **~4×** |

Datasheet confirms the ceiling: RTX PRO 6000 Blackwell **FP8 = 2 PFLOPS, BF16 = 1
PFLOP** ([datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf)).
**Per-tensor FP8 leaves half the tensor cores idle; block-scaled is the peak.**

## 2. FP8 GEMM path — CUTLASS sm_120 block-scaled, not DeepGEMM, not cuBLASLt

- **DeepGEMM: NOT available on sm_120, will not be soon.** Upstream is Hopper sm_90 +
  sm_100 (tcgen05) only ([DeepGEMM README](https://github.com/deepseek-ai/DeepGEMM);
  sm_120 asserts "Unsupported architecture"). Only a partial third-party fork
  (`jasl/DeepGEMM`) has any sm_120 kernels; FP8/FP4 GEMM still missing. vLLM's own fix
  for a DeepGEMM-on-sm_120 load crash is to **turn it off** (`VLLM_USE_DEEP_GEMM=0` →
  routes to CUTLASS, [vLLM #47436](https://github.com/vllm-project/vllm/issues/47436)).
- **CUTLASS sm_120 block-scaled collective = the framework consensus.** sm_120 is a
  first-class CUTLASS arch target with its own examples: `79_blackwell_geforce_gemm/`
  — 79a/79b NVFP4, **79c mixed MXFP8/MXFP6→BF16**, 79d NVFP4 grouped ([CUTLASS ex79](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm.cu)).
  vLLM shipped the block-FP8 (128×128) sm_120 path via CUTLASS in **PR #22131**
  (merged 2025-08-08, "based on cutlass", 8.5–57.9% faster than Triton). SGLang
  backported it (#9233) and tracks the broader sm_120 lane in #19637.
- **cuBLASLt = mature fallback, throttled.** Per-tensor `CUDA_R_8F_E4M3` FP8 works on
  sm_120, but the block-scale modes `BLK128x128_32F`/`VEC128_32F` are **Hopper-sm_90
  ONLY** (software-emulated); sm_120 exposes only `SCALAR_32F` (per-tensor) + native
  `VEC32_UE8M0` (MXFP8) + `VEC16_UE4M3` (NVFP4) ([CUDALibrarySamples #310](https://github.com/NVIDIA/CUDALibrarySamples/issues/310)).
  cuBLASLt sm_120 also **lacks non-TN GEMM layouts** (MXFP8 forward-only,
  [TransformerEngine #2668](https://github.com/NVIDIA/TransformerEngine/issues/2668)).
  Per-tensor path hits the throttled legacy MMA → 2× not 4×.

## 3. Scale format — 128-block E4M3 with UE8M0 scales

- CUTLASS sm_120 block-scaled **does** consume 128×128-block E4M3 (vLLM PR #22131 uses
  `block_shape=[128,128]`), **provided the scales are UE8M0** (power-of-2 exponent).
  A checkpoint whose `scale_fmt ≠ ue8m0` crashes the loader ([vLLM #47436](https://github.com/vllm-project/vllm/issues/47436)).
- cuBLASLt cannot take 128×128 at all on sm_120 — to use its *native* block path you
  must requantize to **MXFP8 (32-element UE8M0)** or **NVFP4 (16-element UE4M3)**. The
  TransformerEngine `Float8BlockScaling`→MXFP8 broadcast does exactly this on all
  Blackwell (~3% scale-data overhead; UE8M0 is power-of-2 only, so it is a genuine
  requant, not a reshape) ([TE #2668](https://github.com/NVIDIA/TransformerEngine/issues/2668)).
- **Action item**: confirm our DeepSeek-format weights store **UE8M0** 128-block scales
  (then CUTLASS sm_120 takes them directly) vs arbitrary f32 (then a UE8M0 requant is
  required). DeepGEMM's SM100 path already repacks to UE8M0, so the format likely
  exists in the checkpoint or the loader.

## 4. Attention — no Blackwell-native FMHA for hd256; peak is FA2

[EVIDENCE, stream 4] FA3 **explicitly refuses Blackwell** (`compute_cap >= 10` gated
off, [Dao-AILab #1853](https://github.com/Dao-AILab/flash-attention/issues/1853)); FA4
is sm_100/sm_103 + hd≤128; CUTLASS FMHA (ex77) is sm_100 (tcgen05); FlashInfer has
sm_120 **MoE** (MXFP4 CUTLASS, ~0.6.6, +13–28% vs Marlin) but **not attention**
(prefill is an open RFC, [#3628](https://github.com/flashinfer-ai/flashinfer/issues/3628)).
**The working peak for hd256 on sm_120 is FA2** (mma.sm80-class, hd256-capable,
PTX-forward-compiled). Our TileLang `batch_prefill_paged_hd256` uses the generic
`GemmMMASm70` MMA path — this IS the FA2-class kernel, so if TileLang emits it for
sm_120, it is the attention peak; the in-tree SIMT `nonpaged_prefill_attention` is the
correctness floor.

## 5. TileLang on sm_120 — compiles, but buggy

[EVIDENCE, stream 4] TileLang emits sm_120 cubins via nvcc (CUDA ≥12.8) through the
**sm_89-class MMA path** (not tcgen05/TMEM, which is sm_100-only). But sm_120 is in
open issues, not changelog features (latest v0.1.12 is sm_100-centric), and carries
active bugs on this exact GPU: **JIT hang on RTX PRO 6000** ([#2328](https://github.com/tile-ai/tilelang/issues/2328)),
**data-race → non-deterministic wrong results** ([#2703](https://github.com/tile-ai/tilelang/issues/2703)),
**99 KiB shared-mem overflow** ([#2201](https://github.com/tile-ai/tilelang/issues/2201)).
Workaround: pin **sm_90/sm_80 tile configs (`ARLE_TILELANG_CUDA_ARCH=90`) compiled
`arch=sm_120`** (avoids the templates sm_120 can't lower, keeps SMEM ≤99 KiB). This is
the load-bearing assumption under empirical test in the live Colab build.

## 6. Real numbers — frameworks already run these models on RTX PRO 6000

[EVIDENCE, [lastloop-ai vllm-blackwell-guide](https://github.com/lastloop-ai/vllm-blackwell-guide),
May 2026, RTX PRO 6000 96GB]:
- **Qwen3.6 35B-A3B MoE FP8** (flashinfer + MTP n=3 + FP8 KV, 256K ctx): **170 tok/s**
  decode.
- Qwen3.6 27B INT4: 24 (eager) → 75 (CUDA graphs + flash_attn) → 100 (flashinfer + MTP
  + FP8KV) tok/s; MTP mean acceptance 3.19.
- DeepSeek-V4-Flash-FP8 on 8× RTX PRO 6000: 40–50 tok/s decode, ~2000 tok/s prefill.
- FP8-KV gotcha: ~15% penalty at short ctx, win only past ~128K (bandwidth-bound).
- NVFP4: Qwen3-32B NVFP4 ≈ 2× BF16 throughput, lowest TTFT at all concurrencies.

No clean matched FP8-vs-BF16 same-model tok/s on a single RTX PRO 6000 surfaced — that
is our A/B to run.

## 7. Recommendation for ARLE (evidence-grounded)

1. **Peak FP8 GEMM = adopt the CUTLASS sm_120 block-scaled collective** (integrate the
   CUTLASS example-79c MXFP8 / vLLM PR #22131 kernel as a new `.cu`), wired as the
   `major==12` route. NOT DeepGEMM (absent), NOT cuBLASLt-fast (throttled), NOT
   hand-tcgen05 (unusable on sm_120).
2. **Prove-the-route first with cuBLASLt per-tensor FP8** (mature, reuses the
   `gemv.cu:539` cuBLASLt scaffold) to validate FFI + dispatch + sm_120 build
   end-to-end at 2× — then swap in the CUTLASS block-scaled kernel for the ~4× peak.
3. **Force DeepGEMM off on sm_120** — the G1 fix does this for the dense proj path;
   confirm the grouped/MoE path (G2) also routes away from DeepGEMM (the CUTLASS
   block-scaled *grouped* GEMM is the sm_120 MoE path, per SGLang #19637 / vLLM).
4. **Scales**: confirm UE8M0 128-block; requant if arbitrary-f32.
5. **Attention**: TileLang hd256 (FA2-class) with `ARLE_TILELANG_CUDA_ARCH=90`;
   SIMT floor as fallback. No Blackwell-native FMHA to adopt.
6. **Strategic (long-term)**: **NVFP4 is the ecosystem-consensus Blackwell-native
   format** and the only path to un-throttled ~4× that also fits 96 GB comfortably.
   For a workstation-Blackwell deployment target, NVFP4 (a separate 4-bit requant) may
   beat FP8 — evaluate as a follow-up, not the first cut.

## Streams (all read-only web research, repo-issue/doc EVIDENCE, not local builds)
1. cuBLASLt FP8 sm_120 scale modes · 2. DeepGEMM/CUTLASS Blackwell FP8 · 3. Framework
practice (vLLM/SGLang/TRT-LLM) · 4. Attention + TileLang sm_120. Full source URLs
inline above. The one thing needing a local verify (in-flight on Colab): does our
CUDA 12.8 + TileLang actually emit a correct sm_120 hd256 cubin, or hit #2703/#2328.
