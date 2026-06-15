# DSv4 lm_head decode: kernel survey ("is there a better kernel online?")

**Date:** 2026-06-15 · **Track:** CUDA / DSv4-Flash decode throughput (8×H20 TP=8/EP=8)
**Question (ckl):** the lm_head GEMV is the #1 nsys decode hog — use cuBLAS for
now, and *check whether there's a better kernel online*.

## TL;DR verdict

The lm_head decode step is **weight-read bandwidth-bound**, not kernel-quality
bound and not logits-bound. Each call reads the **full ~1 GB bf16 vocab weight**
(vocab 129 280 × hidden 4096 × 2 B, **replicated** on every TP rank). At M≥2 a
batched cuBLAS GEMM already runs at the HBM floor for that read. So **no exotic
kernel helps**; the only real levers are *traffic-reduction* levers, both lifted
from the SGLang/vLLM playbook:

1. **Batch the per-row calls** — read the 1 GB weight **once per step** instead of
   16–48×. (My committed `5d6eb0da` does this for the *verify*; **Lever 2a batched
   draft** does it for the *draft*.) — **primary, do first.**
2. **Vocab-parallel shard the lm_head** (`ParallelLMHead`) — each rank reads
   **1/8** the weight (~132 MB) + a cheap logits all-gather. — **secondary,
   structural, diminishing once (1) lands.**
3. Keep **bf16**. **Do not** FP8/FP4-quantize lm_head.
4. **Do not** swap the GEMM kernel (cuBLAS is at the floor); **do not** chase
   fused-sampling kernels (they save the tiny logits tensor, not the 1 GB read).

## The physics (why kernel choice is a red herring)

| quantity | value |
|---|---|
| lm_head weight (bf16, replicated/rank) | 129 280 × 4096 × 2 B ≈ **1.06 GB** |
| logits per row (bf16) | 129 280 × 2 B ≈ **0.26 MB** |
| measured hand-GEMV (`gemv_handwritten`) | **558 µs/call** ⇒ ~1.9 TB/s ≈ HBM floor |
| arithmetic intensity at M=1 | ~2 FLOP/byte → **bandwidth-bound** |

The weight read dwarfs the logits tensor by **~4000×**. Any optimization that
touches the logits (fused sampling, sorting-free top-k) addresses 0.05 % of the
traffic. The only way to move the needle is to **read the 1 GB weight fewer
times** (batch) or **read less of it per rank** (shard) or **read fewer bytes**
(quantize — rejected on precision).

## What SGLang / vLLM actually do

- **lm_head stays high-precision (bf16).** Quantization configs **explicitly skip
  `lm_head`** — it is not FP8/FP4 even in otherwise-FP8 DeepSeek deployments. So
  DSv4 keeping lm_head bf16 is *correct by industry convention*, not an oversight.
  ([vLLM logits_processor][vllm-lp], [SGLang DeepSeek-V3 FP8/BF16][lmsys-h20])
- **lm_head is vocab-parallel sharded** via `VocabParallelEmbedding` /
  `ParallelLMHead` (Megatron TP): each rank computes its `vocab/TP` slice of the
  logits, then an **all-gather** assembles the full logits for sampling. Weight
  and bias are padded to be divisible by TP. ([vLLM vocab_parallel_embedding][vllm-vpe])
  — **ARLE currently does NOT do this**: `load_dsv4_global_matrix` loads the full
  vocab on every rank (the 1 GB/rank read above).

## Candidate kernels evaluated

| candidate | verdict for our shape (M=1..32, N=129 280, K=4096, bf16) |
|---|---|
| **cuBLAS batched GEMM** (current fix) | ✅ at HBM floor for the weight read once M≥2; the right primitive for bf16. |
| **DeepGEMM** | ❌ FP8-primary; **bf16 is "work-in-progress"**; groups only the M-axis (MoE), not a vocab GEMM. Using it would force FP8 lm_head (rejected). ([DeepGEMM][deepgemm]) |
| **FlashInfer fused top-k/top-p sampling** | ➖ sorting-free, real win *for sampling*, but still materializes logits; irrelevant to the weight-read bottleneck. ([FlashInfer sampling][flashinfer-sampling]) |
| **FlashSampling** (fuses sampling into the lm_head matmul, never materializes logits) | ➖ elegant, but saves only the 0.26 MB logits/row — 0.05 % of the 1 GB weight traffic. Also incompatible with MTP's need for full-vocab argmax on the verify path. ([FlashSampling][flashsampling]) |
| **CUTLASS split-K / Marlin / Machete (W4A16)** | ❌ W4A16 = 4-bit weight = quantized lm_head (rejected); split-K helps compute-bound GEMMs, ours is bandwidth-bound. |
| **Custom tall-skinny GEMM (TSM2X etc.)** | ❌ academic gains target compute-bound TS-GEMM; the 1 GB weight read is already at bandwidth. |

## Recommendation (ordered, license-or-kill)

1. **Lever 2a — batched draft (`ARLE_DSV4_BATCHED_MTP_DRAFT`), re-measure + flip.**
   The default `draft_chain` is per-slot serial → c=16 fires 16× m=1 lm_head GEMMs
   = 16 full weight reads/step. `mtp_forward_level_batched` collapses this to **one
   m=16 GEMM = one weight read**. Its prior "marginal" verdict is **stale** — it
   was measured before `5d6eb0da` made the batched lm_head a real cuBLAS GEMM (it
   was internally still a per-row loop), so batching slots saved nothing on the
   lm_head. **Re-A/B on the fixed binary.**
2. **Vocab-parallel lm_head shard** (`ParallelLMHead` equivalent): shard the weight
   by vocab rows at load, compute partial logits, all-gather. 8× less weight/rank.
   Bigger change; marginal value once (1) drops lm_head reads to ~2/step. Note as a
   follow-on, not a blocker.
3. **Rejected:** FP8/FP4 lm_head (off-policy precision), new GEMM kernel
   (bandwidth-bound), fused sampling (wrong bottleneck).

## Sources

- [vLLM `logits_processor`][vllm-lp]
- [vLLM `vocab_parallel_embedding` / ParallelLMHead][vllm-vpe]
- [LMSYS — SGLang DeepSeek-R1 on H20 (FP8/BF16)][lmsys-h20]
- [DeepGEMM (FP8-primary, bf16 WIP)][deepgemm]
- [FlashInfer sorting-free sampling][flashinfer-sampling]
- [FlashSampling (fused sampling, no logits materialization)][flashsampling]

[vllm-lp]: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/logits_processor.py
[vllm-vpe]: https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/vocab_parallel_embedding/
[lmsys-h20]: https://www.lmsys.org/blog/2025-09-26-sglang-ant-group/
[deepgemm]: https://github.com/deepseek-ai/DeepGEMM
[flashinfer-sampling]: https://flashinfer.ai/2025/03/10/sampling.html
[flashsampling]: https://arxiv.org/html/2603.15854
