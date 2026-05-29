# Operator library — SOTA-kernel replacement gap

Date: 2026-05-29
Companion to the kernel registry ([`../reviews/kernel-registry.md`](../reviews/kernel-registry.md))
and the Govern gate ([`../plans/gpu-dispatch-governance.md`](../plans/gpu-dispatch-governance.md)
§Phase 4). Question per op family: **what does the SOTA library hand ARLE for
free that ARLE currently hand-rolls, and is replacement licensed?**

**Evidence discipline (CLAUDE.md §0).** A launch-count / source-survey argument
is *hypothesis*, not evidence. Every candidate carries a verdict from this set:

| Verdict | Meaning |
|---|---|
| **wire-it** | SOTA is the canonical path, already proven elsewhere in-tree or by direct port; no further license needed |
| **needs-paired-A/B** | plausible win, but only a same-binary / same-shape paired bench (or `ncu`/CUDA-event profile under the runtime's sync framing) licenses landing |
| **killed-already** | tried and reverted with hard data — cite the errors entry |
| **roofline-share-too-low** | the op's % of per-request wall-clock at the binding SLO shape is below the ~5% kernel-work floor; not worth a licensed bench |

The corpus lesson that bounds everything below: **bandwidth-bound and
tensor-core-amortized ops did not benefit from launch-fusion or from a
"fewer launches" rewrite.** Three kills anchor it —
[swiglu-fused-kill](../experience/errors/2026-05-21-arle-cuda-opd-swiglu-fused-kill.md)
(fusing silu+mul moved OPD step +0.006%, bandwidth-bound),
[fp8-kv-pair-quantize](../experience/errors/2026-05-12-fp8-kv-pair-quantize-fusion-no-license.md)
(K+V two-call fusion showed no two-launch penalty under runtime sync framing),
[r4-hybrid-dispatch](../experience/errors/2026-05-08-r4-hybrid-dispatch-killed-batch4-decode-regression.md)
(small-batch BF16-native GEMV to dodge Marlin's 5 launches → +60.7% ITL because
the launches *amortize tensor-core compute*). The throughline: **launch count is
not the cost; bytes moved and tensor-core occupancy are.** A SOTA replacement is
only worth it when it changes one of those, not just the launch tally.

---

## Per-op-family gap

### Linear / GEMM

| Op | ARLE today | SOTA library provides | Roofline share | Verdict |
|---|---|---|---|---|
| BF16 GEMM/GEMV | cuBLASLt (`Bf16*Gemm`) + hand-rolled M<16 GEMV | cuBLASLt (same) | bandwidth-bound | **wire-it (already is)** — ARLE routes cuBLASLt with tensor-op math + algo cache, identical to TRT-LLM/vLLM. The M<16 hand-roll is correct (cuBLAS sub-optimal there). No gap. |
| W4 GEMM | Marlin (IST-DASLab port) | Marlin (vLLM/SGLang ship identical file) | tensor-core | **wire-it (already is)**. Small-batch BF16-native fallback is **killed-already** (r4-hybrid +60.7% ITL). |
| DSv4 FP8 grouped GEMM, SM_90 | DeepGEMM v3 wrapper | DeepGEMM (this *is* it) | tensor-core | **wire-it (already is)** — `cute::UMMA`+TMA, no faster FP8 grouped GEMM exists. The only past bug was *reachability* (built+cached, never branched-to, `errors/2026-05-27-b33-deepgemm-not-wired-on-native-deepep.md`), which the Govern Observe gate closes — not an optimality gap. |
| **DSv4 FP8/FP4 batch GEMV, decode B≤16 (GAP-A)** | hand-rolled **scalar FP32-FMA**, no tensor cores (`quantized_gemv.cu`) | CUTLASS `mma.m16n8k16`; FlashInfer `bgmv_*`; SGLang `fp8_blockwise_scaled_grouped_mm` | **≈16% of decode GPU time** (2026-05-14 trace); scalar at `frac_peak=0.04` on H20 — 96% HBM3 unused, compute/issue-bound (`wins/2026-05-28-cuda-gap-a-micro-license-pass-h20.md`) | **needs-paired-A/B** — MMA port (`quantized_gemv_mma.cu`) is **written, wired behind env, parity-tested**; pod TPOT A/B (≥4% decode gate) is the only thing missing. This is the single most-licensed candidate in the tree: roofline share is high *and* the license-or-kill already PASSED at Phase 2. |
| **DSv4 FP8 grouped GEMM, SM_89 fallback (GAP-D)** | hand-rolled scalar (same pattern as GAP-A + expert grid-Z) | SGLang grouped CUTLASS path | on DSv4-prefill SLO-kill path, **but SM_89 is not a DSv4 deploy target** | **needs-paired-A/B, deferred** — mechanical once GAP-A lands; medium priority because SM_89 doesn't serve DSv4 in production (SM_90 uses DeepGEMM). |
| INT W{2,4,8}A16 batch GEMV | hand-rolled CUDA-core GEMV | Marlin (already default ≥batch2) | — | **killed-already** — preferring the GEMV at batch=4 to save launches gave +60.7% ITL (r4-hybrid). Marlin tensor-core dominates. |
| GGUF Q{3-6}_K | hand-rolled dequant + cuBLASLt | none faster in OSS | bandwidth-bound dequant | **roofline-share-too-low** to pursue a rewrite; llama.cpp/vLLM use the same parallel-bitmask extract. |
| TurboQuant rotated W2/W4 | hand-rolled FWHT + GEMV | research-only (QuaRot/TurboQuant), no vLLM/SGLang path | decode-path, small | **roofline-share-too-low**. ARLE is already at the research frontier here. |

### Attention

| Op | ARLE today | SOTA library provides | Roofline share | Verdict |
|---|---|---|---|---|
| Paged BF16 prefill / decode | TileLang AOT (`cute::SM80::mma`) | FlashInfer / FlashAttention-v3 | TileLang **beats SGLang ITL +8.6%→+21.8%** (c=4→16) on L4 (head-to-head win) | **wire-it (already is)** — ARLE is at/above SOTA on SM_89 for the BF16 paths. Don't touch. Caveat is a *reachability* hole: TileLang head-configs hard-fail at runtime on un-baked `(qo,kv)` shapes (`attention.rs:1477`), only `(8,2),(16,2),(16,4)` compiled — fixed by the Declare gate's SKU capability check, not by a kernel swap. |
| **FP8 fused-decode attention softmax (GAP-C)** | hand-rolled **scalar warp-shuffle softmax**, no mma; FP8 path was sync-load until `ab850f7a` | FA-v3 (`flash_attn_with_kvcache`, FP8 KV + WGMMA/mma.m16n8k16); FlashInfer `BatchDecodeWithPagedKVCache` (cp.async + cuBLASDx MMA softmax) | Qwen3.5 high-conc decode; DSv4 hybrid-attn ≈10.3% GPU time (different model, same authoring style) | **needs-paired-A/B (two halves).** cp.async-cheap-half **landed** (FP8 now mirrors INT8 double-buffer, ~111 LoC, `wins/2026-05-28-cuda-gap-c-cheap-fp8-cpasync.md`) — pod TPOT A/B `pending-remote` (c=16 ≥1% gate). MMA-softmax-medium-half **not started**; may reuse the FlashMLA-decode template once it lands — sequence after, don't fork the work. |
| INT8 per-channel-K decode | hand-rolled + cp.async (since `8afecffe`) | FlashInfer mma softmax | — | **needs-paired-A/B (low)** — folded into GAP-C-medium scope; cp.async half already present. |
| DSv4 MLA decode | dispatch shim → TileLang / FlashMLA | FlashMLA (this *is* it) | — | **wire-it (already is)** — FlashMLA is SOTA; owned by a separate subagent, out of this gap's scope. |
| DSv4 SWA + compressor hybrid prep | hand-rolled, partially fused (A2.0/A2.1 landed) | SGLang V4 `flashmla_hybrid_attention` (SWA+compressed in one call) | launch-churn axis, not arithmetic | **needs-paired-A/B, blocked** — wait for FlashMLA-decode subagent; SWA may collapse into the fused-hybrid call. Re-audit after. |
| Single-token GQA fused attn (`fused_attention.cu`) | shared-mem online-softmax, HEAD_DIM=128 hardcoded, no mma | FA-v3/FlashInfer head-dim template + mma | low on L4, matters on H100 | **roofline-share-too-low** on the L4 target; revisit only if H100 becomes a binding SLO target. |

### KV quantization / movement

| Op | ARLE today | SOTA library provides | Roofline share | Verdict |
|---|---|---|---|---|
| KV quantize (FP8/INT8/INT4 paged write) | hand-rolled warp-per-token-per-head | FlashInfer/vLLM same per-channel-K + per-token-V recipe | bandwidth-bound | **wire-it (already is)**. K/V two-call fusion is **killed-already** — paired component bench showed no two-launch penalty under runtime sync (`fp8-kv-pair-quantize`). The canonical anti-launch-count case. |
| KV append / scatter / tier-transfer | hand-rolled (transfer.cu is a *direct SGLang port*) | SGLang / FlashInfer (same patterns) | bandwidth-bound, at HW limit | **wire-it (already is)**. No gap. |

### MoE routing / prep

| Op | ARLE today | SOTA library provides | Roofline share | Verdict |
|---|---|---|---|---|
| **MoE top-k route (GAP-B)** | hand-rolled, **`threadIdx.x==0` only — 255/256 idle**, serial expert scan + selection-sort | SGLang `topk_softmax.cu` (1 warp/token, warp-parallel softmax + bitonic/tournament top-k); FlashInfer `top_k_renorm_probs` (data-parallel); vLLM Triton tree-reduction | **≈4.4% of decode GPU time** (2026-05-14 trace) | **needs-paired-A/B at production shape.** The warp-parallel rewrite is *unwritten*. ⚠ **Do not mistake the prior route kills as killing this.** `errors/2026-05-16-p3-3-dsv4-route-launch-shape-kill.md` (block-size tune) and `…-persistent-kernel-deferred.md` tested a **different axis** (CTA size + persistent kernel) on the **local 1B E16/top2 microbench** — not warp-parallelism on the E256/top8 production shape. The "clearly wrong, one-day win" framing from the audit stands until benched. |
| EP/TP index mask | hand-rolled + TileLang sister | DeepSeek TileKernels (same algorithm) | — | **wire-it (already is)**. |

### Elementwise / norm / recurrent / sampling

| Op | ARLE today | SOTA library provides | Roofline share | Verdict |
|---|---|---|---|---|
| RMSNorm + residual fuse | hand-rolled bf16x4 + FP32-accum + HF-exact 2nd pass | vLLM `layernorm_kernels` / FlashInfer `rmsnorm.cuh` (same) | bandwidth-bound | **wire-it (already is)**. |
| SiLU·up activation | hand-rolled bf16x4 | vLLM `activation_kernels` (same) | bandwidth-bound | **wire-it (already is)**. Fusion **killed-already** (swiglu, +0.006% step). |
| K-quant MLP fallback (`fused_mlp.cu`) | hand-rolled dot-product fallback | cuBLASLt (faster at large intermediate; ARLE already bypasses to it) | mostly dead in DSv4 | **roofline-share-too-low**; removal candidate, not a SOTA-swap candidate. |
| Gated delta rule / conv1d (Qwen3.5 linear-attn) | hand-rolled + TileLang | **no SOTA baseline exists** (Qwen3.5-specific) | — | **wire-it (already is)** — ARLE-original, well-tuned (six-principles fixed cross-slice re-read). Nothing to replace. |
| Sampling (top-k+top-p) | hand-rolled single-block, serial across batch | FlashInfer `sampling.cu` multi-block argmax | ~5% decode at c=16 vocab=152k *(hypothesis, unmeasured)* | **needs-paired-A/B (low)** — the 5% share is itself a hypothesis; measure the share before licensing the FlashInfer swap. |

### Metal (MLX)

ARLE hand-rolls **nothing** on the Metal linear/attention/norm hot path — it
dispatches to MLX primitives (`quantized_matmul`, `gather_qmm`,
`fast::scaled_dot_product_attention`, `fast::rope`, `fast::rms_norm`). MLX *is*
the SOTA library for Apple Silicon. The kernel chosen inside `mx::eval` is a
black box below the FFI line, with no introspection hook (governance non-goal
§"Not building MLX kernel introspection"; `feedback_mlx_async_eval_is_caller_thread`).
**No SOTA-replacement gap on Metal** — the only Metal levers are
primitive-selection / encode-batching, which are a separate workstream from
kernel-internals replacement.

---

## Prioritized shortlist

### Worth a licensed bench (ranked by roofline-share × confidence)

| Rank | Adoption | Roofline share | Confidence | Why ranked here |
|---|---|---|---|---|
| **1** | **GAP-A — `mma.m16n8k16` quantized GEMV** (`Dsv4Fp8/Fp4 *Gemv`) | ≈16% decode | **high** | Highest share *and* the cheapest remaining step: kernel written, dispatch wired, parity PASS, Phase-2 license-or-kill PASS (scalar `frac_peak=0.04`). Only the pod TPOT A/B remains. Single best ROI. |
| **2** | **GAP-B — warp-parallel MoE route** (`dsv4_route_kernel`) | ≈4.4% decode | **medium-high** | "255/256 threads idle" is unambiguous waste; SGLang/FlashInfer give the exact template. Confidence is below GAP-A only because the rewrite is unwritten and must be benched on the **production E256/top8** shape, not the 1B microbench the prior kills used. |
| **3** | **GAP-C-cheap — FP8 decode `cp.async`** (`decode_attention_fp8_…`) | ~2-3% Qwen3.5 quant decode *(estimate)* | **medium** | Already landed; pure mirror of the proven INT8 sibling, so low risk. Ranked third only because the share estimate is smaller and unmeasured. The pod A/B is the gate. |

After these three: **GAP-D** (mechanical port of GAP-A to grouped SM_89) follows
GAP-A automatically; **GAP-C-medium** (MMA softmax) follows GAP-C-cheap and
should reuse the FlashMLA-decode template rather than fork it.

### Explicitly NOT worth it (and why)

| Non-adoption | Why killed / not worth it |
|---|---|
| Small-batch BF16-native GEMV to dodge Marlin launches | **killed-already** +60.7% ITL — Marlin's launches amortize tensor-core compute (`r4-hybrid`). |
| K+V KV-quantize two-call fusion | **killed-already** — no two-launch penalty under runtime sync framing (`fp8-kv-pair-quantize`). |
| SiLU·up SwiGLU fusion | **killed-already** — bandwidth-bound, +0.006% step (`swiglu-fused-kill`). |
| Route block-size / persistent-kernel tune | **killed/deferred** — different axis from GAP-B; KILL on the 1B microbench (`p3-3-route-launch-shape-kill`, `…-persistent-kernel-deferred`). |
| `fused_attention.cu` FA-v3 head-dim template | **roofline-share-too-low** on L4; matters only if H100 becomes a binding SLO. |
| GGUF K-quant / TurboQuant rewrites | **roofline-share-too-low**; ARLE already matches OSS (GGUF) or the research frontier (TurboQuant). |
| `fused_mlp.cu` rewrite | **roofline-share-too-low** — mostly dead, bypassed by cuBLASLt; remove rather than optimize. |
| MHC / sampling micro-rewrites | **needs-paired-A/B (low)** — sampling's ~5% share is itself a hypothesis; measure share first. |
| MLX kernel-internals replacement (Metal) | No gap — MLX is the SOTA library; kernel choice is below the FFI black box. |

---

## Self-check (CLAUDE.md §0 SOLID gate)

- **Evidence vs hypothesis.** Roofline % are SOLID (2026-05-14 nsys trace,
  H20 micro `frac_peak`). The "~2-3% GAP-C" and "~5% sampling" shares are
  marked *estimate* / *hypothesis* — they gate a paired-A/B, not a land.
- **Framing.** All shares are per-decode-window; per CLAUDE.md §0 the
  per-request wall-clock is ground truth, so each share **caps** the achievable
  wall-clock win — no candidate is claimed to dominate beyond its kernel budget.
- **Confounder isolation.** Each candidate is a single-kernel axis; the
  shortlist ranks by share × confidence, and every "not worth it" cites the
  specific kill or the roofline floor.
- **Gap deferred.** No current-main nsys re-trace was run (out of docs-only
  scope); the 2026-05-14 binding-constraints table is the canonical source and
  the registry's roofline column inherits its caveat. Caller-count
  instrumentation (governance Phase-1 counters) is the right next step before
  landing any candidate.

## Cross-refs

- Kernel registry (the data): [`../reviews/kernel-registry.md`](../reviews/kernel-registry.md)
- Govern gate plan: [`../plans/gpu-dispatch-governance.md`](../plans/gpu-dispatch-governance.md)
- Kernel-vs-SOTA audit (richest source): [`2026-05-28-arle-kernel-vs-sota-audit.md`](2026-05-28-arle-kernel-vs-sota-audit.md)
- GAP-A license-or-kill: [`../experience/wins/2026-05-28-cuda-gap-a-micro-license-pass-h20.md`](../experience/wins/2026-05-28-cuda-gap-a-micro-license-pass-h20.md)
- GAP-A MMA kernel: [`../experience/wins/2026-05-28-gap-a-phase3-mma-kernel-partial.md`](../experience/wins/2026-05-28-gap-a-phase3-mma-kernel-partial.md)
- GAP-C cp.async: [`../experience/wins/2026-05-28-cuda-gap-c-cheap-fp8-cpasync.md`](../experience/wins/2026-05-28-cuda-gap-c-cheap-fp8-cpasync.md)
- Anti-launch-count kills: [swiglu](../experience/errors/2026-05-21-arle-cuda-opd-swiglu-fused-kill.md) · [fp8-kv-pair](../experience/errors/2026-05-12-fp8-kv-pair-quantize-fusion-no-license.md) · [r4-hybrid](../experience/errors/2026-05-08-r4-hybrid-dispatch-killed-batch4-decode-regression.md)
