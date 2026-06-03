# SGLang operator selection + overlap — DSv4 (MLA+FP8 MoE) & Qwen3-MoE, vs ARLE

Source: SGLang HEAD `8980eb82` (2026-06-03), source-read only (hypothesis-grade;
license each with a matched same-binary A/B on the SLO shape before any flip).
Feeds the perf phase (task #16) + the operator-selection mandate.

## FP8 compute precision (settled)
`f8f8bf16` = **FP8×FP8 tensor-core inputs, FP32 accumulate, BF16 output**. With DeepGEMM,
weights stay FP8 on tensor cores — **no upcast to BF16**. The BF16-dequant
(`bmm(x.to(bf16), w.to(bf16)*scale)`) is ONLY the no-DeepGEMM / cublas-12.9-bug fallback.
→ ARLE FP8 path = DeepGEMM `f8f8bf16` (masked decode / contiguous prefill, 128×128 block scale), not dequant.

## ARLE already at parity (kernels) — no action
- Absorbed-MLA latent attention (W_UK/W_UV absorption) + FlashMLA FP8 MODEL1 decode (ARLE vendors the same upstream 584B/token layout, byte-identical).
- Fused route kernel: `dsv4_route.cu` (sigmoid+bias+grouped-topk+routed_scaling) ≈ SGLang `moe_fused_gate`.
- 128-block FP8 scaling (`GRAN_K=128`). Full-decode CUDA-graph + eager-prefill collectives.

## SGLang ahead (the perf-phase targets — all OVERLAP/scheduling, not kernel quality)
| # | Gap | SGLang ref | ARLE state |
|---|---|---|---|
| 1 [Highest] | **Decode low-latency DeepEP dispatch** (vs normal all-to-all) | `deepep.py:191-209,785` LL dispatch/combine | sidecar has no LL/normal split — biggest decode-ITL lever |
| 2 [High] | **SBO**: combine↔down-GEMM + shared-expert↔dispatch two-stream | `single_batch_overlap.py:97-124` (DeepGEMM signal + partitioned SMs) | shared-experts run inline/serial (`mlp.rs:1596-1748`) — clearest overlap gap |
| 3 [High] | **TBO** two-micro-batch overlap | `batch_overlap/operations_strategy.py:94-150` | collectives eager (`forward.rs:715`) — largest structural gap |
| 4 [Med] | Default to native DeepGEMM; justify custom M-tile=32 by A/B | routes 100% through DeepGEMM | ARLE keeps custom `dsv4_grouped_gemm.cu` — must beat native masked or retire |
| 5 [Med] | Fuse SiLU+mul+requant between gate/up & down GEMM | `silu_and_mul_masked_post_quant` | verify ARLE isn't separate launches |
| 6 [Low] | Prefill un-absorbed MHA at prefix==0 | `attention_backend_handler.py:89-104` | TTFT trap if ARLE always absorbs in prefill |
| 7 [Low] | AllReduce+residual+RMSNorm fusion (TP decode) | `flashinfer_allreduce_residual_rmsnorm` | removes a hidden-size r/w per TP-decode layer |

## Backend selection
- MLA: prefill→MHA/un-absorbed (prefix==0) or MLA-absorbed; decode→MLA-absorbed. Hopper backends: FlashMLA / FlashInfer-MLA / FA3 / TRT-LLM-MLA / dsv4-dsa-sparse. Per-prefill/decode backend choice (`deepseek_v2.py:1673`).
- Qwen3-MoE: plain GQA (q_norm/k_norm + fused_qk_norm_rope), plain-softmax topk (no grouped/noaux) — but **same DeepEP dispatch + DeepGEMM grouped FP8 GEMM as DeepSeek**, so all DSv4 MoE/overlap work transfers to Qwen3.5-MoE. (Do NOT use upstream `qwen3_5.py` (GatedDeltaNet) as ARLE parity ref; use `qwen3_moe.py` for the MoE operators.)

## Implication for the rewrite port
Carry the parity kernels (shared `cuda-kernels`). The 3 overlap items (LL-dispatch, SBO, TBO) are gaps in BOTH legacy and the rewrite → the perf phase adds genuine value beyond legacy. Functional port first (correctness), then #16 layers these overlaps, A/B-licensed on the SLO shape.
