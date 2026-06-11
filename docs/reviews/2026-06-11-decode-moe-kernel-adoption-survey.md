# Decode-band MoE expert GEMM:业界可抄对象普查(per "直接抄业界最好的" 指令)

**Date:** 2026-06-11. **Op:** BF16 grouped expert GEMM, R=8..64 routed rows,
experts gate/up [512,2048] + down [2048,512], sm_90a H20,
weight-read-bound(E_active × 6.29 MB → 12.6 µs/层理想)。
**Sources(shallow clones,commit 已钉):** vLLM @7852e50e,
SGLang @22c7285a,TRT-LLM @84b349fa,cutlass @1fc71b3e,
FlashInfer @28406af5,llama.cpp @18ef86ec,DeepGEMM @714dd1a4(在树),
kernel-set(本地)。

## Verdict

**业界最好的可抄对象已经在树里,且已被实测**:DeepGEMM bf16 MGroupedMasked
(DeepSeek 生产 decode 同款)decode −8%→持平 —— 输因是 ARLE 侧编排
(masked 全带 [G,128,K] 布局下 silu_mul 扫 ~134 MB×3 填充缓冲 ≈ 8× 理想字节),
kernel 调度本身健康(空组零 tile、TMA exactly-once;但 BLOCK_M 候选仅
{64,128},R≤8 时每活跃 expert 烧 ≥64 行 tile)。

| 候选 | decode 带核? | 提取面 | 结论 |
|---|---|---|---|
| DeepGEMM masked | ✓(产线同款) | **0 LOC,已接** | LIFT(已完成);修 silu 边带账后可重赛 |
| llama.cpp mmvf/mmf+mmid | ✓ **业界唯一 GEMV 类先例**(warp-per-row、向量化、fused gate-silu、per-expert token 批) | ~2.1k LOC,MIT,torch-free | PORT 级参照——**与 in-house 候选同算法族,验证而非替代** |
| TRT-LLM MoE GEMM | tensor-core 类(同 DG) | 30-50k LOC | SKIP(被 DG 严格支配) |
| vLLM | **无 bf16 CUDA expert GEMM**(Triton,BLOCK_M=16 地板) | — | SKIP;W4 后续记 marlin_moe_wna16 |
| SGLang sgl-kernel | 无(fp8/w4a8/cutlass;bf16=Triton;DeepSeek decode→DeepGEMM) | — | SKIP |
| FlashInfer | group_gemv.cuh 是空 TODO;masked grouped=Blackwell CuteDSL | — | SKIP |
| cutlass examples | 通用 grouped,masked 调度要自己写=重新推导 DG | — | SKIP |
| kernel-set | 路由 rank-1 = DeepGEMM masked(heuristic,sm90 未实测 small-M) | — | 印证 #1 |

## A/B 计划(进行中)

- **A**(候选,默认 ON):in-house decode kernel(exactly-once 权重读,
  fused swiglu,`ARLE_QWEN35_MOE_DECODE_KERNEL=1`)
- **B**(基线):旧手写(=0;40.86 tok/s 参考)
- **C**(可选重赛):DG-masked + count-aware/banded silu 修复后;A≈C 则取 A
  (无 JIT 依赖、少 1 launch、sm_70 可移植)
- 门:needle + smoke×3 + c=1/2/4 sweep(c=4 平台是 claim 的一部分)+
  nsys 每层 MoE µs 机制核对(25-60 µs 预测)。

## Rule

- "先抄业界最好的"的前提是普查证明它存在:本 op 在 sm90 bf16 小 M 带
  业界没有未被采纳的更优实现——主线(Triton BLOCK_M=16)甚至不做
  exactly-once;GEMV 类先例(llama.cpp)与自研同族。普查本身就是结论。
- LIFT 过的 kernel 输了 ≠ kernel 不行:先查自己的边带账
  (这里是 padded-band silu)再下采纳结论。
