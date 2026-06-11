# SGLang Qwen3-Next P/D 对照 — ARLE qwen35 的重点定位

**Date:** 2026-06-11. **Sources:** sgl-project/sglang @ `2a51479a` (shallow,
/tmp scan); BBuf AI-Infra-Auto-Driven-SKILLS @ `d45b5cc1` (qwen3-next 61 PR +
qwen35 52 PR cards, sglang + vllm). Survey-grade per skill rules — every
"effect" below is hypothesis until ARLE A/B; structural facts are
source-cited.

## The verdict (重点)

ARLE 的差距主轴不是单算子质量——是 **decode 的执行单位**。SGLang 每个 forward
跑整个 running batch(bucket [1,2,4,8,12,…,512],FULL CUDA graph 每桶一张,
含 logits processor;sampling 在图外),mamba/GDN 状态走 **池 + 槽索引**
(per-bucket 静态 index buffer,replay 前只拷索引)。ARLE qwen35 是
single-row-per-tick,c≥2 直接杀引擎——并发下聚合吞吐不随 c 增长。其余
kernel 差距(FA3、FLA chunked、fused GDN decode)都排在这个结构差之后。

## P/D 路径对照表(要点)

| 维度 | SGLang qwen3-next | ARLE qwen35(今日) | 类别 |
|---|---|---|---|
| Decode 批 | running batch 整批 + bucketed FULL graph + 槽索引 replay | 单行/tick;c≥2 引擎死亡 | **结构 #1** |
| GDN decode | 1 个 fused packed Triton kernel/层(split+L2norm+gating+delta 全融合,`fused_recurrent_..._packed_decode`;vllm #35777 同型) | in_proj×4 GEMM + conv + GDR + gated-norm ≈ 7 launch/层 | 融合(跟随批量化) |
| 状态管理 | `MambaPool` conv bf16 / ssm fp32 池,槽分配器;chunked prefill 状态**按槽链接**(chunk k+1 读 chunk k 写的同一槽,零 host 拷贝) | 每 slot 独立 buffer,地址烘进 graph key | 池化设计是批量 graph 的前提 |
| 全注意力 | FA3 `flash_attn_with_kvcache`(paged)decode / `flash_attn_varlen` prefill;partial rope 0.25;sigmoid 输出门 | 手写 nonpaged 每 (head,token) 一 block | 既定 #4(TileLang paged HD256) |
| GDN prefill | FLA `chunk_gated_delta_rule` varlen(chunk 64,intra fused KKᵀ/solve,initial_state 按槽 INPLACE) | 串行 recurrent 32-block | 既定 #3(FlashQLA) |
| MoE | gate GEMM + TopK kernel(**无 router 融合**)+ triton fused_moe;shared expert 在 alt_stream 与 routed 重叠 | 设备路由 ✓ + DeepGEMM grouped ✓(needle −74.5% 已 license);shared expert 串行 | **ARLE 此项已达/超 parity**;可学 alt_stream 重叠 |
| Prefill chunk | 8192(H100 级默认) | 2048(刚从 64 修上来) | 待 A/B 8192 |
| Hybrid prefix cache | 默认 no_buffer = **不做**(extra_buffer 模式才按 256-token 界快照) | 禁用 | **parity**——上游默认也不做,我们的禁用是对的 |
| Sampling | 图外,flashinfer 后端 | 图外 argmax | parity |
| 双流重叠 | decode capture 模式下 qkvz/ba 投影、shared expert 走 alt_stream | 无 | 跟随批量化 |

## Kept(映射到 ARLE 行动)

1. **Batched decode + bucketed graph + state-slot 索引化**(结构主轴;ARLE
   树内已有 DSv4 batched 模式 + monolith packed-batch 前科 + 死的
   `gdr_decode_batch.cu`/`conv1d_decode_batch.cu` FFI)。
2. GDN decode 全融合单 kernel(vllm #35777 形态;sglang #10466 的 L2-norm
   eps-inside-sqrt 数值锚;vllm #31722 BV tile 32 扫参提示——ARLE AOT 离线扫
   后钉死,不带 autotune)。
3. FA3 类 paged 全注意力(既定 #4)。
4. FLA chunked GDN prefill(既定 #3),含按槽 initial_state 链接设计。
5. shared expert / 投影双流重叠(批量化后评估)。
6. prefill chunk 8192 A/B。

## Killed

- Router-fusion 假设:SGLang 也没融(gate GEMM 与 topk 分离)——我们的设备路由已同型。
- Hybrid prefix cache 焦虑:上游默认同样关闭;ARLE 的 kind 门禁是正确姿势。
- flashinfer_trtllm MoE runner:SM100+fp8/fp4 专属,与 H20/bf16 无关。
- torch.compile tc_piecewise prefill graph:依赖 torch 栈,ARLE 无 Python 热路径,不适用。

## 已由本战役达成的对照项

设备路由(SGLang 同型)✓;MoE grouped GEMM(DeepGEMM ≥ triton fused_moe
类)✓;chunk 64→2048 ✓;workspace 零分配(SGLang 靠 graph 池化,等效)✓;
whole-step graph 机器(等批量化后变 bucketed)✓。
