# 2026-04-02 · CUDA 链路优化复盘：哪些做了但没走通

## Context

本次 review 目标不是列“还能做什么”，而是找出仓库里已经出现过实现、开关、设计或计划，但没有真正进入当前 CUDA 热路径的优化项。

重点检查范围：

- Qwen3 / Qwen3.5 decode 与 batched decode 主链路
- FlashInfer metadata / plan / graph 交互
- recurrent 路径的 batching 与 kernel 复用
- repo 内已有性能记录、计划文档与当前实现是否一致

结论先说：当前最关键的断点集中在 `Qwen3.5 batched decode`。单请求 decode 有 CUDA Graph，但一进入并发热路径就回退成 eager；recurrent 相关也只做到 batched GEMM，没有做到 batched recurrent kernel。

## Findings

### 1. Qwen3.5 的 batched decode 没有吃到 CUDA Graph

**影响**: 高

`--cuda-graph` 仍然是服务层公开配置，描述为 decode path 开关：

- `infer/src/main.rs`

但 Qwen3.5 真正的 batched decode 主路径在代码里直接写死为不用 graph：

- `infer/src/model/qwen35/batch_decode.rs:252`

当前实际情况是：

- 单 token / 单请求路径会走 graph capture/replay
  - `infer/src/model/qwen35/forward.rs:122`
  - `infer/src/model/qwen35/prefill.rs:302`
- batched decode 一旦进入 scheduler 热路径，就完全走 eager
  - `infer/src/model/qwen35/forward.rs:166`
  - `infer/src/model/qwen35/batch_decode.rs:252`

这不是“没有 CUDA Graph 实现”，而是“实现只覆盖了单请求路径，没有覆盖并发热路径”。

### 2. Qwen3.5 的 recurrent batching 只做了一半

**影响**: 高

Qwen3.5 batched decode 已经把线性投影做成 batched GEMM：

- `infer/src/model/qwen35/batch_decode.rs:455`

但 linear attention 的 recurrent 部分仍然是按请求循环：

- 从 batch buffer 提取单条 `qkv / b / a`
- 对每个请求单独执行 `conv1d`
- 对每个请求单独执行 `gated_delta_rule_decode`
- 再把结果写回 batch buffer

对应代码：

- `infer/src/model/qwen35/batch_decode.rs:466`
- `infer/src/model/qwen35/batch_decode.rs:471`
- `infer/src/model/qwen35/batch_decode.rs:495`
- `infer/src/model/qwen35/batch_decode.rs:505`
- `infer/src/model/qwen35/batch_decode.rs:520`

repo 自己的性能记录已经把这条链路定性为 Qwen3.5 C>1 ITL gap 主因：

- `docs/experience/wins/2026-04-01-qwen35-scheduler-support.md:48`

而 parity 计划里也明确把下面两项列为关键缺口：

- `conv1d_decode_batch`
- `gated_delta_rule_decode_batch`

见：

- `docs/plans/qwen35-sglang-parity.md:59`
- `docs/plans/qwen35-sglang-parity.md:70`

也就是说，这条优化方向不是没识别，而是已经进入计划，但还没真正接通。

### 3. Qwen3.5 batched decode 因 recurrent state 不稳定，连 graph invalidate 逻辑都没有意义

**影响**: 高

`FlashInferDecodeMetadata::update()` 会在 `kv_indices` 重新分配时返回 `reallocated=true`，调用方本该用它去失效 graph cache：

- `infer/src/flashinfer_metadata.rs:102`
- `infer/src/flashinfer_metadata.rs:169`

Qwen3 这条链路确实这样做了：

- `infer/src/model/qwen3/batch_decode.rs:199`
- `infer/src/model/qwen3/batch_decode.rs:202`

但 Qwen3.5 这里只能把返回值吃掉：

- `infer/src/model/qwen35/batch_decode.rs:226`
- `infer/src/model/qwen35/batch_decode.rs:229`

原因不是这里漏了 invalidate，而是整个 batched graph 根本没接上。这进一步说明，Qwen3.5 在并发热路径上仍停留在 graph 之前的状态。

### 4. Qwen3 的 tensor-core decode 已接好 plan/run，但被硬编码关闭

**影响**: 中

Qwen3 batched decode 中，tensor-core decode 的实现并不是 stub：

- metadata 里有 `tc_plan`
  - `infer/src/flashinfer_metadata.rs:257`
- attention 里有 `flashinfer_tc_run_layer`
  - `infer/src/ops/attention.rs:844`

但主路径里两处都硬编码：

- `let use_tc_decode = false`

位置：

- `infer/src/model/qwen3/batch_decode.rs:209`
- `infer/src/model/qwen3/batch_decode.rs:412`

注释给出的原因也很明确：A100 上这条路径不比保留 CUDA Graph 更快，因为瓶颈不在 attention，而在 GEMV。

所以这条优化不是“没实现”，而是“已实现，但当前默认热路径明确不走”。如果后面切到更长上下文、不同 GPU 或不同 head/group 配置，这条路径仍然是一个尚未产品化的潜在优化分支。

### 5. FlashInfer metadata 的“增量更新”只减少了 CPU rebuild，没有减少 H2D

**影响**: 中

`FlashInferDecodeMetadata::update()` 在 steady-state decode 下，确实不再 O(total_tokens) 重建全部 indices：

- `infer/src/flashinfer_metadata.rs:136`

但它最后仍然执行整段 `indices_scratch -> kv_indices` 的 H2D：

- `infer/src/flashinfer_metadata.rs:157`

这意味着当前收益主要是：

- 少了 CPU 端 rebuild 成本

没有做到：

- partial H2D
- GPU-side metadata update

而仓库里当时的历史实现说明也已经把 “GPU-side metadata” 单列为剩余优化方向。

所以这条链路属于“思路已经落到一半，但没走完”。

### 6. Qwen3.5 当前单 token decode 退回了通用流水线，早期 fused decode 形态没有保住

**影响**: 中

仓库里仍然有成熟的 fused decode kernel 入口：

- `fused_add_rms_norm_offset_into`
  - `infer/src/ops/norm.rs:187`
- `fused_mlp_into`
  - `infer/src/ops/linear.rs:41`

Qwen3 单 token decode 仍然在用 fused path：

- `infer/src/model/qwen3/decode.rs:68`
- `infer/src/model/qwen3/decode.rs:146`
- `infer/src/model/qwen3/decode.rs:155`

但 Qwen3.5 当前单 token 路径已经退成了 prefill-as-decode 的通用流水线，仍保留 CUDA Graph，却没有继续使用 fused residual/norm/MLP：

- `infer/src/model/qwen35/prefill.rs:341`
- `infer/src/model/qwen35/prefill.rs:398`
- `infer/src/model/qwen35/prefill.rs:444`
- `infer/src/model/qwen35/prefill.rs:455`

这和项目文档中的较早 decode 设计并不完全一致。文档里曾描述过：

- `fused_attention_hd256_decode`
- `fused_add_rms_norm_offset`
- `fused_mlp`

见：

- `infer/docs/projects/qwen35-4b-optimization.md:132`
- `infer/docs/projects/qwen35-4b-optimization.md:151`

但同一篇文档在后面的“current”剖析里又说明现在的实现已经回到了分开的 GEMV + silu_mul：

- `infer/docs/projects/qwen35-4b-optimization.md:164`

这说明 fused decode 不是完全不存在，而是曾经存在过目标形态，后续为了统一实现或正确性/维护性退回了更通用的路径。

### 7. Qwen3 的 batched fused add+norm 已接通，Qwen3.5 batched path 还没接

**影响**: 中

Qwen3 batched decode 已经接上了 `fused_add_rms_norm_batch_into`：

- `infer/src/model/qwen3/batch_decode.rs:456`

Qwen3.5 batched decode 仍然是分开的：

- `add_batch_into`
- `rms_norm_batch_offset_into`

位置：

- `infer/src/model/qwen35/batch_decode.rs:567`
- `infer/src/model/qwen35/batch_decode.rs:570`
- `infer/src/model/qwen35/batch_decode.rs:600`

这和 repo 内对剩余差距的判断一致：早期实现说明里已经把
`FusedAddRMSNorm` 列成剩余关键优化。

所以这里也属于“优化方向已经有实现样板，也被性能分析明确点名，但 Qwen3.5 并发热路径没有吃到”。

### 8. Qwen3.5 的吞吐扩展建议已经写进计划，但默认 slots 仍是 4

**影响**: 中

Qwen3.5 parity 计划里把“默认 slots 从 4 提到 8”列为 Step 1：

- `docs/plans/qwen35-sglang-parity.md:52`

但服务 CLI 默认还是：

- `num_slots = 4`
  - `infer/src/main.rs:35`

这不是 kernel 级问题，但它直接限制了 CUDA 链路最终能体现出来的并发吞吐，尤其是在 Qwen3.5 KV cache 较小的前提下。

## What Actually Landed

为了避免把“暂时不启用”和“完全没做”混在一起，这里单独列当前已经稳定接入主链路的优化：

- Qwen3 batched decode 的 CUDA Graph
  - `infer/src/model/qwen3/batch_decode.rs:246`
- Qwen3 batched decode 的 graph cache invalidation
  - `infer/src/model/qwen3/batch_decode.rs:199`
- FlashInfer plan-once / dirty-replan
  - `infer/src/flashinfer_metadata.rs:192`
- Qwen3 batched fused add+norm
  - `infer/src/model/qwen3/batch_decode.rs:456`
- Qwen3.5 单 token decode 的 CUDA Graph
  - `infer/src/model/qwen35/prefill.rs:302`

这次 review 的重点不是这些已落地项，而是它们没有平移到 Qwen3.5 并发热路径。

## Priority

建议按下面顺序处理：

1. 先补 `Qwen3.5 batched conv1d + batched GDR`
2. 再把 recurrent state 映射到稳定 batch buffer，接通 `Qwen3.5 batched CUDA Graph`
3. 把 `kv_indices` 更新从“CPU 增量 + 全量 H2D”推进到真正 partial H2D，或直接做 GPU-side metadata
4. 评估是否把 `fused_add_rms_norm_offset` / `fused_mlp` 重新接回 Qwen3.5 单 token 或 batched 路径
5. 至少把 Qwen3.5 serving 默认 `num_slots` 提到 8，让吞吐优化能被默认配置体现出来

## Rule

判断“优化有没有做成”，不能只看：

- 有没有 kernel
- 有没有开关
- 有没有计划文档

必须看它是否真正进入了当前 workload 的热路径。

以当前仓库状态看，Qwen3 的并发 decode 链路已经比较完整；Qwen3.5 则明显停在“batched GEMM 已落地，但 recurrent batching 和 batched graph 还没接通”的阶段。
