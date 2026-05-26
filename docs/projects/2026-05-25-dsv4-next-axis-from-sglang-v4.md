---
title: DSv4 next-axis backlog — SGLang V4 借鉴
date: 2026-05-25
type: backlog + execution-order index
status: live — codex 自上而下 pick；Claude 维护；license-or-kill per axis
owner: ckl
related:
  - docs/research/2026-05-15-dsv4-decode-memaccess-binding-constraints.md
  - docs/experience/wins/2026-05-25-bench-guidellm-cuda-l4-arle-vs-sglang-headtohead.md
  - docs/plans/2026-05-25-cuda-perf-codex-collab.md
  - https://www.lmsys.org/blog/2026-04-25-deepseek-v4/
---

# DSv4 next-axis backlog — SGLang V4 借鉴

## Why now

session 4（DSv4 H20 pod）当前正调 `quantized_gemv.cu` 的 `DSV4_BATCH_TILE`，
tile=32 拿到 prefill TTFT −11.8%（35.5s → 31.3s），tile=64 −56.4% 被 kill。
这是 L3 (in-kernel GEMV) 这一层的局部最优。

SGLang 2026-04-25 DSv4 day-0 blog 给出的 V4 工程动作集中在 **跨 kernel boundary
fusion** 和 **CUDA graph 内化 metadata**——属于另一个量级的 lever。本 backlog
把这些 SGLang pattern 套到 ARLE 已有 binding-constraints 表（L1–L6）上，给
codex 自上而下取活。

## Serving SLO Baseline

后续 DSv4 性能优化统一按这个用户-facing 目标 framing，不再用短 token smoke
代替吞吐判断：

| Metric | Target / baseline |
|---|---:|
| Workload length | input 32K / output 1.5K |
| SLO TTFT | <= 5000 ms |
| SLO TPOT | <= 30 ms |
| Hardware | H20 |
| QPS | 8 |
| Concurrency | 8 |
| Current target TTFT | 4800 ms |
| Current target TPOT | 18 ms |
| Current total throughput | 8402 |

`max_tokens=1` 只允许标成 prefill/TTFT smoke；decode 或 wall-clock PASS/KILL
必须用 `max_tokens >= 32`，最终回到上表的 32K/1.5K、c=8、qps=8 framing。

## Mapping — SGLang V4 trick × ARLE binding constraint

| ARLE L 层 | 当前现状 | SGLang V4 对应解法 | 本 backlog axis |
|---|---|---|---|
| L6 NCCL combine | 20 ms / rank-range | DeepGEMM Mega MoE（EP dispatch + GEMM1 + SwiGLU + GEMM2 + EP combine 单 symmetric-memory kernel） | A1 |
| L4 attention | hybrid + CSA select ≈ 10% GPU | FlashMLA hybrid attention（SWA + 压缩 attention 单 fused kernel） | A2 |
| L5 DtoH | 344 次/token，44 KiB，全 sync overhead | In-graph metadata（captured kernels 在 graph 内重建 metadata） | A3 |
| 多流准备 | DSv4 prepare/route/MoE 串行 | Hierarchical Multi-Stream Overlap（4 prep op + 细粒度 event） | A4 |
| 长 ctx 容量 | longseq 单卡 TTFT 31.3 s | HiSparse CPU offload（C4 inactive KV → CPU） | A5 |
| 长 prompt prefix reuse | ARLE RadixCache 是单池物理槽 | ShadowRadix（virtual full-token slot indexing） | A6 |
| 长 prefill TP scaling | TP 暴露 NCCL，CP 才能线性 scale | Context Parallelism（token round-robin to attn ranks） | A7 |
| spec decode | DSv4 spec 未 throughput-positive | MTP 单层 head + graph-fused metadata | A8 |

## Rules

- 一 axis 一组 commit。PASS 落 `docs/experience/wins/`，KILL 落
  `docs/experience/errors/`。
- license-or-kill 用 **wall-clock framing**（per-request total ms 或 token/s），
  不许只看 nsys window 占比（避免 M_pf-graph v2 那种"55.7%/window 但 0.32%/
  wall-clock"陷阱，见 CLAUDE.md §0 实证 anchor）。
- 单变量原则：每 axis 只动一处 source，绝不和别的 axis 叠改后归因。
- Codex 有重排权——按 nsys 实测 ranking 调整顺序，把理由写进 wins/errors。
- 取活前先看 §Live state 看 P5 / GPU 是否占用，session 4 当前在跑就别动它。

## Live state

- session 4 codex 持有 sgl-pod 的 `cargo build` + bench cycle，DSv4 tile=32
  收敛中。**A1–A4 不要在它收尾前触发**（资源冲突 + 归因混淆）。
- A1（Mega MoE）需要 `DeepGEMM` 或自写 symmetric memory kernel — codex 进入
  此 axis 前先和 ckl 走一道 architecture license。
- A6 / A7 涉及 RadixCache 和 TP→CP 重构，**先 Claude 写 design doc 再 codex
  动手**。

## Queue（执行顺序按预期 wall-clock 杠杆排序，可改）

### A3 · In-graph metadata（先做，单卡，最便宜）

**SGLang 机制**：DSv4 prepare 阶段的 expert count / batch shape readback 全部
塞进 CUDA graph，"captured kernels rebuild metadata inside the graph"。下一
kernel 的 launch param 直接由 device 在 graph 内产，**不出 graph**。

**ARLE 当前**：单 token nsys 仍有 344–347 个 `cuMemcpyDtoHAsync_v2`，payload
44 KiB，**完全是 sync/launch overhead 不是带宽**（L5 binding constraint）。

**License-or-kill**：
- **PASS**：单 token decode 路径 DtoH 次数 ≤ 50；per-token wall-clock 减少
  ≥ 5%；output 与 baseline byte-identical（greedy）。
- **KILL**：DtoH 减少但 wall-clock 持平 / 退步，或者引入 sync 漏洞。

**Implementation boundary**：`crates/cuda-kernels/csrc/moe/dsv4_route.cu`
+ `infer/src/model/deepseek/mlp.rs` 的 prepare 路径；从 single-token decode
入手，prefill 后做。

---

### A2 · FlashMLA hybrid attention 单 kernel（单卡，~10%→~5% GPU 时间）

**SGLang 机制**：SWA（sliding window）和 sparse/dense compressed attention
**一次 fused kernel call**，"share metadata construction"。Hopper sm_90 head
padding 到 64 的倍数。

**ARLE 当前**：`dsv4_hybrid_attention_kernel` 6.4% + `dsv4_csa_select_kernel`
3.9% ≈ 10% GPU 时间（2026-05-14 trace）。两个独立 kernel，中间 metadata 落
smem/L2。

**License-or-kill**：
- **PASS**：fused kernel 单 token decode 时间下降至原 hybrid + csa 之和的
  60%；output byte-identical；尾部 latency p99 不退步。
- **KILL**：fused kernel register pressure 超阈，occupancy 降低导致 wall-clock
  退步；或 metadata 共享引入数值漂移。

**Implementation boundary**：`crates/cuda-kernels/csrc/attention/dsv4_*.cu`
新增 fused entry；老路径保留作回滚开关；同一 nsys 对照 trace。

**Execution log**：
- 2026-05-26 A2.0 fused B=1 decode window-cache update into SWA / hybrid
  attention tails. It removed 9504 standalone `dsv4_update_window_cache_kernel`
  launches in the measured H20 `max_tokens=32` trace and kept greedy output
  byte-identical. This is a launch-churn substep, not full FlashMLA.
- 2026-05-26 A2.1 fused Q RMSNorm+RoPE prep and K RoPE prep into one CUDA
  launch with `ARLE_DSV4_FUSE_QK_PREP=0` fallback. Split nsys had
  `dsv4_prepare_q_kernel` 7352 calls and `dsv4_prepare_k_kernel` 7342 calls;
  fused nsys had `dsv4_prepare_qk_fused_kernel` 7290 calls and byte-identical
  `max_tokens=32` output. This still does not close full CSA + hybrid
  attention fusion.

---

### A4 · Hierarchical multi-stream overlap（A1 / A2 落地后做，scheduler 层）

**SGLang 机制**：4 个 prep op 并发跑（q_lora / q_scale / k cache / kv layout），
用细粒度 event（`q_lora_ready`、`q_scale_ready`）串依赖，整段在小 batch 下
塞进 CUDA graph 内。

**ARLE 当前**：DSv4 prepare → route → MoE → combine 是串行 stream 调度。

**License-or-kill**：
- **PASS**：单 token decode wall-clock 下降 ≥ 8%；NCCL/D2H sync 不增加；
  prefill 不退步。
- **KILL**：event 风暴导致 driver 排队，或多 stream 引入 race。

**Implementation boundary**：`infer/src/model/deepseek/mlp.rs` stream 编排
+ `crates/cuda-kernels/src/ffi/moe.rs` event 暴露；从单 layer prototype 起，
按 nsys 对照逐 layer 推。

---

### A1 · DeepGEMM Mega MoE（多卡 EP，预期最大杠杆，需 architecture license）

**SGLang 机制**：单一 symmetric-memory mega-kernel 内完成 **EP dispatch →
FP8×FP4 GEMM1 → SwiGLU → FP8×FP4 GEMM2 → EP combine**。NVLink 通信和
tensor-core 计算在 kernel 内重叠，weight layout aliasing 避免两份 expert
weight 常驻。

**ARLE 当前**：DeepEP 已落地，但 dispatch / GEMM / combine 仍是分离 kernel +
ncclReduce 同步。L6 trace 显示 reduce-scatter 段 ≈ 20 ms / rank-range。

**License-or-kill**：
- **PASS**：8-rank H20 长 prompt prefill TTFT ≥ 20% 提升；step_prefill_kernel_
  launch 在 nsys NVTX 下 reduce-scatter 占比 ≥ 减半；output byte-identical。
- **KILL**：symmetric memory 路径要求改 NCCL allocator 且引入 cross-process
  alias bug；或 fused kernel 在 H20 上 register pressure 导致退步；或合并后
  PCIe / NVLink 不饱和暴露新瓶颈。

**Implementation boundary**：这是**架构级改动**，不要单 PR 推。先写
`docs/plans/2026-XX-dsv4-mega-moe.md` 拿 ckl 的 architecture license。需求面：
DeepEP / NVSHMEM symmetric memory primitive + 自写 GEMM kernel
（或者集成 SGLang 同款 DeepGEMM，但 license/兼容要查）。

**2026-05-26 gate update**：native DeepEP 现在是 DSv4 通信侧最高优先级，
但不是现有 one-process multi-thread worker 的 drop-in。远端 8xH20 evidence：
official DeepEP multi-process LL 通过（dispatch+combine about 48.7 us/rank），
official intranode multi-process DSv4 decode shape 通过（BF16 dispatch best
42.05 us, combine best 36.34 us）；ARLE same-process 8-thread LL gate 180 s
timeout，same-process intranode gate 在 `cudaIpcOpenMemHandle` 报
`invalid device context`。下一步必须先做 process-per-rank DeepEP transport
设计/接入；继续同进程强塞或继续小 launch 轴不是收益优先级。Evidence：
[`../experience/errors/2026-05-26-dsv4-native-deepep-process-model-gate.md`](../experience/errors/2026-05-26-dsv4-native-deepep-process-model-gate.md).

---

### A5 · HiSparse CPU offload — C4 inactive KV 卸 CPU（长 ctx 单卡 3×）

**SGLang 机制**：long-ctx 下 C4（compressed layer）的 inactive KV 卸到 CPU
DRAM，按需 prefetch；3× 容量/吞吐。

**ARLE 当前**：DSv4 longseq smoke 单卡 H20 TTFT 31.3 s 是字节容量天花板；
KV pool 全 HBM 驻留。

**License-or-kill**：
- **PASS**：相同 long prompt 下 KV pool HBM 占用 ≤ 50%；TTFT 退步 ≤ 5%；
  output byte-identical；吞吐（每秒处理 prompt tokens）≥ 提升 2×。
- **KILL**：CPU↔HBM PCIe 带宽不够，offload 比 swap-out 还慢；或 cache miss
  pattern 让 TTFT p99 暴涨。

**Implementation boundary**：`infer/src/kv_tier/` + `crates/kv-native-sys/`
已有 tier 抽象，加 CPU layer 是渐进改动；C4 是哪些 DSv4 layer 要先查 spec
（`crates/deepseek-spec/`）。

---

### A6 · ShadowRadix prefix cache（长 prompt 复用 + 多请求）

**SGLang 机制**：radix tree 索引"virtual full-token slot"，per-pool index
mappings 到 physical pool。10K token 请求只保留 128 SWA + C4/C128 compressed
KV。

**ARLE 当前**：RadixCache 单池物理 slot，virtual layer 不存在。

**License-or-kill**：
- **PASS**：相同 prompt 在第二次请求 prefix 命中率 ≥ 95%；HBM 占用相同
  prompt 下降 ≥ 60%；输出与 cold path 一致。
- **KILL**：virtual→physical mapping 引入 race / staleness；或 mapping 自身
  开销 ≥ 5% wall-clock。

**Implementation boundary**：`infer/src/kv_tier/` RadixCache 改造，**先 design
doc**。

---

### A7 · Context Parallelism（长 prefill 多卡 scaling）

**SGLang 机制**：CP round-robin token 到 attention ranks，每 rank 拥有
`1/cp_size` 序列。"long-context TTFT scale"。

**ARLE 当前**：TP 暴露 NCCL，长 prefill 的 attention 是 reduce-scatter 重灾区
（A1 解决一部分），但 CP 是结构性 scale 路径。

**License-or-kill**：
- **PASS**：8-rank long prompt（≥ 32K token）TTFT 相对 TP 提升 ≥ 40%；输出
  与 TP byte-identical；NCCL 流量在 attention 段下降 ≥ 70%。
- **KILL**：CP 在短 prompt 上反而退步 > 10%（admission 必须切回 TP）；
  或多 KV pool 排布让 RadixCache 失效。

**Implementation boundary**：架构级，先 design doc，再 code。

---

### A8 · MTP 单层 spec decode head（最后做，需要训练侧配合）

**SGLang 机制**：独立训练的 DSv4 decoder layer（SWA-only attention），accept
length ~2.5；hybrid attn metadata 在 CUDA graph 内重建（与 A3 同款）。

**ARLE 当前**：DSv4 spec decode 未 throughput-positive，没有 MTP head。

**License-or-kill**：
- **PASS**：c=1 和 c=4 wall-clock output tok/s ≥ 1.25×（CLAUDE.md §SGLang
  T7-C gate 同款）；temperature=0 输出与 no-spec 一致；draft KV 不挤压
  target KV 池。
- **KILL**：accept length 看着好但 wall-clock < 1.25×；或 draft KV 让长 ctx
  admission 退步。

**Implementation boundary**：A3 必须先落地（hybrid attn metadata 在 graph 内
是 MTP 的前置）；MTP head 训练侧由 OPD pipeline 负责。

## Cross-refs

- ARLE binding-constraints 主表（L1–L6 实证）：
  [`../research/2026-05-15-dsv4-decode-memaccess-binding-constraints.md`](../research/2026-05-15-dsv4-decode-memaccess-binding-constraints.md)
- 同日 ARLE vs SGLang head-to-head wins 实证：
  [`../experience/wins/2026-05-25-bench-guidellm-cuda-l4-arle-vs-sglang-headtohead.md`](../experience/wins/2026-05-25-bench-guidellm-cuda-l4-arle-vs-sglang-headtohead.md)
- CUDA perf 主 plan：
  [`../plans/2026-05-25-cuda-perf-codex-collab.md`](../plans/2026-05-25-cuda-perf-codex-collab.md)
- SGLang V4 day-0 blog 原文：
  https://www.lmsys.org/blog/2026-04-25-deepseek-v4/
- SGLang V4 attention backend matrix（FlashMLA / CutlassMLA / FlashInfer /
  TRTLLM-Gen）：https://docs.sglang.io/basic_usage/deepseek_v3.html
