# KV 量化存储与 Agent 长上下文解决方案

> Status: **Active** | Created: 2026-04-08

---

## 核心命题

Agent 推理的核心矛盾：**多轮对话产生的 KV cache 线性增长 vs GPU 显存恒定**。

一个 8B 模型在 A100-40GB 上，模型权重占 15.5GB，剩余 ~24GB 给 KV cache。BF16 下每个 token 的 KV 开销为 144KB（36 层 x 2(K+V) x 8 heads x 128 dim x 2B），4 个并发 slot 各 4K 上下文就吃掉 2.25GB。Agent 场景下 10 轮对话轻松到 8K-16K token，单 slot 就需要 1.1-2.3GB。这意味着：

- **并发能力被 KV 吞噬** — 长上下文 slot 越多，能服务的请求越少
- **TTFT 线性退化** — 实测 ~0.28ms/token，3K context 时 TTFT 已达 834ms
- **ITL 随 KV 增长** — 从 14ms (T1) 到 23ms (T8)，1.5x 退化

sglang 和 vllm 在这个方向上的投入有限：sglang 聚焦 prefix reuse（RadixAttention），vllm 有 FP8 KV 但缺乏 agent 场景优化。**我们的目标是构建业界最强的 KV 压缩 + 长上下文方案，让 agent-infer 成为 agent 长对话推理的首选引擎。**

---

## 第一部分：KV 量化存储演进

### 现状

已实现 INT8 per-head per-token 对称量化：

```
量化：scale = max(|x|) / 127,  x_q = round(x / scale)
反量化：x = x_q * scale
存储：INT8 数据 + FP32 scale (每 head 每 token 一个)
```

- CUDA kernel：`kv_quant.cu`（quantize + dequantize）
- Rust ops：`ops/kv_quant.rs`（prepare_layer / commit_layer 模式）
- 内存节省：~46%（INT8 + scales + 1 层 bf16 工作缓冲区 vs 全 bf16）
- 工作方式：每层 attention 前 dequant 全部 token 到 bf16 工作区，attention 后 quant 新 token 回 INT8

**当前限制：**

1. **全量 dequant 瓶颈** — 每层都要将全部已有 token 从 INT8 dequant 到 bf16，O(seq_len) 带宽开销
2. **未接入 paged KV pool** — INT8 只在 contiguous KVCache 中，不在 TokenKVPool 中
3. **未接入 scheduler** — 只有 single-request engine 可用 INT8，batched decode 不支持
4. **不支持 CPU offload** — `offload_if_needed()` 对 INT8 直接返回
5. **精度单一** — 只有 INT8，没有 FP8/INT4/混合精度

### 阶段 A：INT8 全链路打通（基线能力）

**目标**：INT8 KV 从 "可用" 变为 "生产就绪"，覆盖 scheduler batched decode 全链路。

#### A1. TokenKVPool INT8 支持

当前 `TokenKVPool` 的 K/V buffer 是 `CudaSlice<u16>`（bf16）。需要增加 INT8 存储路径：

```
TokenKVPool {
    // BF16 路径（现有）
    k_buffers: Vec<CudaSlice<u16>>,
    v_buffers: Vec<CudaSlice<u16>>,

    // INT8 路径（新增）
    k_buffers_q: Vec<CudaSlice<i8>>,
    v_buffers_q: Vec<CudaSlice<i8>>,
    k_scales: Vec<CudaSlice<f32>>,   // [max_total_tokens, num_kv_heads]
    v_scales: Vec<CudaSlice<f32>>,

    dtype: KVCacheDtype,
}
```

**关键设计**：pool 内的 INT8 token 按 NHD layout 存储，与 bf16 路径一致。scatter/gather kernel 需要适配 i8 数据类型。

内存节省：bf16 pool 预分配 24GB → INT8 pool 预分配 ~13GB，同样显存可存 ~1.85x token。

#### A2. FlashInfer INT8 Attention

当前的 prepare/commit 模式每层都要全量 dequant，这在长序列时是严重瓶颈（3K token x 8 heads x 128 dim = 3MB/层，36 层 = 108MB 的 dequant 带宽）。

**两条路径**：

1. **短期（v1）**：保持 prepare/commit，但优化为 **增量 dequant** — 只 dequant 新 token，旧 token 的 bf16 工作区保留。代价是多一层 bf16 缓冲区不释放，但避免重复 dequant。
2. **中期（v2）**：接入 **FlashInfer 原生 FP8 attention** — FlashInfer 从 v0.2 开始支持 FP8 E5M2/E4M3 KV cache，attention kernel 内部直接读 FP8 数据并以 FP32 累加。跳过 dequant 环节。

v2 是正确方向。FlashInfer FP8 attention 在 A100 上几乎无性能损失（FP8 tensor core → FP32 accumulate），且 FP8 E4M3 的精度优于 INT8 对称量化（8-bit 中 3 bit mantissa vs INT8 的 uniform grid，对长尾分布更友好）。

#### A3. Scheduler 集成

Scheduler 的 `step_decode_batch()` 需要感知 KV dtype：

- `prefill_with_pool`：prefill 后 scatter 到 pool 时，如果 pool 是 INT8/FP8，需要 quantize-on-scatter（一个 fused kernel：bf16 attention output → quantize → scatter to pool indices）
- `step_decode_batch`：FlashInfer decode 直接读 INT8/FP8 pool（v2 路径），或 prepare+commit 模式（v1 路径）
- CUDA Graph：INT8/FP8 路径的 graph capture 需要确保 quantize kernel 也在 graph 内

#### A4. INT8 CPU Offload

当前 `offload_if_needed()` 对 INT8 直接跳过。INT8 offload 比 bf16 更有价值——数据量减半，CPU↔GPU 传输减半：

```
BF16 offload: 64 tokens x 144KB/token = 9.2MB/block (per direction)
INT8 offload: 64 tokens x ~78KB/token = 5.0MB/block (per direction)
```

实现：直接 memcpy INT8 数据 + scales 到 host pinned memory。不需要 dequant/requant，因为 INT8 数据本身就是持久化格式。

---

### 阶段 B：FP8 原生存储（性能跃迁）

**目标**：从 INT8 symmetric 升级到 FP8 E4M3，利用硬件原生支持消除 dequant 开销。

#### 为什么 FP8 优于 INT8

| 维度 | INT8 Symmetric | FP8 E4M3 |
|------|----------------|-----------|
| 精度 | Uniform grid，长尾分布损失大 | 对数分布，自然匹配 activation 分布 |
| Attention 兼容 | 需 dequant→bf16→attention | FlashInfer 原生 FP8 input，零 dequant |
| 硬件支持 | 通用但无 tensor core 加速 | Ada/Hopper tensor core 原生 |
| Scale 开销 | FP32 per-head per-token | 可以 per-tensor/per-channel，更少 |
| A100 支持 | 原生 | 软件模拟，10-20% 开销 |

**策略**：A100 用 INT8（已有），H100/H200 用 FP8 E4M3（新增）。运行时根据 SM 版本自动选择。

#### B1. FP8 TokenKVPool

```rust
pub enum KVPoolDtype {
    BF16,
    INT8,  // A100 默认
    FP8,   // H100+ 默认
}
```

FP8 pool 的 buffer 类型为 `CudaSlice<u8>`（E4M3 用 8 bit 表示），scale 可选 per-tensor 或 per-head-group。

#### B2. FP8 Quantize-on-Write Kernel

Fused kernel：attention output (bf16) → FP8 quantize → scatter to pool indices。一次 kernel launch 完成三步，避免中间 bf16 存储。

```c
// Grid: (num_kv_heads, token_count)  Block: (head_dim)
__global__ void quantize_scatter_fp8_kernel(
    const __nv_bfloat16* kv_bf16,     // attention output [tokens, kv_dim]
    __nv_fp8_e4m3* kv_pool,           // pool buffer [max_tokens, kv_dim]
    float* scales,                     // [num_kv_heads] or per-tensor
    const int32_t* scatter_indices,    // pool indices for new tokens
    int head_dim, int num_kv_heads
);
```

#### B3. FlashInfer FP8 Decode 集成

FlashInfer `BatchDecodeWithPagedKVCacheWrapper` 支持 `data_type=torch.float8_e4m3fn`。对应的 C API：

```c
// 已有 flashinfer_batch_decode 签名，增加 kv_dtype 参数
cudaError_t flashinfer_batch_decode_fp8(..., int kv_dtype);
```

预期效果：decode attention 延迟不变（FP8→FP32 accumulate 在 H100 上零额外开销），KV 内存减半。

---

### 阶段 C：K8V4 异构精度（极致压缩）

**核心洞察**：Key 对精度敏感（参与 softmax 归一化），Value 容忍低精度（线性加权求和）。

#### C1. 异构存储布局

```
K cache: INT8/FP8 (8-bit)  — 保持现有精度
V cache: INT4 (4-bit)      — 压缩 50%
```

总内存：相比 BF16，K 节省 50%，V 节省 75%，综合节省 ~62.5%。

#### C2. INT4 V Cache Kernel

INT4 per-head per-token 对称量化：

```
每个 bf16 值 → 4-bit 有符号整数 [-7, 7]
每 2 个 4-bit 值打包到 1 个 byte
scale = max(|v|) / 7.0 per head per token
```

存储：`[num_kv_heads, max_seq_len, head_dim/2]` as uint8（packed INT4）+ `[num_kv_heads, max_seq_len]` f16 scales。

#### C3. FlashInfer Mixed-Precision Attention

需要修改 attention kernel 支持 K=8bit, V=4bit 的混合读取。FlashInfer 目前不原生支持此模式。

**方案 A**：分离 K 和 V 的读取路径 — K 用标准 FP8/INT8 读取，V 用自定义 INT4 dequant + FP32 accumulate。需要 fork FlashInfer 或自研 attention kernel。

**方案 B**：V 在 attention kernel 入口处 on-the-fly dequant 到 bf16 寄存器，利用 shared memory pipeline 隐藏 dequant 延迟。Decode 时 V 读取是带宽瓶颈，INT4 直接减半带宽需求。

我们选方案 B——**decode 是 memory-bandwidth bound，V 读取带宽减半直接提升 2x V 读取效率**。

#### C4. 精度保障

K8V4 的精度风险集中在 V 量化。应对策略：

1. **选择性高精度**：attention score 最高的 top-K 个 token 的 V 保持 8-bit（outlier-aware）
2. **渐进量化**：新 token 写入时 8-bit，被推远后（position > threshold）降级到 4-bit
3. **评估矩阵**：WikiText-2 PPL delta < 0.1，GSM8K accuracy drop < 1%，LongBench < 2%

---

### 阶段 C+：TurboQuant 式可变位宽（灵活性跃迁）

借鉴 [inferrs/turbo_quant.rs](https://github.com/ericcurtin/inferrs/blob/main/inferrs/src/turbo_quant.rs) 的设计理念，将固定 INT8/INT4 扩展为 **可配置 1-8 bit 量化框架**。

#### C+1. 统一可变位宽 Kernel

TurboQuant 的核心抽象：任意 bit 数的 uniform quantization + group-wise absmax scale。inferrs 在 CPU 上实现，我们将其 GPU 化：

```c
// 统一量化 kernel：支持 2/4/6/8 bit
// GROUP_SIZE=32（每 32 个元素共享一个 absmax scale）
__global__ void quantize_kv_variable_kernel(
    const __nv_bfloat16* kv_bf16,     // input [num_heads, seq_len, head_dim]
    uint8_t* kv_packed,                // output packed bits
    float* scales,                     // [num_heads, seq_len, head_dim/GROUP_SIZE]
    int head_dim, int max_seq_len,
    int start_pos, int bits,           // 2, 4, 6, or 8
    int group_size                     // default 32
);
```

**Group quantization vs per-head per-token**：当前 INT8 用 per-head per-token（整个 head_dim 共享一个 scale），head_dim=128 时一个 scale 覆盖 128 个元素。TurboQuant 用 GROUP_SIZE=32，4 个 scale 覆盖 128 个元素，精度更高（outlier 影响范围更小），代价是 4x scale 存储。

对于低 bit（2-4 bit），group quantization 的精度收益远超 scale 存储开销：

| 方案 | 数据存储 | Scale 存储 | 总计/token | 压缩比 (vs bf16) |
|------|----------|-----------|-----------|-----------------|
| BF16 | 256 B | 0 | 256 B | 1x |
| INT8 per-head | 128 B | 4 B | 132 B | 1.94x |
| INT8 group32 | 128 B | 16 B | 144 B | 1.78x |
| INT4 per-head | 64 B | 4 B | 68 B | 3.76x |
| INT4 group32 | 64 B | 16 B | 80 B | 3.20x |
| INT2 group32 | 32 B | 16 B | 48 B | 5.33x |

**结论**：8 bit 用 per-head，4 bit 及以下用 group32。

#### C+2. Nibble Packing（sub-byte 高效存储）

4-bit 量化需要将两个 4-bit 值打包到一个 byte。TurboQuant 采用 MSB-first packing：

```
byte = (idx_high << 4) | idx_low
```

CUDA kernel 中用 vectorized load (`uint4`) 一次读取 16 bytes = 32 个 INT4 值，正好匹配一个 warp（32 threads）。

2-bit 量化：4 个值打包到一个 byte。`uint4` load = 64 个 INT2 值 = 2 个 warp 的工作量。

#### C+3. Warmup 优化（短序列免量化）

TurboQuant 的 warmup 设计很巧妙：序列长度 < 256 时不量化，直接 bf16 存储。避免短序列的 quantize/dequantize 开销（短序列内存不是瓶颈，带宽开销才是）。

我们的实现：

```rust
pub struct WarmupConfig {
    pub warmup_tokens: usize,  // default 256
    // 当 seq_len < warmup_tokens 时，KV 保持 bf16 不量化
    // 超过 warmup_tokens 后，batch 量化所有历史 token
}
```

与 token-age degradation（D2）互补：新 token 先 bf16 → 超过 warmup 阈值后 batch 量化 → 继续 age-based 降级。

### 阶段 D：自适应混合精度（超越竞品的核心差异化）

sglang 没有 KV 量化。vllm 只有固定精度 FP8。**自适应混合精度是我们的独家优势。**

#### D1. Per-Layer Precision Profiling

不同层对量化的敏感度不同。通过 calibration 数据集自动分析：

```rust
pub struct LayerQuantConfig {
    pub k_dtype: KVDtype,  // BF16 / FP8 / INT8
    pub v_dtype: KVDtype,  // BF16 / FP8 / INT8 / INT4
}

pub struct AdaptiveQuantProfile {
    pub layers: Vec<LayerQuantConfig>,
    // 从 calibration 自动生成
}
```

典型结论：early layers (0-3) 和 late layers (30-35) 需要高精度（8-bit K+V），middle layers (4-29) 可用 K8V4。这是学术研究（PM-KVQ, KVTuner）的一致结论。

#### D2. Token-Age Degradation

Agent 场景下，早期 turn 的 token（system prompt、历史对话）的精确度对当前生成的影响递减。利用这一特性：

```
Token age 0-256    (recent)  → BF16 或 FP8 (最高精度)
Token age 256-2048 (active)  → INT8
Token age 2048+    (stale)   → INT4 或 evict
```

实现为后台异步 quantize 任务：当 decode step 空闲时（GPU utilization < 80%），后台 stream 上运行 re-quantize kernel 将老 token 从高精度降级到低精度。

#### D3. Attention-Score-Guided Compression

运行时利用 attention score 反馈：

1. Prefill 结束后，统计每个 KV head 每个 token position 的 attention score 累积值
2. Score 最低的 token 标记为 "可降级" — 降到 INT4 或直接 evict
3. 每 N 个 decode step 更新一次 score 统计（增量更新，不需要重新 softmax）

这是 H2O（Heavy-Hitter Oracle）的工程化实现：只保留 "重要 token" 的高精度 KV，其余压缩或丢弃。

---

## 第二部分：Agent 长上下文解决方案

### 现状

当前长上下文支持：

1. **CPU Offload**（BF16 only）：GPU 存不下时，64-token block 粒度搬到 CPU，attention 前搬回
2. **Prefix Cache**（RadixTree）：相同前缀的请求共享 KV，跳过重复 prefill
3. **Token KV Pool**：全局 token 级 pool，LIFO 分配，FlashInfer 兼容 metadata

**实测性能退化**（Qwen3-8B, A100-40GB, 2 slots）：

| Context | TTFT | ITL | tok/s |
|---------|------|-----|-------|
| 39 tok | 37ms | 14.1ms | 70.8 |
| 1K tok | 282ms | 16.7ms | 58.1 |
| 3K tok | 834ms | 23.1ms | 40.5 |

Agent 对话 10 轮后 context 轻松到 8K-16K，TTFT 将达 2-4 秒，tok/s 降到 30 以下。

**竞品对比**：

| 能力 | agent-infer | sglang | vllm |
|------|-------------|--------|------|
| Prefix Cache | RadixTree | RadixAttention (最强) | Automatic prefix cache |
| KV Swap | CPU offload (BF16) | KV swap + recompute | KV swap |
| KV 量化 | INT8 (single-req only) | 无 | FP8 |
| 智能驱逐 | 无 | 无 | 无 |
| Agent 优化 | 无 | 无 | 无 |

**机会**：三家都没有针对 agent 长对话的专项优化。这是我们的差异化切入点。

### 阶段 E：CPU Offload 全链路（基线能力）

#### E1. INT8 CPU Offload

当前 `offload_if_needed()` 对 INT8 路径直接跳过。补全：

- INT8 数据 + FP32 scales 一起搬到 host pinned memory
- 搬回时直接 memcpy（不需要 re-quantize）
- 对比 BF16 offload 带宽节省 ~45%

#### E2. Async Prefetch Pipeline

当前 `ensure_on_gpu()` 是同步的 — CPU→GPU memcpy 阻塞 compute stream。改为异步：

```
Stream 0 (compute): ... layer N attention ...
Stream 1 (transfer): prefetch layer N+2 KV from CPU → GPU (overlap)
```

双 stream pipeline：compute stream 做当前层 attention，transfer stream 提前搬下一层 KV。对于 36 层模型，理论上可以完全隐藏 transfer 延迟。

#### E3. Paged KV Pool CPU Offload

当前 CPU offload 只在 contiguous KVCache 上。TokenKVPool 也需要支持 CPU shadow：

```rust
pub struct TokenKVPool {
    // GPU buffers（现有）
    k_buffers: Vec<CudaSlice<u16>>,

    // CPU shadow（新增）
    k_host: Vec<Vec<u8>>,     // pinned host memory
    v_host: Vec<Vec<u8>>,
    host_token_map: HashMap<u32, usize>,  // pool_idx → host offset
}
```

策略：LRU eviction — 最久未被 attention 访问的 token 从 GPU pool evict 到 CPU，新 token 分配到 GPU。当 attention 需要被 evict 的 token 时，async prefetch 提前搬回。

---

### 阶段 F：Prefix Cache 深度优化（TTFT 攻坚）

Agent 对话的 TTFT 退化主要来自重复 prefill system prompt + 历史对话。Prefix cache 是最高杠杆的优化。

#### F1. 修复 Qwen3.5 Prefix Cache

当前 prefix cache 对 Qwen3.5 禁用，原因是 recurrent layers 的 state 在 prefix hit 时没有正确重置：

```rust
// infer/src/scheduler/cuda/prefill.rs:17
// TODO: Prefix cache disabled for Qwen3.5 due to recurrent state contamination
```

修复方案：prefix cache hit 时，对 recurrent layers 重新 replay prefix tokens 以重建 state（不需要全量 prefill，只需 recurrent 部分）。Full-attention layers 的 KV 从 cache 复用，recurrent layers 的 state 从 scratch recompute。

代价：Qwen3.5 的 24 个 recurrent layer 的 GDR state recompute。但这远快于全量 prefill（只需 forward recurrent 部分，不需要 attention）。

#### F2. Multi-Level Prefix Cache

Agent 对话有天然的分层结构：

```
Level 0: System prompt (所有对话共享)
Level 1: Tool definitions (同类 agent 共享)
Level 2: Conversation history (同一会话的 turn 间共享)
Level 3: Current turn (不共享)
```

当前 RadixTree 是单层的。升级为 multi-level：

```rust
pub struct HierarchicalPrefixCache {
    system_cache: RadixTree,        // 全局共享，永不 evict
    tool_cache: RadixTree,          // per-agent-type 共享
    conversation_cache: RadixTree,  // per-session 共享
}
```

好处：system prompt 和 tool definitions 的 KV 永远驻留在 GPU，不参与 eviction。Agent 切换会话时只需要 prefill conversation delta。

#### F3. Prefix Cache + KV 量化协同

Prefix cache 中的 KV 可以用更低精度存储（因为 prefix 是 "冷" 数据）：

- 热路径（当前 turn 的新 KV）：BF16 或 FP8
- 温路径（最近几个 turn 的 KV）：INT8
- 冷路径（system prompt、历史 turn）：INT4

转换在 prefix cache entry 被缓存时执行，不在热路径上。

---

### 阶段 G：智能上下文管理（超越竞品的核心能力）

这是 agent-infer 的杀手级特性。sglang 和 vllm 把 KV cache 当作无差别的 FIFO/LRU buffer。但 agent 对话中，不同 token 的重要性差异极大。

#### G1. Token Importance Scoring

在 attention 计算过程中，以极低开销收集每个 KV token 的 "重要性分数"：

```
importance[token_i] = sum_{decode_steps} sum_{q_heads} softmax_score(q, k_i)
```

实现：在 FlashInfer decode kernel 输出后，用一个轻量 reduction kernel 统计每个 KV position 的 attention weight 累积值。每 N 步更新一次（N=32 or 64），开销 < 1% decode 延迟。

#### G2. Importance-Guided Eviction

当 GPU KV pool 满时，不是 FIFO/LRU 驱逐，而是驱逐重要性最低的 token：

```rust
pub enum EvictionPolicy {
    FIFO,           // 现有：最老的先驱逐
    LRU,            // 最久未被访问的先驱逐
    Importance,     // 重要性最低的先驱逐
    Hybrid,         // importance * recency 加权
}
```

**Hybrid 策略**：`score = importance_score * decay(age)^alpha`，其中 `alpha` 控制 recency 的权重。默认 `alpha=0.5`。

被驱逐的 token：
1. 如果是 "可丢弃" 的（importance < threshold）→ 直接丢弃，不搬到 CPU
2. 如果有中等重要性 → 搬到 CPU，按需搬回
3. 如果重要性高 → 永不驱逐（system prompt、关键上下文）

#### G3. Sliding Window + Sink Token

对于超长上下文（>16K），引入 StreamingLLM 式的 sink token 机制：

```
保留：[sink_tokens (4-8个)] + [sliding_window (最近 N 个)] + [important_tokens (动态)]
丢弃：其余
```

Sink token 是序列最开头的几个 token，它们吸收了 softmax 的 "注意力汇聚" 现象。丢弃它们会导致 PPL 爆炸。

与 G2 的 importance scoring 结合：important_tokens 是动态选出的 "heavy hitter"，不受 sliding window 限制。

#### G4. Agent-Aware Context Compression

Agent 对话有结构化信息：tool call + tool result 对。旧的 tool result 通常可以被大幅压缩：

1. **Tool result summarization**：将旧 tool result 的 KV 替换为 summary 的 KV（需要 re-encode summary）
2. **Tool result eviction**：直接丢弃旧 tool result 的 KV，保留 tool call 的 KV（让模型知道调用过什么）
3. **Selective KV retention**：保留 tool result 中 importance score 最高的 top-K 个 token

方案 3 最适合我们的架构——纯 KV 操作，不需要 re-encode，与 G1/G2 自然集成。

---

## 第三部分：两条线的融合 — 统一 KV 生命周期

KV 量化和长上下文管理不是独立的特性，它们在 agent 场景下自然融合为一个统一的 **KV 生命周期管理系统**：

```
Token 生命周期：

  [产生] → BF16 (attention output)
    ↓ quantize-on-write
  [热存储] → FP8/INT8 (GPU pool, 最近 256 token)
    ↓ age-based degradation (async, background stream)
  [温存储] → INT4 (GPU pool, 256-4096 token)
    ↓ importance-guided eviction
  [冷存储] → INT4 (CPU pinned memory, 4096+ token)
    ↓ low importance + old age
  [丢弃] → 释放 (importance < threshold)
```

### 统一 API

```rust
pub struct KVLifecycleManager {
    gpu_pool: TokenKVPool,           // GPU 多精度 pool
    cpu_pool: HostKVPool,            // CPU pinned memory pool
    importance_tracker: ImportanceTracker,  // per-token scores
    prefix_cache: HierarchicalPrefixCache, // multi-level prefix

    // 策略配置
    hot_dtype: KVDtype,              // FP8 (H100) 或 INT8 (A100)
    warm_dtype: KVDtype,             // INT4
    cold_dtype: KVDtype,             // INT4
    hot_window: usize,               // 最近 N token 保持高精度
    eviction_policy: EvictionPolicy, // Hybrid
}

impl KVLifecycleManager {
    /// 新 token 写入（热存储）
    fn write_tokens(&mut self, slot: usize, kv_bf16: &DeviceVec, count: usize);

    /// Decode step 前：确保所有活跃 token 在 GPU
    fn prepare_for_decode(&mut self, slots: &[usize]) -> FlashInferMeta;

    /// Decode step 后：更新 importance，触发降级/驱逐
    fn post_decode_maintenance(&mut self, slots: &[usize], attention_scores: &DeviceVec);

    /// 请求结束：释放 slot，KV 可能留在 prefix cache
    fn release_slot(&mut self, slot: usize, keep_prefix: bool);
}
```

### 与 Scheduler 集成

Scheduler 的 step 循环变为：

```
fn step(&mut self) {
    // 1. KV lifecycle maintenance (async, background stream)
    self.kv_manager.background_maintenance();

    // 2. Batched decode
    let meta = self.kv_manager.prepare_for_decode(&active_slots);
    self.model.forward_decode_batch(..., &meta);
    self.kv_manager.post_decode_maintenance(&active_slots, &attn_scores);

    // 3. Prefill chunks
    //    prefix cache hit → skip prefill for matched portion
    //    new tokens → write to hot storage
}
```

---

## 第四部分：实施路线图

### Phase 1：基线打通（2-3 周）

| 任务 | 文件 | 依赖 | 预期效果 |
|------|------|------|----------|
| A1. TokenKVPool INT8 | paged_kv.rs, kv_quant.cu | 无 | Pool 支持 INT8 存储 |
| A3. Scheduler INT8 | scheduler/cuda/ | A1 | Batched decode INT8 全链路 |
| A4. INT8 CPU Offload | kv_cache.rs | 无 | INT8 offload 带宽减半 |
| E2. Async Prefetch | kv_cache.rs | 无 | Offload 延迟隐藏 |
| F1. Qwen3.5 Prefix Cache 修复 | scheduler/cuda/prefill.rs | 无 | Qwen3.5 TTFT 大幅改善 |

**验证**：C=4 Qwen3-8B INT8 KV，throughput 对比 BF16 无退化，内存减半。Qwen3.5 prefix cache hit 验证。

### Phase 2：FP8 + Prefix 优化（2-3 周）

| 任务 | 文件 | 依赖 | 预期效果 |
|------|------|------|----------|
| B1. FP8 TokenKVPool | paged_kv.rs | A1 | H100 FP8 存储 |
| B2. FP8 Quantize-on-Write | csrc/cuda/kv_quant_fp8.cu | B1 | 零 dequant 开销 |
| B3. FlashInfer FP8 Decode | ops/attention.rs, ffi.rs | B1 | Attention 原生 FP8 |
| F2. Hierarchical Prefix Cache | prefix_cache.rs | F1 | Agent 多层 prefix 复用 |

**验证**：H100 FP8 KV decode 延迟 = BF16（零退化），PPL delta < 0.05。Agent 多轮对话 TTFT 对比。

### Phase 3：混合精度 + 智能管理（3-4 周）

| 任务 | 文件 | 依赖 | 预期效果 |
|------|------|------|----------|
| C+1. 可变位宽 Kernel (TurboQuant) | csrc/cuda/kv_quant_variable.cu | A1 | 2/4/6/8 bit 统一框架 |
| C+3. Warmup 免量化 | kv_cache.rs, paged_kv.rs | 无 | 短序列零开销 |
| C1-C2. K8V4 存储 | kv_quant.cu (new INT4) | C+1 | 62.5% 内存节省 |
| C3. Mixed-Precision Attention | csrc/cuda/attention_mixed.cu | C1 | K8V4 decode kernel |
| G1. Importance Scoring | ops/attention.rs | 无 | 每 token importance 分数 |
| G2. Importance Eviction | paged_kv.rs | G1, E1 | 智能驱逐策略 |
| D2. Token-Age Degradation | paged_kv.rs | C1, G1 | 自动降精度 |

**验证**：K8V4 PPL < BF16 + 0.1，LongBench < 2% drop。智能驱逐 vs FIFO 对比。

### Phase 4：Agent 专项 + 极致优化（2-3 周）

| 任务 | 文件 | 依赖 | 预期效果 |
|------|------|------|----------|
| G3. Sliding Window + Sink | scheduler/ | G2 | 无限上下文 |
| G4. Agent-Aware Compression | scheduler/ | G2 | Tool result 智能压缩 |
| D1. Per-Layer Profiling | 新 calibration tool | C1 | 自动混合精度配置 |
| D3. Attention-Score Compression | ops/attention.rs | G1 | 实时自适应压缩 |
| 统一 KVLifecycleManager | 新模块 | 所有 | 完整生命周期管理 |

**验证**：20+ 轮 agent 对话无精度退化。128K context 可服务。对比 sglang/vllm 相同硬件 agent benchmark。

---

## 第五部分：竞争力分析

### vs sglang

sglang 的核心优势是 RadixAttention（prefix reuse）和调度效率。但它：
- **没有 KV 量化** — 所有 KV 存储为 bf16，内存利用率是我们的 2-4x
- **没有智能驱逐** — 简单的 FIFO/LRU
- **没有 agent 专项优化** — 通用 serving 引擎

我们超越 sglang 的路径：
1. **同等 prefix reuse**（RadixTree 已有）+ **KV 量化**（同样显存 2-4x 并发）
2. **Agent-aware context management** — sglang 不区分 system prompt 和 tool result
3. **智能驱逐** — 基于 attention score 而非时间

### vs vllm

vllm 有 FP8 KV quantization 和 PagedAttention，但：
- **FP8 需要离线 calibration**（llm-compressor）— 我们是 online quantize，零配置
- **固定精度** — 不支持混合精度、per-layer 适配
- **PagedAttention 是 block 级** — 我们是 token 级（更细粒度，更少碎片）
- **没有 agent 优化** — 通用 serving

我们超越 vllm 的路径：
1. **Online FP8/INT8 零配置** vs vllm 需要离线 calibration
2. **自适应混合精度** vs 固定 FP8
3. **Token 级 pool** vs block 级 PagedAttention
4. **Agent-aware lifecycle** — 独家能力

### 量化目标

| 指标 | 当前 | Phase 1 后 | Phase 4 后 |
|------|------|-----------|-----------|
| KV 内存 (per token) | 144 KB | 78 KB (INT8) | 36-54 KB (K8V4/INT4) |
| 同等显存最大并发 | 4 slots | 7 slots | 12-16 slots |
| Agent 最大 context | 8K | 16K (INT8 + offload) | 128K+ (mixed + eviction) |
| TTFT @ 4K context | 1.1s | 200ms (prefix cache) | 50ms (hierarchical cache) |
| Agent 20 轮可用 | 否 (8 轮 OOM) | 是 (INT8) | 是 (自适应) |

---

## 第六部分：PPL 与精度验证体系

每一阶段的量化方案必须通过完整的精度验证才能合入主线。单一 PPL 指标不足以保证生产质量——需要多维度、多长度、多任务的系统性验证。

### 验证流水线

```
                ┌──────────┐
                │ 量化方案  │
                └────┬─────┘
                     │
          ┌──────────▼──────────┐
          │ Gate 1: PPL 快筛    │  WikiText-2 + C4
          │ delta < 0.1 → pass  │  delta > 0.5 → reject
          └──────────┬──────────┘
                     │ pass
          ┌──────────▼──────────┐
          │ Gate 2: 推理任务    │  GSM8K + MMLU + HumanEval
          │ accuracy drop < 1%  │
          └──────────┬──────────┘
                     │ pass
          ┌──────────▼──────────┐
          │ Gate 3: 长上下文    │  NIAH 4K→128K + LongBench
          │ accuracy drop < 2%  │
          └──────────┬──────────┘
                     │ pass
          ┌──────────▼──────────┐
          │ Gate 4: Agent E2E   │  多轮 tool-use 对话
          │ tool call 正确率    │  无退化
          └──────────┬──────────┘
                     │ pass
          ┌──────────▼──────────┐
          │ Gate 5: 吞吐/内存   │  throughput, peak mem
          │ 压缩比达标          │
          └──────────┘
```

### Gate 1：语言建模 PPL

**基线**：BF16 KV cache 下的 PPL。

**测试集**：
- WikiText-2 (test split)：通用英文语言建模
- C4 (validation, 1000 samples)：多领域 web 文本

**指标**：
```
PPL_delta = PPL_quantized - PPL_baseline
```

**阈值**：

| PPL Delta | 判定 | 动作 |
|-----------|------|------|
| < 0.05 | 优秀 | 自动 pass |
| 0.05 - 0.10 | 合格 | pass，但记录 warning |
| 0.10 - 0.50 | 边缘 | 需要 Gate 2-4 全部通过才 pass |
| > 0.50 | 不合格 | reject，不进入后续 gate |

**实现**：

```bash
# PPL 评估脚本（新增）
python scripts/eval_ppl.py \
  --model-path models/Qwen3-8B \
  --server-url http://localhost:8000 \
  --kv-dtype int8 \
  --datasets wikitext2,c4 \
  --output results/ppl_int8.json
```

核心逻辑：滑动窗口 PPL 计算，每次送入 `stride` 个 token（default 512），用 `/v1/completions` 的 `logprobs` 字段获取 log probability。

```python
def compute_ppl(model_url: str, tokens: list[int], stride: int = 512) -> float:
    """Sliding window PPL via logprobs API."""
    nlls = []
    for i in range(0, len(tokens), stride):
        input_ids = tokens[max(0, i - max_length):i + stride]
        response = completions(input_ids, max_tokens=0, logprobs=True, echo=True)
        # 收集 [i:i+stride] 范围内的 logprobs
        nlls.extend(response.logprobs[...])
    return math.exp(-sum(nlls) / len(nlls))
```

### Gate 2：推理任务准确率

**测试集**：

| 任务 | 数据集 | 指标 | 采样 |
|------|--------|------|------|
| 数学推理 | GSM8K (全集 1319) | Accuracy (EM) | greedy |
| 知识问答 | MMLU (5-shot, 57 subjects) | Accuracy | greedy |
| 代码生成 | HumanEval (164 问题) | pass@1 | temperature=0.0 |
| 常识推理 | ARC-Challenge (1172) | Accuracy | greedy |
| 科学推理 | GPQA-Diamond (198) | Accuracy | greedy |

**阈值**：每个任务 accuracy drop < 1%（绝对值）。任一任务超过 2% drop 则 reject。

**实现**：

```bash
# 推理评估脚本
python scripts/eval_reasoning.py \
  --model-path models/Qwen3-8B \
  --server-url http://localhost:8000 \
  --kv-dtype int8 \
  --tasks gsm8k,mmlu,humaneval,arc,gpqa \
  --output results/reasoning_int8.json
```

集成 lm-evaluation-harness（EleutherAI）作为后端，通过 OpenAI-compatible API 适配器调用 agent-infer 的 HTTP server。

### Gate 3：长上下文准确率

**核心洞察**：KV 量化的精度损失随序列长度**累积**。短序列 PPL 正常不代表长序列没问题。

**测试集**：

| 任务 | 长度范围 | 指标 |
|------|----------|------|
| Needle-in-a-Haystack (NIAH) | 4K, 8K, 16K, 32K, 64K, 128K | 检索准确率 |
| LongBench (6 tasks) | 2K-32K | Task-specific accuracy |
| Ruler (4 tasks) | 4K-128K | Accuracy |

**NIAH sweep**：在不同深度（10%, 25%, 50%, 75%, 90%）插入 needle，测量检索准确率。BF16 baseline 应该是 100%。量化方案必须保持 98%+。

**阈值**：
- NIAH：所有长度所有深度 accuracy ≥ 98%
- LongBench：average accuracy drop < 2%
- Ruler：average accuracy drop < 3%（128K 长度允许 5%）

**实现**：

```bash
python scripts/eval_longcontext.py \
  --model-path models/Qwen3-8B \
  --server-url http://localhost:8000 \
  --kv-dtype int8 \
  --lengths 4096,8192,16384,32768,65536,131072 \
  --tasks niah,longbench,ruler \
  --output results/longctx_int8.json
```

### Gate 4：Agent 端到端

**测试场景**：模拟真实 agent 多轮对话，验证量化不影响 tool calling 正确性。

```python
AGENT_SCENARIOS = [
    {
        "name": "code_agent_10turn",
        "turns": 10,
        "tools": ["bash", "read_file", "write_file"],
        "task": "Debug a Python script with 3 bugs",
        "expected_tool_calls": ["read_file", "bash", "write_file", ...],
    },
    {
        "name": "research_agent_15turn",
        "turns": 15,
        "tools": ["web_search", "read_url", "summarize"],
        "task": "Research and compare 3 ML frameworks",
        "expected_outputs": [...],
    },
]
```

**指标**：
- Tool call 正确率（格式正确 + 参数正确）
- Task completion rate
- 与 BF16 baseline 的输出一致性（ROUGE-L > 0.85）

**阈值**：Tool call 正确率无下降。Task completion rate 无下降。

### Gate 5：性能与内存

每个量化方案必须证明性能收益：

| 指标 | 测量方式 | 期望 |
|------|----------|------|
| KV 内存占用 | `nvidia-smi` peak memory | 按方案目标压缩 |
| Decode throughput (C=1) | `bench_throughput.py --concurrency 1` | ≥ 95% of BF16 |
| Decode throughput (C=4) | `bench_throughput.py --concurrency 4` | ≥ BF16 (应更高) |
| TTFT (512 prompt) | `bench_throughput.py --input-len 512` | ≤ 110% of BF16 |
| Max context length | OOM 前最大 seq_len | ≥ 1.5x BF16 |
| Quantize kernel 延迟 | nsys profile | < 5% of decode step |

### 每阶段验证矩阵

| Gate | Phase 1 (INT8) | Phase 2 (FP8) | Phase 3 (K8V4) | Phase 4 (Adaptive) |
|------|---------------|---------------|----------------|-------------------|
| G1: PPL | WikiText-2, C4 | WikiText-2, C4 | WikiText-2, C4 | WikiText-2, C4 |
| G2: Reasoning | GSM8K, MMLU | GSM8K, MMLU | GSM8K, MMLU, HumanEval | 全部 5 tasks |
| G3: Long Context | NIAH 4K-32K | NIAH 4K-32K | NIAH 4K-64K, LongBench | NIAH 4K-128K, LongBench, Ruler |
| G4: Agent E2E | 5 轮 basic | 5 轮 basic | 10 轮 multi-tool | 15 轮 full scenario |
| G5: Performance | C=1,4 throughput | C=1,4 throughput | C=1,4,8 throughput | C=1,4,8,16 throughput |

### 自动化 CI 集成

长期目标：将 Gate 1-2 集成到 CI pipeline，每个涉及 KV 量化的 PR 自动运行：

```yaml
# .github/workflows/kv-quant-eval.yml
kv-quant-eval:
  runs-on: [self-hosted, gpu-a100]
  steps:
    - run: cargo build --release
    - run: cargo run -p infer --release -- --model-path models/Qwen3-8B --kv-dtype int8 &
    - run: python scripts/eval_ppl.py --datasets wikitext2 --kv-dtype int8
    - run: python scripts/eval_reasoning.py --tasks gsm8k --kv-dtype int8
    # Gate 3-5 在 nightly 或 release 前手动触发
```

### 历史基线记录

每次量化方案验证后，将结果记录到 `docs/experience/wins/` 中：

```
docs/experience/wins/YYYY-MM-DD-kv-quant-<method>-eval.md
```

格式：
```markdown
# YYYY-MM-DD · KV Quant <method> Evaluation

## Config
- Model: Qwen3-8B
- KV dtype: INT8 per-head per-token symmetric
- Hardware: A100-40GB

## Results
| Gate | Metric | BF16 Baseline | Quantized | Delta |
|------|--------|---------------|-----------|-------|
| G1   | WikiText-2 PPL | 7.23 | 7.28 | +0.05 |
| ...  | ...    | ...           | ...       | ...   |

## Verdict: PASS / FAIL
```

---

## References

- [KVQuant (NeurIPS 2024)](https://arxiv.org/abs/2401.18079) — 异构 K/V 精度，NUQ
- [PALU (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/7da6e0e00702c60607a6ae05c802ef85-Paper-Conference.pdf) — Low-rank + quantization
- [H2O: Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14048) — Attention-based eviction
- [StreamingLLM](https://arxiv.org/abs/2309.17453) — Sink tokens + sliding window
- [PM-KVQ](https://openreview.net/forum?id=Vem6FQvRvq) — Per-layer mixed precision
- [FlashInfer FP8 Attention](https://docs.flashinfer.ai/) — 原生 FP8 KV decode
- [vLLM KV Quantization](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)
- [NVIDIA NVFP4 KV Cache](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)
- [inferrs TurboQuant](https://github.com/ericcurtin/inferrs/blob/main/inferrs/src/turbo_quant.rs) — 可变位宽 (1-8 bit) group quantization，warmup 优化，nibble packing
