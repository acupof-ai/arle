# SGLang HiCache:从零到专家的一篇文章

> 目标:读完这一篇,你掌握 HiCache 的全部核心机制、能读源码、能在生产环境调参,并知道下一步成为顶级专家应该做什么。
> 适合飞机离线阅读。所有数字都标注了来源。

---

## 0. 30 秒结论

**HiCache 是 SGLang 在 2025 年 9 月推出的分层 KV cache 子系统**,把 RadixAttention 从"只用 GPU 显存"扩展到三级缓存:

- **L1 = GPU 显存**(原 RadixAttention 的领地)
- **L2 = CPU 主机 pinned memory**(扩容 2× ~ 8×,本地)
- **L3 = 分布式存储**(Mooncake / 3FS / NIXL / AIBrix,跨实例共享、近乎无限扩容)

核心创新只有 4 个,记住就够了:

1. **HiRadixTree** — 在 RadixTree 节点上加一个"我现在在哪个 tier"的元数据指针,让一棵树跨三层管理。
2. **GPU-assisted I/O kernels** — 替代 `cudaMemcpyAsync`,CPU↔GPU 传输吞吐 **3×**(LMSYS 官方数据)。
3. **Layer-wise compute-transfer overlap** — 第 N 层在算的时候,第 N+1 层的 KV 在传,延迟被算力吃掉。
4. **三种 write-back × 三种 prefetch 策略** — 让你按延迟/吞吐/容量做工程取舍。

**官方收益**(LMSYS 2025-09-10 blog):多轮场景下 **吞吐 6×**、**TTFT 降 80%**;DeepSeek-R1-671B PD 解耦部署中,cache 命中时 **TTFT 降 84%**;Qwen3-Coder-480B 多轮 agent 场景命中率从 **40% → 80%**,TTFT 降 56%。

如果你只想看到这,就够了。下面是从底到顶的完整推导。

---

## Part I — 前置知识(纯小白起点)

### 1.1 为什么 LLM 推理需要"缓存"

LLM 推理分两个阶段:

| 阶段 | 工作 | 计算复杂度 | 你能感知到的 |
|------|------|-----------|-------------|
| **Prefill** | 把整个 prompt 喂进模型,算出每个 token 的 K、V 向量 | O(n²) — 跟 prompt 长度平方相关 | "首字延迟"(TTFT) |
| **Decode** | 一个 token 一个 token 往外吐,每步只算 1 个新 token 的 attention | O(n) — 但每步都要看前面所有 K、V | 输出速度(tokens/s) |

Decode 的时候,你需要前面所有 token 的 K、V 向量做 attention 计算。每次都重算太贵,所以把它们存起来 —— 这就是 **KV Cache**。

KV Cache 大小的简单估算公式:

```
KV_size = num_layers × 2 (K and V) × num_kv_heads × head_dim × seq_len × batch_size × dtype_bytes
```

举例:Llama-3-70B,80 层,8 个 KV heads,head_dim=128,fp16,序列长 8K,batch=1
≈ 80 × 2 × 8 × 128 × 8192 × 2 ≈ **2.7 GB**(单个请求的单条序列)。
batch=32、seq=32K 就是几十到上百 GB。GPU 显存装不下。

### 1.2 共享前缀的机会

在真实应用里,大量请求共享前缀:

- **System prompt** — "You are a helpful assistant…" 每个 session 开头都一样
- **Few-shot examples** — 同一个 task 的所有请求共用同样的示例
- **Multi-turn 对话** — 第 N 轮的 prompt 是前 N-1 轮的累积
- **Tool definitions / RAG context** — 同一份长文档被多个 query 复用
- **Agent loop** — ReAct 的 thought-action-observation 历史不断追加

**如果两个请求前 N 个 token 完全一样,它们在第 N 个 token 之前的所有 K、V 计算结果就完全一样**(transformer 是 causal 的,前面的不看后面的)。可以重用。

这是 RadixAttention 的全部动机:**把 KV Cache 变成可以跨请求共享的结构,而不是请求结束就扔。**

---

## Part II — RadixAttention(HiCache 的基础)

### 2.1 数据结构:为什么是 Radix Tree

**Radix Tree(基数树)= 压缩前缀树(trie)**。普通 trie 一个节点存一个字符,radix tree 一个节点存一个**字符序列**。

举例:三个 prompt

```
A: [101, 203, 77, 55, 900, 12, 44, 88]
B: [101, 203, 77, 55, 900, 12, 44, 91]
C: [101, 203, 77, 55, 900, 12, 99, 11]
```

Radix tree 长这样:

```
root
 └── [101, 203, 77, 55, 900, 12]   ← 节点 1,共享前缀
      ├── [44]                     ← 节点 2
      │    ├── [88]                ← A 独有
      │    └── [91]                ← B 独有
      └── [99, 11]                 ← C 独有
```

**每个节点对应一段连续 token 的 KV cache 在 GPU memory 里的物理地址(indices)。**

为什么不用 hash?Hash 不能做"最长前缀匹配"。新请求来了你要快速找:"我历史上有没有缓存过这个前缀的某段?最长能匹配到哪?" Trie/radix 是 O(prompt_len) 时间。

### 2.2 关键源码字段

`python/sglang/srt/mem_cache/radix_cache.py` 里的 `TreeNode`:

```python
class TreeNode:
    children: dict        # 子节点
    parent: TreeNode
    key: RadixKey         # 这个节点对应的 token 序列
    value: torch.Tensor   # 对应的 KV cache 在 GPU 上的 indices
    lock_ref: int         # 引用计数 — 正在被 batch 用的节点不能 evict
```

**核心操作**:

- `match_prefix(tokens)` — 从 root 走,返回最长能匹配的前缀长度和节点
- `insert(tokens, kv_indices)` — 插入新前缀,自动分裂节点
- `lock_ref(node)` / `dec_lock_ref(node)` — 加/减引用,防止运行中被驱逐
- `evict(num_tokens)` — 显存不够时,按策略驱逐叶子节点

### 2.3 Eviction 策略

源码 `python/sglang/srt/mem_cache/radix_cache.py:55-64` 支持:

- **LRU**(默认)— Least Recently Used
- **LFU** — Least Frequently Used
- **SLRU** — Segmented LRU(冷区+热区,防止扫描污染)
- **priority** — 自定义优先级

驱逐只能驱叶子节点,且 `lock_ref > 0` 的跳过。父节点变成新叶子后再次入堆。

### 2.4 性能数据(SGLang 原始论文)

RadixAttention 在共享前缀场景下,相比 vLLM 等无前缀复用的系统,**吞吐最高 6.4×**,首 token 延迟显著降低(arxiv 2312.07104)。开销在 cache miss 时几乎为零,所以默认开启。

**RadixAttention 的根本局限:只用 GPU 显存。** 显存满了就 evict,evict 掉的下次就要重算。这个局限就是 HiCache 要解决的问题。

---

## Part III — HiCache 为什么必须出现

三个真实场景把 RadixAttention 逼到极限:

### 3.1 Agent 长上下文

Agentic Coding(Cursor、Claude Code 风格)单个 session 上下文常超 25K token,平均 8 轮对话(LMSYS 实测,Qwen3-Coder-480B)。每轮新增的 prompt 都建立在前 N-1 轮的累积之上。

**问题**:GPU 显存 evict 太快,大部分前缀活不到下一轮请求。命中率被压到 40%。

### 3.2 多实例集群

生产环境通常 N 个 SGLang 实例做负载均衡。Load balancer 把请求随机或按 round-robin 分发,**同一个 user 的连续请求可能落到不同实例**,每个实例都要重新 prefill 一遍 system prompt + RAG context。

**问题**:RadixAttention 只在单实例内有效,跨实例零共享。

### 3.3 PD 解耦部署

Prefill 和 Decode 拆到不同节点(Mooncake/Splitwise 范式)。Prefill 节点产出 KV,Decode 节点消费。如果中间没有共享存储,KV 要走传统传输,带宽和延迟都很差。

**HiCache 的回答**:把 KV cache 视为可以分层存储 + 跨节点共享的资源。

---

## Part IV — HiCache 总体架构

### 4.1 三层缓存类比 CPU L1/L2/L3

| 层级 | 介质 | 容量 | 延迟 | 范围 |
|------|------|------|------|------|
| **L1** | GPU HBM | 几十 GB(显存留给 KV 的部分) | < 1 μs | 单卡 |
| **L2** | CPU pinned DRAM | 100 GB ~ TB(可配) | ~10 μs(走 PCIe/NVLink-C2C) | 单机 |
| **L3** | RDMA 远端 / NVMe / 对象存储 | 几十 TB ~ PB | 50 μs ~ ms | 整个集群 |

**capacity ratio 配置**:`--hicache-ratio` 控制 L2 是 L1 的多少倍(默认必须 > 1)。例如 `--hicache-ratio 2` 表示 L2 = 2 × L1 大小。也可以用 `--hicache-size 30` 直接指定 30GB。

### 4.2 关键组件

```
┌─────────────────────────────────────────────────┐
│  Scheduler  (continuous batching, PD, PP, DP)   │
└──────────────┬──────────────────────────────────┘
               │ 每个 request 进来先做 match_prefix
               ▼
┌─────────────────────────────────────────────────┐
│  HiRadixCache  (extends RadixCache)             │
│   - HiRadixTree: 节点 + tier 元数据             │
│   - match_prefix() 跨 L1/L2/L3 找最长匹配      │
└──────────────┬──────────────────────────────────┘
               │ 触发数据移动
               ▼
┌─────────────────────────────────────────────────┐
│  HiCacheController                              │
│   - CacheOperation (L1↔L2)                      │
│   - StorageOperation (L2↔L3, prefetch/writeback)│
│   - TransferBuffer  (overlap I/O 与计算)        │
└──────┬───────────────────────────┬──────────────┘
       │                           │
       ▼                           ▼
┌──────────────┐            ┌──────────────────┐
│ HostKVCache  │            │ HiCacheStorage   │
│ (L2 pool)    │            │ (L3 backend ABC) │
│ pinned mem   │            │ - mooncake       │
│ MHA/MLA 变体  │            │ - hf3fs (3FS)   │
└──────────────┘            │ - nixl           │
                            │ - aibrix         │
                            │ - file (本地)    │
                            └──────────────────┘
```

源码定位:

- `python/sglang/srt/mem_cache/hiradix_cache.py:65-67` — HiRadixCache 主类
- `python/sglang/srt/mem_cache/memory_pool_host.py:154-166` — HostKVCache pinned 分配
- `python/sglang/srt/managers/cache_controller.py:117-135` — 控制器调度
- `python/sglang/srt/mem_cache/hicache_storage.py` — L3 backend 抽象

---

## Part V — HiRadixTree 数据结构详解

HiRadixTree 在 RadixTree 节点上加了**位置元数据**:

```
TreeNode (HiRadixTree 版):
  - key: token 序列
  - value: KV indices (L1 GPU 地址,如果在 L1)
  - host_value: KV indices (L2 host 地址,如果在 L2)
  - storage_hash: L3 上的 key/hash(如果在 L3)
  - tier_flags: bitmap, 标记当前节点 KV 同时存在哪些 tier
  - lock_ref: 同 RadixTree
  - last_access_time: for eviction
```

**一个节点可以同时存在于多个 tier。** 比如 write_through 策略下,一个新计算出的 KV 会立刻同时存在于 L1、L2、L3。

**Match 算法**:

```python
def match_prefix(tokens):
    # 1. 先在本地树(覆盖 L1+L2)做最长前缀匹配
    matched_local, node = local_match(tokens)

    # 2. 剩下的 token 去 L3 查 metadata
    if matched_local < len(tokens):
        remaining = tokens[matched_local:]
        l3_hit_len = storage.query(remaining)  # 只查 metadata,不传数据

    # 3. 如果 L3 命中长度 > 阈值(默认 256),触发 prefetch
    if l3_hit_len > prefetch_threshold:
        controller.prefetch(remaining[:l3_hit_len])

    return matched_local + l3_hit_len, node
```

**关键工程点**:L3 query 只问"你有没有,有多长",**不传数据**。数据传输由 prefetch 异步触发。这是控制流和数据流分离的标准设计。

---

## Part VI — 让你成为专家的 7 个核心优化

> 前面是"是什么"。下面是"为什么这么设计、不这么设计会怎样"。
> 这部分是面试和架构评审的真正分水岭。

### 6.1 GPU-assisted I/O kernels

**问题**:`cudaMemcpyAsync` 走 PCIe DMA engine,但对于"很多个小段、不连续"的 KV 拷贝,DMA 调度开销大,实际带宽利用率低。

**解法**:写专用 CUDA kernel 用 GPU SM 来搬数据。SM 数量多,可以同时启动大量 thread 并行做 strided copy,把零散块聚合成大传输。

**收益**:LMSYS 官方数据 **3× 吞吐**,相比裸 `cudaMemcpyAsync`。

**配置**:`--hicache-io-backend {kernel, direct}`。`kernel` 是 GPU-assisted(默认推荐),`direct` 是 cudaMemcpyAsync 兜底。

### 6.2 Layer-wise Compute-Transfer Overlap

**问题**:模型有 N 层(70B 模型典型 80 层)。如果等所有层的 KV 都从 L2 拉到 L1 才开始算,延迟 = 传输时间 + 计算时间。

**解法**:**逐层流水线**。第 1 层的 KV 一传完就开始算第 1 层的 attention,同时后台传第 2 层的 KV;第 2 层算完时第 3 层 KV 已经到位……

```
时间轴 →
传输:  [L1传][L2传][L3传][L4传] ...
计算:        [L1算][L2算][L3算][L4算] ...
                  ↑ 计算时间吃掉了传输时间
```

**结果**:有效延迟 ≈ max(单层传输, 单层计算) × N,而不是 (传输 + 计算) × N。在大模型上传输几乎完全被计算 hide。

源码:`python/sglang/srt/mem_cache/memory_pool_host.py` 里的 layer-wise transfer kernel。

### 6.3 Memory Layout:layer_first vs page_first

L2 上的 KV 怎么排?两种布局:

**layer_first**(默认):

```
[layer_0 的所有 token 的 K,V] [layer_1 的所有 token 的 K,V] ...
```

- 优点:跟 GPU 上的 layout 一致,L1↔L2 拷贝是连续的,GPU kernel 友好
- 缺点:L2↔L3 传输某段连续 token 时,要在每一层各取一片,strided 访问,L3 backend(尤其 RDMA)效率低

**page_first**:

```
[token_page_0 在 layer_0 的 K,V][token_page_0 在 layer_1 的 K,V]...[token_page_1 在 layer_0 的 K,V]...
```

- 优点:连续 token 的所有层 KV 是物理连续的,L2→L3 一次大块 RDMA 传输,zero-copy 友好
- 缺点:L1↔L2 时需要 layout 转换,有 transpose 开销(就是上面 PR #8651 解决的事)

**配置**:`--hicache-mem-layout {layer_first, page_first, page_first_direct}`。

**经验法则**:

- 只用 L1+L2(无外部存储):`layer_first`
- 启用 L3(尤其 Mooncake/3FS):`page_first` 或 `page_first_direct`,zero-copy 收益大

### 6.4 Write-back 策略

新算出的 KV 怎么往下层写?三种(`--hicache-write-policy`):

| 策略 | 行为 | 适用场景 | 代价 |
|------|------|---------|------|
| **write_through** | 每次新 KV 立刻同步写到下一级 | 带宽充裕、希望最高命中率 | I/O 流量大 |
| **write_through_selective** | 只把"被访问过 N 次以上"的热数据写下去 | 通用生产环境(推荐默认) | 需要计数 |
| **write_back** | 仅当上层要 evict 时,才写到下一级 | 存储紧张、上层利用率优先 | 命中数据可能晚到 L3 |

**关键认知**:write 是异步并行的。Granularity 是 HiRadixTree node(一段连续 token),不是单 token。

### 6.5 Prefetch 策略

L3 命中长度超过阈值(默认 256 tokens)时,触发预取(`--hicache-storage-prefetch-policy`):

| 策略 | 行为 | 适用 |
|------|------|------|
| **best_effort** | GPU 一空就开始 prefill,不等 prefetch 完 | TTFT 极敏感 |
| **wait_complete** | 等所有 prefetch 完才开始 prefill | 命中率最大化 |
| **timeout** | 等到 `prefetch_timeout_base + prefetch_timeout_per_ki_token × N/1024` ms 自动放弃 | 平衡(推荐生产) |

**为什么 L3 prefetch 单独设计**(L1↔L2 直接 layer-wise overlap 就够):

- L3 延迟比 L2 高一个量级(50 μs ~ ms vs 10 μs),且抖动大
- L3 是网络/磁盘,失败率高,需要超时保护
- L3 命中部分往往很长(整段历史 context),比逐层 overlap 更适合提前批量拉

### 6.6 MLA 写回去重

**MHA(Multi-Head Attention)** 在 TP=N 下,每个 rank 持有 1/N 的 KV head。所以每个 rank 都要独立 write-back 自己那份。

**MLA(Multi-head Latent Attention,DeepSeek-V2/V3 的设计)** 把 KV 压缩到 latent vector,**所有 TP rank 持有完全相同的 KV**。如果每个 rank 都写,就重复 N 倍。

**HiCache 的优化**:MLA 模型下只有 rank 0 做 write-back,其他 rank 跳过。直接省 N 倍 L3 写流量。

这个优化在 DeepSeek-R1 / V3 / V3.2 系列上意义巨大,因为它们是当前主流大 MLA 模型。

### 6.7 Zero-copy + RDMA(L2→L3 with Mooncake/NIXL)

L2→L3 走 Mooncake 时:

- **Zero-copy**:HiCache 把 L2 pinned memory 的地址和大小直接传给 Mooncake,Mooncake 通过 RDMA 直接从 pinned 内存搬到远端,**不经过额外内存拷贝**。
- **多 NIC 聚合**:Mooncake 利用单机多张 RDMA NIC(`MOONCAKE_DEVICE="mlx5_0,mlx5_1,..."`),线性扩展带宽。
- **GPU Direct RDMA**(部分场景):某些 backend 可以从 GPU 显存直接发起 RDMA,跳过 host 内存。

带宽对比量级感(典型生产配置):

- PCIe Gen4 x16:~32 GB/s 单向
- 单 NIC 200Gb RDMA:~25 GB/s
- 4 NIC 聚合:~100 GB/s
- NVMe SSD:~7 GB/s 单盘

L3 backend 选型本质就是在选你的"带宽 × 容量 × 延迟"组合。

---

## Part VII — L3 Backend 选型矩阵

`--hicache-storage-backend` 可选:`file, mooncake, hf3fs, nixl, aibrix, dynamic`。

| Backend | 介质 | 跨实例共享 | 部署复杂度 | 典型延迟 | 主要场景 |
|---------|------|----------|----------|---------|---------|
| **file** | 本地文件系统 | ❌ | 极低(开箱即用) | ms 级 | 单机 demo / 小流量 / debug |
| **Mooncake** | 分布式 RDMA DRAM/SSD pool | ✅ | 中(需 metadata server + Transfer Engine) | 50~100 μs | **生产首选**,Kimi 同款 |
| **HF3FS (DeepSeek 3FS)** | 分布式存储,K8s native | ✅ | 中高(需 K8s operator) | 100 μs ~ ms | 大规模历史 KV 持久化 |
| **NIXL** | 统一 API → 3FS / GDS / S3 | ✅ | 中(NVIDIA 维护) | 取决于 plugin | 混合多种存储 |
| **AIBrix** | 生产级 KV offload framework | ✅ | 中(字节开源) | 几十 μs ~ ms | 生产环境的 cross-engine reuse |

**选型经验法则**:

- 你只想试一下:`file`
- 生产、Kimi 路线、有 RDMA 集群:`mooncake`
- 已有 3FS 部署:`hf3fs` 或 `nixl`
- 混合云、要 S3 兜底:`nixl`
- 字节内部生态:`aibrix`(字节背书)

**额外选项**:`--enable-lmcache` 可以换成 LMCache 全替代方案。LMCache 是另一个独立的分层 KV cache 项目,设计哲学和 HiCache 不同(更偏跨引擎、跨框架),适合需要在 vLLM 和 SGLang 之间共享 KV 的多引擎场景。**HiCache 和 LMCache 是互斥替代关系,不是叠加。**

---

## Part VIII — 端到端请求流程(完整时序)

一个新请求从进来到吐出第一个 token,完整路径:

```
[1] Request 到达
     ├─ tokens = tokenize(prompt)
     │
[2] HiRadixCache.match_prefix(tokens)
     ├─ 步骤 a: local_match → 找到 L1+L2 命中长度 m1
     ├─ 步骤 b: storage.query(tokens[m1:]) → L3 命中长度 m2
     │
[3] 决策分支
     ├─ if m2 > prefetch_threshold:
     │     controller.prefetch(L3 段) → 异步从 L3 拉到 L2
     │
[4] Scheduler 把请求加入 batch
     ├─ 计算需要 prefill 的 token 数 = len(tokens) - m1 - m2
     │   (m2 部分预期能 prefetch 上来,不重算)
     │
[5] Prefill 开始
     ├─ if 命中段在 L2: layer-wise 把 KV 从 L2 流水线传到 L1
     ├─ if 命中段还在 L3: 按 prefetch policy 等或不等
     │     - best_effort: 不等,L3 段重算
     │     - wait_complete: 等,延迟换命中
     │     - timeout: 等到超时
     ├─ 不命中段在 GPU 上做 prefill,产出新 KV
     │
[6] Write-back(异步,与 decode 并行)
     ├─ 按 write-policy 把新算的 KV 同步到 L2 / L3
     ├─ HiRadixTree 节点 tier_flags 更新
     │
[7] Decode 开始,吐第一个 token = TTFT 完成
     │
[8] Decode 持续,直到 EOS / max_tokens
     │
[9] 请求结束
     ├─ dec_lock_ref(节点),允许后续 evict
     ├─ KV 留在 cache 里供后续请求复用
```

**理解这条时序你就理解了 HiCache 的全部决策点**。每一步都对应了第 6 节的某个优化:

| 时序步 | 对应优化 |
|-------|---------|
| [2b] L3 query | 控制流/数据流分离 |
| [3] prefetch 决策 | 6.5 prefetch 策略 |
| [5] L2→L1 传输 | 6.1 GPU kernel + 6.2 layer-wise |
| [5] memory layout | 6.3 layer_first vs page_first |
| [6] write-back | 6.4 write 策略 + 6.6 MLA 去重 |
| [6] L2→L3 写 | 6.7 zero-copy + RDMA |

---

## Part IX — 配置实战

### 9.1 关键 CLI 参数全集

```bash
# 启用
--enable-hierarchical-cache

# 容量
--hicache-ratio 2                       # L2 = L1 × N(必须 > 1)
--hicache-size 30                       # 或直接指定 GB(覆盖 ratio)

# 内存布局
--hicache-mem-layout layer_first        # 仅 L1+L2 用
--hicache-mem-layout page_first         # 启用 L3 用
--hicache-mem-layout page_first_direct  # zero-copy 优化版

# I/O 后端
--hicache-io-backend kernel             # GPU kernel(推荐,3×)
--hicache-io-backend direct             # cudaMemcpyAsync 兜底

# 写策略
--hicache-write-policy write_through
--hicache-write-policy write_through_selective   # 推荐
--hicache-write-policy write_back

# L3 后端
--hicache-storage-backend file
--hicache-storage-backend mooncake
--hicache-storage-backend hf3fs
--hicache-storage-backend nixl
--hicache-storage-backend aibrix

# Prefetch 策略
--hicache-storage-prefetch-policy best_effort
--hicache-storage-prefetch-policy wait_complete
--hicache-storage-prefetch-policy timeout         # 推荐

# 替代方案
--enable-lmcache                        # 用 LMCache 替代 HiCache
```

### 9.2 三个典型配置 preset

**(a) 单机 + 长上下文 agent(无 L3)**

```bash
--enable-hierarchical-cache \
  --hicache-ratio 4 \
  --hicache-mem-layout layer_first \
  --hicache-io-backend kernel
```

适用:Cursor/Copilot 风格本地 agent 服务,单机够用,要尽可能多缓存历史。

**(b) 多实例集群 + Mooncake L3(生产典型)**

```bash
export MOONCAKE_MASTER=10.x.x.x:50051
export MOONCAKE_PROTOCOL="rdma"
export MOONCAKE_DEVICE="mlx5_0,mlx5_1,mlx5_2,mlx5_3"
export MOONCAKE_GLOBAL_SEGMENT_SIZE=25769803776   # 25 GB

python -m sglang.launch_server \
  --model-path <path> \
  --tp 8 --dp 8 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-mem-layout page_first \
  --hicache-io-backend kernel \
  --hicache-storage-backend mooncake \
  --hicache-write-policy write_through_selective \
  --hicache-storage-prefetch-policy timeout
```

适用:多卡多机,跨实例共享前缀的 QA / 客服 / RAG 场景。

**(c) DeepSeek V3.2-Exp(MLA + sparse + HiCache)**

```bash
python -m sglang.launch_server \
  --model-path /path/to/DeepSeek-V3.2-Exp \
  --tp 8 --dp 8 --enable-dp-attention \
  --chunked-prefill-size 8192 --max-prefill-tokens 16384 \
  --max-total-tokens 262144 \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-storage-backend mooncake \
  --hicache-write-policy write_through \
  --hicache-storage-prefetch-policy timeout
```

注意:这个配置在 2025 年 11 月还有 `transfer_kv_all_layer_direct_lf_pf` kernel 维度不匹配的 bug(GitHub issue #13431),关注后续修复版本。这是 sparse + HiCache 集成还在演进的信号。

---

## Part X — 性能基线(量化)

所有数据来自 LMSYS 官方 blog 2025-09-10 + Mooncake 文档:

| 场景 | 模型 | 指标 | 收益 |
|------|------|------|------|
| Long-context bench | — | 吞吐 | **6×** |
| Long-context bench | — | TTFT | **降 80%** |
| Multi-turn QA + Mooncake | DeepSeek-R1-671B (PD 解耦) | TTFT(命中时) | **降 84%** |
| Agentic Coding + 3FS | Qwen3-Coder-480B (25K tok / 8 turns) | TTFT 平均 | **降 56%** |
| Agentic Coding + 3FS | 同上 | 吞吐 | **2×** |
| Agentic Coding + 3FS | 同上 | 命中率 | **40% → 80%** |
| GPU I/O kernel vs cudaMemcpyAsync | — | CPU↔GPU 吞吐 | **3×** |

**这些数字的隐含信号**:

- 收益的 **可信区间**:在前缀复用密集的场景下 2× ~ 6×,在前缀复用稀疏的场景接近无收益(命中率决定)
- **TTFT 降幅 > 吞吐提升**:意味着 HiCache 主要是省 prefill,而 prefill 是 TTFT 的主导项
- **命中率从 40% 翻到 80% 是分水岭**:这背后是 L3 把"被 evict 的历史"接住了,这是 HiCache 相对于纯 RadixAttention 最大的差异化

---

## Part XI — 与生态对比

### 11.1 vs LMCache

| 维度 | HiCache | LMCache |
|------|---------|---------|
| 归属 | SGLang 原生 | 独立项目 |
| 跨引擎 | SGLang only | 支持 vLLM、SGLang |
| 与 RadixAttention 集成 | 深度(同一棵树) | 外接 |
| 生产成熟度(2025 末) | 字节、月之暗面、DeepSeek 验证 | 企业级用户 |
| 部署 | SGLang 内置 | 独立 service |

**SGLang 文档明确说**:`--enable-lmcache` 是 alternative,不是叠加。LMCache 提供了 HiCache 之外的另一种实现路径。

### 11.2 vs Strata(学术对手)

Strata(arxiv 2508.18572,2025-08)是学术界最近的同类系统,主打"长上下文 hierarchical context caching"。它的论文里就用 SGLang-HiCache 做基线,page_size=32。Strata 的差异化在于更细粒度的 prefetch 调度和 cross-tier eviction。

**这是 HiCache 后续会借鉴或被借鉴的方向。** 学术上这条线还在快速演进。

### 11.3 vs vLLM PagedAttention(根本路径差异)

| 维度 | SGLang HiCache | vLLM PagedAttention |
|------|----------------|---------------------|
| KV 组织 | Radix Tree(token 序列结构化) | Page Table(固定大小 page,无结构) |
| 共享单位 | 任意长度的连续 token 段(节点) | 固定 page(典型 16 token) |
| 跨请求共享 | 自动,基于 token 序列匹配 | Prefix caching,后期补的能力 |
| 分层缓存 | 原生支持 L1/L2/L3 | 主要靠 LMCache 等外接 |

根本差异:**SGLang 一开始就把 KV 视为"有结构的、可跨请求共享的资源",vLLM 一开始把 KV 视为"非结构化、按 page 管理的内存"**。这个设计差异决定了 SGLang 在 prefix-heavy 场景天然占优,vLLM 在朴素 batching 场景成熟稳定。

---

## Part XII — 局限与开放问题(顶级专家视角)

如果你想成为这条线上的专家,这些是你应该知道并有看法的开放问题。

### 12.1 Hybrid / Sparse 模型支持

**问题**:Qwen3-Next(linear + full attention 混合)、DeepSeek V3.2-Exp(sparse attention)、GPT-OSS(SWA 滑动窗口)等新架构,KV 不再是"每层每 token 都有一份完整 K、V"的统一形态:

- Linear attention 没有显式 KV
- Sparse attention 只关注部分 token
- SWA 只缓存窗口内的 token

**HiCache 的应对**:GitHub issue #12826(2025-11)明确把 sparse + HiCache 列为优先方向,目标是"selective KV cache loading to GPU memory, reducing memory footprint and allowing larger batch sizes"。但工程实现还在演进,issue #13431 就是这条路上的实际 bug。

**你应该有的看法**:这是 HiCache 未来 6-12 个月最重要的演进方向。新架构层出不穷,谁能先做到"统一抽象 + 高性能 sparse KV 分层缓存",谁就是下一个标准。

### 12.2 Workflow-aware 调度(KVFlow 路线)

**问题**:HiCache 的 eviction 是基于 LRU/LFU,prefetch 是基于 token 命中长度。这些都是"通用"策略,**不知道上层 agent workflow 的语义**。

**KVFlow**(arxiv 2507.07400)的洞察:agent 系统知道"下一步 agent A 会被调用,它会用前缀 P",可以**主动预取**和**workflow-aware eviction**。报告 1.83× speedup over SGLang HiCache。

**这是个分层架构的天然空白**:HiCache 在 inference engine 内部,看不到上层 agent 拓扑。要么 agent framework 主动告诉 inference engine 调度信息(API 扩展),要么 HiCache 自己学会预测 workflow。

### 12.3 Multi-tenant 隔离与安全

**没解决的问题**:

- 跨 tenant 的 KV 共享是否泄露 prompt?(common system prompt 被命中是否能被推测出来?)
- 配额(单个 tenant 占多少 L2/L3 容量)?
- 优先级(VIP tenant 的 KV 是否更难被 evict)?

生产环境如果是多租户 SaaS,这些都是现实问题。HiCache 当前没有正式答案。

### 12.4 写一致性

`write_back` 策略下,L1 的最新 KV 没及时写到 L3。如果实例宕机,这些 KV 丢失。下次同一前缀的请求只能重算。**这是 cache 的本质 trade-off,不是 bug**,但要在 SLA 设计时考虑。

`write_through` 把流量打满 L3,可能成为瓶颈。生产里通常是 `write_through_selective`(热数据写),但 "热" 的阈值怎么调,没有通用最优解,要按 workload 实测。

### 12.5 Page size 与 fragmentation

`--page-size` 越大,L2/L3 传输效率越高(大块更友好),但 cache 内部碎片越大(尾部不足一个 page 的 token 被丢弃,见 RadixCache 源码 line 110-114)。

DeepSeek V3.2 配置里用 `--page-size 64` 是大模型的取舍。小模型可能 16 更优。**没有银弹,要 profile**。

---

## Part XIII — 顶级专家学习路径

读到这,你已经超过 90% 了解 HiCache 的人。要进入剩下的 10%,做这些事:

### 13.1 必读源码(按推荐顺序)

```
python/sglang/srt/mem_cache/
  ├── radix_cache.py            # 先看:基础树结构和 LRU
  ├── hiradix_cache.py          # 核心:三层扩展
  ├── memory_pool.py            # L1 GPU pool
  ├── memory_pool_host.py       # L2 host pool + layer-wise transfer kernel
  ├── hicache_storage.py        # L3 抽象基类
  └── (各 backend 子目录)        # mooncake/, hf3fs/, nixl/

python/sglang/srt/managers/
  ├── cache_controller.py       # CacheOperation / StorageOperation 调度
  └── scheduler.py              # 整体 batching 与 cache 的交互
```

**建议**:用一个 toy prompt(`"Hello, world."` 重复 100 次)跑通,加 print/profiler,看每个节点何时被创建、何时被 evict、何时下沉到 L2/L3。**亲手追一次完整生命周期是最快的理解方式**。

### 13.2 必读论文清单(按时间)

1. **SGLang / RadixAttention**(arxiv 2312.07104)— 一切的源头
2. **vLLM / PagedAttention**(arxiv 2309.06180)— 对照系
3. **CachedAttention**(Gao 2024)— 早期分层 KV 工作
4. **Mooncake**(arxiv 2407.00079)— L3 backend 思想源
5. **LMCache**(2024)— 替代路线
6. **KVFlow**(arxiv 2507.07400)— workflow-aware 优化
7. **Strata**(arxiv 2508.18572)— 最新学术对手
8. **DeepSeek V3 / V3.2**(技术报告)— MLA、sparse attention 的 KV 形态
9. **PALU**(低秩 KV 压缩)、**DDTree**、**FlashInfer** — 周边技术,理解 KV 优化全景

### 13.3 必跑实验

- 跑通 LMSYS blog 末尾的 reproducibility benchmark(long-context + multi-turn)
- 用 `nsys` profile 一次完整 prefill,看 layer-wise overlap 真的有没有 hide 传输
- 改写一个 mini L3 backend(比如基于 Redis),实现 HiCacheStorage ABC 的 4-5 个方法,跑通
- 用 Mooncake + 2 个 SGLang 实例做 cross-instance hit 实验,验证集群级共享的实际收益

### 13.4 贡献切入点

GitHub `sgl-project/sglang` 上的 HiCache-related issue 列表是最好的入口。当前(2025 末)活跃方向:

- Hybrid model HiCache 支持(#12826)
- Sparse + HiCache(DSv3.2 集成,#13431 这种 bug 修复)
- 新 L3 backend 适配
- write-policy / prefetch-policy 的自适应调整(目前都是静态参数)

如果你目标是成为"被这个社区记住"的专家,贡献一个新的 L3 backend 或一个自适应调度算法,是最快的路径。

### 13.5 你已经掌握的(检查清单)

- [ ] 能 5 句话讲清 HiCache 是什么、解决什么、关键创新
- [ ] 能画出 L1/L2/L3 + HiRadixTree 架构图
- [ ] 能解释 layer_first vs page_first 的 trade-off 和何时选哪个
- [ ] 能解释三种 write policy 和三种 prefetch policy 的工程权衡
- [ ] 能讲清 MLA write-back 去重为什么对 DeepSeek 系列特别重要
- [ ] 能选出给定场景的 L3 backend 并解释理由
- [ ] 能评价 HiCache vs LMCache vs vLLM PagedAttention 的设计差异
- [ ] 能指出 HiCache 当前 3 个最大的开放问题
- [ ] 能在 sglang 源码里 5 分钟内定位任何上面提到的概念

打勾 6 个以上,你在生产环境就已经能 own 这块了。9 个全打勾,你可以去面 Inference Infra 架构师。

---

## 附录 A:术语速查

| 术语 | 解释 |
|------|------|
| **TTFT** | Time To First Token,首字延迟,prefill 阶段的核心指标 |
| **KV Cache** | 每层 attention 的 K、V 中间结果,decode 阶段必须访问 |
| **Prefill** | 处理整个 prompt、生成所有 K、V 的阶段,O(n²) |
| **Decode** | 一个一个 token 生成的阶段,O(n) per step |
| **PD Disaggregation** | Prefill 和 Decode 拆到不同节点 |
| **MHA** | Multi-Head Attention,标准多头 |
| **MLA** | Multi-head Latent Attention,DeepSeek V2/V3 的低秩压缩 KV 设计 |
| **RDMA** | Remote Direct Memory Access,绕过 CPU 的网络传输 |
| **GDS** | GPU Direct Storage,GPU 显存直接读写存储 |
| **TP / DP / PP** | Tensor / Data / Pipeline Parallel |
| **page** | KV cache 的最小连续分配单元 |
| **lock_ref** | 节点引用计数,防止运行中被 evict |

## 附录 B:参考链接(全部权威源)

- LMSYS HiCache 官方 blog:https://lmsys.org/blog/2025-09-10-sglang-hicache/
- SGLang HiCache 设计文档:https://docs.sglang.io/advanced_features/hicache_design.html
- Mooncake × SGLang 设计:https://kvcache-ai.github.io/Mooncake/design/hicache-design.html
- Mooncake 集成指南:https://kvcache-ai.github.io/Mooncake/getting_started/examples/sglang-integration/hicache-integration-v1.html
- SGLang 源码(DeepWiki):https://deepwiki.com/sgl-project/sglang
- RadixAttention 原始 blog:https://lmsys.org/blog/2024-01-17-sglang/
- SGLang 原论文:https://arxiv.org/abs/2312.07104
- KVFlow:https://arxiv.org/abs/2507.07400
- Strata:https://arxiv.org/abs/2508.18572
- Hybrid/Sparse roadmap issue:https://github.com/sgl-project/sglang/issues/12826

---

**结束。** 如果飞机时间还有,从 Part VI 重读一遍,把每个优化的"为什么不这么设计会怎样"在脑子里推一遍。这是从"知道"到"会用"的最短路径。
