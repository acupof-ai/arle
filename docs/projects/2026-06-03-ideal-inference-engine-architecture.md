# 高效・高性能・可演进推理引擎的理想架构

**类型:** 架构北极星(survey + ideal design)。指导 ARLE 后端 seam 重构。
**分支:** `arch/ideal-inference-engine`
**关联:** [`2026-06-03-backend-seam-redesign.md`](2026-06-03-backend-seam-redesign.md)(ARLE 现状审计 + 迁移序列)
**方法:** `arle-upstream-runtime-scan`。SGLang 读源(`sgl-project/sglang@3e681d7`),
vLLM/Dynamo 读官方文档,其余读架构知识——**全部 hypothesis-grade**,落地 ARLE 前以本地
bench/test license-or-kill。来源见末节。

---

## 0. 心智模型:推理引擎是一条数据流水线,不是一个函数

一个请求的生命周期是一条 **stage 流水线**:

```
ingress → tokenize → schedule → forward(layer stages) → sample → detokenize → stream
  (CPU)     (CPU)      (CPU)         (GPU)              (GPU/CPU)   (CPU)      (CPU/net)
```

整套架构由三个**根本张力**塑形,所有主流引擎的设计都是对它们的回答:

1. **CPU↔GPU 必须重叠。** tokenize/schedule/sample-prep/detokenize 是 CPU 串行小活;
   forward 是 GPU 大活。若 CPU stage 与 GPU stage 串行,GPU 在每一步之间空转。
   → 解法:**overlap scheduler**(CPU 调度第 N+1 步,与 GPU 跑第 N 步并行)。
2. **GPU 要静态形状 + 大批。** kernel launch 开销、CUDA graph、cuBLAS 选优都偏好固定 shape
   和满批。但请求是动态到达、变长的。→ 解法:**连续批 + chunked prefill + 静态形状
   graph(padding 到桶)**。
3. **设备/并行在演进,调度逻辑不应跟着重写。** NVIDIA/AMD/Intel/NPU,TP/PP/EP/DP,
   disaggregation——这些是**正交的变化轴**。→ 解法:**后端无关的 engine core +
   窄契约 seam**,把设备/并行/kernel 关进可插拔层。

理想架构 = 把这三个回答**分层**固化下来:数据流是流水线,执行是分层 seam,变化轴是插件。

---

## 1. 主流引擎调研

| 引擎 | scheduler↔backend 分离 | 连续批 / chunked prefill | TP / PP / EP / DP | CPU↔GPU overlap | CUDA graph | KV 内存 | 设备可移植 | disagg P/D |
|---|---|---|---|---|---|---|---|---|
| **vLLM V1** | ✅ 进程级(API ‖ EngineCore ‖ Worker) | ✅ token-budget,无 prefill/decode 相位区分 | TP+PP(+EP MoE) | ✅ async,zero-overhead | ✅ piecewise + torch.compile | paged + prefix cache | platform 抽象:CUDA/ROCm/TPU/XPU/CPU | 支持(KV connector) |
| **SGLang** | ✅ Scheduler / ModelRunner / AttentionBackend | ✅ chunked,RadixAttention | TP+PP(`scheduler_pp_mixin`)+EP(DeepEP)+**DP-attention** | ✅ overlap scheduler(`FutureMap`) | ✅ 标准 + piecewise + breakable + CPU graph | paged + **radix** + 分层 | `AttentionBackend` 24 实现:NV/**AMD aiter·hip·wave**/Intel xpu·amx/NPU | 支持 |
| **TensorRT-LLM** | 部分(Executor API + 编译引擎) | ✅ in-flight batching | TP+PP+EP | ✅ | ✅ | paged + reuse | **仅 NVIDIA**(AOT 编译,per-shape profile) | 支持 |
| **NVIDIA Dynamo** | ✅✅ **编排层之上**(后端=vLLM/SGLang/TRT-LLM) | 委托给后端 | 委托 + 多节点扩缩 | — | — | **KVBM** + NIXL 点对点 KV 搬运 | engine-agnostic | ✅✅ **核心卖点**(PrefillRouter + KV-aware 路由) |
| **DeepSpeed-FastGen** | 部分 | ✅ **Dynamic SplitFuse**(chunked prefill 起源之一) | TP | 部分 | ✅ | blocked KV | NVIDIA | — |
| **Mooncake** | KV-中心架构 | ✅ | — | — | — | **KVStore 池**(分离 prefill/decode/KV 三池) | — | ✅✅ disagg 先驱 |
| **LMDeploy(TurboMind)** | 部分 | ✅ persistent batch | TP | ✅ | ✅ | blocked KV | NVIDIA 为主 | — |
| **HF TGI** | 较弱 | ✅ | TP | 部分 | 部分 | paged(FlashAttn) | NV/部分 ROCm | — |
| **llama.cpp** | ✅ **ggml-backend 接口** | 弱(批量为主) | 有限 | — | 部分 | 简单 KV | ✅✅ CUDA/Metal/Vulkan/ROCm/CPU/SYCL | — |
| **MLC-LLM / TVM** | 编译式 | ✅ | 有限 | — | ✅(codegen) | paged | ✅✅ **编译器跨硬件**(含 WebGPU) | — |

**每个引擎的架构一句话本质:**

- **vLLM V1** = 把 V0 的单进程拆成 *API 进程 ‖ EngineCore 进程 ‖ Worker*,scheduler 退化成
  `{request_id: num_tokens}` 的 token 预算字典(无相位),换来 CPU 活与 GPU 核心循环的最大重叠。
- **SGLang** = 设备无关 Scheduler 产 `ForwardBatch`,`AttentionBackend` ABC 是窄 kernel seam
  (24 实现横跨所有硬件),`FutureMap` 让调度跑在 GPU 前面;DP-attention+EP 是 MoE 的标准答案。
- **TRT-LLM** = 极致单卡/单厂性能,代价是 AOT 编译 + NVIDIA 锁定,可移植性最差。
- **Dynamo** = 不造引擎,造**引擎之上的数据流编排层**:KV-aware 路由 + KVBM + NIXL 把 prefill/decode
  拆成独立 worker 池,KV 在 VRAM 间点对点搬。这是"数据流心智"的极致表达。
- **llama.cpp / MLC** = 可移植性的两条路:**手写 backend 接口(ggml)** vs **编译器 codegen(TVM)**。

---

## 2. 全行业收敛的六个模式(你抄就对了)

跨这 10 个引擎,设计已经收敛。这六条是"高效・高性能・可演进"的最大公约数:

1. **进程/循环三分:Frontend ‖ EngineCore ‖ Workers。** API/tokenize/detokenize(CPU,GIL/锁敏感)
   独立于 scheduler+execute 的核心循环(vLLM V1 进程分离,SGLang tokenizer_manager 分离)。
   单进程把 CPU 活和 GPU 循环耦在一起 = GPU 饿死。
2. **后端无关 scheduler + 逻辑 ForwardBatch IR;设备相关执行在窄 seam 之下。**
   scheduler 产数据(plan),不碰 kernel。设备多样性塞进 `AttentionBackend`+`KVCache`(SGLang)
   / platform+attn backend(vLLM)。**一个 scheduler 服务所有硬件。**
3. **token 预算连续批,取消 prefill/decode 相位;chunked prefill 统一二者。**
   不再"先 prefill 队列再 decode 队列",而是每步给一个 token 预算,prefill chunk 与 decode 行
   混在同一 forward(vLLM token-budget,SGLang chunked,DeepSpeed SplitFuse)。
4. **Overlap scheduler:CPU 调度第 N+1 步 ‖ GPU 跑第 N 步,用 future-token 缓冲解耦。**
   SGLang `FutureMap`:scheduler 为第 N 步的输出**发布占位 future token**,第 N+1 步直接从
   `output_tokens_buf` 索引读取,**不等结果回 host**。vLLM 称 "zero-overhead"。
   这是把 §0 张力①落地的关键机制——也是 CPU/GPU 天然并行的本体。
5. **CUDA graph = padding 到静态桶 + metadata 分两段。** capture 时形状固定(batch 桶
   1/2/4/.../max,padding 补齐);per-iter 动态 metadata 走 **out-of-graph**(host op,
   `.item()`/`.cpu()` 在这),可录制的静态 GPU op 走 **in-graph**。SGLang 把这做成
   `init_forward_metadata_{out,in}_graph` 两段;vLLM 用 **piecewise graph**(只把
   attention 之外的部分录进 graph,attention 留动态)+ torch.compile。
6. **KV-中心内存:paged pool + radix 前缀 + 分层/offload +(新)disagg KV 搬运。**
   KV cache 是显存大头也是复用金矿。paged(碎片化分配)+ radix(跨请求前缀共享,SGLang)+
   host/disk 分层 + disaggregation(Dynamo KVBM/NIXL、Mooncake)把 KV 当一等公民管理。

---

## 3. 并行轴心智:DP / TP / EP / PP 各在哪一层

这是"良好支持 DP/EP/TP"的前提——它们**不是一个开关,是四个正交轴,在不同层**。理想架构必须
让它们**正交组合**(TP×EP×DP×PP),而不是互相纠缠。

| 轴 | 切什么 | 通信原语 | 住在哪层 | 性质 |
|---|---|---|---|---|
| **TP** tensor parallel | 层内:权重矩阵按列/行切到多卡 | 每层 all-reduce / reduce-scatter | 模型 linear/attn 层 + `Communicator` | 延迟敏感,需 NVLink 级互联;KV 在 TP 组内**复制** |
| **PP** pipeline parallel | 层间:把 L 层切成 S 段,各段一卡 | 段间 P2P send/recv(activations) | **scheduler 编排 microbatch** + 段间通信 | 吞吐导向;引入 bubble,scheduler 必须 microbatch 化(SGLang `event_loop_pp`:async-send/sync-recv 压 bubble) |
| **EP** expert parallel | MoE:experts 按卡分,token 路由到 expert 所在卡 | all-to-all dispatch/combine(DeepEP) | MoE 层 + all-to-all `Communicator` | MoE 专属;负载不均是主敌 |
| **DP** data parallel | 复制整引擎 / **复制 attention**(DP-attention) | 副本间无(或 EP 组内) | engine 副本 / attention 层 | 吞吐;**DP-attention 是 MoE 关键**:attention 走 DP(避免 KV 跨 TP 复制),FFN/experts 走 EP |

**最重要的一条(DeepSeek/Qwen-MoE 的标准答案,SGLang `enable_dp_attention`):**
大 MoE 上,**attention = DP(每 rank 独立 KV,不跨 TP 复制),experts = EP(all-to-all)**。
纯 TP 会把 KV cache 在每个 TP rank 上复制 N 份,显存爆炸;DP-attention + EP 让 KV 只存一份/rank,
是 ARLE 跑 Qwen3.6-MoE 必须走的路。

**架构含义:scheduler 只在 plan 层面感知并行**(知道有几个 rank/microbatch、token 怎么路由),
**collectives(all-reduce/all-to-all/P2P)在 seam 之下的 `Communicator`**。四轴组合 = 一个
**topology/mesh 描述符**(`{tp, pp, ep, dp}` + rank 映射)注入 executor,scheduler 据此产
microbatch 化的 plan。

---

## 4. 理想架构

### 4.1 分层视图(谁依赖谁)

```
┌──────────────────────────────────────────────────────────────┐
│ Frontend          HTTP/OpenAI · tokenize · detokenize · stream │  CPU, async, 独立进程/线程池
│                   (与核心循环解耦,GIL/锁不污染 GPU 循环)        │
└───────────────────────────┬──────────────────────────────────┘
                            │ Request / StreamDelta
┌───────────────────────────▼──────────────────────────────────┐
│ Engine Core  (后端无关, 1 份, 单写者)                           │
│   admission · 连续批 · radix prefix · slot 生命周期 · retract    │
│   chunked-prefill 策略 · microbatch 编排(PP)                   │
│   ────────────────────────────────────────────────────────    │
│   产出 ForwardPlan(= ForwardBatch IR):mode + 行布局 + KV 索引   │
└───────────────────────────┬──────────────────────────────────┘
        ForwardPlan(数据契约)│         ▲ future tokens(overlap 解耦)
┌───────────────────────────▼─────────┴────────────────────────┐
│ Executor Seam(窄契约,trait,编译期泛型)                        │
│  • BackendExecutor: execute(plan) → prefill/decode/mixed        │
│  • KvPool:          alloc/free/page/migrate                     │
│  • Communicator:    all_reduce / all_to_all / p2p (TP·EP·PP)    │
│  • Sampler:         logits → tokens                             │
│  • GraphRunner:     capture/replay,padded 桶,metadata 两段      │
└──┬───────────────┬───────────────┬───────────────┬────────────┘
   │ CudaExecutor  │ MetalExecutor │ HipExecutor   │ …(NPU/XPU)   ← 每个 ~1-2k 行,0 scheduler
   │ +CudaKvPool   │ +MetalKvPool  │ +HipKvPool    │
┌──▼───────────────▼───────────────▼───────────────▼────────────┐
│ Kernels:  attention(flash/paged) · gemm · moe(all-to-all) · quant│  per-device,手写或 codegen
└────────────────────────────────────────────────────────────────┘

正交注入:Topology{tp,pp,ep,dp}+rank mesh → 决定 Communicator 接线 + plan microbatch 化
正交注入:KV tier(host/disk/remote)、disagg role(prefill-only/decode-only)→ KvPool 之上
```

### 4.2 流水线视图(时间轴:CPU 怎么和 GPU 天然并行)

稳态下,engine core 跑在 GPU 前面一步,future-buffer 解耦:

```
step:        N-1            N              N+1            N+2
GPU forward: [===== fwd N-1 =====][===== fwd N =====][===== fwd N+1 =====]
CPU core:        [sched N][              ][sched N+1][          ][sched N+2]
                     │ publish future(N)      │ resolve future(N) as input(N+1)
Frontend:    [detok N-2 ‖ tokraw N+1] [detok N-1 ‖ tokraw N+2] …  (另一线程/进程)

关键:sched N+1 不等 fwd N 的 token 回 host —— 它读 future buffer 里 fwd N 的输出槽位索引。
      GPU 永不为"等 CPU 调度"而空转。detokenize/tokenize 在 Frontend 线程再叠一层重叠。
```

CUDA graph 落在 GPU forward 那条:decode 步形状固定 → replay 已 capture 的 graph(launch 开销→0);
动态 metadata(seq_lens、page table)在 capture 之外由 CPU stage 准备好,graph 内只读静态指针。

### 4.3 五个契约(seam 的精确定义)

| 契约 | 类型 | 作用 | 谁实现 |
|---|---|---|---|
| `ForwardPlan` + `ForwardMode{Prefill,Decode,Mixed,Idle,Verify,Draft}` | **数据** | engine core ↔ executor 的唯一桥;含 token 布局/positions/KV 索引/spec | engine core 产,executor 读 |
| `BackendExecutor` | **行为(窄)** | `execute_prefill/decode/mixed(plan, kv, comm)`;async launch/readback overlap 在此实现 | 每后端 |
| `KvPool` | **行为** | `alloc/free/seq_len/page_indices/migrate`;paged/quant/分层 layout 在此 | 每后端 |
| `Communicator` | **行为** | `all_reduce`(TP)/`all_to_all`(EP)/`send_recv`(PP);拓扑无关接口 | 每后端(NCCL/RCCL/MPI/Gloo) |
| `Sampler` + `GraphRunner` | **行为** | 采样 logits→token;graph capture/replay + 桶 + metadata 两段 | 每后端 |

`ModelArch`(层定义)横跨其上:用 `Communicator` 写 TP/EP collective,用 `KvPool` 读写 KV,
被 `BackendExecutor` 驱动。模型与设备解耦——加模型不碰后端,加后端不碰模型。

---

## 5. 可演进性:每个变化轴 = 一个局部插件

| 想加什么 | 改哪里 | **不**改哪里 |
|---|---|---|
| 新后端(HIP/ROCm) | `BackendExecutor`+`KvPool`+`Communicator`(RCCL)impl,~1-2k 行 | **scheduler / engine core 一行不改** |
| 新 attention kernel(FlashMLA…) | 一个 executor 内的 attention 分支 | plan / scheduler / 其他 kernel |
| 新并行轴(disagg P/D) | router + KV 搬运(NIXL 类)+ 两个 engine role | plan IR / KvPool(已抽象) |
| 新模型 | `ModelArch` impl(用 Communicator/KvPool) | 后端 / scheduler |
| 新量化/KV dtype | `KvPool` 变体 | scheduler / executor 主干 |
| 新并行组合(TP×EP×DP) | topology 描述符 + Communicator 接线 | scheduler 主干(只读 plan microbatch) |

这就是"可演进":变化被**契约边界**吸收,核心循环稳定。Dynamo 更进一步——把"引擎"本身做成可换的,
在其上做数据流编排;那是 ARLE 成熟后的下一个抽象层(单机引擎做扎实后再上)。

---

## 6. 映射回 ARLE:已有什么、缺什么、怎么走

ARLE 现状(详见 backend-seam-redesign.md 审计):**2 个 scheduler**(cuda 13.7k + metal 1.1k)、
`ModelForward` 50 方法 god-trait、scheduler 直接持 `PagedKVPool`。但好消息——**核心循环只有 3.7%
碰 CUDA**,且**理想架构的零件 ARLE 大半已有**:

| 理想零件 | ARLE 现状 | 差距 |
|---|---|---|
| ForwardPlan IR | ✅ `LogicalServePlan`(CUDA+Metal **已共享同一类型**) | CUDA 还在从 `StepPlan` shadow 转换(往返)→ 规范化 |
| Overlap scheduler | ⚠️ 已有 async `pending_decode`/`pending_prefill` 跨 loop turn | 但绑在 `ModelForward`,非 future-buffer 显式契约 |
| Communicator | ✅ `LayerCommunicator` 已存在 | 未从 god-trait 拆出为独立 seam |
| EP all-to-all | ✅ DeepEP 已接(DSv4) | 未抽象为 `Communicator::all_to_all` |
| KvPool | ❌ `PagedKVPool` 具体类型 scheduler 直持 | 抽 `KvPool` trait(L0.2,seam ~14 方法已测绘) |
| BackendExecutor | ❌ 融在 `ModelForward` 50 方法 | 拆窄 seam(L0.3) |
| 后端无关 scheduler | ❌ `scheduler/cuda/` + 独立 `MetalScheduler` | 合一(L1+L2) |
| DP-attention(MoE) | ❌ | Qwen3.6-MoE 必需,新轴 |
| PP microbatch | ❌ | 参照 `scheduler_pp_mixin` |
| disagg P/D | ❌ | 远期(Dynamo 层) |

**采纳优先级(承接 backend-seam-redesign 的 L 序列,按 ROI):**
1. **L0.1 ForwardPlan 规范化** —— CUDA 直接产 `LogicalServePlan`(像 Metal),删 `StepPlan` 往返 +
   `unified_scheduler` flag。自包含、已被你认可。
2. **L0.3 拆 god-trait** —— `BackendExecutor`/`Sampler`/`Communicator`(LayerComm 升格)/`KvPool`
   从 `ModelForward` 切出;overlap 重写为显式 future-buffer 契约(对齐 SGLang `FutureMap`)。
3. **L1+L2 合一 scheduler** —— `scheduler/cuda`→`scheduler/`,泛型 `<B:BackendExecutor,K:KvPool>`,
   删 `MetalScheduler`。**ARLE 自此一个 scheduler。**
4. **DP-attention + EP** 作为 MoE 并行轴接进 Communicator/topology(Qwen3.6 直接受益)。
5. **L3 HIP** 验证抽象;**disagg** 远期。

---

## 7. 北极星 + 反模式

**北极星:** 一个设备无关 engine core;后端是薄插件;CPU 全程重叠 GPU(future-buffer);
静态形状 graph;KV 当一等公民;DP/TP/EP/PP 正交组合(topology 注入);数据流是显式 stage 流水线。

**反模式(ARLE 已踩或易踩,grounded):**
- ❌ 每后端一个 scheduler(现状:cuda+metal,加 HIP=第三个)。
- ❌ god-trait seam(`ModelForward` 50 方法,混 forward+KV+graph+NCCL+sample+spec)。
- ❌ scheduler 持设备具体类型(`PagedKVPool`)。
- ❌ CPU stage 与 GPU 串行(无 future-buffer → GPU 等调度)。
- ❌ 动态形状阻塞 graph capture(必须 padding 到桶 + metadata 两段)。
- ❌ KV 跨 TP rank 复制(大 MoE 必用 DP-attention)。
- ❌ 为假想后端预塑接口(从 **CUDA↔Metal 两个真实后端的共性**抽 seam,HIP 验证;
  非 HIP-first 凭空设计)。

---

## 来源与方法

- **SGLang** `sgl-project/sglang@3e681d7`(读源):`base_attn_backend.py`(24 attn backends,
  含 AMD aiter/hip_radix/wave、Intel xpu/amx、NPU)、`forward_batch_info.py`(ForwardMode)、
  `mem_cache/memory_pool.py`(KVCache ABC)、`managers/overlap_utils.py`(FutureMap)、
  `scheduler_pp_mixin.py`(event_loop_pp)、`server_args.py`(tp/dp/ep/pp/dp-attention)、
  `model_executor/*cuda_graph*`(标准/piecewise/breakable)。
- **vLLM V1** 设计博客:<https://vllm.ai/blog/2025-01-27-v1-alpha-release>(进程分离、
  token-budget scheduler、zero-overhead overlap、piecewise CUDA graph + torch.compile、
  platform 抽象)。
- **NVIDIA Dynamo** 文档:<https://docs.nvidia.com/dynamo/>(disagg P/D、PrefillRouter、
  KVBM、NIXL 点对点 KV 搬运),NIXL 背景:<https://www.spheron.network/blog/nvidia-nixl-disaggregated-inference-guide/>。
- TRT-LLM / DeepSpeed-FastGen / Mooncake / LMDeploy / TGI / llama.cpp / MLC-LLM:架构知识,
  **hypothesis-grade**,未逐一读源。

**纪律(skill 要求):** 以上 survey 是 hypothesis-grade;落地 ARLE 任一 L 步前,以本地
`bench_guidellm` + `greedy_consistency` + nsys 在绑定 SLO shape 上 license-or-kill。
narrow-window 占比 ≠ wall-clock 影响。
