# MLX 性能上界与动态量化调研报告

> 调研日期：2026-03-31
> 硬件参考：Apple Silicon M1–M5 系列
> 模型参考：Qwen2.5-0.5B-4bit（278 MB）

---

## 一、MLX 后端的性能上界

### 1.1 Apple Silicon 内存带宽规格

| 芯片 | 内存带宽（理论峰值） |
|------|---------------------|
| M1 | ~68 GB/s |
| M1 Pro | 200 GB/s |
| M1 Max | 400 GB/s |
| M1 Ultra | 800 GB/s |
| M2 | 100 GB/s |
| M2 Pro | 200 GB/s |
| M2 Max | 400 GB/s |
| M2 Ultra | 800 GB/s |
| M3 | 100 GB/s |
| **M3 Pro** | **150 GB/s** |
| M3 Max (14-core GPU) | 300 GB/s |
| M3 Max (16-core GPU) | 400 GB/s |
| M3 Ultra | 819 GB/s |
| M4 | 120 GB/s |
| M4 Pro | 273 GB/s |
| M4 Max (32-core) | 410 GB/s |
| M4 Max (40-core) | 546 GB/s |
| M5 | 153 GB/s |
| M5 Pro | 307 GB/s |

**实际可达带宽**：STREAM benchmark 实测约为理论峰值的 **85%**（内存控制器开销、非顺序访问损耗）。

---

### 1.2 理论性能上界推导

**Decode 阶段是 memory-bandwidth bound**。每生成一个 token，GPU 需要从内存中完整读取一次模型权重（用于矩阵乘法）加上 KV cache（用于 attention）。

**简化公式**（忽略 KV cache，适用于短上下文）：

```
theoretical_max_tok/s = memory_bandwidth_GB_s / model_size_GB
```

**Qwen2.5-0.5B-4bit 参数**：
- 参数量：0.5B
- 量化精度：4-bit
- 模型文件大小：**278 MB = 0.278 GB**（实测 mlx-community 版本）

| 芯片 | 带宽 | 理论极限 | 85% 效率修正后 |
|------|------|---------|--------------|
| M3 Pro | 150 GB/s | **540 tok/s** | ~459 tok/s |
| M4 | 120 GB/s | 432 tok/s | ~367 tok/s |
| M4 Pro | 273 GB/s | 982 tok/s | ~835 tok/s |
| M4 Max (546 GB/s) | 546 GB/s | 1,964 tok/s | ~1,669 tok/s |
| M1 Ultra | 800 GB/s | 2,878 tok/s | ~2,446 tok/s |

---

### 1.3 实测性能与利用率

| 硬件 | 模型 | 实测速度 | 理论极限（85%校正） | 带宽利用率 |
|------|------|---------|-------------------|-----------|
| M3 Pro | Qwen2.5-0.5B-4bit | ~285 tok/s | ~459 tok/s | **62%** |
| M3 Pro | （本项目实测参考） | ~354 tok/s | ~459 tok/s | **77%** |
| M4 Max (546 GB/s) | Qwen3-0.6B | 525 tok/s (单请求) | ~1,669 tok/s | **31%** |
| M4 Max (546 GB/s) | Qwen3-0.6B | 1,642 tok/s (16并发) | ~1,669 tok/s | **98%** |
| M2 Max | Llama-2-7B Q4_0 | 31 tok/s | ~175 tok/s | **18%** |
| M2 Max | Llama-2-7B FP16 | 19 tok/s | ~28 tok/s | **68%** |

**关键洞察**：
- 模型越小，单请求利用率越低（M4 Max 跑 0.6B 模型只有 31% 利用率）
- 通过并发批处理可大幅提升（16 并发达到 98%）
- 较大模型（7B FP16）在单请求下已接近内存带宽瓶颈（68%）

---

### 1.4 剩余性能 Gap 的归因分析

以 M3 Pro 单请求 354 tok/s vs 理论 459 tok/s（差距 23%）为例：

#### (a) Metal kernel launch overhead（约 5-10%）
- MLX 动态构建计算图并逐层提交 Metal 命令
- 每次图评估都有固定开销（framework 文档承认存在 "some fixed overhead with each graph evaluation"）
- 0.5B 小模型层数少、每层计算量小，kernel dispatch 占比更高

#### (b) Attention 计算（compute bound，约 5-8%）
- Attention 的 softmax + QK^T 矩阵乘法是 compute bound，不受内存带宽限制
- 即使是 decode 阶段，attention 对新 token 与所有历史 token 的计算随上下文线性增长
- 小模型 embedding 维度小，attention kernel 无法充分利用 SIMD 并行度

#### (c) bf16 在 M1/M2 的隐性损耗（M3+ 无此问题）
- M1/M2 GPU **不支持原生 bf16**，内部将 bf16 升级到 fp32 计算
- M3+ 增加了原生 bf16 支持（famstack.dev 实测验证）
- MLX 默认使用 bf16 权重，M1/M2 用户面临隐性性能折损

#### (d) Python GIL 竞争（约 3-5%）
- mlx-lm 采用 Python 实现，token 生成循环受 GIL 限制
- 框架曾记录 "GIL starvation in the _generate thread when batch is idle" 问题（已在近期版本修复）
- 每个 token 的 Python 层调度开销约 0.1-0.5 ms

#### (e) Sampling overhead（约 2-3%）
- 逐 token 的 top-k/top-p/temperature 采样需 GPU→CPU 数据传输或 GPU kernel 启动
- 0.5B 小模型生成极快（<2ms/token），采样开销占比显著

#### (f) 单请求 KV cache 读取开销
- 随上下文增长，每个 decode step 需读取 KV cache
- 在 M3 Pro 150 GB/s 带宽下，KV cache + 权重的总读取量会超出纯权重计算

---

### 1.5 进一步优化路径

| 优化方案 | 预期收益 | 实现难度 |
|---------|---------|---------|
| 并发批处理（连续批处理） | 单机吞吐 2-5x（至带宽上限） | 中等 |
| KV cache 量化（kv_bits=4） | 长上下文速度 20-40% 提升 | 低（已支持） |
| 专用 Metal kernel（Flash Decoding） | 单请求延迟 10-20% 降低 | 高 |
| 模型权重缓存预热（避免首次加载延迟） | 首 token 延迟降低 | 低 |

---

## 二、为什么 MLX 没有支持动态量化

### 2.1 概念厘清：三种"量化"

| 类型 | 量化对象 | 时机 | MLX 支持？ |
|------|---------|------|-----------|
| **静态权重量化** | 模型权重（W） | 离线，部署前 | ✅ 支持（4-bit/8-bit） |
| **混合精度权重量化** | 逐层权重，敏感层用更高精度 | 离线，敏感度分析 | ✅ 支持（`mlx_lm.LEARNED_QUANTS`） |
| **KV cache 量化** | KV cache 张量 | 运行时，但固定精度 | ✅ 支持（`--kv-bits` 参数） |
| **动态量化（真正意义）** | 激活值（A） + 权重（W） | 运行时，per-token | ❌ **不支持** |

**注意**：MLX 文档中出现的 "dynamic quantization" 指混合精度权重分配（按层灵活分配 bit 数），并**不是**激活值的运行时量化。

### 2.2 什么是真正的动态量化

在 CUDA 生态中，动态量化（W8A8、W4A8）指：
- **W**: 权重以低精度存储（INT8/INT4）
- **A**: 激活值在**每次前向传播时**动态量化为 INT8/INT4
- 实际计算：`INT8(weights) × INT8(activations)` → 利用硬件 INT8 tensor core 加速矩阵乘法

代表实现：SmoothQuant、LLM.int8()（bitsandbytes）、GPTQ act-order

### 2.3 MLX 不支持动态量化的根本原因

#### 原因一：Apple GPU（M1-M4）没有独立的 INT8 矩阵乘法硬件路径

CUDA 生态动态量化的收益来自 **Tensor Core 的 INT8 加速**（A100 上 INT8 吞吐是 FP16 的 2x）。

Apple GPU（M1-M4）的情况截然不同：
- 不提供离散的 INT8 TFLOPS 规格
- INT8 矩阵乘法通过 **FP16 pipeline** 处理（simdgroup_matrix API）
- 实测无加速：Apple Core ML 工具文档明确指出"激活量化会导致在使用运行时权重解压缩的计算单元（CPU 有时 GPU）上出现显著的推理减速"
- Metal benchmarks 库确认："未提供离散的 INT8 性能数据，该架构优先 FP16/FP32 混合精度运算"

**结论**：在 M1-M4 上做 W8A8 运算时，INT8 乘法会被转为 FP16，没有任何加速收益，反而增加量化/反量化的开销。

#### 原因二：Apple Neural Engine（ANE）不适用于 LLM 推理

Apple Silicon 的真正 INT8 加速在 **ANE（神经引擎）** 而非 GPU：
- M4 ANE：38 TOPS（INT8 为主）vs GPU 4.6 TFLOPS（FP16）
- M5 Neural Accelerator：13.4 TOPS vs 7.4 TFLOPS FP16（约 **1.85x** INT8 优势）

但 ANE 用于 LLM 有严重限制：
- **上下文长度限制**：ANE 只支持 512-2048 token，无法处理通常 LLM 推理场景
- **访问开销**：通过 Core ML 访问 ANE 有 2-4x 额外开销
- **架构不匹配**：ANE 为卷积/矩阵乘法的固定形状优化，不适合变长序列的自回归生成

MLX 走 GPU 路径，ANE 的 INT8 优势无法被利用。

#### 原因三：Memory-bandwidth bound 场景不需要动态量化

LLM decode 阶段在 Apple Silicon 上是 **内存带宽瓶颈（memory-bandwidth bound）**，而非计算瓶颈（compute bound）。

这是核心洞察：
- 内存带宽瓶颈 → 优化**权重的存储大小**（weight-only 量化）即可最大化性能
- 计算瓶颈 → 才需要降低 **FLOPS**（W8A8 激活量化的适用场景）

对比：
| 场景 | 瓶颈类型 | 最优量化策略 |
|------|---------|------------|
| CUDA，大批量（batch=32+） | Compute bound | W8A8 动态量化（INT8 Tensor Core） |
| Apple Silicon，单请求 decode | Memory-bandwidth bound | W4 weight-only 量化 |
| Apple Silicon，高并发 decode | 接近 Memory-bandwidth bound | W4 + KV cache 量化 |

Weight-only 4-bit 量化已经让 decode 阶段的带宽需求降低 4x，基本达到苹果硬件上单请求 decode 的最优解。在这个场景下，增加激活量化带来的收益几乎为零，而增加的运行时开销（动态量化/反量化）会造成净负收益。

#### 原因四：Metal graph compilation 与动态 shape 兼容性差

MLX 的执行模型基于**惰性求值 + 自动图编译**：
- 编译时需要确定张量形状（否则无法生成最优 Metal shader）
- 激活量化的 scale 因子是逐 token、逐 batch 动态计算的
- 这要求在编译期不确定的值流入 Metal kernel，破坏编译优化

相比之下，权重量化在加载时完成，shapes 在编译时已知，与 MLX 的惰性图求值完美兼容。

#### 原因五：社区需求优先级低

查看 mlx GitHub issue #13、#135、#262，以及 mlx-lm 相关讨论：
- 量化相关需求集中在**权重量化格式**（支持更多 GGUF 格式、AWQ 格式）和**混合精度分配**
- 无高优先级 issue 要求激活量化
- 维护者 Awni Hannun 在 2024 年 1 月关闭主要量化 issue 时，重心在 weight quantization
- MLX 定位是 Apple Silicon 研究框架，在 weight-only 量化下已达到良好性能，激活量化的 ROI 对 Apple 平台用户不高

---

### 2.4 与 CUDA 生态的对比

| 特性 | CUDA（PyTorch/vLLM） | MLX |
|------|---------------------|-----|
| W8A8（SmoothQuant） | ✅ via llm-compressor | ❌ |
| LLM.int8()（bitsandbytes） | ✅ | ❌ |
| GPTQ with act-order | ✅ | ❌ |
| AWQ（weight-only） | ✅ | ✅（格式转换后） |
| W4A16（纯权重 4-bit） | ✅ | ✅ |
| 混合精度权重量化 | ✅ via GPTQ/AWQ | ✅（sensitivity-based） |
| KV cache 量化 | ✅（多种方案） | ✅（4-bit uniform + TurboQuant） |
| MXFP4 | ✅ H100 via NVFP4 | ✅ M5 via Neural Accelerator（实验性） |
| **动态激活量化** | ✅ 完整支持 | ❌ 无 |

**根本区别**：CUDA Tensor Core 对 INT8 矩阵乘法提供真实 2-4x 加速，使得运行时激活量化的开销值得付出。Apple GPU（M1-M4）无此硬件路径。M5 引入的 Neural Accelerator 在技术上具备 1.85x INT8 优势，未来可能推动 MLX 支持受限场景下的激活量化。

---

## 三、总结

### 性能上界

- **M3 Pro 理论极限**（Qwen2.5-0.5B-4bit）：**540 tok/s**（85% 效率校正后 **~459 tok/s**）
- **实测 354 tok/s → 带宽利用率 77%**，属于良好水平
- 剩余 23% gap 主要来自：Metal kernel dispatch overhead、attention compute bound、Python GIL、sampling overhead
- **提升单机吞吐**的最有效方式：并发批处理（可达 98% 带宽利用率）

### 动态量化缺失原因

1. **硬件无收益**：M1-M4 GPU 无独立 INT8 计算路径，动态量化无加速
2. **场景不匹配**：Memory-bandwidth bound 场景只需 weight-only 量化，无需 compute 优化
3. **架构限制**：ANE 的 INT8 优势因上下文限制无法用于 LLM
4. **框架设计**：MLX 图编译与动态激活 shape 兼容性差
5. **社区优先级**：Apple 平台用户在 weight-only 量化下已达最优，需求不迫切

**展望**：M5 Neural Accelerator 的 1.85x INT8 优势 + 未来更长上下文 ANE 支持，可能推动 MLX 在高并发场景下引入动态量化。

---

## 参考资料

- Apple Silicon 内存带宽：Apple 官方规格页面 + Wikipedia
- STREAM benchmark 实测：community measurements（~85% 效率）
- MLX-LM 性能测试：awni's gists, SiliconBench, famstack.dev 分析
- vllm-mlx 论文：arXiv:2601.19139（Qwen3-0.6B on M4 Max 1,642 tok/s at 16 concurrent）
- MLX 量化文档：ml-explore/mlx GitHub issues #13, #135, #262
- Apple Core ML Tools 文档：激活量化推理减速警告
- metal-benchmarks：Apple GPU INT8/FP16 对比
- famstack.dev：bf16 on M1/M2 GPU 性能回归分析
