# Gap ② — KV 缓存预算:静态/heuristic → profiling-run 实测

研究日期 2026-06-08。配套主文档
[`2026-06-08-model-loading-preprocessing-arle-vs-industry.md`](2026-06-08-model-loading-preprocessing-arle-vs-industry.md)。

**Verdict**:这是三个 gap 里**最实**的一个。ARLE 的 KV 预算两条路径都不测真实 forward
激活峰值——Qwen3 dense **根本没有自动预算**(用户手填 `total_pages`),DSv4 用
`cudaMemGetInfo×0.9` 盲猜 margin。vLLM 跑一次 profiling forward **实测**非 KV 峰值再分配
剩余给 KV,精确且免手调、免高并发 OOM。建议优先落地。

Evidence tier:✅ code-verified(ARLE) / 📖 upstream source-survey / ⚗️ needs experiment。

---

## 1. ARLE 现状(✅)

### 1.1 Qwen3 dense:全静态,无自动预算
`CudaExecutor::from_*`(`infer-cuda/executor.rs:280`):
```
token_budget = total_pages * SUPPORTED_PAGE_SIZE          // total_pages 由调用方/CLI 给
budget_bytes = PagedKVPool::budget_bytes_for_tokens(
                 layers, kv_heads, head_dim, token_budget, KVFormat::BF16)
```
`total_pages` 是 `SchedulerConfig` 透传的**用户输入**,**没有任何 GPU 显存查询**。
后果:填高 → 稳态 forward 激活叠加 KV 超显存 → **runtime OOM**;填低 → KV 容量浪费、
`max_total_tokens` 偏小、并发上限被卡。等于把"分多少显存给 KV"这个决策甩给用户手调。

`budget_bytes_for_tokens`(`cuda-kernels/paged_kv.rs:188`)只做 per-token 字节会计
(`(kv_dim·bpe·2 + scale + norm)·layers`),把 token 数翻成字节——它是**会计器,不是预算器**,
不决定"该给多少 token"。

### 1.2 DSv4:动态但 heuristic
`Dsv4*::...`(`infer-cuda/dsv4.rs:796-813`):
```
const MEM_FRACTION: f64 = 0.9;
free = cudarc::driver::result::mem_get_info().free      // 查当前空闲显存
num_slots = free * 0.9 / per_slot_kv_bytes              // 砍 0.9 当 margin
```
commit 史:`aa445112` 加这段修了 c=32 OOM 崩溃,`3981225d` 验了 c=1 byte-identical。
**比 Qwen3 路径好**(至少自适应显存),但两个结构性弱点:
1. **查询时机**:`mem_get_info` 在权重已载、但 **forward 的 graph/workspace/激活尚未分配**
   时查 free。那 0.9 是给"所有还没分配的东西"留的**盲猜 margin**——没人测过 max-batch
   forward 的激活峰值到底吃多少。margin 偏大 → 浪费;偏小 → 高并发仍可能 OOM。
2. **per-slot 满额预留**:`per_slot_kv_bytes` 按 max_seq_len 整额留,不是按 paged 增长留,
   进一步放大保守度。

---

## 2. 业界做法(📖)

### 2.1 vLLM — `determine_num_available_blocks` + profiling forward(标杆)
worker 启动序列(概念,精确符号名 hypothesis until pinned):
1. **载权重**,记 `weights_memory`。
2. **`profile_run`**:用 dummy token 构造**最坏批**(`max_num_seqs ×
   max_num_batched_tokens`)跑一次 forward,**实际触发** attention/MLP/MoE 的激活 +
   workspace 峰值,用 `torch.cuda.max_memory_allocated()` 取 high-water。
3. `peak_non_kv = max_memory_allocated`(权重 + 激活 + 临时 buffer 的真实峰值)。
4. `available_kv = total_gpu_mem × gpu_memory_utilization(默认 0.9) − peak_non_kv`。
5. `num_gpu_blocks = available_kv / cache_block_bytes`;`initialize_cache` 精确分配这么多。
6. 之后 `capture_model` 抓 CUDA graph,graph 显存也算进已 reserved(因为在 KV 分配后捕获,
   或预留固定份额)。
- **关键差异**:`gpu_memory_utilization` 砍的是**总显存**,而被减掉的 `peak_non_kv` 是
  **实测值不是猜值**。所以 0.9 只是"留给碎片/未覆盖项"的小余量,激活那块是量出来的。
- 旁路:`--kv-cache-memory-bytes` 可直接指定,跳过 profiling。

### 2.2 SGLang — `--mem-fraction-static`
`init_memory_pool`:`max_total_num_tokens ≈ (total_gpu_mem × mem_fraction_static −
model_size − activation_estimate) / per_token_kv_bytes`。default `mem_fraction_static`
≈0.9。激活用启发式估(部分版本结合一次 profile / `--max-running-requests` 约束),比 vLLM
的纯实测略粗,但仍**显式扣激活**,不是只砍空闲显存。

### 2.3 TensorRT-LLM — `kv_cache_free_gpu_mem_fraction`
default 0.9 of **free** memory after engine load。**与 ARLE DSv4 的 `free×0.9` 几乎同构**——
但 TRT 引擎的激活/workspace 是 AOT 已知(build 时定),所以"free×0.9"在 TRT 语境下比在
ARLE 语境下更安全(ARLE 的激活峰值是运行时才知道的)。

### 2.4 llama.cpp
`n_ctx × n_layer × kv 字节` 直接按 context 长度静态算,用户给 `-c`;无并发批,问题维度不同。

---

## 3. Gap 精确表述

| | Qwen3 dense(ARLE) | DSv4(ARLE) | vLLM | SGLang | TRT-LLM |
|---|---|---|---|---|---|
| 自适应显存? | ❌ 全静态 `total_pages` | ✅ `free×0.9` | ✅ | ✅ | ✅ |
| 扣**实测**激活峰值? | ❌ | ❌(盲猜 margin) | ✅ profiling-run | ~(估) | ❌(但 AOT 已知) |
| 免用户手调? | ❌ 必须填 `total_pages` | ✅ | ✅ | ✅ | ✅ |
| 高并发 OOM 风险 | 高(填高即崩) | 中(0.9 没测峰值) | 低 | 低 | 低 |

**核心 gap**:① Qwen3 路径是**裸缺口**(连 DSv4 的 free×0.9 都没有);② 两条路径都**没有
profiling-run** 去实测 max-batch forward 的非 KV 峰值,vLLM 的精确性来自这一步。

---

## 4. 落地建议(⚗️ license-or-kill）

**目标**:统一一条 `profile_kv_budget()`,替换 Qwen3 的静态 `total_pages` 推导与 DSv4 的
`free×0.9`。复用已存在的 warmup 钩子(`executor.rs:474` 已有 `warmup()` 跑一步)。

**算法(对齐 vLLM,适配 cudarc 无 `max_memory_allocated` 的现实)**:
1. 载权重后,`(free0, total) = mem_get_info()`(`cuda-kernels/tensor.rs:319` 已封装)。
2. **profiling forward**:用 dummy token 跑一次**最坏批**(`num_slots ×
   max_batched_tokens`,或 scheduler 的 max 并发形状)的 prefill+decode,strict `ctx.sync()`。
3. `free1 = mem_get_info().free`;`peak_non_kv_delta = free0 − free1`(这次 forward 新增的
   激活/workspace/graph 占用近似)。注意 cudarc 给的是瞬时 free 不是 high-water,需在 forward
   **内部峰值点**或 forward 后立即量(保守:量 forward 中途 + 结束取 min free)。
4. `kv_budget = total × util(默认 0.9) − (total − free0) − peak_non_kv_delta`
   = `free0 × util_adjusted − peak_non_kv_delta`(精确公式按是否 util 砍总额定)。
5. `num_slots / total_pages` 由 `kv_budget / per_token_kv_bytes` 反算。

**CLI 旋钮(遵循 [[feedback_runtime_config_cli_flags_not_env]])**:`--gpu-memory-utilization`
(industry 标准名,对齐 vLLM,遵循 [[feedback_use_industry_env_vars]] 的命名取向)+
`--kv-cache-memory-bytes` 旁路。env 仅作 test-harness shim。

**Gate(必须全过才 license)**:
1. **c=1 不退**:profiling-run 算出的 num_slots ≥ 当前手调值,c=1 tok/s 同 harness A/B 不退。
2. **高并发不 OOM**:c=32(DSv4)/ Qwen3 max 并发跑通,无 OOM,且实测 free 余量 >0。
3. **精度对照**:profiling 算出的 `peak_non_kv` 与 nsys/`mem_get_info` 实测峰值误差 <10%
   (否则 profiling 形状不对,等于换个盲猜)。
4. **DSv4 不退**:替换 `free×0.9` 后 num_slots 不低于现值(否则容量回退)。

**风险**:
- cudarc 无 torch 式 high-water,`free0−free1` 只是 forward **结束后**的残留 delta,可能**漏掉
  中途瞬时峰值**(临时 buffer 已 free)。缓解:warmup 内在激活峰值点(attn/MoE 后)插
  `mem_get_info` 取最小 free;或预留固定 workspace 头(类似 DSv4 现状的 0.9 但只覆盖未测部分)。
  这是本 gap 落地的**主技术难点**,需先验证 cudarc 能否量到真实峰值,否则退化成 SGLang 式估算。
- profiling 形状必须是**生产最坏批**,不是 smoke 形状(见
  [[feedback_measure_batching_before_ceiling]] 同类教训:小形状预测高并发会翻)。

---

## 关联
- ✅ `infer-cuda/executor.rs:280`(Qwen3 静态)/ `dsv4.rs:796-813`(DSv4 free×0.9)/
  `cuda-kernels/paged_kv.rs:188`(per-token 字节会计)/ `tensor.rs:319`(`gpu_memory_info`)。
- [[feedback_runtime_config_cli_flags_not_env]] / [[feedback_use_industry_env_vars]] — 旋钮命名。
- [[feedback_measure_batching_before_ceiling]] — profiling 形状必须是生产最坏批。
- commit `aa445112` / `3981225d` — DSv4 free×0.9 的引入与 c=1 验证。
