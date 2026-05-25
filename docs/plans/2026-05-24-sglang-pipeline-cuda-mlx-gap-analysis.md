---
title: SGLang pipeline ↔ ARLE CUDA/MLX 全链对照、gap 分析与落地计划
date: 2026-05-24
type: analysis + plan
status: draft — analysis 落地,plan 待 license-or-kill 决策
owner: ckl
related:
  - docs/codebase-map.md
  - docs/plans/2026-05-09-prefill-graph-phase0v3-validation-protocol.md
  - docs/plans/M_e1-metal-paged-kv-hot-path.md
  - docs/plans/M_e1-omlx-c-multi-step-pipelining.md
  - docs/plans/M_e5-mlx-multi-stream-pipelining.md
  - docs/plans/M_e2-prompttrie-prefix-cache.md
  - docs/plans/M_ibp-in-batch-prefix-caching.md
  - docs/plans/2026-05-04-kv-tier-hicache-borrowed-improvements.md
  - infer/src/backend/metal/AGENTS.md
---

# SGLang pipeline ↔ ARLE CUDA/MLX 全链对照、gap 分析与落地计划

> 一句话:用 SGLang 经典 5 阶段流水线(启动 → 调度 → prefill → 采样 → 贯穿
> 机制)作为参考系,把 ARLE 当前 CUDA(`infer/src/scheduler/cuda/` +
> `crates/cuda-kernels/`)和 Metal(`infer/src/backend/metal/` +
> `crates/mlx-sys/`)两条链路逐项对照,标清楚 ✅/⚠️/❌/N/A,把"看起来已经
> 做了"和"真的进了热路径"分开来,然后按 SOLID + license-or-kill 给出排序
> 后的落地计划。

---

## 0. SOLID 自检 — 这份 doc 的证据等级

按 §0 第一原则:推断 ≠ SOLID。本 doc 里:

- **Evidence(✅)** = `file:line` 引用 + 实际函数体读过 + 行内引号。本次审计
  由两个 `Explore` subagent 在 working tree 上跑出来(2026-05-24)。
- **Hypothesis(🔬)** = 推断/对照 SGLang 文档/未做 nsys/未做 wall-clock
  对照实验得出的结论。下文用 🔬 标。**Hypothesis 必须先 cheap-experiment
  验证才能转 commit;不验证就实施 = §0 反模式**。
- **License-or-kill 阈值**:每条落地动作显式标 PASS / KILL 阈值,且必须用
  wall-clock framing(不是 narrow window),参照 EOD+19 `M_pf-graph v2` 教训。

---

## 1. SGLang 参考流水线(对照轴)

| 阶段 | SGLang 做什么 | 关键产物 |
|---|---|---|
| **A 启动** | weights H2D + KV pool 测显存 + req_to_token + 空 radix 树 + FlashInfer workspace + CUDA Graph 多 batch 预录 + 3 进程 ZMQ | 静态资源就位 |
| **B 调度** | tokenize → Scheduler 入队 → RadixCache 前缀命中 → 只为后缀分槽 | 每 tick 的 prefill/decode batch |
| **C Prefill** | embed →(RMSNorm → fused QKV GEMM → fused RoPE → KV scatter → extend attention → o_proj → fused MLP)× L → final RMSNorm → LM head | 最后位置 logits |
| **D 采样** | GPU 上 temp/top-p/top-k 采样 → token D2H → 增量 detokenize → 插回 radix 树 | 第一个 token + 树更新 |
| **E 贯穿** | overlap 调度 / decode CUDA Graph 重放 / HiCache HBM↔DRAM↔NVMe↔远端 | 隐性 wall-clock 加速 |

> 「与朴素 PyTorch 差别压在三处」:**radix 树零重算 + token_to_kv_pool O(1)
> KV 写 + CUDA Graph decode 下发归一**。下面的所有结论围绕这三点收敛。

---

## 2. CUDA 链路逐阶段对照(evidence-grounded)

### A. 启动期

| 项 | 状态 | Evidence |
|---|---|---|
| A1 safetensors → TP shard → H2D | ✅ | `infer/src/backend/cuda/bootstrap.rs:33-100` `load_shard_info` / `mmap_shards`;`infer/src/tp/load_context.rs:35-68` `TpLoadContext::column/row` |
| A2 测显存 → 推 token 槽数 | ✅ | `crates/cuda-kernels/src/paged_kv.rs:127-180` `compute_budget_breakdown`;`bootstrap.rs:91-100` 用 SGLang 公式 `free × (1 - mem_fraction_static)`;page_size=16,BF16/FP8/INT8 均尊重 |
| A3 req_to_token + 空 radix 树 | ✅ | `infer/src/prefix_cache.rs:308-350` `RadixCache::new`;`Node` (138-195) 带 `tier_location` / `session_id` / `fingerprint` |
| A4 FlashInfer workspace | ❌ → N/A | **不接 FlashInfer**,用自家 TileLang AOT `csrc/attention/{prefill_attention,prefill_attention_hd256,decode_prep_paged,fused_attention}.cu`;dispatch 在 `infer/src/ops/attention.rs:54-67` |
| A5 多 batch-size CUDA Graph 预录 | ✅ | 已有 batch-size graph warmup/cache:scheduler 启动预热 batch-size 列表(`infer/src/scheduler/cuda/core/warmup.rs:26,418`);Qwen3 `graph_cache[batch_size - 1]`(`infer/src/model/qwen3/batch_decode.rs:171`);Qwen3.5 piecewise `graph_cache[group_idx][batch_size - 1]`(`infer/src/model/qwen35/batch_decode.rs:168`) |
| A6 3 进程 ZMQ | ❌ → N/A | 单进程,grep `zmq`/`TokenizerManager` 无结果 |

### B. 请求 → 调度

| 项 | 状态 | Evidence |
|---|---|---|
| B1 HTTP 边界分词 | ✅ | `http_server/openai_v1.rs:75-100`;`tokenizer.rs:80-86`;fingerprint SHA256 做 RadixCache 命名空间(`tokenizer.rs:28-60`) |
| B2 调度器队列 + continuous batching | ✅ | `scheduler/cuda/runtime/scheduler_loop.rs:101-150` `Scheduler::run`;`scheduler/cuda/prefill.rs:111-200` `step_new` / `step_running_decode` / `step_running_prefill` |
| B3 Radix 命中 → 只分后缀 | ✅ | `prefix_cache.rs:674-768` `lookup_or_stage`;`prefill.rs:139-200` "if full_prompt_reuse_hit" 直接复用;否则 `effective = prompt_tokens[attach_prefix_len..]` |

### C. Prefill 前向

| 项 | 状态 | Evidence |
|---|---|---|
| C1 per-layer 融合 | ✅ | `model/qwen35/prefill.rs:23-200`:embedding gather (36-41)、`batched_rms_norm_offset` (97)、QKV `gemm` (159)、**`prefill_attention_hd256_batch`** (176-189) **融合 QK norm + RoPE + K/V scatter**,attention 读 cached-prefix+new KV 无 S×S 物化 |
| C2 final RMSNorm + LM head | ✅ | `prefill.rs:65-73` `compute_logits_batch` |

### D. 采样 → 第一个 token

| 项 | 状态 | Evidence |
|---|---|---|
| D1 GPU 采样无 logits D2H | ✅ | `ops/sampling.rs:94-276` + `csrc/misc/sampling.cu`;`gpu_sample_cuda` 在 device 上做 temp → softmax → top-k → top-p → multinomial;只 D2H i32 (279-285) |
| D2 新序列插回 radix 树 | ⚠️ | `RadixCache::insert` 存在,但 scheduler 侧 insert 时机 / 触发点端到端未完全 trace 验证 |
| D3 token D2H + detokenize | ✅ | `model/qwen35/batch_decode.rs:140-160` 异步 readback;`tokenizer.rs:94-120` 增量 decode |

### E. 贯穿机制

| 项 | 状态 | Evidence |
|---|---|---|
| E1 overlap 调度(CPU 准备下一轮 / GPU 跑当轮) | ✅ | `scheduler_loop.rs:132-138` 注释"keeps decode/prefill readback pending across loop turns so this iteration's intake/admission work can overlap the previous iteration's GPU compute" |
| E2 decode CUDA Graph 重放 | ⚠️ | on-demand 单 graph,无 bucket pre-capture |
| E3 HiCache 多层 | ⚠️ | T0+T1 活的(`paged_kv` + `HostPinnedPool`);`lookup_or_stage` 有 staged readmission + `Coordinator` 队列;**T2 disk 在 `kv_tier/transport/disk.rs` 在,T3 NIXL 是 stub-only**(`kv_tier.rs:18` "remote tier remains skeletal") |

---

## 3. Metal/MLX 链路逐阶段对照(evidence-grounded)

> 规范模型:**Qwen3.6-35B-A3B-4bit MoE**(CLAUDE.md §Metal canonical model)。
> CUDA-vs-Metal 设计差异(unified memory、无 PCIe、无独立 graph API、无 TP)
> 计入 N/A,不计 gap。

### A. 启动期

| 项 | 状态 | Evidence |
|---|---|---|
| A1 MLX load + Q4 + MoE 权重 | ✅ | `infer/src/backend/metal.rs:200-250`;`MetalQwen35MoeWeights`(router、switch_gate/up/down、shared_gate/up/down)`qwen35.rs:63-79`;auto wired-limit `metal_serve.rs:32-84` |
| A2 KV pool sizing | ⚠️ | `backend/metal/kv_pool.rs` token-level `SlotLedger::new(max_total_tokens)` **已建,但未进 Qwen3.5 packed-decode 热路径**(`metal_serve.rs:161` "dual-write lands in P2.1");当前仍走 per-request KV tensor |
| A3 RadixCache for Metal | ✅(数据结构)/ ⚠️(命中收益) | `backend/metal/prefix_cache.rs` 桥接到同一 `infer/src/prefix_cache.rs::RadixCache`;**因 A2 未通,命中后的"零重算"未真正绕过 prefill** |
| A4 attention workspace | ✅ | vendored MLX `scaled_dot_product_attention`(`crates/mlx-sys/src/lib.rs:126`)+ varlen `build_varlen_decode_mask`(`backend/metal/mlx.rs`) |
| A5 CUDA-Graph 等价 | ❌ | `mlx_metal_capture.mm` 只支持 `INFER_CAPTURE_STEP` profiling 抓帧;**无 command-buffer pre-encode + replay** |
| A6 进程切分 | ❌ → N/A | 单进程 `metal_serve.rs` |

### B. 请求 → 调度

| 项 | 状态 | Evidence |
|---|---|---|
| B1 continuous batching + 变长 packed decode | ✅ | `backend/metal/request_state.rs:797-1911` `Qwen35PackedDecodeBatch` 真的有 `left_padding: Vec<i32>` + 加性 mask + per-row RoPE offsets(mlx-lm BatchKVCache pattern,符合 `backend/metal/AGENTS.md` §7) |
| B2 prefix 命中跳过重算 | ⚠️ | 路径在,但 A2 未通,**packed decode 没真用 pool 写入**,命中实际仍走一次 prefill |
| B3 resumable prefill + one-step decode | ✅ | `request_state.rs:118-292` `ResumableRequestState` trait;`prefill_chunk(budget)` 返回 `PrefillChunkResult` |

### C. Prefill 前向

| 项 | 状态 | Evidence |
|---|---|---|
| C1 MoE step path | ✅ | `crates/mlx-sys/src/mlx_qwen35_model.cpp` + `backend/metal/qwen35.rs`;router → top-k → expert dispatch |
| C2 KV scatter / append | ⚠️ | `pool.write_kv(slot_indices, k_rows, v_rows)` **当 `use_kv_pool=true` 时**;否则 fallback `extend_kv_cache`(per-request tensor)。**默认 flag 未启** |
| C3 attention kernel | ✅ | MLX vendored kernel |

### D. 采样

| 项 | 状态 | Evidence |
|---|---|---|
| D1 GPU 采样 | ⚠️ | `backend/metal/sampling.rs:44-63` `gpu_sample_token` **只支持 greedy + temperature**(line 10-26);**top-k/top-p/penalties 缺**(CUDA 侧完整) |
| D2 新序列回写 radix | ✅ | `backend/metal/prefix_cache.rs:59` `insert(tokens, slot_indices)` |
| D3 detokenize | ✅ | 同进程,`.item_i32()` → tokenizer |

### E. 贯穿机制

| 项 | 状态 | Evidence |
|---|---|---|
| E1 overlap | ⚠️ | `async_eval` 异步派发 `qwen35.rs:2912, 2997`,但 encode 本身 sync 且占 95%(`backend/metal/AGENTS.md`、`docs/experience/wins/2026-05-07-bench-qwen36-encode-bottleneck.md`)→ overlap 收益被 encode 吃掉;`MLX_GUARD` 全进程 mutex(`crates/mlx-sys/src/lib.rs:12-21`)序列化 FFI |
| E2 graph replay 等价 | ❌ | 不存在 |
| E3 KV tiering | ❌ → 部分 N/A | unified memory 决定的;`backend/metal/gdr.rs` 只管 GDR recurrent state,工作集超 wired limit 时无降级 |

---

## 4. Cross-cut gap 矩阵(SGLang「三处压差异」对齐)

| SGLang 关键压差 | CUDA 状态 | Metal 状态 | 端到端兑现? |
|---|---|---|---|
| **radix 树命中前缀零重算** | ✅ 端到端通(C1 + B3) | ⚠️ 数据结构通,**热路径未接**(A2 deferred P2.1) | CUDA yes / Metal **no** |
| **token_to_kv_pool O(1) KV 写不搬历史** | ✅ `paged_kv` page-aware + `prefill_attention_hd256_batch` 融合 scatter | ⚠️ scatter 在 `pool.write_kv`,但 flag 默认未启 | CUDA yes / Metal **no** |
| **CUDA Graph decode 下发归一** | ⚠️ 单 graph on-demand,**无 bucket** | ❌ **完全没有 graph replay 等价物** | CUDA partial / Metal **no**(95% encode 占比 ground truth) |

> 结论:**SGLang 三处差异化收益,CUDA 兑现了 2.5/3,Metal 兑现了 0.5/3。**
> Metal 链路当前最大的"看起来对、其实没兑现"集中在 A2 + E2。

---

## 5. Gap 清单(按 wall-clock 影响 + license-or-kill 排序)

> 排序依据:**预测对 wall-clock TTFT/ITL 的影响**(不是 narrow window),
> 高 → 低。证据强度区分:🟢 已有实测 baseline / 🟡 推断需 cheap experiment /
> 🔴 纯 hypothesis。

### G1 🟢 Metal — Qwen3.5/3.6 packed decode 接 KV pool dual-write(P2.1)

- **现状**:`kv_pool.rs` + `prefix_cache.rs` 已建好,但 packed decode 默认
  `use_kv_pool=false`,prefix 命中收益不到 GPU。
- **影响**:多请求 / 重复 system prompt 场景下,每个请求白跑一遍 prefill。
  对 agent workload(W3/W4,system prompt 经常 1–4k token)是直接 TTFT 损失。
- **Evidence 强度**:🟢 — 命中应跳的 token 数可直接由 RadixCache 实测(已有
  逻辑),只差通到 pool 写。
- **动作**:把 `pool.write_kv` 从 flag-gated 改成默认路径,删 `extend_kv_cache`
  fallback;packed decode 的 K/V 索引从 `slot_indices` 来。
- **License-or-kill**:**PASS** 阈值 = Qwen3.6-35B-A3B 在 c=4 + 共享 1k system
  prompt 下,**第 2 请求 TTFT < 0.5 × 第 1 请求 TTFT**(命中应零重算);
  否则 hot path wiring 有 bug。**KILL** = 接通后任意 single-request bench
  ITL 退化 > 3%(写多了一份 KV 不应有可观开销)。

### G2 🟢 Metal — command-buffer pre-encode + replay 等价物

- **现状**:每 step 重编 600–1000 个 primitive;`mx::async_eval` 同步编 95%
  占比(CLAUDE.md + `2026-05-07-bench-qwen36-encode-bottleneck.md` 实测)。
- **影响**:**Metal 第一性瓶颈**。CUDA 用 CUDA Graph 把 decode launch 归一,
  Metal 当前完全没有等价。
- **Evidence 强度**:🟢 — bottleneck 已实测 wall-clock 95%,不是 narrow window。
- **难点**:MLX 0.31.1 没原生 "compile-once-replay-many" API。**先 license**
  小实验,再决定路线:
  - 路线 A:**MLX 算图缓存复用**(M_e1-omlx-c-multi-step-pipelining 已有
    研究)— 同 shape 的 decode 复用 lazy graph 物化结果。
  - 路线 B:绕过 MLX,直接 vended Metal command buffer indirect command
    encoder(ICB)预录定 shape decode pipeline。投入大,需评估。
- **License-or-kill 实验**(必跑,2 天):用 MLX C++ API 实验 lazy graph
  内部 op-list 是否可在 same-shape 下复用,**measure async_eval encode 时间
  占比下降幅度**。**PASS** = encode 占比从 95% 降到 ≤ 50%(对应整体 ≥ 2×
  decode 加速);**KILL** = MLX 实测无可复用接口 → 转路线 B 单独立项,或
  接受 Metal 上限 = MLX 上限(86 tok/s on M4 Pro)。
- **关联**:`2026-05-19-metal-mtp-qwen36-native-design.md` 已 on-hold,说明
  上 MTP 不是当前最高 ROI(因为 MLX-parity 已经达到 78% 带宽天花板)。**G2
  是先于 MTP 的瓶颈**。

### G3 🟡 CUDA — CUDA Graph efficacy 实测

- **现状**:已有 batch-size graph warmup/cache:scheduler 启动预热 batch-size
  列表(`infer/src/scheduler/cuda/core/warmup.rs:26,418`);Qwen3
  `graph_cache[batch_size - 1]`(`infer/src/model/qwen3/batch_decode.rs:171`);
  Qwen3.5 piecewise
  `graph_cache[group_idx][batch_size - 1]`(`infer/src/model/qwen35/batch_decode.rs:168`)。
- **修正**:原 PASS/KILL("上不上 bucket pool") **KILLED** — 问题本身错,
  因为当前主线已经有 batch-size warmup/cache。G3 现在只回答"现有 graph
  path 对 P5/low-c decode 是否仍有实测收益"。
- **影响**:🟡 **预测**现有 graph path 可能削低 batch decode launch overhead;但
  **EOD+19 `M_pf-graph v2` 教训**:nsys "X% of NVTX window" 必须 cross-check
  "(Y ms / per-request total time)" framing,取保守者。前次 `M_pf-graph
  Phase 0 KILL` 就是因为 graph capture 不是 SGLang 主因。
- **Evidence 强度**:🟡 — 必须先 nsys + wall-clock framing 双 cross-check。
- **License-or-kill 实验**(必跑,先于实现):
  1. Qwen3.5 decode c=1/2/4/8,成对测 `--cuda-graph` vs
     `--disable-cuda-graph`。
  2. 记录 wall-clock per-request latency(TTFT/ITL/tok/s/mean request)和
     nsys CUDA API stats(`cuLaunchKernel`,`cuGraphLaunch`,`cuStreamSynchronize`)。
  3. 与 T2 P5 trace 的 sync-bound 结论 cross-correlate:若 graph 改不了
     wall-clock,该 axis 让位给 student_rollout/backward。
- **PASS** = `--cuda-graph` 比 `--disable-cuda-graph` mean step/request
  latency ≥ 5% lower,或 c≤4 范围内 `cuLaunchKernel` time ≥ 30% lower。
  **KILL** = wall-clock < 5% 且 `cuLaunchKernel` < 30% 差 → graph cache
  已是次要 bottleneck,不继续投入 G3 实施。

### G4 🟢 Metal — GPU sampler 补 top-k / top-p / penalties

- **现状**:`backend/metal/sampling.rs:44-63` 只 greedy + temperature。CUDA
  侧 `ops/sampling.rs` + `csrc/misc/sampling.cu` 完整(temp→softmax→top-k→
  top-p→multinomial 全 GPU)。
- **影响**:**正确性**(不是性能)。当前 Metal serve 在非 greedy 场景下
  fallback CPU 或拒绝参数,与 OpenAI v1 协议有差,影响 W3/W4 agent workload。
- **Evidence 强度**:🟢 — 协议差异直接 grep 可证。
- **动作**:把 CUDA `gpu_sample_cuda` 的 logic port 到 MLX(用 MLX primitive
  组装,不需要写 .metal),复用现有 `gpu_sample_token` 入口。
- **License-or-kill**:**PASS** = OpenAI v1 sampling 参数集与 CUDA 一致,且
  与 mlx-lm 相同种子 / 相同参数下 top-k=k / top-p=p 采样分布 KS test ≤ 0.05。

### G5 🟡 CUDA — HiCache T2 disk 路径接入 scheduler 热路径

- **现状**:`kv_tier/transport/disk.rs` 在,但 `Coordinator` 未消费到 scheduler
  fetch/store 决策;`kv_tier.rs:18` 自述 "remote tier remains skeletal"。
- **影响**:🟡 **预测**对长 session(超 HBM 工作集)有 evict-and-readmit 收益,
  但对当前 short-prompt bench 收益不明。
- **Evidence 强度**:🟡 — 需 long-session bench 验证收益;参见
  `2026-05-04-kv-tier-hicache-borrowed-improvements.md`。
- **License-or-kill**:**先跑 wall-clock measure**:T0 pool 容量耗尽频率,
  以及 evict 后重 prefill 的成本。**PASS** = 真实 workload 中 T0 满载 ≥ 20%
  时间,且 evict 后 90% 命中 T1/T2(避免重 prefill)。**KILL** = T0 几乎不
  满 → 不上 T2/T3。

### G6 🔴 CUDA — RadixCache `insert` 时机端到端验证

- **现状**:`RadixCache::insert` 存在,但 scheduler 侧 insert 的 trigger 没
  trace 完;**有可能**新序列没插回,造成 follow-up 请求命中率 < 预期。
- **Evidence 强度**:🔴 — 纯推断,必须先写一个 e2e test 验证。
- **动作**:**先验证再决定**。写 `infer/tests/radix_insert_e2e.rs`:同 prompt
  连发 2 次,assert 第 2 次 `req.reusable_prefix_len == full prompt len`。
  - 若 pass → 关闭 G6,无需动作。
  - 若 fail → 找到 insert 缺失位置补一行,**禁止 over-engineer**。

### G7 🔴 Metal — `MLX_GUARD` 全进程 mutex 解锁

- **现状**:`crates/mlx-sys/src/lib.rs:12-21` 全 FFI 都 hold 同一 mutex。
- **Evidence 强度**:🔴 — 没测过 contention,可能根本不是热路径瓶颈(因为
  E1 已经被 encode 卡住了)。
- **License-or-kill**:**先 measure mutex hold time / step**,< 5% wall-clock
  → KILL(不动)。这条放在 G2 之后,因为 G2 不解决,mutex 即使解了也吃不到。

### 不排序(N/A 设计取舍)

- **3 进程 ZMQ split**(CUDA A6 / Metal A6):单进程更简单,失去 detokenize
  并行收益但拿回 IPC 开销。无证据表明 ARLE 当前 workload 卡在 detokenize。
- **FlashInfer 集成**(CUDA A4):自家 TileLang AOT 已是路径选择(见
  `2026-05-05-cuda-kernel-tilelang-unification.md`),不回头。
- **Metal KV tiering**(E3):unified memory 模型下 T0/T1 边界模糊;工作集
  超 wired limit 时考虑 backpressure / reject 而非降级。

---

## 6. 排序后的落地建议(执行顺序)

| 序 | Gap | 类型 | 投入 | 预期 wall-clock 收益 | 前置 |
|---|---|---|---|---|---|
| 1 | **G1** Metal P2.1 dual-write | 接通 | S(~1 周) | 共享 prompt TTFT −50% 起 | — |
| 2 | **G6** CUDA radix insert e2e 验证 | 验证 | XS(~半天) | bug-or-noop | — |
| 3 | **G4** Metal GPU sampler 补全 | 补齐 | S(~半周) | 正确性,非性能 | — |
| 4 | **G2 license 实验** Metal encode 复用可行性 | 实验 | M(~2 天 spike) | 决定 G2 路线 A/B | — |
| 5 | **G2 实施**(路线 A 或 B,看 4 的结果) | 实装 | L(~2-4 周) | decode 2× 起 / 或 KILL 接受现状 | G2 license |
| 6 | **G3 license 实验** CUDA graph efficacy nsys | 实验 | XS(~半天 nsys) | 决定现有 graph path 是否值得继续优化 | — |
| 7 | **G3 实施**(若 license PASS) | 实装 | M(~1 周) | 低 batch decode launch/sync overhead 削减 | G3 license |
| 8 | **G5 license measure** T0 pool 满载频率 | 测量 | S(~1 天 bench) | 决定 G5 上/不上 | — |
| 9 | **G5 实施**(若 license PASS) | 实装 | L(~2-3 周) | long-session evict 后命中 | G5 license |
| 10 | **G7 license measure** mutex hold time | 测量 | XS(~半天 profile) | 决定 G7 上/不上 | G2 完成 |

**关键纪律**:任何 license-or-kill 实验不通过,**直接关闭对应 gap,不实施**;
不变 silent 放过,不 over-engineer。每条 PASS 后产出 `docs/experience/wins/`
entry(或 KILL 时 `errors/`),用 wall-clock framing 标 ground truth。

---

## 7. 不在本计划内(显式 deferred)

- **TP > 1 实生产 NCCL collective** — F0–F4 scaffold 已在,production wiring
  是另一条 axis(`docs/plans/2026-04-28-single-node-multi-gpu.md`)。
- **DeepSeek V4 kernel** — `#1 next-model` 优先级在 ROADMAP,不属于 SGLang
  pipeline gap。
- **Spec decode / Medusa / MTP** — Metal MTP 已 on-hold 等 G2 决策;CUDA spec
  在 `M_b-tilelang-fused-draft-verify-kernel.md` 独立 track。

---

## 8. 自检 — 这份 doc 够 SOLID 吗?

- ✅ 每条 gap 标了 evidence 强度(🟢/🟡/🔴),不混淆推断 = SOLID。
- ✅ 每条落地动作有 license-or-kill 阈值,wall-clock framing 作 ground truth。
- ✅ 显式列了 deferred 项,不 silent 放过。
- ⚠️ **本身仍是 hypothesis**:G1/G3/G5 的预测影响是基于 SGLang 文档 + 当前
  bench 数推断,**没有 ARLE 自己的 A/B 对照实验数**。所以 §6 把
  "license 实验" 放在 "实施" 之前 — 不是规划洁癖,是兑现 §0 第一原则。
- ⚠️ 本 doc 不替代任何 sub-plan 的 design;G2 真实施时要起一份独立的
  `M_*` design doc(参考 `M_e1-omlx-c-multi-step-pipelining.md` 已有材料)。
