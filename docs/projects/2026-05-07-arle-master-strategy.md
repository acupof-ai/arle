# ARLE 战略主文档 — 唯一信息源(coding/agent runtime + DSV4 training)

> 2026-05-07 用户 directive:全力分析 + 顶尖大师姿态 + 至少 3 遍 review + SOLID
> 无不确定性 + **唯一的文档**。本文是 ARLE 项目的唯一战略主文档,**任何与本
> 文冲突的旧 doc 以本文为准**。
>
> **本文 3 遍 review 履历**:
> 1. 事实/逻辑(数据精确,引用准确,因果链合理)
> 2. 完整性(blind spots 覆盖,multi-dim,部署模式)
> 3. 行动清晰度(P0/P1/P2 序列依赖,kill criteria,可执行)

---

## §0 一句话核心论断

**ARLE = Rust-native AI 开发工具的双线产品**:
- **推理侧**:服务 coding/agent workload(Cursor / Claude Code / Aider / Continue),
  **目标 = 在 agent shapes(W3/W4)上 lead #2 by 1.30×**,目前 4-shape baseline
  显示 high-conc ✓ 但 long-ctx 4k/8k 落后 50%,binding constraint 是 **prefill
  路径的 launch + dispatch + scheduler overhead**(per R1)
- **训练侧**:**DSV4 架构 from-scratch repro**(以 HF replica
  `kshitijthakkar/deepseek-v4-mini-1B-init` 为唯一架构真理),**当前 ARLE
  deepseek-spec V4 操作覆盖 = 0%**(V3/MLA scaffold,需大改才能解析 V4 config)

**Defensible moat = 5 项 capability 组合**(2 项 ✓ 已具备,3 项 ⏳ 待 land):
Rust hot path ✓ + TileLang 自定义 attention ✓ + Piecewise Prefill CUDA Graph ⏳
+ Speculative decoding(Medusa)⏳ + Grammar 约束(xgrammar FFI)⏳。**没有
任何竞品同时具备 5 项**,但当前 moat 是 forward-looking 而非现实 — 必须
执行交付 3 项 plan。

**最终 product-market fit 验证**:DSV4 训出来后,**ARLE 推理侧能直接 serve
自训 DSV4** — 这是真正的"训练-推理一体"护城河,内部模型不经过 PyTorch/HF
transformers 中转。

---

## §1 服务核心目标

### §1.1 是什么

ARLE 服务以下 3 类用户(优先级排序):

| 用户类型 | 例子 | 关键需求 | 优先级 |
|---|---|---|---|
| **Coding agent**(单用户本地)| Cursor 本地 / Claude Code Mac | 单 request TTFT + ITL p99 + Mac/Linux 双 backend | **P0** |
| **Coding agent**(服务多租户)| Cursor cloud / 自托管 API | 综合 throughput + p99 + 成本 + RadixCache | **P0** |
| 自训 SOTA 模型推理(DSV4)| ARLE 内部训练 → ARLE 服务 | 训-推一体,无 PyTorch 中转,FP8 expert + MoE 路由 | **P1** |

### §1.2 不是什么

| 反向定位 | 不做的理由 |
|---|---|
| ❌ Generic LLM 服务运营商 | OpenAI/Anthropic API 复刻不是 ARLE 的 product fit |
| ❌ "在所有 canonical shape 都赢" | 4-shape benchmark 框架不反映 agent 痛点 |
| ❌ 单点最强(单点 kernel / 单点 scheduler)| moat 是 5 项组合,不是单点 |
| ❌ 量化(4bit/2bit)优先 | 量化是后置工程,先解决 BF16 路径正确性和 latency |
| ❌ 多模态(vision)| 暂时 out of scope,先做 text agent |
| ❌ Distributed 大模型(72B+)| 本 sprint 聚焦单卡 4B-32B BF16,distributed 是后期 |

---

## §2 Workload 真相

### §2.1 Coding/Agent workload 的实际形态

每 request 形态(per real agent 调用日志分析):

| 维度 | 典型值 |
|---|---|
| 输入 tokens | 5k-32k(system prompt + 文件内容 + 历史 + 工具结果)|
| 输出 tokens | 50-2000(JSON tool call 或代码,大多 < 500)|
| 并发 | c=1-8 典型(单用户 agent loop,**不是**批服务 c=64)|
| Multi-turn | 一个 session 10-50 turn,每轮 TTFT 都吃 |
| Prefix 复用 | system prompt 跨 turn 不变,**命中率 80%+** |
| 输出结构 | tool call JSON 占输出 70%+,**grammar 约束**关键 |
| Decoding 友好 | 输出短 → **speculative decoding 杠杆大** |

### §2.2 部署模式

| 模式 | 例子 | 主要优化 | ARLE 当前 |
|---|---|---|---|
| **本地单用户** | Cursor 本地 / Claude Code Mac | 单 request TTFT + ITL p99 | Metal backend(`infer/src/backend/metal/`,`crates/mlx-sys/`)+ CUDA 16GB 卡 |
| **服务多租户** | Cursor cloud / 自托管 API | 综合 throughput + p99 + 成本 | CUDA 服务(`infer/src/backend/cuda/`)|

ARLE 必须**双 backend 都强**。本文档主要分析 CUDA;Metal backend 维度参考
[`infer/src/backend/metal/AGENTS.md`](../../infer/src/backend/metal/AGENTS.md)。

### §2.3 Canonical 4-shape 与 agent shape 的相关度重排

| Canonical shape | Coding/agent 相关度 | 备注 |
|---|---|---|
| high-conc 1k/256/c=64 | **LOW** | agent 不批服务 64 路 |
| **long-ctx 4k/c=4** | **HIGHEST** | 典型小代码库 agent turn |
| **long-ctx 8k/c=4** | **HIGH** | 中等代码库 + 历史 |
| multi-tenant prefix | **MEDIUM-HIGH** | system prompt 复用,c 值仍偏批 |

**Agent 真实 shape**(见 [`docs/plans/2026-05-02-agent-load-bench-spec.md`](../plans/2026-05-02-agent-load-bench-spec.md) W3/W4):
- **W3 短-prompt multi-turn**:`base 1024 ± 32` → `tail 64 ± 8/turn` × `4 turn`,
  全局并发 16,80% warm + 20% cold,prefix-cache 命中率验证
- **W4 tool-call resume**:`8192 ± 64` 上下文 → 暂停 → 注入 `256 ± 16` tool
  output → resume 256 token,128 个 session,验证 `avoided-prefill ratio`

**Mission 阈值**(per 2026-05-02 plan):`min(throughput_ratio, p99_ratio_inverse) >= 1.30`。

---

## §3 现状 — bench 证据

### §3.1 4-shape 三方 baseline(2026-05-07 实测)

| Shape | ARLE | vLLM | SGLang | Δ vs #2 | Verdict |
|---|---:|---:|---:|---:|---|
| **high-conc** 1k/256/c=64(tok/s)| **843** ⭐ | 647 | 499 | **+30% / +69%** | ✓✓ #1 |
| **4k/c=4 longctx** TTFT | 1976 ms | 1177 | **973** ⭐ | ARLE −51% | ✗ #3 |
| **8k/c=4 longctx** TTFT | 4574 ms | **2362** ⭐ | 8054 | ARLE −48% | ✗ #2(throughput tied) |
| 8k/c=4 longctx tok/s | 103.07 | **104.74** ⭐ | 78.05 | ARLE −1.6% | ⚠ #2 持平 |
| multi-tenant TTFT | 318 ms | 573 ms | TBD | +80% vs vLLM | ✓ #1 (vs vLLM)|

**关键事实**:**ARLE 在 high-conc 已大幅领先**(对 SGLang +69%),但 **long-ctx
4k/8k TTFT 落后约 50%**,且 **agent W3/W4 shape 实测尚未落地**(只有 generic
4-shape;W3/W4 driver `scripts/bench_agent_trace.py` 已存在,跨引擎实测为 P0)。

### §3.2 R1 关键 finding(`docs/research/2026-05-07-sglang-prefill-stack-survey.md`)

SGLang 在 4k/c=4 比 ARLE 快 2.03× TTFT 的真因:

1. **NOT custom kernel**:SGLang BF16 prefill GEMM 走 `torch.nn.functional.linear`
   → cuBLAS / cuBLASLt(**与 ARLE 走同 cuBLAS 路径**)
2. **NOT operator fusion**:SGLang 用 `MergedColumnParallelLinear` /
   `QKVParallelLinear`(同 vLLM),table stakes,不是差异
3. **YES Piecewise CUDA Graph for prefill**:`PiecewiseCudaGraphRunner`,
   `disable_piecewise_cuda_graph=False` 默认,42 个 num_token sizes(4-2048
   token bucket),capture 整个 prefill 层 loop(norms / QKV / RoPE / KV write
   / attention / output proj / MLP / residual)
4. **NOT decode graph**:decode graph ARLE 也有(`batch_decode.rs:1703` B=1..8)

**结论**:SGLang 的 lead 在**dispatch + launch overhead 消除**,不在 kernel 时间。
ARLE 因 Rust hot path 本来 dispatch overhead 较小,但**仍落后 50%** —
意味着不止 launch overhead,还有 scheduling / metadata / kernel implementation 因素。

### §3.3 ROI 真相 — Graph capture 单点不够

per `docs/plans/M_pf-graph-prefill-capture.md`(commit 939008f)的诚实数学:

```
Launch floor saved: 36 layers × 7 ops/layer × 2 chunks × 7.5us = 3.8 ms
ARLE 4k TTFT 1976ms → 1969-1972 ms (only)
SGLang 4k TTFT = 973 ms
SGLang gap to ARLE = 1003 ms
Graph capture 单点解释: 3.8 / 1003 = 0.38% of gap
```

**Graph capture 单点不能闭合 4k 缺口**。剩余 ~999ms 来自:
- Cuda graph 还消除 host-side dispatch / cudarc 调用 / event 流量 / 动态调度 —
  但 ARLE Rust 本来就少,杠杆比 SGLang(Python)小
- FlashInfer paged prefill kernel vs ARLE TileLang HD128 — kernel impl 差异
- KV layout / page boundary / 调度 overhead

**真闭合路径**:graph + TileLang prefill 优化 + FP8 paged KV + 调度细节联合
组合(per `M_pf-graph-prefill-capture.md` Phase 2)。

---

## §4 架构盘点 — 推理侧

### §4.1 已具备 ✓(对 coding/agent 直接对路)

| Capability | 文件路径 | Coding/agent 价值 |
|---|---|---|
| Continuous batching scheduler | `infer/src/scheduler/cuda/` | 多 request 复用 |
| RadixCache prefix cache | `infer/src/kv_tier/` | system prompt 跨 turn 复用 |
| Decode CUDA Graph(B=1..8)| `infer/src/model/qwen3/batch_decode.rs:1703` | 短输出(tool call)latency |
| Paged KV + FP8 | `infer/src/kv_tier/`, `crates/cuda-kernels/csrc/quant/` | 长上下文内存效率 |
| TileLang HD128 attention | `crates/cuda-kernels/tools/tilelang/batch_prefill_paged_hd128.py`, `batch_decode_paged_hd128.py` | 自定义 kernel 无 Triton/CUTLASS 依赖 |
| Rust hot path | 整 `infer/`,`crates/` | 无 Python 解释器 overhead(真护城河) |
| Metal backend(Apple Silicon)| `infer/src/backend/metal/`, `crates/mlx-sys/` | Mac 本地 coding/agent |
| Qwen3 / Qwen3.5 spec | `crates/qwen3-spec/`, `crates/qwen35-spec/` | 当前主用 Qwen3-4B |

### §4.2 已 plan / 进行中 ⏳

| Capability | 计划 | 当前状态 |
|---|---|---|
| **Prefill CUDA Graph capture** | `docs/plans/M_pf-graph-prefill-capture.md`(commit 939008f)| Phase 0 audit by codex 0:0 进行中 |
| Agent W3/W4 bench harness | `docs/plans/2026-05-02-agent-load-bench-spec.md` + `scripts/bench_agent_trace.py` | spec ✓ + driver ✓,跨引擎实测未跑 |

### §4.3 缺口 ❌(对 coding/agent 关键)

| 缺口 | 影响 | 优先级 | 备注 |
|---|---|---|---|
| Speculative decoding(Medusa)| tool call 短输出 tok/s × 2-3(条件 acceptance ≥70%)| P1 | Medusa 多头优先 EAGLE(降数据/训练风险)|
| Grammar 约束生成(xgrammar)| JSON tool call 强制有效 | P1 | **FFI 包 xgrammar C++** 优先,Rust 重写 deferred |
| 32k+ long-ctx 验证 | Claude Code class 大代码库 | P2 | 未 benched |
| Tool call 快路径 | 结构化短输出 latency | P2 | 通用 streaming 不够优 |
| MoE 推理 production | Qwen3.5-MoE / DeepSeek 服务 | P2 | substrate 已有,需 prod 验证 |

### §4.4 已 KILLED — 不要重做

| 实验 | 原因 | 禁忌 |
|---|---|---|
| M_pf-gemm Phase 0(cuBLAS algo 选择)| top-1 已最优,−1.3% 在噪声内 | 不要再做 cuBLAS algo 实验 |
| M_pf-fuse Phase 0(gate-up fusion)| +1.5% regression(cuBLAS 22k N 反而吃亏)| 不要再做单点 GEMM fusion |
| M_b.2.2 split-KV BF16(opt-in)| bench regression(ITL +31.6%, tok/s -18.8%)+ e2e hang 33m | 不要 enable INFER_TILELANG_BF16_SPLIT_KV |
| M_pf-gemm Phase 2/2.5(custom prefill GEMM)| ⏸ DEFER(R1 证 SGLang 用 cuBLAS 也赢)| graph capture 落地后再评估,不是 binding constraint |

---

## §5 架构盘点 — 训练侧

### §5.1 DSV4 唯一架构真理(HF replica)

**Authoritative source**:[`kshitijthakkar/deepseek-v4-mini-1B-init`](https://huggingface.co/kshitijthakkar/deepseek-v4-mini-1B-init)
config.json,**已下载到** `infer/models/dsv4-mini-1B-init/`(配置 + tokenizer
完成,model.safetensors 2GB 下载完整)。

完整 `config.json`:

```json
{
  "architectures": ["DeepseekV4ForCausalLM"],
  "model_type": "deepseek_v4",
  "dtype": "bfloat16",

  "vocab_size": 129280,
  "hidden_size": 1024,
  "num_hidden_layers": 24,
  "num_attention_heads": 16,
  "num_key_value_heads": 1,
  "head_dim": 64,
  "hidden_act": "silu",
  "swiglu_limit": 10.0,

  "q_lora_rank": 384,
  "o_lora_rank": 384,
  "o_groups": 4,
  "qk_rope_head_dim": 32,

  "n_routed_experts": 16,
  "n_shared_experts": 1,
  "num_experts_per_tok": 2,
  "moe_intermediate_size": 512,
  "routed_scaling_factor": 1.5,
  "norm_topk_prob": true,
  "scoring_func": "sqrtsoftplus",
  "topk_method": "noaux_tc",

  "index_n_heads": 8,
  "index_head_dim": 64,
  "index_topk": 128,
  "num_hash_layers": 2,
  "sliding_window": 64,
  "compress_ratios": [0,0,4,96,4,96,4,96,4,96,4,96,4,96,4,96,4,96,4,96,4,96,4,0],
  "compress_rope_theta": 160000.0,

  "hc_mult": 4,
  "hc_sinkhorn_iters": 20,
  "hc_eps": 1.0e-6,

  "num_nextn_predict_layers": 1,

  "max_position_embeddings": 1048576,
  "rope_theta": 10000.0,
  "rope_parameters": {
    "rope_type": "yarn", "factor": 16.0,
    "original_max_position_embeddings": 65536,
    "beta_fast": 32.0, "beta_slow": 1.0
  },

  "rms_norm_eps": 1.0e-6, "initializer_range": 0.02,
  "tie_word_embeddings": false, "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": null
}
```

**关键架构特征**:
- **混合注意力**(per-layer `compress_ratios`):`0` → SWA(window 64);
  `4` → CSA(stride-4 + Lightning Indexer top-128);`96` → HCA(stride-96 dense)
- **V4 abandons MLA**:Q-LoRA(384)+ 单 KV 头(num_kv_heads=1)+ O-LoRA grouping(rank 384,o_groups=4)
- **Lightning Indexer**:`index_n_heads=8`,`index_head_dim=64`,`index_topk=128`,`num_hash_layers=2`
- **MoE**:16 routed + 1 shared,top-2/16,scoring `sqrtsoftplus`(V4 新),`noaux_tc`
- **mHC(Manifold-Constrained Hyper-Connections)**:`hc_mult=4` 流,Sinkhorn 20 步投影
- **MTP**:`num_nextn_predict_layers=1`(spec-decode-friendly)
- **YaRN**:base 65536 × factor 16 = **1,048,576 tokens** 有效上下文
- **双 RoPE**:`rope_theta=10000`(dense)/ `compress_rope_theta=160000`(compressed)

**实际参数估算**(BF16 untied):~836M params / ~280M activated(约 33.5% sparsity ratio)。

### §5.2 ARLE 训练侧现状(`crates/deepseek-spec/` + `crates/train/`)

per R2 audit(`docs/research/2026-05-07-deepseek-spec-truth-audit.md`,commit fa6a5ea):

| 维度 | 状态 |
|---|---|
| **`DeepSeekConfig` 解析 HF V4 config** | **0% 操作覆盖**(必填 V3 字段 `kv_lora_rank` / `qk_nope_head_dim` / `v_head_dim` / `first_k_dense_replace` 缺失阻断 parse)|
| Schema field 覆盖 | 47%(21/45 keys 有同名或别名)|
| 缺字段(关键)| `swiglu_limit` `o_lora_rank` `o_groups` `scoring_func` `topk_method` `index_*`(4 项)`num_hash_layers` `sliding_window` `compress_ratios` `compress_rope_theta` `hc_*`(3 项)等 |
| Tensor name 覆盖 | <30% by family,前缀大多错(V3 `model.layers.N.self_attn.kv_a_proj_with_mqa` vs V4 `layers.L.attn.wq_a/q_norm/wq_b/wkv/kv_norm/wo_a/wo_b/attn_sink/...`)|
| Shard 策略 | 不处理 V4 single-KV-head GQA / O-LoRA grouping / mHC streams / Lightning Indexer / MTP |
| MLA references | `DeepSeekMlaTensorNames` `kv_a_proj_with_mqa` `kv_b_proj` `kv_lora_rank` 全部 V3 only(应标 deprecated)|
| `crates/train/src/deepseek.rs` | V3 假设,需新 DSV4 实现路径 |
| Muon optimizer | ❌ 未实现(Moonlight 论文参考)|
| FP8 dense matmul(TransformerEngine 等价)| ❌ 推理侧 cuBLAS BF16 已通,FP8 训练需新增 |
| Sinkhorn 投影(mHC)| ❌ 未实现 |
| MTP loss(t+1 + t+2 双预测)| ❌ 未实现 |
| Sequence packing + segment_ids | ❌ 未实现 |

**结论**:**ARLE 训练栈不能跑 DSV4** — 当前是 V3 scaffold。需要新建
`DeepSeekV4Config` + `DeepSeekV4TensorNames` + 完整 V4 实现,**不允许 retrofit
V4 进 V3 结构**。

---

## §6 创新组合护城河

### §6.1 5 项 capability 组合

| Capability | ARLE | SGLang | vLLM | TRT-LLM |
|---|:-:|:-:|:-:|:-:|
| **Rust hot path**(整 stack) | ✓ | ✗ Python | ✗ Python | C++ engine + Python frontend |
| **Custom TileLang HD128 prefill+decode attention** | ✓ | (Triton)| (Triton/CUDA)| (CUTLASS)|
| Piecewise Prefill CUDA Graph | ⏳ plan(939008f)| ✓ default ON | partial | partial(engine 编译)|
| Speculative decoding(Medusa)| ⏳ | ✓ | ✓ | ✓ |
| Grammar 约束(xgrammar)| ⏳ | ✓ | ✓ | partial |

**单点比 → 没赢**:SGLang 也有 graph + spec + grammar(Python overhead)。**ARLE 独家 = Rust hot path + TileLang custom kernel**。

### §6.2 Moat 真相

- **现 moat**:Rust + TileLang(2 项)
- **Forward-looking moat**:5 项全 land
- **Land 风险**:

| Plan | LOC 估 | 风险 | 替代降风险方案 |
|---|---|---|---|
| Prefill graph(Phase 0 939008f)| ~200 | 中(scope 已限,7 项 kill criteria)| 无 |
| Spec decode(Medusa 多头)| ~500-800 | **高**(数据/训练 + kernel + 调度)| Medusa 优先 EAGLE 后置 |
| xgrammar 集成 | FFI ~200 / Rust 重写 ~1000+ | **高**(FSM 复杂)| **FFI 包 C++** 优先,Rust 重写 deferred |

### §6.3 训-推一体 — 真正的护城河

DSV4 训练完(per §8 plan),**ARLE 推理侧能直接 serve 自训权重**:
- 内部模型不经过 PyTorch / HF transformers 中转
- BF16 / FP8 expert 权重直接加载到 ARLE Rust runtime
- 验证路径:训练 checkpoint 在 ARLE serve 出 valid logits,与 reference 实现 numerical match

**这是其他竞品没有的**:vLLM/SGLang/TRT-LLM 都是"服务他人训出的模型"。ARLE
是"自训 + 自服"完整闭环。当前不是 moat(因为 ARLE 没自训过),**land DSV4
后是 moat**。

---

## §7 解决方案 — 推理侧 P0/P1/P2

### §7.0 序列依赖图(关键)

```
P0.0 (W3/W4 bench harness 跨引擎实测)
   │
   ├─▶ P0.1 (M_pf-graph Phase 0 实施 + 验证)
   │     │
   │     └─▶ P1.x (Phase 1 piecewise + Phase 2 + 内核组合)
   │
   ├─▶ P1.1 (Spec decode Medusa)
   │
   └─▶ P1.2 (xgrammar FFI)
```

**P0.0 必须先做**:没 W3/W4 跨引擎数据,P0.1 落地后**不知道 agent shape 是否真受益**。

### §7.1 P0.0 — Agent W3/W4 跨引擎 baseline(必须先做)

- **Driver**:`scripts/bench_agent_trace.py`(已实现)
- **Spec**:`docs/plans/2026-05-02-agent-load-bench-spec.md`(W3 + W4 完整)
- **跨引擎**:ARLE / SGLang / vLLM 三方 H1 baseline
- **Acceptance**:三方 baseline 表 + 每方 mission_margin(`min(throughput_ratio, p99_inverse) >= 1.30`)
- **Owner**:Claude(运行)+ general-purpose(协助)
- **GPU 串行**:与 codex 协调,~30-60 min 总时间

### §7.2 P0.1 — Prefill CUDA Graph capture Phase 0

- **Plan**:`docs/plans/M_pf-graph-prefill-capture.md`(commit 939008f)
- **Scope**:opt-in `INFER_PREFILL_GRAPH=1`,单 bucket 2048-token,~200 LOC,只 capture body(GPU-only)
- **License threshold**:1850-1950ms TTFT 进 Phase 1;<1700ms 立即 promote;
  <10ms 改善 + nsys 无 launch overhead 减少 → KILL
- **Owner**:codex 0:0(进行中,audit phase)+ Claude(review/bench/wins)
- **依赖**:P0.0 完成才有 agent shape 验证(否则只能 generic 4k bench)

### §7.3 P0.2 — Phase 1 piecewise + Phase 2 内核组合(若 P0.1 license)

- 42 num_token buckets piecewise(mirror SGLang)
- Phase 2:graph capture + TileLang HD128 + FP8 paged KV combine
- Acceptance:4k TTFT < 748 ms(SGLang × 1/1.30)+ 8k TTFT < 1816 ms

### §7.4 P1.1 — Speculative decoding(Medusa 优先)

- **Medusa 多头**(单模型加多个 prediction head)优先 EAGLE(独立 draft)
- 复用 ARLE Rust runtime + TileLang verify kernel
- 目标:tool call 短输出(50-500 tok)tok/s × 2-3,acceptance ≥70%
- LOC 估:500-800
- Owner:codex(impl)+ Claude(plan + review)
- 触发:P0.1 license 通过

### §7.5 P1.2 — Grammar 约束(xgrammar FFI 优先)

- **FFI 包现成 xgrammar C++ 库**(降风险),不重写 Rust
- Hook 到 ARLE sampling(`infer/src/scheduler/cuda/sampler.rs` 等价路径)
- 目标:JSON tool call 100% 有效 + 解码 overhead < 10%(简单 schema)
- LOC 估:200(FFI)
- 触发:P0.1 license + P1.1 plan 草

### §7.6 P2(后置)

- 32k-128k long-ctx 优化(Claude Code class)
- Tool call fast path(短结构化输出 lazy KV write)
- MoE 推理 production(Qwen3.5-MoE / DSV4 自训 weights)
- 量化(AWQ/GPTQ/INT8)
- Vision/Multi-modal
- Distributed inference(TP/PP)
- Metal backend coding/agent 优化(单独 track,本文 out of scope)

### §7.7 KILL / DEFER(明确停做)

| 项 | 决定 | 原因 |
|---|---|---|
| ❌ M_pf-gemm Phase 0 | KILLED | top-1 cuBLAS 已最优 |
| ❌ M_pf-fuse Phase 0 | KILLED | regression |
| ❌ M_b.2.2 split-KV BF16 | KILLED | regression + e2e hang |
| ❌ M_b.3 G1(segment-aware mixed batch)| **DEFERRED** | 非 binding constraint per R1,P0 graph capture 后再评估 |
| ⏸ M_pf-gemm Phase 2/2.5(custom GEMM)| **DEFERRED** | R1 证 SGLang 用 cuBLAS,不是 binding;graph 后再评估 |
| ⏸ "Win at every canonical shape" 框架 | **REPLACED by W3/W4 框架** | agent shape 才是 product fit |
| ⏸ 持续优化 high-conc | **DEFEND only** | 已 +69%,守住即可 |

---

## §8 解决方案 — 训练侧 P0/P1/P2

### §8.0 序列依赖图

```
P0 (deepseek-spec V4 重构) ──▶ P1.0 (推理跑通 HF init 权重 numerical match)
   │                                │
   └─▶ P1.1 (Muon + FP8 dense)      └─▶ P1.2 (从 init 权重 nano 训练 跑 1 step OK)
              │
              └─▶ P1.3 (Sinkhorn mHC + MTP loss + sequence packing)
                          │
                          └─▶ P2 (40B token from-scratch pre-train)
```

### §8.1 P0 — DeepSeek V4 spec 重构(deepseek-spec)

per R2 audit(commit fa6a5ea):
- **新建** `crates/deepseek-spec/src/v4.rs` 含 `DeepSeekV4Config` + `DeepSeekV4TensorNames`
- **保留** `DeepSeekMlaTensorNames` `kv_*_proj` 等为 V3 legacy(标 deprecated 但不删)
- **覆盖 HF config 全 45 字段**(尤其 `swiglu_limit` `o_lora_rank` `o_groups`
  `scoring_func` `topk_method` `index_*` `compress_*` `hc_*` `num_hash_layers`)
- **新 tensor name map**:`embed.weight` / `head.weight` / `layers.L.attn.wq_a` /
  `q_norm` / `wq_b` / `wkv` / `kv_norm` / `wo_a` / `wo_b` / `attn_sink` /
  `compressor.{wkv,wgate,ape,norm}` / `indexer.{wq_b,weights_proj,compressor.*}` /
  `ffn.gate.{weight,bias,tid2eid}` / `ffn.experts.J.{w1,w2,w3}` / `mtp.K.*` / `hc_*`
- **Shard 策略**:V4 single-KV-head 复制;Q-LoRA dim 0 query head;O-LoRA 按
  `o_groups`;mHC / norm / attn_sink 复制;Indexer head 数除 TP;MoE 用 `ExpertParallel`
- **Acceptance**:能 parse `infer/models/dsv4-mini-1B-init/config.json` 并 layer
  tensor names byte-match HF safetensors index
- **LOC 估**:400-600(新 module + tests + V3 legacy 标记)
- **Owner**:codex(impl)+ Claude(review)

### §8.2 P1.0 — 推理 HF init 权重 numerical match

- 加载 `infer/models/dsv4-mini-1B-init/model.safetensors`
- ARLE Rust runtime 前向 → logits
- 对照 HF transformers reference impl(`infer/models/dsv4-mini-1B-init/code/deepseek_v4/modeling_deepseek_v4.py`)
- Acceptance:logits cosine similarity > 0.999 over 100-token prompt
- **意义**:证明 ARLE DSV4 推理路径正确,training pipeline 验证用
- 触发:P0 完成

### §8.3 P1.1 — Muon optimizer + FP8 dense matmul

- **Muon**:per Moonlight 论文,Newton-Schulz 5 步正交化,hidden 2D weights only
  - LR ≈ 0.02,weight_decay 0.1,momentum 0.95,Nesterov
  - 实现位置:`crates/autograd/`(已有 AdamW + lr-schedule + AdamW codec)
  - LOC 估:300-500
- **FP8 dense matmul**:
  - 推理侧 cuBLAS BF16 已通,训练新 path
  - 用 cuBLAS Lt FP8(E4M3)+ 动态 per-tensor scaling
  - 主权重 + Muon momentum 留 FP32(精度需要),CPU offload(per `dsv4-small-repro.md §4.3`)
  - LOC 估:300-500
- 触发:P0 完成

### §8.4 P1.2 — 从 init 权重 nano training 跑 1 step

- 用 HF init 权重起步(不是随机初始化),验证 Muon + FP8 + 数据 pipeline 整体跑通
- 1 step forward + backward + optimizer update,loss decrease 检查
- Acceptance:1 step 不 crash,loss 下降 > 0,GPU 内存 < 14 GiB
- 触发:P1.0 + P1.1 完成

### §8.5 P1.3 — Sinkhorn mHC + MTP loss + sequence packing

- **mHC Sinkhorn 投影**:4×4 矩阵,20 步迭代,Birkhoff polytope
  - 微秒级 forward overhead,可忽略
  - LOC 估:100
- **MTP loss**:t+1 + t+2 双预测,辅助损失
  - LOC 估:200
- **Sequence packing + segment_ids**:多文档同 seq 拼接,attention 不 leak
  - LOC 估:200
- 触发:P1.2 完成

### §8.6 P2 — 40B token from-scratch pre-train

- per `docs/plans/dsv4-small-repro.md §3-§4`(数据源 + pipeline)
- 数据:FineWeb 16B + Chinese-Fineweb-Edu 12B + Stack-v2-dedup 8B + OpenWebMath 2B + RedPajama-arXiv 1.5B + OPUS 0.5B
- Curriculum:4k seq(0-2B 暖身)→ 4k(2-25B 主 dense)→ 8k(25-35B sparse switch)→ 16k(35-40B 长上下文)
- 步时 0.5 s × 610k step ≈ **85 hr ≈ 4-7 day** 连续(per `dsv4-small-repro.md §4.5`)
- 触发:P1.3 完成 + 全栈 cargo test green

---

## §9 决策点 — Top 7 排序(用户拍板)

按战略关键度排序:

| # | 决策 | 关键度 | Default 推荐 |
|---|---|---|---|
| **D1** | 接受 ARLE 双线产品定位(coding/agent runtime + DSV4 from-scratch)? | ★★★★★ | **YES** |
| **D2** | P0.0 W3/W4 跨引擎 baseline 先做(P0.1 graph capture 之前)? | ★★★★★ | **YES** |
| **D3** | M_pf-graph Phase 0 是 P0.1(per codex 939008f plan)? | ★★★★★ | **YES** |
| **D4** | 训练侧 P0 = 重构 deepseek-spec 为 V4(新建 V4 module,V3 legacy 保留)? | ★★★★★ | **YES** |
| **D5** | Spec decode 选 Medusa 多头优先 EAGLE 后置? | ★★★★ | **YES** |
| **D6** | xgrammar FFI 优先 Rust 重写后置? | ★★★★ | **YES** |
| **D7** | M_pf-gemm Phase 2 / M_b.3 G1 全部 DEFER(非 binding constraint)? | ★★★ | **YES** |

---

## §10 已知不确定性(透明)

| 不确定性 | 当前判断依据 | 解除条件 |
|---|---|---|
| Prefill graph 在 ARLE Rust runtime 真实 ROI | Codex plan 数学 + R1 evidence | Phase 0 实测 |
| Spec decode acceptance rate(Qwen3-4B coding tool call)| 行业典型 70-80% | 训练 / 选 draft 后实测 |
| 32k-128k long-ctx ARLE 表现 | 未 benched | P2 工作 bench |
| FlashInfer paged prefill vs TileLang HD128 kernel-time | 未对照 bench | nsys A/B + ncu |
| Metal backend coding/agent 现状 | 未深入分析 | 单独 Metal track plan |
| HF replica config 是否 byte-exact = DSV4-Pro/Flash 缩比 | 未对照官方 | DeepSeek 官方 config + 技术报告 |
| `hc_mult` `scoring_func` 在 transformers 源码 | 未 review | transformers ≥ 4.57.1 review |
| FP4 expert(`expert_dtype=fp4`)在 sm_89 不支持 | 已 plan(用 FP8 替代)| RTX 4070Ti SUPER 硬件限制 |
| Muon Rust 实现复杂度 | Moonlight 论文 + PyTorch impl | 实现后实测 |
| W3/W4 跨引擎 baseline | 未跑 | P0.0 实施 |

---

## §11 Cross-references(本文 supersede 一切冲突)

### 推理侧
- M_world1 roadmap:[`docs/plans/M_world1-30-percent-lead-roadmap.md`](../plans/M_world1-30-percent-lead-roadmap.md)
- W3/W4 bench spec:[`docs/plans/2026-05-02-agent-load-bench-spec.md`](../plans/2026-05-02-agent-load-bench-spec.md)
- W3/W4 mission:[`docs/projects/2026-05-02-agent-load-mission-expansion.md`](./2026-05-02-agent-load-mission-expansion.md)
- M_pf-graph plan:[`docs/plans/M_pf-graph-prefill-capture.md`](../plans/M_pf-graph-prefill-capture.md)
- R1 SGLang prefill stack survey:[`docs/research/2026-05-07-sglang-prefill-stack-survey.md`](../research/2026-05-07-sglang-prefill-stack-survey.md)
- Prefill graph readiness audit:[`docs/research/2026-05-07-arle-prefill-graph-readiness-audit.md`](../research/2026-05-07-arle-prefill-graph-readiness-audit.md)
- ARLE prefill GEMM callgraph:[`docs/research/2026-05-07-arle-prefill-gemm-callgraph.md`](../research/2026-05-07-arle-prefill-gemm-callgraph.md)
- 4-shape baseline P0.1 wins:`docs/experience/wins/2026-05-07-m_world1-p0-sglang-baseline.md`
- 4-shape baseline P0.2 wins:`docs/experience/wins/2026-05-07-m_world1-p0-sglang-baseline-extended.md`

### 训练侧
- DSV4 small-scale repro(rewriting 中):[`docs/plans/dsv4-small-repro.md`](../plans/dsv4-small-repro.md)
- DSV4 small-VRAM substrate:[`docs/plans/2026-05-05-deepseek-v4-small-substrate.md`](../plans/2026-05-05-deepseek-v4-small-substrate.md)
- deepseek-spec truth audit(R2):[`docs/research/2026-05-07-deepseek-spec-truth-audit.md`](../research/2026-05-07-deepseek-spec-truth-audit.md)
- HF replica weights:`infer/models/dsv4-mini-1B-init/`(下载完成)
- HF replica HF page:https://huggingface.co/kshitijthakkar/deepseek-v4-mini-1B-init

### 代码 entry points
- DSV4 推理(待新建):`infer/src/model/dsv4/`(还未存在)
- DSV4 训练(待新建):`crates/train/src/dsv4.rs`(`deepseek.rs` V3 假设)
- DSV4 spec(需重构):`crates/deepseek-spec/src/v4.rs`(还未存在)
- 推理 HTTP server(已有):`infer/src/http_server/`
- Decode CUDA Graph(已有):`infer/src/model/qwen3/batch_decode.rs:1703`
- Prefill TileLang HD128(已有):`crates/cuda-kernels/tools/tilelang/batch_prefill_paged_hd128.py`

---

## §12 Rules

1. **本文是 ARLE 战略唯一信息源**。其他文档不允许复制本文 §0-§10 内容,只能 link。
2. **架构数据(§5.1)出现差异 = 修改 truth**。HF replica config 改了,本文 §5.1 同步更新,其他 doc 自动跟进。
3. **新 plan 必须从本文 §7-§8 派生**(P0/P1/P2 序列)。
4. **新实验必须 cite 本文的 KILL 列表**(§4.4)— 不要重做已 KILLED 的事。
5. **决策(§9)D1-D7 的"YES default"在用户未明确否决前生效**。
6. **不确定性(§10)必须显式列出而非隐藏**。新发现的不确定性立即加入 §10。
7. **Supersede 规则**:本文超过的旧 doc 在自身 header 加 `> ⚠️ SUPERSEDED by docs/projects/2026-05-07-arle-master-strategy.md` 标记,但不删 — 历史 reasoning 保留。

---

## §13 一句话总结(再次,加粗)

**ARLE 是双线产品双第一要义:推理侧 = Rust-native coding/agent runtime
(W3/W4 mission_margin ≥ 1.30);训练侧 = DSV4 架构 from-scratch repro(HF
replica 1B-init 为唯一架构真理,当前 ARLE 训练栈是 V3 scaffold 必须重构);
Defensible moat = 5 项 capability 组合(Rust + TileLang ✓,Prefill graph
+ Spec decode + Grammar ⏳),land 后是真正的"训-推一体"护城河。**
