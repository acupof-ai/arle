# DeepSeek-V4 Small-VRAM Substrate Plan

**Reference:**
[`crates/deepseek-spec/`](../../crates/deepseek-spec/) (架构 ground truth) ·
[`docs/plans/2026-05-01-mla-kernel-design.md`](2026-05-01-mla-kernel-design.md) ·
[`docs/plans/2026-05-01-longctx-spec-decode-phase2.md`](2026-05-01-longctx-spec-decode-phase2.md) ·
[`infer/src/kv_tier/AGENTS.md`](../../infer/src/kv_tier/AGENTS.md) ·
[`infer/src/model/AGENTS.md`](../../infer/src/model/AGENTS.md) ·
[`infer/src/backend/AGENTS.md`](../../infer/src/backend/AGENTS.md)
**Owner:** unassigned
**Status:** drafted, not yet started

---

## 0. Premise

- **架构 ground truth = `crates/deepseek-spec/`** (MLA + DeepSeekMoE + MTP)。spec crate 当前以 V3 reference config 作为 known-good fixture, AGENTS.md 中定位为 *DeepSeek V4 readiness scaffold*。**V4 正式公开后的增量** (预计在 FP8 scaling 元数据形态、router 正则、context 长度三处) 通过补 spec hook 收敛, 不改本计划路线。
- **"infer-first" 不变**: 本计划交付物是 *runtime substrate* —— 一个用来给 `infer/` 跑通 V4 family 的小模型, 不是 SOTA 玩家。
- **PyTorch 用在 cold path 合规**: AGENTS.md 禁止的是 *热路径引入 PyTorch*。pretrain 是一次性 cold-path 工作, 走外部 PyTorch 栈合规; 推理热路径仍 100% Rust + CUDA / Metal。

## 1. Goal / Non-Goals

**Goal**

1. 产出 2 个 checkpoint, 分别命中 4–6 GB 与 8–12 GB 卡 sweet spot。
2. 端到端打通 `deepseek-spec → 外部 pretrain → safetensors → infer/src/model/deepseek.rs → CUDA / Metal MLA + MoE kernel → 量化 → 小卡 bench`。
3. 把 MLA、DeepSeekMoE、MTP 三件套在自家 runtime 的首套数值与吞吐 bench 落 [`docs/experience/wins/`](../experience/wins/), 作为 V4 正式发布前的"热身轨道"。

**Non-Goals**

- 不与 DeepSeek-V3-Lite / Distill 在公开 benchmark 正面对标。
- 不投入 RLHF / 复杂对齐, 产物以 base + 轻 SFT 为限。
- 不在 [`crates/autograd/`](../../crates/autograd/) 内跑万亿 token pretrain (autograd crate 不为此而生); 仅做小规模数值对齐验证。

## 2. Model SKUs (three sizes)

> `tied_embeddings = true` 在所有 SKU 上启用 —— 小模型对 vocab 矩阵冗余敏感的工程取舍, 与 V3 默认 (`tie_word_embeddings = false`) 不同。spec crate 的 `lm_head_tensor_name()` 已经支持。

| | nano (fixture) | A · Tiny-Dense-MLA | B · Mini-DeepSeekMoE |
|---|---|---|---|
| 用途 | unit-test fixture / CI | 4–6 GB 卡 (GTX 1660 / RTX 3050 4 G) | 8–12 GB 卡 (RTX 3060 12 G / 4060 Ti 16 G) |
| 总参 | ~12 M | ~200 M | ~450 M |
| 激活参 (MoE only) | – | – | ~190 M |
| `vocab_size` | 4 096 | 65 536 | 65 536 |
| `tie_word_embeddings` | true | true | true |
| `hidden_size` | 256 | 768 | 1 024 |
| `num_hidden_layers` | 2 | 12 | 18 |
| `num_attention_heads` | 4 | 12 | 16 |
| `num_key_value_heads` | 4 | 12 | 16 |
| `qk_nope_head_dim / qk_rope_head_dim / v_head_dim` | 32 / 16 / 32 | 64 / 32 / 64 | 64 / 32 / 64 |
| `q_lora_rank` | None | None | 384 |
| `kv_lora_rank` | 64 | 128 | 192 |
| `intermediate_size` (dense FFN) | 512 | 3 072 | 2 816 |
| `moe_intermediate_size` | – | – | 768 |
| `num_experts / num_experts_per_tok` | 0 | 0 | 32 / 4 |
| `n_shared_experts` | – | – | 1 |
| `n_group / topk_group` | – | – | 1 / 1 (group routing 关) |
| `first_k_dense_replace` | – | – | 2 |
| `num_nextn_predict_layers` (MTP) | 0 | 0 (留 hook) | 1 |
| `max_position_embeddings` | 1 024 | 4 096 (YaRN→16 k) | 4 096 (YaRN→16 k) |
| Pretrain 体量 | 0.5 B token | 30 B token | 80 B token |
| BF16 权重 | ~24 MB | ~400 MB | ~900 MB |
| INT4 权重 (GPTQ) | ~6 MB | ~110 MB | ~250 MB |

**SKU 选型背后的核心取舍**

- **MLA 在小 SKU 上的收益是 KV 压缩, 不是 FLOPs**。SKU-A 标准 GQA-12 KV ≈ `2 × 12 × 96 × 4 dtype = 9 KB/token`; MLA latent (`kv_lora_rank + qk_rope_head_dim` = 160) BF16 = `0.32 KB/token`, **~28× 削减**。这就是 small-VRAM long-context 的命脉。
- **MoE 总参权重必须全量驻留** (路由不可预测), 所以 SKU-B 在 BF16 下 ~900 MB 权重 + activations + KV 后, 4 k context 下需要 ~1.6 GB 显存预算 → 8 GB 卡舒适, 6 GB 卡需 INT4。
- **`n_group=1, topk_group=1`** (即 group routing 关闭): 32 routed expert 体量太小, V3 的 group-limited routing (256 expert / 8 group) 规模不适用, 简化为 plain top-4。这是相对 V3 的有意识缩减, spec crate 的 `topk_group` 字段已经支持任意值。
- **MTP 仅在 SKU-B 启用**: 把 spec-decode 的 proposer / verifier 同源路打通, 与 [`docs/plans/2026-05-01-longctx-spec-decode-phase2.md`](2026-05-01-longctx-spec-decode-phase2.md) 的 token tree 数据结构共用。

## 3. Tokenizer & Data

- **65 k 词表 BPE**, 使用 [`crates/train/examples/build_bpe_tokenizer.rs`](../../crates/train/examples/build_bpe_tokenizer.rs) 已有的 trainer。容量为 V3 的一半 (V3 = 129 280), 节省约 60 MB BF16 vocab embedding。
- **数据配比** (按 token 体积): 英 50% / 中 30% / code 15% / math 5%。
  - 英: The Pile v2 (去 Books3) + FineWeb-Edu 高质量子集
  - 中: Wudao 开放子集
  - code: The Stack v2 dedup (python/rust/c/cpp/go/js/ts, 去许可问题分区)
  - math: OpenMathInstruct + OpenWebMath
- **Packing**: 4 096-长度 sequence packing; 不跨 doc 截断 attention (packed-attn mask)。
- **nano 数据**: 从 SKU-A 数据集随机 0.5 B token 子采样, 仅用于跑通流水线。

## 4. Pretrain Stack (external, cold path)

**主路: PyTorch + FSDP-zero3 + DeepSpeed-MoE expert-parallel**, repo 不入树, 单独 git submodule 或同账号下另一 repo (`agent-infer-pretrain`)。

**算力 sanity** (Chinchilla 6 FLOPs/param/token, H100 BF16 50% MFU ≈ 4 PFLOPS/卡):

| SKU | 总 FLOPs | 8 × H100 GPU-hours |
|---|---|---|
| nano | 12 M × 0.5 B × 6 = 3.6e16 | < 0.02 |
| A · 200 M | 200 M × 30 B × 6 = 3.6e19 | ~2.5 |
| B · 450 M | 450 M × 80 B × 6 = 2.16e20 | ~15 |

**整个 pretrain 的 GPU 预算 ≤ 一周 8 × H100 算力额度, 瓶颈是 kernel 工程量, 不是 GPU 时间。**

- 优化器: AdamW, β=(0.9, 0.95), wd=0.1, cosine LR + 2 k warmup, peak LR=3e-4 (SKU-A) / 2e-4 (SKU-B)。
- 精度: BF16 mixed (master weights)。**FP8 (E4M3 fwd / E5M2 bwd, per-tile fine-grained scaling) 作为 SKU-B 的可选试验路**, 仅 H100+ 启用 —— 这是把 V4-FP8 路径在我们这边落地的预演, 对 runtime FP8 weight load 直接受益。
- MoE: auxiliary-loss-free 路径 (DS-V3 的 router bias adjustment), 保留 entropy 日志做事后审计; 备开关到 standard aux-loss 兜底。
- 中段评估: 每 2 B token 在 hellaswag / arc-easy / mmlu-tiny / humaneval-tiny / cmmlu-tiny 上跑一遍 (lm-eval-harness 走 PyTorch, 不污染 runtime)。

**备路** (明确为 stretch, 不阻塞主线): 在 [`crates/autograd/`](../../crates/autograd/) 上跑 nano SKU 的 forward+backward, 与 PyTorch reference 数值对齐到 1e-4。这是"runtime-led train"能力的小验证里程碑, **不是 SKU-A/B 的训练路径**。

## 5. Weight Export → safetensors

- 导出脚本 (PyTorch 端 utility, 落外部 pretrain repo) 从 PyTorch state_dict 直出 safetensors, **命名严格遵循 `DeepSeekConfig::layer_tensor_names() / mtp_tensor_names() / shard_for_global_tensor()` 当前定义的全套字符串**。
- Round-trip 校验: safetensors 重新 load 回 PyTorch, 参数 bitwise 相同 (atol=0)。
- `config.json` 字段名匹配 `RawDeepSeekConfig` 的 serde 别名 (`num_kv_heads`、`n_routed_experts` alias 已就位), 保证 `DeepSeekConfig::from_json_file` 直接认。
- **Sanity 验证**: 用 `deepseek-spec` 的 `shard_for_global_tensor` + `layer_tensor_names(i).shard_for(...)` + `mtp_tensor_names(0).layer.shard_for(...)` 把每个 tensor 名都过一遍 —— 任意 tensor 名未被任一 shard 规则命中即认为命名错位, build 失败。
- **`q_lora_rank=None` 路径需要新单元测试**: 当前 spec crate 的 V3 fixture `q_lora_rank=1536`, `q_proj` 直连分支没有真测过; SKU-A/nano 必须打开这条路径, 前置补 `parses_tiny_dense_no_q_lora` 测试 —— 进 §10 出口清单。

## 6. Runtime Adaptation — `infer/src/model/deepseek.rs`

新建模型文件, 遵守 [`infer/src/model/AGENTS.md`](../../infer/src/model/AGENTS.md) 的 `ModelForward + weights vs state` 拆分契约。

### 6.1 MLA Op

- 新增 `infer/src/ops/attention/mla.rs`。CUDA 路径优先复用 [`docs/plans/2026-05-01-mla-kernel-design.md`](2026-05-01-mla-kernel-design.md) 的设计; Metal 路径走 MLX bridge, 实现 latent-cache + RoPE 解耦的 fused decode kernel。
- KV cache 形态: `(B, S_total, kv_lora_rank + qk_rope_head_dim)` 单 head —— SKU-A=160, SKU-B=224。**注意 page size 要按 latent dim 重算**, 写入 [`docs/plans/2026-05-01-mla-kernel-design.md`](2026-05-01-mla-kernel-design.md) 的扩展条目, 并与 [`infer/src/kv_tier/AGENTS.md`](../../infer/src/kv_tier/AGENTS.md) 的 page 体系对齐 —— RadixCache 不变, page size 是 backend-side 参数。
- `q_lora_rank=None` 与 `Some(rank)` 两种 q-projection 形态在 `Model::load_weights()` 阶段按 config 字段二选一; 选错即报错, 不做 silent fallback。

### 6.2 DeepSeekMoE Op (SKU-B only)

- 复用 spec crate 已有的 `DeepSeekMoeForwardBatch::expert_inputs() / ep_forward_plan()` —— 单卡推理 `world_size=1`, `local_experts = 0..32`, 全部本地。
- ops 层把 `DeepSeekExpertForwardInput` 列表映射成 grouped-GEMM: CUDA 走 cutlass-MoE 路径 (已在 [`crates/cuda-kernels/csrc/`](../../crates/cuda-kernels/csrc/) 视野内), Metal 端先实现 expert-by-expert loop (M-系列 grouped matmul 后置优化)。
- shared expert 走 dense MLP path 与 routed 并行计算然后加和 (spec 已编码 `include_shared_experts` 字段)。
- 路由: sigmoid-gating + plain top-4 (`n_group=1`), auxiliary-loss-free 的 router bias 已 fold 进 router 权重, inference 无额外状态。

### 6.3 MTP Head (SKU-B only)

- 加载 `mtp_tensor_names(0)` 全套权重 (`embed_tokens / enorm / hnorm / eh_proj / 内嵌 layer / shared_head_norm / lm_head`), 作为可选 module。
- 推理两挡: 默认关闭走标准 decode; 打开后接 spec-decode proposer, 与 [`docs/plans/2026-05-01-longctx-spec-decode-phase2.md`](2026-05-01-longctx-spec-decode-phase2.md) 的 verifier 共用 token tree。

### 6.4 Quantization / Small-VRAM Deployment

- BF16 baseline → **INT4 weight-only (GPTQ 或 AWQ, 离线一次性)** → 落 `models/deepseek-tiny-dense/`、`models/deepseek-mini-moe/`。
- **FP8 weight load** 作为 V4-FP8 前向兼容钩子: H100 上无损直跑, 消费卡退回 INT4; scaling tile 的 per-block 元数据按 V3 layout 落 safetensors aux 张量。
- KV 默认 BF16; **不优先做 KV 量化** —— MLA 已经把 KV 压到 1/28, 4 k context KV 只占几十 MB, 再压收益不抵风险。

### 6.5 Backend Isolation

严格遵守 [`infer/src/backend/AGENTS.md`](../../infer/src/backend/AGENTS.md) 的 `cfg(feature = "cuda" / "metal")` 隔离; `mla.rs` / DeepSeekMoE 的 `cudarc` / MLX 类型不穿透到 cross-backend 模块, 跨 backend 走 `server_engine::InferenceEngine`。

## 7. Verification & Benchmarks

按 [`docs/bench-and-trace-spec.md`](../bench-and-trace-spec.md) 与 AGENTS.md §Benchmarks (每个 runtime 改动都必须出 bench 入 `wins/`):

1. **数值对齐 baseline**: 对每个 SKU 用 PyTorch reference 的 logits 落 `infer/test_data/deepseek-{nano,tiny-dense,mini-moe}.json` (top-128 prob 误差 ≤ 1e-3)。新增 `infer/tests/e2e_deepseek.rs`, 参考现有 `e2e_qwen35` 风格。
2. **Throughput / TTFT**: `scripts/bench_guidellm.sh deepseek-tiny-dense-cuda-rtx3050`、`deepseek-mini-moe-cuda-rtx3060`、`deepseek-mini-moe-metal-m3`, 落 `docs/experience/wins/<datestamp>-bench-guidellm-<label>.md` (datestamp = 当次 run 实际时间, 由 CC 在 commit 时填)。
3. **显存 ceiling**: 在 4 / 6 / 8 / 12 GB 卡上跑并发 1 / 2 / 4 / 8 × context 1 k / 4 k / 16 k, 记录 max admissible 工作点表 —— 这是"小卡推理适配"的核心交付。
4. **MLA KV-cache 收益对比**: 相同模型规模做一次 if-not-MLA 对比 (手工把 MLA 替成 GQA-K 跑同一组), 量化 KV 压缩比; 进 `wins/`。
5. **MoE 路由 entropy / 利用率**: SKU-B inference 时随机采样 1 k 个 batch, 每个 expert 的 token 命中分布写进同一份 wins entry。

## 8. Phase Ordering (task volume, no calendar)

每个 phase 标注相对工作量 (XS / S / M / L / XL) 与建议执行模式 (single-session / multi-session / delegate to subagent)。**不写日历**; 实际节奏由 CC session 推进决定。phase 间是 *依赖序*, 不是 *时间序* —— `P3` 不要求 `P2` 先 ship, 但要求 `P2` 已经设计 freeze。

| Phase | Volume | Mode | Deliverables |
|---|---|---|---|
| **P1** Spec presets | S | single-session | `crates/deepseek-spec/`: `presets::{nano, tiny_dense, mini_moe}` 构造器 + 三份 `config.json` 模板 + `q_lora_rank=None` 单元测试 |
| **P2** Model skeleton + MLA CPU ref | M | multi-session | `infer/src/model/deepseek.rs` skeleton, MLA CPU reference forward, nano e2e (CPU, logits 对齐外部 PyTorch ref) |
| **P3** MLA CUDA kernel | L | delegate (general-purpose, plan-mla-kernel-design 作为 brief 附件) | `infer/src/ops/attention/mla.rs` CUDA path, nano CUDA e2e, bench entry |
| **P4** MLA Metal kernel | L | delegate | MLA on MLX bridge, nano Metal e2e, bench entry |
| **P5** Safetensors export | S | single-session, 外部 PyTorch repo | export 脚本 + round-trip 校验 + tensor name shard 全覆盖 sanity |
| **P6** SKU-A pretrain | S (CC 端工作量小, GPU 算力 ~2.5 H100·h) | external | 触发 + 监控外部 pretrain run; SKU-A INT4 量化离线 |
| **P7** SKU-A 落地 | M | multi-session | SKU-A on CUDA + Metal e2e; 三篇 wins (CUDA / Metal / MLA-vs-GQA 对比) |
| **P8** DeepSeekMoE forward | L | delegate | CUDA cutlass MoE path + Metal expert-loop; MTP head 加载; SKU-B model wiring |
| **P9** SKU-B pretrain | M (GPU 算力 ~15 H100·h, 需要 retry 预算) | external | SKU-B BF16 + INT4 |
| **P10** SKU-B 落地 + spec-decode 串联 | M | multi-session | SKU-B e2e, 小卡显存 ceiling 表, MTP→spec-decode integration, 两篇 wins |
| **P11** Reflect | XS | single-session | 复盘文档: "V4 正式 delta 在哪里需要打补丁"检查表 |

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| DS V4 正式公开规格与本计划假设差量大 | 架构需重做局部 | spec crate 已覆盖 V3 全部字段; 预计 V4 增量在 FP8 scaling 形态、router 正则、超长 context 三处 —— 留 hook 不写死 |
| 自训 65 k tokenizer 中文表现弱 | 评估指标偏低 | 接受 —— 这是 substrate; 若必要换开源 DS tokenizer 但需法律确认 |
| MLA Metal kernel 不达预期 | M-系列推理体感差 | 已立项 plan; 最坏情况 Metal 先走 MQA fallback, CUDA 跑完整 MLA |
| auxiliary-loss-free MoE 收敛不稳 | SKU-B 数据效率差 | 8 B token 烟测; 不通过则切 standard aux-loss |
| Rust 端 grouped-MoE kernel 工作量被低估 | 推迟 SKU-B | Metal 端先 expert-loop 兜底; CUDA 走 cutlass 现成路径 |
| INT4 量化在 MLA latent KV 配合下精度掉太多 | 小卡不可用 | 先验证 W4-only INT4, KV 留 BF16; 如仍掉, 回退 INT8 |
| 外部 PyTorch repo 与本 repo 的 spec 版本漂移 | 命名不对齐导致 load fail | safetensors 导出脚本必须 import `deepseek-spec` 的 JSON 配置作为唯一 source; CI 强制校验 |

## 10. Definition of Done

- [ ] `crates/deepseek-spec/`: `presets::{nano, tiny_dense, mini_moe}` 构造器 + 对应 JSON 模板 + `q_lora_rank=None` 测试
- [ ] `infer/src/model/deepseek.rs` 在 CUDA + Metal 双 backend 跑通 generation; `infer/tests/e2e_deepseek.rs` 全绿
- [ ] `infer/src/ops/attention/mla.rs` (CUDA + Metal kernel)
- [ ] `infer/src/model/deepseek/moe.rs` (DeepSeekMoE forward) —— 仅 SKU-B 必须
- [ ] 三个 safetensors checkpoint + tokenizer 落 `models/deepseek-{nano,tiny-dense,mini-moe}/`
- [ ] `docs/experience/wins/` 至少 5 篇 bench: tiny-dense (CUDA / Metal)、mini-moe (CUDA / Metal)、MLA-vs-GQA 对比
- [ ] V4 公开后的迁移指南一页 (`docs/plans/` 下, 含字段 diff + kernel 影响矩阵)
