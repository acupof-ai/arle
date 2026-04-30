# 32k 长上下文吞吐 — 架构与技术设计

**Status:** Active — 单一目标设计，2026-04-30 立项；2026-04-30 v2 critique-pass 修订；2026-04-30 v3 §6.1 无损下一站清单
**Owner:** ckl
**Scope:** CUDA backend, Qwen3-4B (HD128, FP8 + INT8 KV)。HD256 / Qwen3.5 / Metal / 64k+ / 128k 全部显式不做
**Parent context:**
[`2026-04-23-cuda-decode-sglang-alignment.md`](2026-04-23-cuda-decode-sglang-alignment.md)
是更宽的对齐计划；本文是其在长上下文方向的最小子集。
**Bench note:** docs-only; 实施落地的每一刀都按
[`../bench-and-trace-spec.md`](../bench-and-trace-spec.md) 提交 wins/ 条目

---

## 1 · 唯一目标

```
Qwen3-4B FP8 KV, L4 单卡, prompt=32768, output=256
  primary    c=4, ARLE tok/s within 5% of SGLang  (i.e. ARLE/SGLang ≥ 0.95×)
  secondary  c=1, ARLE tok/s within 5% of SGLang  (同 sweep 内免费数据)
  stretch    ARLE/SGLang ≥ 1.00×（不作为 retry 触发条件）
```

围绕 primary 一行展开。任何不直接服务它的优化推迟。

为什么是这个数据点：
- **32k = Qwen3-4B 训练上下文上限**（RoPE base 不需要 YaRN 外推），先把"训练分布内最长"打透
- **c=4 = 单卡 KV 容量约束生效点**（FP8 下 32k×4 ≈ 9.6 GB，权重 + workspace 后正好饱和 L4 22 GB 实际可用）
- **c=1 = 同 sweep 免费数据**，验证 split-KV kernel 单请求 ceiling
- **L4 = 项目当前 wins/ 主基线**，CI 和远端机器都在；不切换硬件
- **vs SGLang = 已建立的对照**，不发明新基线
- **`within 5%` 不用 `≥1.00×`**：±2% measurement noise 下窄窗口会反复抖到 fail/pass 边缘

**output=256 的取舍**：选 256 因为对照基线 (SGLang/vLLM default) 也在 256，**不代表 long-ctx 真实 workload**（真实 RAG / agent loop 通常 512–2048）。代表性长 decode 不在本计划，独立项目。

---

## 2 · 审计基线（直接代码证据）

### 2.1 当前 attention 调度面（5 路并存）

| Phase × Format | Entry (Rust) | Kernel (CUDA) | HD |
|---|---|---|---|
| Prefill paged (BF16/FP8/INT8 共享) | `infer/src/model/qwen3/prefill.rs:318` | `prefill_attention_paged_prep.cu` + TileLang/FlashInfer cubin | 128 |
| Decode batched **BF16** | `infer/src/model/qwen3/batch_decode.rs:1699` | `flashinfer_tc_decode.cu` | 128 |
| Decode batched **FP8** | `infer/src/model/qwen3/batch_decode.rs:1647` | `decode_attention_quantized.cu:307` (split-KV) | 128 |
| Decode batched **INT8** | `infer/src/model/qwen3/batch_decode.rs:1670` | `decode_attention_quantized.cu:46` (split-KV) | 128 |
| Mixed (decode+prefill) **BF16** | `infer/src/model/qwen3/batch_decode.rs:474`，dispatch `:729` | `flashinfer_tc_decode.cu` (varlen TC) | 128 |
| Mixed **FP8/INT8** | **缺失** — 早 return at `batch_decode.rs:481`，gate at `forward.rs:585` | — | — |

5 个独立调度位 + 1 个空格子。后面每一节都围着这张表走。

### 2.2 已落地但未串通的资产

- **`crates/cuda-kernels/csrc/attention/decode_attention_varlen_fp8.cu`**（commit `4e4906f5`）
  - `decode_attention_varlen_fp8_kernel<HEAD_DIM=128, CAUSAL>` 已写
  - C API `decode_attention_varlen_fp8_cuda` (`:314`)
  - Rust FFI `crates/cuda-kernels/src/ffi/attention.rs:543`
  - Rust 函数 `crates/cuda-kernels/src/kv_quant.rs:426 decode_attention_varlen_fp8`
  - **Single-CTA-per-(q_token, q_head)**，没有 split-KV — 32k 下 HBM 带宽饥饿
  - **HD128-only**，page_size=16-only

### 2.3 Mixed 路径的两个守门人

```rust
// infer/src/model/qwen3/forward.rs:579
fn supports_mixed_batch(&self, kv_pool_format: crate::model::kv_cache::KVFormat) -> bool {
    self.prefill_uses_paged_pool()
        && self.lora.is_none()
        && matches!(kv_pool_format, crate::model::kv_cache::KVFormat::BF16)  // ← 只放 BF16
}

// infer/src/model/qwen3/batch_decode.rs:481
if self.lora.is_some() || paged_kv_pool.format != KVFormat::BF16 {
    return Ok(false);  // ← 同一道门第二把锁
}
```

### 2.4 Bench 现状

`scripts/bench_guidellm.sh:68` 写死了 `prompt_tokens=4096`。
[`bench-matrix-design-2026-04-29.md`](bench-matrix-design-2026-04-29.md) 提议过 `longctx-32k` workload 但**没落地**。`docs/experience/wins/` 里 32k 数据点为零。

历史 SGLang 引用值（如 `~201 tok/s @ c=16/4096`）是 hand-recorded，**不是 reproducible 同硬件同 commit 跑出来的**。本计划要求 SGLang 对照变成 1st-class 工件（详见 §5.5 + S4）。

### 2.5 FP8 KV 数值漂移现状

[`../experience/errors/2026-04-30-arle-fp8kv-numerical-drift.md`](../experience/errors/2026-04-30-arle-fp8kv-numerical-drift.md)：
两个 FFI/scale 漏洞已修，Qwen3.5 spot-check 从 2.79% 提到 39.08%
token-match。**Qwen3 路径未单独 spot-check**——本计划必须补上。

---

## 3 · "Done" 的 5 个条件

以下 5 条同时成立才算这个目标达成：

1. **Mixed plan 在 FP8 下进 hot path**：`scheduler` 日志在 32k c=4 跑下出现 `StepPlan::Mixed`，Qwen3-4B 路径下 `Split` 计数 = 0
2. **数值正确性（分层）**：Qwen3-4B FP8 在 32k 输入下，与 BF16 对照 token-match 分层
   - **≥70% = pass**（与 SGLang 短 prompt 77.54% 的 -7pp 衰减估计一致）
   - **60–70% = degraded**：必须开 `errors/` 条目说明，**不**进 wins
   - **<60% = stop**：回 S1 修 reduction 或回 S2 重检 wire-up
3. **e2e 文本可读**：`infer/tests/e2e.rs` non-degeneracy gate 在 32k 输入扩展用例下通过（无 NaN、无 token-0 重复、无空输出），TileLang feature on/off 两种构建都过
4. **吞吐**：`docs/experience/wins/YYYY-MM-DD-bench-guidellm-longctx-32k-c4-fp8.md` 一份，含 `(ARLE tok/s) / (SGLang tok/s) ≥ 0.95`（within 5%），SGLang 同 commit / 同硬件 reproducible
5. **Qwen3-4B 不再回退 Split**：`forward.rs:585` 不再有 BF16-vs-FP8 分支；`StepPlan::Split` 物理代码保留服务 LoRA + Qwen3.5（**不假装已删除**），但 Qwen3-4B 路径走不到（运行期 `debug_assert!` 守护 + 日志计数 = 0 验证）

第 5 条**显式不是**"删除式重构完成"——`Split` enum variant 整体删除有独立触发条件，见 §6。本计划的删除式分母是 `:481` 的 KV-format 检查（1 处删 + 1 处 gate 收紧）。

---

## 4 · 架构设计（Post-state）

### 4.1 Qwen3 mixed gate 从 BF16-only 扩到 BF16+FP8+INT8

目标后表（仅 Qwen3 维度，Qwen3.5 不变）：

| Phase × Format | Entry | Kernel |
|---|---|---|
| Prefill paged (BF16/FP8/INT8) | `prefill.rs:318` 不变 | `prefill_attention_paged_prep.cu` 不变 |
| Decode batched **BF16** | `batch_decode.rs::decode_batch` | `flashinfer_tc_decode.cu` |
| Decode batched **FP8/INT8** | `batch_decode.rs::decode_batch` | `decode_attention_quantized.cu` (split-KV，保留——见 §4.4) |
| Mixed varlen (BF16/FP8/INT8) | `batch_decode.rs::decode_batch_with_prefill` | **BF16**: `flashinfer_tc_decode.cu`；**FP8**: `decode_attention_varlen_fp8.cu` (扩 split-KV)；**INT8**: 同 kernel + `K_scales` 可空指针 |

按 Rust dispatch call site 数：本计划完成后 Qwen3 仍有 4 个 attention 调度位（3 个 `decode_batch` 的 format-switch + 1 个 `decode_batch_with_prefill` 的 format-switch）。**没有"实质 2 路"这种修辞**——本计划是**新增 2 个 Mixed 分支 + 删除 1 道 BF16-only 守门**，不是路径合并。

### 4.2 删除式收敛（真实分母 = 1）

具体被删的代码：
- `batch_decode.rs:481` 的 `format != KVFormat::BF16` 检查 — **删除**，只剩 LoRA gate
- `forward.rs:585` 的 `matches!(..., KVFormat::BF16)` — **改为** `matches!(..., KVFormat::BF16 | KVFormat::FP8E4M3 | KVFormat::INT8)`

明确**保留**（不假装删）：
- `StepPlan::Split` enum variant 和 fallback 路径 — Qwen3.5 + LoRA 仍在用
- `decode_attention_quantized.cu` qlen=1 split-KV kernel — 纯 decode 路径仍在用（除非 S1 acceptance 证明 varlen kernel 在 qlen=1 不慢于它）

### 4.3 不引入的抽象 + 明确触发器

显式拒绝以下"看起来通用"的设计：

- ❌ `trait BatchAttention { fn run(...); }`
- ❌ `enum AttentionDispatch { Bf16(...), Fp8(...), Int8(...) }`
- ❌ 把 `decode_attention_varlen_fp8` 改名为 "通用 varlen kernel"
- ❌ 给 `KVFormat` 加 `supports_mixed()` 方法

但**写死立 trait 的触发条件**（避免下次又借 no-speculative-shaping 拖延）：

> 当满足 (a) 同型 dispatch shape 在 ≥2 模型族出现，且 (b) 至少一族已有 ≥3 实现，且 (c) 新增分支需要碰 ≥2 个 call site 的 `match` 才能添完 — 立 trait/enum。

按此判据：**Qwen3.5 mixed 启动时（不在本计划）必须立**——届时会同时满足 (a)+(b)+(c)。本计划只到 Qwen3-4B mixed FP8/INT8，不触发。

### 4.4 qlen=1 split-KV decode kernel 的 sunset 路径

`decode_attention_quantized.cu:307` (FP8 qlen=1 split-KV) 在纯 decode (`StepPlan::Decode`) 路径仍在用。理论上 varlen kernel 喂 `qo_indptr=[0,1,2,...]` 就退化为 qlen=1，可以替代。

**S1 acceptance 加一条对照**：varlen kernel 在 qlen=1 / S=32k / c=4 配置下，对 quantized:307 的性能差距 ≤ 5%。如果通过，**S6 加 sunset 计划**：先迁移 `StepPlan::Decode` FP8/INT8 路径到 varlen kernel，再删 quantized:307。如果不通过，两个 kernel 并存写入 `feedback_no_half_states` 例外条款。

---

## 5 · 技术设计

### 5.1 Kernel 改造：varlen FP8 加 split-KV

**问题**：当前 kernel 是 `grid = (total_q_tokens, num_q_heads)`，每个 block 独占地流过该序列**全部** KV 页（`decode_attention_varlen_fp8.cu:197 for (int p = 0; p < num_kv_pages; p++)`）。32k 序列下 `num_kv_pages = 32k/16 = 2048`，单 CTA 串行流式读 2048 页 KV，HBM 带宽利用率 ~10%。L4 KV decode 是纯 memory-bound，单 CTA = 性能下限。

**方案**：每 (q_token, q_head) 拆 `KV_SPLIT` 个 CTA，每 CTA 处理一段连续 KV，phase-2 reduction kernel 合并 (m, l, o)。

```
Phase 1 (split kernel):
  grid  = (total_q_tokens, num_q_heads, KV_SPLIT)
  block = 128 threads
  output: tmp_o[total_q_tokens, num_q_heads, KV_SPLIT, HEAD_DIM]
          tmp_m[total_q_tokens, num_q_heads, KV_SPLIT]
          tmp_l[total_q_tokens, num_q_heads, KV_SPLIT]

Phase 2 (reduction kernel):
  grid  = (total_q_tokens, num_q_heads)
  block = HEAD_DIM threads
  reduces over KV_SPLIT axis: standard FlashAttention online-softmax merge
  output: O[total_q_tokens, num_q_heads, HEAD_DIM]  (bf16)
```

#### KV_SPLIT 选择 — **动态**

```c
int KV_SPLIT = clamp(kv_total_len / 4096, 1, 16);
```

- 4k 序列：`KV_SPLIT=1`（退化为现有 single-CTA，4k workload 不回归）
- 16k 序列：`KV_SPLIT=4`
- 32k 序列：`KV_SPLIT=8`（核心目标）
- 64k+ 序列：`KV_SPLIT=16` 上限

**直接抄 `decode_attention_quantized.cu` 现有动态 num_splits 路径**（同 4096 token 甜点），不发明新策略。**不**预先做 work-estimation 框架（违反 no-speculative-shaping）。

#### Phase-1 kv chunk offset 传递

每个 (split_idx) CTA 必须重算它处理的 KV 段在序列内的全局起点：

```c
int kv_chunk_size = (kv_total_len + KV_SPLIT - 1) / KV_SPLIT;
int kv_chunk_start = split_idx * kv_chunk_size;
int kv_chunk_end   = min(kv_chunk_start + kv_chunk_size, kv_total_len);
```

不能继续用现 kernel 的 `kv_pos_running` 累加器（它是 per-page 累加，跨 split 失效）。Per-CTA 入口直接由 `(kv_chunk_start, kv_chunk_end)` 决定起止 page 范围。

#### Phase-2 reduction kernel 模板来源

**强制抄 `decode_attention_quantized.cu:282-298` 现有 split-KV merge**（已被远端验证正确），grid 维度从 `(num_qo_heads × batch)` 改为 `(total_q_tokens, num_qo_heads)`。**不**从 FlashAttention paper 重写——bug 表面积差 3×。

Phase-2 grid 在 mixed batch 上界估算：c=4 / 32k decode-only / chunk_prefill_size=2048 时，`total_q_tokens ≤ 4 + 2048 = 2052`，phase-2 grid `2052 × 32 = 65,664` blocks。L4 60 SM × 4 active block ≈ 240 并发 → ~273 wave，每 block ~1μs → ~0.3ms phase-2 overhead per layer × 36 layers = ~10ms per forward step。**不可忽略，但占 25s 单请求 < 0.05%**。预先记录这个数。

#### Causal mask 与 split 的交互

- 仅"包含 `causal_limit` 的 split 段 + qlen>1"需要 diagonal mask 检查
- 完全可见段（kv_chunk_end ≤ causal_limit）：无需 mask
- 完全不可见段（kv_chunk_start > causal_limit）：CTA 入口直接 return（写 `tmp_m=-FLT_MAX, tmp_l=0`）
- 这个静态裁剪同时是 prefill chunk 的免费优化

#### varlen 可独立 split-KV 的论证

现有 kernel 注释（`decode_attention_varlen_fp8.cu:11-13`）说 "varlen makes split-KV hard because per-row qlen varies"——错的。split-KV 的复杂度来自**跨行**合并，不来自 per-row qlen 变化。本设计每行独立 split，reduction 也按 (q_token, q_head) 独立，qlen 变化只影响 causal mask 范围（已在 phase-1 处理）。Phase-1 的 CTA work 不均（prefill 行最后一段 mask 掉大部分）只影响 wave 尾巴，不影响正确性。

#### 不复用 FlashInfer plan_info 的理由

FlashInfer 的 `BatchDecodeWithPagedKVCacheWorkEstimationDispatched` (`flashinfer_decode.cu:73`) 写死 DType=bf16 + paged_kv_t HND 布局，FP8 pool 是 NHD + E4M3 自描述无 scale。**结构性不兼容**，不是品味问题。

#### 触点文件

- 改：`crates/cuda-kernels/csrc/attention/decode_attention_varlen_fp8.cu`（拆 phase-1/2，改 grid 维度，加 K_scales 可空指针支持 INT8）
- 改：`crates/cuda-kernels/src/ffi/attention.rs:543`（C API 增加 `tmp_o / tmp_m / tmp_l` workspace + `K_scales / V_scales` 可空指针）
- 改：`crates/cuda-kernels/src/kv_quant.rs:426`（Rust 包装管理 workspace，`decode_attention_varlen_fp8` 现支持 FP8 / INT8）

### 5.2 Wire-up：Mixed 路径接 FP8 + INT8

**Step 1** — `forward.rs:585`：

```rust
fn supports_mixed_batch(&self, kv_pool_format: crate::model::kv_cache::KVFormat) -> bool {
    self.prefill_uses_paged_pool()
        && self.lora.is_none()
        && matches!(
            kv_pool_format,
            crate::model::kv_cache::KVFormat::BF16
                | crate::model::kv_cache::KVFormat::FP8E4M3
                | crate::model::kv_cache::KVFormat::INT8,
        )
}
```

**Step 2** — `batch_decode.rs:481`：

```rust
// Was: if self.lora.is_some() || paged_kv_pool.format != KVFormat::BF16
if self.lora.is_some() {
    return Ok(false);
}
let kv_format = paged_kv_pool.format;
debug_assert!(matches!(
    kv_format,
    KVFormat::BF16 | KVFormat::FP8E4M3 | KVFormat::INT8
));
```

**Step 3** — `batch_decode.rs:720-750` attention dispatch 处加 FP8 + INT8 分支：

```rust
match kv_format {
    KVFormat::BF16 => {
        ops::flashinfer_tc_run_layer(/* 现有调用，不变 */)?;
    }
    KVFormat::FP8E4M3 | KVFormat::INT8 => {
        // 同一个 kernel，K_scales/V_scales 在 INT8 时非空，FP8 时为 null
        kv_quant::decode_attention_varlen_fp8(
            &self.ctx,
            &mixed.q_batch,
            &mixed.metadata.qo_indptr,
            paged_kv_pool, layer_idx,
            &mixed.metadata.kv_indptr,
            &mixed.metadata.kv_indices,
            &mixed.metadata.kv_last_page_len,
            &mut mixed.attn_output,
            sm_scale,
            /* causal: */ true,
        )?;
    }
    KVFormat::TurboQuant => unreachable!("TurboQuant 不进 mixed plan"),
}
```

**Step 4** — `batch_decode.rs::ensure_mixed_buffers` 给 quantized 路径分配 split-KV workspace。Workspace 上界（c=4 / 32k / KV_SPLIT=8）：`4 × 32 × 8 × (128×4 + 8) ≈ 540 KB`，可忽略。

**Step 5** — `forward.rs::forward_mixed_batch` 已经路由到 `decode_batch_with_prefill`（`forward.rs:588-601`），无需改动。

### 5.3 INT8 mixed 顺手做（不再独立立项）

**改动量：~30 LoC**（critique 期间核实：`decode_attention_quantized.cu:46 vs :307` 差只在每 token 一个 `K_scales[scale_offset]` 标量乘）。

把 varlen kernel 模板化吃 `const float* K_scales / V_scales`，`nullptr` 时退化为 FP8 E4M3 自描述；非空时按 INT8 per-page scale 表读取并乘到 dequant 路径里。

不顺手做的代价（即如果推迟到独立计划）：
- §3 condition 5（`forward.rs:585` KV-format 分支收紧）必须保留 BF16/FP8 vs INT8 二分，删除式分母从 1 缩到 0.5
- 同一份 varlen kernel 模板要在两个计划里各动一次

**因此本计划范围扩 +30 LoC，把 INT8 一起带进 Mixed gate**。

### 5.4 数值正确性 gate（三层防线）

**第一层 — 短/长 prompt e2e**（已有 + 扩展）

`infer/tests/e2e.rs:148-184` non-degeneracy gate 在 prompt=32k 输入下复跑：

```rust
// 新增到 e2e.rs cases：
("Summarize the following 30000-token document: ...", 64),  // 加一条 long-prompt
```

要求：非空、首 5 字符不全相同、运行不 panic。**TileLang feature on/off 两种构建都跑一遍**（避免 `cfg(not(feature = "tilelang-attn"))` 盲区，参考 errors/2026-04-28 short-qlen NaN）。

**第二层 — FP8-vs-BF16 内部一致性**

复用 `errors/2026-04-30` 的 16-prompt × 64-token 协议，在 32k 输入下跑：

```bash
# Pair: ARLE FP8 KV vs ARLE BF16 KV (same model, same scheduler)
# Prompts: 16 × 32k 真实长文 (ShareGPT-long 或合成)
# Sampling: temperature=0, seed=20260429, max_tokens=64
# Pass condition (分层):
#   ≥70%  avg token-match  ⇒ pass
#   60–70%                 ⇒ degraded, open errors/, 不进 wins
#   <60%                   ⇒ stop, 回 S1
# 同时要求:
#   exact_pairs / 16          ≥ 3
#   earliest divergence       ≥ generated token 1
```

**第三层 — SGLang BF16 锚点**（防 ARLE-BF16 自己漂移）

防止"ARLE-FP8 和 ARLE-BF16 token-match 高，但两个都和真实分布偏远"的 self-consistent failure。复用同 16 prompt × 64 token：

```bash
# ARLE-BF16 vs SGLang-BF16, same prompts/seed
# Pass: avg token-match ≥ 80%（短 prompt 已知 90%+，32k 留 10pp 余量）
# Fail: 表示 ARLE-BF16 自己已漂移，FP8 数据不可信，回 S1 排查 prefill kernel
```

**第四层 — 长尾扫**（覆盖率从 ~3% 拉到 ~85%）

16×64=1024 tokens 对 32k 输入空间覆盖率 ~3%，FP8 漂移真实 bug 在 token 30+ 才稳定显形。追加：

```bash
# 32 prompts × 256 tokens FP8-vs-BF16
# Pass condition:
#   divergence_p50 ≥ generated token 30   (errors/2026-04-30 经验值)
#   tail_match_last_64tok ≥ 50%           (前 192 tokens 漂移容忍，尾部 64 tokens 不能崩)
# 增量跑时间 ~5 min / 16k tokens generated
```

四层全过才进 §7.2 wins。任意一层 fail 按其分层规则处理。

### 5.5 Bench harness：`--workload longctx-32k` + SGLang 同硬件对照脚本

**ARLE 侧**：最小改动 `scripts/bench_guidellm.sh`：

```bash
case "${WORKLOAD:-default}" in
  default)
    DATA="prompt_tokens=4096,...,output_tokens=256,..."
    PROFILE_ARGS="--profile sweep"
    MAX_SECONDS=60
    ;;
  longctx-32k)
    DATA="prompt_tokens=32768,prompt_tokens_stdev=1,prompt_tokens_min=32768,prompt_tokens_max=32768,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256"
    PROFILE_ARGS="--profile concurrent --concurrency 1,4"
    MAX_SECONDS=300       # c=4 升 300s（180s 仅 ~28 sample，p99 = max）
    SECONDARY_C1_SECONDS=360  # c=1 单独跑 360s
    ;;
esac
```

**SGLang 侧（新增 1st-class 工件）** — `scripts/bench_sglang_longctx.sh`：

```bash
# 1. SGLang server 启动（commit pinned）
SGLANG_COMMIT="214c35b03184c354acf1f86f99746799e1c9b3a9"  # 与 docs/plans/2026-04-23-cuda-decode-sglang-alignment.md 同 anchor
python -m sglang.launch_server \
  --model-path $MODEL \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 16 \
  --mem-fraction-static 0.85 \
  --disable-radix-cache \      # 与 ARLE 当前默认对齐（prefix-cache 单独验）
  --max-total-tokens 131072

# 2. guidellm 同 shape 同 seed 跑 ARLE 那份相同 DATA / PROFILE_ARGS / MAX_SECONDS
guidellm --target http://localhost:30000/v1 ...

# 3. 输出 sglang_headline.json，由 ARLE wins 模板合并算 Δ%
```

**对齐项明细表**（写进 wins 条目 §7.1）：

| 维度 | ARLE | SGLang | 备注 |
|---|---|---|---|
| KV dtype | FP8 E4M3 | `fp8_e4m3` | 显式开 |
| max running req | `--num-slots 16` | `--max-running-requests 16` | 对齐 |
| mem fraction | `--mem-fraction-static 0.85` | 同左 | 对齐 |
| prefix cache | off (default) | `--disable-radix-cache` | 显式关，独立 workload 测 prefix |
| max ctx | 131072 | `--max-total-tokens 131072` | 对齐 |
| commit | ARLE HEAD SHA | SGLang `214c35b...` | 写进 wins |

### 5.6 Δ% 在 SGLang 失败时的处理

| SGLang 状态 | wins Δ% 字段 | success criterion 替代 |
|---|---|---|
| 跑通 | `(ARLE/SGLang - 1) × 100%` | `ARLE/SGLang ≥ 0.95` |
| OOM | `N/A (SGLang OOM @ <shape>)` | 退化为 "ARLE within historical L4 envelope ±20%"（envelope 取 4096-in/c=16 历史 wins 推算的 32k 等效）|
| NaN / 不收敛 | `N/A (SGLang 数值失败)` | 同上，但**额外要求**开 errors/ 条目记录 SGLang fail，并不阻塞本计划 |

无论哪种情形，**SGLang commit + 启动参数都钉在 wins 条目里**——后人复现需要这两条。

---

## 6 · 显式不做（Out of Scope）

| 不做 | 为什么 / 触发条件（如有） |
|---|---|
| 64k / 128k | 32k 数稳之前任何 64k+ 都是猜的；64k 接续基线在本计划完成后画 |
| HD256 / Qwen3.5 mixed | varlen kernel 当前 HD128-only，HD256 模板化 + Qwen3.5 hybrid layer 调度是独立工程。**触发立 trait/enum**：启动该工作时同步评估 §4.3 触发器，届时大概率三条件全成立 |
| Speculative decode | 长 decode 才回报，本目标 output=256 不属于长 decode 区段 |
| Tiered KV 长 S 端到端 bench | 依赖本目标的 32k 单请求基线，先打基线 |
| SnapKV / KV 压缩 | 改变 KV 语义，需独立 spec + 数值评估 |
| Sequence parallel (CP) | 32k 单卡可装下，CP 是 64k+ 才必要 |
| **`StepPlan::Split` 整体删除** | 还要服务 Qwen3.5 + LoRA。**触发删除**：(a) Qwen3.5 mixed 落地 + (b) LoRA mixed 落地 + (c) 一周生产日志 `Split` 计数 = 0 — 三条全绿才删 enum variant |
| **`BatchAttention` trait** | 当前不立。**触发立**：满足 §4.3 三条件（≥2 模型族 + 至少一族 ≥3 实现 + 新增分支需碰 ≥2 call site）|
| `quantized:307` qlen=1 split-KV kernel | 等 §4.4 sunset 路径触发 |
| TileLang prefill long-qlen 优化 | 32k 用 `chunked_prefill_size=2048` 切 16 段；不重写 |
| Metal 镜像跟跑 | 完全独立路径，不在 CUDA 长上下文这条线上 |
| c=8 OOM 探测 | L4 22GB 实际可用，FP8 KV 32k×8 = 19.2 GB > 可用 KV 预算，结构性不可达 |

### 6.1 严格无损下一站（本计划完成后选一）

本计划只走 split-KV varlen + Mixed wire（一种"补课式"无损改动）。
完成后下一站从下面 5 项中选一个推进，**不并行**，按 ROI 顺序：

每条满足 (a) 数学上输出分布等价（拒绝采样 / 通信精确实现 / 缓存复用 / 调度优化）或 (b) 经验近无损 ≤1pp on LongBench（仅 X 项标注）。

| 序 | 方向 | 严格无损 | 仓库现状 | 32k–128k 数字 | 触发条件 |
|---|---|---|---|---|---|
| **A** ★★★ | **Long-ctx spec decode (MagicDec/TriForce/LongSpec)** | 是（拒绝采样保证 target 分布）| `infer/src/speculative.rs` 625 行 CPU 框架已在；`infer/src/speculative/cuda.rs` 桩；CUDA verifier 未接 | MagicDec 2.51× @ batch 32-256；TriForce 128k 8× faster than offloading；LongSpec 32k+ 1.8-2.5× | 本计划 wins 落地；`speculative.rs` SpecConfig + Verifier 现状 ≥ 70% 完整 |
| **B** ★★★ | **Tiered KV + Radix prefix 在 32k+ workload 验证** | 是（输入相同 → KV 复用是字面等价）| RadixCache + T0/T1/T2 完整；agent loop / RAG 真实流量在 32k 数据点 = 0 | llm-d 经验：prefix hit 60%+ 把长 ctx prefill 时间砍 60%+ | 本计划 wins 落地；`bench-matrix-design-2026-04-29.md` 的 `prefix-cache` workload 落 `--workload` |
| **C** ★★ | **Disaggregated prefill/decode (Mooncake-aligned)** | 是（仅切分阶段，attention 数学不变）| `kv_tier/` + `kv-native-sys/` + `transport/` 接口齐；未拼成 disagg | DistServe 7.4× more requests / 12.6× tighter SLO；Mooncake 长 ctx 真实流量容量 +59-498% | 本计划 §9 风险表 #5（prefill 段长 qlen 缺口）实测确认；`tiered-kv-cache.md` Phase E 远端验证完成 |
| **D** ★★ | **Sequence Parallel (Ulysses / Ring / USP)** | 是（attention 精确分布式实现）| `plans/2026-04-28-single-node-multi-gpu.md` F4 留位置；F0–F3 TP 骨架未落 | DeepSpeed-Ulysses 已开源；Arctic Ulysses 专为 inference 优化；64k+ 单卡装不下时唯一解 | 64k 基线立项时启动；32k 阶段不必 |
| **E** ★ | **Async pipeline gap closure** | 是（纯调度，无算力变化）| `wins/2026-04-29-scheduler-overlap-gap-instrumentation.md` 已量化 idle gap | 项目内 instrumentation 已显示有可压缩窗口；上界单数 % | 本计划 S5 bench 跑出 ARLE/SGLang 在 0.95-0.99 区间徘徊（拼最后几个百分点）|

**经验近无损工具箱（用户判断接受度后启用）**：

| 方法 | LongBench 退化 | 节省 | 严格性 |
|---|---|---|---|
| DuoAttention (ICLR 2025) | <1pp | KV 2× / decode 2× | 近似（head 分类）|
| Quest (ICML 2024) | <1pp | decode 2-3× | 近似（页级 top-k）|
| MInference 1.0 (NeurIPS 2024) | <1pp | **prefill 10× @ 1M ctx** | 近似（pattern 检测）|
| SnapKV / PyramidKV | 持平 | KV 4-8× | 近似（KV 选择）|

**与量化的关键区别**：稀疏丢的是 attention edges（"哪些 token 进 attention"），量化丢的是数值精度（"3.7 vs 3.6875"）。长 ctx 上经验稀疏更稳，但**严格论仍是近似**。是否进工具箱由用户判断（默认不进，避免范围漂移）。

**推荐次序**：A → B → C → D，**E 仅在 A-D 都未达到 SGLang ≥1.00× 时启用**。

**显式不做（即使无损）**：

| 不做 | 为什么 |
|---|---|
| **vAttention** (ASPLOS 2025, 1.97× vs vLLM) | 用 CUDA virtual memory 替代 paging，**与现 PagedKVPool 投资冲突**；迁移代价 > 收益 |
| **跨请求 KV deduplication** beyond RadixCache | RadixCache 已覆盖主要场景；通用 dedup 收益 < 工程成本 |

---

## 7 · 成功文档（Success Criteria 落地形态）

完成时必须存在以下 4 份产物：

### 7.1 Wins 主条目

`docs/experience/wins/YYYY-MM-DD-bench-guidellm-longctx-32k-c4-fp8.md`，按
[`TEMPLATE-bench-guidellm.md`](../experience/wins/TEMPLATE-bench-guidellm.md)
模板。**强制字段**：

- Goal: 引用本文 §1
- Hypothesis: K2 wire + split-KV varlen 在 32k c=4 within 5% of SGLang
- Params: prompt=32768 / output=256 / c=4 / 300s / FP8 KV / `--workload longctx-32k`
- Env: L4 / CUDA / Qwen3-4B
- **ARLE commit SHA + SGLang commit SHA** — 两个都写
- **SGLang 启动参数** — §5.5 对齐项明细表全文
- Results 表：`tok/s`, `TTFT p50/p99`, `ITL p50/p99`, `Δ% vs SGLang`, plan 分布 (`Mixed` / `Split` 计数)
  - SGLang fail 时按 §5.6 写 N/A + envelope check 结果
- Problems: 跑过程中观察到的异常
- Learnings: 可推广的 finding（不要写"K2 有效"这种废话；写"split-KV 在 S>X 时收益曲线变化点"这类）
- **Statistical note**: c=4/300s 完成请求数 ≈ 56，p99 = sample[55]（Wilson 95% CI 标注）

### 7.2 Wins 数值健康条目

`docs/experience/wins/YYYY-MM-DD-qwen3-fp8-bf16-spotcheck-32k.md`：四层防线全过的数据表
- 短/长 prompt e2e 通过状态（含 TileLang feature on/off 两份）
- FP8-vs-BF16 16×64 token-match 表
- ARLE-BF16-vs-SGLang-BF16 16×64 token-match 表
- 32×256 长尾扫 divergence_p50 + tail_match 表

### 7.3 Errors（如果有）

任何 ≥1 次失败重试的 bug 入
`docs/experience/errors/YYYY-MM-DD-<slug>.md`，含 Context / Root Cause / Fix /
Rule 四节。

### 7.4 ROADMAP 更新

`ROADMAP.md` "Active Priorities" P0 / P3 行更新一句：32k FP8 长上下文 within 5% of SGLang 已落 wins；下一步基线（64k 或 hybrid Qwen3.5 mixed）。

---

## 8 · 执行方案（按工作量切片，不按时间）

每一刀是一个独立 commit。看依赖、看 acceptance，不看时间。

### 8.1 总览（进度 checklist）

复制此清单到日常 status 用：

- [ ] **S1** — Kernel: varlen FP8+INT8 split-KV（无前置）
- [ ] **S2** — Wire-up: Mixed FP8+INT8 路径（依赖 S1）
- [ ] **S3** — 数值 gate 四层防线（依赖 S2）
- [ ] **S4** — Bench harness ARLE + SGLang baseline 脚本（无前置；可与 S1/S2/S3 全程并行）
- [ ] **S5** — Bench 实跑 + 主 wins（依赖 S2 + S3 + S4）
- [ ] **S6** — Qwen3-4B Split 路径守护 + ROADMAP 更新（依赖 S5）

### 8.2 依赖图

```
S1 ──► S2 ──► S3 ─────────┐
                          ├──► S5 ──► S6
S4 ───────────────────────┘
```

- **关键链** = `S1 → S2 → S3 → S5 → S6`（5 步串行）
- **旁路** = `S4`，从立项第一秒起就可以做，与关键链全程并行
- **不存在双向依赖**；任何一步的 fail 只回退到自己的上游

### 8.3 工作切片

#### S1 — Kernel: varlen FP8 + INT8 加 split-KV

**Blocked by:** —
**Blocks:** S2

**改动文件 (3)**：

| 文件 | 操作 | 估算 (+/−) |
|---|---|---|
| `crates/cuda-kernels/csrc/attention/decode_attention_varlen_fp8.cu` | 拆 phase-1 / 新增 phase-2 reduction kernel；grid `(q_token, q_head)` → `(q_token, q_head, KV_SPLIT)`；动态 `KV_SPLIT = clamp(kv_total_len / 4096, 1, 16)`；causal mask 静态裁段；加 `K_scales / V_scales` 可空指针支持 INT8（`nullptr` = FP8 E4M3 自描述） | +260 / −110 |
| `crates/cuda-kernels/src/ffi/attention.rs` | C API 加 `tmp_o / tmp_m / tmp_l` workspace ptr + `K_scales / V_scales` 可空 ptr；新增 `decode_attention_varlen_fp8_reduce_cuda` FFI | +60 / 0 |
| `crates/cuda-kernels/src/kv_quant.rs` | workspace 分配；两段 kernel launch；FP8 与 INT8 共用 entry，对外 API 不变 | +70 / −20 |

**Phase-2 reduction 模板**：抄 `decode_attention_quantized.cu:282-298` 的 split-KV merge，grid 维度从 `(num_qo_heads × batch)` 改为 `(total_q_tokens, num_qo_heads)`。**禁止**从 FlashAttention paper 重写。

**新增测试 (1)**：

| 文件 | 内容 | 估算 |
|---|---|---|
| `crates/cuda-kernels/tests/attention_varlen_fp8_split.rs` | qlen=1 单行 / qlen 混合 / KV_SPLIT∈{1,4,8,16} vs single-CTA reference / FP8 vs INT8 输出一致性（INT8 scale=1.0 时应等价 FP8 + scale 修正） | +220 |

**Acceptance（每条要绿才算完）**：
- [ ] `cargo test --release -p cuda-kernels --test attention_varlen_fp8_split` 全绿
- [ ] KV_SPLIT=1 输出与改动前 single-CTA kernel **bit-identical**
- [ ] KV_SPLIT=8 与 KV_SPLIT=1 的 bf16 输出 max-abs-diff < 1e-3
- [ ] INT8 路径 `K_scales=[1.0]` 退化输出与 FP8 路径 max-abs-diff < 5e-3
- [ ] qlen=1 / S=32k / c=4 配置下，varlen kernel 性能差距 ≤ `decode_attention_quantized.cu:307` + 5%（用 micro-bench 验，不用 e2e）
- [ ] commit 单独成 PR/diff，CI 类型检查 green

---

#### S2 — Wire-up: Mixed plan 接 FP8 + INT8

**Blocked by:** S1
**Blocks:** S3、S5

**改动文件 (2)**：

| 文件 | 操作 | 估算 (+/−) |
|---|---|---|
| `infer/src/model/qwen3/forward.rs` | `:585` `matches!(..., BF16)` → `matches!(..., BF16 \| FP8E4M3 \| INT8)` | +3 / −1 |
| `infer/src/model/qwen3/batch_decode.rs` | `:481` 删 `format != KVFormat::BF16` 检查；`:720-750` 加 FP8/INT8 分支；`ensure_mixed_buffers` 接 split-KV workspace | +75 / −10 |

**Acceptance**：
- [ ] `cargo check -p infer --no-default-features --features cuda,no-cuda` green（Mac 本地）
- [ ] `cargo build --release`（Linux CUDA，default feature）green
- [ ] **`cargo build --release --features tilelang-attn`** green（避免 `cfg(not(feature="tilelang-attn"))` 盲区，参考 errors/2026-04-28）
- [ ] `cargo test --release --test e2e` BF16 现有 case 不回归（两种 feature 各跑一次）
- [ ] 短 prompt FP8 e2e non-degeneracy gate 通过（两种 feature 各跑一次）
- [ ] 启动 server 跑一次 4k prompt c=4，scheduler 日志出现 `StepPlan::Mixed`（FP8 配置下）
- [ ] INT8 配置下同上验证

---

#### S3 — 数值 gate 四层防线

**Blocked by:** S2
**Blocks:** S5
**可并行：** S4（不互相阻塞）

**改动文件 (1) + 新脚本 (2)**：

| 文件 | 操作 | 估算 |
|---|---|---|
| `infer/tests/e2e.rs` | 加一条 prompt≈32k 的 long-prompt case；TileLang feature on/off 两个构建都跑 | +60 |
| `scripts/spotcheck_fp8_bf16_long.py` | 16 prompts × 64 token，复用 `errors/2026-04-30` 协议；prompt 用 32k 输入 | +180 |
| `scripts/spotcheck_arle_sglang_bf16.py` | 16 prompts × 64 token，ARLE-BF16 vs SGLang-BF16 锚点（防 self-consistent drift） | +160 |
| `scripts/spotcheck_long_tail.py` | 32 prompts × 256 token，FP8-vs-BF16 长尾扫，输出 divergence_p50 + tail_match | +200 |

**新增 wins 条目**：
`docs/experience/wins/YYYY-MM-DD-qwen3-fp8-bf16-spotcheck-32k.md`（§7.2，含四层防线数据）

**Acceptance（四层全过）**：

第一层 — e2e：
- [ ] 长-prompt e2e（TileLang on/off）：不 panic、首 5 字符不全相同、非空

第二层 — FP8-vs-BF16：
- [ ] `exact_pairs ≥ 3/16`
- [ ] `avg token-match`：分层处理（≥70% pass / 60–70% degraded+errors / <60% stop）
- [ ] `earliest divergence ≥ generated token 1`

第三层 — ARLE-BF16-vs-SGLang-BF16：
- [ ] `avg token-match ≥ 80%` （fail = ARLE-BF16 自漂移，回 S1 排查 prefill kernel）

第四层 — 长尾扫：
- [ ] `divergence_p50 ≥ generated token 30`
- [ ] `tail_match_last_64tok ≥ 50%`

- [ ] §7.2 wins 条目落地，含四层数据表

**未达标处理**：
- 第一/二层 fail → 回 S1 修 reduction
- 第三层 fail → ARLE-BF16 自漂移，**不是** FP8 问题，回 S1 排查 prefill kernel
- 第四层 fail → token 30 之后崩，bug 在 split-KV reduction 累积误差，回 S1

---

#### S4 — Bench harness：ARLE workload + SGLang baseline

**Blocked by:** —
**Blocks:** S5
**可并行：** S1 / S2 / S3 全程

**改动文件 (1) + 新脚本 (1)**：

| 文件 | 操作 | 估算 (+/−) |
|---|---|---|
| `scripts/bench_guidellm.sh` | `:68` 周边硬编码 `DATA=` 替换为 `case "${WORKLOAD:-default}"`；新增 `longctx-32k` 分支（prompt=32768、c=1,4、c=4 = 300s、c=1 = 360s） | +50 / −5 |
| `scripts/bench_sglang_longctx.sh` | SGLang server 启动 + guidellm 同 shape 跑；commit pin = `214c35b...`；启动参数对齐表（§5.5） | +200 |

**Acceptance**：
- [ ] `WORKLOAD=longctx-32k scripts/bench_guidellm.sh longctx-32k-smoke` (5s 烟测) 跑通
- [ ] 产出 `headline_table.md` 含 32k 行
- [ ] `scripts/bench_guidellm.sh smoke` (default) 不回归
- [ ] 默认调用方式 (`scripts/bench_guidellm.sh <label>` 不带 `WORKLOAD`) 行为完全不变
- [ ] `scripts/bench_sglang_longctx.sh smoke` 拉起 SGLang server + guidellm 跑通 30s 烟测
- [ ] SGLang 启动日志中 commit SHA 与 `214c35b...` 一致

---

#### S5 — Bench 实跑 + 主 wins

**Blocked by:** S2 + S3 + S4
**Blocks:** S6
**改动**：无代码改动（若诊断后需调参，回退到 S1）

**新增 wins 条目**：
`docs/experience/wins/YYYY-MM-DD-bench-guidellm-longctx-32k-c4-fp8.md`（§7.1）

**执行**：
```bash
# ARLE
WORKLOAD=longctx-32k scripts/bench_guidellm.sh longctx-32k-c4-fp8
# SGLang 同硬件同 shape
scripts/bench_sglang_longctx.sh longctx-32k-c4-fp8
# 合并两份 headline 表算 Δ%，按 TEMPLATE-bench-guidellm.md 写 wins
# Statistical note: c=4/300s 应有 ≥56 完成请求；c=1/360s 应有 ≥14
```

**Acceptance（primary）**：
- [ ] ARLE tok/s **within 5%** of SGLang（`ARLE/SGLang ≥ 0.95`，c=4 / 32k / FP8）
- [ ] ARLE 日志 `StepPlan::Mixed` 计数 > 0
- [ ] ARLE 日志 Qwen3-4B 路径 `StepPlan::Split` 计数 = 0
- [ ] wins 表含：tok/s、TTFT p50/p99、ITL p50/p99、plan 分布、Δ% vs SGLang、ARLE/SGLang commit SHA
- [ ] §7.1 wins 条目落地

**Acceptance（secondary）**：
- [ ] c=1 数据同 wins 落地（bonus，不阻塞 primary）

**未达标诊断树**：

| 现象 | 诊断 | 回退到 |
|---|---|---|
| `Mixed` 计数 = 0 | admission 不放 mixed | 调 `mixed_prefill_token_budget`，**不**回 S1 |
| `Mixed` 命中但 ARLE/SGLang < 0.95，ITL p99 长尾 | admission burst 或 K7 slot leak | ≤2 次旋钮迭代（c=2 重测 / `mixed_prefill_token_budget`）；无果 = stop 本计划 + fork K7 子任务 |
| `Mixed` 命中但 ARLE/SGLang < 0.95，ITL p50 慢 | KV_SPLIT 没自适应到 / kernel 带宽利用率 | 回 S1，用 `ncu` profile，调 KV_SPLIT 上下界 |
| `Mixed` 命中但 ARLE/SGLang < 0.95，prefill TFLOPs < SGLang 50% | **prefill 段 FP8 long-qlen 单 CTA 模式无 tensor core**（已知结构性缺口）| 接受差距 < 5% 进 wins；> 5% 时开 errors/ 立项独立 prefill kernel 工程 |
| 数值漂移退化 | reduction bug 复发 | 回 S3 验证四层，再回 S1 修 |
| SGLang OOM / NaN | SGLang 同 shape 不收敛 | 按 §5.6 退化 success criterion，wins 写 N/A + envelope check |

**目标处理规则**：
- ≤2 次旋钮迭代未达标 → 进入诊断树对应分支
- 诊断树指向"独立工程"或"K7 子任务"时 → stop 本计划，wins 按 degraded 写、开 errors/、ROADMAP 列后续
- **不在本计划内 in-flight 修 K7 / prefill kernel**

---

#### S6 — Qwen3-4B Split 路径守护 + ROADMAP 更新

**Blocked by:** S5
**Blocks:** —

**改动文件 (3)**：

| 文件 | 操作 | 估算 (+/−) |
|---|---|---|
| `infer/src/scheduler/cuda/execution.rs` | `:362-364` 加 `debug_assert!(model_family != Qwen3)`：Qwen3-4B/8B/14B 路径不应进 Split；注释明示 Split 仅服务 LoRA + Qwen3.5 + 触发删除条件（§6） | +18 / −2 |
| `infer/src/model/qwen3/forward.rs` | `supports_mixed_batch` 注释更新：删 "BF16 only" 表述；指向 §4.3 立 trait 触发条件 | +5 / −3 |
| `ROADMAP.md` | "Active Priorities" P0 行加一句：32k FP8 长上下文 within 5% of SGLang 已落 wins | +5 / −2 |

**Acceptance**：
- [ ] `cargo test --workspace --release` green
- [ ] `cargo clippy --workspace -- -D warnings` clean
- [ ] 复跑 S5 同样 bench，`debug_assert!` 不触发
- [ ] ROADMAP 更新指向新 wins 条目
- [ ] §4.4 sunset 路径明确：S1 acceptance 第 5 条（varlen qlen=1 性能 ≤ quantized:307+5%）通过 → 在 §6 OOS 更新 `quantized:307` 行的触发条件为 "本计划完成"；未通过 → 更新为 "永久并存（feedback_no_half_states 例外）"

---

### 8.4 工作量汇总

| 维度 | 数 |
|---|---:|
| Commits | 6 |
| 改动 Rust/CUDA 文件 | 9 unique |
| 新增测试 / 脚本 | 1 cargo test + 4 spot-check / bench scripts |
| 新增 wins | 2 |
| 新增 errors | 0–N（踩坑时）|
| 净 LoC | +1100 / −150 |

LoC 上调主因：S1 加 INT8 模板 (+30) + S3 增加两层防线 (+360) + S4 SGLang baseline 脚本 (+200)。

### 8.5 通用 verify 命令

```bash
# Mac CUDA-Rust 类型检查（开发本地）
cargo check -p infer --no-default-features --features cuda,no-cuda

# Linux CUDA full build（默认 feature）
CUDA_HOME=/usr/local/cuda cargo build --release

# Linux CUDA + TileLang（S2 必须额外验）
CUDA_HOME=/usr/local/cuda cargo build --release --features tilelang-attn

# 步骤特定
cargo test --release -p cuda-kernels --test attention_varlen_fp8_split   # S1
cargo test --release --test e2e                                          # S2/S3
cargo test --release --workspace                                         # S6
cargo clippy --workspace -- -D warnings                                  # S6
```

---

## 9 · 风险与回退

| 风险 | 触发信号 | 回退动作 |
|---|---|---|
| Split kernel reduction 数值不稳 | S3 第二层 spot-check < 60% | 回 S1，先把 KV_SPLIT=1 跑通（等价于现 single-CTA），再逐步上 split |
| ARLE-BF16 自漂移 | S3 第三层（ARLE-BF16 vs SGLang-BF16）< 80% | 回 S1，**不是** FP8 问题，排查 prefill kernel + RoPE |
| 32k Mixed plan 触发率低 | S5 日志 `StepPlan::Mixed` 占比 < 50% | 检查 `chunked_prefill_size` 和 `mixed_prefill_token_budget`，调 admission |
| FP8 varlen 比 BF16 varlen 慢 | S5 ARLE FP8 < ARLE BF16 | KV_SPLIT 调到上界 16；若仍慢，问题在 dequant 路径，profile `ncu` |
| **Prefill 段 FP8 长 qlen 单 CTA 无 tensor core**（关键裂缝）| S5 prefill TFLOPs < SGLang 50% | 接受差距 < 5% 进 wins；≥ 5% 时开 errors/ + ROADMAP 立项独立 "FP8 prefill tensor-core kernel" 工程，本计划接受 degraded 收尾 |
| ARLE 32k FP8 仍 < SGLang ≥5% | S5 Δ% < -5% 且诊断树未指向单参数旋钮 | ≤2 次迭代后 stop 本计划，按诊断树分支处理（K7 fork / errors / degraded wins）|
| 长 prompt 触发 OOM | S3 e2e panic | `mem_fraction_static` 调到 0.85；KV_SPLIT workspace 算入 budget |
| TileLang feature on 时 mixed 路径炸 | S2 acceptance `--features tilelang-attn` 构建 / e2e fail | `batch_decode.rs:551` 周边补 cfg 守护，确保 mixed FP8/INT8 在 TileLang 下也走对的 planning 路径 |
| SGLang baseline 不收敛 | S5 SGLang OOM / NaN | 按 §5.6 退化 criterion；wins 写 N/A + envelope；额外 errors/ 记录 SGLang 失败原因（不阻塞本计划）|

**目标可改但必须数据驱动**：跑数后发现 32k FP8 within 5% of SGLang 不可达，开 errors/ 条目说明结构性原因（如 prefill kernel 长 qlen 缺口），并按 ROADMAP 重画下一基线（例如先做 prefill kernel，再回头打 32k）。

---

## 10 · 关联文档

- 父计划：[`2026-04-23-cuda-decode-sglang-alignment.md`](2026-04-23-cuda-decode-sglang-alignment.md)
- 现状分析：[`../projects/2026-04-29-throughput-gap-analysis.md`](../projects/2026-04-29-throughput-gap-analysis.md) (K2 ≈ 50 tok/s lever，本计划是它在长上下文上的具体落地)
- Pipeline map：[`../projects/2026-04-29-scheduler-pipeline-map.md`](../projects/2026-04-29-scheduler-pipeline-map.md)
- Bench 协议：[`../bench-and-trace-spec.md`](../bench-and-trace-spec.md)
- Bench 矩阵设计：[`bench-matrix-design-2026-04-29.md`](bench-matrix-design-2026-04-29.md)
- 数值漂移历史：[`../experience/errors/2026-04-30-arle-fp8kv-numerical-drift.md`](../experience/errors/2026-04-30-arle-fp8kv-numerical-drift.md)
- TileLang feature 盲区先例：[`../experience/errors/2026-04-28-tilelang-prefill-short-qlen-nan.md`](../experience/errors/2026-04-28-tilelang-prefill-short-qlen-nan.md)
- 反速 模式：[`../../memory/feedback_no_speculative_interface_shaping.md`](../../memory/feedback_no_speculative_interface_shaping.md)、[`../../memory/feedback_no_half_states.md`](../../memory/feedback_no_half_states.md)

### §6.1 引用论文 / 系统

- [MagicDec, ICLR 2025](https://arxiv.org/abs/2408.11049) · [project](https://infini-ai-lab.github.io/MagicDec-part1/)
- [TriForce, COLM 2024](https://infini-ai-lab.github.io/TriForce/) · [code](https://github.com/Infini-AI-Lab/TriForce)
- [LongSpec, 2025-02](https://arxiv.org/abs/2502.17421)
- [DistServe, OSDI 2024](https://arxiv.org/abs/2401.09670) · [code](https://github.com/LLMServe/DistServe)
- [Mooncake, FAST 2025 Best Paper](https://arxiv.org/abs/2407.00079) · [code](https://github.com/kvcache-ai/Mooncake)
- [Sarathi-Serve, OSDI 2024](https://arxiv.org/abs/2403.02310)
- [Disaggregated Inference 18-month retro (UCSD Hao AI Lab)](https://haoailab.com/blogs/distserve-retro/)
- [DeepSpeed-Ulysses](https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md)
- [USP / Hybrid SP (Ulysses + Ring)](https://github.com/feifeibear/long-context-attention)
- [Arctic Ulysses (Snowflake) inference adaptation](https://insujang.github.io/2024-09-20/introducing-context-parallelism/)
- [llm-d blog (prefix cache offload + disagg)](https://llm-d.ai/blog)
- [DuoAttention, ICLR 2025](https://arxiv.org/abs/2410.10819) · [code](https://github.com/mit-han-lab/duo-attention)
- [Quest, ICML 2024](https://arxiv.org/abs/2406.10774)
- [MInference 1.0, NeurIPS 2024](https://arxiv.org/abs/2407.02490)
- [SnapKV, NeurIPS 2024](https://arxiv.org/abs/2404.14469)
- [PyramidKV / PyramidInfer](https://arxiv.org/abs/2406.02069)
- [vAttention, ASPLOS 2025](https://dl.acm.org/doi/10.1145/3669940.3707256)
- [vLLM speculative decoding 文档](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
