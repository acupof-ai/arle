# Speculative Decoding 接入计划

_Date: 2026-04-14_

## 1. 目标

在当前 `agent-infer` 架构上接入标准 speculative decoding，优先落 CUDA backend，采用：

- draft model：`Qwen3-0.5B`
- target model：现有 CUDA 主模型
- KV 组织：`paged prefix (committed) + non-paged suffix buffer (draft / verify tentative)`
- verify：target model 对每个 request 一次 forward 验证全部 draft tokens
- 调度：在现有 CUDA scheduler 外包一层 `SpeculativeScheduler`

本轮目标是把架构、接口和落地顺序定清楚，并补一个 `DraftEngine` skeleton，先把“第二个 CUDA 模型可加载、可跑 draft token”这件事做出来。

## 2. 现状结论

### 2.1 `infer/src/speculative.rs`

当前已经具备：

- `SpecConfig`
- `TokenProposal`
- `VerificationResult`
- `verify_tokens()`
- `AcceptanceTracker`
- `DraftModel` trait

缺口也很明确：

- `DraftModel` 还是 phase-0 stub，`draft_batch(&self, token_ids, k)` 只有单 request 视角，没有 request-local state / suffix handle。
- `verify_tokens()` 目前只有 `p(t_i)` / `q(t_i)` 标量；一旦发生 rejection，并没有足够的信息从 `max(0, p - q)` 精确重采样。
- 现有 scheduler decode 路径只支持 “每个 request 一步一个 token”。

### 2.2 CUDA scheduler / model 路径

关键现状：

- `infer/src/scheduler/cuda/decode.rs` 现在是 batched single-token decode。
- `infer/src/model.rs` 里的 `ModelForward` 只有：
  - `forward_prefill()`
  - `forward_decode()`
  - `forward_decode_batch()`
- `PagedKVPool` 现在只有：
  - `alloc_tokens(slot, n)`：向 slot 末尾追加
  - `free_slot(slot)`：整槽释放
  - 没有 “commit suffix / rollback suffix / truncate suffix” 语义
- `Scheduler::warmup_cuda_graphs()` 只为 decode 单阶段做 graph warmup，没有 speculative 的 draft/verify 两阶段概念。

### 2.3 KV pattern 结论

`docs/research/speculative-decoding-feasibility.md` 的结论是对的，而且和当前实现最匹配：

- committed prefix 保持在 `PagedKVPool`
- speculative suffix 不写进 paged pool，而是放 request-local contiguous buffer
- 全部 acceptance 之后，再把 accepted suffix fold 进 paged prefix
- rejection 时直接丢 suffix，不对 paged pool 做 page-level rollback

这比给 `PagedKVPool` 做事务回滚安全得多，也更贴合当前 token-level pool 的实现方式。

## 3. 推荐架构

## 3.1 总体数据流

每个 decode round：

1. `SpeculativeScheduler` 选出本轮 decode requests
2. `DraftEngine` 为每个 request 生成最多 `K` 个 draft tokens
3. target model 对每个 request 的 `prefix + K draft` 做一次 verify forward
4. CPU 侧执行 acceptance / rejection sampling
5. accepted tokens 被提交到 request 状态，并把对应 KV 从 suffix 提交到 paged prefix
6. rejection 之后的 suffix 直接丢弃
7. 更新 `AcceptanceTracker`；如果均值低于阈值，退回普通 decode

## 3.2 组件拆分

### A. `DraftEngine`

位置建议：

- runtime 实现先放在 `infer/src/speculative/cuda.rs`
- 只做 draft model 的 load / single-request draft token 生成

第一版建议：

- draft model 只支持 `Qwen3-0.5B`
- 先复用 `load_qwen3_components()` / `Qwen3Model`
- draft pass 先做 greedy-only，避免一开始就要求 sampled-token full-prob readback
- skeleton 阶段允许 “每次调用 fresh state + prefill prefix”，先把接口跑通

后续 production 化再补：

- request-local draft state
- draft KV 的 prefix 复用
- batched draft generation

### B. target-side suffix buffer

新增 request-local 结构：

- `TargetSuffixBuffer`
- 容量固定为 `max_spec_tokens`
- 存 target verify 阶段的 tentative K/V
- 生命周期只覆盖一个 speculative round

建议放置位置：

- request 级元数据：`infer/src/scheduler/cuda/request.rs`
- GPU buffer / helper：`infer/src/model/*` 或 `crates/cuda-kernels/src/*`(post `a4e12f5` kernel-crate 抽取)

这个 buffer 的职责：

- verify forward 时作为 append-only 写目标
- acceptance 后按 accepted 长度 commit 到 paged pool
- rejection 后整个清空

### C. `SpeculativeScheduler`

新增文件建议：

- `infer/src/scheduler/cuda/speculative.rs`

职责：

- 包装现有 `Scheduler<M>`
- 复用已有：
  - slot admission
  - prefix reuse
  - prefill
  - `emit_delta`
  - cleanup
- 只替换 decode 路径：
  - 原 `step_decode_batch()`
  - 变成 `step_speculative_decode_batch()`

建议结构：

```rust
pub struct SpeculativeScheduler<M, D>
where
    M: ModelForward,
    D: DraftModel,
{
    inner: Scheduler<M>,
    draft: D,
    spec: SpecRuntime,
}
```

这里不建议从零重写 scheduler；应该直接复用现有 `Scheduler` 的 slot / queue / metrics / cleanup 机制。

## 4. 需要改的接口

## 4.1 `DraftModel` 需要升维

当前 trait：

```rust
fn draft_batch(&self, token_ids: &[u32], num_draft_tokens: usize) -> Result<TokenProposal>;
```

问题：

- 名字叫 batch，实际上只有单 request
- 没有 request-local draft state 句柄
- 没有返回 full draft distribution，无法精确 rejection resample

建议分两阶段：

### Phase 1

保留现有 trait，先把 `DraftEngine` 落下去，供 scheduler prototype 使用。

### Phase 2

扩成显式 batch 形态，例如：

```rust
pub struct DraftRequest<'a> {
    pub slot: usize,
    pub prefix_last_token: u32,
    pub max_draft_tokens: usize,
    pub sampling: &'a SamplingParams,
}

pub struct DraftBatchOutput {
    pub proposals: Vec<TokenProposal>,
    pub draft_logits: DraftLogitBatch,
}
```

production 版本必须至少保留：

- `q(t_i)` for all sampled draft tokens
- draft full logits for each speculative position，供 rejection 时构造 `max(0, p - q)`

否则只能做近似算法，不是严格 speculative decoding。

## 4.2 `ModelForward` 需要 verify 专用入口

现在的 `forward_prefill()` / `forward_decode_batch()` 都不直接适配 verify。

verify 需要：

- 读 committed paged prefix
- 一次 forward 处理 `K` 个 draft tokens
- 对每个位置拿到 logits / sampled-token prob
- tentative K/V 写入 suffix buffer，而不是直接写 paged pool

建议新增：

```rust
pub trait ModelForward {
    type VerifyContext: Send;

    fn create_verify_context(
        &self,
        max_batch_size: usize,
        max_spec_tokens: usize,
        pool: &PagedKVPool,
    ) -> Result<Self::VerifyContext>;

    fn forward_verify_batch(
        &self,
        draft_tokens: &[u32],
        states: &mut [Self::State],
        slot_indices: &[usize],
        draft_lens: &[usize],
        paged_kv_pool: &PagedKVPool,
        verify_ctx: &mut Self::VerifyContext,
    ) -> Result<VerifyBatchOutput>;
}
```

`VerifyBatchOutput` 至少要包含：

- 每个 speculative position 的 target sampled-token logprob / prob
- 每个 request 的 bonus position logits
- rejection 时可重采样所需的 full target logits

建议不要把整块 `[B, K+1, vocab]` 全量 D2H；应保持 logits 在 GPU，上层只拉：

- 被采样 token 的标量概率
- bonus row
- 真正发生 rejection 的那一行

## 4.3 `PagedKVPool` 只加“提交 accepted suffix”能力，不加回滚事务

不要做：

- paged pool 内页级事务
- speculative allocate 然后 rollback page tables

建议加：

```rust
pub fn append_committed_suffix(
    &mut self,
    slot: usize,
    suffix_token_indices: &[u32],
    accepted: usize,
) -> Result<()>;
```

但更推荐的职责分配是：

- `PagedKVPool` 仍只关心 committed token indices
- suffix buffer 自己管理 tentative indices / contiguous rows
- commit 时调用一个显式 copy helper，把 accepted rows scatter 到 paged pool 新分配的 slots

也就是说：

- paged pool 只知道 committed rows
- suffix buffer 负责 tentative rows
- rollback 只发生在 suffix buffer

这样和当前 token-level `PagedKVPool` 的语义最一致。

## 5. CUDA Graph 方案

## 5.1 为什么现有 graph 不够

当前 graph warmup 只覆盖：

- batch size 维度
- decode 单 token 路径

speculative 之后至少有两类图：

- draft pass graph
- verify pass graph

而 verify pass 还多一维：

- `spec_len = 1..K`

## 5.2 推荐 graph key

建议 graph cache key 变成：

```rust
enum SpecPhase {
    Draft,
    Verify,
}

struct SpecGraphKey {
    phase: SpecPhase,
    batch_size: usize,
    spec_len: usize,
}
```

### Draft pass

第一版可以不 capture，先 eager 跑通。

原因：

- draft model 初版本来就是 skeleton
- 先把 verify / suffix buffer 接起来更关键

### Verify pass

verify pass 才是主收益路径，优先 capture：

- `batch_size` 复用现有 warmup schedule
- `spec_len` 只 warmup `1..=K`

如果显存压力太大，先只 warmup常见配置：

- `spec_len = K`
- `batch_size = 1, 2, 4, 8, ...`

然后 fallback 到 eager。

## 6. 分阶段实施顺序

## Phase 0：本轮已落 / 可直接落

- `infer/src/speculative.rs` 保持 CPU 可验证逻辑
- 新增 `DraftEngine` skeleton，能加载 `Qwen3-0.5B`
- 文档明确 scheduler / verify / KV 的下一步设计

## Phase 1：draft engine production 化

- 给 draft engine 增加 request-local state
- scheduler 为 active slot 维护 draft-side state
- 避免每个 speculative round 都重跑 draft prefill

文件：

- `infer/src/speculative/cuda.rs`
- `infer/src/scheduler/cuda/request.rs`

## Phase 2：verify API + suffix buffer

- target model 增加 `VerifyContext`
- 加 `forward_verify_batch()`
- 加 request-local `TargetSuffixBuffer`

文件：

- `infer/src/model.rs`
- `infer/src/model/qwen3/*`
- `crates/cuda-kernels/src/paged_kv.rs`
- `infer/src/scheduler/cuda/request.rs`

## Phase 3：scheduler 接 speculative

- 新增 `SpeculativeScheduler`
- decode step 改为：
  - draft
  - verify
  - accept/reject
  - emit
- acceptance 低于阈值时自动 fallback 到普通 decode

文件：

- `infer/src/scheduler/cuda/speculative.rs`
- `infer/src/scheduler/cuda/decode.rs`
- `infer/src/scheduler/cuda/core.rs`

## Phase 4：CUDA Graph

- verify pass graph cache
- optional draft pass graph cache
- warmup / invalidation 策略接进现有 graph 管理

文件：

- `infer/src/scheduler/cuda/core.rs`
- model-specific `VerifyContext`

## 7. 我建议的取舍

### 7.1 Draft KV 不要一开始就 paged 化

建议 V1：

- target：`paged prefix + suffix buffer`
- draft：contiguous per-request state

原因：

- target path 已经强绑定 `PagedKVPool + FlashInfer`
- draft model 更小，先用 contiguous state 能显著降低接入复杂度
- 真正的 correctness / rollback 难点在 target verify，不在 draft

后面如果 draft memory 成为瓶颈，再把 draft prefix 也升级到 paged 模式。

### 7.2 先只支持 greedy speculative

原因：

- 当前 `select_token_with_logprob()` 在 greedy fast path 已经有现成支持
- 非 greedy 需要 sampled-token prob / full logits 的稳定 API
- 先把 greedy 跑通，后面再补 temperature / top-p

### 7.3 `SpeculativeScheduler` 先包装，不改现有 `Scheduler` 对外接口

原因：

- 现有 `Scheduler` 已经承担太多职责
- speculative 是 decode-path enhancement，不该把 prefill / emit / cleanup 重写一遍
- wrapper 更利于灰度开关和 fallback

## 8. 风险

- **精确 rejection sampling 需要 full draft + full target logits。**
  这不是当前 `TokenProposal` 的标量接口能表达的，必须升维。
- **verify path 的 suffix buffer 会碰到新的 CUDA kernel / metadata 组合。**
  这里是最大实现风险。
- **CUDA Graph 组合数会增加。**
  如果对 `batch_size × spec_len × phase` 全量捕获，显存会膨胀；必须允许 eager fallback。
- **prefix cache / slot reuse 要和 speculative state 一起清理。**
  request cleanup 需要同时释放：
  - committed paged prefix
  - target suffix
  - draft state

## 9. 完成标准

第一阶段可接受的 done 定义：

- `DraftEngine` 可加载 `Qwen3-0.5B`
- `cargo check -p infer --no-default-features --features cuda,no-cuda` 通过
- 有一份明确的实现文档，把 API、KV、graph、scheduler 的落地顺序说清楚

第二阶段 production done：

- greedy speculative decode 跑通
- verify 一次 forward 校验全部 draft tokens
- acceptance/rejection 正确
- fallback 可靠
- CUDA graph 在 verify path 可用

## 10. 本轮已落的代码边界

本轮 `DraftEngine` skeleton 的边界应明确理解为：

- 已完成：
  - second CUDA model load
  - greedy draft token generation
  - draft token prob 占位接线
- 未完成：
  - request-local draft KV 复用
  - target verify integration
  - exact rejection resampling
  - scheduler wiring

也就是说，这一轮是把 speculative decoding 的“地基接口和实现顺序”钉住，而不是假装整条链路已经完成。
