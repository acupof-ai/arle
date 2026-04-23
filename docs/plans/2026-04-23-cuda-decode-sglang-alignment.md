# 2026-04-23 CUDA Decode / Current SGLang Alignment Plan

**Status:** Active — current source of truth for CUDA decode alignment as of 2026-04-23  
**Commissioned by:** repo-grounded audit of "解码部分是否对齐当前 SGLang"  
**Scope:** current CUDA continuous-batching decode path only; compare against SGLang `main` as of 2026-04-23  
**Bench note:** docs-only planning update; no benchmark is required for this doc edit  
**Supersedes for current decode truth:** [`qwen35-sglang-parity.md`](qwen35-sglang-parity.md), [`p99-unified-mixed-batch.md`](p99-unified-mixed-batch.md), [`2026-04-21-sglang-control-exec-alignment.md`](2026-04-21-sglang-control-exec-alignment.md), [`2026-04-22-sglang-gap-closure-execution.md`](2026-04-22-sglang-gap-closure-execution.md)  
**Out of scope:** Metal runtime, PD cluster split, speculative decode, quant-specific custom decode rewrites, and copying SGLang's pool types verbatim

## 一句话结论

现在不能说 "`decode` 已经对齐当前 SGLang 了"。更准确的说法是：

- `scheduler policy` 已经对齐不少。
- `decode runtime contract`、`mixed-chunk execution shape`、`Qwen3.5 decode ownership` 还没有对齐。

## 本文把“对齐”定义成什么

本文把“和当前 SGLang 对齐”定义成下面 5 个条件同时成立，而不是只看某一条启发式或某一个 benchmark 点：

1. 一个 scheduler tick 只产出一个执行批次，而不是同一 tick 里分开 launch `prefill` 和 `decode`。
2. `mixed-chunk` 在 CUDA 上是真正的一次 mixed forward，而不是“同一轮先 prefill、再 decode”的双 launch 近似。
3. decode metadata、attention planning、graph input buffer 的 ownership 落在 execution/backend 边界，而不是散落在 scheduler 显式时序里。
4. `Qwen3`、`Qwen3.5`、`GLM4` 在 scheduler 可见层面消费同一个 batch contract；模型差异留在 model/backend 内部。
5. 剩余差异是刻意保留且有 bench/traces 支撑的实现差异，不是历史路径残留。

## 当前审计结论

### 已对齐的部分

| Area | Local repo | Current SGLang | Judgment |
| --- | --- | --- | --- |
| decode retract heuristic | [`infer/src/scheduler/cuda/decode.rs`](../../infer/src/scheduler/cuda/decode.rs) 先按 generated token 少的撤回；相同再按 prompt 更长的撤回 | [`schedule_batch.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py#L2064-L2083) 同样按 `(len(output_ids), -len(origin_input_ids))` | aligned |
| prefill admission 粒度 | [`infer/src/scheduler/cuda/execution.rs`](../../infer/src/scheduler/cuda/execution.rs) 已经是 chunk 预算，不再整条 prompt 一次性预留 | [`schedule_policy.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py#L767-L860) 的 `PrefillAdder` 也是 chunk 粒度 | aligned |
| decode 前置 staging 顺序 | scheduler 显式做 `upload_token_ids -> update_metadata -> plan_attention -> forward_decode_batch` | [`model_runner.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py#L2782-L2854) + [`flashinfer_backend.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/flashinfer_backend.py#L1097-L1199) 也是先准备 metadata/plan 再 decode | broadly aligned |

### 还没对齐的关键点

| Area | Local repo | Current SGLang | Why this blocks a real “aligned” claim |
| --- | --- | --- | --- |
| scheduler tick 形态 | [`infer/src/scheduler/cuda/execution.rs`](../../infer/src/scheduler/cuda/execution.rs) 的 `step()` 还能在同一 tick 里先 launch prefill、再 launch decode | [`scheduler.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L2308-L2400) 的 `get_next_batch_to_run()` 一次返回一个 batch | 现在是“同 tick 双执行”，不是 “单批次执行 contract” |
| mixed-chunk 执行形态 | CUDA 当前仍是同一 tick 的两次独立路径；[`crates/cuda-kernels/src/flashinfer.rs`](../../crates/cuda-kernels/src/flashinfer.rs) 里的 `update_mixed_batch()` 存在，但当前 CUDA path 没有消费它 | [`scheduler.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L2654-L2667) 会 `prepare_for_decode()` 后 `mix_with_running(...)`，形成一个真实 mixed batch | 这是当前最大缺口；没有这一点，不能说 decode shape 对齐 |
| metadata / graph ownership | scheduler 直接编排 `upload -> update_metadata -> plan_attention -> forward` | SGLang 把 lowering + graph inputs 放在 `model_runner` / attention backend | contract 分层还不是一套 |
| Qwen3.5 decode shape | [`infer/src/model/qwen35/batch_decode.rs`](../../infer/src/model/qwen35/batch_decode.rs) 仍是 piecewise graph；linear groups graph，full-attn 层 eager | SGLang 当前 decode ownership 更集中在 backend / runner 层 | Qwen3.5 仍是最明显的 model-specific decode outlier |
| KV contract | 本地是 native `PagedKVPool + slot_epoch + page table`，主路径 `page_size=16` | SGLang 还是 `req_to_token_pool + token_to_kv_pool_allocator` | 语义接近，但结构不同；这里需要先定义“要语义对齐还是实现同构” |

## 为什么旧计划现在会让人误读

过去几份计划各自解决了真实问题，但它们混合了三类内容：

- 架构意图，比如 `ScheduleBatch -> ForwardBatch`
- 已落地 runtime cleanup，比如 mixed legacy 删除、paged prefill 打通
- 当时版本的 SGLang parity 目标，比如 `0.5.9` 时代的 Qwen3.5 对齐

这会产生一个常见误读：  
“mixed legacy surface 已删掉” 被读成了 “当前 CUDA mixed path 已经和 SGLang 一样”。  
这不成立。当前 truth 是：legacy surface 删除了，但 canonical mixed path 仍然不是 SGLang 那种单次 mixed forward。

## 目标终态

达到本计划终态后，我们才可以对外说“CUDA decode 基本对齐当前 SGLang”：

1. scheduler 每轮只 lower 一次，输出一个 `ForwardBatch`。
2. mixed batch 在 CUDA 上只有一个 canonical path，并且是一次真实 forward。
3. scheduler 只表达“这轮要跑什么 batch”，不再手写 metadata / graph / attention planning 的时序细节。
4. `Qwen3`、`Qwen3.5`、`GLM4` 共用同一 scheduler-visible batch contract。
5. 如果保留 native `PagedKVPool` 而不照搬 SGLang pool types，这个差异被明确记录为“实现差异”，不是“未对齐”。

## 最小可交付 patch 集

如果目标只是把“不能说对齐”推进到“可以诚实地说大体对齐当前 SGLang”，最小 patch 集是下面 4 个：

### Patch 1 — 单批次执行 contract

把当前 `step()` 的 `decode + prefill` 双 launch 形态收敛成显式的 `ScheduleBatch -> CudaForwardBatch`。

交付物：

- scheduler 每 tick 只产出一个 batch variant：`DecodeOnly | PrefillOnly | Mixed`
- lowering 负责把 decode rows / prefill rows / sampling rows 装进一个 batch 结构
- `step()` 不再独立触发两条 canonical 执行路径

主要文件：

- `infer/src/scheduler/cuda/execution.rs`
- `infer/src/scheduler/cuda/decode.rs`
- `infer/src/scheduler/cuda/prefill.rs`
- `infer/src/model.rs`

验收：

- 主循环里没有“先 prefill 再 decode”的 canonical 双 launch 分支
- 日志/metrics 能直接看出每轮跑的是哪一种 batch

### Patch 2 — 真正的 CUDA mixed batch

让 mixed-chunk 真的变成一个 batch，而不是“同 tick 两次 launch”的近似。

交付物：

- 要么复用 `FlashInferDecodeMetadata::update_mixed_batch()`，要么用一个新的 `update_forward_batch()` 把它替代掉
- decode rows 和 extend rows 进入一次 attention plan
- 一个 mixed tick 只发生一次 model forward
- 删除剩余的 scheduler-visible mixed special path

主要文件：

- `crates/cuda-kernels/src/flashinfer.rs`
- `infer/src/model/qwen3/batch_decode.rs`
- `infer/src/model/glm4/batch_decode.rs`
- `infer/src/scheduler/cuda/execution.rs`

验收：

- mixed tick 的 trace 里只有一次 forward
- mixed rows、decode rows、prefill rows 都能在同一个 batch 结构上观测到

### Patch 3 — metadata / graph ownership 下沉

把 decode metadata、graph input buffers、attention plan 的 ownership 从 scheduler 下沉到 execution/backend 层。

交付物：

- scheduler 只构造逻辑 batch，不再直接调 `upload_token_ids -> update_metadata -> plan_attention`
- execution/backend 保持 stable device buffers，并自行决定 graph capture / replay / eager
- 让本地 contract 更接近 SGLang 的 `model_runner -> attention_backend`

主要文件：

- `infer/src/model.rs`
- `infer/src/model/qwen3/forward.rs`
- `infer/src/model/qwen35/forward.rs`
- `infer/src/model/glm4/forward.rs`
- `crates/cuda-kernels/src/flashinfer.rs`
- `infer/src/scheduler/cuda/decode.rs`

验收：

- scheduler decode path 不再显式串联 metadata lowering 细节
- graph reuse 与 buffer 生命周期主要由 backend/execution 管理

### Patch 4 — Qwen3.5 decode 合同化

Qwen3.5 可以继续保留内部 hybrid 复杂度，但不能继续作为 scheduler-visible 的 decode contract outlier。

交付物：

- `Qwen3.5` 消费和 `Qwen3` 相同的 `ForwardBatch` 接口
- piecewise graph 细节留在 model 内部，不再影响 scheduler shape
- 如需分两步，先统一 contract，再考虑是否把 full-attn eager 收进更统一的 graph 策略

主要文件：

- `infer/src/model/qwen35/batch_decode.rs`
- `infer/src/model/qwen35/forward.rs`
- `infer/src/model.rs`

验收：

- scheduler 不再对 `Qwen3.5` 持有额外的 mixed/decode 语义分叉
- remote bench 时 `Qwen3.5` 不再因为 scheduler contract 差异而天然掉队

## 推荐执行顺序

1. 先做 Patch 1，先把“每 tick 一个 batch”的骨架立住。
2. 再做 Patch 2，把 mixed-chunk 从双 launch 变成真 mixed forward。
3. Patch 2 落稳后做 Patch 3，把 ownership 从 scheduler 挪下去。
4. 最后做 Patch 4，清掉 Qwen3.5 这块最容易把结论搞混的 outlier。

这个顺序的目的很简单：  
先收敛 scheduler contract，再谈 backend ownership；先让 mixed shape 成立，再谈更深的 graph 整理。  
否则很容易再次出现“结构看起来更像 SGLang 了，但 canonical hot path 仍然不是同一个 shape”的半状态。

## 非目标

- 不要求把本地 `PagedKVPool` 改成 SGLang 一模一样的 pool 类型。
- 不在这轮顺手改 Metal。
- 不把 `page_size=16` vs `page_size=1` 的策略争论提前到 mixed path 修好之前。
- 不把 `Qwen3.5` 强行改成和 `Qwen3` 完全一样的内部 forward 形状；要求统一的是 scheduler-visible contract。

## 风险

### 风险 1 — 先做 contract，性能短期可能不升反降

原因：单批次 contract 和 ownership 下沉先改的是 shape，不一定立刻减少 kernel 时间。  
应对：每一步都要求 trace 可见，先验证 shape 变对，再用 bench 判性能。

### 风险 2 — Qwen3.5 hybrid 让“真 mixed forward”实现复杂化

原因：full-attn layers 和 linear-attn layers 本来就不是同一类算子。  
应对：要求的是统一外部 batch contract，不要求内部一步到位完全同构。

### 风险 3 — 继续保留旧 mixed path 会再次形成 half-state

原因：最容易出现“新 contract 落了，但旧 dual-launch path 还留着兜底”。  
应对：按 patch tranche 删除旧 canonical path，而不是长期并行保留。

## 验证与验收

### 静态检查

- `cargo check -p infer --release --no-default-features --features cuda,no-cuda`
- `cargo test --release --test e2e`
- `cargo test --release --test e2e_qwen35`
- `cargo clippy -p infer --release --no-default-features --features cuda,no-cuda -- -D warnings`

### 运行时观测

- 为每个 tick 记录 batch kind：`DecodeOnly | PrefillOnly | Mixed`
- mixed tick 记录 `decode_rows`、`prefill_rows`、`single_forward=true/false`
- 记录 graph path：`capture / replay / eager`

### 基准对照

- 远端 CUDA 用 `scripts/bench_guidellm.sh <label>` 做 before/after
- 对照对象是当前 SGLang `main`，不是旧的 `0.5.x`
- 至少覆盖 `c1/c2/c4/c8/c16`
- 输出 TTFT、ITL、output tok/s、完成率
- 落一条新的 `docs/experience/wins/`；如果本地不能跑，先开 `pending-remote` stub

## 历史文档怎么读

- [`qwen35-sglang-parity.md`](qwen35-sglang-parity.md)  
  这是 `SGLang 0.5.9` 时代的 Qwen3.5 parity 计划，保留做历史参考，不再代表当前 `main`。

- [`p99-unified-mixed-batch.md`](p99-unified-mixed-batch.md)  
  这是更大的 mixed/paged prefill 总计划，解释了为什么要做这些重构，但它不是今天的 decode truth。

- [`2026-04-21-sglang-control-exec-alignment.md`](2026-04-21-sglang-control-exec-alignment.md)  
  这是 `ScheduleBatch -> ForwardBatch` 思路的架构前置稿，可继续借用它的分层方向。

- [`2026-04-22-sglang-gap-closure-execution.md`](2026-04-22-sglang-gap-closure-execution.md)  
  这是 runtime gap-closure 的执行台账，记录了已经落地的工作，但不应被当成“当前 decode 已对齐”的证明。

## 审计参考

### Local repo

- [`infer/src/scheduler/cuda/decode.rs`](../../infer/src/scheduler/cuda/decode.rs)
- [`infer/src/scheduler/cuda/execution.rs`](../../infer/src/scheduler/cuda/execution.rs)
- [`infer/src/model/qwen35/batch_decode.rs`](../../infer/src/model/qwen35/batch_decode.rs)
- [`infer/src/model/qwen35/forward.rs`](../../infer/src/model/qwen35/forward.rs)
- [`crates/cuda-kernels/src/flashinfer.rs`](../../crates/cuda-kernels/src/flashinfer.rs)
- [`crates/cuda-kernels/src/paged_kv.rs`](../../crates/cuda-kernels/src/paged_kv.rs)

### Current SGLang

- [`schedule_batch.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py)
- [`schedule_policy.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py)
- [`scheduler.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py)
- [`model_runner.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py)
- [`flashinfer_backend.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/flashinfer_backend.py)
