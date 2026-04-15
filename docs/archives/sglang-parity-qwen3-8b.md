> **Archived 2026-04-15** — Steps 1–4 shipped; C=4 exceeds SGLang on
> Qwen3-8B per the win notes
> [`../experience/wins/2026-04-01-sglang-parity-steps1-4.md`](../experience/wins/2026-04-01-sglang-parity-steps1-4.md)
> and
> [`../experience/wins/2026-04-01-throughput-vs-sglang.md`](../experience/wins/2026-04-01-throughput-vs-sglang.md).
> The remaining C=1 gap is kernel-bound and outside this plan's scope.
>
> **Disambiguation**: this is the **Qwen3-8B** parity plan. For the
> still-active **Qwen3.5** parity work (batched prefill outstanding),
> see [`../plans/qwen35-sglang-parity.md`](../plans/qwen35-sglang-parity.md).
> The two were renamed during this archive to make the model split
> obvious.

---

# Plan: 追平 sglang 0.5.9 (Qwen3-8B · A100-40GB · 同配置)

> Status: **Done — Archived** — Steps 1-4 complete. C=4 exceeds sglang. C=1 gap is kernel-bound.
> Created: 2026-04-01
> Updated: 2026-04-01
> Goal: 在相同硬件和模型下，匹配或超越 sglang 0.5.9 的默认配置吞吐量

---

## Baseline 数据

| 配置 | infer (当前) | sglang 0.5.9 | Gap |
|------|-------------|-------------|-----|
| 128in/512out C=1 | 70.2 tok/s | 76.3 tok/s | **-8%** |
| 512in/256out C=1 | 68.1 tok/s | 75.8 tok/s | **-10%** |
| 1024in/256out C=1 | 63.5 tok/s | 75.5 tok/s | **-15%** |
| 2048in/256out C=1 | FAIL (503) | 74.8 tok/s | **broken** |
| 512in/256out C=4 | FAIL (503) | 255.9 tok/s | **broken** |
| ITL p50 (128in) | 13.7ms | 13.0ms | **-5%** |
| TTFT p50 (128in) | 19ms | 49ms | **+158%** (infer wins) |

## sglang 默认开启的优化

以下是 sglang 0.5.9 在 A100-40GB 上 serve Qwen3-8B 默认开启的优化列表。**只实现这些，不做额外优化**：

| 优化 | sglang 默认 | infer 现状 |
|------|------------|-----------|
| FlashInfer decode attention | ON | ON (FlashInfer batched decode) |
| CUDA Graph for decode | ON (batch 1-32) | ON (batch 1-4) |
| Chunked prefill | ON (4096 tokens) | ON (512 tokens, decode 时 64) |
| RadixCache prefix caching | ON (LRU eviction) | OFF (数据结构有但没接入) |
| Continuous batching (FCFS) | ON | 部分 (固定 slot, 无 dynamic insert) |
| Batched GPU sampling | ON (FlashInfer kernel) | 部分 (greedy batched, 但 per-token D2H sync) |
| Overlap scheduling (H2D/D2H) | ON | OFF |
| torch.compile | OFF | N/A |
| Speculative decoding | OFF | N/A |
| Priority scheduling | OFF | OFF |

## 差距根因分析

### 1. ITL 差距 (13.7ms vs 13.0ms, -5%)

**根因**: per-token D2H sync
- `ops/sampling.rs:120`: 每个 token 采样后都 `ctx.sync()` + `clone_dtoh()`
- sglang 的 sampling 是 batched 的，一次 kernel launch 采样整个 batch，然后一次 D2H

**文件**: `infer/src/ops/sampling.rs` (lines 99-128)

### 2. 长 context 吞吐差距 (63.5 vs 75.5 tok/s at 1024in, -15%)

**根因**: chunked prefill 太小
- infer: 512 tokens/chunk (decode 活跃时 64 tokens/chunk)
- sglang: 4096 tokens/chunk
- 更小的 chunk = 更多的 kernel launch overhead + 更多的 scheduler 轮次

**文件**: `infer/src/scheduler/cuda/mod.rs:28-30`
```rust
pub(super) const PREFILL_CHUNK_SIZE: usize = 512;
pub(super) const PREFILL_CHUNK_SIZE_WITH_DECODE: usize = 64;
```

### 3. C>1 并发失败

**根因 A**: Scheduler queue 拒绝
- `SchedulerHandle::submit()` 在 `waiting_count >= max_waiting` 时返回 503
- `max_waiting = num_slots * 4` (4 slots → 16)
- 但问题不是队列大小，而是 slot 释放和新请求到达之间的竞态

**根因 B**: 无 continuous batching
- infer 的 scheduler 是 round-robin 固定 slot：每个 request 占一个 slot 直到完成
- sglang 有 dynamic batch：一个 decode step 可以处理所有 active requests，不限 slot 数
- 新 prefill 可以在任何 decode step 之间插入

**文件**: `infer/src/scheduler/cuda/runtime.rs` (全文)

### 4. CUDA Graph batch size 限制

- infer: warmup batch 1-4 (= num_slots)
- sglang: warmup batch 1-32
- 当 C>4 时 infer 没有对应的 graph

**文件**: `infer/src/scheduler/cuda/core.rs:196`

---

## 实施计划

### Step 1: 消除 per-token D2H sync

**目标**: ITL 13.7ms → 13.0ms
**修改文件**: `infer/src/ops/sampling.rs`, `infer/src/scheduler/cuda/decode.rs`

现状:
```rust
// sampling.rs — gpu_sample_core()
fn gpu_sample_core(...) -> Result<u32> {
    launch_sample_kernel_inner(...);
    ctx.sync();                    // ← 阻塞 CPU 等 GPU
    let result = out.clone_dtoh()?; // ← 单 token D2H copy
    Ok(result[0] as u32)
}
```

改为:
```rust
// 在 decode.rs step_decode_batch() 中:
// 1. 所有 token 的 sampling kernel launch (不 sync)
// 2. 一次 ctx.sync()
// 3. 一次 batch D2H readback
// 4. 分发结果到各 request
```

具体步骤:
1. 在 `BatchDecodeBuffers` 中预分配 `sampled_tokens_gpu: CudaSlice<i32>` 和 `sampled_tokens_host: Vec<i32>` (max_batch_size)
2. `sample_batch_greedy()` 已经是 batched 的 — 确认它不做 per-token sync
3. 非 greedy 路径: 改 `select_tokens_batch()` 为一次 kernel launch + 一次 sync + 一次 batch D2H
4. 移除 `gpu_sample_core()` 中的 `ctx.sync()` 和 `clone_dtoh()`，改为 caller 负责 sync

**验证**: `cargo test --release`, bench ITL 对比

### Step 2: 增大 chunked prefill size

**目标**: 长 context 吞吐 +8-10%
**修改文件**: `infer/src/scheduler/cuda/mod.rs`

```rust
// 改前
pub(super) const PREFILL_CHUNK_SIZE: usize = 512;
pub(super) const PREFILL_CHUNK_SIZE_WITH_DECODE: usize = 64;

// 改后 (匹配 sglang 默认)
pub(super) const PREFILL_CHUNK_SIZE: usize = 4096;
pub(super) const PREFILL_CHUNK_SIZE_WITH_DECODE: usize = 512;
```

注意事项:
- 需要确认 prefill buffers 能处理 4096 token 的 batch
- 检查 `prefill_buffers.rs` 中 buffer 分配是否动态或固定
- 如果 hidden states 是固定大小分配，需要扩大到 4096 * hidden_dim

**验证**: bench 1024in/256out C=1 吞吐对比

### Step 3: 修复 Scheduler queue (支持 C>1)

**目标**: C=4 不再 503
**修改文件**: `infer/src/scheduler/types.rs`, `infer/src/scheduler/cuda/core.rs`

改动:
1. `SchedulerHandle::submit()` — 当 queue 满时不立即返回 503，改为 bounded channel 的 blocking send with timeout
2. 或者: 大幅增加 `max_waiting` (从 `num_slots * 4` 改为 256)
3. HTTP handler 侧加入重试逻辑 (当 503 时等待一个 decode step 的时间后重试)

```rust
// core.rs line 124 — 改前
let handle = SchedulerHandle::with_max_waiting(tx, model_id, num_slots * 4);
// 改后
let handle = SchedulerHandle::with_max_waiting(tx, model_id, 256);
```

**验证**: bench 512in/256out C=4

### Step 4: Continuous batching — dynamic batch decode

**目标**: C=4 吞吐从 0 → 接近 sglang 256 tok/s
**修改文件**: `infer/src/scheduler/cuda/runtime.rs`, `execution.rs`, `decode.rs`, `core.rs`

这是最大的改动。当前 scheduler 模型:
```
Request → assign_slot → prefill (chunked) → decode (round-robin in slot) → finish → free slot
```

改为:
```
Request → queue → batch_prefill (如果没有 active decode) → join decode batch
         所有 active requests 一起做 batched decode
         新 request 可以在任何 decode step 后 join
```

具体步骤:
1. **移除 fixed slot 概念**: 不再是 "N slots, each with its own state"。改为一个统一的 "active requests" 列表
2. **Unified KV pool**: 所有 request 共享 PagedKVPool (已有实现)。每个 request 用 pool 分配的 page，不再有 per-slot contiguous KV
3. **Batch decode**: `step_decode_batch()` 已经支持 arbitrary batch size — 不需要改 forward path
4. **Dynamic batch insert**: `assign_slots()` 改为 `try_start_prefill()` — 每个 decode step 后检查是否有新 request 可以开始 prefill
5. **CUDA Graph batch sizes**: warmup 1-32 (当前 1-num_slots)

核心修改:
```rust
// runtime.rs — 新的 main loop
loop {
    drain_incoming_requests();
    
    // Step 1: batch decode all active requests
    if !active_decode.is_empty() {
        step_decode_batch(&active_decode);
        emit_deltas(&active_decode);
        remove_finished(&mut active_decode);
    }
    
    // Step 2: start new prefills (if budget allows)
    while let Some(req) = waiting.front() {
        if can_start_prefill(req) {
            let req = waiting.pop_front();
            step_prefill_or_chunk(req);
            if req.prefill_done() {
                active_decode.push(req);
            }
        } else {
            break;
        }
    }
}
```

**关键约束**:
- CUDA Graph capture 需要固定 batch size → 对不同 batch size 各 capture 一个 graph (1, 2, 4, 8, 16, 32)
- PagedKVPool 必须完全接入 (当前 partially wired)
- FlashInfer metadata (indptr, indices, last_page_len) 需要按 dynamic batch 更新

**验证**: bench 512in/256out C=1,2,4,8

### Step 5: 接入 RadixCache prefix caching

**目标**: 多轮 agent 场景 TTFT 大幅下降
**修改文件**: `infer/src/radix_tree.rs` (已有), `infer/src/scheduler/cuda/runtime.rs`

当前状态: `radix_tree.rs` 有完整的 radix tree 数据结构，但没有接入 scheduler。

实现:
1. 每个 request 提交时，在 radix tree 中查找最长 prefix match
2. 匹配的 prefix tokens 跳过 prefill，直接从 pool 中引用已有的 KV cache pages
3. 只 prefill 未匹配的 suffix tokens
4. Request 完成时，将其 KV cache pages 注册到 radix tree (而非立即释放)
5. LRU eviction: 当 pool 空间不足时，从 radix tree 驱逐最久未使用的 prefix

**验证**: 连续发相同 system prompt 的请求，第二个应该 TTFT ≈ 0

### Step 6: Overlap scheduling

**目标**: H2D/D2H 与 compute 重叠
**修改文件**: `infer/src/scheduler/cuda/decode.rs`

实现:
1. 在 decode batch 的 GPU forward 阶段，同时在另一个 CUDA stream 上做:
   - 下一轮的 token IDs H2D copy
   - 上一轮的 sampled tokens D2H copy
2. 需要 2 个 CUDA stream: compute stream + copy stream
3. 用 CUDA events 做 stream 间同步

这个优化在 batch size 大的时候收益更明显。C=1 时收益可能 <2%。

**验证**: bench C=4,8 对比

---

## 实施顺序和依赖

```
Step 1 (D2H batch)  ─────────────────────────→ 独立, 立即开始
Step 2 (chunk size)  ─────────────────────────→ 独立, 立即开始  
Step 3 (queue fix)   ─────────────────────────→ 独立, 立即开始
Step 4 (cont. batch) ←── 依赖 Step 3 ────────→ 最大改动
Step 5 (prefix cache) ←── 依赖 Step 4 ────────→ 需要 dynamic KV pool
Step 6 (overlap)     ←── 依赖 Step 1 + Step 4 → 收尾优化
```

**并行策略**: Step 1, 2, 3 可完全并行开发。Step 4 是关键路径。

## 预期结果

| 配置 | 当前 | Step 1-3 后 | Step 4-6 后 | sglang |
|------|------|------------|------------|--------|
| 128in/512out C=1 | 70.2 | ~75 | ~77 | 76.3 |
| 512in/256out C=1 | 68.1 | ~74 | ~76 | 75.8 |
| 1024in/256out C=1 | 63.5 | ~73 | ~76 | 75.5 |
| 2048in/256out C=1 | FAIL | ~72 | ~75 | 74.8 |
| 512in/256out C=4 | FAIL | ~100 | ~250+ | 255.9 |
| ITL p50 | 13.7ms | ~13.0ms | ~13.0ms | 13.0ms |

## 不做的事情

以下是 sglang 默认 OFF 的优化，我们也不做:
- torch.compile (sglang 默认 OFF)
- Piecewise CUDA Graph (sglang 默认 OFF)
- Speculative decoding (sglang 默认 OFF)
- DP Attention (单卡不适用)
- Priority scheduling (sglang 默认 OFF)
- FP8/INT8 量化 (sglang 默认不量化)

## 测量方法

每个 Step 完成后跑:
```bash
source .venv/bin/activate
# 启动 server
LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
  ./target/release/infer --model-path /root/models/Qwen3-8B --port 8000 \
  --num-slots 4 --max-seq-len 4096

# 跑 bench
python scripts/bench_throughput_sweep.py --url http://localhost:8000 --label "infer-stepN"
```

对比同配置 sglang:
```bash
python -m sglang.launch_server --model-path /root/models/Qwen3-8B --port 30000
python scripts/bench_throughput_sweep.py --url http://localhost:30000 --label sglang
```
