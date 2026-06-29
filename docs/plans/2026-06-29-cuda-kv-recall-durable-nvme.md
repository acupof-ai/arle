# CUDA KV Recall — Durable NVMe Spill 完整方案

**日期:** 2026-06-29. **作者:** ckl / Claude. **状态:** 方案阶段.

---

## 1. 架构总览：三层 KV Tier

```
┌─────────────────────────────────────────────────────────────────┐
│ L1: Device HBM (PagedKVPool / TokenKVPool)                      │
│    活跃请求的 KV pages — attention 直接读取                        │
│    容量: ~VRAM × mem_fraction_static                             │
│    延迟: GPU 显存带宽 (~1.6 TB/s H20)                             │
├─────────────────────────────────────────────────────────────────┤
│ L2: Host DRAM (CudaKvTierStore::host: BTreeMap<u64, Vec<u8>>)   │
│    降级的 prefix pages / 写穿的 recall pages / G3 slot images      │
│    容量: dram_fraction × MemAvailable (default 0.5, ~850 GiB)    │
│    延迟: CPU memcpy (~50-100 GB/s DDR5)                          │
│    持久性: 进程内; 崩溃即丢失                                      │
├─────────────────────────────────────────────────────────────────┤
│ L3: NVMe Disk (CudaKvTierStore::disk: DiskTier)                 │
│    溢出的 pages / 跨重启持久化 recall                              │
│    容量: ssd_fraction × free_disk (default 0.5)                  │
│    延迟: NVMe read (~10 µs + ~2-3 GB/s 顺序)                     │
│    持久性: 跨进程 (durable namespace + manifest)                   │
│    模式: ephemeral (prefix tier, 进程退出即 wipe)                  │
│          durable  (recall tier,  跨重启保留)                      │
└─────────────────────────────────────────────────────────────────┘
```

`CudaKvTierStore` (`crates/infer-cuda/src/kv_tier.rs`) 统一管理 L2+L3：
- L2 总是存在（构造时 budget ≥ 0 即有 host map）
- L3 通过 `set_disk` (ephemeral) 或 `set_disk_durable` (durable) 可选接入
- 插入时 L2 满了自动 spill 到 L3（LRU eviction）
- L3 durable 有 manifest 持久化，重启后 `load()` 重放恢复

---

## 2. 当前接线状态

### 2.1 活路径

| Tier 实例 | 持有者 | 用途 | L2 | L3 |
|---|---|---|---|---|
| `QwenCudaExecutor::tier` | dense Qwen3 prefix | 前缀降级/提升/写穿 | ✅ DRAM | ✅ ephemeral (`set_disk`) |
| `Qwen35CudaExecutor::slot_tier` | Qwen3.6 G3 | 整槽 capacity spill | ✅ DRAM | ❌ |
| `Qwen35CudaExecutor::recall_tier` | Qwen3.6 recall | 写穿 evict + 预取 | ✅ DRAM | ❌ |
| `Dsv4CudaExecutor::slot_tier` | DSv4 G3 | 整槽 capacity spill | ✅ DRAM | ❌ |
| `Dsv4CudaExecutor::recall_tier` | DSv4 recall | 写穿 evict + 预取 | ✅ DRAM | ❌ |

### 2.2 死代码

`kv_tier.rs` 中被 `#[allow(dead_code)]` 标记、无调用方的路径：

| 函数 | 功能 |
|---|---|
| `set_disk_durable` | 创建跨重启 persistent NVMe namespace |
| `load` | 重放 manifest 恢复上进程的 spill index |
| `durable_namespace` | 生成 per-process stable namespace 路径 |
| `weights_epoch_tag` | 模型权重版本 hash（detect stale KV） |
| `DiskTier::write_manifest` | 持久化 `{key, byte_len}` 索引 |
| `DiskTier::parse_manifest` | 解析 manifest |
| `CudaKvTierStore::persist` | Drop 时 flush manifest（但 durable 永假） |

---

## 3. 接入目标

让 `--kv-recall --kv-ssd-path /nvme/kv` 时，Qwen3.6 的 `recall_tier` 使用 durable NVMe spill：
- 进程运行中：L2 host DRAM 满 → spill 到 L3 NVMe
- 进程重启后：`load()` 重放 manifest → 恢复 spill index → 之前 evict 的页面可寻址

### 3.1 调用链

```
loaded.rs::build_cuda_engine (or cuda_serve_handle)
  │
  ├─ executor.set_dram_fraction(fraction)      // L2 budget from measured DRAM
  ├─ executor.set_kv_tier_budget_bytes(bytes)  // explicit --kv-t1-budget-bytes override
  ├─ executor.set_kv_recall(true)              // 构建 recall_tier (L2 DRAM)
  └─ executor.set_kv_tier_disk(root, budget)   // 接入 L3 durable NVMe ← 本次改动
```

`set_kv_recall` 和 `set_kv_tier_disk` 调用顺序取决于 CLI 参数顺序，两种顺序都需支持。

### 3.2 文件变更清单

#### 文件 1: `crates/infer-cuda/src/kv_tier.rs`

- 去掉 `set_disk_durable` `#[allow(dead_code)]`
- 去掉 `load` `#[allow(dead_code)]`
- 去掉 `durable_namespace` `#[allow(dead_code)]`
- `weights_epoch_tag` 已是 `pub(crate)`，无需改动

#### 文件 2: `crates/infer-cuda/src/executor.rs`

**2a. `Qwen35CudaExecutor` 新增字段**（已在上一轮加入）：

```rust
model_path: PathBuf,          // from_qwen35_safetensors 初始化
weights_epoch: String,        // weights_epoch_tag(&model_path)
disk_root: Option<PathBuf>,   // None → set_kv_tier_disk 设置
disk_budget: Option<usize>,   // None → set_kv_tier_disk 设置
```

**2b. `Qwen35CudaExecutor` 新增方法 `set_kv_tier_disk`**：

```rust
pub(crate) fn set_kv_tier_disk(&mut self, root: PathBuf, budget_bytes: usize) -> bool {
    self.disk_root = Some(root);
    self.disk_budget = Some(budget_bytes);
    if let Some(tier) = self.recall_tier.as_mut() {
        let page_bytes = tier.page_bytes();
        tier.set_disk_durable(
            self.disk_root.clone().unwrap(),
            self.disk_budget.unwrap(),
            page_bytes,
            self.weights_epoch.clone(),
        );
    }
    true
}
```

**2c. `Qwen35CudaExecutor::set_kv_recall`** 扩展：

在 `self.recall_tier = Some(tier)` 前插入 durable load/attach 逻辑：

```rust
if enabled && self.recall_tier.is_none() {
    let page_bytes = ...;                          // 已有代码
    let tier = CudaKvTierStore::with_budget(...);  // 已有代码

    // Try loading prior session's durable NVMe spill
    if let (Some(root), Some(budget)) = (self.disk_root.as_ref(), self.disk_budget) {
        let loaded = tier.load(
            root.clone(), *budget, page_bytes, self.weights_epoch.clone(),
        );
        if !loaded {
            tier.set_disk_durable(
                root.clone(), *budget, page_bytes, self.weights_epoch.clone(),
            );
        }
    }

    self.recall_tier = Some(tier);
}
```

**2d. `RealCudaExecutor::set_kv_tier_disk`**：

```rust
// Before: Self::Qwen35(_) | Self::Dsv4(_) => false,
// After:
Self::Qwen35(q) => q.set_kv_tier_disk(root, budget_bytes),
Self::Dsv4(_) => false,
```

#### 文件 3: `crates/infer-api/src/loaded.rs`

更新 `--kv-ssd-path` 错误消息（行 ~1643）：

```rust
// Before:
"Qwen3-dense only today; DSv4/hybrid pending #85"

// After:
"Qwen3-dense + Qwen3.6 recall; DSv4 pending"
```

### 3.3 不涉及的模块

- **Qwen dense**: prefix tier 已有 ephemeral L3 — 不需要 durable
- **Qwen3.6 G3 `slot_tier`**: 整槽 spill 是容量优化，不需要跨重启持久化 — 不需要 L3
- **DSv4**: `slot_swap_store` 是纯 `BTreeMap`，不经过 `CudaKvTierStore`
- **Metal**: 有独立的 `MetalSsdTier`

---

## 4. 已知 Bug

### 4.1 `durable_namespace` 使用 PID

```rust
fn durable_namespace(root: PathBuf) -> PathBuf {
    root.join(format!("arle-kv-recall-{}", std::process::id()))
}
```

注释说 "stable across restarts"，但 PID 每次启动都变 → `load()` 永远找不到上一进程的 manifest → 永远从冷启动开始。

实际行为：
- 第一次运行 `set_disk_durable`：写入 blobs + manifest → OK
- 第二次运行 `load()`：PID 变了 → `namespace.join(MANIFEST_FILE)` 不存在 → `parse_manifest` returns `None` → `load` returns `false` → 走 `set_disk_durable` 建新 namespace → 上一进程的 blobs 变成孤儿文件（不可寻址，不回收）
- **不会 crash，但跨重启恢复功能实际上是 broken 的**

正确修复：namespace 用 model path 的 hash（而非 PID），这会让同一模型+同一 SSD 根路径的多次启动复用同一 namespace。需要修改 `DiskTier::Drop` 的 wipe 语义（durable namespace 在 Drop 时不能被 wipe）。

---

## 5. 风险分析

### 5.1 风险矩阵

| # | 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|---|
| R1 | PID-based namespace → 跨重启恢复 broken | 100% | **func** — blobs 孤儿子无法寻址 | 写进 Known Issue；本次仅接通链路，实际生效需 fix namespace |
| R2 | NVMe 延迟 spike（wear leveling / GC）→ TTFT 不可预测 | 中 | **perf** — prefetch sync 完成，TTFT 含 L3 read 延迟 | 控制面：后续可改 async prefetch。当前 `prefill_row_recall` 中 prefetch 是 sync 的 |
| R3 | Manifest 损坏 → panic | 极低 | **func** | ✅ `parse_manifest` returns `None`, `load` returns `false`, falls through to `set_disk_durable` |
| R4 | NVMe 满 → write_through 失败 | 中 | **perf** — recall 效果退化，KV 正确性不变 | ✅ `tier.insert` returns `false` → caller keeps page resident |
| R5 | Manifest incremental write 开销 | 低 | **perf** | ✅ `write_block_cache_sharded` 无 fsync（cache 语义）；manifest 用 atomic temp+rename |
| R6 | OPD 权重更新 → 旧 epoch manifest 跳过 | 低 | **correctness** | ✅ `weights_epoch_tag` 基于 safetensors 文件 hash → epoch 变化 → `load` returns `false` |

### 5.2 核心问题：L3 能做稳定性无关层吗？

**可以。** 而且当前架构已经是这么设计的。

**L3 的故障模型**（从 recall tier 视角）：

```
L3 失败场景                 →  系统行为                  →  影响
─────────────────────────────────────────────────────────────────────
L3 write 失败              →  insert 返回 false         →  页面保留在 HBM，不 evict
L3 read 失败               →  read 返回 Err             →  跳过该 block，stays evicted
L3 满                     →  insert/lru spill 拒绝     →  降级回纯 L2 行为
L3 慢                     →  prefetch 延迟增加          →  TTFT 变差，但 correct
进程崩溃                   →  L2 + L3 全丢              →  重启后 load 失败 → 冷启动
Manifest 损坏              →  parse_manifest → None    →  冷启动，blobs 变孤儿
DiskTier Drop wipe(durable=false) →  durable=true 不 wipe → ✅
```

**关键设计点**：tier 每一层都是 *optional luxury*。最底层 ground truth 始终是 HBM 中的活跃页面 + 可以重算的 attention。Tier 是 "如果可以的话，少算一点" 的优化。每一处 `tier.insert()` / `tier.read()` 失败都有一条 no-tier fallback path，不产生 correctness 错误。

**L3 不做的事**：
- 不做唯一数据源 — L2 始终有一份（L3 是 L2 的 spill）
- 不做同步 durability 保证 — `write_file_atomic_cache` 无 fsync
- 不做 crash recovery — crash 后重新 prefill 即可

**结论**：L3 出任何问题，系统降级到 L2-only 或 L1-only 模式，输出正确性不变。唯一可能的影响是 TTFT 延长或 HBM 压力增加。L3 是稳定性无关的优化层。

---

## 6. 性能验证方案

### 6.1 当前可用的测量手段

| 测量维度 | 工具 | 覆盖范围 |
|---|---|---|
| 客户端延迟+吞吐 | `scripts/bench_guidellm.sh <label>` | TTFT p50/p99, ITL p50/p99, tok/s |
| GPU kernel timeline | `scripts/profile_nsys_guidellm.sh` | cudaMemcpy D2H/H2D 字节+带宽 |
| GPU kernel 效率 | `scripts/profile_ncu_guidellm.sh` | SM 利用率, mem 带宽 |
| 服务端 tier 计数 | `/v1/stats` → `KvSystemMetrics` | demote_mset_* / promote_mget_* / fetch_wait_ms / fallback_recompute |
| Prometheus | `GET /metrics` | 同上，持续 scrape |

### 6.2 当前缺失的 Recall Tier 测量

Recall write-through + prefetch 路径（`prefill_row_recall` executor.rs:4093-4250）的 I/O **完全不被 `KvSystemMetrics` 追踪**。缺失：

| 缺失指标 | 建议新增字段 | 采集位置 |
|---|---|---|
| recall write bytes / count / ms | `recall_write_bytes / recall_write_count / recall_write_ms` | `prefill_row_recall` write-back-evict 循环 |
| recall prefetch bytes / count / ms | `recall_prefetch_bytes / recall_prefetch_count / recall_prefetch_ms` | `prefill_row_recall` prefetch 循环 |
| recall prefetch fallback | `recall_prefetch_fallback` | `reinstate_slot_page` 返回 None 时 |
| recall tier location (L2 vs L3) | 复用 `KvTierLocation` | `recall_tier.location(key)` 查询 |

### 6.3 写放大定义

```
写放大 = total_bytes_written_to_tier / total_bytes_evicted_from_hbm

其中:
  total_bytes_written_to_tier = recall_write_bytes (计数器新字段)
  total_bytes_evicted_from_hbm = evict_pages × page_bytes

理论最优值: 1.0 (每页写一次 → 一次 page evict = 一次 tier write)
实际情况: 同一页可能被多次 evict 和 prefetch，写放大 > 1.0
```

写放大只能通过新增计数器测量，nsys 只能看总 D2H 带宽，不能区分 "新写" vs "重复写"。

### 6.4 建议的 e2e 验证流程

```
1. bench_guidellm.sh recall_baseline     → TTFT/ITL/tok-s 基线 (no --kv-recall)
2. bench_guidellm.sh recall_dram         → 同 workload + --kv-recall (L2 only)
3. bench_guidellm.sh recall_nvme (
     --kv-recall --kv-ssd-path /nvme)    → 同 workload + L2+L3
4. /v1/stats diff                        → prefix tier 计数器变化
5. nsys profile diff                     → D2H/H2D memcpy 的时机+量
     预期: no-recall 时 decode 无 D2H/H2D
     预期: recall 时 prefill 后有 D2H (write-through evict)
          下一个请求 prefill 有 H2D (prefetch)
6. 写放大: 计数器新字段计算
```

---

## 7. 实施步骤

| Step | 文件 | 内容 | 预估 |
|---|---|---|---|
| S1 | `kv_tier.rs` | 去 dead_code 标记 | 1 commit |
| S2 | `executor.rs` | Qwen35 `set_kv_tier_disk` 新方法 | 1 commit |
| S3 | `executor.rs` | `set_kv_recall` 扩展 (load→set_disk_durable) | 1 commit |
| S4 | `executor.rs` | `RealCudaExecutor::set_kv_tier_disk` Qwen35 arm | 1 commit |
| S5 | `loaded.rs` | 更新 `--kv-ssd-path` 错误消息 | 1 commit |
| S6 | — | `cargo check --features cuda,no-cuda` | verify |
| S7 | — | `cargo test -p infer-api --features cuda,no-cuda --lib` | verify |

Steps S1-S5 是独立变更，可以各自提交。
