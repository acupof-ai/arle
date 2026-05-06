# M_d — Tier-KV × spec-decode 协同

> Sub-plan of [`longctx-spec-tilelang-combo.md`](longctx-spec-tilelang-combo.md) §M_d.
> 前置:M_c(hybrid spec-rollback)和 unification M2(Metal kv-tier T2 adapter,已 land `f8f063d`)。
> Goal:把 Tier-KV(T0 GPU / T1 host pinned / T2 disk / T3 remote)和
> spec-decode 路径对齐,避免 (a) verify 时 cold-tier prefix 拖慢 spec step,
> 和 (b) 失败 verify 污染 RadixCache 留下"draft 看过的伪前缀"。

## 0. 现状盘点

- `infer/src/kv_tier/tier.rs::Tier` enum + `BlockLocation` 已定义,
  CUDA(T0/T1/T2)和 Metal(T0/T2)路径已分别由 `KvTierAdapter`(`f8f063d`)
  覆盖。
- `RadixCache`(`infer/src/prefix_cache.rs`)负责前缀公布 + 回收。
  spec_path.rs 当前没有"verify 失败回退"路径,所有 verify 中读到的 K/V
  默认走标准 admission path → 落到 RadixCache。
- spec-verify 当前不主动预取 T1/T2 数据,如果 prefix 在 T1 命中:第一次 verify
  时阻塞等 T1→T0 staging,几十 ms 抖动。

## 1. 设计

### 1.1 Spec-aware prefetch

scheduler 在 admission 阶段已经能看到 prefix 命中位置(T0/T1/T2/miss)。
spec-decode 进 verify 之前,scheduler 触发 **eager promote**:
- T1 → T0:host pinned → GPU,DMA 异步,verify 实际启动时基本 ready
- T2 → T0:走 (T2 → T1 → T0) 链,触发越早越好
- T3:不预取,fall back 到 vanilla decode

**接入点**:`spec_path.rs::SpecPath` 在 spec step 计划阶段(每 step 开始)
往 `kv_tier_coordinator.eager_stage(slot_idx, …)` 投递 hint。Coordinator 已
实现 stage,只缺 caller。

### 1.2 Spec-rollback 不污染 RadixCache

verify 失败位置 j ∈ [0, K):
- 当前路径:K 个 draft token 的 K/V 已经写进 paged-KV(decode_prep_paged)
  并跟随 RadixCache `publish` 流程,如果失败 verify 直接 abort,这 K 个 page
  会以"未引用"状态被 prefix 回收 — 但是 **当前实现没有显式区分 spec-tentative
  vs committed publish**,在并发场景下(同 prefix 的别的请求查 cache)可能命中
  这些"draft 看过但 reject 的 page",拿到的 K/V 不是 target 看到的真实 K/V,
  造成 prefix 污染下一请求结果。

**验证假设**:这是 *疑似* 问题(感觉应该有 bug,但还没构造 repro)。M_d Q1
就是写一个 repro test 先确认。如果实际上 publish 已经条件化在"committed"
位置之后,M_d Q2-Q5 删半工作量。

### 1.3 设计目标(假设 Q1 confirms 污染问题)

- spec-tentative 的 K/V 写入用 **scratch page**(分配在 spec workspace,
  不进 RadixCache 的 publish set);verify 接受后才 commit page-id 到主 KV
  + RadixCache。
- 等价地:`spec_path.rs` 显式持有 K 个 "pending publish" entry,verify
  done 后按 k_acc 决定 publish / drop。

## 2. 实现拆分

| 任务 | 文件 | LOC 估 | Owner |
|---|---|---|---|
| Q1. 写 repro test:两个 request,A 跑 spec-decode reject,B 之后用相同 prefix 查 cache,断言 B 的 token 输出跟 vanilla 一致 | `infer/tests/spec_decode_radix_pollution.rs`(新) | ~120 | Claude |
| Q2(if Q1 fails). spec-tentative scratch page model + commit barrier | `infer/src/scheduler/cuda/spec_path.rs` + `infer/src/prefix_cache.rs` | ~150 | Codex |
| Q3. eager-prefetch hint API on coordinator | `infer/src/kv_tier/coordinator.rs` | ~60 | Codex |
| Q4. spec_path.rs 调 eager-prefetch | `infer/src/scheduler/cuda/spec_path.rs` | ~40 | Codex |
| Q5. bench:64k context + spec-decode + Tier-KV(T0:T1 = 8GB:24GB)端到端 throughput vs SGLang HiCache + 无 spec | `bench_ab.sh` + 一段 longctx-32k profile 配置 | ~50 | Claude |
| Q6. wins entry | `docs/experience/wins/<date>-m_d-tier-spec-coordination.md` | 一个文件 | Claude |

总 LOC ~420(若 Q1 confirms),~270(若 Q1 disconfirms 污染但仍要做 prefetch)。
工期 4-5 天。

## 3. Acceptance

- Q1 新测试 pass(无论原 implementation 是否有污染,test 锁住未来回归)。
- vanilla longctx-32k bench 不回退。
- spec longctx-32k(c=4)bench:vs vanilla longctx-32k 至少 +30% throughput
  (combo plan §M_e 总目标 1.8× SGLang HiCache + 无 spec — M_d 单条贡献部分)。
- 无 RadixCache 污染:`infer/tests/spec_decode_radix_pollution` 在 N=20 个
  随机 reject 位置场景下 100% 一致性。

## 4. Risks + retreat

- **R1 — spec-tentative scratch page 增加内存压力**:每 spec step 临时占
  K 个 page。K=4, page=16 token, num_kv_heads=8, head_dim=128,bf16 →
  4 × 16 × 8 × 128 × 2 = 128 KB / spec step / request。c=16 同时 spec →
  2 MB。**不显著,可忽略**。Retreat 不需要。
- **R2 — eager prefetch 抢 PCIe 带宽,跟正常 admission staging 打架**:
  限制 prefetch 优先级 ≤ admission staging,coordinator 已经有 priority queue
  field。Mitigation:Q3 把 eager-prefetch 标 `Priority::Background`,
  admission staging 是 `Priority::Foreground`。
- **R3 — verify 失败发生时,K/V 的 page 已经被并发 admission 看走** —
  这种 race 罕见但理论存在。Mitigation:spec-tentative scratch page 有
  ref-count = 1,只有 owning request 引用;commit barrier 切到主 KV
  之前不可见。

## 5. 不在范围

- T3(remote)spec-decode 协同:涉及 NIXL / RDMA 的 spec verify,留到
  长 context disagg plan(`docs/projects/2026-04-30-longctx-32k-128k-leadership.md`)。
- Metal spec-decode:等 unification M5(Qwen3 cross-backend)。

## 6. References

- 父计划:[`longctx-spec-tilelang-combo.md`](longctx-spec-tilelang-combo.md) §M_d
- Tier 模型 + block location:`infer/src/kv_tier/tier.rs:14-30`
- KvTierAdapter trait(`f8f063d`):`infer/src/kv_tier.rs`
- spec_path.rs current admission stub:`infer/src/scheduler/cuda/spec_path.rs:280-291`
- RadixCache publish path:`infer/src/prefix_cache.rs`(verify Q1 起点)
