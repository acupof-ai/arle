# Plan — CUDA serving 性能优化 codex 协作 (2026-05-25)

**Status**: Draft · **Driver**: ckl · **Collab**: codex CLI 二次诊断 + adversarial 挑战

## Why now

2026-05-25 一天的实测 + nsys 把 §11 七条 hypothesis 走完一遍,最终落点:

- ✅ H1 (prefill queue 排队) confirmed — TTFT c=1→16 17.6× 线性放大
- ✅ H5 (event poll spin) confirmed — fix 落地 (50us backoff, 17× 削减)
- ❌ H2/H3/H4/H6 refuted — sampling/D2H/CUDA Graph/lm_head 全干净
- 同日 vs SGLang 0.5.12 头对头:**decode ITL 反超 21.8%,prefill TTFT 慢 115%,
  总 tok/s +21.5% (decode 段占长导致),TTFT p99 -54% (FIFO 严格,SGLang dynamic 长尾)**

同日 attempted A/B/C 全都没出 wall-clock win:

- A (`INFER_PREFILL_GRAPH=1` default) — c=16 -86% 直接 kill,errors entry 见
  [`docs/experience/errors/2026-05-25-prefill-graph-default-kill.md`](../experience/errors/2026-05-25-prefill-graph-default-kill.md)
- B (decode batch pipeline) — ±2% 噪音内,autoregressive 约束让"早 relaunch"
  理论收益就小
- C (prefill multi-stream) — partial implementation,inner kernel
  (attention/MLP/RMSNorm 等 ~30 处) 仍 hardcode `ctx.stream`,**实际没 overlap**

所以 50% GPU idle 还在,真正的 lever 没找到。

## 当前 wall-clock 分布 (nsys c=16 bench window 25s,post-H5)

| 项 | 时间 | % | 是否可压 |
|---|---:|---:|---|
| GPU kernel busy | 12.07s | 48% | 已基本满 (cutlass + tilelang) |
| **GPU idle (183 个 10-100ms gap)** | **12.62s** | **50%** | **核心目标** |
| 微 gap (<1ms) | 10ms | <0.1% | 正常 kernel boundary |
| 极少 outlier (>100ms) | <100ms | <0.4% | 忽略 |

CUDA API time (bench window):

| API | 时间 | % | 含义 |
|---|---:|---:|---|
| `cudaLaunchKernel` 5992 × 380µs | 2.29s | 9.2% | raw launch,prefill 未走 graph |
| `cuLaunchKernel` 1908 × 235µs | 0.45s | 1.8% | tilelang AOT |
| `cuGraphLaunch` 182 × 310µs | 0.06s | 0.2% | decode 走 graph ✓ |
| **GPU idle** (跨 launch) | **12.62s** | **50%** | **scheduler 决策** |

## 为什么 H1 是真敌人,但是 plan-level fix 而非 kernel-level

Bench plan tick 实测: `idle=22425 / decode=6382 / prefill=95 / split=0 / mixed=0`

- prefill 只占 0.33% (95/28902 ticks),split 和 mixed **完全 0**
- 16 个 session,c=16 时全用 paged KV → kv_util peak 100% → `select_prefill_candidates`
  返回空 → 退化到 `StepPlan::Decode` (`execution.rs:553`)
- 即使有 candidates,默认 `mixed_policy=Split`,prefill 在 has_decode 时只走 Split 单独 batch
- prefill 因此走串行 FIFO,新请求 TTFT 线性堆积

Plan agent design B-3 提出"早 relaunch 同 tick" — 实测无效,因为 autoregressive
constraint:decode N+1 必须等 N apply 完才能 launch,**而 N apply 占 ~微秒级,
本来就不是 gap 大头**。Gap 大头是 N 的 kernel 在 GPU 上跑 + scheduler 在
poll 等结果。

## 真正应该做的 5 个 axis (待 codex 挑战)

### Axis 1: **Admission policy 改造 — 让 prefill 不饿死**

bench 显示 c=16 时 ARLE 16 个 session 全 active(KV 锁定),new prompts 永远
排不进 `prefill_queue`(`Peak prefill_queue=15` 表示一次只能 14 个 wait)。
SGLang 的赢家是 `chunked-prefill + admission permissive at high c`。

可能的改动点:
- `infer/src/scheduler/cuda/runtime/admission.rs` — admission 不再要求 prefill
  immediately 拿满 KV,允许 chunk granularity admission
- `infer/src/scheduler/cuda/execution.rs:514` `collect_prefill_candidates` —
  当 KV 紧张时不要返回空,允许 chunked-prefill 走小 batch
- `infer/src/scheduler/policy.rs` — 加 `prefill_priority_boost` 参数,使长
  waiting 队列的请求 chunk 优先

**风险**: 改动 admission 可能影响 KV pressure 下的 robustness;需 c=16 W3
agent workload 验证不退化 (2026-05-08 deadlock 教训)

### Axis 2: **Mixed plan 强制激活 — `mixed_policy=Mixed` default**

当前 `mixed_policy` 默认 Split (`scheduler/types.rs:334`)。Mixed 允许同一 GPU
batch 同时跑 prefill rows + decode rows。Plan tick `mixed=0` 表示**从未触发**,
但代码路径存在 (`scheduler/cuda/execution.rs:558` 当 `allows_mixed=true && supports_mixed_batch`)。

可能的改动点:
- `scheduler/types.rs:334` — 默认改 Mixed
- 验证 `model.supports_mixed_batch` 在 Qwen3 dense 返回 true
- bench c=16 短 prompt + 长 prompt 都要测

**风险**: Mixed batch 需 model.forward 支持 prefill+decode 混合 row,可能 fail-fast
on Metal/CPU backend;需 cfg-gate

### Axis 3: **chunked prefill 默认 size 调大 + per-tick prefill chunk 数提升**

当前 `chunked_prefill_size=2048` (`scheduler/types.rs:403`),`max_prefill_tokens=16384`。
c=16 prompt 4096 → 每 prefill 需 2 个 chunk × 16 = 32 个 chunk 才完。如果一次 admit
1 个 chunk,scheduler 每 tick 只前进 1 chunk。

可能的改动点:
- chunked_prefill_size 加大到 4096 或 8192
- 允许同一 tick 内 admit 多个 prefill chunk (跨 session)
- `prefill_max_requests` 调小 (现 unbounded?) 让 mixed batch 更小

### Axis 4: **CUDA Graph cache for prefill — 真正可用版本**

A (`INFER_PREFILL_GRAPH=1`) 被 kill 是因为 8-slot LRU 在 c=16 thrash。
重新设计:

- 扩大 cache 到 32-64 slot
- 加更激进 bucketing (`PAGE_INDICES_BUCKET` 64→256, `PREFIX_ROWS_BUCKET` 128→512)
- 加 hit rate metric,bench 时确认 c=16 hit rate ≥ 80%
- 加 graph capture 失败 fallback 到 raw kernel(目前 capture 失败直接挂)

**风险**: A 之前 -86% kill 教训;改完必须 c=1/2/4/8/16 全跑通才能默认开

### Axis 5: **Multi-stream prefill 完整实现 — inner kernel 全部 stream 参数化**

C-1..C-4 框架已搭起来 (CudaPipelineStreamKind::Prefill 变体已在 cuda-kernels,
prefill_stream 分配,fence API 复用 record_pipeline_fence/wait_on_pipeline_fence)。

未完成:
- attention/MLP/RMSNorm/embedding/sampling 等 ~30 处 kernel call site 仍 hardcode
  `&ctx.stream` (例:`infer/src/ops/attention.rs:104+`、`infer/src/ops/linear.rs`)
- 需要新 `ComputeView` wrapper 或 `&CudaStream` 参数透传
- KV cache write fence 跨 stream 的正确性测试

**风险**: 改面巨大 (~30 文件);CUDA Graph capture 在 cross-stream 下行为未知

## 协作模式 — codex 二次诊断 + adversarial 挑战

每个 axis 独立向 codex 提问,要求:

1. **codex review** 现有 plan/diff 找 missing edge case
2. **codex challenge** 假设和数据,例如:
   - "Axis 1 改 admission 是否会让 KV 紧张时 OOM?"
   - "Axis 2 Mixed default 是否会让短 prompt c=1 退化?"
   - "Axis 5 multi-stream 在 L4 1-GPU 上是否真有 SM 并发?"
3. **codex consult** 实现细节 (CUDA Graph cross-stream / chunked prefill admission ordering)

## Implementation order (按"低风险 × 高信号"乘积)

| # | Axis | 改动量 | 实测难度 | bench-window 预估 |
|---|---|---|---|---|
| 1 | Axis 2 (Mixed default) | 极小 (1 行 + cfg-gate) | 容易 (c-sweep) | TTFT -20~40% (prefill 不再单独 batch) |
| 2 | Axis 3 (chunked prefill size + per-tick) | 小 (几行 default) | 容易 | TTFT -10~20% |
| 3 | Axis 1 (admission permissive) | 中 (admission policy 改) | 中 (需 W3 deadlock 防回归) | TTFT -30~50% |
| 4 | Axis 4 (graph cache 32 slot + bucketing) | 中 | 难 (c=16 需 hit rate ≥ 80%) | launch overhead -50~80% (9% wall) |
| 5 | Axis 5 (multi-stream 全套) | 大 (30 文件) | 极难 | 50% idle 中的一大块 |

**先做 Axis 2 + Axis 3 — 单行 default + 几行配置改动,bench c-sweep 直接验。
两者都成立才考虑 Axis 1。Axis 4/5 单独 plan 做。**

## Bench-loop 设计

每个 axis 改完跑同一组 baseline 对照:

- L4 远端 (`outdoors-arrow-guide-participate.trycloudflare.com`)
- Qwen3-4B BF16,4096 prompt / 256 output
- c=1, 4, 8, 16, max-seconds=60, warmup=5
- 关键指标:**TTFT p50/p99, ITL p50/p99, out tok/s, plan_label 分布**
- nsys c=16 30s 取 trace 对比 GPU idle gap 数 + cuStreamSynchronize / cuEventQuery 计数

baseline (no change): TTFT p50 c=16 = 12913ms, ITL p50 = 71.85ms, out tok/s = 164.

## Acceptance — 单 axis 不亏 baseline + 1 项指标改善

任一 axis 改动若:
- TTFT p50 / ITL p50 / out tok/s 任意一项回归 > 5% → revert,errors entry 记录
- 所有指标在噪音 ±3% 内 → close as "structural cleanup, no perf win"
- 至少 1 项指标 > 10% 改善 + 其余无 > 3% 回归 → land + wins entry

## Cross-refs

- 本日 H1+H5 nsys evidence:
  [`docs/experience/wins/2026-05-25-bench-guidellm-cuda-l4-arle-nsys-h1-h5-confirmed.md`](../experience/wins/2026-05-25-bench-guidellm-cuda-l4-arle-nsys-h1-h5-confirmed.md)
- 本日 vs SGLang 0.5.12 头对头:
  [`docs/experience/wins/2026-05-25-bench-guidellm-cuda-l4-arle-vs-sglang-headtohead.md`](../experience/wins/2026-05-25-bench-guidellm-cuda-l4-arle-vs-sglang-headtohead.md)
- A kill 教训 (graph cache thrash):
  [`docs/experience/errors/2026-05-25-prefill-graph-default-kill.md`](../experience/errors/2026-05-25-prefill-graph-default-kill.md)
- 待办:B/C 实测 ±2% 教训 errors entry (next session)
