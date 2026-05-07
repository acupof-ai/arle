# M_pgc — Prefill CUDA graph capture(对标 SGLang piecewise prefill graph)

> **Created 2026-05-07 EOD+9** by Claude after dual-evidence locked the
> M_world1 P0.1 root cause:
> - [SGLang prefill stack survey](../research/2026-05-07-sglang-prefill-stack-survey.md):
>   SGLang BF16 GEMM 走 torch.nn.functional.linear → cuBLAS,**跟 ARLE 同一路径**;
>   2.03× TTFT 优势主因是 piecewise CUDA graph capture for prefill(2048-token
>   bucket,整 layer loop graph)。ARLE 跑 prefill eagerly。
> - [ARLE prefill GEMM callgraph](../research/2026-05-07-arle-prefill-gemm-callgraph.md):
>   ARLE 走 cublasLt 同 algo space,M_pf-gemm Phase 0 KILL 已证 algo selection
>   不是 leverage point;Phase 2/2.5(TileLang prefill GEMM port)demoted P3。
>
> 这两份证据合起来锁死方向:**prefill graph capture 是主路径**。

## Priority & ROI

**Priority**: **P0 #1**(其他所有 prefill 优化 candidate 都让位)。

**ROI evidence**:

| Engine | longctx 4k/c=4 TTFT p50 | out tok/s | prefill graph? |
|---|---:|---:|---|
| SGLang 0.5.11 | **972.9 ms** ⭐ | 164.3 | ✅ piecewise(2048 bucket)|
| vLLM s8 | 1177 ms | 159.1 | ✅(MergedColumnParallelLinear + graph)|
| ARLE post-`3e0ed5a` | 2005.9 ms | 152.49 | ❌ eager |
| ARLE pre-fuse(`786a20a`)| 1976.4 ms | 153.83 | ❌ eager |

ARLE vs SGLang gap = **2.03×**;M_world1 +30% lead target = TTFT ≤ **748 ms** /
out tok/s ≥ **213.6**。

**预期收益数学**:

per-layer prefill kernel launches × 36 layer × 4 chunks/req = **~17k host launches**
per request。每 launch ~5-10 µs CPU overhead → **~85-170 ms per request 纯 launch
overhead**(占当前 1976.4 ms 的 4.3-8.6%)。

但 SGLang 优势远大于 8.6%(972 ms vs 1976 ms = -50.8%),说明 graph capture 的
真正收益不是消除 host launch,**而是消除 host-driven dispatch 的 cache thrashing
+ 让 GPU 跑满 stream pipeline**。这跟 H_LP3 数据(`Kernel2` 56.7% TTFT 但 launch
overhead < 5%)对得上 — kernel 时间占大头的真因是每 launch 都 stalled,而不是
launch 数量。

**保守预测**:Phase 0(单 2048-bucket capture 整 prefill layer loop)→ TTFT
**1976 → ~1400 ms**(-29%),仍未达 SGLang(972 ms)但缩进 30% target 边界。
Phase 1(42 桶 + 元数据 graph integration)→ 进一步 -15% 进入 1200 ms 区间。

**Negative case**:
- **cuBLASLt graph capture 不稳**:linear.rs:934 已有 `gemm_graphsafe_cuda`
  (no-workspace cuBLAS variant,decode path 已用),prefill route 过去即可。
  风险点:graphsafe 路径可能比 cublasLt 路径慢 ~5-10%,导致 graph capture 收益被
  本身慢路径吃掉。Phase 0 必须 bench `gemm_graphsafe_cuda` eager 路径作 control。
- **bucket fragmentation**:多桶(Phase 1)每桶 ~10 buffer × 2-4 MB = 20-40 MB,
  42 桶 × 30 MB = 1.26 GB 显存(KV pool 占了 11.7 GB,剩 4.3 GB 给 graph buckets
  紧)。Phase 0 单桶约 30-50 MB,无压力。
- **e2e 正确性回归**:graph 只 capture 一次,后续 replay 必须 deterministic。
  greedy_consistency 测试是关键 gate。

**Kill criteria**:
- Phase 0 实现完 bench Δ TTFT < 10% improvement → 假设错误,demote P3,
  重审 SGLang survey 推断;切换到 Phase 2 TileLang prefill GEMM port
- e2e / greedy_consistency 跑挂(replay deterministic 假设 broken)→ rollback
  + errors entry
- 实现 LOC > 500 → 暂停,reframe scope(decoder graph pattern 应能 ≤ 360)

## P0 survey(代码为真理)

### 已有基础设施(complete reuse)

- `infer/src/model/cuda_graph.rs` — `CudaGraphState`,capture-on-first-call,
  replay-thereafter(decode-path 已 production)
- `infer/src/model/qwen3/forward.rs::supports_cuda_graph_decode` — feature gate
  pattern,`enable_cuda_graph && self.lora.is_none()`
- `infer/src/main.rs` — `--cuda-graph` / `--disable-cuda-graph` CLI 已 wire
- `infer/src/ops/linear.rs:934 gemm_graphsafe_cuda` — no-workspace cuBLAS
  graph-safe GEMM,decode 已用

### Hard blockers(必须改)

| File:line | 问题 | 改造 | LOC |
|---|---|---|---:|
| `infer/src/model/qwen3/prefill.rs:42-80` | `PrefillBuffers::new()` 每次 `HiddenStates::zeros()` 10 个 buffer | bucket-级别 pre-alloc + pool reuse | 40-60 |
| `infer/src/model/qwen3/prefill.rs:485-503` | 3× `clone_htod` page table 在 capture 内 | 移到 `prepare()` 阶段 capture 之前 | 20-30 |
| `infer/src/ops/linear.rs:672,688,692-695` | Marlin W4 workspace `alloc_zeros` 在 hot path | Phase 0 disable Marlin in graph path,fallback eager | 30-50 |
| `infer/src/ops/linear.rs:945-955` | cuBLASLt 默认非 graph-safe | 强制 route 到 `gemm_graphsafe_cuda:934` for prefill | 20-30 |

### Soft blockers(reorder 即可)

| File:line | 问题 | 改造 | LOC |
|---|---|---|---:|
| `infer/src/model/qwen3/prefill.rs:472-475` | host-side `packed_tokens` 构造 | 移到 capture 前 `prepare()` | 0(reorder) |
| `infer/src/ops/attention.rs:591-621` | per-seq prep_cuda loop | bucket-fixed 后 unrolled 即 graph-safe | 10-20 |
| `infer/src/ops/attention.rs:548-569` | TileLang kernel dispatch by (q_heads, kv_heads) | capture 前选定 kernel pointer | 0 |
| `infer/src/model/qwen3/prefill.rs:438-448` | logits buffer on-demand alloc | Phase 0 init pre-alloc max-bucket logits | 20 |

### 已 graph-safe(无需改)

- `infer/src/ops/attention.rs:574-664` — TileLang paged prefill execution
- `infer/src/ops/embedding.rs` — embedding gather kernel
- RMSNorm / SiLU / residual add — 单 kernel launch 无 alloc

## Scope

### Phase 0 — Single 2048-bucket capture(license-or-kill)

**目标**:整 prefill layer loop 在 single 2048-token bucket 下 graph-captured,
跑 longctx 4k/c=4 bench 验证 ROI。

**工作分解**:

1. **`PrefillGraphState`** — 镜像 `CudaGraphState` decode pattern,key 为
   `(bucket_token_count)`,Phase 0 写死 2048(`infer/src/model/qwen3/prefill_graph.rs`,~80 LOC 新文件)
2. **`PrefillBufferPool`** — bucket-级别 pre-alloc 10 个 device buffer + page
   table device buffer + logits device buffer(`prefill.rs::PrefillBuffers` 加
   `from_pool()` constructor,~50 LOC)
3. **prepare-then-capture flow** — host metadata 计算 + H2D 在 `prepare()`,
   capture 内只有 kernel launches(~30 LOC reorder)
4. **`gemm_graphsafe_cuda` route** — Phase 0 prefill GEMM 全走 graph-safe 路径,
   bench 验证 vs eager 是否退化(`linear.rs::gemm_into` 加 `prefer_graph_safe`
   flag,~20 LOC)
5. **Marlin guard** — `if use_marlin && in_graph_capture { fallback eager }`
   (~30 LOC)
6. **`supports_cuda_graph_prefill()` trait method** — 镜像
   `supports_cuda_graph_decode`,Phase 0 仅 Qwen3 enable
   (`forward.rs` impl,~10 LOC)
7. **CLI flag** — `--cuda-graph-prefill` 默认 off(opt-in),
   `--cuda-graph-prefill-bucket` 默认 2048(`main.rs`,~15 LOC)
8. **Telemetry counters** — `prefill_graph_hit` / `prefill_graph_miss` /
   `prefill_graph_fallback_reason`(`/v1/stats` 扩展,~25 LOC)

**Phase 0 总 LOC est**:**~260**(audit 估 360 - 复用 decode CudaGraphState ~100)。

**Validation**:

| 测试 | 命令 |
|---|---|
| 编译 | `cargo check --release -p infer --features cuda` |
| 静态 lint | `cargo clippy --release -p infer --features cuda -- -D warnings` |
| no-cuda 类型 gate | `cargo check -p infer --no-default-features --features cuda,no-cuda` |
| e2e 正确性 | `cargo test --release -p infer --features cuda --test e2e` |
| greedy 一致性 | `cargo test --release -p infer --features cuda --test greedy_consistency` |
| 微 bench | `INFER_CUDA_GRAPH_PREFILL=1 scripts/bench_guidellm.sh m_pgc-phase0-2048bucket --concurrencies 4 --max-seconds 120 --warmup 10 --data 'prompt_tokens=4096,...,output_tokens=256,...'` |
| Δ vs control | 对照 `bench-output/2026-05-07-longctx-4k-c4`(1976.4 ms TTFT)+ `2026-05-07-sglang-longctx-4k-c4`(972.9 ms)|

**License decision**:

- ≥ 25% TTFT improvement(1976 → ≤ 1480 ms)→ Proceed Phase 1(多桶)
- 10-25% improvement → Proceed but lower priority(opt-in flag,document trade-off)
- < 10% improvement → KILL,demote P3,errors entry,重审 SGLang survey 推断

### Phase 1 — Multi-bucket scheduler(after Phase 0 license)

| 子任务 | LOC |
|---|---:|
| 42-bucket table(2048 / 1792 / 1536 / 1280 / 1024 / 960 / ... / 64) | 30 |
| `PrefillGraphState` 多 key + LRU eviction(显存压力)| 60 |
| Bucket selection from incoming chunk size(round-up)| 20 |
| Multi-shape e2e + greedy bench | 0 |

**Phase 1 总 LOC est**:**~110**。

### Phase 2 — Conditional follow-ups(gated on bench)

- 若 Phase 1 仍 ≥ 1.5× SGLang → 重审 attention kernel(TileLang vs FlashInfer
  paged)A/B 对照,M_b.3 G1 segment-aware grid 复活
- 若 cuBLASLt graphsafe 退化 ≥ 5% → Phase 2 TileLang prefill GEMM port 复活
- 若 Marlin disable in graph path 限制 quantized 模型 → 单独 Phase 2.x Marlin
  graph-safe substrate

## Acceptance criteria

- Long-ctx 4k/c=4: ARLE TTFT ≤ **1400 ms**(-29% vs 1976 ms,缩进 SGLang 1.44×)
- Long-ctx 8k/c=4: TTFT improvement ≥ 25% vs ARLE 8k baseline(待补,M_world1 P0.2)
- High-conc 1k/256/c=64: out tok/s 不回归(decode-dominated,graph 收益小但不应负)
- All e2e + greedy_consistency tests pass
- Wins entry with per-shape Δ table + SGLang `bench-output/2026-05-07-sglang-longctx-4k-c4` 三方对照
- Phase 0 LOC ≤ 280(audit 260 + 20 buffer)

## Tasks

| # | Task | File | LOC | Owner | Trigger |
|---|---|---|---:|---|---|
| 0.1 | `PrefillGraphState` 镜像 decode `CudaGraphState` | `prefill_graph.rs` 新文件 | ~80 | Codex | 现在(M_pf-fuse 收尾后) |
| 0.2 | `PrefillBufferPool` bucket pre-alloc | `prefill.rs::from_pool` | ~50 | Codex | 0.1 后 |
| 0.3 | prepare/capture flow reorder | `prefill.rs:472-503` | ~30 | Codex | 0.2 后 |
| 0.4 | `gemm_graphsafe_cuda` route + Marlin guard | `linear.rs:945,672-695` | ~50 | Codex | 0.3 后 |
| 0.5 | `supports_cuda_graph_prefill` trait + CLI flag | `forward.rs` `main.rs` | ~25 | Codex | 0.4 后 |
| 0.6 | telemetry counters | `prefill.rs` `/v1/stats` | ~25 | Codex | 0.5 后 |
| 0.7 | Phase 0 bench `m_pgc-phase0-2048bucket` | `scripts/bench_guidellm.sh` | 0 | Claude | 0.6 后 |
| 0.8 | wins entry + license decision | `docs/experience/wins/...` | 0 | Claude | 0.7 done |
| 1.x | Phase 1 多桶 scheduler | (conditional) | ~110 | Codex | License fires |

## Cross-references

- 双侧 evidence:
  - [SGLang prefill stack survey](../research/2026-05-07-sglang-prefill-stack-survey.md)
  - [ARLE prefill GEMM callgraph](../research/2026-05-07-arle-prefill-gemm-callgraph.md)
- M_world1 baseline:[`12c4c86`](../experience/wins/2026-05-07-m_world1-p0-sglang-baseline.md)
- M_pf-gemm Phase 0 KILL(autotune 无收益):[`267fcfa`](../experience/wins/2026-05-07-m_pf-gemm-phase0-killed-cublas-heuristic-already-optimal.md)
- M_pf-fuse Phase 0 KILL(fusion 无收益):[`3e0ed5a`](../experience/wins/2026-05-07-m_pf-fuse-phase0-gateup-killed.md)
- decode graph pattern reference:`infer/src/model/cuda_graph.rs`
- M_world1 roadmap:[`docs/plans/M_world1-30-percent-lead-roadmap.md`](M_world1-30-percent-lead-roadmap.md)

## Rules(per memory `feedback_docs_priority_roi_evidence.md`)

- **Trace-driven hypothesis**:M_pgc 不是 intuition,SGLang survey + ARLE callgraph
  双侧 ground truth 锁死方向。在 M_pf-gemm/M_pf-fuse 两条 KILL 后,**这是当前唯一
  evidence-supported P0**。
- **Phase 0 single-bucket license-or-kill**:~260 LOC 在 ~6 小时实现窗口验证
  假设;Phase 1 多桶 gated on Phase 0 ≥ 25% improvement。
- **Reuse decode infra,不重复造轮子**:`CudaGraphState` 已 production,Phase 0
  通过新 `PrefillGraphState` 复用 90% 模式,只改 buffer pool + reorder。
