# 2026-05-07 · ARLE prefill 路径 graph-readiness audit

> 目的:为 [`M_pf-graph-prefill-capture`](../plans/M_pf-graph-prefill-capture.md)
> Phase 0 提供代码级 ground truth — 哪些点在当前 prefill 调用栈里是 graph-killer,
> 改造路径 + LOC est。
>
> 读完 [SGLang prefill stack survey](2026-05-07-sglang-prefill-stack-survey.md)
> + [ARLE prefill GEMM callgraph](2026-05-07-arle-prefill-gemm-callgraph.md)
> 后做的代码审计,扫 `infer/src/model/qwen3/`、`infer/src/ops/`、`crates/cuda-kernels/`。

## 已有基础设施(complete reuse,非新建)

- `infer/src/model/cuda_graph.rs` — `CudaGraphState`,capture-on-first-call,
  replay-thereafter,decode-path 已 production
- `infer/src/model/qwen3/forward.rs::supports_cuda_graph_decode` —
  feature gate `enable_cuda_graph && self.lora.is_none()`
- `infer/src/main.rs` — `--cuda-graph` / `--disable-cuda-graph` CLI 已 wire
- `infer/src/ops/linear.rs:608` — `gemm_graphsafe_cuda` no-workspace cuBLAS
  graph-safe GEMM,decode 已用
- `infer/src/scheduler/cuda/core/warmup.rs:11` — graph warmup 已有

**最重要的 simplification**:Phase 0 不需要从零起 graph 基础设施,只要 mirror
decode `CudaGraphState` pattern 扩到 prefill,key 从 `batch_size` 改为
`prefill_token_count`。

## 🔴 Hard blockers(必须改才能 graph capture)

| File:line | 问题 | 改造 | LOC |
|---|---|---|---:|
| `infer/src/model/qwen3/prefill.rs:42-80` | `PrefillBuffers::new()` 每次调用 `HiddenStates::zeros()` 10 个 device buffer(hidden_out, normed, q/k/v_batch, o_buf, gate_out, up_out, act_out, attn_output)+ 条件 `residual_f32` | bucket-级别 pre-alloc + pool reuse(`from_pool()` constructor)| 40-60 |
| `infer/src/model/qwen3/prefill.rs:485-503` | 3× `clone_htod`(page_indices, prefix_token_rows_upload, prefill_token_rows)在 capture 内 | 移到 `prepare_prefill_graph_inputs()` capture 之前 | 20-30 |
| `infer/src/ops/linear.rs:672,688,692-695` | Marlin W4 quant path:`alloc_zeros(m*k)` for `x_fp16` + `alloc_zeros(m*n)` for `y_fp16` + `alloc_zeros(ws_elems)` for workspace,sm_count + N runtime 决定 size | Phase 0 disable Marlin in graph path,fallback eager;Phase 2 单独做 graph-safe Marlin substrate | 30-50 |
| `infer/src/ops/linear.rs:945-955` | `gemm_cuda` cuBLASLt 默认 capture 不稳(handle/algo selection 时序问题)| 强制 prefill 走 `gemm_graphsafe_cuda:608`(no-workspace,decode 已用)| 20-30 |

## 🟡 Soft blockers(graph 能跑但要 explicit 处理)

| File:line | 问题 | 改造 | LOC |
|---|---|---|---:|
| `infer/src/model/qwen3/prefill.rs:472-475` | host-side `packed_tokens` Vec 构造 | 移到 capture 前 `prepare()` | 0(reorder) |
| `infer/src/ops/attention.rs:591-621` | per-seq prep_cuda loop launch + per-seq H2D | bucket-fixed 后 unrolled,seqs 元数据 capture 前 pre-compute | 10-20 |
| `infer/src/ops/attention.rs:548-569` | TileLang kernel dispatch by `(num_q_heads, num_kv_heads)` | capture 前选定 kernel pointer(once per layer)| 0 |
| `infer/src/model/qwen3/prefill.rs:438-448` | logits buffer on-demand `DeviceVec::zeros()` | Phase 0 init pre-alloc max-bucket logits buffer | 20 |
| `infer/src/model/qwen3/prefill.rs:135-138` | `completion_event.record(&ctx.stream)` in `set_pending()` | 已经在 capture 体外,structurally OK | 0 |

## 🟢 已 graph-safe(无需改)

- `infer/src/ops/attention.rs:574-664` — TileLang paged prefill execution
  (kernel launch 单 entry,无 alloc 无 event 创建)
- `infer/src/ops/embedding.rs::get_embeddings_batch` — single gather kernel
- RMSNorm / SiLU / residual add — 单 kernel launch 无 alloc
- `gemm_graphsafe_cuda:608` — no-workspace cuBLAS 已 graph-safe(decode 用)

## 总 LOC est

| 改造类别 | LOC |
|---|---:|
| Hard blocker 改造合计 | 110-170 |
| Soft blocker 改造合计 | 30-40 |
| 新增 `PrefillGraphState`(mirror `CudaGraphState`)| ~80 |
| 新增 `PrefillBufferPool` | ~50 |
| `supports_cuda_graph_prefill` trait + CLI | ~25 |
| Telemetry counters(graph_hit / miss / fallback_reason)| ~25 |
| **Phase 0 合计** | **~320-390**(plan kill 设 200,需 aggressive consolidation 复用 decode infra → 落到 ~210-260)|

## Token bucket 推荐

- **Phase 0**:单 2048 token bucket(对标 SGLang 16GB default 的 `chunked_prefill_size=2048`)
- **Phase 1**:42 桶按 SGLang table:`[2048, 1792, 1536, 1280, 1024, 960, 896, 832, 768, 704, 640, 576, 512, 480, 448, 416, 384, 352, 320, 288, 256, 240, 224, 208, 192, 176, 160, 144, 128, 112, 96, 80, 64, 48, 32, 28, 24, 20, 16, 12, 8, 4]`
- **Memory cost**:Phase 0 单桶 ~30-50 MB;Phase 1 42 桶 × ~30 MB = ~1.26 GB(KV pool 占 11.7 GB,剩 4.3 GB 给 graph buckets,紧但可行)
- **Page size**:TileLang HD128 锁 `page_size==16`,无 flexibility

## 风险点(top 3)

1. **cuBLASLt graph capture contract**:`gemm_graphsafe_cuda` 路径在 decode 已 production,
   但 prefill shape (大 M = 8192) 是否仍 graph-safe 需要 Phase 0 e2e + greedy_consistency 双 gate
2. **Marlin W4 在 graph 路径直接 disable**:Phase 0 限 Qwen3-4B BF16 一种 path,
   quantized 模型(Qwen3 GPTQ etc.)走 eager fallback。task `supports_cuda_graph_prefill`
   trait 加 `weights.is_quantized()` guard
3. **prefill async pending 与 graph replay 协调**:当前 `PendingPagedPrefill`
   依赖 event query,graph replay 必须在 event record 之前完整执行 — 现 reorder 后
   graph body 在 set_pending 之前,structurally 兼容但需要 e2e 验证

## Phase 0 LOC consolidation 路径(达到 plan kill 200 的方法)

| 措施 | 节省 LOC |
|---|---:|
| `PrefillGraphState` 直接复用 `CudaGraphState` 模板,只改 key type | 30 |
| `PrefillBufferPool` 复用 decode 的 buffer pool 模式 | 20 |
| Telemetry 复用现有 `/v1/stats` schema | 15 |
| 不引新 trait,只在 `supports_cuda_graph_decode` 加 prefill flag | 10 |
| 总 saved | **75** |

→ Phase 0 现实 LOC 落到 **~245-315**,还需 plan owner 在实现时进一步压缩到 ≤ 200。
若 Phase 0 实测 LOC > 250,plan kill criteria 触发 → reframe scope。

## Cross-references

- 双侧 evidence:
  - [SGLang prefill stack survey](2026-05-07-sglang-prefill-stack-survey.md)(`7ef707d`)
  - [ARLE prefill GEMM callgraph](2026-05-07-arle-prefill-gemm-callgraph.md)(`7ef707d`)
- M_world1 P0.1+P0.2 baseline:[`12c4c86`](../experience/wins/2026-05-07-m_world1-p0-sglang-baseline.md) +
  [`9ee4644`](../experience/wins/2026-05-07-m_world1-p0-sglang-baseline-extended.md) +
  [`4ae3b7b`](../experience/wins/2026-05-07-m_world1-p0-sglang-baseline-extended.md)
- Plan owner:[`docs/plans/M_pf-graph-prefill-capture.md`](../plans/M_pf-graph-prefill-capture.md)(`939008f`)
- Decode graph pattern:`infer/src/model/cuda_graph.rs` `infer/src/model/qwen3/batch_decode.rs:169,1691`
- Graph-safe BF16 GEMM:`infer/src/ops/linear.rs:608`
