# ARLE Backend Unification Roadmap

> Updated 2026-05-07. Mission: 力争世界第一 inference runtime (vs vLLM / TGI / llama.cpp / SGLang) by 收敛 CUDA 与 Metal 两条分叉到一个共享 scheduler / ops / model / kv-tier / contract 平面。

## 0. Why this roadmap

ARLE 现状是 **两个独立产品挤在一个 crate 里**:

- **CUDA 路径**: `infer/src/scheduler/cuda/*` (≈6900 行) + `infer/src/ops/*` (≈4500 行 全部 cudarc) + `infer/src/model/qwen3/*` + `infer/src/model/qwen35/*` + `cuda_kernels::PagedKVPool`/`TokenKVPool` + `kv_tier` (T0/T1/T2/T3)。
- **Metal 路径**: `infer/src/backend/metal/*` (≈19,000 行,全部 self-contained) + `infer/src/backend/metal/scheduler.rs` `MetalScheduler` + `infer/src/backend/metal/runtime.rs` `run_metal_scheduler_runtime` + `MetalKVPool` + `mlx-sys` C++ 桥。

两边只在 **HTTP/CLI 入口的 trait** (`InferenceEngine` / `RequestHandle` in `infer/src/server_engine/types.rs` + `infer/src/request_handle.rs`) 收敛。其他 stack 各自一份。

Result:
- 任何 scheduler/policy/kv-tier 提升要写两次,实际只写一次 — Metal 永久落后。
- 性能数字不可比 (CUDA 跑 W1/c4 1.609× SGLang;Metal 仍是单请求 305 tok/s 单点)。
- 对标 vLLM / TGI 的"continuous batching + paged KV + prefix cache + tiered KV"四件套,我们 CUDA 边接近,Metal 边只到 continuous batching。

## 1. 现状分叉盘点 (基于 grep + Read)

### 1.1 `infer/src/backend/`

| 模块 | CUDA | Metal | 分叉根因 |
|---|---|---|---|
| `backend/cuda.rs` (12 行) | 仅 re-export `cuda_kernels::*` + `cuda/bootstrap.rs` | — | CUDA 真正的"backend impl"是 scheduler;`InferenceBackend` 这个串行 trait 在 CUDA 上是 vestigial |
| `backend/metal.rs` (922 行) | — | `MetalBackend` 实现 `InferenceBackend` + `StreamingInferenceBackend`,自己做 weight loading / generate loop | Metal 一开始走的是单请求 backend,后来才在 `runtime.rs` 上面加了 scheduler |
| `backend/cpu.rs` | feature `cpu`,基本是 stub | — | dev only,可忽略 |

**分叉点**: `InferenceBackend` trait 在 `backend.rs` 里只有 `load`/`generate`/`tokenize`,适合单请求,不适合 multi-tenant — CUDA 整条 scheduler 路径根本不实现这个 trait,而是通过 `RequestHandle` 接到 `RequestHandleInferenceEngine`;Metal 两条都用 (`BackendInferenceEngine` + `MetalSchedulerHandle`)。

### 1.2 `infer/src/scheduler/`

| 文件 | 状态 |
|---|---|
| `scheduler/batch.rs` (448 行) `BatchScheduler` | **声称 backend-agnostic**,但实际只有 `scheduler/tests.rs` 用,生产路径不接 |
| `scheduler/cuda/core.rs` (1742 行) `Scheduler<M: ModelForward>` | CUDA 真正的生产 scheduler,泛型在 `ModelForward` 上 |
| `scheduler/cuda/{prefill,decode,execution,spec_path,budget,policy,request,runtime}.rs` (≈3800 行) | 全部 CUDA-only,直接调 `crate::ops::*` + `cuda_kernels::PagedKVPool` |
| `scheduler/cuda/runtime/{scheduler_loop,admission,fetch,helpers}.rs` | scheduler thread 拆分 |
| `backend/metal/scheduler.rs` (1015 行) `MetalScheduler` + `MetalSchedulerConfig` + `MetalScheduleStep` | Metal 自己的 CPU-only 决策层 |
| `backend/metal/runtime.rs` (2918 行) `run_metal_scheduler_runtime` | Metal 自己的 hot path,decode-first 连续批处理,packed varlen decode,执行 `MetalScheduleStep` |
| `backend/metal/plan.rs` (126 行) `MetalLogicalDecodeRow` / `MetalScheduleStep` | Metal 专属决策载体 |

**分叉根因**: 两边 scheduler 的 KV state 形状根本不一样 — CUDA 用 page index (`u32`) 进 `PagedKVPool`;Metal 用 token slot 进 `MetalKVPool` + 每行 left-pad 后的 packed cache。`BatchScheduler` 没成型成统一 IR 是因为它出生太早,没考虑 sparse-KV / mixed-batch / spec-decode。

### 1.3 `infer/src/ops/`

**全部 8 个文件 (`attention/elementwise/embedding/kv_ops/linear/norm/recurrent/sampling`) 都是 CUDA-only**,顶着 `use cudarc::driver::*` + `use cuda_kernels::ffi`。Metal 完全没接入这一层 —— Metal 等价物在 `backend/metal/ops.rs` (109 行,只有 `linear`/`extend_kv_cache`/`clear_metal_cache`) 和 `backend/metal/mlx.rs` (1592 行,`MlxArray` + 所有 RMSNorm/RoPE/attention 在那)。

| Op | CUDA 位置 | Metal 等价位置 | 对齐难度 |
|---|---|---|---|
| GEMM / GEMV / linear | `ops/linear.rs` (944 行) + `crates/cuda-kernels/csrc/gemm/{gemv,marlin_kernel,quantized_gemv}.cu` | `metal/ops.rs::linear` (调 `mlx::matmul`/`quantized_matmul`/`gguf_quantized_matmul`) | 中:量化路径多版本 (Marlin / TurboQuant / MLX 4bit / GGUF),需要统一 weight format enum |
| Attention prefill (HD128 + HD256, paged + nonpaged) | `ops/attention.rs` (1313 行) + `crates/cuda-kernels/csrc/attention/{prefill_attention,prefill_attention_hd256,prefill_attention_paged_prep,fused_attention,nonpaged_prefill_attention}.cu` | `metal/forward.rs::build_forward_graph` (走 mlx scaled_dot_product_attention) + `metal/qwen35.rs` 走 mlx-sys C++ step model | 高:CUDA 两个 head dim + 4 个 path,Metal 只有一个 |
| Attention decode (BF16/INT8/FP8/MLA/quantized) | `ops/attention.rs` + `csrc/attention/{decode_attention_quantized,decode_attention_turboquant,decode_attention_varlen_fp8,mla_decode,decode_prep_paged*}.cu` | `metal/forward.rs` packed varlen decode (left-pad + additive mask + per-row RoPE offsets) | 高 |
| KV cache append / scatter / paged metadata | `ops/kv_ops.rs` + `csrc/kv/{paged_kv_append,paged_kv_metadata,scatter_kv,kv_cache_to_paged,kv_quant}.cu` | `metal/ops.rs::extend_kv_cache` + `MetalKVPool::alloc_tokens` | 中 |
| RMSNorm (+ fused add) | `ops/norm.rs` + `csrc/misc/norm.cu` | `metal/mlx.rs::rms_norm` | 低 — 接口干净 |
| Sampling (greedy/argmax/top-k) | `ops/sampling.rs` + `csrc/misc/sampling.cu` | `metal/sampling.rs::gpu_sample_token` | 中 |
| Recurrent / GDR (Qwen3.5 linear-attn) | `ops/recurrent.rs` + `csrc/misc/{conv1d*,gated_delta_rule,gdr_*}.cu` | `metal/qwen35.rs` 全在 mlx-sys C++ 内 | 高:Metal 等价整个委托给 C++ step model,不是 op-level 拼装 |
| Quant (TurboQuant / dtype convert) | `ops/linear.rs` + `csrc/quant/{turboquant,turboquant_fast,dtype_convert}.cu` | mlx 量化矩阵乘 + `gguf_quantized_matmul` | 中 |
| Elementwise (silu_mul, add) | `ops/elementwise.rs` + `csrc/misc/{elementwise_basic,fused_mlp,split_qkv}.cu` | mlx `*_eval` graph nodes | 低 |
| Embedding | `ops/embedding.rs` | `metal/forward.rs::take_axis` 内联 | 低 |

**分叉根因**: 没有 `Op` trait;`crate::ops::*` 是一组自由函数,签名带 `&DeviceVec`/`CudaSlice`/`PagedKVPool` 等 CUDA-only 类型。Metal 等价 op 没法被 model 代码统一调用。

### 1.4 `infer/src/model/qwen3*/`

| 文件 | CUDA-only? | 备注 |
|---|---|---|
| `model/qwen3/forward.rs` | 是 (`use cuda_kernels::*` + `crate::ops::*`) | 实现 `ModelForward` for Qwen3,decode/prefill/batched/spec-verify 全绑 CUDA |
| `model/qwen3/{prefill,decode,batch_decode,decode_buffers,weights}.rs` | 是 | 全部依赖 `cudarc` / `cuda_kernels::PagedKVPool` |
| `model/qwen3/lora.rs` | 是 (`#[cfg(feature = "cuda")]`) | LoRA 在 Metal 不存在 |
| `model/qwen35/forward.rs` + 所有同级 `qwen35/*.rs` | 是 | Qwen3.5 hybrid (linear + full attn) 也只在 CUDA |
| `backend/metal/forward.rs` (182 行) | Metal-only | Qwen3 forward graph builder,绝大部分逻辑写在这里 |
| `backend/metal/qwen35.rs` (3871 行) | Metal-only | 委托给 `crates/mlx-sys/src/mlx_qwen35_model.cpp` C++ step model |
| `backend/metal/request_state.rs` (4601 行) | Metal-only | 整个 packed varlen decode + KV state 管理 |

**分叉根因**: `crate::model::ModelForward` trait 签名硬绑 `cudarc::driver::CudaSlice` 和 `cuda_kernels::PagedKVPool`,Metal 永远无法实现。

### 1.5 `infer/src/kv_tier/`

`kv_tier` 整个模块在 `Cargo.toml` 上 **不带 cuda feature 门**,但内部有 `#[cfg(feature = "cuda")]` 守住 PCIe 注册:

| 文件 | 备注 |
|---|---|
| `kv_tier/host_pool.rs` | T1 host pinned;有 `#[cfg(feature = "cuda")]` 包 `cuMemHostRegister_v2` (Metal 上 unified memory 跳过整个 T1 — 见 `kv_tier/AGENTS.md` 注释) |
| `kv_tier/transport/disk.rs` | T2 disk store,**两端都能用,但只在 CUDA scheduler 接** (`scheduler/cuda/policy.rs::TieredKvPolicy`) |
| `kv_tier/transport/local_cuda.rs` | CUDA-only |
| `kv_tier/transport/{nixl,shared_fs}.rs` | T3 远程,后端无关但 Metal scheduler 没接 |
| `kv_tier/coordinator/*.rs` | 协调器自身 backend-agnostic,但生产路径只挂在 CUDA scheduler |

**分叉根因**: `kv_tier::policy::{PrefetchPolicy,WritePolicy}` 是后端无关的,但 `cuda/policy.rs::TieredKvPolicy` 是 wiring;Metal 没有等价 wiring,所以 T2 disk + T3 remote 在 Metal 路径上**事实未启用**。

### 1.6 `infer/src/server_engine/`

唯一干净的"已收敛"层:

- `server_engine/types.rs::InferenceEngine` trait — both backends 实现
- `server_engine/loaded.rs::LoadedInferenceEngine` — 共用包装
- `server_engine/pool.rs::EnginePool` — 共用,multi-model 路由
- `server_engine/request_handle_engine.rs::RequestHandleInferenceEngine<H>` — 泛型在 `RequestHandle`,CUDA 和 Metal 各传自己的 handle

这一层是统一的 **唯一锚点**。所有下面的统一动作都应保持这个 trait 签名稳定。

## 2. Milestones (5–7 个,顺序按"小→大、收敛性强→弱")

每个 milestone:
- ≤2 周
- 自包含,可独立验收
- 有 acceptance + 风险 + 回退预案

### M1 — Unified Backend Telemetry & Engine Lifecycle Trait (Week 1)

**Why first**: 最痛点最少、最容易拿 win。今天 CUDA 侧 metrics 走 `scheduler/metrics.rs`,Metal 侧走 `backend/metal/runtime.rs::maybe_refresh_runtime_metrics` 自己一套,bench wins doc 里数字格式都不一致。先把 telemetry 收敛,后面任何 perf 工作都直接受益。

**范围**:
- 在 `infer/src/server_engine/types.rs` 加 `EngineTelemetry` snapshot struct (TTFT, ITL, queue depth, batch occupancy, KV-tier hit rates)。
- 改 `InferenceEngine` trait 增 `fn telemetry(&self) -> EngineTelemetry`。
- CUDA: 把 `scheduler/metrics.rs` 的 `SchedulerMetrics::snapshot()` 投影到新 struct。
- Metal: 把 `runtime.rs::maybe_refresh_runtime_metrics` 投影到同一 struct。
- 改 `infer/src/http_server/` 的 `/v1/stats` endpoint 用新 trait。
- 改 `scripts/bench_guidellm.sh` 拉到的 `service_stats_trace.jsonl` 直接 dump 这个 struct。

**涉及文件**:
- `infer/src/server_engine/types.rs`
- `infer/src/server_engine/loaded.rs`
- `infer/src/scheduler/metrics.rs`
- `infer/src/scheduler/cuda/runtime/scheduler_loop.rs`
- `infer/src/backend/metal/runtime.rs`
- `infer/src/http_server/` (找出 stats handler)
- `scripts/bench_guidellm.sh`

**Acceptance**:
- (test) `cargo test -p infer --features cuda telemetry::` 和 `--features metal,no-cuda telemetry::` 都过。
- (bench) `bench_guidellm.sh cuda-h100 --quick` + `bench_guidellm.sh metal-m3max --quick` 输出的 `service_stats_trace_summary.md` 字段完全对齐。
- 老 `/v1/stats` 字段保持向后兼容 (新字段只追加)。

**风险 / 回退**:
- 风险:Metal `maybe_refresh_runtime_metrics` 字段缺失 → 用 `Option<f64>` + `None` 填充。
- 回退:rev `server_engine/types.rs`,trait 加默认实现返回 empty snapshot,CUDA-only override。

---

### M2 — KV-tier Policy Adapter for Metal (Week 2)

**Why second**: T0/T1/T2 policy enums 已经 backend-agnostic,只差 wiring。这一步把 Metal 接进 `kv_tier::policy`,后续 prefix-cache / disk persistence 在两端复用。

**范围**:
- 在 `infer/src/kv_tier.rs` 提出 `KvTierAdapter` trait — 抽象 `paged_pool_pressure() -> f64` + `submit_demote(block) -> Result<()>` + `submit_promote(...)`。
- CUDA `scheduler/cuda/policy.rs::TieredKvPolicy` 改成实现该 trait。
- Metal `backend/metal/runtime.rs` 也实现该 trait,wiring `MetalKVPool` 压力 + `kv_tier/transport/disk.rs`。
- Metal 路径首次启用 T2 disk persistence (`MetalKvDiskOptions` 已经在 `metal.rs`,但没接到 coordinator)。

**涉及文件**:
- `infer/src/kv_tier.rs`
- `infer/src/kv_tier/policy.rs`
- `infer/src/kv_tier/coordinator.rs` + `infer/src/kv_tier/coordinator/*.rs`
- `infer/src/scheduler/cuda/policy.rs`
- `infer/src/backend/metal/runtime.rs`
- `infer/src/backend/metal/kv_pool.rs`
- `infer/src/backend/metal/prefix_cache.rs`

**Acceptance**:
- (test) Metal disk persistence smoke test,验证 KV block 落盘 → 重启 → readmission。
- (bench) `bench_guidellm.sh metal-m3max --workload longctx-32k` 在 disk persist 开启时,跨次运行 prefix-cache hit rate ≥ 50%。
- (test) 现有 `kv_tier/coordinator/tests.rs` 全过。

**风险 / 回退**:
- 风险:Metal 的 unified memory 让 T1 概念不适用,会在 trait 上漏抽象 → trait 签名加 `tier: Tier` 参数,Metal 直接拒掉 T1 请求。
- 回退:把 Metal 实现下沉成 noop,keep adapter trait,留 hook 给将来。

---

### M3 — Unify Scheduler Decision Layer (Logical IR) (Week 3-4)

**Why third**: 收益最大也最容易卡壳的是 scheduler 决策层。关键洞察:`scheduler/batch.rs::BatchScheduler` 已经写了"backend-agnostic CPU 决策层"的雏形,但被两边架空了 (CUDA 用 `scheduler/cuda/core.rs`,Metal 用 `MetalScheduler`)。把两边决策层 **同时** 改成调 `BatchScheduler` 出 `ScheduleDecision`,然后各自再 lower 到自家执行层。

**范围**:
- 扩展 `scheduler/batch.rs` 的 `ScheduleDecision` 让其能描述 Metal 现有的 packed varlen decode (左 padding + per-row RoPE offset)。
- 让 `scheduler/cuda/core.rs::Scheduler::step()` 内部委托给 `BatchScheduler::schedule_step()`,然后只做 lowering (页表分配 + GPU launch)。
- 让 `MetalScheduler::step()` 也委托。
- 把 `scheduler/policy.rs` 里 `DecodeAwareChunking` 升级成两端共享 (现在只 batch.rs 用)。
- **不动**:CUDA 的 prefix-cache / paged-KV / spec-decode 高级路径仍可绕过 (走 `scheduler/cuda/prefill.rs` 的特殊 path),只走 happy path 用 `BatchScheduler`。

**涉及文件**:
- `infer/src/scheduler/batch.rs`
- `infer/src/scheduler/policy.rs`
- `infer/src/scheduler/cuda/core.rs` (主要改 `step()` / `step_new()` 入口)
- `infer/src/scheduler/cuda/execution.rs::plan_step`
- `infer/src/backend/metal/scheduler.rs`
- `infer/src/backend/metal/runtime.rs::run_metal_scheduler_runtime`
- `infer/src/backend/metal/plan.rs` (改成用 `BatchScheduler::ScheduleDecision`)
- `infer/src/block_manager.rs` (BlockManager 已经 backend-agnostic,Metal 也接)

**Acceptance**:
- (test) `scheduler/tests.rs` + `infer/tests/e2e*.rs` (CUDA) + Metal e2e 全过。
- (bench) `bench_guidellm.sh cuda-h100` 不回退 ≥ 95% baseline (回归 ≤5%)。Metal 同。
- (lines) `scheduler/cuda/core.rs` + `backend/metal/scheduler.rs` 总行数下降 ≥ 800 行。

**风险 / 回退**:
- 风险:CUDA spec-decode + sparse-KV draft view 是 `BatchScheduler` 没考虑的 → 让 CUDA 在这种 case 时仍走旧 path,只在 happy path 委托。
- 风险:回归大 → 在 Cargo feature `unified_scheduler` 后面 gate,默认关。
- 回退:rev `core.rs::step()` change,保留 `BatchScheduler` 的扩展。

---

### M4 — Unified `Op` Trait + Metal `crate::ops::*` Implementor (Week 5-6)

**Why fourth**: ops 层是工作量大头但收敛性强。一旦 `crate::ops::rms_norm` / `linear` / `attention_decode` 是 backend-agnostic trait,model 层就能跨后端共享。

**范围**:
- 在 `infer/src/ops.rs` 引入 `OpsBackend` trait — `fn rms_norm(&self, x, w, eps) -> Tensor`,等等。每个 op 一个方法。
- 引入 `Tensor` 抽象 (enum DeviceTensor: `Cuda(DeviceVec)` | `Metal(MlxArray)`)。
- CUDA `OpsBackend` impl 复用现有 `ops/*.rs` 里的自由函数。
- Metal `OpsBackend` impl 调 `mlx::*` (在 `backend/metal/mlx.rs`)。
- 优先级:先 norm + linear + sampling + elementwise (5 个 op 占 model forward 调用量 70%)。attention prefill/decode 留 M5。

**涉及文件**:
- `infer/src/ops.rs` (整个改成 trait + module)
- `infer/src/ops/{norm,linear,elementwise,sampling,embedding}.rs` (新增 trait impl block)
- `infer/src/backend/metal/ops.rs` (扩成全套 impl)
- `infer/src/backend/metal/mlx.rs` (可能需要新加几个 wrapper)
- `infer/src/model/qwen3/{forward,prefill,decode}.rs` (改 callsite 走 trait)

**Acceptance**:
- (test) `ops/tests.rs` 现有 1629 行测试全过 (CUDA);新增等价 Metal 测试集。
- (correctness) 跑 `infer/tests/greedy_consistency.rs` Metal 版,greedy 输出 vs Qwen3-0.6B reference output bit-identical。
- (lines) `backend/metal/forward.rs` + `backend/metal/qwen35.rs` 重复 op 实现行数下降。

**风险 / 回退**:
- 风险:`Tensor` enum 引入 dispatch overhead → benchmark `bench_throughput.py` 单请求,要求 ≤ 2% 退化;否则用 monomorphization (`OpsBackend<T>` 泛型)。
- 风险:Metal 的 lazy graph 与 CUDA 的 eager 语义不同 → trait 强制 eager (mlx `async_eval`/`eval` 调用点上移到 trait 实现内部)。
- 回退:每个 op 单独迁;失败的 op 撤回到 free fn。

---

### M5 — Unified `ModelForward` Trait + Qwen3 Cross-Backend Path (Week 7-8)

**Why fifth**: 在 M4 ops trait 落地后,这一步把 Qwen3 model 写一份,跑两端。

**范围**:
- 改 `infer/src/model.rs::ModelForward` trait,把 `cudarc` / `cuda_kernels::PagedKVPool` 类型替换成 M4 的 `Tensor` + 后端无关 `KVPoolHandle` trait。
- `KVPoolHandle` 两个实现:`CudaPagedKVPool`、`MetalKVPool`。
- 把 `model/qwen3/forward.rs` 改成 backend-agnostic — 通过 `OpsBackend` 调 op,通过 `KVPoolHandle` 操作 KV。
- 删除 `backend/metal/forward.rs::build_forward_graph` (Qwen3 那段) — 改成共用 `model::qwen3`。
- **Qwen3.5 暂不动**(linear-attn 在 Metal 端是整个 C++ step model,M5 不动它,留 M6)。

**涉及文件**:
- `infer/src/model.rs` (`ModelForward` trait 重大改动)
- `infer/src/model/qwen3/forward.rs` + `decode.rs` + `prefill.rs` (改 generic over backend)
- `infer/src/model/qwen3/weights.rs` (weight loading 也需要 backend-agnostic)
- `infer/src/backend/metal/forward.rs` (大半删除)
- `infer/src/backend/metal/runtime.rs` (改成 dispatch 到 `model::qwen3`)
- `infer/src/scheduler/cuda/core.rs` 模板参数收紧

**Acceptance**:
- (correctness) Metal Qwen3-0.6B-bf16 greedy output bit-identical 与 CUDA Qwen3-0.6B-bf16 greedy output (同一 prompt)。
- (test) Metal `bench_4bit` + `bench_bf16` `#[ignore]` test 全过且 tok/s 不回退 > 3%。
- (lines) `backend/metal/forward.rs` 行数 ≤ 60 (只剩 Metal-specific glue)。

**风险 / 回退**:
- 风险:numerical drift (bf16 在 MLX vs cuBLAS 计算路径不同)。允许一定 epsilon (greedy_consistency 已经处理过 CUDA 内部);Metal 单独 baseline。
- 风险:Metal eager graph 需要每 op 后 `eval`,vs CUDA stream 同步,影响 perf → 在 Metal `OpsBackend` 内部用 fused graph 段 (mlx `compile`)。
- 回退:保留 `backend/metal/forward.rs` 旧实现作为 fallback,feature gate 切换。

---

### M6 — Cross-Backend Bench Matrix + World-#1 Gauntlet (Week 9-10)

**Why sixth**: M1–M5 完成后,arle 已经是真正"unified backend" runtime。这个 milestone 把对标做完,不动代码,只跑 bench + 立 wins doc。

**范围**:
- 标准化 `scripts/bench_guidellm.sh` 输出到 `docs/experience/wins/<date>-bench-{backend}-{workload}.md`。
- 新增 4 个 workload preset:`prefill-heavy` (4096-in/16-out)、`decode-heavy` (128-in/2048-out)、`longctx-32k` (已有)、`high-conc` (c=64)。
- 跑 ARLE-CUDA-H100 vs vLLM (latest) vs SGLang vs TRT-LLM (Qwen3-4B + Qwen3-8B)。
- 跑 ARLE-Metal-M3Max vs llama.cpp Metal vs MLX-LM (Qwen3-0.6B / 4B)。
- 立 `docs/experience/wins/2026-XX-world-rank-snapshot.md`,正面对比表格。

**涉及文件**:
- `scripts/bench_guidellm.sh` (扩 preset)
- `scripts/bench_compare.py` (扩 baseline runner)
- `docs/experience/wins/<date>-*.md` (新增)

**Acceptance**:
- (data) 8 个 workload × 5 baseline 矩阵填满,误差棒 ≤ 5%。
- (verdict) ARLE 在至少 4/8 workloads 第一名;两个后端各贡献 ≥ 1 第一。
- (doc) wins 文档有可重现 commit hash + 命令。

**风险 / 回退**:
- 风险:ARLE 在某些场景显著落后 → 那些 workload 单独立 follow-up plan,不挡 M6 完成。
- 回退:此 milestone 不修改代码,无回退需求。

---

### (可选) M7 — Qwen3.5 Hybrid Cross-Backend Path (Week 11-12)

如果 M5 完成顺利且 Qwen3.5 hybrid (linear-attn) ops 能在 mlx-sys 用 op-level 拼装 (而非整个 C++ step model),把 Qwen3.5 也 unify。否则保留两端各自实现。

## 3. 世界第一对标策略

### 3.1 对标 baseline + 测试场景 → 涉及的 ARLE milestone

| Workload (`bench_guidellm.sh`) | 对手优势点 | 我们目前差距 (假设) | 由哪个 milestone 收割 |
|---|---|---|---|
| **prefill-heavy** (4096-in / 16-out, c=1) | vLLM/TGI cuBLASLt + chunked prefill;llama.cpp Metal Flash-Attn-Metal | TTFT 假设落后 vLLM ~10-15% on H100,Metal 落后 mlx-lm ~5% | M3 (统一 chunking policy) + M4 (统一 op,Metal 拿到 batched prefill) |
| **decode-heavy** (128-in / 2048-out, c=1) | TRT-LLM 持续 decode + CUDA Graph;mlx-lm packed decode | CUDA 假设接近 vLLM,Metal 已经领先 mlx-lm | 维持现状 + M5 (Metal Qwen3 path 跟上 CUDA 的 sample_batch_greedy) |
| **longctx-32k** (32k-in / 256-out, c=4) | SGLang HiCache + Mooncake disagg | 我们 P0 目标 1.609× SGLang,vs vLLM/TRT-LLM panel 仍缺 | M2 (Metal 接上 disk tier,长上下文 prefix 重用) + M6 (panel 数据) |
| **high-conc** (1024-in / 256-out, c=64) | vLLM continuous batching + paged-KV;TRT-LLM in-flight batching | CUDA 假设打平到领先 5%,Metal 现在不能跑 (单请求 backend → M3 后才能跑 c>1 well) | M3 (统一 scheduler — Metal 拿到 mixed-batch happy path) |
| **prefix-reuse** (8 同 prefix,c=8) | SGLang RadixAttention | CUDA 已有 radix prefix cache,Metal `prefix_cache.rs` 但未 wired 到运行时持续 reuse | M2 (Metal disk-backed prefix) + M3 (统一调度面把 prefix-hit 类别也走 BatchScheduler) |

### 3.2 哪些 milestone 直接带 bench 收益

| Milestone | 预计 bench 移动 (假设) |
|---|---|
| M1 | 不动数字,但能让 bench 数据可比 |
| M2 | Metal longctx prefix-cache hit rate +30~50% → 长 context 多次访问场景 TTFT -30% |
| M3 | Metal 拿到真 continuous batching → c=4 throughput +60~100% (vs 现在单请求 305 tok/s 串行) |
| M4 | CUDA 维持;Metal 减少 self-coded op duplication 小幅 +2~5% |
| M5 | Metal 拿到 CUDA 的 batched greedy sampling + decode buffer 复用 → decode-heavy +10~20% |
| M6 | 不动数字,只确认 leadership |

## 4. Out of Scope (这次明确不做)

- **单后端纯优化**:CUDA 的 split-KV varlen FP8 / TileLang 新 tile / TRT-LLM-style in-flight batching 等单后端 perf,走 P0 longctx leadership 那条线。
- **重写 `cudarc` 或 `mlx-sys`**:依赖层稳定,不动。
- **新硬件后端**:ROCm / TPU / Vulkan(虽然 `docs/plans/2026-05-05-multi-backend-tilelang-rocm-vulkan.md` 已有研究,但本路线图只统一现存两端)。
- **Train / RL / Agent runtime 收敛**:这些走 P4 计划,不在本路线图。
- **DeepSeek V4 / 新模型架构**:走 P0'' 单独项目。

## 5. 完成的判定标准 (Definition of Done)

- M1–M5 全部 acceptance 通过。
- `scheduler/cuda/core.rs` + `backend/metal/scheduler.rs` 不再各自实现 `ScheduleDecision`。
- Metal Qwen3 forward 走 `model::qwen3::*`,Metal 自有 forward 删除 ≥ 80%。
- `infer/src/ops/*` 提供 backend-agnostic trait,Metal 实现完整 5 个核心 op。
- `bench_guidellm.sh` 对两个后端输出可对比矩阵。
- M6 wins doc 在仓库内,世界排名快照公开。

## 6. References

- `infer/src/scheduler/AGENTS.md`
- `infer/src/backend/metal/AGENTS.md`
- `infer/src/model/AGENTS.md`
- `infer/src/kv_tier/AGENTS.md`
- `docs/architecture.md`
- `docs/projects/2026-04-30-longctx-32k-128k-leadership.md` (P0 long-context)
- `docs/plans/guidellm-integration.md` (bench protocol)
- `ROADMAP.md`
