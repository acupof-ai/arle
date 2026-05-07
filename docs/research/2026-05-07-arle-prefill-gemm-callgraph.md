# 2026-05-07 · ARLE longctx 4k/c=4 prefill GEMM callgraph(对应 SGLang 调研)

> ⚠ **2026-05-07 EOD+9 update — Phase 2.5/2 demoted to P3 by SGLang evidence.**
> [SGLang survey](2026-05-07-sglang-prefill-stack-survey.md) §1 + Bottom Line
> 证明 SGLang BF16 dense GEMM 就是 `torch.nn.functional.linear → cuBLAS/cuBLASLt`
> **跟 ARLE 同一条路径**。SGLang 的 2.03× TTFT 优势不在 GEMM kernel,而在
> **piecewise CUDA graph capture for prefill**(2048-token bucket,整 layer loop graph,
> 不只 attention)。本 brief 的 "Phase 2.5 dispatch" 部分(§Phase 2.5 dispatch threshold)
> **不再是 P0 路径** — 见新 task M_pgc(Prefill CUDA graph capture)。
> 本 brief 仍作为"如果 graph capture 后 cuBLASLt 仍是 limiter"的 fallback 证据保留。
>
> 对应 [SGLang prefill stack survey](2026-05-07-sglang-prefill-stack-survey.md),
> 双侧 ground truth 锁定结论:**GEMM 不是主因**。

## Context

- Qwen3-4B BF16:`hidden=2560`, `intermediate=9728`, `num_q_heads=32`, `num_kv_heads=8`, `36 layers`
- longctx 4k/c=4 一次 prefill chunk = 2048 token × 4 reqs → **M = 8192 batched rows**
- H_LP3([`cae08b7`](../experience/wins/2026-05-07-h_lp3-diagnosed-cutlass-small-tile-gemm-bottleneck.md))
  实测:cutlass `Kernel2` (16×16×128 wmma bf16) = **606 launches × 2.3 ms = 1417 ms = 56.7% of TTFT window**

## Total GEMM count per forward step

**252 GEMMs** = 36 layer × 7 GEMM/layer。

per-layer 7 GEMM(`infer/src/model/qwen3/prefill.rs:forward_layer_batch_paged`):
- **QKV** 3 个:`q_proj` `k_proj` `v_proj`(L652-654;current Qwen3 path **separate**,非 fused)
- **attention 后**:`o_proj`(L718)
- **MLP** 3 个:`gate_proj` `up_proj` `down_proj`(via `forward_mlp_batch_into` L753-762)

H_LP3 reconciliation:**606 launches ÷ 36 layer ≈ 17/layer**;7 GEMM/layer 中只有 ~4 个落到 Kernel2 小 tile(其余 3 个走更大 tile)→ 17 ÷ 4 ≈ 4 chunks/req × ... 与 longctx 4k/c=4 实际 prefill 调用次数对得上。

## Per-GEMM (M, N, K) shape

| Op | M | N | K | 推断 cuBLAS tile |
|---|---:|---:|---:|---|
| q_proj | 8192 | **4096** | 2560 | 128×128 OK |
| k_proj | 8192 | **1024** | 2560 | ⚠ Kernel2 16×16 |
| v_proj | 8192 | **1024** | 2560 | ⚠ Kernel2 16×16 |
| o_proj | 8192 | **2560** | 4096 | ⚠ Kernel2 16×16 |
| gate_proj | 8192 | 9728 | 2560 | 256×128 OK |
| up_proj | 8192 | 9728 | 2560 | 256×128 OK |
| down_proj | 8192 | 2560 | **9728** | ⚠ Kernel2 16×16 |

**4 个 Kernel2 候选**(k/v/o/down):2.3 ms × 4 = 9.2 ms/layer × 36 = **331 ms 单 chunk**,× ~4 chunks(c=4 per req prefill 切 4 chunks)≈ 1324 ms,与 H_LP3 实测 1417 ms 误差 6.5% → 假设站得住。

## Dispatch path

所有 252 GEMM 都走 **cuBLASLt** 单一路径:

```
infer/src/ops/linear.rs:gemm_into()→gemm_cuda()
  → crates/cuda-kernels/src/ffi/gemm.rs FFI
  → crates/cuda-kernels/csrc/gemm/gemv.cu:gemm_cublaslt_impl() (L298-482)
  → cublasLtMatmulAlgoGetHeuristic() returns 8 candidates
  → 当前用 heuristic_results[0] (top-1, L371)
  → cached in GemmKey hash map
```

- **无 Marlin** 走法(Qwen3-4B 标准 BF16 权重,Marlin 仅 quantized 模型用)
- **无 TileLang prefill GEMM**(TileLang AOT 仅覆盖 attention,prefill GEMM 是 cuBLAS 独占)
- M_pf-gemm Phase 0 KILLED([`267fcfa`](../experience/wins/2026-05-07-m_pf-gemm-phase0-killed-cublas-heuristic-already-optimal.md))已证 8 candidates 中 top-1 ≈ 最优,问题不在 algo selection 而在 algo space 本身

## 为什么 cuBLAS 给小 N 选 Kernel2

H_LP3 diagnosis(L108-115):cuBLAS heuristic 在 varlen(4 reqs × 2048 chunk = 不对齐 M=8192)+ 小 N(≤2560)选小 16×16 tile 防止 shared memory blow-up。cuBLAS 偏向 conservative tile;vLLM/SGLang 用 pre-tuned CUTLASS templates 直接 hardcode 大 tile。

evidence escalation(M_pf-fuse Phase 0 KILL `3e0ed5a`):fused N=19456 反而比 2× separate N=9728 慢 +1.5% → cuBLAS algo space **non-monotonic in N**,大 N 也踩坑,不只小 N。

## Phase 2.5 dispatch threshold(具体提议)

**dispatch 规则**:

```cpp
// gemv.cu::gemm_cublaslt_impl 入口前
if (N <= 2560 && M >= 2048) {
    return tilelang_prefill_gemm(...);  // hand-tuned 256×128 tile
}
// 否则继续 existing cublasLt path
```

**影响 4 GEMM/layer**(k_proj/v_proj/o_proj/down_proj),× 36 layer = **144 dispatch hits/forward step**。

**LOC est**(细分):

| 文件 | 工作 | LOC |
|---|---|---:|
| `crates/cuda-kernels/tools/tilelang/batch_prefill_bf16_gemm.py` | 新 TileLang IR(镜像 batch_prefill_paged_hd128.py 模式)| ~120 |
| `crates/cuda-kernels/csrc/gemm/tilelang_prefill_dispatch.cu` | C++ wrapper + tile dispatch | ~50 |
| `crates/cuda-kernels/csrc/gemm/gemv.cu` | 前置 dispatch 判断 | ~20 |
| `crates/cuda-kernels/build.rs` | AOT 注册 | ~15 |
| `crates/cuda-kernels/src/ffi/gemm.rs` | Rust FFI | ~15 |
| `infer/src/ops/linear.rs` | 路由(若 dispatch 在 Rust 侧)| ~5 |
| **合计** | | **~225 LOC** |

比 plan 估的 ~150 多 50% — 因为 plan 假设 Phase 2.5 复用 attention TileLang 基础设施,实际新 GEMM IR 仍要 ~120 LOC scratch。Phase 2 (full port)估 250-300 LOC,Phase 2.5 reduce 不到 50%。

## Decision notes for #18 vs #19

- **Phase 2.5 优势**:scope 小(225 LOC),dispatch 后兜底 cublasLt,risk 低
- **Phase 2 (full port) 优势**:同样的 IR 一次写完所有 prefill GEMM,长期维护单一路径
- **关键 trigger**:codex #2 SGLang 调研出来如果显示 SGLang 是 **flashinfer pre-tuned 大 tile + 静态 dispatch**,Phase 2.5 是直接对标;如果是 **dynamic graph capture + per-shape JIT**,Phase 2 才能复用其架构

## Cross-refs

- H_LP3 trace 数据:[`cae08b7`](../experience/wins/2026-05-07-h_lp3-diagnosed-cutlass-small-tile-gemm-bottleneck.md)
- M_pf-gemm Phase 0 KILL:[`267fcfa`](../experience/wins/2026-05-07-m_pf-gemm-phase0-killed-cublas-heuristic-already-optimal.md)
- M_pf-fuse Phase 0 KILL:[`3e0ed5a`](../experience/wins/2026-05-07-m_pf-fuse-phase0-gateup-killed.md)
- M_world1 P0.1 SGLang baseline:[`12c4c86`](../experience/wins/2026-05-07-m_world1-p0-sglang-baseline.md)
- Phase 2.5 plan(待 codex #2 ground-truth 后细化):[`docs/plans/M_pf-gemm-cublaslt-autotune.md`](../../plans/M_pf-gemm-cublaslt-autotune.md) §Phase 2.5
