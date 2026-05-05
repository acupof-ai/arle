# 2026-04-14 · CUDA kernel 六要素审计与第一波优化

## Scope

审计对象是当前生产路径下的 CUDA kernel crate：

- `crates/cuda-kernels/csrc/` 下 `33` 个 `.cu` 文件
- `crates/cuda-kernels/tools/triton/` 下 `13` 个 Triton AOT kernel

方法不是“逐个猜哪段会快”，而是按六要素过一遍，再结合现有 profiling / benchmark 文档给热路径排优先级：

1. Global Memory Coalescing
2. Shared Memory Bank Conflicts
3. Occupancy
4. Tiling & Data Reuse
5. Warp Divergence
6. Launch Config & Tail Effect

相关上下文来自：

- `docs/resources/profiling-guide.md`
- `docs/experience/reviews/2026-04-02-cuda-link-optimization-gaps.md`

## Heat Map

### P0 · decode hot path

- `csrc/gemm/gemv.cu`
- `csrc/gemm/quantized_gemv.cu`
- `csrc/misc/norm.cu`
- `csrc/attention/decode_prep_paged.cu`
- `csrc/attention/decode_prep_paged_hd256.cu`
- `csrc/misc/gated_delta_rule.cu`
- `csrc/misc/gdr_decode_batch.cu`
- `csrc/misc/conv1d_decode_batch.cu`
- `csrc/attention/flashinfer_decode.cu`
- `csrc/attention/flashinfer_decode_hd256.cu`
- `csrc/attention/flashinfer_tc_decode.cu`

这些文件决定 decode TPOT / ITL。现有 profiling 一致指向这里，而不是 scheduler。

### P1 · prefill / mixed hot path

- `csrc/attention/prefill_attention.cu`
- `csrc/attention/prefill_attention_hd256.cu`
- `csrc/attention/flashinfer_prefill.cu`
- `csrc/attention/fused_attention.cu`
- `csrc/misc/fused_mlp.cu`
- `csrc/misc/conv1d.cu`
- `tools/triton/flash_attention_prefill_hd256_kernel.py` *(historical — deleted in Phase 0, commit 38d4d773; superseded by FlashInfer + TileLang HD256 paths)*
- `tools/triton/gated_delta_rule_chunkwise_kernels.py`

这些 kernel 对 TTFT 和长 prompt 更敏感，但不是当前 decode 吞吐的第一瓶颈。

### P2 · KV / quant / utility / cold path

- `csrc/kv/kv_cache_to_paged.cu`
- `csrc/kv/kv_quant.cu`
- `csrc/kv/paged_kv_append.cu`
- `csrc/kv/scatter_kv.cu`
- `csrc/quant/dtype_convert.cu`
- `csrc/quant/turboquant.cu`
- `csrc/quant/turboquant_fast.cu`
- `csrc/gemm/marlin_kernel.cu`
- `csrc/gemm/marlin_repack.cu`
- `csrc/gemm/turboquant_weight_gemv.cu`
- `csrc/attention/decode_attention_quantized.cu`
- `csrc/attention/decode_attention_turboquant.cu`
- `csrc/attention/flashinfer_metadata.cu`
- `csrc/misc/pos_enc.cu`
- `csrc/misc/sampling.cu`
- `csrc/misc/split_qkv.cu`
- `tools/triton/basic_kernels.py`
- `tools/triton/silu_mul_kernel.py`

这些文件都看过了，但这轮不该和 P0 热路径抢改动预算。

## Six-Principles Findings

### 1. Global Memory Coalescing

整体结论：主热路径大体是对的，但 recurrent decode 里有一处明确的重复全局读。

- `gemv.cu` / `quantized_gemv.cu` 主要是按 row 连续访问权重、按 `k` 连续访问输入，coalescing 基本成立。
- `decode_prep_paged*.cu` 对 K/V pool 的 HND 写入布局是连续的，只要 `stride_page = kv_dim * page_size` 传对。
- `gated_delta_rule.cu` 和 `gdr_decode_batch.cu` 之前让四个 `j_slice` 都重复读取同一组 `q/k/v`，不是访存模式错，而是**完全可消掉的重复全局读**。

这轮已修：

- `csrc/misc/gated_delta_rule.cu`
- `csrc/misc/gdr_decode_batch.cu`

做法是只让 `j_slice == 0` 读取 `q/k/v`，然后通过 shared memory 复用给所有 slice。

### 2. Shared Memory Bank Conflicts

整体结论：没有发现 P0 级 bank-conflict blocker。

- `gemv.cu` 已经把 cross-warp reduction 改成了 padded / transposed shared layout。
- `gdr_decode_batch.cu` / `gated_delta_rule.cu` 的 shared partial buffer 访问模式按 `val_idx` 连续，不是当前最大问题。
- `fused_attention.cu` shared footprint大，但从静态访问模式看，当前更像 occupancy / launch shape 问题，不像显式 bank-conflict 问题。

因此这轮没有为了“可能存在的 bank conflict”去重写 hot kernel 的 shared layout。

### 3. Occupancy

整体结论：decode 热路径里，`GDR` 和 `HD256 decode prep` 是 occupancy 敏感点，不能再随便增大 shared / register 压力。

- `gated_delta_rule.cu` / `gdr_decode_batch.cu` 用 `512` 线程块，本来就偏重。
- `decode_prep_paged_hd256.cu` 用 `256` 线程块，每个 block 只做一个 `(kv_head, batch)`。
- `flashinfer_*` / `marlin_*` 更依赖上游库和 tile 选择，这轮不适合盲调 launch。

这轮对 occupancy 的处理是保守的：

- 不增加 GDR 的 j-slice 数
- 不扩大 decode prep 的 shared buffers
- `conv1d_decode_batch.cu` 改成 kernel-size 专门化，减少热路径里的 runtime branch 和无效寄存器活跃区

### 4. Tiling & Data Reuse

整体结论：这是这轮最有把握落地的一条。

- FlashInfer / Marlin / Triton attention 的 tile reuse 已经由对应库负责，不该在 wrapper 层“再发明一遍”。
- 自写 recurrent decode kernel 的主要问题不是 tile 太小，而是**已经读进来的数据没有跨 j-slice 复用**。

这轮已修：

- `gated_delta_rule.cu`
- `gdr_decode_batch.cu`

把 `q/k/v` 从“每 slice 各读一遍”改成“一次全局读，四个 slice 共享”。

### 5. Warp Divergence

整体结论：decode 热路径分歧不大，但 `conv1d_decode_batch.cu` 有一处可以直接消掉的 runtime branch。

- `sampling.cu` 天然更容易分歧，但不在当前第一波优化范围。
- `conv1d_decode_batch.cu` 虽然只有 `kernel_size <= 4`，但之前每个线程都走 runtime 分支和条件加载。

这轮已修：

- `csrc/misc/conv1d_decode_batch.cu`

改成 `kernel_size ∈ {2,3,4}` 的模板专门化，由 launcher `switch` 分发，热路径不再背 runtime branch。

### 6. Launch Config & Tail Effect

整体结论：这是全仓 CUDA kernel 的**长期约束**，不是这轮通过调 `blockDim` 就能彻底解决的。

- 很多 decode kernel 是 `(num_heads, batch)` 小网格，本质上在低 batch 下天然吃不满 L4。
- 真正的解法是更宽的 batching / 更多 slot / 更大粒度 fusion，而不是把 `128/256/512` 线程块换个数字碰碰运气。

因此这轮没有做“纯 launch 参数微调”。没有 profiling 证据时，这类改动风险大于收益。

## Implemented In This Pass

### 1. Recurrent decode 去掉重复全局读取

文件：

- `crates/cuda-kernels/csrc/misc/gated_delta_rule.cu`
- `crates/cuda-kernels/csrc/misc/gdr_decode_batch.cu`

改动：

- 只让 `j_slice == 0` 从 global 读取 `q/k/v`
- 通过 shared memory 复用给其它 `j_slice`
- 保持数值路径和状态更新顺序不变

原因：

- 这是明确违反 “Tiling & Data Reuse” 的地方
- 同时减少 global load 压力，且不改变线程映射和输出布局

### 2. `conv1d_decode_batch` 模板专门化

文件：

- `crates/cuda-kernels/csrc/misc/conv1d_decode_batch.cu`

改动：

- 新增 `kernel_size = 2/3/4` 模板实例
- launcher 用 `switch` 选择实例
- 去掉热路径里的 runtime 条件判断和死代码

原因：

- 这是当前 decode recurrent 路径里最安全的 “Warp Divergence / Occupancy hygiene” 改动

### 3. `decode_prep_paged*` 的 `stride_page` 修正

文件：

- `infer/src/ops/attention.rs`

改动：

- `stride_page = kv_dim * page_size`

原因：

- 对当前 `page_size = 1` 路径没有行为变化
- 但这是 HND paged layout 的正确 stride，避免把 `page_size > 1` 支持继续建立在错误 stride 之上

## Remaining Actions

### Next P0

- 给 `conv1d_decode_batch` / `gdr_decode_batch` 增加独立 microbench，避免只能看端到端噪声
- 跑 Qwen3.5 decode-heavy baseline / after snapshot
- 用 `nsys` 验证这轮改动是否真的减少了 recurrent decode 的 global load / kernel time

### Next P1

- 重新审 `decode_prep_paged*.cu` 的 block-to-work mapping，看是否值得把 `(kv_head, batch)` 做更粗粒度融合
- 审 `norm.cu` 和 `quantized_gemv.cu` 的 register pressure / occupancy，上 profiler 再决定是否加 launch-bounds 或改 tile

### Deferred

- `flashinfer_*`：以真实 trace 为准，不在 wrapper 层盲改
- `marlin_*`：除非有明确 profile 证据，否则不动上游 kernel
- `sampling.cu`：属于低占比 utility kernel，这轮不抢热路径预算
- `kv_cache_to_paged.cu`：保留为 correctness / page-size 扩展项，不和当前 decode 吞吐优化混做

## Rule

“review 所有 CUDA kernel” 不等于 “同时改所有 CUDA kernel”。

正确顺序是：

1. 先按六要素把全体 kernel 分级
2. 用 profiling 选热路径
3. 只改有证据、低风险、可验证的那一小批
4. benchmark / trace 证明有效后，再开下一波
