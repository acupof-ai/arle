# W4A8 PR #31 vs QQQ main 真实 diff finding(纠正之前 +21 LOC bf16 误报)

> 接续 `2026-05-09-w4a8-upstream-qqq-survey.md`(误报 +21 LOC bf16 activation)。
> 实际 diff QQQ main `b6582d1` qqq_gemm.cu 1106 LOC vs ARLE
> `marlin_w4a8_kernel.cu` 987 LOC = **+119 LOC delta**,核心是
> **thread config auto-tune dispatch**(不是 bf16 activation)。

## 纠错 — diff 实际内容

### ❌ 不是 bf16 activation
之前误以为 +21 LOC 是 bf16 activation 支持(`#include <cuda_bf16.h>`)。
实际 +1 LOC 只是 header,kernel 内部仍 fp16-only。

### ❌ ARLE 也不是更老版本
看 diff 反而 **ARLE 已经包含 sm_89 specific L2 cache eviction hint**:
- `cp_async4_stream` with `createpolicy.fractional.L2::evict_first` + `cp.async.cg.shared.global.L2::cache_hint`
- `cp_async1_stream` with same L2 hint
- QQQ main upstream 是更基础的 `cp_async4` / `cp_async1`(no L2 hint)
- ARLE PR #31 cherry-pick 包含 vLLM team 加的 sm_89 优化

### ✅ 真实 +119 LOC = thread config auto-tune dispatch

QQQ main `qqq_gemm.cu` 行 820-960 加入:

```cpp
typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

thread_config_t small_batch_thread_configs[] = {  // prob_m <= 16(decode/小 batch)
    {128, 128, 256},  // Default
    {128, 64, 128},   // Reduce N 2X, same K
    {64, 256, 256},   // Reduce K 2X, increase N 2X
    {64, 128, 128},
};

thread_config_t large_batch_thread_configs[] = {  // prob_m > 16(prefill/大 batch)
    {64, 256, 256},   // Default
    {128, 128, 256},  // Reduce N 2X, increase K 2X
    {64, 128, 128},
    {128, 64, 128},
};

bool is_valid_config(thread_config_t const& th_config, int prob_m, int prob_n, int prob_k) {
  // K/N divisible, thread_k ∈ {64, 128}, num_threads >= 128, etc.
  ...
}

thread_config_t determine_thread_config(int prob_m, int prob_n, int prob_k) {
  if (prob_m <= 16) {
    for (auto th_config : small_batch_thread_configs)
      if (is_valid_config(...)) return th_config;
  } else {
    for (auto th_config : large_batch_thread_configs)
      if (is_valid_config(...)) return th_config;
  }
  return {-1, -1, -1};
}

#define __CALL_IF(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, GROUP_BLOCKS, NUM_THREADS) \
  else if (thread_m_blocks == ... && num_threads == NUM_THREADS) {                              \
    cudaFuncSetAttribute(Marlin<NUM_THREADS, ...>, ...);                                        \
    Marlin<NUM_THREADS, ...><<<blocks, NUM_THREADS, max_shared_mem, stream>>>(...);             \
  }
```

vs ARLE 当前固定 `const int THREADS = 256`,**单一 dispatch path**。

## 这意味着什么

QQQ main 的 **核心改进**:

1. **小 batch(decode prob_m=1..8)走 `(128,128,256)` default,fallback 到 `(128,64,128)` / `(64,256,256)` / `(64,128,128)`**
2. **大 batch(prefill prob_m=16+)走 `(64,256,256)` default,fallback 到 `(128,128,256)` 等**
3. ARLE 当前 W4A8 全用同一 tile config(`THREADS=256`),**未根据 batch size 调优**

## 预估 perf gain(预测,not 实测)

- **B7 c=4 decode batch**(prob_m=4):走 `small_batch_thread_configs` `(128,128,256)` default — 接近 ARLE 当前 default,可能 持平
- **B5 W4A16 c=4 prefill seq=4096 prob_m=4096**:走 `large_batch_thread_configs` `(64,256,256)` default — **不同 tile**,**可能 ITL -10-20% in prefill**(thread_n=256 vs current 不同)
- 重要的是 **小 batch fallback paths**(1-2-4 token decode):允许更小 thread_k(64) reduce K 2x → 可能让 small-batch decode **launch 更多 blocks per SM**,occupancy 提升

最大 ROI 在:
- **W4A8 decode batch=1 路径**(B6 W4A8 c=1) — 单 token 走 small_batch fallback,可能 ITL -5-10%
- **W4A8 prefill 长 seq**(B7 c=4 4096-in 的 prefill 部分,影响 TTFT)— 大 batch tile 优化

## ⚠ Risk:dispatch correctness

QQQ main 的 multi-config dispatch 需要每个 config 都被 `__CALL_IF` 实例化为 cubin。如果忘记某个 (thread_m, thread_n, thread_k, num_threads) 组合,就会 fallback 到 unconditional error。需要全 enumerate 列表 + 测试。

ARLE 当前 cherry-pick 的 `CALL_IF` 宏已经有 (THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, GROUP_BLOCKS) 4 维 dispatch,加 NUM_THREADS 维度 → 4×4×3 = ~48 个新 instance(粗估)→ 编译时间 + cubin size 增加。

## 移植 patch outline(实际)

1. **Header includes**:不变(我们已有 cuda_fp16.h)
2. **Add `thread_config_t` struct + 2 arrays + `is_valid_config` + `determine_thread_config`** at marlin_w4a8_kernel.cu line ~820
3. **Replace `const int THREADS = 256` with `USER_THREADS = 256`** + dynamic per-call config
4. **Upgrade `CALL_IF` macro to `__CALL_IF`**(add NUM_THREADS template param)
5. **Enumerate new config combinations**(每个 small/large batch fallback × M/N/K blocks × group_blocks)
6. **Adapter (linear.rs:1307)**:无变化(kernel-internal dispatch)

LOC delta:**约 +130-150 LOC(QQQ +119 + 列出更多 instance)**。

## 推荐 Phase 顺序(更新)

| Phase | 内容 | LOC | 风险 | 预估 gain |
|-------|------|-----|------|----------|
| **1** | **QQQ main thread_config auto-tune dispatch port** | +130-150 | 中(dispatch table 全 enumerate) | **B6 ITL -5-10%,B7 prefill TTFT -5-15%** |
| 2 | sm_89 specific re-tune of new configs | + ncu sweep | 低 | -3-5% 上 |
| 3(可选)| QQQ i4fp8 FP8 activation | +167 | 中 | accuracy ↑,perf ~0% |

**修正之前 estimate**:
- ❌ 之前说 "+21 LOC bf16 activation,ITL -2-5%" → 错误
- ✅ 实际 "+130-150 LOC thread_config dispatch,B6 -5-10% / B7 prefill -5-15%"

## 立即 next step

GPU 当前 codex 占用(SGLang baseline),Claude 不能 bench。但可以 **CPU work 把 patch 写好**:
1. Read full `__CALL_IF` macro from QQQ main
2. Read existing `CALL_IF` from ARLE marlin_w4a8_kernel.cu
3. Write port patch as a separate `.diff` file ready for apply
4. Commit prep ready for build/bench when GPU 释放

## Cross-references

- Diff 详细 dump:`/tmp/qqq_diff.txt`(local cache,not in repo)
- QQQ source local cache:`/tmp/qqq_main_kernel.cu`
- ARLE current:`crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu`
- Adapter:`infer/src/ops/linear.rs:1307 run_marlin_w4a8_linear`
- 之前误报 brief:`docs/research/2026-05-09-w4a8-upstream-qqq-survey.md`(纠正中)

## 状态

QQQ main 真实 +119 LOC 是 **thread_config auto-tune dispatch**(纠正之前 bf16
activation 误报)。ARLE 已有 sm_89 L2 cache hint(没倒退)。下一步:写 patch
outline 等 GPU 释放再 bench。
