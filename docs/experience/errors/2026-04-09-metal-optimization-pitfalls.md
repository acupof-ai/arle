# 2026-04-09 · Metal MLX 优化踩坑记录

## Context

将 Qwen3.5-4B-MLX-4bit 从 45 tok/s 优化到接近 mlx_lm 的 77 tok/s。

## 踩坑清单

### 1. C API 开销假设 (错误)
- **假设**: Rust→C API wrapper 每次调用有 ~18μs 开销，425 ops × 18μs = 7.5ms
- **事实**: micro-benchmark 证实 C API vs C++ direct API 开销为 **零** (199ns vs 185ns)
- **教训**: 不要猜测，要 benchmark

### 2. Stream context 假设 (错误)
- **假设**: mlx_lm 的 `mx.stream()` context 启用 GPU 流水线
- **事实**: `StreamContext` 只是 RAII 设置线程本地默认 stream，不影响调度
- **测试**: 换 generation stream 反而变慢 (39.6 vs 45.2)
- **教训**: 读源码确认机制，不要望文生义

### 3. async_eval 假设 (部分错误)
- **假设**: async double-buffering 能让 CPU 图构建和 GPU 计算重叠
- **事实**: Qwen3.5 的 GDR 有 recurrent state 依赖链，async 不稳定
- **Qwen3 有效**: Qwen3 纯 self-attention 没有 recurrent state，async 有帮助
- **教训**: 有状态依赖的模型，async 需要更谨慎

### 4. mlx-sys ABI mismatch (已解决)
- **症状**: C++ fused blocks 运行时 SIGSEGV
- **根因**: target/release/build 下残留旧版 mlx-sys 目录，build.rs 扫到错误的 headers
- **修复**: 删除旧目录
- **教训**: cargo 增量编译可能留下 stale artifacts

### 5. split() 函数重载歧义
- **症状**: `split(q_full, {head_dim}, -1)` 结果是 [1,1,16,2] 而不是 [1,1,16,256]
- **根因**: `{head_dim}` 被解释为 int (split into N parts) 而非 vector<int> (split at indices)
- **修复**: `split(q_full, std::vector<int>{head_dim}, -1)`
- **教训**: C++ 花括号初始化列表有歧义风险

### 6. SDPA null mask_mode crash
- **症状**: decode 第 2 个 token 时 SIGSEGV
- **根因**: `std::string(nullptr)` 是 UB，C wrapper 不检查 null
- **修复**: 总是传 "" 而非 null
- **教训**: C API null 指针 vs C++ std::string 不兼容

### 7. Metal kernel crash (未解决)
- **症状**: Rust MetalKernel::apply 在 eval 时 SIGSEGV
- **事实**: 相同的 kernel source 在 Python 中正常工作
- **可能原因**: mlx_fast_metal_kernel_apply 的 config 传参或输出提取问题
- **教训**: 需要写最小可复现用例隔离问题

### 8. pmetal-mlx-sys 0.2.4 升级 (失败)
- **症状**: Qwen3 跑到 145 tok/s (提速!) 但 Qwen3.5 crash
- **原因**: API 变化 (quantized_matmul 加了 mode/optional 参数等)
- **修复了 API**: dequantize, quantized_matmul, sdpa — 但仍 crash
- **可能原因**: mlx_load_safetensors 或其他未发现的 API 变化
- **教训**: 大版本升级需要系统性测试，不能逐个函数修

## 根因分析

最终确认的性能差距来源:

| 因素 | 影响 | 状态 |
|------|------|------|
| 手动 RMS norm (10 ops → 1 fast op) | +5% | ✅ 已修复 |
| 冗余 as_dtype 转换 | +3% | ✅ 已修复 |
| mmap 加载 | 内存 -67% | ✅ 已修复 |
| GDR state update (14 ops → 1 kernel) | +30%~40% | ❌ Metal kernel crash |
| C API 开销 | 0% | 证伪 |
| Stream/async | 0~-5% | 证伪 (有副作用) |

## Rule

1. **先 profiling，后优化**: 用数据驱动，不要凭假设
2. **比较 op 数量**: 性能差距先看总 op 数
3. **看源码，别猜**: StreamContext/split/sdpa 都是看了源码才搞清楚
4. **增量编译坑多**: stale artifacts 会导致 ABI mismatch
5. **大升级全量测**: mlx-sys 升级需要全链路回归
