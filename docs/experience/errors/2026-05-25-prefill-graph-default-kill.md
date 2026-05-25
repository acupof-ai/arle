# INFER_PREFILL_GRAPH=1 默认化 KILL — 高 c shape thrash

## Context

2026-05-25 L4 nsys 分析显示 bench window GPU 50% idle(12.6s of 25s),其中 ~9%
wall 是未走 CUDA Graph 的 raw `cudaLaunchKernel` overhead(5992 × 380µs)。
decode 已走 graph(288 cuGraphLaunch 命中),prefill 没走。

发现 `INFER_PREFILL_GRAPH=1` env flag 已实现 paged prefill graph capture
(`infer/src/model/qwen3/prefill.rs:362-367` + `:825-867`),试图默认开启。

## Root Cause

bench c=1..16 sweep on L4 / Qwen3-4B / 4096 prompt / 256 out / 60s:

| c | baseline tok/s | A 启用 tok/s | Δ |
|--:|--:|--:|--:|
| 1 | 26.18 | 27.85 | **+6.4%** |
| 2 | 45.10 | 47.23 | **+4.7%** |
| 4 | 75.58 | 63.62 | **-15.8%** |
| 8 | 119.07 | 27.65 | **-77%** |
| 16 | 167.19 | 24.32 | **-86%** |

c=8 / c=16 出现大量 `scheduler_channel_closed` 错误,bench validation 失败
(`TTFT p50 = 0.0`)。

**根因**:`QWEN3_PREFILL_GRAPH_CACHE_MAX_KEYS = 8`(`prefill.rs:369`)是 LRU
上限。每个 prefill graph 的 key 由 `Qwen3PrefillGraphKey { token_count,
page_indices_len, prefix_rows }` 组成,带 bucketing
(`QWEN3_PREFILL_GRAPH_PAGE_INDICES_BUCKET=64`、`_PREFIX_ROWS_BUCKET=128`)。

c=1 / c=2 时每秒新增 ≤ 2 个 distinct shape variant → 8 个 slot 够 →
命中率高 → 收益。

c=16 时每秒 16 个并发 session,每个 session prompt 长度 / KV layout 都不同
→ 每秒 > 8 个 distinct shape → cache evict + re-capture → 每个 prefill 都
重新走 capture path(本来该走 replay) → CPU 阻塞 → scheduler tick 失速 →
请求超时 channel close。

## Fix

不默认 `INFER_PREFILL_GRAPH=1`。在 `qwen3_prefill_graph_requested()` 之上
加 doc comment 说明 high-c shape thrash,引此 errors entry。

如果未来要默认 on,二选一:
- 加宽 bucketing(`PAGE_INDICES_BUCKET` 64 → 256,`PREFIX_ROWS_BUCKET`
  128 → 512)让 c=16 collapse 到 ≤8 unique keys
- 扩大 `CACHE_MAX_KEYS` 到 32+,加 LRU eviction 计数器,bench c=16 确认 hit
  rate ≥ 80%

## Rule

- CUDA Graph cache 默认化前**必须** sweep c=1..16 验证 hit rate,不能只
  c=1/c=2 测了就 ship default。
- Per-shape graph 的 cache 容量必须 ≥ peak concurrency × per-session shape
  variant 数,否则在 high-c 比纯 kernel launch 更慢(re-capture stall)。
- bench validation 失败(TTFT=0、ttft_ms=null)是 scheduler 阻塞信号,不是
  metric 计算 bug — 先怀疑 server,别先怀疑 guidellm。

## Cross-refs

- nsys 发现 GPU 50% idle 来源:[`2026-05-25-bench-guidellm-cuda-l4-arle-nsys-h1-h5-confirmed.md`](../wins/2026-05-25-bench-guidellm-cuda-l4-arle-nsys-h1-h5-confirmed.md)
- 代码注释位置:`infer/src/model/qwen3/prefill.rs:362` 上方 block doc
- A bench 原始数据: archived L4 bench-output directory
