# 2026-04-15 · Metal Qwen3.5 Live Prefix Reuse

## Context

`Qwen3` 的 live prefix reuse 已经在 Metal scheduler runtime 里打通，但
`Qwen3.5` 还缺这条路。它的 compiled MLX path 把 full-attention KV 和
linear-attention recurrent state 都保存在 Rust 持有的 `MlxArray` 里，所以
可以先走 Rust-side snapshot replay，而不必先扩一轮新的 C++ bridge。

## What Worked

- 给 `MetalRequestState` / `Qwen35StepDriver` 加了 Qwen3.5 prefix snapshot
  import/export。
- publish 端不做 zero-copy pool，而是 replay 编译态 prefill，在每个
  block-aligned prefix boundary 生成可复用 snapshot。
- runtime admission 端按最长已缓存前缀做 snapshot import，只把 suffix 交给
  scheduler。
- 本地 `Qwen3.5-4B-MLX-4bit` HTTP smoke 打出了真实命中：
  - run 1: `533.9 ms`
  - run 2: `145.6 ms`
  - `/metrics`: `infer_prefix_hits_total=1`,
    `infer_prefix_hit_rate=0.3333`

## Rule

对于带 recurrent state 的 Metal 模型，先做 copy-based snapshot reuse，把真实
serving 命中跑通，再决定是否值得继续投入 zero-copy shared-state 设计。
