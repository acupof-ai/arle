# `mean` lazy forward on Metal — CE-loss head chokepoint（M5.3b.19）

status: pending-remote (Mac-local parity + eval-count assertions pass;
end-to-end Qwen3.5 training-step `METAL_EVAL_COUNT` reduction must be
re-measured on the training box — **expected to be the biggest single
win in the M5.3b series** since `mean` was the one remaining eager op
in the CE-loss path)

## Title

CE loss 头部 `log_softmax → gather_last_dim → mean → mul_scalar(-1)`
端到端 lazy — 复用 M5.3b.1 `sum_all` + M5.3b.13 `mul_scalar` 组合

## Context

M5.3b.1–18 把 Qwen3.5 forward pass 每层 attention/MLP 的所有 op 都驻留
MLX lazy 图。但 `ops::mean` 在 `ops.rs:264-267` 仍然是
`store.ensure_host(a)?; reduce::mean(a, store, tape)`。mean 被 CE loss
使用（`crates/train/src/loss.rs:14`：`mean(target_log_probs, ...)`），
也就意味着**每个训练步**，lazy log_probs `[batch,seq]` 被整块 flush
回 host，只为了计算一个标量 loss —— 反噬了所有 M5.3b 的上游努力。这
是 M5.3b 系列剩下唯一一个 single-highest-value 修复。

## What Worked

**不加新的 Backend trait method — 直接在 ops 层用已有 lazy primitives
组合：`sum_all + mul_scalar(1/numel)`。**

1. `crates/autograd/src/ops/reduce.rs` — `mean` 拆成 Dirty dispatcher +
   `mean_device_lazy` + `mean_host_eager`。Dirty::Device 路径先跑
   `store.backend().sum_all(&input_handle, &input_shape)` 得到 scalar
   handle（M5.3b.1 引入的 lazy sum），再跑
   `store.backend().mul_scalar(&sum_handle, 1.0/numel, &[])` 得到均值
   scalar handle（M5.3b.13 引入的 lazy rank-0 multiply）。两个 FFI 调
   用组合进 MLX 图，0 eval。numel=0 guard 避免 div-by-zero。
2. `crates/autograd/src/ops.rs` — 公共 `mean` 入口剥掉 `ensure_host`，
   加 M5.3b.19 说明注释。
3. `mean_backward` 无需改动 —— 读取的只是 saved `numel` 和 output_grad
   的 `.data[0]` 标量，不涉及输入 tensor 的 Dirty 状态。

## Rule

**"无新 trait method，纯 primitives 组合" 是一个优雅的变体模板。** M5.3b
绝大多数 op 都新增了 `Backend::xxx` 方法，但 `mean` 语义上就是
`sum / numel`，而 `sum` 和 `mul_scalar` 各自已经 lazy。在 ops 层直接
组合这两条 FFI 比新增一个 `mean_all` trait method 更简单、更少代码、
对 CpuBackend/CudaBackend 零侵入（它们通过默认 readback fallback 仍
然正确）。

**CE-loss head 的隐形 chokepoint。** `ops::mean` 看起来是一个小标量 op，
但在 loss 路径上它 gate 了整个 forward pass 的 lazy 保留。同类 pattern
需要检查任何"最后一步收到标量"的聚合 op（mean / sum / max_entropy
等），它们单独看是一次 flush，但在 training step 维度上是反噬全链的
amplifier。

## Verification

Mac local:

```bash
cargo check  -p autograd --features metal --release
cargo test   -p autograd --features metal --release --test test_device_handle metal_mean_forward
cargo test   -p autograd --features metal --release       # 25 tests (+1), all green
cargo test   -p train    --features metal --release       # 30+ groups, all green
cargo clippy -p autograd --features metal --release --tests -- -D warnings \
    -A clippy::arc_with_non_send_sync -A clippy::type_complexity
```

全绿；新增 `metal_mean_forward_stays_lazy` 跑 `matmul → mean` 链，前向
0 eval、backward ≤3 eval，CPU 参考对拍 ≤ 1e-4。

pending-remote (train box, Qwen3.5-4B):

- 训练步 `METAL_EVAL_COUNT` 对拍 pre-M5.3b.19 vs 当前 commit，预计**显著**
  下降（上游 lazy `log_softmax → gather_last_dim` 的值不再 per-step
  flush）。
- `cargo run -p train --release --bin train_multi_turn --features metal
  -- --backend metal --iters 30` 数值稳定性 vs M5.3b.18 baseline 无漂移。
- 这是判定 M5.3b 系列工程价值的最终 benchmark：如果 per-step eval 数
  没有显著下降，说明还有其他隐形 chokepoint（候选：`softmax`
  backward、`log_softmax` backward、`gather_last_dim` backward 里的
  scatter_add_rows 路径，或 `mul_scalar` 后续的标量运算）。

## Cross-links

- Plan row: `docs/plans/rust-agent-rl-single-node.md` M5.3 — M5.3b.19
  entry 已添加。
- 前置/姐妹: `2026-04-20-lazy-sum-metal.md` (M5.3b.1, `sum_all` 基础)，
  `2026-04-21-lazy-mul-scalar-metal.md` (M5.3b.13, `mul_scalar` 基础)，
  `crates/train/src/loss.rs:14` (CE loss 调用点)。
- 姐妹 M5.3b 模板全集见 M5.3b.1–18 各 wins 条目。
