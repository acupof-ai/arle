# `add_broadcast` lazy forward on Metal（M5.3b.14）

status: pending-remote (Mac-local structural + parity assertions pass;
end-to-end Qwen3.5 `METAL_EVAL_COUNT` reduction must be re-measured on
the training box)

## Title

attention causal-mask add + Linear bias add 驻留 MLX lazy graph — 修正
dispatch 语义为 **OR**（不是 AND）以避免 host 常量反噬 device 产出

## Context

M5.3b.1–13 把 forward pass 大部分 op 上了设备，但 attention 路径上
`scores + causal_mask` 仍走 host-eager：`ops::add_broadcast` 的公共入口
`ensure_host(a) + ensure_host(b)`，把刚 matmul 出来的 device-resident
scores flush 回 host 再做加法。每层 × 28 层在 Qwen3.5 里就是大量
readbacks。Linear bias 的 `linear_out + bias` 同理。

## What Worked

**trait device method + Metal override + dispatch 语义修正。**

1. `crates/autograd/src/backend.rs` — 新增 `Backend::add_broadcast(
   a: &DeviceHandle, a_shape, b: &DeviceHandle, b_shape) ->
   Result<DeviceHandle>`，默认 readback→`add_broadcast_forward`→upload
   fallback。
2. `crates/autograd/src/backend_metal.rs` — override 直接用 `mlx_add`。
   **关键**：MLX 的 `mlx_add` 已原生实现 NumPy-style right-aligned
   broadcasting，`[merged_heads, seq, seq] + [1, seq, seq]` 自动广播成
   `[merged_heads, seq, seq]`，**不需要显式 `mlx_broadcast_to`**，单个
   FFI 调用即搞定。
3. `crates/autograd/src/ops/broadcast.rs` — `add_broadcast` 拆成
   dispatcher + `add_broadcast_device_lazy` + `add_broadcast_host_eager`。
4. `crates/autograd/src/ops.rs` — 公共入口剥掉 `ensure_host`。

## Rule

**Dispatch 语义：`a_lazy || b_lazy`，不是 `&&`。** 常见场景：一个
device-resident activation（scores, linear_out）加一个 host-only 常量
（mask, bias）。若用 AND，"b 不 lazy" 就会让整个 op 走 host_eager，
`ensure_host(a)` 直接 flush 掉上游 matmul 的 lazy 图 — 正是要消灭的
readback。用 OR 意味着：只要 **任一** side 已经 lazy，就把另一 side
`ensure_device` 上传（小张量 upload 便宜，大 activation readback 昂贵）。

**host_eager 内部必须 `ensure_host(a) + ensure_host(b)`。** CPU 后端的
matmul 也走 `alloc_device_tensor`（`DeviceHandle::Cpu(Vec<f32>)`），产出
`Dirty::Device`。下游的 Linear bias 是 host-only（device_handle=None）。
在我们的 dispatch 里，`a_lazy=false`（Cpu handle 的 Dirty::Device 算
lazy？其实 `t.device_handle.is_some() && t.dirty != Dirty::Host`
在 CPU 上也成立）—— 所以实测 CPU 后端的 Linear bias case 会走
**device_lazy**，因为 OR 一分一走。但若某个场景仍落入 host_eager，
`Tensor::clone()` 的 `dirty != Device` 断言会触发 panic，所以 host_eager
头部显式同步两边到 host。这条在 `module::tests::linear_forward*` 两例
上触发过，修之。

**OR dispatch 的性能直觉：小张量 upload ≪ 大 activation readback。**
mask / bias 通常 ≤ 一个 attention matrix 的千分之一大小，MLX 的 upload
是 memcpy 级别；readback 要先 `mlx_eval` 等 GPU 队列排空再拷贝。方向
非对称，dispatch 设计就要对齐这个方向。

## Verification

Mac local:

```bash
cargo check  -p autograd --features metal --release
cargo test   -p autograd --features metal --release --test test_device_handle metal_add_broadcast
cargo test   -p autograd --features metal --release
cargo test   -p train    --features metal --release
cargo clippy -p autograd --features metal --release --tests -- -D warnings \
    -A clippy::arc_with_non_send_sync -A clippy::type_complexity
```

全绿；新增 `metal_add_broadcast_forward_stays_lazy` 通过；autograd 全 20
路 metal 测试 + train 全 36 路无回归；早前 `module::tests::linear_forward*`
panic 修复。clippy 两条 `-A` 为预存遗留。

pending-remote (train box, Qwen3.5-4B):

- 训练步 `METAL_EVAL_COUNT` 对拍 pre-M5.3b.14 commit vs 当前 commit，
  验证 attention path 的 mask-add + bias-add evals 消除。
- `cargo run -p train --release --bin train_multi_turn --features metal
  -- --backend metal --iters 30` 数值稳定性 vs M5.3b.13 baseline 无漂移。

## Cross-links

- Plan row: `docs/plans/rust-agent-rl-single-node.md` M5.3 — M5.3b.14
  entry 已添加。
- 前置: `docs/experience/wins/2026-04-21-lazy-mul-scalar-metal.md`
  (M5.3b.13).
- 姐妹 M5.3b 模板全集见 M5.3b.1–13 各 wins 条目。
