# `mul_scalar` lazy forward on Metal（M5.3b.13）

status: pending-remote (Mac-local structural + parity assertions pass;
end-to-end Qwen3.5 `METAL_EVAL_COUNT` reduction must be re-measured on
the training box)

## Title

attention q-scaling `q * (1/sqrt(d_head))` 驻留 MLX lazy graph — 每 token
潜在消除 28 evals (Qwen3.5 × 28 层)

## Context

M5.3b.1–12 把 forward pass 绝大多数 op 上了设备，但 attention 路径上 q
projection 之后、softmax 之前还有一个 scalar rescale。`ops::mul_scalar`
走 `store.ensure_host(a)?` + `backend.mul_scalar_forward(&host, s)`，
每次都要把 q matmul 的 lazy 结果 flush 回 host、rescale、再 upload 一份
回设备，Qwen3.5 28 层 attention × 1 次/层 = 28 次 eval/token 被烧掉。

## What Worked

**同 M5.3b 模板：trait device method + Metal override + Dirty dispatch。**

1. `crates/autograd/src/backend.rs` — 新增 `Backend::mul_scalar(x:
   &DeviceHandle, s: f32, shape: &[usize]) -> Result<DeviceHandle>`，
   默认 readback→`mul_scalar_forward`→upload fallback，CPU/CUDA 零改动。
2. `crates/autograd/src/backend_metal.rs` — override：
   - `mlx_array_new_float32(s)` 分配 rank-0 scalar array（开销可忽略）。
   - `mlx_multiply(x_handle, scalar)` 接入 MLX 图；MLX 自动把 rank-0
     标量 broadcast 到 x 的任意秩。
   - `mlx_array_free(scalar)` 即刻释放，输出 handle 接管乘法结果。
3. `crates/autograd/src/ops/elementwise.rs` — `mul_scalar` 拆成 dispatcher
   + `mul_scalar_device_lazy`（走 `store.ensure_device + backend.mul_scalar`）
   + `mul_scalar_host_eager`（原始 body）。dispatch 门：
   `device_handle.is_some() && dirty != Dirty::Host`（同 rope/embedding/
   reshape 模板，覆盖 `Dirty::Both`）。
4. `crates/autograd/src/ops.rs` — 公共 `mul_scalar` 剥掉 `ensure_host`，
   加 M5.3b.13 注释解释 inner dispatcher 接管。
5. `crates/autograd/tests/test_device_handle.rs` — 新增
   `metal_mul_scalar_forward_stays_lazy`：`matmul → mul_scalar(1/√8) →
   sum` 链断言 `before_mul == after_mul == 0`（readback 才 eval）；CPU
   参考对拍 ≤ 1e-5。

## Rule

**rank-0 MLX 标量是最轻的广播乘法载体。** 不需要 `mlx_full` 扩到 x 的形状，
不需要手写 `mul_scalar` kernel — MLX 把 rank-0 array 视作 broadcast-scalar，
对所有秩的乘法都成立，分配开销就是一个 1-元素 `mlx_array`。这个模式可以
照搬给 `neg`（`mul_scalar(x, -1.0)`）等其他标量-elementwise op 的未来
lazy 化。

**每个新增的 lazy op override 都要成对新增一个 `metal_*_forward_stays_lazy`
acceptance test。** 单纯改代码不留 fixture，将来有人 "顺手"
`ensure_host` 一下，就会把这十几步攒下的 eval 优化静默打穿。

## Verification

Mac local:

```bash
cargo check  -p autograd --features metal --release
cargo test   -p autograd --features metal --release --test test_device_handle metal_mul_scalar
cargo test   -p autograd --features metal --release
cargo test   -p train    --features metal --release
cargo clippy -p autograd --features metal --release --tests -- -D warnings \
    -A clippy::arc_with_non_send_sync -A clippy::type_complexity
```

全绿；新增 `metal_mul_scalar_forward_stays_lazy` 通过；autograd 全 19
路 metal 测试 + train 全 36 路无回归。clippy 两条 `-A` 为预存遗留。

pending-remote (train box, Qwen3.5-4B):

- 训练步 `METAL_EVAL_COUNT` 对拍 pre-M5.3b.13 commit vs 当前 commit，
  验证 28 层 attention q-scale evals 从 28/step 降到 0。
- `cargo run -p train --release --bin train_multi_turn --features metal
  -- --backend metal --iters 30` 数值稳定性 vs M5.3b.12 baseline 无漂移。

## Cross-links

- Plan row: `docs/plans/rust-agent-rl-single-node.md` M5.3 — M5.3b.13
  entry 已添加。
- 前置: `docs/experience/wins/2026-04-21-lazy-reshape-transpose-metal.md`
  (M5.3b.12).
- 姐妹 M5.3b 模板全集见 M5.3b.1–12 各 wins 条目。
