# `reshape` + `transpose_axes_swap` lazy forward on Metal（M5.3b.12）

status: pending-remote (Mac-local structural + parity assertions pass;
end-to-end Qwen3.5 `METAL_EVAL_COUNT` reduction must be re-measured on
the training box)

## Title

attention 路径每层 q/k/v projection 后的 `reshape` + `transpose` 组合
全部驻留 MLX lazy graph — 每 token 潜在消除 252 evals（28 层 × 9 op）

## Context

M5.3b.1–11 已经把 matmul / sum / softmax / log_softmax / silu / exp /
rope / rmsnorm / embedding / gelu / gather_last_dim / AdamW 上设备。
forward pass 剩下两个高频 op 仍走 host-eager：`reshape` 和 `transpose`。
Qwen3.5 attention 每层每 token 6 次 reshape + 3 次 transpose（q/k/v
projection 后 `[B, S, H*D] → [B, S, H, D] → [B, H, S, D]` + 末尾 output
projection 前 reverse），× 28 层 = 252 强制 readback + upload，把前
面十一步 lazy 累积的收益直接烧掉。

## What Worked

**trait-level lazy 入口 + Metal override + ops 层 Dirty dispatch，沿用
M5.3b 模板；transpose 的关键坑点是 MLX view 语义。**

1. `crates/autograd/src/backend.rs`
   - 新增 `Backend::reshape(x: &DeviceHandle, new_shape: &[usize]) ->
     Result<DeviceHandle>`，默认 readback→upload fallback。
   - 新增 `Backend::transpose_axes_swap(x: &DeviceHandle, old_shape:
     &[usize], axis1: usize, axis2: usize) -> Result<(DeviceHandle,
     Vec<usize>)>`，默认用 `cpu_transpose_swap` helper。
   - 新增 `pub fn cpu_transpose_swap(data: &[f32], old_shape: &[usize],
     axis1: usize, axis2: usize)` CPU 参考实现（nested loop swap axes）。
2. `crates/autograd/src/backend_metal.rs`
   - `reshape` override：直接调 `mlx_reshape` 拿到元数据节点（MLX 的
     reshape 在连续 row-major 数据上就是 stride 改写，~0 cost）。
   - `transpose_axes_swap` override：
     - 恒等 case（`axis1 == axis2`）走 `mlx_reshape` alias 避免 FFI
       overhead。
     - 非恒等 case 用 `mlx_transpose_axes` 拿到 **non-contiguous view**，
       然后**立即包一层 `mlx_contiguous`** 并 free 掉中间 view，最后
       返回 contig handle。关键细节见下方 "Rule" 段。
3. `crates/autograd/src/ops/layout.rs`
   - `reshape` / `transpose` 各拆成 public dispatcher + 两条分支
     (`*_device_lazy` / `*_host_eager`)。
   - dispatch 门：`t.device_handle.is_some() && t.dirty != Dirty::Host`
     （同 rope/embedding，覆盖 `Dirty::Both`）。
4. `crates/autograd/src/ops.rs`
   - 剥掉公共 `reshape` / `transpose` 入口的 `store.ensure_host(x)?`，
     加 M5.3b.12 注释解释 inner dispatcher 接管。
5. `crates/autograd/tests/test_device_handle.rs`
   - `metal_reshape_forward_stays_lazy`：`matmul → reshape` 链
     `before_reshape == 0 && after_reshape == 0`，readback 时才 eval；
     CPU 参考对拍 ≤ 1e-4。
   - `metal_transpose_forward_stays_lazy`：`matmul → transpose` 链
     `before_transpose == 0 && after_transpose == 0`；CPU
     `cpu_transpose_swap` 对拍 ≤ 1e-4。

## Rule

**MLX transpose 返回 non-contiguous view；readback 读原始 storage 会拿到
pre-transpose 布局的 bytes。** 第一次实现时 transpose parity fail 为
`metal=-0.4118606 cpu=-0.31485343` — 不是数值精度问题，是**布局错位**。
`mlx_transpose_axes` 只改 strides + shape，不移动 bytes；`mlx_eval` 即便
触发，storage 仍然是原始连续布局；`mlx_array_data_float32` 给的是裸指针，
完全忽略 view strides。修复：**transpose override 内部必须显式
`mlx_contiguous`**，把 view 物化成新布局下的 contig array。MLX 对已
contig 的 array 在 contiguous 调用上短路（内部 no-op），所以
reshape-only / identity-swap 路径不会有额外成本。

**reshape 不受影响**：row-major 连续数据的 reshape 就是元数据改写，
bytes 顺序本就对的，`mlx_reshape` 的 view 在 readback 时直接正确。

**更广的启示**：任何 MLX op 返回 view 而不是新 buffer 的（transpose、
slice、take 轴上的非起点 slice）都要在 override 内部包一次 contiguous，
**不能指望 caller 记得**。trait 契约：override 返回的 `DeviceHandle` 必须
可以无条件 readback 成正确字节序的 row-major array。

## Verification

Mac local:

```bash
cargo check -p autograd --features metal --release
cargo test  -p autograd --features metal --release --test test_device_handle metal_reshape metal_transpose
cargo test  -p autograd --features metal --release
cargo test  -p train    --features metal --release
cargo clippy -p autograd --features metal --release --tests -- -D warnings \
    -A clippy::arc_with_non_send_sync -A clippy::type_complexity
```

全绿；新增 `metal_reshape_forward_stays_lazy` /
`metal_transpose_forward_stays_lazy` 通过；autograd 全 18 路 metal 测试
+ train 全 36 路无回归。clippy 两条 `-A` 为预存遗留（非本 diff 引入）。

pending-remote (train box, Qwen3.5-4B):

- 训练步 `METAL_EVAL_COUNT` 对拍 pre-M5.3b.12 commit vs 当前 commit，
  验证 28 层 attention 路径的 reshape/transpose evals 从 252/step 降到 0。
- `cargo run -p train --release --bin train_multi_turn --features metal
  -- --backend metal --iters 30` 数值稳定性 vs M5.3b.11 baseline 无漂移。

## Cross-links

- Plan row: `docs/plans/rust-agent-rl-single-node.md` M5.3 — M5.3b.12
  entry 已添加。
- 前置: `docs/experience/wins/2026-04-21-adamw-batched-eval-metal.md`
  (M5.3b.11).
- 姐妹 M5.3b 模板全集见 M5.3b.1–11 各 wins 条目。
