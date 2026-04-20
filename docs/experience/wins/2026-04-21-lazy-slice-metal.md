# `slice` lazy forward on Metal（M5.3b.16）

status: pending-remote (Mac-local parity + eval-count assertions pass;
end-to-end Qwen3.5 `METAL_EVAL_COUNT` reduction must be re-measured on
the training box)

## Title

Qwen3.5 attention layer 的 `q_full` → `q` + `gate` 窗切片驻留 MLX lazy
graph — `mlx_slice` 是 lazy view op，触发 contiguous 物化修复视图 readback
陷阱

## Context

M5.3b.1–15 把 forward pass 的绝大多数 op 都在 Metal 上 lazy 化了，
但 `slice` 仍走 host-eager：`ops::slice` 公共入口 `ensure_host(x)?`，
把上游 matmul 刚产出的 device-resident `q_full` flush 回 host 再做
per-axis 拷贝。Qwen3.5 每层 attention 里 q_proj 实际输出
`[batch, seq, heads, head_dim*2]`，随后切成 `q = q_full[..., :head_dim]`
和 `gate = q_full[..., head_dim:]`（`crates/train/src/qwen35.rs:100-113`），
每层 2 次 slice × 28 层 = 56 次 readback/token。

## What Worked

**trait device method + Metal override（`mlx_slice` + `mlx_contiguous`）
+ ops dispatcher 模板。**

1. `crates/autograd/src/backend.rs` — 新增 `Backend::slice(
   x: &DeviceHandle, old_shape, starts, ends) -> Result<DeviceHandle>`，
   默认 readback → `cpu_slice` → upload fallback。`cpu_slice` 也新 pub
   到 backend.rs，作为数值 reference，device-default 和 host-path 共享。
2. `crates/autograd/src/backend_metal.rs` — override 走
   `mlx_slice(start, stop, strides=all 1)` 拿到一个非连续视图，再包
   `mlx_contiguous` 强制下一次 eval 时在窗口布局下物化。**关键**：跟
   M5.3b.12 transpose 一模一样的 view-materialization 陷阱 — `mlx_slice`
   返回的视图是 lazy 的元数据变换（偏移+stride），`mlx_array_data_float32`
   的裸指针 readback 忽略视图信息，读出原布局的 bytes。MLX 对已连续数组
   短路 `mlx_contiguous`，零开销；只有真正的非全切片（always）才付一次
   拷贝，且那次拷贝正是我们想要的（与老 host_eager 输出 byte-identical）。
3. `crates/autograd/src/ops/layout.rs` — `slice` 拆成 dispatcher +
   `slice_device_lazy` + `slice_host_eager`，dispatch 门同 reshape /
   transpose / rope 的 `device_handle.is_some() && dirty != Host`。
4. `crates/autograd/src/ops.rs` — 公共入口剥掉 `ensure_host`。

## Rule

**`mlx_slice` 返回的是 lazy 视图，readback 必须 `mlx_contiguous` 包一层。**
和 `mlx_transpose_axes` 是同一族 trap — MLX 把 slice 当作 stride/offset
层面的元数据变换，返回的 `mlx_array*` 指向的 buffer 还是原数组的，
`mlx_array_data_float32` 读到的是原布局的 byte range。下游如果再送 FFI
（`mlx_matmul`、`mlx_multiply` 等）是安全的（MLX 自己处理视图语义），
但 readback 不行。**"wrap contiguous" 的成本只在真实物化时出现**，MLX
对已连续 array 短路，所以常见 case 是零开销；不常见 case（slice
followed by readback，比如我们的测试）付一次 copy，换来正确性。

**复合 op 的 dispatcher 和 inner op 的 dispatcher 对称。** `ops/layout.rs`
的 `slice`/`reshape`/`transpose` 都按 `device_handle.is_some() &&
dirty != Host` 分流，ops.rs 公共入口就只负责 "不在入口 flush"。这个
分工保证了 `Dirty::Host` 的 host 输入走 host 路径（CPU 后端或单元测试
场景），`Dirty::Device`/`Dirty::Both` 的输入沿着 MLX graph 下游走。

## Verification

Mac local:

```bash
cargo check  -p autograd --features metal --release
cargo check  -p autograd --release                        # CPU-only; cpu_slice fallback
cargo test   -p autograd --features metal --release --test test_device_handle metal_slice
cargo test   -p autograd --features metal --release       # 22 tests (+1), all green
cargo test   -p train    --features metal --release       # 30 groups, all green
cargo clippy -p autograd --features metal --release --tests -- -D warnings \
    -A clippy::arc_with_non_send_sync -A clippy::type_complexity
```

全绿；新增 `metal_slice_forward_stays_lazy` 跑 `matmul → reshape →
slice(q) + slice(gate) → sum` 链，前向 0 eval、两次 backward ≤6 eval，
并对 q/gate 窗两路都 CPU 参考对拍 ≤ 1e-4。clippy 两条 `-A` 为预存遗留。

pending-remote (train box, Qwen3.5-4B):

- 训练步 `METAL_EVAL_COUNT` 对拍 pre-M5.3b.16 commit vs 当前 commit，
  验证 attention 路径每层 q/gate 两次 slice × 28 = 56 evals/token 消除。
- `cargo run -p train --release --bin train_multi_turn --features metal
  -- --backend metal --iters 30` 数值稳定性 vs M5.3b.15 baseline 无漂移。

## Cross-links

- Plan row: `docs/plans/rust-agent-rl-single-node.md` M5.3 — M5.3b.16
  entry 已添加。
- 前置: `docs/experience/wins/2026-04-21-lazy-reshape-transpose-metal.md`
  (M5.3b.12, 同 view-materialization 陷阱模式);
  `2026-04-21-lazy-causal-sdpa-metal.md` (M5.3b.15, 复合 op 收尾模板).
- 姐妹 M5.3b 模板全集见 M5.3b.1–15 各 wins 条目。
