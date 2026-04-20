# Elementwise `sigmoid` lazy forward on Metal（M5.3b.18）

status: pending-remote (Mac-local parity + eval-count assertions pass;
end-to-end Qwen3.5 `METAL_EVAL_COUNT` reduction must be re-measured on
the training box)

## Title

attention gate `gate = sigmoid(gate_proj)` 驻留 MLX lazy graph — 跟
`exp` / `silu` 同族的单 FFI activation 模板

## Context

M5.3b.1–17 已经把 attention/MLP forward pass 的大部分 ops 驻留在 Metal
lazy 图里，但 `sigmoid` 仍然走 host-eager：`ops::sigmoid` 入口
`ensure_host(x)?` 把上游产出 flush 回 host 再做 sigmoid。Qwen3.5 在
attention gated-multiply 之前每层有一处 `gate = sigmoid(gate_full)`
（`crates/train/src/qwen35.rs:142`），共 28 层 = 28 evals/token 潜在损失。
修复后 `q_full → slice(gate) → sigmoid → mul(attn, ·)` 整条链可以端到
端驻留设备。

## What Worked

**trait device method + Metal override（`mlx_sigmoid`）+ Dirty-dispatch
模板（同 `exp` / `silu`）。**

1. `crates/autograd/src/backend.rs` — 新增 `Backend::sigmoid_forward(a)
   -> Vec<f32>`（默认走 `cpu_sigmoid_forward`）+ `Backend::sigmoid(x,
   shape) -> DeviceHandle`（默认 readback→`sigmoid_forward`→upload
   fallback）。
2. `crates/autograd/src/backend_metal.rs` — override `sigmoid` 直接
   `mlx_sigmoid(x_handle)` 单 FFI 调用，返回驻留 handle（与 `exp`
   同构）。
3. `crates/autograd/src/ops/activation.rs` — `sigmoid` 拆成 Dirty
   dispatcher + `sigmoid_device_lazy` + `sigmoid_host_eager`。dispatch
   条件 `device_handle.is_some() && dirty != Host`，覆盖 Dirty::Device
   和 Dirty::Both（ensure_device 后重入场景）。
4. `crates/autograd/src/ops.rs` — 公共入口剥掉 `ensure_host(x)`，加
   M5.3b.18 说明注释。

## Rule

**`sigmoid_backward` 用 `SavedContext::SigmoidCtx { y: output_id }` 保
存输出 + `.clone()` 读 y，安全性依赖 `tape.backward` pre-walk
`flush_to_host_batch` 把所有 entry output 批量置 `Dirty::Both`。** 同
M5.3b.4 `exp` / M5.3b.17 `mul` 的 saved-output/saved-input 路径一致，
不需要额外改 backward。

**Qwen3.5 attention gate 现在端到端 lazy。** q_proj 输出的
`q_full[batch,seq,heads,head_dim*2]` → `slice(gate)` (M5.3b.16 lazy) →
`sigmoid(gate)` (M5.3b.18 lazy) → `mul(attn, gate)` (M5.3b.17 lazy)
一条链都在 MLX 图内，不再有 per-layer readback。

## Verification

Mac local:

```bash
cargo check  -p autograd --features metal --release
cargo test   -p autograd --features metal --release --test test_device_handle metal_sigmoid_forward
cargo test   -p autograd --features metal --release       # 24 tests (+1), all green
cargo test   -p train    --features metal --release       # 30+ groups, all green
cargo clippy -p autograd --features metal --release --tests -- -D warnings \
    -A clippy::arc_with_non_send_sync -A clippy::type_complexity
```

全绿；新增 `metal_sigmoid_forward_stays_lazy` 跑 `matmul(x·w) → sigmoid
→ sum` 链，前向 0 eval、backward ≤3 eval，CPU 参考对拍 ≤ 1e-4。

pending-remote (train box, Qwen3.5-4B):

- 训练步 `METAL_EVAL_COUNT` 对拍 pre-M5.3b.18 commit vs 当前 commit，
  验证 attention gate `sigmoid(gate)` × 28 层 = 28 evals/token 消除。
- `cargo run -p train --release --bin train_multi_turn --features metal
  -- --backend metal --iters 30` 数值稳定性 vs M5.3b.17 baseline 无漂移。

## Cross-links

- Plan row: `docs/plans/rust-agent-rl-single-node.md` M5.3 — M5.3b.18
  entry 已添加。
- 前置/姐妹: `2026-04-21-lazy-mul-metal.md` (M5.3b.17, attention
  `attn * gate` 的乘法侧), `2026-04-20-lazy-silu-metal.md` /
  `2026-04-20-lazy-exp-metal.md` (activation 单 FFI 模板原型)。
- 姐妹 M5.3b 模板全集见 M5.3b.1–17 各 wins 条目。
