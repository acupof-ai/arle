# Elementwise `mul` lazy forward on Metal（M5.3b.17）

status: pending-remote (Mac-local parity + eval-count assertions pass;
end-to-end Qwen3.5 `METAL_EVAL_COUNT` reduction must be re-measured on
the training box)

## Title

attention gate 融合 `attn * gate` + MLP SwiGLU `silu(gate) * up` 驻留 MLX
lazy graph — 跟 `add_broadcast` 同族的 OR-dispatch 模板

## Context

M5.3b.1–16 把 forward pass 的几乎所有 ops 都在 Metal 上 lazy 化了，
但 elementwise `mul` 仍走 host-eager：`ops::mul` 入口 `ensure_host(a)?+
ensure_host(b)?` 把两条上游产线都 flush 回 host 再做元素乘。Qwen3.5 每
attention 层有一处 gated multiply（`attn * sigmoid(gate)`），每 MLP
层有一处 SwiGLU activation（`silu(gate) * up`），共 2 次/层 × 28 层 =
56 次 evals/token 潜在损失。

## What Worked

**trait device method + Metal override（`mlx_multiply`）+ OR-dispatch
模板（同 `add_broadcast`）。**

1. `crates/autograd/src/backend.rs` — 新增 `Backend::mul(a, b, shape) ->
   DeviceHandle`，默认 readback→`mul_forward`→upload fallback。
2. `crates/autograd/src/backend_metal.rs` — override 直接 `mlx_multiply(
   a_handle, b_handle)`。形状由 caller 保证相等（`ops::mul` 的老契约）
   所以不需要 broadcast 处理。单个 FFI 调用。
3. `crates/autograd/src/ops/elementwise.rs` — `mul` 拆成 OR-lazy
   dispatcher + `mul_device_lazy` + `mul_host_eager`。跟
   `add_broadcast` 同样的 **OR（不是 AND）** 理由：hot path 是两条都
   Dirty::Device（`attn`/`gate` 皆从上游 matmul/sigmoid 链出），若 AND
   会因"只有一个 lazy"就走 host_eager，反噬 matmul 的 lazy 图。
4. `crates/autograd/src/ops.rs` — 公共入口剥掉 `ensure_host(a)+
   ensure_host(b)`。

## Rule

**Elementwise `mul` 的 dispatch 是 OR-lazy — 与 `add_broadcast` 的教训
同构。** 一旦 attention/MLP 里有混合 host 常量的 elementwise 二元 op，
也应套同样模板；但 Qwen3.5 里 `mul` 的两侧几乎永远都 Dirty::Device，
所以 OR 和 AND 在实际路径上同性。OR 作为保险丝是写对的默认，因为它
永不退化（upload 小端比 readback 大端廉价）。

**`.clone()` 在 backward 路径仍安全，因为 `tape.backward` pre-walks
`flush_to_host_batch`。** `mul_backward` 用 `SavedContext::Tensors([a,b])`
保存输入，之后 `a_tensor = store.tensor(a)?.clone()` 依赖 `.clone()`
的 `dirty != Device` 断言。但 tape.backward 第 161 行 pre-walk 把所有
entry `output_id`（包括 `a`/`b` 作为上游 op 输出的那些）批量 flush 到
`Dirty::Both`，所以 backward 里的 `.clone()` 不会 panic。不需要额外改
backward。

## Verification

Mac local:

```bash
cargo check  -p autograd --features metal --release
cargo test   -p autograd --features metal --release --test test_device_handle metal_mul_forward
cargo test   -p autograd --features metal --release       # 23 tests (+1), all green
cargo test   -p train    --features metal --release       # 30 groups, all green
cargo clippy -p autograd --features metal --release --tests -- -D warnings \
    -A clippy::arc_with_non_send_sync -A clippy::type_complexity
```

全绿；新增 `metal_mul_forward_stays_lazy` 跑 `matmul(a*wa) * matmul(b*wb)
→ sum` 链，前向 0 eval、backward ≤6 eval，CPU 参考对拍 ≤ 1e-4。

pending-remote (train box, Qwen3.5-4B):

- 训练步 `METAL_EVAL_COUNT` 对拍 pre-M5.3b.17 commit vs 当前 commit，
  验证 attention `attn*gate` + MLP SwiGLU `silu(gate)*up` 2 × 28 = 56
  evals/token 消除。
- `cargo run -p train --release --bin train_multi_turn --features metal
  -- --backend metal --iters 30` 数值稳定性 vs M5.3b.16 baseline 无漂移。

## Cross-links

- Plan row: `docs/plans/rust-agent-rl-single-node.md` M5.3 — M5.3b.17
  entry 已添加。
- 前置: `docs/experience/wins/2026-04-21-lazy-add-broadcast-metal.md`
  (M5.3b.14, OR-dispatch 模板的原教训)，
  `2026-04-21-lazy-mul-scalar-metal.md` (M5.3b.13, `mlx_multiply` +
  scalar 的姐妹 case).
- 姐妹 M5.3b 模板全集见 M5.3b.1–16 各 wins 条目。
