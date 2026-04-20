# `causal_sdpa` composite op lazy forward on Metal（M5.3b.15）

status: pending-remote (Mac-local parity + eval-count assertions pass;
end-to-end Qwen3.5 `METAL_EVAL_COUNT` reduction must be re-measured on
the training box)

## Title

attention composite op 端到端驻留 MLX lazy graph — inner 已全 lazy 后的
收尾 strip，每层 attention 从 token ids 一路保留在设备上

## Context

M5.3b.1–14 把 `causal_sdpa` 的 body 里每一个 inner op 单独上了设备：
reshape / transpose (M5.3b.12)、matmul (M5.3a.2)、mul_scalar (M5.3b.13)、
add_broadcast (M5.3b.14)、softmax (M5.3b.2)、另一个 matmul、最后 reshape。
但 `ops::causal_sdpa` 的公共入口仍有三个 `store.ensure_host(q)?` /
`ensure_host(k)?` / `ensure_host(v)?`，把 Linear 投影刚 matmul 出来的
q/k/v flush 回 host 再传给已经 lazy 的 body —— 等于入口把所有下游收益
抹掉。Qwen3.5 × 28 层每层都走这条路径。

## What Worked

**Strip-only diff。** Body 是纯 dispatch，inner ops 已全部按
Dirty-dispatch 模板开好了 lazy 分支，公共入口只需要不再 `ensure_host`：

```rust
pub fn causal_sdpa(
    q: TensorId, k: TensorId, v: TensorId,
    store: &mut TensorStore, tape: &mut Tape,
) -> Result<TensorId> {
    // Body 纯 dispatch，inner 已全 lazy；stripping ensure_host 让 attention
    // 链端到端驻留 MLX 图内。
    attention::causal_sdpa(q, k, v, store, tape)
}
```

## Rule

**复合 op 的最后一步是"不做什么"。** 当 body 的每一个 inner op 都按
Dirty-dispatch 正确处理了 `Dirty::Device` 输入后，复合 op 的公共入口
只剩一件事：**不要在入口 flush**。多一个 `ensure_host(q)?` 就等于一笔
勾销下游所有 inner op 的 lazy 收益 —— inner 的 lazy 分支看到的是被
flush 过的 `Dirty::Host` 输入，直接走 host_eager，MLX 图不复存在。

这条的对称原则：**composite op 的入口不应再有 residency 决策**，应由
inner op 自己的 dispatcher 负责，因为 inner 才知道自己能不能在设备上做
（CPU 后端的 MLX 专用 kernel 不存在，fallback 到默认 readback→host→
upload trait 方法）。

## Verification

Mac local:

```bash
cargo check  -p autograd --features metal --release
cargo test   -p autograd --features metal --release --test test_device_handle metal_causal_sdpa
cargo test   -p autograd --features metal --release          # 21 tests, all green (+1 vs M5.3b.14)
cargo test   -p train    --features metal --release          # 36 tests, all green
cargo clippy -p autograd --features metal --release --tests -- -D warnings \
    -A clippy::arc_with_non_send_sync -A clippy::type_complexity
```

全绿；新增 `metal_causal_sdpa_forward_stays_lazy` 断言复合 op forward
eval_count=0、backward ≤6、`sum` 后仍 0（sum_all 也 lazy）。clippy 两条
`-A` 为预存遗留。

pending-remote (train box, Qwen3.5-4B):

- 训练步 `METAL_EVAL_COUNT` 对拍 pre-M5.3b.15 commit vs 当前 commit，
  验证 attention 路径从 3 × 28 = 84 evals/token（q/k/v ensure_host）
  降到 0 evals/token。
- `cargo run -p train --release --bin train_multi_turn --features metal
  -- --backend metal --iters 30` 数值稳定性 vs M5.3b.14 baseline 无漂移。

## Cross-links

- Plan row: `docs/plans/rust-agent-rl-single-node.md` M5.3 — M5.3b.15
  entry 已添加。
- 前置: `docs/experience/wins/2026-04-21-lazy-add-broadcast-metal.md`
  (M5.3b.14), `2026-04-21-lazy-mul-scalar-metal.md` (M5.3b.13),
  `2026-04-21-lazy-reshape-transpose-metal.md` (M5.3b.12) 完成了 body
  内每个 inner op 的 lazy 化。
- 姐妹 M5.3b 模板全集见 M5.3b.1–14 各 wins 条目。
