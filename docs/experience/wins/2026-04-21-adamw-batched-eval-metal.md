# AdamW 终端 eval 批量化（M5.3b.11）

status: pending-remote (Mac-local structural assertions pass; end-to-end
Qwen3.5 `METAL_EVAL_COUNT` reduction must be re-measured on the training
box)

## Title

`AdamW::step_device` 一次 `backend.eval` 覆盖所有 params — per-step eval
cost 从 O(num_params) 降到 O(1)

## Context

M5.3b.10 把 AdamW 的公式组合进 MLX lazy graph，但 `backend_metal::adamw_step`
仍然在每次调用的末尾自触发 `mlx_eval([new_param, new_m, new_v]) +
bump_eval_count()`。单 param 场景 1 eval/step，没问题；但 Qwen3.5 级模型
~200 trainable params，每个 optimizer step 就是 ~200 次 `mlx_eval`，足以
把 M5.3b.1–10 十步优化攒下的收益全部消耗掉。

## What Worked

**契约移交：让 caller 批量 eval，而不是每个 op 自己 eval。**

1. `crates/autograd/src/backend_metal.rs` — `adamw_step` override 删掉终端
   `mlx_eval + bump_eval_count`，直接返回三个 `MlxHandle` 包住的图节点，
   docstring 指向新契约。
2. `crates/autograd/src/backend.rs` — trait-level `adamw_step` docstring
   增一段 "Eval contract (M5.3b.11)" 明确 *实现 MUST 返回未 eval 的
   handle；caller 负责批量 eval；独立 per-param 链无共享子节点，批量安全*。
3. `crates/autograd/src/optim.rs` — `AdamW::step_device` 里新增
   `pending_eval: Vec<DeviceHandle>`，每个 param 的 `(new_param, new_m,
   new_v)` Arc 克隆塞进去（`DeviceHandle::Metal(MlxHandle(Arc<…>))` 克隆是
   ref-count，开销可忽略），循环后如果非空就一次 `backend.eval(&refs)`。
   CPU 默认 `Backend::eval` no-op 行为不变。
4. `crates/autograd/tests/test_device_handle.rs` — 新增
   `metal_adamw_step_batches_eval_across_params`：N=1 vs N=8 两次测量，
   断言 `eight <= one + 1`，即 8-param 的 eval 数不能线性涨。实测 N=1=1、
   N=8=1；既有 `metal_adamw_step_stays_device_resident` 单 param 路径
   per-step delta ≤ 2 仍通过。

## Rule

**Lazy-graph op 的 eval 契约要在 trait 层显式声明，不能埋在某个实现里。**
一旦某个 override "顺手" 在结尾 eval，会强制 API 语义变成 *每 op 一次
eval*，外层 caller 再怎么批也补救不回来。M5.3b.1–9 从来不自己 eval；
M5.3b.10 的 adamw_step 是漏网之鱼，M5.3b.11 把它纳入同一纪律。

**独立 per-param 更新链 → 一次 eval。** MLX 的 lazy 图只要子节点不共享
就可以任意并拢；AdamW 每个参数的 m/v 更新互相独立，整组 `Vec<&Handle>`
塞进 `mlx_eval` 一次评估是安全、正确、便宜的。

## Verification

Mac local:

```bash
cargo check -p autograd --features metal --release
cargo test  -p autograd --features metal --release --test test_device_handle metal_adamw
cargo test  -p autograd --features metal --release
cargo clippy -p autograd --features metal --release --tests -- -D warnings \
    -A clippy::arc_with_non_send_sync -A clippy::type_complexity
```

全绿；新增 `metal_adamw_step_batches_eval_across_params` 通过；既有 47
个 autograd 测试无回归。clippy 两条 `-A` 为预存遗留项（非本 diff 引入，
stash-pop 验证过）。

pending-remote (train box, Qwen3.5-4B):

- 用 `scripts/bench_guidellm.sh` 以 pre-M5.3b.11 commit 与当前 commit
  各跑一次单训练步的 `METAL_EVAL_COUNT`，验证 ~200-param 模型的
  AdamW 部分从 ~200 evals/step 降到 1 eval/step。
- 训练 smoke（`cargo run -p train --release --bin train_multi_turn
  --features metal -- --backend metal --iters 30`）数值稳定性 vs
  M5.3b.10 baseline 无漂移。

## Cross-links

- Plan row: `docs/plans/rust-agent-rl-single-node.md` M5.3 — M5.3b.11
  entry 已添加。
- 前置: `docs/experience/wins/2026-04-21-adamw-on-device-metal.md`
  (M5.3b.10).
