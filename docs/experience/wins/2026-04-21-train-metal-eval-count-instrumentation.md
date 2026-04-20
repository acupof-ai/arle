# train_multi_turn Metal per-step eval-count instrumentation（baseline reveal）

status: landed local (Mac Qwen3.5-style tiny 2-layer fixture)；Qwen3.5-4B
pending-remote

## Title

`train_multi_turn` 每步 Metal `METAL_EVAL_COUNT` 按阶段拆分记录（fwd /
bwd / opt），M5.3b.1–19 forward-path lazy 实测值 + backward chokepoint
量化

## Context

M5.3b.1–19 shipped lazy forward coverage for every op in the Qwen3.5
forward hot path (embedding, rmsnorm, matmul, rope, softmax, log_softmax,
silu, sigmoid, gelu, exp, slice, reshape, transpose, mul, mul_scalar,
add, add_broadcast, gather_last_dim, mean, sum)，同时 M5.3b.10–11 把
AdamW 打成 device-resident 且 terminal eval 批量化。但到 2026-04-21 为
止没有任何 binary 实测 per-step eval count —— `test_device_handle.rs`
只覆盖单 op 链（≤3 evals），真实 training step 是否 scale 未知，
backward-path 该不该投资也没依据。

这一条 commit 给 `train_multi_turn` 加了 cfg-guarded 的 Metal
`METAL_EVAL_COUNT` 阶段化采样（forward / backward / optimizer），让
任何 Mac 本地或训练机都能一眼看到瓶颈在哪。非 Metal 构建 snapshot 返
回 `None`，metric 字段直接省略 —— 对 CPU/CUDA 路径零侵入。

## What Worked

**instrumentation-first over optimization-first**。在投资 backward-path
overhaul 之前先加纯观测，拿到实测数再决策。

1. `crates/train/src/bin/train_multi_turn.rs` — 加两个 cfg-guarded fn：
   `metal_eval_reset()`（metal 下调 `autograd::backend_metal::
   reset_eval_count`，非 metal 下 no-op）+ `metal_eval_snapshot() ->
   Option<u64>`（metal 下 `Some(eval_count())`，非 metal 下 `None`）。
2. 训练循环在 rollout 后 / loss 前 reset_count + snapshot start；loss
   读回 host 后 snapshot after_forward；`tape.backward` 完 snapshot
   after_backward；`optimizer.step` 完 snapshot after_step。
3. Metric emit 路径从固定数组改成 `Vec<(&str, f64)>`，conditionally
   push `metal_evals_fwd / _bwd / _opt` delta；stdout sink 自动按
   `key=value` 格式吐出，jsonl sink 同样兼容。

## Rule

**"Measure before overhaul"**。Backward-path device port 是
architectural-change-level commitment（改 `Backend::matmul_backward` +
`flush_to_host_batch` 调度 + `accumulate_grad` 设备感知 + 所有 backward
op 支持 Dirty::Device grad）；在没拿到 per-step eval count 的实测数之
前，不清楚这工是否值得。实测一跑出来决策立刻清晰：

**Mac 本地 2-layer Qwen3.5-style tiny fixture 实测（iter=3，group=2，
turns=1，seq=8，d_model=32）：**

| 阶段 | evals/step | 占比 |
|------|-----------|-----|
| forward (rollout + ref_model + kl + loss) | 6 | 5% |
| backward (`tape.backward` 全程) | **113** | **95%** |
| optimizer (AdamW step) | 0 | 0% |

M5.3b.1–19 的 forward-path 压到 6 evals/step（rollout 两个 episode +
ref_model forward + final loss），M5.3b.10–11 的 AdamW 批量化做到 0
eval/step（step 内部批一次 terminal eval，reset 窗口正好盖住）。
backward 是唯一的巨块，对照 2-layer 模型的 ~14 个 matmul（q/k/v/o/
gate/up/down × 2 layer）× 2 evals/matmul = 28，加 group_size=2 + ref
backward + kl 共 ~113，跟理论模型吻合。

Qwen3.5-4B（28 层）按同比例外推 backward ≈ 1500 evals/step。**这是接
下来做 `matmul_backward` 设备版本 / backward-path 设备化 的定量依据**
—— 理论上限能把 95% 的 per-step eval 压到 1，但需要协调 matmul_backward
+ `flush_to_host_batch` 调度 + `accumulate_grad` 路径。

## Verification

Mac local:

```bash
cargo build -p train --release --features metal --bin train_multi_turn
cargo build -p train --release                                         # non-metal still builds
cargo test  -p train --release --features metal                        # 36+ groups all green
target/release/train_multi_turn --backend metal --iters 3 \
  --group-size 2 --turns 1 --prompt-len 4 --agent-tokens 4 \
  --obs-tokens 0 --d-model 32 --n-layers 2 --n-heads 2 --d-head 16 \
  --d-ff 64 --vocab 128
# step=1 loss=0.000000 ... metal_evals_fwd=6 metal_evals_bwd=113 metal_evals_opt=0
# step=2 loss=-0.000001 ... metal_evals_fwd=6 metal_evals_bwd=113 metal_evals_opt=0
# step=3 loss=-0.000001 ... metal_evals_fwd=6 metal_evals_bwd=113 metal_evals_opt=0
```

pending-remote (train box, Qwen3.5-4B):

- 跑 `target/release/train_multi_turn --backend metal --iters 10 --save-path
  /tmp/qwen35-out`（或实际训练 config），记录 `metal_evals_fwd / _bwd /
  _opt`，对拍 2-layer tiny fixture 线性外推的 ~1500 bwd / step。
- 确认 per-step wall-clock 比例与 eval count 比例一致（backward 占 wall
  的百分比应 ≈ 95%，如果不是，说明还有 non-eval 开销 dominant）。

## Cross-links

- Plan row: `docs/plans/rust-agent-rl-single-node.md` M5.3 残余条款 "剩
  余：eval_count 目标收敛到严格 1 per step"，这一条把 residual 的定量
  落地。
- 前置全集：M5.3b.1–19 各 forward-path lazy wins（`2026-04-20-lazy-
  sum-metal.md` 起到 `2026-04-21-lazy-mean-metal.md` 止）。
- 后续候选：`matmul_backward` 设备版本（大概率 M5.3b.20 单 op 单 FFI
  override + 默认 readback→host→upload fallback，与 forward 的
  M5.3b.12/.13/.14 同型），但需要配合 `accumulate_grad` 调度才能落到
  strict 1 eval/step —— 实测 backward=113 的 95% 占比是这笔投资的依据。
