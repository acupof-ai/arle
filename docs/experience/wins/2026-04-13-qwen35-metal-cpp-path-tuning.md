# 2026-04-13 · Qwen3.5 Metal C++ Path Tuning

## Goal

继续把 `mlx-community/Qwen3.5-4B-MLX-4bit` 的 Metal 路径往 `mlx_lm` 推近，并保留清晰的实验记录。

这轮工作的重点不是再猜测 Rust/FFI 开销，而是直接在已经确认启用的 `C++ full generate` 路径上做 A/B。

## Confirmed Runtime Path

- 默认 benchmark 路径已经走 `CppQwen35Model::generate()`
- 运行时日志确认是 `Metal forward path: C++ full generate (all in C++)`
- 因此本轮优化的收益和回归，都来自 `infer/mlx-sys/src/mlx_qwen35_model.cpp`

## Competitor Alignment Notes

这轮重新对照了本机 `mlx_lm 0.31.2`：

- `mlx_lm` 的 Qwen3.5 已经是模型专用路径，不是 generic Qwen fallback
- `mlx_lm` 的 prompt prefill 仍然返回整段 `logits[:, -1, :]` 之前的完整 logits 张量
- 所以“我们可能在 prefill 多算了整段 lm_head，而 Python 只算最后一个 token”不是当前 gap 的解释

结论：剩余 prompt gap 仍应优先在 GDR / prefill hot path 里找，而不是 lm_head 形状假设。

## A/B Matrix

工作负载：

- model: `mlx-community/Qwen3.5-4B-MLX-4bit`
- prompt/generation: `128 / 128`
- warmup/runs: `1 / 2`

四组组合：

| GDR attention proj | GDR MLP | Prompt TPS | Gen TPS | TTFT | Repo E2E |
|---|---|---:|---:|---:|---:|
| separate | separate | 564.9 | 66.7 | 226.6 ms | 59.63 |
| separate | merged | 610.1 | 73.7 | 209.8 ms | 65.73 |
| merged | separate | 604.8 | 73.5 | 211.6 ms | 65.57 |
| merged | merged | 605.2 | 72.9 | 211.5 ms | 65.07 |

结论：

- 不能简单照搬“全部 merged”
- 最优默认组合是：
  - GDR attention projections 保持 `separate`
  - GDR MLP 默认切回 merged `gate_up`

## Code Changes

### 1. Split `separate proj` and `separate MLP` into independent switches

新增两个开关，避免把 attention-side 和 MLP-side 的收益绑死在一起：

- `AGENT_INFER_QWEN35_CPP_SEPARATE`
- `AGENT_INFER_QWEN35_CPP_SEPARATE_MLP`

默认值：

- projection: `on`
- MLP: `off`

即默认走 `separate proj + merged MLP`。

### 2. Add C ABI for separate-MLP only

之前 `qwen35_compiled_set_separate_proj()` 同时打开了：

- `use_separate_proj`
- `use_separate_mlp`

现在拆出独立的：

- `qwen35_compiled_set_separate_proj()`
- `qwen35_compiled_set_separate_mlp()`

这样可以做干净的四象限 A/B。

### 3. Fuse two gate hot paths with compiled helpers

把两个高频小 elementwise 串收进 `mlx::core::compile(..., shapeless=true)` helper：

- full-attention output gate: `sigmoid(gate) * attn_out`
- GDR output gate: `silu(z) * normed`

目的不是改变数学，而是减少 decode/prefill 中的小 kernel 数量。

### 4. Precompute `-exp(A_log)` once at load time

`compute_g` 原来每层每步都做一遍：

- `exp(A_log)`
- `negative(...)`

现在在加载 GDR layer 时直接把 `A_log` 转成 `neg_exp_a = -exp(A_log.f32)`，运行时只做：

- `exp(neg_exp_a * softplus(...))`

这是一个小优化，但只作用在真正的 hot path 上。

## Final Result

最终默认路径再次 benchmark：

- `load_ms`: `469.4`
- `ttft_ms`: `199.5`
- `prompt_tps`: `641.5`
- `generation_tps`: `81.3`
- `repo_e2e_tps`: `72.17`
- `peak_rss_mb`: `2508`

对比本轮起点：

- `prompt_tps`: `564.9 -> 641.5`  `+13.6%`
- `generation_tps`: `66.7 -> 81.3` `+22.0%`
- `repo_e2e_tps`: `59.63 -> 72.17` `+21.0%`
- `ttft_ms`: `226.6 -> 199.5` `-11.9%`

## Python Baseline (same 128/128 setup)

本机 `mlx_lm 0.31.2`：

- `load_ms`: `1564.4`
- `prompt_tps`: `749.1`
- `generation_tps`: `79.9`
- `repo_e2e_tps`: `72.24`

当前结论：

- decode 已经小幅超过 `mlx_lm`
- repo 口径的端到端吞吐基本打平
- 剩余差距主要集中在 prefill / TTFT

## Verified

- `cargo test --release -p infer --no-default-features --features metal,no-cuda --bin metal_bench`
- `./target/release/metal_request --model mlx-community/Qwen3.5-4B-MLX-4bit --prompt 'benchmark throughput' --raw-prompt --warmup 0 --max-new-tokens 16 --ignore-eos`

## Follow-up Experiments

### Rejected for now: compile the whole GDR sublayer

后续尝试过把单层 GDR attention 子图单独包进 `mlx::core::compile(...)`，目标是继续减少 prefill / decode 的 command-buffer 数量。

这条路当前不适合作为默认优化：

- `shapeless=true` 时，`split(...)` 会直接卡在 output-shape inference
- 改成 shape-specialized compile 虽然能跑，但还没有拿到稳定、干净、可复现的正收益
- 官方 `mlx_lm` 的做法也更保守：优先 compile 小 helper 和 recurrent 核心，而不是把整层都丢给 compile

当前结论：

- 继续保留 `compute_g` / gate / swiglu 这类小型 compiled helper
- 不把 `compiled GDR sublayer` 留在默认路径里
- 之后如果重试，必须只在 env 开关下做，不污染默认 benchmark 路径

### Benchmark hygiene: clean build is mandatory

这轮还确认了一个很关键的 benchmark hygiene 规则：

- 改过 `infer/mlx-sys/src/mlx_qwen35_model.cpp` 之后，不能直接相信增量构建出来的 A/B 数据
- 同一份源码，在脏 `target/` 下会出现明显偏低的 Metal benchmark
- `cargo clean -p infer -p mlx-sys` 后重建，结果会明显回归正常区间

建议固定流程：

```bash
cargo clean -p infer -p mlx-sys
cargo build --release -p infer --no-default-features --features metal,no-cuda --bin metal_bench
./target/release/metal_bench --model mlx-community/Qwen3.5-4B-MLX-4bit \
  --prompt-tokens 128 --generation-tokens 128 --warmup 1 --runs 2 --json
```

### Kept: trim prefill-only intermediates

`C++ full generate` 路径原来会在 `prefill` 和 `decode` 两个阶段都把大量 intermediate arrays 保活到下一步，目的是复用 GPU buffer。

这轮做了更窄的策略：

- `decode (S=1)` 继续保留原有的全量保活
- `prefill (S>1)` 默认不再保活整步 intermediates
- 需要回退时可用 `AGENT_INFER_QWEN35_CPP_KEEP_PREFILL_INTERMEDIATES=1`

同一份 clean build、同一台机器、`128 / 128 / warmup 1 / runs 2` 的对照：

- trim prefill intermediates (default): `prompt 603.26 tok/s`, `generation 74.26 tok/s`, `repo_e2e 66.12 tok/s`, `TTFT 212.18 ms`
- keep prefill intermediates: `prompt 603.23 tok/s`, `generation 73.37 tok/s`, `repo_e2e 65.42 tok/s`, `TTFT 212.20 ms`

结论：

- prompt / TTFT 基本不变
- decode / repo-e2e 有约 `+1%` 的小收益
- 这是低风险改动，可以保留为默认策略

### Kept: align C++ generate with `mlx_lm` cache clearing

联网重新对照官方 `mlx_lm` 之后，确认它在 `generate.py` 里会做两件事：

- 每个 prefill chunk 完成后 `mx.clear_cache()`
- decode 过程中每 `256` 步再清一次 cache

我们原来的 `Qwen3.5` C++ full-generate 路径没有这两个点。现在补齐为默认行为，并保留环境变量：

- `AGENT_INFER_QWEN35_CPP_CLEAR_CACHE=0`

同一份 clean build、`128 / 128 / warmup 1 / runs 2` 的 A/B：

- clear-cache on (default): `prompt 578.62 tok/s`, `generation 68.37 tok/s`, `repo_e2e 61.14 tok/s`, `TTFT 221.29 ms`
- clear-cache off: `prompt 594.93 tok/s`, `generation 65.41 tok/s`, `repo_e2e 58.93 tok/s`, `TTFT 215.29 ms`

单看这组综合 workload：

- decode / repo-e2e 提升更明显
- TTFT / prompt 略有回退

为了确认不是单一 workload 偶然噪音，又拆了两个极端 profile：

- `1 / 128`:
  - on: `generation 74.23 tok/s`, `repo_e2e 73.21 tok/s`
  - off: `generation 70.92 tok/s`, `repo_e2e 70.03 tok/s`
- `2048 / 1`:
  - on: `TTFT 3144.22 ms`, `prompt 651.35 tok/s`, `repo_e2e 0.3131 tok/s`
  - off: `TTFT 3346.75 ms`, `prompt 611.94 tok/s`, `repo_e2e 0.2959 tok/s`

结论：

- `clear_cache` 不是纯 decode 技巧，也不是纯 prefill 技巧
- 在当前 MLX / Metal allocator 行为下，它对两类 workload 的 `repo_e2e` 都是正收益
- 因此保留为默认路径，与官方 `mlx_lm` 的生成策略对齐

### Kept: prefill computes only last-token logits

进一步对照官方 `mlx_lm` 之后，确认它公开路径里并没有现成的“只算最后一位 logits”开关；默认模型调用仍然返回整段 logits。

这意味着如果要继续追 `prefill / TTFT`，不能只等上游 API，必须在我们自己的 `Qwen3.5` C++ 专用路径里做更激进的专门优化。

现在的做法是：

- `prefill (S>1)` 时，最终 `rms_norm` 之后只切出最后一个 position 再做 `lm_head`
- `decode (S=1)` 路径完全不变
- 需要回退时可用 `AGENT_INFER_QWEN35_CPP_PREFILL_LAST_LOGITS_ONLY=0`

同一份 clean build、`128 / 128 / warmup 1 / runs 2` 的对照：

- last-logits-only on (default): `prompt 687.52 tok/s`, `generation 74.18 tok/s`, `repo_e2e 66.95 tok/s`, `TTFT 186.18 ms`
- last-logits-only off: `prompt 595.93 tok/s`, `generation 66.04 tok/s`, `repo_e2e 59.45 tok/s`, `TTFT 214.80 ms`

极端 workload 也保持正收益：

- `1 / 128`:
  - on: `generation 74.94 tok/s`, `repo_e2e 73.79 tok/s`, `TTFT 26.65 ms`
  - off: `generation 63.72 tok/s`, `repo_e2e 62.86 tok/s`, `TTFT 27.67 ms`
- `2048 / 1`:
  - on: `prompt 725.76 tok/s`, `TTFT 2821.87 ms`, `repo_e2e 0.3492 tok/s`
  - off: `prompt 626.64 tok/s`, `TTFT 3268.24 ms`, `repo_e2e 0.3009 tok/s`

额外做了贪心生成等价性检查，同一 prompt 的正文输出一致。

结论：

- 这是目前最有效的一刀 prefill 优化
- 收益不是噪音，在综合 workload 和两种极端 workload 上都成立
- 因此保留为默认路径

### Rejected: slice before the final `rms_norm`

在 “prefill 只算最后一位 logits” 继续往下挖时，又试了一刀更激进的变体：

- 原方案：先对整段 hidden states 做最终 `rms_norm`，再只切最后一个 position 去做 `lm_head`
- 新尝试：先切最后一个 position，再只对这一位做最终 `rms_norm`

直觉上，新方案应该少算一整段 final norm；但 clean-build 复测并不支持这个判断。

同机、同一 build、`128 / 128 / warmup 1 / runs 3`：

- norm-then-slice (kept): `prompt 683.59 tok/s`, `generation 74.54 tok/s`, `repo_e2e 67.21 tok/s`, `TTFT 187.26 ms`
- slice-then-norm (rejected): `prompt 666.51 tok/s`, `generation 69.86 tok/s`, `repo_e2e 63.22 tok/s`, `TTFT 192.54 ms`

结论：

- 这不是“更少工作 = 更快”的简单关系
- 在当前 MLX / Metal 图优化行为下，提前切片反而让整体表现更差
- 因此最终保留 `norm-then-slice`，不把 `slice-before-final-rms_norm` 留在默认路径里

### Kept: phase-specific GDR MLP selection

继续沿着 `separate / merged` 方向往下挖时，发现真正该分开的不是 `final norm` 这类尾巴，而是 GDR block 里的 MLP 形式。

同一份 build，直接拆两个极端 workload 看：

- `decode-heavy (1 / 128)`：
  - `sep_proj + merged_mlp`: `generation 73.53 tok/s`, `repo_e2e 72.54 tok/s`
  - `sep_proj + sep_mlp`: `generation 74.63 tok/s`, `repo_e2e 73.62 tok/s`  ← 最优
  - `merged_proj + merged_mlp`: `generation 72.19 tok/s`, `repo_e2e 71.23 tok/s`
  - `merged_proj + sep_mlp`: `generation 71.10 tok/s`, `repo_e2e 70.12 tok/s`
- `prefill-heavy (2048 / 1)`：
  - `sep_proj + merged_mlp`: `TTFT 2869.14 ms`, `prompt 713.81 tok/s`  ← 最优
  - `sep_proj + sep_mlp`: `TTFT 2926.01 ms`, `prompt 699.97 tok/s`
  - `merged_proj + merged_mlp`: `TTFT 2976.49 ms`, `prompt 688.06 tok/s`
  - `merged_proj + sep_mlp`: `TTFT 2981.24 ms`, `prompt 686.96 tok/s`

结论很直接：

- `projection` 两个阶段都继续保持 `separate`
- 真正分裂的是 GDR `MLP`
  - `decode (S=1)` 更适合 `separate gate/up`
  - `prefill (S>1)` 更适合 merged `gate_up`

现在的默认路径改成：

- `projection`: 继续 `separate`
- `MLP`: 默认按 phase 切换
  - `decode (S=1)` → `separate`
  - `prefill (S>1)` → merged
- 如果显式设置 `AGENT_INFER_QWEN35_CPP_SEPARATE_MLP=0/1`，则继续按环境变量强制全局模式

对 `128 / 128 / warmup 1 / runs 3` 的综合 workload，三种模式对照如下：

- phase-specific default: `prompt 679.93 tok/s`, `generation 74.89 tok/s`, `repo_e2e 67.46 tok/s`, `TTFT 188.28 ms`
- force merged: `prompt 684.70 tok/s`, `generation 74.51 tok/s`, `repo_e2e 67.20 tok/s`, `TTFT 186.95 ms`
- force separate: `prompt 689.16 tok/s`, `generation 75.39 tok/s`, `repo_e2e 67.95 tok/s`, `TTFT 185.74 ms`

后续又用更长的 runs 重做了一轮 clean-build 复测，结论反而更简单：

- `128 / 128 / warmup 1 / runs 5`
  - merged: `prompt 681.31 tok/s`, `generation 74.09 tok/s`, `repo_e2e 66.82 tok/s`, `TTFT 187.91 ms`
  - separate: `prompt 686.32 tok/s`, `generation 74.67 tok/s`, `repo_e2e 67.35 tok/s`, `TTFT 186.51 ms`
- `1 / 128 / warmup 1 / runs 5`
  - merged: `generation 73.71 tok/s`, `repo_e2e 72.71 tok/s`
  - separate: `generation 73.72 tok/s`, `repo_e2e 72.72 tok/s`
- `2048 / 1 / warmup 1 / runs 3`
  - merged: `prompt 691.12 tok/s`, `TTFT 2963.30 ms`, `repo_e2e 0.3323 tok/s`
  - separate: `prompt 690.44 tok/s`, `TTFT 2966.22 ms`, `repo_e2e 0.3320 tok/s`

这说明先前看到的 “prefill 长序列更偏向 merged MLP” 并不够稳定，至少在当前代码和机器态下，不足以支撑更复杂的默认策略。

最终结论：

- 继续保留环境变量 `AGENT_INFER_QWEN35_CPP_SEPARATE_MLP=0/1` 作为强制开关
- 但默认策略从 “phase-specific” 再次简化为：
  - **global separate MLP**
- 原因不是审美，而是更长 runs 下它在 mixed workload 上小幅更优，在 decode-heavy / prefill-heavy 上也没有可测的负收益

### Kept: helper-level cleanup inside GDR hot path

在把默认 MLP 重新简化回 `global separate` 之后，又继续回到真正的 GDR 热区，做了两刀更小但更扎实的 helper-level 优化：

- `prefill (S>1)` 时，把
  - `beta = sigmoid(b_raw)`
  - `g = exp(neg_exp_a * softplus(a_raw + dt_bias))`
  收进一个 compiled helper，减少每层一个 elementwise launch 和一个临时张量
- 把 `q/k rms_norm + scale` 收进一个 compiled helper，让两次 GDR q/k 归一化不再从 host 侧分开调度
- `gdr` Metal kernel 用到的 `T` 标量改成每次 `forward()` 只创建一次，不再每层重建

这里刻意没有再去碰 `decode` 的数学路径：`g/beta` fused helper 只在 `prefill` 启用，`decode (S=1)` 保持旧算式，避免为了追 prompt 去拖慢 decode。

对比上一版默认（`39d95dc`）：

- `128 / 128 / warmup 1 / runs 5`
  - before: `prompt 683.70 tok/s`, `generation 74.32 tok/s`, `repo_e2e 67.03 tok/s`, `TTFT 187.22 ms`
  - after:  `prompt 686.64 tok/s`, `generation 74.67 tok/s`, `repo_e2e 67.35 tok/s`, `TTFT 186.42 ms`
- `1 / 128 / warmup 1 / runs 5`
  - before: `generation 74.35 tok/s`, `repo_e2e 73.37 tok/s`, `TTFT 23.07 ms`
  - after:  `generation 75.17 tok/s`, `repo_e2e 74.16 tok/s`, `TTFT 23.20 ms`
- `2048 / 1 / warmup 1 / runs 3`
  - before: `prompt 721.19 tok/s`, `TTFT 2840.43 ms`, `repo_e2e 0.3472 tok/s`
  - after:  `prompt 724.97 tok/s`, `TTFT 2824.94 ms`, `repo_e2e 0.3484 tok/s`

结论：

- 这不是“大跃进”式优化，但三类 workload 都是正方向
- 收益来自真正的高频层内 helper，而不是尾部一次性算子
- 因此可以保留为默认路径

### Control check on newer HEAD

在 `37a8a82` 之后，仓库主线合入了 `prefix_cache / kv_tier` 相关提交。为了确认它们没有把 `metal_bench` 的 direct Metal path 弄慢，做了同机对照：

- old worktree (`37a8a82`, fresh build): `prompt 575.8 tok/s`, `generation 68.7 tok/s`, `repo_e2e 61.4 tok/s`, `TTFT 222.3 ms`
- current `HEAD` (fresh clean build): `prompt 577.7 tok/s`, `generation 69.5 tok/s`, `repo_e2e 62.0 tok/s`, `TTFT 221.6 ms`

结论：

- `metal_bench` 不走 `server_engine` prefix-cache 流程
- 当前主线相对 `37a8a82` 没观察到直接的 Metal source-level regression
- 先前看到的大幅掉速，主要是 build artifact / machine-state 噪音，不是 `kv_tier` skeleton 本身

## Next Targets

下一轮只打 prefill，不再在 wrapper/FFI 上浪费时间：

1. 继续 profile Qwen3.5 GDR prefill，但只接受 clean-build 之后的 trace / benchmark
2. 优先沿着 `mlx_lm` 的思路继续做 helper-level compile / fuse，不优先做 whole-sublayer compile
3. 如果 prompt gap 仍明显存在，再评估更激进的 sequence-level GDR prefill kernel 设计
