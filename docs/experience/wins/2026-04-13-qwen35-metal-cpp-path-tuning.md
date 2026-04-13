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

## Next Targets

下一轮只打 prefill，不再在 wrapper/FFI 上浪费时间：

1. 继续 profile Qwen3.5 GDR prefill，确认 `compute_g / conv / output gate` 以外的剩余热点
2. 看 `prefill` 是否还能把部分 shape-stable子图进一步 compile/fuse
3. 如果 prompt gap 仍明显存在，再评估更激进的 sequence-level GDR prefill kernel 设计
