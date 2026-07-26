# Carry-aware chunked GDN backward — killing the O(seq) `state_history` wall

> Plan / approach doc. Closes the 97 GB OPD-writeback device wall at its root.
> Companion to `docs/experience/wins/2026-07-25-backward-reoffload-device-wall-24576-to-32768.md`
> (offload lifted 24576→32768; this lifts it to ~256K).

## 1. 一句话

OPD masked-writeback 的反传掉进一条 **host 全序列 recompute**,它一次性物化
`state_history`(每 token 一份完整 K⊗V 外积态,≈86 GB @ seq=40960)。而
**device 反传早就 chunk 化了**(峰值与 seq 无关),只是它吃不到 carry 场景的输入。
补齐 device forward 的 carry seed,让 OPD 路径复用已有的 chunked device 反传,
删掉 host recompute —— 峰值 `O(seq·KV)` → `O(seq/64·KV)`,约 **64× 降**。

## 2. 背景:GDN 反传的两个世界

Gated DeltaNet(线性注意力)的态是 `S = Σ kᵢ ⊗ vᵢ`,一个
`[value_heads, key_dim, value_dim]` 的矩阵,随序列递归演化。反传要沿序列
反向重建每步的态。有两种实现:

**A. device chunked(默认,非 OPD)** — `cuda_linear_attention_backward_device_row`
(`backend_cuda.rs:4461`)。三 stage(`:4661` 的 `else` 分支):

1. `linear_attention_chunk_transfer_f32` —— 每 chunk 并行算仿射 grad-state 转移 (M_c, B_c)。
2. `linear_attention_chunk_carry_f32` —— num_chunks 步边界进位,得每 chunk 的入口 grad-state。
3. `linear_attention_chunk_grad_f32` —— 每 chunk 从其边界态**重放 64 token** 精确重算 grad。

峰值 scratch = `grid·64·state_elems`,`grid = wave·rows`,**`wave = min(num_chunks, 8)`**
(`LA_BWD_CHUNK_WAVE`,`backend_cuda.rs:139`、`:4670`、`:4772-4781`)。**与 seq 无关。**
它靠 forward 存的**每 chunk 一份**边界态 `chunk_state`(`num_chunks` 份,
`backend_cuda.rs:4024`;每 chunk 跑之前在 `:4305` 快照)喂 stage-1/3。

**B. host 全序列 recompute(OPD carry 路径)** — `linear_attention_backward`
(`linear_attention.rs:1151`)。当 `has_carry`(`:1221`)时**跳过** device 快路
(`:1222`),重跑整段 host forward(`:1304`),分配 `state_history`
(`linear_attention.rs:1776-1783`):

```
batch · seq_len · value_heads · key_dim · value_dim   (f32)
```

**每 token 一份完整态**。40960 · 32 · 128 · 128 · 4 ≈ **86 GB** —— 就是 97 GB 峰值的
主体,`concat_axis2` 只是压垮的最后一根稻草。

## 3. 缺口:carry 路径为什么走 B 而非 A

OPD phase-2 用 `linear_attention_core_with_carry_taped`(`linear_attention.rs:503`,
从 `qwen35.rs:2375` 调),它 **host-only**:录 `SavedContext::LinearAttentionCtx` 时
device 张量全填 `None`(`preact/qkv_conv/q/k/v/g/g_cumsum/beta/a_inv/chunk_state/raw_output`,
`linear_attention.rs:655-665`),只带 `initial_state`/`initial_conv_window`(`:666-667`)。

于是反传时 `chunk_state = None`,device chunked 反传的输入不存在 → 只能 host recompute。
注释自己承认这是欠债(`linear_attention.rs:499-501`、`:1216-1220`):

> Host-only: the carry-aware device kernel is **deferred to the pod increment**.

**关键:欠的不是反传算法(A 已经 chunk 化且与 seq 解耦),欠的是它的输入 —— 一个
carry-seeded 的 device forward。**

### carry 有两半

1. **递归态 carry** (`initial_state`):前段 prompt 结束时的 `S`。device forward 的
   递归循环从 `final_state = alloc_zeros`(`backend_cuda.rs:4032`)起步 —— 零态。
2. **卷积窗 carry** (`initial_conv_window`):前段最后 `conv_kernel-1` 行 qkv,喂
   causal conv1d 的左侧 tap。device conv1d 核
   (`linear_attention.cu:87-113`)把越界 tap 当零(`:105-106`)—— 隐式假设无历史。

## 4. 方案:seed device forward,复用 chunked 反传,删 host recompute

### 4.1 为什么这是最低熵解

`chunk_state[0]` 在 `:4305` 是"跑 chunk 0 之前的 `final_state` 快照"。**只要把
`final_state` 初值 seed 成 `initial_state`,`chunk_state[0]` 自动等于 carry 态**,
下游三 stage 反传核**一行不用改**。这就是复用点。

> **验证已完成(2026-07-26 两轮只读探查)——工作量比初估更省:**
> - 三个反传核(`chunk_transfer :1005` / `chunk_carry :1252` / `chunk_grad :1291`)对
>   `chunk_state[c]` 用统一 `(chunk_idx*num_value_heads+value_head)*state_elems` 索引,
>   **无任何 `chunk_idx==0` 特判**;`chunk_carry` 根本不读 `chunk_state`,其唯一零假设
>   (`g_in[last]==0`)是**反向 dstate** 的边界,与前向 seed 无关。→ **seed 非零初始态,
>   反传三核零改动。**
> - `chunk_grad` 在 chunk 0 累积的 `grad_state` **从不写回**(`:1733-1737`)——即不产生
>   `d_initial_state`。这对 OPD **恰好正确**:carry 是冻结 prompt 态(`requires_grad=false`,
>   `linear_attention.rs:493`),梯度本就不该回流。天然对齐,无需截断。
> - host recompute / `state_history` / `scan_backward` 的**唯一消费者**是 `has_carry`
>   backward 分支 → 可干净删,无隐藏 caller。
> - carry 两张量 layout 确定:`initial_state` = `[batch, value_heads, key_dim, value_dim]`;
>   `initial_conv_window` = `[batch, conv_kernel-1, qkv_dim]`(通道序 `[q|k|v]`)。
> - OPD 实际 batch==1,走 `backend_cuda.rs:3870` shortcut,不碰 fan-out 切片。

### 4.2 改动清单(到文件:符号)

**① device forward 收 carry(~3 行 D2D + 参数线)**
`cuda_linear_attention_forward_device_row` (`backend_cuda.rs:3918`):
- 入参 `LinearAttentionDeviceForwardArgs` 加 `initial_state`/`initial_conv_window`
  两个 `Option<&DeviceHandle>`(batch>1 fan-out 复用 `cuda_row_slice :3801`,OPD 走
  batch==1 shortcut)。
- 递归循环前(`:4032` alloc_zeros 之后、`:4298` 之前):`initial_state` 存在 → D2D 拷进
  `final_state`;则 `:4305` 的快照令 `chunk_state[0]` = carry 态。否则维持零初始化。
- **反传核零改动**(探查已证)。

**② device conv1d forward 读左侧历史(~10 行 CUDA)**
`linear_attention_conv1d_silu_forward_f32_to_bf16`(`linear_attention.cu:87`):
- 加参 `const float* conv_tail`(可空)+ `int tail_len`(= `conv_kernel-1`)。
- 越界 tap(`:104-109` 的 `t+tap+1<kernel_size` 分支)不再当零:从
  `conv_tail[(src_t+tail_len)*channels + c]` 读(`src_t<0` 时)。`conv_tail==nullptr`
  → 保持当零(默认路径零改动)。layout 对齐 host `conv_window_input`(`:1793-1810`)。

**②b device conv1d backward 补边界 grad_weight(~10 行 CUDA,accuracy 必需)**
`linear_attention_conv1d_silu_backward_f32`(`linear_attention.cu:1741`):
- 现状(`:1760-1763`)把边界 tap 直接 `continue` —— 既不算 `grad_weight` 也不写
  `grad_input`。加了 ② 后这些 tap **对前向有真实贡献**,漏它们的 `grad_weight` = 漏真
  梯度,违反 accuracy-first。
- carry 是冻结常量 → **不写** `grad_input` 到 `conv_tail`(梯度不回流 prompt,正确);
  **但必须补** `grad_weight`:`atomicAdd(&grad_weight[c*kernel_size+tap], dpre *
  conv_tail[(src_t+tail_len)*channels+c])`。`conv_tail==nullptr` → 保持现状(默认零改动)。

**③ device backward 复用(零核改动)**
device forward 现产出 carry-seeded `chunk_state`,taped ctx 不再填 `None`。
`cuda_linear_attention_backward_device_row`(`backend_cuda.rs:4461`)三 stage 原样工作
—— 探查已证零假设不成立。

**④ 让 carry 路径走 device(删 host recompute 分支)**
`linear_attention_core_with_carry_taped`(`linear_attention.rs:503`):
- 删 "always host"(`:548`)分支,改为:device 支持 → 走
  `try_linear_attention_forward_device`(带 carry),录**真实** device ctx
  张量(不再 `None`);否则 host 回退保留(CPU/不支持 dtype)。
- `linear_attention_backward`(`:1151`):`has_carry` 门(`:1221`)从"强制 host"
  改为"carry 且 device ctx 存在 → 走 device chunked 反传(seed carry)"。host
  recompute 分支(`:1304` 起)降级为回退,不再是 carry 的默认路径。

**⑤ host recompute 降级为 CPU 回退(路由,不删)**
CUDA 支持(`key_dim==value_dim==128 && conv_kernel<=5`,`backend_cuda.rs:3857`)的 carry
tape 走 device;host `state_history` recompute(`linear_attention.rs:1776`)**保留**作
非 CUDA / 非 128×128 的回退。CPU seq 小(仅测试),不触 86GB。device-快路 + CPU-回退
是一职一路,**非 half-state**——故不删 `state_history` / `scan_backward`,只改路由让 CUDA
carry 不再落到它们。

### 4.3 数字

| 项 | host recompute(现状) | carry-seeded chunked(方案) |
|---|---|---|
| 主 buffer | `state_history` = seq·KV | `chunk_state` = (seq/64)·KV |
| @ seq=40960, 32h, 128×128 | ≈86 GB | ≈1.34 GB |
| 反传 scratch 峰值 | O(seq·KV) | O(wave·64·KV),wave=8,与 seq 无关 |
| 可训 seq(单卡 H20) | 32768(offload 后) | ~256K 量级 |

## 5. 三条路对比(为什么先做这条)

| 方案 | 峰值 | 熵/工作量 | 判断 |
|---|---|---|---|
| **carry-aware chunked backward** | O(seq/64·KV) | 低 —— 反传核已 chunk 化,只补 forward carry seed + conv 历史 | ✅ 本计划 |
| 序列并行(TP 切 seq) | O(seq/N·KV) | 高 —— 训练路径零基础(无 all_gather/reduce_scatter),只降 N=world_size 倍 | 正交,跨机才需要 |
| 更多 offload | 不降峰值 | —— | 已证否(companion win) |

单卡降 64×,机器现成;序列并行只降 world_size 倍(H20 单机 ≤8×)且改动大得多。
先做前者,序列并行留到真跨机 256K+ 再上。

## 6. 正确性门槛(accuracy-first)

host 不是 ground truth,只是另一条数值路。真正的 oracle 是解析梯度。

- **有限差分梯度检验(绝对基准)**:`cargo test -p autograd linear_attention` 新增一条
  小 seq(跨 chunk,如 seq=128)carry 单测:`(L(θ+ε)−L(θ−ε))/2ε` vs device chunked
  grad,对 8 个投影输入逐一验。这是唯一证明"反传核是对的"的绝对判据,不依赖 host。
- **device vs host A/B**:同一 carry 输入,device chunked grad vs host recompute grad,
  作交叉验证(两条独立数值路一致 → 双重确证);差异只作诊断,不作通过判据。
- **端到端 needle/lever ×3**:真训练路径,与项目 correct-inference 门一致
  (`scripts/needle_gate.py` + `scripts/lever_gate.sh`)。
- **VRAM trace**:`--synthetic-writeback-seq {24576, 40960, 65536}` × `ARLE_OPD_VRAM_TRACE=1`,
  device 峰值应从 O(seq) 平成 O(seq/64) —— 40960 由 OOM 变通过,峰值 ≈ base + ~1.5 GB
  而非 +86 GB。loss 与 companion win 的 loss 表同 seq 对齐。
- **A/B 保命**:`--la-backward-mono`(`backend_cuda.rs:4570`)作数值回归黄金对照。

性能是**第二轮**:第一版只求数值对 + 峰值降,不追 chunked 核吞吐。正确性落地、bench
有基线后,再看 `LA_BWD_CHUNK_WAVE` / chunk 内 matmul 等杠杆,独立成轮。

## 7. 风险 / 未决(探查后重估)

- ~~stage-2 `chunk_carry` 零假设~~ → **已消除**:三反传核无 `chunk_idx==0` 特判,
  `chunk_carry` 不读 `chunk_state`,seed 非零态零核改动(§4.2 探查结论)。
- ~~conv backward 对称性~~ → **已定位为 ②b**:边界 tap 的 `grad_weight` 必须补
  (accuracy 必需),`grad_input` 不写(carry 冻结)。~10 行,default 路径零改动。
- **`d_initial_state` 丢弃** → **非风险,是正确**:carry 冻结,梯度本不回流
  (`chunk_grad :1733-1737` 天然不写回)。
- **CPU 回退**:非 CUDA / 非 128×128 仍走 host recompute —— 保留,不是 half-state
  (device 是快路,host 是明确回退,单一真源仍是 device)。

## 8. 落地顺序

1. **tranche 1**:① forward state seed + ② conv forward 读历史 + ②b conv backward 补
   grad_weight。带有限差分单测(小 seq carry,§6)。default 路径 byte-identical。
2. **tranche 2**:④ 切路由(CUDA carry taped forward/backward 走 device chunked;host
   recompute 降级为 CPU 回退,§4.2⑤——不删,一职一路)。带端到端 VRAM trace。
3. bench 条目(`docs/experience/wins/`)+ CHANGELOG + companion win 交叉引用;订正
   companion win 根因段(峰值主体是 `state_history` 的 `[seq,heads,key_dim·value_dim]`,
   非 `[heads,seq,dim]` 激活)。
