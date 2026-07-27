# OPD writeback 显存墙:到底卡在哪(实测分解)

一句话:**40960 训练不了,不是因为 GDN 线性注意力层,而是因为「前向留存的激活」+「全注意力层的 backward」。之前打算做的「chunk GDN backward / GDN full-BPTT」是在优化一个不到 4 GB 的次要项——建错了东西。**

## 怎么测的

ThinkingCap-Qwen3.6-27B-FP8,单张 H20(97.5 GB),`--synthetic-writeback-seq N` 跑一次
masked-CE 写回。逐 checkpoint-group、逐 op 打显存 `used/free`。两个长度:32768(过)、40960(OOM)。

## 显存账本(27B,单卡)

| 项 | 大小 | 随 seq 变? | 说明 |
|---|---|---|---|
| 固定地板(FP8 权重+Adam+留存) | **37.5 GB** | 否 | 32768/40960 完全相同 |
| 前向峰值(进 backward 前) | 71.6 / **79.7 GB** | 是,~1 MiB/token | 地板之上的 34~42 GB 是前向留存激活 |
| **单个全注意力层 backward** | **+12~16 GB** | 是 | 峰值项 |
| 单个 GDN 层 backward | +2~4 GB | 是 | 次要项 |

按层型分的 backward 峰值(Run A 跑完全 64 层):

| 层型 | backward 峰值 |
|---|---|
| **全注意力 self_attn** | **87.5 GB** ← 最高 |
| GDN linear_attn | 75.5 GB |

40960 时:进 backward 已占 79.7 GB → **第一层(layer 63,恰好是全注意力层)**的 backward 就推到 97.1 GB,只剩 399 MiB,`concat_axis2` 分配失败 → OOM。**GDN 层根本没轮到就挂了。**

## 三个结论

1. **墙 = 前向留存(34~42 GB)+ 全注意力单层 backward(12~16 GB)叠加。** 两者都随 seq 线性增长,合起来在 40960 撞满 97.5 GB。

2. **GDN 不是瓶颈。** 单个 GDN 层 backward 只多吃 2~4 GB。chunk 它、给它做 full-BPTT,省的是这 2~4 GB——抬不动墙。这正是「杠杆作用在错误的项」。

3. **offload 是有效的,但没抓全。** 前向逐层 `used` 几乎持平(60 层只涨 1 GB),说明 checkpoint offload 在正常工作;但进 backward 时仍留存 42 GB——这部分是 offload 没覆盖的前向激活。

## 真正该动的两个杠杆(按主项排序)

- **全注意力层 SDPA backward 峰值(+12~16 GB/层)** —— 单层最高瞬态,降它直接抬墙。SDPA backward 已按 q 分块(`attention.rs:490`),但 `q_chunk` 可能偏大,或是 proj/LoRA 的 grad 中间量占大头。
- **前向留存的 42 GB** —— 进 backward 前就压在那,offload 没抓住的前向激活。降它腾出地板上方的全部空间。

**下一步待定的唯一事实**:上面两块的具体 buffer 构成——是 SDPA recompute 的 `[heads, q_chunk, seq]` 分数/概率(调一个参数就行,便宜),还是 proj/LoRA 的 grad 中间量(要改 backward)。这决定修法是「调一个数」还是「重写一段 backward」,成本差一个量级。定位到再出精确设计。

## 一处方法教训

后台 agent 的 summary 把墙归给了 GDN(只看了 backward plateau 高度)。**按层型拆开、并注意到 OOM 发生在全注意力层之后,归因才反转。** 峰值高度相同不代表来源相同——必须拆到层型 + 具体 op 才算数。
