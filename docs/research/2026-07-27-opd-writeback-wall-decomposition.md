# OPD writeback 显存墙：当前证据与无损 128K 单卡路径

## 结论先说

**现在不能把 40960 OOM 归因于 GDN、全注意力 backward，或 33 GB 的单层重算。当前 main 的直接证据是：CUDA 驱动显示约 95 GiB 已用，但真正支撑活张量的 mempool used 只有约 34.7 GiB；约 56 GiB 是 mempool 保留但未使用的缓存高水位。**

因此，旧文档基于 driver used 得出的显存账本、每 token 斜率、128K 上限，以及“先实现 MLP recompute”的方案全部撤回。下一步不是改注意力或 MLP，而是先完成同一二进制下的 mempool retain matched A/B，再用 pool used 重建真实账本。

## 已确认的事实

### 模型和训练路径

- 模型：ThinkingCap-Qwen3.6-27B-FP8。
- 64 层：48 个 GDN linear-attention 层，16 个 full-attention 层；full-attention 位于 3、7、…、63 层。
- `hidden_size=5120`，`intermediate_size=17408`，24 个 attention heads，4 个 KV heads，`head_dim=256`。
- 该模型配置没有 expert、top-k 或 MoE 字段；训练侧使用 `Qwen35Mlp::Dense`。旧文档中的“MoE FFN”是错误描述。
- 长序列下 `ckpt_group_size()` 已返回 1；继续缩小 checkpoint group 不可能。
- full-attention 已使用 `causal_sdpa_recompute`，其 score/prob backward transient 已按 query chunk 约束。

### current-main 的 40960 实测

测量基于 commit `b5cb3b5f6` 加未提交的 checkpoint allocator 诊断；单张 H20 的总显存读数为 97508 MiB。运行 `--synthetic-writeback-seq 40960`，在每个 checkpoint group 后同时读取：

- driver used/free；
- CUDA mempool reserved current；
- CUDA mempool used current；
- live tensor 数量。

结果：

| checkpoint group | driver used | pool reserved | pool used | live tensors |
|---:|---:|---:|---:|---:|
| 1 / 64 | 94740 MiB | 90336 MiB | 34672 MiB | 1850 |
| 2 / 64 | 95540 MiB | 91136 MiB | 34680 MiB | 1851 |
| 3 / 64 | 95764 MiB | 91360 MiB | 34688 MiB | 1852 |

随后 forward 因 `cuda alloc_zeros failed` 退出，`RUN_EXIT=1`，没有进入 backward。

最关键的分解是 group 1：

```text
pool reserved - pool used
= 90336 MiB - 34672 MiB
= 55664 MiB
```

即 mempool 持有约 55.7 GiB 当前没有支撑活张量的缓存。group 1 到 group 3，pool used 只增加 16 MiB，而 pool reserved 增加 1024 MiB。driver used 因此不能当作活激活显存，也不能用于外推序列长度上限。

### 并发 WIP 已排除

远端曾包含 `fused_linear_distill.rs` 和 `update_strategy.rs` 的并发未提交改动。将这两个文件恢复到 committed main、只保留 allocator 诊断后重新构建，40960 的 group trace 与失败结果相同。

因此，这两个 WIP 不是当前显存现象的来源。

### allocator 机制

CUDA allocator 当前默认：

```text
MEMPOOL_RETAIN = true
CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = u64::MAX
```

其目的原本是服务 decode：同步后继续保留释放的块，避免下一步重新向驱动申请。对推理热循环这是合理的 caching allocator 策略；对一次长序列训练 forward，它可能把早期 transient 的高水位一直留在 pool 中，最终让后续分配看到卡已接近满载。

这与实测的 `pool reserved ≫ pool used` 一致，因而是当前最强候选机制。**但它还不是已确认根因**：尚未完成 `retain=true` 对 `retain=false` 的同二进制因果 A/B。

### train 侧控制入口

此前 `--cuda-mempool-retain` 只属于 serve 参数，`train agent-opd` 会在参数解析阶段拒绝它：

```text
error: unexpected argument '--cuda-mempool-retain' found
```

现在已把同名布尔开关接入 OPD runtime flags，并在 student store/context 创建前调用现有 setter；默认仍为 `true`，`false` 仅作为 matched A/B treatment。`DeviceContext` 现在对两种取值都显式写入 release threshold（`true → u64::MAX`，`false → 0`），随后读回并记录有效值，避免把写入失败或进程内残留状态误当成 treatment。该接线已通过 CUDA/no-CUDA typecheck 和 CLI help gate，但 H20 上的容量行为验证仍待执行。

## 哪些旧结论已撤回

以下说法都使用了被 allocator reserved 高水位污染的 driver used，不能继续作为事实：

1. “40960 的墙是单层 checkpoint replay 一次性物化 33 GB。”
2. “MLP 中间量 11.4 GB 是当前最大主项，必须先实现 `MlpRecompute`。”
3. “前向活激活约按 1 MiB/token 增长。”
4. “65536 必然超出单卡容量。”
5. “MLP 沿序列分块即可无损达到 128K。”
6. 任何由 32768/40960 driver used 两点直接推导的 128K 或 256K 上限。

旧 reference 二进制曾表现为较低的 group-1 driver used，并能走到 backward；current-main 则在 forward 早期 OOM。reference 的准确 commit、allocator 初始状态和构建内容尚未固定，因此两者不能构成有效 A/B，也不能据此归因代码回归。

历史逐层 trace 仍能说明：那次运行的失败发生在 layer 63 full-attention 路径，GDN backward 尚未执行，所以把那次失败归因于 GDN 是错误的。但这不等于已经证明 full-attention 或 MLP 是 current-main 的容量根因。

## 无损 128K 单卡方案

目标是单张 H20 上完成 128K masked-CE writeback，不改变模型、注意力语义、精度或梯度定义。方案按证据 gate 推进。

### Gate 1：固定来源并验证 train-side mempool 开关

先记录 commit、dirty diff、构建参数、模型配置、CUDA 版本、GPU、完整命令和二进制哈希；来源不完整的历史运行不进入 A/B。

然后使用已接入的 train runtime flag，保证它在 student `DeviceContext` 创建前生效。不修改 allocator 实现，不修改 attention、MLP 或 autograd 数学。运行 A/B 前先从 release-threshold 读数确认参数确实生效。

该开关只改变空闲块何时归还给驱动，因此理论上不改变数值；实际仍需用 matched A/B 验证。

### Gate 2：同一二进制跑 40960 matched A/B

固定模型、参数、进程启动方式和二进制，只改变：

```text
A: mempool retain = true
B: mempool retain = false
```

两臂都记录：

- 每个 checkpoint group 的 driver used、pool reserved、pool used、live tensors；
- forward 是否完成 64 个 group；
- 是否进入并完成 backward；
- loss 和失败位置；
- wall time，确认释放策略的性能成本。

判定：

- 若 B 中 pool reserved 接近 pool used，且 40960 越过当前 forward OOM，则 allocator retain 是已验证的第一容量墙。
- 若 B 中 pool reserved 仍远高于 pool used，则继续定位未被 release threshold 控制的缓存或异步释放点。
- 若 B 中 pool used 本身接近设备容量，则 allocator 缓存不是主墙，转入真实活张量分解。

### Gate 3：用 pool used 跑长度阶梯

仅在 Gate 2 通过后，使用 `retain=false` 的同一实现依次测：

```text
64K → 96K → 128K
```

每个长度都记录 forward、backward 各阶段的 pool-used 峰值和最大单次增长。只有 pool used 才进入容量账本；driver used 只用于观察驱动和 pool 的差额。

128K 达成的判据：

- 完整 forward + backward 成功；
- 没有降低精度、截断 attention、缩短有效序列或改变 loss mask；
- 与 retain=true 的可运行短序列 matched A/B 保持 loss/gradient correctness parity；
- 记录 wall-time 代价和 pool-used 峰值。

### Gate 4：只有 live bytes 真正撞墙才改算子

若关闭 retain 后，某个长度的 pool used 本身逼近 H20 容量，再按实际最大 live buffer 决定最小改动：

1. 先定位到具体 op、buffer、shape、dtype 和生命周期；
2. 若是无必要的同时存活，先缩短生命周期或复用输出；
3. 若是数学上可精确分块的 transient，再做 seq-chunked recompute；
4. 只有实测证明 MLP 中间量是最大主项，才实现 MLP custom recompute；
5. 只有实测证明 attention backward 是最大主项，才调整其精确分块路径。

在 Gate 3 前实现 MLP recompute、重写 attention 或修改 GDN，都是没有证据支撑的提前优化。

## 当前判断

- **已确认：** current-main 40960 OOM 时，约 34.7 GiB pool used 对应活张量，而约 90.3–91.4 GiB 被 pool reserved；driver used 不能代表活内存。
- **强候选：** train 继承了面向 decode 的无限 mempool retain 策略，导致 transient 高水位不归还。
- **尚未确认：** 关闭 retain 是否足以让 40960、64K、96K 或 128K 完成。
- **当前正确动作：** 在 H20 上验证开关与 release threshold，并完成 40960 matched A/B；A/B 成立后再跑长度阶梯。

## 方法教训

显存问题必须同时看 `driver used`、`pool reserved` 和 `pool used`。只看 `nvidia-smi` 或 `device_mem_info()`，会把 allocator 缓存误判为活张量，再把错误输入包装成精确的 buffer 账本和序列上限。**先证明数字代表什么，再用它做归因和外推。**
