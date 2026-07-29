# OPD writeback 显存墙：逐 op 实测归因与无损 128K 单卡路径

> **Superseded (2026-07-29)** by
> [`docs/plans/2026-07-29-single-gpu-256k-training-repair.md`](../plans/2026-07-29-single-gpu-256k-training-repair.md),
> the single current capability page. The per-op attribution below remains
> accurate as a decomposition; the up-to-date single-GPU ladder and terminal
> ceiling live in that plan's T5.

## Correction (2026-07-28, post-Move-1 `e736c485a`, backward per-op probe)

Move 1 (free dead MLP+LoRA transients in the tape-disabled forward, commit
`e736c485a`) cleared the **forward** wall: seq=40960 all-linear now completes
64/64 forward groups, flat at free≈20 GB (was OOM at group 3). The wall **moved
to backward** and is now measured, not modeled:

- **Mempool is not the lever.** `--cuda-mempool-retain false` freed the full
  ~35 GB forward-end hoard (driver free 18447 → 53167 MiB, verified
  `effective_release_threshold=0`) — backward still OOMs on the **identical**
  `add [40960,17408]`=2720 MiB. Same op/shape/bytes across a 35 GB free swing ⇒
  live-bytes wall, not hoard, not fragmentation. The mempool A/B is closed.
- **The binding chunk is the ATTENTION recompute inside checkpoint backward, not
  MLP.** Per-op probe (`ARLE_OPD_OP_MEM_CHECKPOINT_FN=63`, layer 63, verbatim
  `pool_used_current`):

  | stage | pool_used | Δ |
  |---|---:|---:|
  | layer_enter | 38357 MiB | — |
  | post_input_norm | 39157 MiB | +800 |
  | **post_attention** | **63497 MiB** | **+24340** ← dominant |
  | post_attention_residual | 64297 MiB | +800 |
  | post_mlp_norm | 65097 MiB | +800 |
  | → MLP `add [40960,17408]`=2720 MiB **fails** (pool already 65 GB) | | |

  The failing alloc is MLP-shaped but MLP is only the straw: the attention
  recompute fills the pool with +24 GB *before* MLP begins. Cutting the MLP alloc
  alone won't help. This corroborates the retracted binary's ~24 GB-attn vs
  ~11 GB-MLP ranking — but that binary's `checkpoint_fn` "3-behind-layer"
  numbering was **wrong**: this binary logs `[checkpoint-map] checkpoint_fn=N
  layers=N..N+1`, i.e. **1:1 with layer index**.
- **Why no `free()` fix works here.** `checkpoint_backward` (tape.rs:897-926)
  re-runs the whole layer forward with the inner tape *enabled*, so the recompute
  intermediates are genuinely needed for the grad walk and are freed only at
  closure exit. Move 1's `!tape.enabled` gate correctly does not fire. The fix is
  structural: seq-chunk the attention forward-recompute (q/k/v proj + SDPA +
  o_proj) inside the backward replay so its peak divides by the chunk count —
  mirroring what Move 1 did for the forward, but at the recompute layer.

Next: an approach doc for seq-chunked attention recompute (autograd op + kernel +
Metal). MLP-SwiGLU chunking (old "Move 5") and GDN dead-store (old "Move 2") both
target non-binding blocks — deferred. Everything below is history.

---

## Correction (2026-07-27, clean HEAD `585e49337`, 4-arm re-measure)

The earlier "40960 passes on main" below is RETRACTED. It rested on a pod probe
binary `6da8e866` that does not exist in the repo — an unrebuildable mystery
binary, not a baseline. On a clean current-HEAD build, 40960 OOMs in **every**
config:

| config | mempool | where it dies | free at the failing alloc |
|---|---|---|---|
| attention-qv | retain=false | backward `add_into` (gated-Q grad, 1920 MiB) | 56 GiB free — fragmentation |
| attention-qv | retain=true | backward `add_into` (1920 MiB) | 18 GiB free |
| all-linear (default) | retain=true | forward group 3 `mul` (SwiGLU, 2720 MiB) | 15.9 MiB — true OOM |
| all-linear (default) | retain=false | forward group 3 `mul` (2720 MiB) | 1680 MiB — true OOM, hoard=0 |

Three plain conclusions:
1. **Not mempool.** all-linear dies on the same `mul` whether the pool hoards
   or releases (hoard=0 still OOMs). The mempool A/B is rejected.
2. **Not a regression.** The "it passed" binary is unrebuildable, so it is not a
   baseline (see `Talk-like-a-human` sibling: a baseline is a commit you can
   rebuild, not a hash nobody has). The wall is the current measurement.
3. **A real capacity wall.** Inside a checkpoint group the forward runs
   tape-disabled and frees intermediates only at the closure's exit
   (`checkpoint.rs` `free_new_except`), so the peak = the sum of ONE layer's
   intermediates. At seq=40960 that single-layer peak eats 58 GiB → 1.6 GiB.
   all-linear carries a full extra ring of LoRA-delta intermediates (live
   tensors 1852 vs attention-qv's 922), pushing that peak past 97.5 GiB.

The forward failure is the gated dense-MLP SwiGLU (`silu(gate)*up`,
`[40960, 17408]` = 2.72 GiB) — the last straw on top of ~53 GiB already live in
the same layer's forward.

Everything below this block is the pre-correction analysis; read it as history.

---

## 结论先说

**40960 在当前 main 上已经不 OOM：rc=0，loss=8.685793，backward 层内峰值 pool_used 85.9 GiB / reserved 89.4 GiB，都在 97.5 GiB 之下。之前的"墙"是 mempool 缓存高水位（forward 后 hoard 39 GiB），不是活张量；per-replay trim 已把它压住。**

**逐 op 实测（layer 63，全注意力层，即历史 OOM 层）推翻旧文档的靶子排序：单层 replay 里最大可动块是「注意力 forward 重算 = +23.4 GiB」，MLP 只有 +11.4 GiB。旧方案"tranche 1 先做 MLP recompute"打错了靶子——它只动第二大项，且当前根本没有墙需要动。**

## 逐 op 实测（H20 GPU4，探针二进制 `6da8e866`，seq=40960）

用 `ARLE_OPD_OP_MEM_CHECKPOINT_FN=60`（=layer 63，冻结层 0–2 使 checkpoint_fn 编号比层号小 3）单臂采样该层 replay+inner-backward 的每个阶段 `pool_used_current`（只读 mempool used，无额外同步）。

**replay forward 逐阶段（相对该层入口 floor 37.9 GiB）：**

| 阶段 | pool_used | Δ | 说明 |
|---|---:|---:|---|
| layer_enter | 37.9 GiB | — | 该层 backward 起点 |
| post_input_norm | 38.7 GiB | +0.8 | RMSNorm |
| **post_attention** | **62.1 GiB** | **+23.4** | ← 单层最大块（q/k/v proj + RoPE + SDPA recompute + gate + o_proj 中间量） |
| post_attention_residual | 62.9 GiB | +0.8 | |
| post_mlp_norm | 63.7 GiB | +0.8 | |
| **post_mlp** | **75.3 GiB** | **+11.4** | Dense SwiGLU（gate/up/silu/mul/down 中间量） |
| post_replay | 76.1 GiB | 整层物化 +38.2 | 全层前向中间量活到 inner backward |
| **inner-backward op 峰值** | **85.9 GiB** | +9.8 | 梯度中间量叠加，此为全局峰值 |
| scope_exit | 37.9 GiB | 回落 floor | free_new_except + re-offload + trim |

**关键读数：**
- 层内 backward 峰值 pool_used **85.9 GiB**，reserved **89.4 GiB**，设备 97.5 GiB → 有约 8 GiB 余量，不 OOM。
- forward 全程 pool_used 平在 34.7 GiB，driver used 却涨到 75.9 GiB——差额 39 GiB 全是 mempool 缓存。
- ledger：`hoarded_fwd/bwd/clean = 39030 / 3167 / 6413 MiB`，per-replay trim 在 backward 把 hoard 从 39 GiB 收到 3.2 GiB。

## 旧结论撤回与修正

- ~~"墙是单层 replay 一次性物化 33 GB"~~ → 实测单层物化是 **38.2 GiB**（attn 23.4 + MLP 11.4 + norm/residual），但它**不撞墙**：峰值 85.9 GiB 有余量。
- ~~"MLP 中间量 11.4 GB 是最大主项，先做 MlpRecompute"~~ → 最大块是 **attention forward 23.4 GiB**，MLP 是第二。且当前无墙可动，不需要任何 recompute。
- ~~"40960 OOM"~~ → 当前 main **完成**，rc=0 loss=8.685793。历史 OOM 是 allocator hoard，非活张量。

## 旧 mempool 分析（保留，仍成立）


### 模型和训练路径

- 模型：ThinkingCap-Qwen3.6-27B-FP8。
- 64 层：48 个 GDN linear-attention 层，16 个 full-attention 层；full-attention 位于 3、7、…、63 层。
- `hidden_size=5120`，`intermediate_size=17408`，24 个 attention heads，4 个 KV heads，`head_dim=256`。
- 该模型配置没有 expert、top-k 或 MoE 字段；训练侧使用 `Qwen35Mlp::Dense`。旧文档中的“MoE FFN”是错误描述。
- 长序列下 `ckpt_group_size()` 已返回 1；继续缩小 checkpoint group 不可能。
- full-attention 已使用 `causal_sdpa_recompute`，其 score/prob backward transient 已按 query chunk 约束。

### current-main 的 40960 实测（探针二进制 `6da8e866`）

单张 H20（97508 MiB）跑 `--synthetic-writeback-seq 40960 --writeback-offload true`，每个 checkpoint group 后读 driver used、mempool reserved/used、live tensors。forward 完成全部 64 group，backward 完成，`RUN_EXIT=0`，`DONE loss=8.685793`。

forward group 采样（`pool_used_current`）：

| checkpoint group | driver used | pool reserved | pool_used | live tensors |
|---:|---:|---:|---:|---:|
| 1 / 64 | 61483 MiB | 60896 MiB | 34664 MiB | 922 |
| 32 / 64 | 75339 MiB | 74752 MiB | 34681 MiB | 953 |
| 64 / 64 | 75915 MiB | 75328 MiB | 35497 MiB | 985 |

**pool_used 全程平在 ~34.7 GiB（64 group 只涨 ~0.8 GiB），driver used 涨到 75.9 GiB。** 差额 39 GiB 是 mempool 保留但未支撑活张量的缓存。driver used 不能当作活激活显存，也不能用于外推序列长度上限。

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

目标是单张 H20 上完成 128K masked-CE writeback，不改变模型、注意力语义、精度或梯度定义。40960 已通过（峰值 pool_used 85.9 GiB / reserved 89.4 GiB），下一步是沿长度阶梯找真正的墙。

### Gate 1：pool_used 长度阶梯

用 `6da8e866` 探针二进制直接测：

```text
49152 → 65536 → 98304 → 131072
```

每个长度记录 forward、layer-63 replay+backward 各阶段的 `pool_used` 峰值和最大单次增长（`ARLE_OPD_OP_MEM_CHECKPOINT_FN=<layer63 的 fn>`），以及 reserved 峰值与 hoard 曲线。只有 pool_used 进入容量账本；driver used 只看差额。

从 40960 的实测外推：层内峰值 pool_used ≈ floor(34.7) + 层内物化(38.2) + grad(9.8) ≈ 82.7 GiB 中，物化与 grad 随 seq 近似线性。128K/40960 ≈ 3.2×，若线性则层内峰值 ≈ 34.7 + 48×3.2 ≈ 188 GiB，**远超 97.5**。所以墙会在阶梯中段出现，且届时是**真实 live bytes**，不再是 hoard。

### Gate 2：撞墙后按实测最大块做最小改动

阶梯里第一个 pool_used 逼近 97.5 GiB 的长度，就是真实容量墙。届时按实测顺序动最大块：

1. **attention forward 重算中间量（40960 时 23.4 GiB，最大）** —— 沿 seq 分块重算 q/k/v proj + SDPA + o_proj，峰值 →/N。SDPA 已 q-chunk bound，主要是 proj 与 gate 中间量。
2. MLP 中间量（11.4 GiB，第二）—— 若 attention 分块后仍不够，再做 MLP custom recompute。
3. grad 中间量（9.8 GiB）—— last-consumer 释放 / tiling。

**在 Gate 1 撞墙前实现任何 recompute 都是无证据的提前优化。** 40960 有 8 GiB 余量，当前不需要动任何算子。

## 当前判断

- **已确认：** 40960 完成，rc=0，loss=8.685793。层内 backward 峰值 pool_used 85.9 GiB / reserved 89.4 GiB < 97.5 GiB。
- **已确认：** 单层最大可动块是 attention forward 重算 23.4 GiB，MLP 11.4 GiB 第二，grad 9.8 GiB —— 推翻旧 MLP-first 方案。
- **已确认：** forward 的 driver-used 涨幅几乎全是 mempool hoard（39 GiB），pool_used 平在 34.7 GiB；per-replay trim 在 backward 收到 3.2 GiB。
- **尚未确认：** 65536/98304/131072 的真实 live-bytes 峰值 —— Gate 1 阶梯待测。
- **当前正确动作：** 跑 pool_used 长度阶梯定位真实墙，不提前改算子。

## 方法教训

显存问题必须同时看 `driver used`、`pool reserved` 和 `pool used`。只看 `nvidia-smi` 或 `device_mem_info()`，会把 allocator 缓存误判为活张量，再把错误输入包装成精确的 buffer 账本和序列上限。**先证明数字代表什么，再用它做归因和外推。**

## Bottom line (post-correction)

40960 does not fit on one H20 today, in any config. The wall is a real
single-layer forward peak (58 GiB → 1.6 GiB in one layer), not mempool and not a
regression. To make it fit, cut that per-layer peak — chunk the attention
forward-recompute and the dense-MLP SwiGLU intermediates along the sequence so
the peak divides by the chunk count. That touches autograd ops, so it needs an
approach doc before code. Until then, 40960 single-card lossless writeback is
blocked by capacity, full stop — no more mempool or regression chasing.
