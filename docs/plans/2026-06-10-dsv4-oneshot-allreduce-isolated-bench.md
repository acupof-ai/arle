# DSv4 decode 链上小消息集合通信 — one-shot AR/AG isolated bench(copy-first)

> Status: **CLOSED — T2 ran, wall-neutral, default back to NCCL** (the 9th
> wall-neutral decode lever; complete causal account in
> [`errors/2026-06-10-dsv4-oneshot-comm-wall-neutral-skew-bound.md`](../experience/errors/2026-06-10-dsv4-oneshot-comm-wall-neutral-skew-bound.md):
> per-op 2-4× real and path-verified on-chain, but the decode wall is
> rank-SKEW-bound, not protocol-bound). One-shot stack stays as opt-in
> `--comm-backend auto` — foundation for multi-node EP and T3 fused AR+norm.
> Verdict: B=1 decode's remaining lever is MTP.
>
> T1 record: **LICENSED** (`3aaa19b9`; results:
> [`wins/2026-06-10-dsv4-comm-bench-oneshot-licensed.md`](../experience/wins/2026-06-10-dsv4-comm-bench-oneshot-licensed.md)).
> Verdict: car_1stage 3.05× @14KB / nccl_sym 2.17× @448KB vs NCCL, all arms
> cross-rank byte-identical. T2 = TpComm integration, car_1stage ≤128KB +
> nccl_sym ≥224KB behind one opt-in flag (C4 short-circuit re-judged with
> staging asymmetry — see wins entry). Commissioned 2026-06-10
> ("写 one-shot AR kernel 的 isolated bench…先copy现成实现…看看 sglang 的是不是更好…给我判断依据")。

## 1. 问题与公式预测(hypothesis,本 bench 负责 license-or-kill)

B=1 decode 25.7 ms/步的墙 = 43 层串行链 attn→AR→MoE→AR(2026-06-08 whole-step-graph
CONCLUSIVE;8 次链下 per-kernel 优化全部 wall-neutral)。链上通信账(同日 nsys kernel med):

| 链上 op | 次数/层 | med | 每步合计 | 消息大小 | NVLink 带宽地板 |
|---|---|---|---|---|---|
| ncclAllReduce(attn + moe) | 2 | 51.9 µs | 4.46 ms | bf16 [1,7168] = 14 KB | ~0.03 µs |
| ncclAllGather(FlashMLA Q) | 1 | 100.5 µs | 4.32 ms | 16 KB/rank → 128 KB | ~0.3 µs |
| **合计** | | | **8.8 ms = 34% of wall** | | **协议/launch 开销 50–100×** |

预测:one-shot AR ~10 µs、one-shot AG ~15 µs ⇒ 省 (2×42 + 85) µs × 43 ≈ **7.3 ms
⇒ 25.7→18.4 ms ≈ 54 tok/s(+40%)**。与 MTP 正交(18.4 ÷ ~1.85 ≈ 10 ms ≈ 100 tok/s)。
此预测必须经 isolated bench 的 exposed-latency 实测修正后才作 license 依据(§5)。

## 2. 候选(全部 copy 现成实现,Apache-2.0;先 copy 后比,不手写)

| # | 来源 | copy 文件 | 血统/特性 | 预期定位 |
|---|---|---|---|---|
| C1 | vLLM | `csrc/custom_all_reduce.cuh` + `.cu` | 此类 kernel 的原型:one-shot(全网格 P2P 读+本地归约,≤512 KB)/two-shot;IPC handle + flag 自旋屏障;world 2/4/6/8 | 参照系 |
| C2 | SGLang sgl-kernel | `sgl-kernel/csrc/allreduce/custom_all_reduce.cu`+`.cuh` | **vLLM 同血统 + sglang 改进**:`copy_mode` 在 kernel 内做 input→IPC buffer 拷贝(省一次显式 D2D)、graph-capture 友好、barrier 方案微调(twoshot 用 2×block_barrier,见 sglang discussion #2918) | **默认 copy 源**(待 C1↔C2 源码 diff + 实测确认) |
| C3 | TRT-LLM(经 flashinfer vendor) | flashinfer `trtllm_allreduce_fusion` 系 | one-shot/two-shot + **AR+residual+RMSNorm 融合**(我们每个 AR 后面都跟 norm/add,可再砍一截链);sm90 multimem/NVLS 路径 | 终局候选,T3 阶段;T1 只测纯 AR 形态 |
| C4 | NCCL 2.27.3(pod 现版本) | **零 kernel 代码** | `ncclCommWindowRegister` symmetric-memory 低延迟 kernel:官方宣称小消息最高 9× 降延迟、NVL8 域 2.5×;约束:持久注册 buffer、fp≤32bit、sum、单 collective/group | **集成成本最低的潜在赢家**;若达标直接当 tranche 1.5 |
| B0 | 现状 | — | 未注册 ncclAllReduce/ncclAllGather | baseline |

AG 侧:vLLM/sgl 不带 one-shot AG;AG 的 one-shot = 同一 IPC 框架下的纯 P2P store + barrier
(~30 行 kernel 变体,从 C2 的 one-shot 骨架派生,属"copy-and-adapt"不属手写)。
另记一个模型级替代:FlashMLA Q-AG 可改为每 rank 冗余算全头 Q(q-proj 复制计算换通信),
T2 时作为 AG 的对照备选,不进 T1。

## 3. 判断依据(决策矩阵 —— "sglang 的是不是更好"由 a–e 实测裁决)

| 维度 | 权重 | 怎么测/怎么判 |
|---|---|---|
| a. **exposed p50 @我们的形状@我们的 pod** | 决定性 | bf16 sum AR [N,7168] N∈{1,2,4,8,16,32}(14–450 KB)+ AG 16 KB/rank;**依赖链计时**(每次 AR 输入依赖上次输出,杜绝流水线重叠),不是 back-to-back |
| b. p99 / 抖动 | 高 | lockstep 放大慢尾:最慢 rank 卡住每层。p99 > 2×p50 即扣分 |
| c. **跨 rank 结果逐字节一致** | 硬性 | 我们的 lockstep 依赖各 rank 输出一致(采样 argmax 翻转 = plan 发散)。one-shot 同序归约天然一致,bench 内逐字节断言;NCCL 构造性一致 |
| d. graph-capture 兼容 | 中 | 无 host sync、IPC buffer 持久注册即可捕获(未来 CUDA graph 复活的前置);sglang 版以此为卖点之一 |
| e. ARLE 集成成本 | 高 | torch 剥离难度(kernel 核心均为纯 CUDA,wrapper 弃用);IPC handle 交换(我们已有 multiproc relay/文件 rendezvous);buffer 模型:持久注册 comm scratch + copy-in/out(拷贝 2–4 µs 计入该 arm 成本,bench 按生产形态诚实计时) |
| f. 融合余量 | 中 | C3 的 AR+norm 融合可再删一个 kernel + 一次依赖;只作 T3 期权,不影响 T1 选型 |
| g. 血统/维护 | 低 | C2 = C1 + 修complement;copy 时 diff 两者,取含 graph 支持与 copy_mode 的版本 |

**赛前预期(供实测推翻)**:C2≈C1(同血统,C2 的 copy_mode 省一次拷贝、capture 友好 →
默认 copy C2);C4 若 exposed p50 ≤ 20 µs 则按集成成本胜出成为 tranche 1.5,custom CA
仅在 Δ(CA − C4) ≥ 10 µs/op(≈ wall +5%)时才值得;C3 融合形态留 T3。

## 4. Bench harness 设计

- **进程模型**:8 进程(与生产 multiproc 同形),复用 `dsv4_parity` 启动模式
  (`INFER_TP_RANK`/`INFER_NCCL_UNIQUE_ID` env + 文件 rendezvous);cudaIpcMemHandle
  经同一 rendezvous 文件交换。单进程 8 卡虽简单但偏离生产 IPC 形态,不取。
- **代码落点**:
  - `crates/cuda-kernels/csrc/comm/oneshot_allreduce.cu` — C2 拷贝(torch 剥离、
    header 内联),+ one-shot AG 变体;`csrc/comm/` 新目录。
  - `crates/cuda-kernels/src/ffi/comm.rs` — extern decl;`src/comm.rs` 安全包装。
  - `crates/infer-cuda/examples/comm_bench.rs` — harness(features cuda,nccl;不需要 deepep)。
- **测量法**:CUDA event 包 100 次/批 × 20 批,p50/p99 取批均值分布;两种模式:
  ① 依赖链(每 op 输入 = 上 op 输出过一个 1µs dummy elementwise)→ **exposed latency,
  license 依据**;② back-to-back → 参考。warmup 50 次丢弃。
- **正确性(每 arm)**:vs ncclAllReduce 参照(bf16 容差 ≤1e-2 相对)+ **跨 rank 逐字节
  一致断言** + same-config-twice。
- **C4 specifics**:`ncclCommWindowRegister` 注册持久 buffer;按 nccl-tests#333 的
  symmetric 开关跑;同样过 copy-in/out 形态计时。
- 离线构建(`CARGO_NET_OFFLINE=1`,无新依赖);单次全矩阵运行 ~3 分钟。

## 5. License-or-kill gates

- **L1(license T2 集成)**:最优 arm 满足 — AR exposed p50@14KB ≤ 20 µs 且全形状
  ≤ 0.5×B0;AG exposed p50 ≤ 25 µs;以实测数代回 §1 公式后预测 wall 增益 ≥ +15%。
  → T2:`TpComm` 加小消息 one-shot 路径(尺寸阈值切换,`--comm-oneshot` opt-in),
  B=1 wall A/B + needle + same-twice + c=2/4/8 sweep(lockstep 指纹一致性照跑)。
- **KILL**:无任何 arm 在 14KB exposed 上 ≥2× 优于 B0 → 协议开销在此 fabric 上不可回收,
  全力转 MTP。
- **C4 短路**:C4 达 L1 且 Δ(最优 CA − C4) < 10 µs/op → 直接 ship C4(零 kernel 维护),
  custom CA 存档。

## 6. Tranches / 预算

| T | 内容 | 预算 |
|---|---|---|
| T1 | vendor C2(+C1 diff)+ C4 wiring + harness + 全矩阵 + 决策表 | ~1 天 |
| T2 | TpComm 集成(opt-in flag)+ 端到端 A/B + bench entry | 1–2 天 |
| T3 | C3 融合(AR+norm)/ NVLS multimem(cuMulticastCreate 管道) | 单独 license |

## 7. 风险

- one-shot 需 8 卡 NVLink fullmesh P2P(H20 NVSwitch ✓,deepep P2P 已验证可用)。
- flag 自旋屏障的内存序 bug 表现为偶发错值:跨 rank 一致性断言 + 1000 次迭代覆盖。
- C4 的 "one collective in flight per group" 约束与我们的串行链天然契合,无冲突。
- 与 lockstep 的交互:custom AR 不经 NCCL launch 队列,反而减少 NCCL 内部排队抖动。

## Refs

- 今日 nsys 分解:`wins/2026-06-10-dsv4-deepep-batched-ab-and-ttft.md` 同日 captures
  (`nsys_b1_allreduce.nsys-rep`,kernel sums)
- 链结论:`errors/2026-06-08-dsv4-mhc-params-1-kernel-overlapped-perkernel-dead.md`、
  `wins/2026-06-08-dsv4-wholestep-graph-wall-neutral-gpu-bound-CONCLUSIVE.md`
- NCCL 2.27 symmetric memory: NVIDIA blog "Enabling Fast Inference and Resilient
  Training with NCCL 2.27"; nccl-tests issue #333
- sglang allreduce barrier 讨论: sgl-project/sglang discussion #2918
