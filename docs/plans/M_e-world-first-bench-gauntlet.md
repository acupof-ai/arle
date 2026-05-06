# M_e — World-first bench gauntlet

> Sub-plan of [`longctx-spec-tilelang-combo.md`](longctx-spec-tilelang-combo.md) §M_e.
> 前置:M_a / M_b.2 / M_c / M_d 全部 land。也复用 backend-unification.md M6
> 的 8-workload × 5-baseline 矩阵骨架(`docs/plans/backend-unification.md` §M6)。
> Goal:把"长上下文 + spec-decode + tilelang + tier-KV"四件套的真实 bench
> 数据放到一张可对比矩阵,公开拿到至少 6/9 workload 第一名。

## 0. 关键差异 vs unification M6

unification M6 的目标是"统一两后端的对比",workload 是 **prefill / decode /
longctx / high-conc / prefix-reuse** 五类;关心的是"两后端可比"。

本计划 M_e 的目标是 **跨厂商对比**(vs vLLM/TGI/SGLang/TRT-LLM/mlx-lm),
workload 加上 **spec-decode 派生**(spec-on / spec-off),关心的是"我们能
在世界排名上拿冠军"。

二者矩阵互不冲突,M6 提供单后端 baseline,M_e 在其上扩 spec 维度 + 厂商对
比。**先 land M6,再 land M_e。** M_e 不重复 M6 已经拿到的数据。

## 1. 矩阵

### 1.1 Workload(8 个)

| ID | 输入 | 输出 | 并发 | 名字 |
|---|---|---|---|---|
| W1 | 4096 | 256 | 1 | prefill-heavy |
| W2 | 128 | 2048 | 1 | decode-heavy |
| W3 | 1024 | 256 | 16 | high-conc-mid |
| W4 | 1024 | 256 | 64 | high-conc-high |
| W5 | 8 同 prefix | 256 | 8 | prefix-reuse |
| W6 | 32k | 256 | 4 | longctx-32k |
| W7 | 64k | 256 | 4 | longctx-64k |
| W8 | 128k | 256 | 1 | longctx-128k |

W3+W4 合并成"high-conc",共 5 类 × 不同 c。所以矩阵实际 8 行。

### 1.2 对手 baseline(5 个)

| Name | Version | Backend | 适用 workload |
|---|---|---|---|
| vLLM | 0.7.2 | CUDA L4 / H100 | W1-W7 |
| TGI | 3.0.1 | CUDA | W1-W4 |
| SGLang | 0.4.0 | CUDA | W1-W7(HiCache for W6/W7) |
| TRT-LLM | 0.17 | CUDA H100 | W1-W4(in-flight batching 强项) |
| mlx-lm | 0.21 | Metal M3 Max | W1-W7 |

W8 (128k) 对手能跑的不多 — W8 主要 vs SGLang HiCache + vLLM long-context patch。

### 1.3 ARLE 配置(4 个)

| ID | 模型 | Backend | spec-decode | 备注 |
|---|---|---|---|---|
| A1 | Qwen3-4B | CUDA(4070 Ti SUPER / L4 / H100) | off | vanilla baseline |
| A2 | Qwen3-4B | CUDA | on (self-spec K=4) | M_a + M_d |
| A3 | Qwen3.5-4B(hybrid) | CUDA | on (self-spec K=4) | M_a + M_c + M_d |
| A4 | Qwen3-4B | Metal M3 Max | off | unification M5 land 之后才能跑 c>1 |

## 2. 数据采集 protocol

每个 cell(workload × backend × ARLE/baseline)产 5 个数:
- TTFT p50 / p99(微秒)
- ITL p50 / p99(微秒)— inter-token latency
- Throughput(tokens/sec, output)

工具:`scripts/bench_guidellm.sh <label> --workload <id>`(已有
canonical-params 锁,不动)。

每 cell 跑 3 次取中位数,误差棒 ≤ 5%。运行前 server warm-up 30 秒。

每个对手 baseline 在 **同硬件 / 同模型权重** 下跑(用各厂家自己的 server),不混淆
模型版本。

## 3. Acceptance(crisp)

- 全部 8 workload × 5 baseline × 4 ARLE 配置 = 160 cells 数据齐全(allowing W8 cell exclusions when baseline does not run that profile)。
- ARLE 在至少 **6 / 9 workload-class 上排第一**(有 9 个不同 workload-class:W1-W8 + W3/W4 共算 high-conc,合算成 9 类)。
- 每个 ARLE 第一名都有可重现 commit hash + 启动命令 + raw guidellm output URL。
- A2 vs A1 在 W2 (decode-heavy) 上 ≥ 1.4× throughput(M_a 数据点)。
- A3 vs A1 在 W6 (longctx-32k) + spec-decode 上 ≥ 1.5× throughput(M_c + M_d 合并)。
- W6 (longctx-32k) c=4 ARLE A3 vs SGLang HiCache + vLLM 至少其一 ≥ 1.8×(M_d 目标)。

## 4. 报告产物

| 产物 | 路径 |
|---|---|
| 总览 wins entry | `docs/experience/wins/<date>-world-first-snapshot.md` |
| 矩阵 raw json | `docs/experience/wins/<date>-world-first-snapshot/<cell>.json`(every cell) |
| 复现脚本 | `scripts/world_first_gauntlet.sh` 一键串跑(新) |

## 5. 实施拆分

| 任务 | 负责 | 工期 |
|---|---|---|
| M_e.1 baseline server 启动脚本(每个对手一个 dockerfile / start script) | Claude(数据搬运) | 2 天 |
| M_e.2 workload preset 落 `bench_guidellm.sh`(已有 longctx-32k preset,补 longctx-64k / 128k / decode-heavy / high-conc presets) | Claude(数据 / 锁参数) | 1 天 |
| M_e.3 spec-decode 启动行(server 加 `--spec-enabled --spec-draft-k 4 --spec-draft-model self`) | 已经有,只需文档 | 0.5 天 |
| M_e.4 串跑 + raw 数据 + 矩阵渲染 | Claude(在远端 H100 + L4 + M3 Max 三台) | 5 天 |
| M_e.5 wins entry 写作 + 公开排名截图 | Claude | 1 天 |
| M_e.6 三方独立审核(可选,如果数据出来确实排名好) | 外部 / 社区 | TBD |

总 ~10 天落地(主要瓶颈是远端硬件 + 对手版本运维)。

## 6. Risks + retreat

- **R1 — ARLE 在某 workload 落后**:那个 workload 单独立 follow-up plan,
  不阻 M_e 完成。例:如果 W4 (c=64) ARLE 落后 vLLM,记 W4 为 P1,继续提交其他
  workload 的第一名。
- **R2 — 远端硬件成本**:H100 时长有限。Mitigation:优先 W6/W7/W8 (long-context)
  + W2 (decode-heavy)+ W5 (prefix-reuse) 这些是 ARLE 强项;W1 (prefill) 和 W3/W4
  (high-conc) 留 L4 / 4070 Ti SUPER 跑。
- **R3 — 对手 server 起不来 / OOM**:fail-record 在 wins entry,标"对手在该
  cell 不能完成 workload"也算 ARLE 第一(但写明背景)。
- **R4 — 数据公布前发现 M_d Q1 污染问题没修**:这种情况下 ARLE
  spec-decode 数据 INVALID,先 close M_d Q1 才能跑 M_e。

## 7. References

- 父计划:[`longctx-spec-tilelang-combo.md`](longctx-spec-tilelang-combo.md) §M_e
- M6 of unification:[`backend-unification.md`](backend-unification.md) §M6
- guidellm canonical params:[`docs/plans/guidellm-integration.md`](guidellm-integration.md) §3
- bench infra:`scripts/bench_guidellm.sh` + `scripts/bench_ab.sh`
- 长 context P0 项目:`docs/projects/2026-04-30-longctx-32k-128k-leadership.md`
