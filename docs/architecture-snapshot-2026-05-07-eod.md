# ARLE 架构 + feature snapshot — 2026-05-07 EOD(以代码为真理)

> 今日(commits since `5月6日 22:02` codebase-map.md 最后一改)的 ARLE
> 真实状态 audit。**不重复** [`docs/codebase-map.md`](codebase-map.md) 已有的
> 模块结构详细描述,只补今日变化点 + 当前 readiness gap + 战况。
>
> 用法:开新 session 先读这份 + `MEMORY.md`,获取最新态势,再按需深读
> codebase-map / AGENTS.md / 各 plan。

## Workspace 拓扑(代码为真理)

14 个 crate(`Cargo.toml` workspace):

| 类 | crate | 入口 |
|---|---|---|
| **runtime 主** | `infer/` | scheduler + model + ops + backend + http + kv_tier + distributed + 8 binary(`infer` `bench_serving` `metal_*` `cpu_serve`) |
| **CUDA 后端** | `crates/cuda-kernels/` | `csrc/{attention,gemm,kv,quant,misc}/` + `tools/tilelang/*.py` AOT + Rust FFI |
| **Metal 后端** | `crates/mlx-sys/` | C++ bridge(MLX vendored 5 cpp + 148 FFI export) |
| **业务** | `agent` `chat` `cli` `tools` `train` | session loop + tool 执行 + train 命令 |
| **模型 spec** | `qwen3-spec` `qwen35-spec` `deepseek-spec` | config + tensor name + DSv4 MLA/MoE shard |
| **辅助** | `autograd` `kv-native-sys` | 自动求导 + AdamW codec + LR sched / KV 磁盘+SHM 持久化 |

Feature flags:`cuda` / `metal` / `cpu` / `no-cuda` / `nccl` / `unified_scheduler`(default)/ `rdma-nixl{,-real}` / `reference`。

## Today 变更点(commits since 22:02)

41 个 commit,关键 24 个分类:

### 🟢 已 land(production)

| commit | 内容 |
|---|---|
| `5cacdcb` | scheduler default mixed = Split(P0' 闭合,−41.9% TTFT) |
| `786a20a` `adb2757` | Phase 1A v3 fix +25.6% wins |
| `9b1fb8c` | profiler:SIGUSR1/USR2 → cuProfilerStart/Stop |
| `f791425` `28b56d0` | `scripts/profile_nsys_signal.sh` M_nsys P1 wrapper |
| `f3ff34f` | M_nsys P1 validated longctx kernel data captured |
| `cae08b7` | **H_LP3 root cause**:`cutlass Kernel2` (16×16 wmma) = 56.7% TTFT |
| `419fdea` | M_pf-gemm Phase 0 substrate(`INFER_GEMM_AUTOTUNE`)|
| `85d3751` | M_b.2.2 BF16 split-KV substrate(opt-in `INFER_TILELANG_BF16_SPLIT_KV`)|
| `3e0ed5a` | M_pf-fuse Phase 0 substrate(opt-in `INFER_QWEN3_FUSED_GATE_UP`)|
| `c97afc5` `d13c95a` `92b33de` | metal pool dual-write + flush sync(M_e.1 P3.1c)|

### 🔴 KILLED / abandoned

| commit | 理由 |
|---|---|
| `267fcfa` | M_pf-gemm Phase 0 KILL — cuBLAS heuristic top-1 已最优(autotune ~1% 在 noise 内)|
| `3e0ed5a` 关联 wins | M_pf-fuse Phase 0 KILL — fused gate-up +1.5% 慢于 separate(L2 thrashing,non-monotonic in N)|
| `5886ac0` errors entry | M_b.2.2 BF16 split-KV KILL — ITL +31.6% / out tok/s -18.8% |
| `c219434` | longctx chunk-size TTFT fix KILL(H_LP1+H_LP2 license-kill)|
| `ba748af` `c54fb5d` `8c209a8` | M3.9 Phase 1A v3 KILL CRITERIA + retrospective traces |

### 📋 plans 起草 / promoted / demoted

| commit | 内容 |
|---|---|
| `7d4a2fe` | M_b.3 plan(FlashInfer parity G1+G2)— 仍 active P3 |
| `a5bef64` | M_world1 +30% lead 路线图 |
| `808f19d` `012d989` | M_pf-gemm Phase 2 + Phase 2.5 plan — **demoted P3**(M_pgc 取代)|
| `63396ff` | M_pf-fuse plan — Phase 0 KILL,plan retired |
| `939008f` `3e01f16` | M_pf-graph-prefill-capture plan + half-state reconcile |

### 🌟 evidence 双侧 ground truth(M_world1 P0.1+P0.2)

| commit | 内容 |
|---|---|
| `12c4c86` | **P0.1**:SGLang 真 #2,TTFT 972.9 ms,ARLE 2.03× 慢 |
| `9ee4644` `4ae3b7b` | **P0.2**:8k + highconc 三方表完成 + ARLE 4-shape verdict |
| `7ef707d` | **R1+R2 双 brief**:SGLang prefill stack survey + ARLE GEMM callgraph |
| `3e01f16` | prefill graph audit data(3 hard + 5 soft + 3 graph-safe blockers)|

## ARLE 4-shape verdict(M_world1 P0 完成,代码为真理)

| Shape | ARLE | vLLM | SGLang | Δ rank | source |
|---|---:|---:|---:|---|---|
| high-conc 1k/256/c=64 | **843 tok/s** ⭐ | 647 | 499 | ARLE **+30% past vLLM, +69% past SGLang** | M3.6 F4-Small `13:14 wins` |
| longctx 4k/c=4 | 2005 ms / 152.49 | 1177 ms / 159.1 | **972 ms** ⭐ / 164.3 | ARLE −51% TTFT,−7% tok/s | P0.1 `12c4c86` |
| longctx 8k/c=4 | 4574 ms / **103.07** | **2362 ms** ⭐ / 104.74 | 8054 ms / 78.05 | ARLE TTFT −48%,throughput **拉平 −1.6%** | P0.2 `4ae3b7b` |
| multi-tenant | pending | pending | pending | — | — |

**关键 insight**:
- **ARLE decode-bound 强**(high-conc +69% past SGLang)— F4-Small substrate 已 land
- **ARLE prefill-bound 弱**(全 longctx 1.4-2× 慢)— 双 brief 锁死主因 = **prefill graph capture missing**
- **SGLang 8k 反向弱**(chunked_prefill 切 4 段顺序跑)→ ARLE 8k 不需要直接抄 SGLang 4k 配方
- **+30% lead targets**:4k TTFT ≤ 748 ms(currently 2005),8k TTFT ≤ 1816 ms(currently 4574)

## 当前 readiness gap(以代码为真理 — grep / 文件存在性)

### Backend

| Backend | 状态 | 文件 |
|---|---|---|
| CUDA continuous batching | ✅ primary | `infer/src/backend/cuda.rs` + `server_engine/` |
| Metal varlen packed decode | ✅ secondary | `infer/src/backend/metal/` + `crates/mlx-sys/` |
| CPU dev | ✅ smoke | `infer/src/backend/cpu.rs` |

### CUDA kernel inventory

| Kernel | 状态 | gap |
|---|---|---|
| HD128 BF16 prefill | ✅ TileLang `batch_prefill_paged_hd128.py` | grid `(ceildiv(max_qlen, 64), num_q_heads, batch_size)` — mixed batch 短 row CTA 空转(M_b.3 G1 scope)|
| HD128 BF16 decode | ✅ TileLang | partial+merge 两 kernel split-KV 已 substrate(`85d3751` opt-in)|
| HD128 FP8 decode | ✅ TileLang A0 单 config | **⚠ 无 split-KV** |
| HD256 prefill / decode(Qwen3.5)| ✅ TileLang | **⚠ decode 无 split-KV** |
| Quantized GEMV / Marlin W4 | ✅ csrc/gemm | **⚠ Marlin W4 在 prefill graph 路径 disable**(M_pf-graph Phase 0)|
| `gemm_graphsafe_cuda` (no-workspace cuBLAS) | ✅ production decode 用 | prefill 路径未走(M_pf-graph Phase 0 改造点)|

### CUDA Graph

| 路径 | 状态 |
|---|---|
| Decode graph | ✅ production(`infer/src/model/cuda_graph.rs` + `supports_cuda_graph_decode`)|
| **Prefill graph** | ❌ **当前最大 P0 gap** — `grep INFER_PREFILL_GRAPH` 无命中,plan + audit 已就位,implementation 未启动 |

### 训练 stack

| 命令 | 文件 | 状态 |
|---|---|---|
| SFT | `crates/train/src/commands/train_sft.rs` | ✅ |
| GRPO | `train_grpo.rs` | ✅ |
| 多轮 | `train_multi_turn.rs` | ✅ |
| pretrain | `pretrain.rs` `pretrain_dsv4.rs` | ✅(DSv4 新增)|
| 数据 | `download_dataset.rs` `convert_dataset.rs` | ✅ |
| RLHF/PPO 独立 | — | ❌ 无 |
| HTTP API `/v1/train/{status,events,stop,save}` | ✅ in `http_server/router.rs` |  |

### 特殊 feature

| Feature | 状态 |
|---|---|
| Speculative decoding(DFlash draft) | ✅ `infer/src/speculative.rs` + `mlx_dflash_draft_model.cpp` |
| Prefix cache (Radix) | ✅ `infer/src/prefix_cache.rs` + `kv_tier/` |
| LoRA | ⚠ **Qwen3 only**(`model/qwen3/lora.rs`,Qwen3.5/DeepSeek 无)|
| Tensor parallel | ✅ `tensor_parallel.rs` + `distributed/` + `tp/` |
| Vision/multimodal | 文件存在(`infer/src/vision/`),状态待审 |
| nsys profiler infra | ✅ `scripts/profile_nsys_signal.sh` SIGUSR1/USR2 |

## 优先级当下(代码为真理)

| Rank | Track | 状态 | Owner |
|---|---|---|---|
| **P0 #1** | **M_pf-graph-prefill-capture Phase 0**(单 2048 bucket,opt-in `INFER_PREFILL_GRAPH=1`)| 📋 plan + audit done,implementation 未启动 | 待派 |
| P0.4 | M_world1 multi-tenant baseline | ⏳ pending | Claude(custom Python runner)|
| P1 | M_b.3 G1 segment-aware prefill grid | 📋 plan,**now unblocked**(split-KV abandoned)| 待 prefill graph Phase 0 后再评估 |
| P3 | M_pf-gemm Phase 2 / 2.5(TileLang prefill GEMM port)| ⬇ DEMOTED by R1 evidence | conditional |
| P3 | TRT-LLM bench(P0.3)| ⏳ deferred | Claude |

## Optimization 战况(KILL log)

```
KILLED  M_pf-gemm Phase 0  267fcfa  cuBLAS top-1 ≈ optimal,搜不到更好 algo
KILLED  M_pf-fuse Phase 0  3e0ed5a  fused N=19456 反而 +1.5% 慢(L2 thrashing)
KILLED  M_b.2.2 split-KV   5886ac0  ITL +31.6% out tok/s -18.8%
KILLED  H_LP1+H_LP2        c219434  longctx chunk-size 不优于 baseline
KILLED  M3.9 Phase 1A v3   ba748af  Mixed re-enable 长 ctx 回归 +132% TTFT
DEMOTED M_pf-gemm Phase 2  012d989  → P3,SGLang 也走 cuBLAS 同路径
DEMOTED M_pf-gemm Phase 2.5  012d989  → P3,同上
PROMOTED M_pf-graph-capture  939008f→3e01f16  → P0,双侧 ground truth 锁定
```

## 与 docs/ 旧文档差异

| Doc | 最后改 | 与今日真理差距 |
|---|---|---|
| `docs/codebase-map.md` | 5月6 22:02 | ⚠ 不反映:M_pf-graph 新 P0 / M_pf-gemm KILL / M_pf-fuse KILL / split-KV KILL / SGLang baseline / 4-shape verdict |
| `docs/architecture.md` | 5月6 22:02 | ⚠ 同上 |
| `docs/support-matrix.md` | 5月6 22:02 | ⚠ 没列 LoRA Qwen3-only,FP8 decode 无 split-KV,HD256 decode 无 split-KV |
| `docs/plans/M_world1-30-percent-lead-roadmap.md` | `a5bef64` | ⚠ P0.1 P0.2 完成数据未 patch 进 plan |

**建议**:codebase-map.md 不 full rewrite,保留结构性内容;在文件顶部加一个
"Latest snapshot pointer" 指向本 doc。本 doc 是 today snapshot,带日期戳,
不重写历史结构,做 incremental ground truth。

## 半状态监控(per `feedback_no_half_states.md`)

✅ 已 reconcile 今日:`M_pgc-...` vs `M_pf-graph-...` 两份重复 plan(`3e01f16`)。

⚠ 仍存在的 half-state(待清理):
- `INFER_GEMM_AUTOTUNE` substrate 在 KILL 后仍 default-off opt-in 留存(`267fcfa`)
- `INFER_QWEN3_FUSED_GATE_UP` substrate 在 KILL 后仍 default-off opt-in 留存(`3e0ed5a`)
- `INFER_TILELANG_BF16_SPLIT_KV` substrate 在 KILL 后仍 default-off opt-in 留存(`85d3751`)
- 这三个 substrate 都标 "kept for future experiments",但若超 1 周无后续动作,
  应 promote 到 main path(并 license)或 delete(无半状态)

## Cross-references

- `docs/codebase-map.md` `docs/architecture.md` `docs/support-matrix.md` — 长期结构 truth
- `docs/plans/M_pf-graph-prefill-capture.md` — 当前 P0 #1
- `docs/plans/M_world1-30-percent-lead-roadmap.md` — 战略目标
- `docs/research/2026-05-07-{sglang-prefill-stack-survey, arle-prefill-gemm-callgraph, arle-prefill-graph-readiness-audit}.md` — 双侧 evidence + 改造 audit
- `docs/experience/wins/2026-05-07-{m_world1-p0-sglang-baseline,...-baseline-extended,h_lp3-...,m_pf-gemm-phase0-killed,m_pf-fuse-phase0-gateup-killed}.md` — 关键 ROI 数据
- `docs/experience/errors/2026-05-07-m_b22-bf16-splitkv-killed-regression-and-hang.md` — split-KV KILL
