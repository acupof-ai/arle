# Baseline Snapshot — 2026-05-09 main `d4c3fc3`(post-46-commit session)

> 当前 main commit `d4c3fc3` 的全维度 baseline。包含 BF16 / W4A16 Marlin / W4A8 GPTQ
> 三种 quant × c=1/c=4 × prompt-shape 4 个 workload 共 5 bench。**保留作为 future
> regression / improvement 比较锚点**。Raw artifacts 在 `baseline-d4c3fc3-snapshot/`
> 子目录(metrics.md + service_stats_trace_summary.md + command.txt 每个 workload)。

## Snapshot 元数据

- **Date**:2026-05-09
- **Main commit**:`d4c3fc3`(post B3 Step 2 + P0.2 + Phase 1.A + R4#6 KILLED + c20b1ce reverted)
- **Hardware**:RTX 4070 Ti SUPER 16GiB sm_89,CUDA 13.2
- **Server config**:`--num-slots 8 --max-seq-len 5120 --kv-cache-dtype bf16`
- **Admission policy**:default `queue-bound`(opt-in `prefix-aware` not benched here)
- **Bench tool**:`scripts/bench_guidellm.sh`(guidellm wrapper)

## 5-workload baseline matrix

| ID | 模型 | workload | 维度 | 用途 |
|----|-----|---------|------|------|
| **A1** | Qwen3-4B BF16 | c=4 / 4096-in / 256-out | 多并发长 prompt | 单租户长上下文基线 / TTFT-gap reference |
| **A2** | Qwen3-4B BF16 | c=1 / 4096-in / 256-out | 单用户长 prompt | 无并发对比基线 |
| **A3** | Qwen3-4B BF16 | c=4 / 512-in / 2048-out | 多并发短 prompt 长 output | decode-dominant throughput |
| **A4** | Qwen3-4B W4A16 Marlin zpfix | c=4 / 4096-in / 256-out | 多并发长 prompt | W4A16 quant baseline |
| **A5** | Qwen3-4B W4A8 GPTQ zpfix | c=4 / 4096-in / 256-out | 多并发长 prompt | W4A8 quant baseline |

## Metrics 全表(median / p50 / p99 from successful)

| ID | TTFT median(ms) | TTFT p99(ms) | ITL median(ms) | ITL p99(ms) | TPOT median(ms) | Out tok/s median | Success % |
|----|----:|----:|----:|----:|----:|----:|----:|
| **A1 BF16 c=4** | **2005.5** | 2072.1 | **25.43** | 25.58 | 33.18 | 79.08 | 52/56 = 93% |
| **A2 BF16 c=1** | 519.4 | 806.3 | 22.80 | 22.81 | — | 43.87 | 10/10 = 100% |
| **A3 BF16 c=4 decode** | 205.9 | 495.8 | 18.29 | 18.34 | — | 112.35 | 9/12 = 75% |
| **A4 W4A16 Marlin c=4** | 2336.9 | 2359.8 | **18.13** | 18.20 | — | **220.09** | 64/68 = 94% |
| **A5 W4A8 c=4** | **1634.3** | 1687.3 | 25.10 | 25.90 | — | 81.01 | **56/56 = 100%** |

## Quant 维度对比(c=4 4096-in/256-out 同条件)

| Quant | TTFT median | ITL median | Out tok/s median | Success | 主要 win |
|-------|----:|----:|----:|----:|---|
| BF16(A1) | 2005.5 ms | 25.43 ms | 79.08 | 93% | 平衡 |
| **W4A16 Marlin(A4)** | 2336.9 ms(+16.5%)| **18.13 ms(-28.7%,1.40× decode)** | **220.09(+178%)** | 94% | **decode 速度 + 吞吐** |
| **W4A8(A5)** | **1634.3 ms(-18.5%)** | 25.10 ms(同 BF16) | 81.01(+2.4%) | **100%** | **prefill 速度 + 100% success** |

**关键 quant 取舍**:
- W4A16 Marlin:**赢 decode**(1.40× ITL,+178% throughput),**输 prefill**(+16.5% TTFT,format 转换开销)→ decode-heavy 场景最优
- W4A8 GPTQ-zpfix:**赢 prefill**(-18.5% TTFT),**decode 持平**,**100% success rate** → prefill-heavy 场景 + 稳定性最优
- BF16:全平衡,无特殊优势,**ground truth 参考**

## 并发维度对比(BF16 c=1 vs c=4 同 4096-in/256-out)

| Concurrency | TTFT median | ITL median | Out tok/s median |
|---|----:|----:|----:|
| c=1(A2)| 519.4 ms | 22.80 ms | 43.87 |
| c=4(A1)| 2005.5 ms(+286%)| 25.43 ms(+12%)| 79.08(+80%)|

c=1 → c=4:TTFT 飙升(队列等待 + 多并发 prefill 串行),ITL 略增,throughput 提升 80%(没到 4× 是 prefill bottleneck)。

## Workload-shape 维度对比(BF16 c=4 长 prompt vs decode-dominant)

| Workload | TTFT median | ITL median | Out tok/s |
|----------|----:|----:|----:|
| 4096-in/256-out(A1)| 2005.5 ms | 25.43 ms | 79.08 |
| 512-in/2048-out(A3)| 205.9 ms | **18.29 ms** | **112.35** |

短 prompt 大 output:TTFT 几乎降到 1/10,ITL 减 28%,throughput 增 42%。**符合 nsys 实证**:prefill 是 c=4 4096-in 的主导成本,而 decode-dominant workload 下 ITL 才是主要 metric。

---

## 已落地 Feature 按维度标注

### 维度 1 — TTFT(首 token 延迟)

| Feature | Commit | TTFT 方向 | 实测 metric | 状态 |
|---------|--------|----------|------------|------|
| W4A8 GPTQ-zpfix | `2a3a6f0` + `b5889b3` | ⬇ 改善 prefill | TTFT **1634ms**(BF16 2005ms,**-18.5%**)| ✅ LANDED |
| cap=8 admission default | `12300c5` | ⬇ 改善多租户 TTFT p99 | -86% TTFT p99(per `cap8-ttft-tail.md`)| ✅ LANDED + caveat |
| B3 Step 2 PrefixAwareAdmission | `b85929b` | ⬇ 改善 multi-tenant warm TTFT | **-24.2% multi-tenant TTFT median**(318→241ms)| ✅ LICENSED + opt-in |
| W4A16 Marlin | (existing) | ⬆ 略增 prefill TTFT | +16.5%(format conversion overhead) | ✅ tradeoff acknowledged |

### 维度 2 — ITL(每 token 间延迟,decode 速度)

| Feature | Commit | ITL 方向 | 实测 metric | 状态 |
|---------|--------|---------|------------|------|
| W4A16 Marlin | (existing) | ⬇ 大幅改善 | **18.13ms vs BF16 25.43ms = 1.40× decode** | ✅ LANDED |
| W4A8 GPTQ | `b5889b3` | 持平 BF16 | 25.10ms ≈ BF16 25.43ms | ✅ LANDED |

### 维度 3 — Throughput(吞吐 tok/s)

| Feature | Commit | Throughput 方向 | 实测 metric | 状态 |
|---------|--------|----------------|------------|------|
| W4A16 Marlin | (existing) | ⬆ 大幅 | **out tok/s 220 vs BF16 79 = +178%** | ✅ LANDED |
| W3+W4 admission deadlock unblock | `b708e00` | ⬆ multi-tenant 持续吞吐 | (eliminates regression) | ✅ SOLVED |
| Hybrid Phase 1b loader | `232aed5` | ⬆ enables Phase 2 dispatch | bench TTFT p50 68.4ms regression gate PASS | ✅ LANDED loader-only |

### 维度 4 — Memory(KV 容量 / VRAM 利用)

| Feature | Commit | Memory 方向 | 实测 metric | 状态 |
|---------|--------|------------|------------|------|
| W4A16 Marlin | (existing) | ⬇ weight 8GB→2GB(-75%)| 释放 ~6 GB → KV pool 扩容 | ✅ LANDED |
| W4A8 GPTQ | `2a3a6f0` | ⬇ activation 也 W4 + KV BF16 | 同 W4A16 但激活也压缩 | ✅ LANDED |

### 维度 5 — Stability(success rate / σ)

| Feature | Commit | Stability 方向 | 实测 metric | 状态 |
|---------|--------|----------------|------------|------|
| W4A8 GPTQ-zpfix | `2a3a6f0` | ⬆ greedy 32/32 0% diff | A5 baseline 56/56 = **100% success** | ✅ LANDED |
| cap=8 + warmup fix(c20b1ce REVERTED, 12300c5 是真正 fix)| `12300c5` | ⬆ multi-tenant 76→100% turn success | per `cap8-final-synthesis`(re-attributed)| ✅ LANDED |
| metal_eval_audit | (existing) | ⬆ Metal materialize regression gate | static-analysis test(unrelated to CUDA path)| 🟡 pre-existing failure documented |

### 维度 6 — 调度 / 准入策略

| Feature | Commit | 维度 | 实测 metric | 状态 |
|---------|--------|------|------------|------|
| `--admission-policy {queue-bound,prefix-aware}` CLI | `b85929b` | 准入策略可配 | default queue-bound(prod-safe);prefix-aware opt-in | ✅ LANDED |
| `--cold-headroom N` CLI | `b85929b` | cold-request 缓冲 | default `max_waiting_requests / 4` | ✅ LANDED |
| Fail-open guard at admission | `b85929b` | 防 PrefixAware 死锁 | `if candidates.is_empty() { take first deferred }` | ✅ codex 自发加 |

### 维度 7 — Substrate 基础设施

| Feature | Commit | 维度 | 状态 |
|---------|--------|------|------|
| Phase 1.A `step_admission_prefix_lookup` nvtx scope | `5a63142` | 性能可观测性 | ✅ LANDED |
| RadixCache production-wired at CUDA admission | (existing per `1217375` audit) | 前缀缓存 substrate | ✅ default-enabled |
| Hybrid W4 Marlin DeviceMatrix side-tensor 加载 | `232aed5` | hybrid checkpoint substrate | ✅ LANDED loader-only |
| Skill kernel-optimization v1.7.0(19 anti-patterns)+ v1.8.0 batch(6 candidates ready)| `c768b70` + memory | 方法论 substrate | ✅ codified |

### 维度 8 — 已 KILL 的 hypothesis(知识沉淀)

| Hypothesis | KILL commit | 原因 |
|-----------|------------|------|
| W4A16BatchGemv override(R4#6)| `3b9cc06` | bench +37% ITL regression vs Marlin(empirically refuted) |
| c20b1ce 是 cap=8 fix 真因 | `3fea979` | 7-layer audit:NO-OP in production-default,12300c5 才是真正 fix |
| (4 prior P0 KILLs from Q1/Q2)| various | M_pf-gemm autotune / M_pf-fuse / M_b.2.2 split-KV / M_pf-graph Phase 0 |

---

## 用法 — future regression / improvement comparison

任何 future 优化 落地后,**重跑 5-workload matrix**(同 protocol)→ 与本 snapshot 对比 Δ%。
- TTFT/ITL/throughput Δ% 落进 wins entry "vs baseline-d4c3fc3" 表
- Success % regression 即 KILL signal
- 单维度 win 单维度 lose 必须 explicit tradeoff 罗列(per skill rule 7)

每 baseline workload 重跑命令保存于 `baseline-d4c3fc3-snapshot/<label>/command.txt`。

---

## 关键 observation

1. **Quant 各有 trade-off,无单一最优**:W4A16 赢 decode/throughput,W4A8 赢 prefill/stability,BF16 平衡。生产应**按 workload 选配**。
2. **A1(BF16 c=4 4096-in)是 4/56 incomplete**:120s 窗口 + c=4 4096-in 长 prompt 接近 server 极限。Future bench 应延长 max-seconds 或 reduce concurrency。
3. **A4 W4A16 throughput 220 tok/s** 是当前最强 single-axis win。Hybrid Phase 2 dispatch 可能进一步提升(prefill 也走 W4A8 path)。
4. **A5 W4A8 100% success rate** 强信号 — W4A8 path 的 stability 最佳,适合 production-default 候选。
5. **multi-tenant warm-prefix(B3 Step 2)** 未在本 snapshot — 需 single-prompt-from-multi-session workload (`scripts/bench_multitenant_burst.py`)单独 bench。

## Cross-references

- 5-workload raw metrics:`baseline-d4c3fc3-snapshot/{A1..A5}/metrics.md`
- Service trace per workload:`baseline-d4c3fc3-snapshot/{A1..A5}/service_stats_trace_summary.md`
- Bench commands:`baseline-d4c3fc3-snapshot/{A1..A5}/command.txt`
- Pickup queue:`docs/plans/codex-pickup-queue-2026-05-09.md`
- Methodology:`memory/feedback_bidirectional_audit_cycle.md`(local auto-memory)

## Status

**5-workload baseline snapshot LANDED**。Future regression / improvement bench
benchmarks against this anchor。Feature dimension matrix 全部已落地特征按 7 维度
(TTFT / ITL / throughput / memory / stability / 调度 / substrate)+ KILL 维度标注。
