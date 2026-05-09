# Baseline 数据快照 — 2026-05-09 main `d4c3fc3`(全维度重测)

> 当前 main commit 在 RTX 4070 Ti SUPER (sm_89) 上的 baseline,**7-workload 矩阵 +
> 走的 path/算子标注 + features 启用状态**。
> 用作 future regression / improvement 比较锚点。Raw artifacts 在
> `baseline-d4c3fc3-snapshot/{B1..B7}/`(metrics + service trace + command)。

---

## 元数据

| 项 | 值 |
|---|---|
| 日期 | 2026-05-09 |
| Main commit | `d4c3fc3` |
| 硬件 | RTX 4070 Ti SUPER 16 GiB(sm_89,100 KB smem/SM,88.5 BF16 / 706 FP8 TFLOPS,672 GB/s HBM) |
| CUDA | 13.2(driver) + cudarc + TileLang AOT cubins |
| 服务器配置 | `--num-slots 8 --max-seq-len 5120 --kv-cache-dtype bf16` |
| 准入策略 | 默认 `queue-bound`(本次 baseline 全用默认,`prefix-aware` 未测) |
| Bench 工具 | `scripts/bench_guidellm.sh`(guidellm 包装) |

---

## 7-workload 测试矩阵

| ID | 模型 | 并发 | prompt-in | output | 持续时间 | 用途 |
|----|-----|------|-----------|--------|---------|------|
| **B1** | Qwen3-4B BF16 | 1 | 4096 | 256 | 60 s | 单用户长 prompt 基线 |
| **B2** | Qwen3-4B BF16 | 4 | 4096 | 256 | 120 s | 多并发长 prompt(主基线) |
| **B3** | Qwen3-4B BF16 | 4 | 512 | 2048 | 120 s | decode-dominant 吞吐测试 |
| **B4** | Qwen3-4B-GPTQ-W4A16-marlin-zpfix | 1 | 4096 | 256 | 60 s | W4A16 单用户 |
| **B5** | Qwen3-4B-GPTQ-W4A16-marlin-zpfix | 4 | 4096 | 256 | 120 s | W4A16 多并发(主基线) |
| **B6** | Qwen3-4B-GPTQ-W4A8-zpfix | 1 | 4096 | 256 | 60 s | W4A8 单用户 |
| **B7** | Qwen3-4B-GPTQ-W4A8-zpfix | 4 | 4096 | 256 | 120 s | W4A8 多并发(主基线) |

---

## 全部 metrics 表(**n=3 重测,均值 ± σ%**)

每个 workload 跑 3 次独立 run(冷启动 + 全 warmup),保留全部 raw 数据于
`baseline-d4c3fc3-snapshot/{prefix}/metrics-n3.md`。**所有 σ 均 < 5%**(per
skill rule 6 σ-tight 标准)→ 数据高度可重复。

| ID | TTFT 中位 (ms) ± σ% | ITL 中位 (ms) ± σ% | 输出 tok/s 中位 ± σ% | 成功率 r1/r2/r3 |
|----|---|---|---|---|
| B1 BF16 c=1 | 522.9 ± 0.9% | 22.8 ± 0.0% | 43.9 ± 0.1% | 10/10 · 10/10 · 10/10 |
| B2 BF16 c=4 | 2012.2 ± 0.0% | 25.5 ± 0.1% | 79.0 ± 0.1% | 52/56 · 52/56 · 51/55 |
| B3 BF16 decode | 205.1 ± 0.2% | 18.3 ± 0.1% | 113.8 ± 0.7% | 13/16 · 13/16 · 13/16 |
| B4 W4A16 c=1 | 569.8 ± 0.5% | **14.6 ± 0.0%** | 68.7 ± 0.0% | 13/13 · 13/13 · 13/13 |
| B5 W4A16 c=4 | 2340.0 ± 0.1% | **18.2 ± 0.0%** | **219.6 ± 0.2%** | 64/68 · 64/68 · 64/68 |
| B6 W4A8 c=1 | 410.6 ± 2.2% | 22.2 ± 0.2% | 45.1 ± 0.2% | 10/10 · 10/10 · 10/10 |
| B7 W4A8 c=4 | **1608.8 ± 0.2%** | 25.1 ± 0.2% | 82.6 ± 4.2% | **56/56 · 56/56 · 56/56** |

**σ 观察**:
- 所有 ITL σ 都极小(<0.2%)→ steady-state decode 几乎完美可重复
- TTFT σ 略大(0-2.2%)→ prefill 受 GPU 状态影响略大,但仍 σ-tight
- W4A8 c=4 throughput σ 4.2% 是最大的,与 c=4 多并发 prefill 串行竞争有关
- **成功率全部 deterministic**(r1=r2=r3 across all),除 B2 略有 1 request 浮动

---

## 走的 path 和算子(per workload 详细标注)

### Path 1:**BF16 prefill + BF16 decode**(B1 / B2 / B3)

**Linear(权重投影 Q/K/V/O/MLP)**
| 阶段 | LinearKernelPlan | 实际算子 |
|------|------------------|---------|
| Prefill(seq>1)| `Bf16CublasGemm` | cuBLAS GEMM(BF16),源:`infer/src/ops/linear.rs:113` |
| Decode 单 token | `Bf16Gemv` | 手写 BF16×4 vectorized GEMV,源:`gemv_cuda` |
| Decode batched(graph-safe)| `Bf16GraphsafeGemm` | cuBLAS GEMM,CUDA Graph 安全(B=1 seq,batched-decode 多 slot)|

**Attention**
| 阶段 | 算子 | 来源 |
|------|------|------|
| Prefill | **TileLang AOT HD128 paged-prefill**(`tilelang_batch_prefill_paged_hd128_q{16,32,40,64}_kv8_run_cuda`) | `crates/cuda-kernels/tools/tilelang/batch_prefill_paged_hd128.py` AOT cubin |
| Decode | **TileLang AOT HD128 paged-decode** | `batch_decode_paged_hd128.py` AOT cubin |
| KV cache | BF16 paged KV(page_size=16) | `crates/cuda-kernels/csrc/kv/` |

**附加算子**
- RoPE / RMSNorm / Sampling:custom CUDA C(`csrc/misc/`)
- Token embedding gather:CUDA C
- Logits → sample:custom kernel

### Path 2:**W4A16 Marlin prefill + W4A16 decode**(B4 / B5)

**Linear**
| 阶段 | LinearKernelPlan | 实际算子 |
|------|------------------|---------|
| Prefill(seq>1)| `MarlinW4Gemm` | **Marlin W4A16 GEMM**(3 launches:bf16→fp16 + GEMM + fp16→bf16),源:`infer/src/ops/linear.rs:705 run_marlin_w4_gemm`,kernel:`crates/cuda-kernels/csrc/gemm/marlin_*.cu` |
| Decode 单 token | `W4A16Gemv` | `w4a16_gemv_cuda`(BF16-native GEMV with 4-bit unpack),`csrc/gemm/qweight_gemv.cu` |
| Decode batched(2..=8)| `W4A16BatchGemv` | `w4a16_gemv_batch_cuda`(BF16-native batch GEMV),1 launch |

**Attention**:**TileLang HD128 paged-prefill / paged-decode**(同 Path 1,attention 不受 W4 影响)
**KV cache**:BF16 paged KV(同 Path 1)

### Path 3:**W4A8 Marlin prefill + W4A8 decode**(B6 / B7)

**Linear**
| 阶段 | LinearKernelPlan | 实际算子 |
|------|------------------|---------|
| 全部(prefill + decode 都通过此路径)| `MarlinW4A8Gemm` | **Marlin W4A8 GEMM**,激活 W8 量化 + 权重 W4 packed,源:`run_marlin_w4a8_linear`,kernel:`csrc/gemm/marlin_w4a8_*.cu` |

**Attention**:**TileLang HD128 paged-prefill / paged-decode**(同上)
**KV cache**:BF16 paged KV(W4A8 仅压缩权重 + 激活,KV 仍 BF16)

### 调度器 / 准入(全部 baseline 共用)

| 组件 | 实现 | 备注 |
|------|------|------|
| Admission policy | `QueueBoundAdmission`(默认) | `infer/src/scheduler/policy.rs`,简单队列上限 |
| Cap admission | `prefill_max_requests=8`(`12300c5` 默认) | 控制单步并发 prefill 数 |
| Decode warmup | `warmup_cuda_graphs()` | 提前 capture B=1..num_slots 的 graph |
| RadixCache | 默认开启(prefix cache lookup) | `infer/src/scheduler/cuda/runtime/admission.rs:189 lookup_or_stage` |
| KV pool | Paged KV(BF16),page_size=16 | `crates/cuda-kernels/csrc/kv/` |
| CUDA Graph | Decode 路径自动 capture B=1..8 | warmup time ~1.25 s |

---

## 同条件 quant 对比(c=4 4096-in/256-out:B2 vs B5 vs B7)

| Quant | TTFT 中位 | ITL 中位 | 输出 tok/s | 成功率 | 主要 win |
|-------|----:|----:|----:|----:|---|
| **BF16(B2)** | 2011.6 ms | 25.46 ms | 79.06 | 93% | 平衡(参考) |
| **W4A16 Marlin(B5)** | 2339.4 ms(+16.3%)| **18.15 ms(-28.7%,1.40× decode)** | **219.94(+178%)** | 94% | **decode 速度 + 吞吐** |
| **W4A8(B7)** | **1652.5 ms(-17.8%)** | 25.13 ms(同 BF16) | 82.88(+4.8%) | **100%** | **prefill 速度 + 100% 成功** |

**核心洞察**:
- **W4A16 Marlin** 赢 decode(1.40× ITL,+178% 吞吐),但**输 prefill**(+16.3% TTFT,因 bf16↔fp16 转换 3 launches)→ decode-heavy 场景最优
- **W4A8** 赢 prefill(-17.8% TTFT),**decode 持平 BF16**,**100% 成功率** → prefill-heavy + 稳定性最优
- **BF16** 平衡参考,无单一优势

---

## 单用户 vs 多并发对比(c=1 vs c=4 同 4096-in/256-out)

| Quant | c=1 TTFT | c=4 TTFT | c=1 ITL | c=4 ITL | c=1 tok/s | c=4 tok/s |
|-------|----:|----:|----:|----:|----:|----:|
| BF16(B1/B2)| 520 ms | 2012 ms(+286%) | 22.79 ms | 25.46 ms(+12%)| 43.88 | 79.06(+80%)|
| W4A16(B4/B5)| 572 ms | 2339 ms(+309%)| 14.56 ms | 18.15 ms(+25%)| 68.68 | 219.94(+220%)|
| W4A8(B6/B7)| 415 ms | 1652 ms(+298%)| 22.25 ms | 25.13 ms(+13%)| 44.92 | 82.88(+85%)|

**观察**:从 c=1 → c=4,TTFT 几乎飙升 3 倍(prefill 串行 + 队列等待),ITL 略增(KV 容量竞争),吞吐增 80-220%(W4A16 因 throughput 上限高,scaling 最好)。

---

## Workload-shape 对比(BF16 长 prompt vs decode-dominant:B2 vs B3)

| Workload | TTFT 中位 | ITL 中位 | 输出 tok/s |
|---------|----:|----:|----:|
| 4096-in / 256-out(B2)| 2012 ms | 25.46 ms | 79.06 |
| 512-in / 2048-out(B3)| **206 ms** | **18.30 ms** | **111.86** |

**观察**:短 prompt + 大 output 的 decode-dominant 场景:TTFT 降至 1/10,ITL -28%,吞吐 +42%。**符合 nsys 实证**(prefill 是 c=4 4096-in 的主导成本,而 decode-dominant workload 下 ITL 才是主要 metric)。

---

## 当前 features 启用状态(全表)

### 已 LANDED 默认启用

| Feature | Commit | 默认状态 | 说明 |
|---------|--------|---------|------|
| **TileLang HD128 paged-prefill / paged-decode** | (existing) | ✅ 默认开启(`cuda` feature 含 `tilelang-attn`)| BF16 attention 默认走 TileLang AOT cubin |
| **TileLang HD64 / HD256** | (existing) | ✅ 默认开启 | 不同 head_dim 模型自动 dispatch |
| **CUDA Graph for decode** | (existing) | ✅ 默认开启 | warmup 时 capture B=1..num_slots,~1.25 s 启动开销 |
| **RadixCache prefix lookup** | (existing) | ✅ 默认开启(`--disable-radix-cache` opt-out)| `prefix_cache.lookup_or_stage` 在 CUDA admission |
| **Paged KV (BF16, page_size=16)** | (existing) | ✅ 默认开启 | 主 KV substrate |
| **Marlin W4A16 GEMM** | (existing) | ✅ checkpoint-driven | `MarlinW4Gemm` plan,需 W4A16 + marlin-packed 权重 |
| **Marlin W4A8 GEMM** | (existing) | ✅ checkpoint-driven | `MarlinW4A8Gemm` plan,需 MarlinW4A8 权重 |
| **W4A16 GEMV / BatchGEMV(BF16-native)** | (existing) | ✅ checkpoint-driven | decode 单 token / batch 2..=8 |
| **GGUF Q3K/Q4K/Q5K/Q6K kernels** | (existing) | ✅ checkpoint-driven | GGUF 量化模型 |
| **TurboQuant** | (existing) | ✅ checkpoint-driven | 旧 quant 路径 |
| **W4A8 GPTQ qzeros 修复** | `2a3a6f0` | ✅ 默认 LANDED | greedy 32/32 0% diff,W4A8 准确性 |
| **cap=8 admission default** | `12300c5` | ✅ 默认 8 | `prefill_max_requests=Some(8)`,提升多并发吞吐 |
| **cap=8 decode warmup** | (内置)| ✅ 默认 | `warmup_cuda_graphs` warm B=1..num_slots |
| **B3 Step 1 admission_allows refactor** | `7c8fd61` | ✅ 默认 LANDED | `SchedulerSignals` 信号管线 |
| **B3 Step 2 PrefixAwareAdmission** | `b85929b` | 🟡 **opt-in**(`--admission-policy=prefix-aware`)| 默认仍 queue-bound,prod-safe |
| **`--cold-headroom N` CLI** | `b85929b` | ✅ 配套 prefix-aware | 默认 `max_waiting_requests / 4` |
| **Fail-open guard at admission** | `b85929b` | ✅ 默认 | 防 PrefixAware 死锁 |
| **W3+W4 admission deadlock unblock** | `b708e00` | ✅ 默认 | codex page_budget 修复 |
| **P0.2 Hybrid Phase 1b loader** | `232aed5` | ✅ checkpoint-driven | `marlin_w4_hybrid` 配置自动识别;Phase 2 dispatch 默认 OFF |
| **P0.2 Hybrid Phase 2 dispatch** | (codex 最新)| 🟡 **opt-in**(`hybrid_w4a8_prefill_enabled()` env gate)| 默认 OFF,需要 hybrid checkpoint + env var |
| **Phase 1.A nvtx scope `step_admission_prefix_lookup`** | `5a63142` | ✅ 默认开启(NVTX no-op without profiler)| 仅在 nsys/ncu 附加时产生数据,zero overhead 否则 |
| **R4#6 W4A16BatchGemv override** | `3b9cc06`(env-gated)| ❌ **opt-in 但 KILLED** | `INFER_R4_W4A16_GEMV_OVERRIDE=1`,实测 +37% ITL regression,**不建议开** |

### 已 KILLED(env-gated 但实证不该开)

| Feature | KILL Commit | 原因 |
|---------|------------|------|
| `INFER_R4_W4A16_GEMV_OVERRIDE=1` | `3b9cc06` | bench +37% ITL regression vs Marlin |
| TileLang BF16 split-KV(`INFER_TILELANG_BF16_SPLIT_KV`)| (KILLED 2026-05-07) | ITL +31.6% / out tok/s -18.8% regression,33m+ hang |
| Spec decode self-spec k5 c4 | (KILLED 2026-05-08)| 多次 KILL,axis dead per `aa00c6a` |
| External draft Qwen3-0.6B k5 c4 | (KILLED) | 同上 |

### 已 LANDED 但有 caveat / 未完成

| Feature | 状态 | 说明 |
|---------|------|------|
| cap=8 bimodal 残差 | 🟡 部分残差(33% degraded path) | `db20d34` 分析 + `3fea979` 7-layer 闭环:c20b1ce 是 NO-OP,12300c5 才是真正 fix。残差需 Phase 0.5 evidence 决定是否值得修 |
| W4A8 graph capture(`#24`)| 🟡 待 hoist | W4A8 prefill -36% TTFT 已 LICENSED 但未 default-on,需 graph capture 改造 |
| Hybrid Phase 2 dispatch wiring | 🟡 substrate landed,bench 待跑 | 默认 OFF,nsys 实证 prefill 97% 主导 → 高 leverage 轴 |
| metal_eval_audit 失败 | 🟡 pre-existing | 与 CUDA 路径无关,Metal-only 静态分析 |

### 当前 known broken / 待做

| Feature | 状态 | 备注 |
|---------|------|------|
| Phase 1.A `step_admission_prefix_lookup` 实际 fire | 🟡 nsys 60s 没看到 | 可能 workload 太轻(curl burst short prompts 不触发 lookup_or_stage)— Stage 9 验证 |
| `arle data download` HF Hub blocker(`#34`)| 🟡 P3 demoted | 用 wget + pandas 绕过 |
| Cell (d) 实验 — 12300c5 attribution 闭环 | 🟡 可选 | ~30 min,验证 12300c5 是真正 fix,closes 7-layer SOLID chain |

---

## 关键 observations

1. **Quant 各有 trade-off,无单一最优**:W4A16 赢 decode/吞吐(220 tok/s),W4A8 赢 prefill/稳定性(100% success),BF16 平衡参考。**生产应按 workload 选配**。
2. **W4A16 在 c=4 throughput 220 tok/s 是当前最强单一 win**(vs BF16 79 tok/s,+178%)。
3. **W4A8 100% success rate 强信号** — W4A8 path 稳定性最佳,production-default 候选。
4. **B2/B3/B5 c=4 长 prompt 有 incomplete**(4-7%)— 120s 窗口对 c=4 4096-in 接近极限,future bench 应延长 max-seconds。
5. **TileLang attention 默认开启但 vs FlashInfer 输 3-10%**(见 `2026-04-29-bench-guidellm-cuda-l4-tilelang-on-vs-off.md`):tile defaults 是 Hopper-tuned,sm_89 100 KB smem 的 occupancy ceiling 受限,**未来高 leverage 优化点**。
6. **多并发不是简单 4× 扩展**:c=1 → c=4,throughput 仅 +80-220%(prefill 串行是主因)。
7. **Multi-tenant warm-prefix(B3 Step 2 -24.2%)未在本 baseline**(默认 queue-bound),需要 `scripts/bench_multitenant_burst.py` 单独跑(仍可用 `b85929b` LICENSE 数据参考)。

---

## Cross-references

- 7-workload raw metrics:`baseline-d4c3fc3-snapshot/{B1..B7}/metrics.md`
- Service trace per workload:`baseline-d4c3fc3-snapshot/{B1..B7}/service_stats_trace_summary.md`
- Bench 命令:`baseline-d4c3fc3-snapshot/{B1..B7}/command.txt`
- TileLang on-vs-off 对比:`docs/experience/wins/2026-04-29-bench-guidellm-cuda-l4-tilelang-on-vs-off.md`
- nsys 4-phase 实证(prefill 97%):`docs/research/2026-05-09-eod113-p1a-nsys-decomposition-evidence.md`
- B3 Step 2 LICENSE:`docs/experience/wins/2026-05-09-bench-b3-step2-prefix-aware.md`
- P0.2 hybrid loader:`docs/experience/wins/2026-05-09-bench-hybrid-phase1b-loader.md`
- Pickup queue:`docs/plans/codex-pickup-queue-2026-05-09.md`

---

## 状态

**7-workload baseline 完整重测 + path/算子标注 + features 启用状态全部记录**。
Future regression / improvement 直接对比此 anchor。每个新 feature 落地后,**重跑 7-workload matrix(同 protocol)**,与本 snapshot 对比 Δ%,落进 wins entry "vs baseline-d4c3fc3" 表。
