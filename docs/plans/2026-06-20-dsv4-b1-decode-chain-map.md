# DSv4-Flash B=1 decode 全链路拆解图 (链路图 v5, TP4)

## 图的逻辑(怎么读这张图 / 我的拆解推理)

1. **基本模型:解码一个 token = 一条串行链.** 43 层固定序列,每层
   `hc_attn → mla_attn →(AR)→ hc_ffn → moe →(AR)`,最后 head→lm_head→sample。
   **墙 = 这条链的 critical path**(最长串行路径),不是各阶段成本之和。
2. **host 发射 vs GPU 执行 —— 谁是墙?判据 = CUDA graph 实验.** host 发射 kernel 是
   异步的,正常 overlap 在 GPU 执行下面(隐藏)。graph 消除发射开销:实测 **graph −41%
   (不是 +)** → 发射不是墙、GPU 执行才是。故图里 host 行画在 GPU 链**下面(隐藏)**。
3. **为什么 host 不能 run-ahead(= foundation-bound):** 每步 `ctx.sync`(ops.rs:467)
   把采样结果同步回 host → host 每 token 阻塞等 GPU → 发不出下一步;+ 每 tick 跨进程
   barrier(4 进程)。**这两个是 foundation,graph 删不掉 → graph 才会 −41%。**
4. **拆解粒度:大阶段→小阶段→kernel,每层都对 floor.** stage profiler 给大阶段(sync
   虚高、**相对可信**),linear profiler 给小阶段 GEMV,nsys 给 kernel(multiproc 4 次失败,
   待修)。每个阶段算物理 floor(bytes/带宽 或 latency),**实际 vs floor 的 gap = 工程开销**。
5. **杠杆判据(三个都满足才是杠杆):** ① 在 critical path 上 ② 实际 >> floor ③ 有可行缩小手段。
   逐项 license-or-kill。
6. **唯一能动 foundation-bound 墙的杠杆 = 摊薄(amortize):** foundation 成本 per-token 固定;
   减少它的唯一办法 = 一条链跑多个 token = **MTP**。故 MTP 是唯一结构杠杆,受接受率门控。

数据可信度:stage/linear profiler 相对可信(绝对值 sync 虚高);foundation 代码核实;
**baseline/接受率以 wins 实测为准**;逐 kernel 待修 nsys。

## Baseline(= 当前最优全开,wins 实测)
- **clean baseline c=1 = ~44 tok/s**(2026-06-16:ON 44.9 / OFF 43.8,TP8,profiling OFF)。
- 全开配置:TP8 · num-slots 64 · max-total 4096 · chunked-prefill 64 ·
  `MOE_BACKEND=allreduce` `EXPERT_BACKEND=deepgemm` `INCREMENTAL_KV=1`
  **`FUSED_DISPATCH_PAYLOAD=1`** + code-default(batched-FlashMLA · fused-wqkv ·
  decode-proj DeepGEMM · **decode-graph**)。
- **我之前测的 37 不是 baseline**:① 漏了 `FUSED_DISPATCH_PAYLOAD` ② TP4 vs TP8。
  正在补 fused-dispatch 重测 TP4。**"38" 那个更早的数是 DECODE_PHASE_TIME/LINEAR_PROFILE
  污染(每步 sync,−25–35%)。**

## 时间尺度
- clean wall ≈ **22.3 ms/token**(44 tok/s,TP8 baseline)。TP4 待补测(预期略高于 22.3)。

## 全链路(端到端,一个 token 从请求到吐出 —— host/算子/通讯三维)
```
[Host-pre]  scheduler tick → admission(跨进程 TickAdmissions relay,4进程)→ build plan(token+pos)
   ↓ launch(异步,overlap 在 GPU 下)
[算子 GPU 串行链 ×43层]
   embed
   每层: hc_attn(Sinkhorn) → wqkv_a GEMV → compressor/indexer GEMV → MLA attn(FlashMLA) → DSA select → wo GEMV
         →【通讯 AllReduce】→ hc_ffn → moe route GEMV → FP8 expert GEMV →【通讯 AllReduce】
   head_hc → lm_head GEMV
   ↓
[Sample]   argmax →【ctx.sync 阻塞 host(ops.rs:467)】→ token 回 host
   ↓
[Host-post] detokenize → stream 返回 → 下一 tick
```
**三维成本归属:**
- **host 路径** = tick + admission relay + plan + ctx.sync 等 + detokenize。除 ctx.sync 外全 overlap 在 GPU 下;
  **ctx.sync 是 host 每 token 唯一阻塞点 = foundation(graph −41% 证明它删不掉)**。
- **算子** = 43 层 kernel 串行链 = **墙**(mla_attn 70% / hc 20% / moe 11%)。
- **通讯** = 2× AllReduce/层 = 86/token(inline compute stream,无 overlap,占 3% → KILL)+ 每 tick 跨进程 barrier。

## 大阶段 → 小阶段(相对占比可信)
| 大阶段 | ~ms | % | 小阶段 | 杠杆(三判据) |
|---|---|---|---|---|
| **mla_attn** | 18 | **~70%** | GEMV 10.7 + 注意力数学 7.8 | 见下 |
| hc_params (Sinkhorn) | 5.2 | ~20% | attn 2.7 + ffn 2.5 | warp-tail 已优化 |
| moe | 3.0 | ~11% | route 1.9 + FP8 experts 1.3 | FP8 lane lock-on |
| comm (AllReduce×2/层) | 2.3 | ~9% | attn_AR + moe_AR(inline compute stream) | **KILL**(3.5×快=0 wall) |
| 其余 | 2.9 | ~11% | shared_expert host 1.6/GPU 0.25=纯发射税 | 可融合(小) |
(% 之和 >100 因 sync-inflated + overlap;clean wall = overlapped critical path)

## mla_attn 内部(70%,主拖累)
- **GEMV(M=1 FP8,10.7ms,串行发射):** wqkv_a_fused 2.7 / compressor_wkv 1.8 /
  compressor_wgate 1.67 / wo_a 1.34 / wo_b 1.1 / indexer_wq_b 0.81 / wq_b 0.74 /
  indexer_weights 0.59 — **KILL**:M=1 是 **latency-bound 不是带宽-bound**(所以 TP4≈TP8
  这部分),uint4 1.8× isolated → wall-neutral(overlap 在 AllReduce 后)。
- **注意力数学(7.8ms):** FlashMLA decode(vendor,<1%)+ DSA official select(换 legacy 后
  correctness-only)+ compressor_update + indexer scoring。**待修 nsys 钉到具体 kernel。**

## 物理 floor —— roofline(最物理层:计算 FLOPs + 访存 HBM,ground-truth 权重 shape)

H20:HBM **4.0 TB/s** · FP8 peak **296 TFLOPS** · ridge = 74 FLOP/byte。
B=1 decode 每个 GEMV M=1 → 算术强度 = 2·M = **2 FLOP/byte ≪ 74 → 深度 memory-bound**,floor = 权重bytes/带宽。

**每层 per-GPU(TP4)权重(实测 safetensors shape):**
| kernel | shape | dtype | /GPU bytes | HBM-floor |
|---|---|---|---|---|
| wq_a(repl) | 1024×4096 | FP8 | 4.2MB | 1.0µs |
| wq_b(col/4) | 32768×1024 | FP8 | 8.4MB | 2.1µs |
| wkv(repl) | 512×4096 | FP8 | 2.1MB | 0.5µs |
| wo_a/wo_b | 8192×4096×2 | FP8 | 16.8MB | 4.2µs |
| 6 routed experts(EP) | 3×2048²×6 | **INT8** | ~18.9MB | 4.7µs |
| shared expert(TP/4) | 3×2048×4096 | FP8 | 6.3MB | 1.6µs |
| router+hc(repl) | — | BF16/F32 | 5.2MB | 1.3µs |
| **每层** | | | **~62MB** | **~15.5µs** |

**全 token:** 43 层 + lm_head(529M BF16/4=265MB)= **~2.9 GB/token/GPU → HBM floor 0.73ms**。
compute floor = 2·13e9/296e12 = **0.088ms**(更小)→ roofline = max = **0.73ms(HBM-bound)**。

**verdict:实测 26ms / floor 0.73ms = ~36× above roofline,HBM 利用率 ≈ 2.8%(读 ~110 GB/s of 4000)。**
**墙不是带宽、不是算力 —— 是 LATENCY**:43 层×~10 个 M=1 小 kernel 串行,M=1 填不满 SM(低
occupancy → 抽不满 HBM)+ foundation(ctx.sync+barrier)禁止 overlap → HBM 97% 闲置。

**物理重述每个杠杆:**
- **GEMV 带宽优化 = KILL**(HBM 已 97% 闲,"读更快"无意义)。
- **唯一逼近 36× headroom 的办法 = 抬高 M** 让 M=1 GEMV 变带宽-bound:**MTP(M=2-3)/ batching(c≥4)**。
  这就是 MTP/batch 是仅有真杠杆的物理原因 —— 只有它们朝 roofline 移动。
- **36× gap = M=1 occupancy/latency 税**,被 foundation 结构性封顶。
- TP4 vs TP8:M=1 latency-bound,GPU 部分接近 → 37 vs 44 是 TP8 注意力/MoE 分片更细(更接近各自 floor)。

## MTP 路径(d2)= 唯一摊薄 foundation 的杠杆
- MTP step(d2)= capture 1.6 + draft 4.4 + **verify ~52(3-row batched,~2× no-spec)** + commit 9。
- emit/step = E[accepted]+1;**break-even ≈ 57% 接受率**(emit 要 > MTP步/no-spec步 ≈ 2.14)。
- **接受率 prompt-dependent**(wins ShareGPT 实测 per-prompt **41–59**:java-code 58.7 高接受,
  prose 41 低接受;聚合 **+14% @B=1**)。我先测的 4 个技术 prompt 接受率 50–53%(偏低端)→ wash;
  **正用 code(高接受)+ prose 混合 prompt 复现 +14%**。
- 2026-06-13 chain-fold 在更高接受 + 优化下 **+18–20%(53 tok/s ×3)**。
- 1-head NextN 架构上限 ~52–62%;**2-head MTP → 稳过 57% break-even → 稳健 B=1 杠杆(训练/OPD)**。

## 杠杆总账(从图的判据导出)
1. **逐 kernel 基本穷尽**:GEMV M=1 latency-bound(KILL)/ comm 3%+3.5×快=0wall(KILL)/
   select correctness-only / graph −41%(KILL)/ 单进程 TP wall-neutral(KILL)。
2. **MTP d2**:接受率门控(>57% 赢);高接受 workload(code/ShareGPT)+14%,低接受 wash。
3. **2-head MTP**:唯一稳健 B=1 杠杆(把接受率推过 break-even)→ 训练。
4. 见下"未分析空缺"。

## 未分析空缺(self-audit;§0:每条要么补、要么明确 defer —— 不静默 pass)

**P0 — 阻塞分析的硬空缺:**
- **G1. mla_attn 70% 内部没拆到 kernel.** 主拖累的内部(FlashMLA / DSA select /
  compressor_update / indexer 谁占大头)还是黑盒 —— nsys 在 multiproc TP4 上 **4 次失败**
  (stale/export/empty/--inherit 没抓 worker)。**补法**:改 `serve_multiproc.rs:474`
  给每个 fork+exec 的 worker 包一层 nsys(per-rank wrap,≤10 行)→ 8 卡 kernel 分解。**最大空缺。**
- **G2. PREFILL 链路完全没分析.** 整张图只有 decode。prefill 是**另一个 regime**
  (M=prompt_len → **compute-bound 不是 memory-bound**,roofline 完全不同),而且**输入敏感性 +
  prefix-reuse win 都在 prefill**。缺:prefill 分阶段 / prefill roofline / chunked-prefill /
  TTFT 分解。**半张延迟图缺失。**

**P1 — 量化空缺(数有了但没拆透):**
- **G3. "36× = latency 税"没拆.** 距 HBM floor 的 gap 归给"latency"但没分解成 kernel 发射延迟
  (×N kernel)+ 访存延迟(每散乱访问一次)+ occupancy(M=1 只占几个 SM)。缺:数每 token 的
  kernel 数 + 每 kernel 发射+延迟(需 G1 的 nsys)。
- **G4. host 路径 + 跨进程 barrier 是推断未实测.** "除 ctx.sync 外全 overlap" + barrier 成本都
  推断。缺:nsys CPU/host 时间线 + 量 barrier 的 µs/token。
- **G5. stage-profiler 绝对值 sync 虚高.** 每阶段绝对 ms 只能信相对;真实 overlapped 绝对值要干净 nsys。

**P2 — 完整性空缺:**
- **G6. batched-decode(c>1)链路没画.** 36× 头寸在批量,但逐行循环怎么挥霍它(重读权重)没进链路。
- **G7. MoE EP 不均(b=1)** 代码确认了(`moe.rs:675`),量级没进图。
- **G8. KV-read bytes 不在 roofline.** 只算权重;sparse-512/window-128 的 KV 读没进 floor。
- **G9. MTP step 分解可能 stale.** verify≈52ms 偏大(no-spec 步才 26ms),earlier profiler 数可能
  confounded,要干净重测。
- **G10. vs 工业 baseline.** 我们 36×-above-floor 是正常还是差?SGLang/vLLM 在这条 roofline 哪?
- **G11. TP4 vs TP8 没拆.** 为什么 37 vs 44、哪个阶段受益于 TP8 分片。
- **G12. 无方差/error bar.** 单次数;§0 要 ≥3 次 + σ 才能下 kernel-efficiency 结论。

**填空缺的关键路径:G1(修 nsys)解锁 G3/G4/G5;G2(prefill 链路)是独立的另一半。其余 P2 跟随。**
