# Gap ③ — CUDA decode graph:仅 B=1 → 多 batch-size 批量捕获

研究日期 2026-06-08。配套主文档
[`2026-06-08-model-loading-preprocessing-arle-vs-industry.md`](2026-06-08-model-loading-preprocessing-arle-vs-industry.md)。

**Verdict**:coverage gap 属实(ARLE graph 只覆盖 c=1,B≥2 完全无 graph),但**收益高度
存疑、必须先量化再 license**。三条已知约束压着它:(a) [[feedback_b1_decode_gpu_bound_overhead_removal_wash]]
证 B=1 decode 是 GPU-bound、overhead 移除是 wash;(b) DSv4 整条 graph 已禁用(MoE host
路由 + NCCL 不可捕获);(c) 主战场 8×H20 是 DSv4。所以本 gap **只对 Qwen3 dense CUDA 在
中等并发(B≥2)有潜在意义**,且要先用 nsys/CUDA-event 量出 B≥2 时 launch-overhead÷compute
比例,够大才做。**不是确定的 win,是一个待测的 coverage 缺口。**

Evidence tier:✅ code-verified(ARLE) / 📖 upstream source-survey / ⚗️ needs experiment。

---

## 1. ARLE 现状(✅)

`DECODE_GRAPH_BATCH: usize = 1`(`infer-cuda/decode_graph_key.rs:9`),注释明写
"B=1 is purely launch-bound, so one `cuGraphLaunch`"。捕获 key:
```
DecodeGraphKey { batch_size: DECODE_GRAPH_BATCH /*=1*/, num_pages }
```
`decode_graph_key_for(page_size, kv_seq_len)`:`num_pages = (kv_seq_len+1).div_ceil(page_size)`
是**唯一变化维**,跨页边界才 recapture(`decode_graph_key.rs:24-29`)。`decode_graph.rs:71`
把 `seq_len = DECODE_GRAPH_BATCH = 1` 写死进捕获的 launch 形状。

**后果**:continuous batching 跑到 B≥2 并发 decode 时,**查不到匹配 graph → 回退 eager
逐 kernel launch**。graph 的全部收益(replay = 一次 `cuGraphLaunch` 省 launch 开销)**只在
单流 c=1 生效**。

进一步:
- TP/MoE 路径 warmup **主动禁 graph**(`executor.rs:127`,NCCL 不可捕获 + MoE host 路由
  每步变)→ Qwen3.5 MoE 与 **DSv4 全程无 graph**。
- 既有 `2026-06-04-load-and-compile-optimization.md` §B-runtime 说 "CUDA decode graph
  essentially solved" —— 准确语境是 **"对 B=1 solved",对 B≥2 absent**。本 doc 不与之矛盾,
  只把 scope 说清。

---

## 2. 业界做法(📖)

### 2.1 vLLM — `capture_model` 多 batch-size 批量捕获
- warmup 末尾对一组 batch size 逐个抓 graph:`cudagraph_capture_sizes`(默认从一个递增表
  截到 `max_num_seqs`,典型如 `[1,2,4,8,16,24,32,48,64,...,256]`)。
- decode 时把真实并发批 **pad 到 ≥它的最近捕获桶**,launch 那个桶的 graph(多算几行 padding
  token,省下全部 launch 开销)。
- 显存代价:每个捕获 size 持有自己的 graph + 静态输入 buffer → 用**有限桶 + padding** 控制
  graph 数量,不是每个 batch size 都抓。
- **V1 piecewise cudagraph**:把模型图在 attention(动态形状)处切开,只捕获静态 piece,配
  `torch.compile`;另有 full-cudagraph 模式可选。解决"attention 形状随 seq 变、不可整图捕获"
  的问题——与 ARLE 用 `num_pages` 当 key 分桶是同一矛盾的两种解法。

### 2.2 SGLang — `CudaGraphRunner`
`--cuda-graph-bs`(可配 batch-size 列表)/ `--cuda-graph-max-bs`,对每个 bs 抓 graph,decode
pad 到桶。机制与 vLLM 同构。也对 MoE/TP 做了可捕获化处理(把 host 决策移出捕获区或固定形状)。

### 2.3 TensorRT-LLM
batched decode 是引擎原生能力;CUDA graph 通过 `--use_cuda_graph` + 一组 batch size 开启,
in-flight batching 调度器在引擎外。

---

## 3. Gap 精确表述

| | ARLE | vLLM / SGLang |
|---|---|---|
| 捕获的 batch size | **仅 B=1** | 一组桶 `[1,2,4,...,max]` |
| B≥2 decode | **回退 eager launch(无 graph)** | pad 到桶,replay graph |
| 变化维 key | `num_pages`(seq 增长) | `(padded_batch_size, 形状)` + piecewise 处理 attn 动态形状 |
| MoE/TP | 全禁 graph | 做了可捕获化(host 决策移出捕获区) |

ARLE 缺两块:**(i) batch_size 维的捕获桶 + padding**;**(ii) MoE/TP 路径的可捕获化**(后者
工程量大,且 DSv4 的 host 路由/NCCL 是硬约束,短期不动)。

---

## 4. 为什么"先量化再做"——三条压制理由

1. **B=1 是 GPU-bound、overhead 移除是 wash**([[feedback_b1_decode_gpu_bound_overhead_removal_wash]]
   3× 实证:decode-graph launch +1.5%、mempool wash)。graph 在 B=1 本就只省一点点;B≥2 的
   compute 更大,launch 开销**占比可能更小**,graph 相对收益**未必更高**——这与"B≥2 才需要
   graph"的直觉相反,必须实测。
2. **DSv4(主战场)无法用**:MoE host 路由 + NCCL 不可捕获(`executor.rs:127`)。Gap ③ 对
   8×H20 DSv4 serving **零收益**。
3. **Qwen3 dense 中等并发**才是唯一受益面,且要 B≥2 时 launch-overhead÷compute 够大。

→ 所以这不是"该不该做",而是"**先 measure**:Qwen3 dense 在 B=2/4/8/16 时,eager launch 的
host 开销占 per-step wall 多少?"够大(比如 >10%)才 license。

---

## 5. 落地建议(⚗️ 仅在 measure 过关后)

**Step 0(gate,先做这个)**:Qwen3 dense CUDA,c=2/4/8/16 各跑一段,nsys 或 CUDA-event
量 **per-step eager launch host 开销 ÷ GPU compute**。
- 若各 B 都 <10% wall → **KILL**(graph 救不回有意义的 wall,符合 B=1 wash 的延伸)。
- 若某 B 段 >10% → 进 Step 1。

**Step 1(若 licensed)**:`DECODE_GRAPH_BATCH` 常量 → 捕获**桶集合**
`{1,2,4,8,...,num_slots}`;key 改 `(padded_batch, num_pages)`;decode 把真实批 pad 到最近桶;
每桶预分配静态输入 buffer(注意 [[reference_disabled_event_tracking_premature_buffer_free]]:
捕获期的 buffer 生命周期 + forward-level keepalive)。

**Gate(Step 1 落地)**:
1. **B=1 不退**:新桶逻辑下 c=1 与现状 byte/ITL 一致(别为多桶把单桶路径拖慢)。
2. **B≥2 真赢**:c=2/4/8 graph-on vs eager 同 harness A/B,ITL/throughput 净增 > padding
   多算 token 的成本。
3. **显存可控**:桶数 × 静态 buffer 显存有界,且与 Gap ② 的 KV 预算不冲突(graph buffer 也要
   算进非 KV 峰值——**与 Gap ② 耦合**,两者一起 measure)。
4. **正确性**:padding token 不污染真实 token 的输出(mask/位置正确)。

**明确不在 scope**:MoE/TP 的可捕获化(DSv4 host 路由 + NCCL 硬约束),除非另起专项。

---

## 6. 与其他 gap 的耦合
- **Gap ②**:每个捕获桶的静态输入 buffer 是**非 KV 显存**,必须算进 profiling-run 的峰值,
  否则 KV 预算会高估。两者落地时一起 measure。
- **B=1 wash 教训**([[feedback_b1_decode_gpu_bound_overhead_removal_wash]]):license 必须用
  wall-clock 同 harness A/B,不用 API-table launch-count % 自欺(§0 framing trap)。

---

## 关联
- ✅ `infer-cuda/decode_graph_key.rs:9`(`DECODE_GRAPH_BATCH=1`)/ `decode_graph.rs:71`
  (seq_len 写死)/ `executor.rs:127`(TP/MoE 禁 graph)。
- [`2026-06-04-load-and-compile-optimization.md`](2026-06-04-load-and-compile-optimization.md) §B-runtime — "decode graph essentially solved"(scope=B=1)。
- [[feedback_b1_decode_gpu_bound_overhead_removal_wash]] / [[reference_dsv4_decode_6ms_path_state]] — B=1 wash + DSv4 graph 不可捕获的实证。
