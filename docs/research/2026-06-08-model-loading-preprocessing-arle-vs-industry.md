# 模型加载预处理:ARLE vs 业界 — 系统性对比

研究日期 2026-06-08。本文是**主对比文档**:把"权重在盘上 → 能吐第一个 token"这条加载
预处理链拆成 7 个阶段,逐阶段对照 ARLE 与业界(vLLM / SGLang / TensorRT-LLM /
llama.cpp / MLX),定位 gap。每个 gap 有独立的专调研文档(§Gaps)。

**Evidence tier(全文统一图例):**
- ✅ **code-verified** — ARLE 源码 `file:line` 实证。
- 📖 **upstream source-survey** — 业界公开 docs/repo 的机制(知识 grounded);精确
  API 名 / 行号未在本地 pin,按 §0 属 *hypothesis until line-pinned*。
- ⚗️ **needs experiment** — 落地前必须本地 license-or-kill。

> §0 提醒:业界侧均为 source-survey,**不是** evidence;ARLE 侧 file:line 是 evidence。
> 任何"业界这么做所以我们也该做"的结论,落地前都要本地实验 license。

---

## 0. 一个先验事实:`infer/` 已拆成 `crates/infer-*`

CLAUDE.md 仍写 `infer/` 单 crate,**已过时**。实际加载链分布在:
`crates/infer-cuda/`(CUDA 后端 + loader)、`crates/infer-metal/`(Metal + MLX)、
`crates/cuda-kernels/`(paged KV)、`crates/{qwen3,qwen35,deepseek}-spec/`(config +
Shard)、`crates/infer-topo/`(TP 切片)、`crates/infer-moe/`(EP split)。本文引用以
实际 crate 路径为准。

---

## 1. 七阶段对比总表

| 阶段 | ARLE(✅ file:line) | 业界(📖) | Gap |
|---|---|---|---|
| **① 文件读取** | CUDA eager `fs::read` 整 shard → host RAM,`shard_cache` 全量常驻、不淘汰 (`infer-cuda/loader.rs:621-630`);per-tensor pageable 同步 H2D,串行在 compute stream (`loader.rs:378`)。Metal 走 MLX mmap (`infer-metal/loader.rs:16`) | safetensors **mmap**(按页 fault);GPU-direct streamer:RunAI Model Streamer、CoreWeave tensorizer、fastsafetensors(GDS);多线程 read+deserialize | **Gap ①**(已在 `2026-06-04-load-and-compile-optimization.md` §A 分析 ARLE 侧)→ 本批补**业界深survey** |
| **② Config/Tokenizer** | 各 spec JSON 解析 + 不变量校验 (`deepseek-spec/v4.rs:72`, `infer-metal/config.rs:104`);tokenizer 推到 serve 层 | HF `config.json`+`tokenizer.json`;llama.cpp 全塞 GGUF 元数据 | 无显著 gap |
| **③ 量化转换** | 消费预量化(DSv4 FP8 E4M3+E8M0 block-scale `loader.rs:727`;Metal group-affine);Metal embed 载时 dequant (`loader.rs:82`);DSv4 norm F32→BF16 (`loader.rs:667`) | 同样消费预量化(AWQ/GPTQ/FP8/GGUF);**TRT-LLM 离线标定量化**(SmoothQuant/AWQ 用标定集);vLLM bitsandbytes on-the-fly | 无显著 gap(标定是离线工具链,不在运行时) |
| **④ 布局变换/融合** | head 对齐 TP 切片;Qwen3.5 fused `in_proj_qkv`;RoPE 载时预算 cos/sin (`infer-cuda/ops.rs:12`);Metal 载时 transpose+eval (`infer-metal/loader.rs:76`) | fused QKV/gate-up;Marlin/Machete 载时 repack;**TRT-LLM 离线把 attn/MLP/norm 融成单 kernel 写进引擎** | 不同战场(无离线引擎编译,刻意取舍) |
| **⑤ TP/EP 分片** | column/row 切片 (`infer-cuda/shard_slice.rs:33,70`);DSv4 `ExpertParallel` 按 expert index 切 (`deepseek-spec/lib.rs:29`);**host 端切完才上传** | vLLM/Megatron column/row + PP;各 rank `weight_loader` 从 mmap **只取自己 slice** | 与 Gap ① 耦合(ARLE 读全量再切;mmap+slice 只 fault 自己那片) |
| **⑥ 设备放置/内存预算** | cudarc `clone_htod` (`loader.rs:378`);Metal `set_wired_limit` 自动 pin (`infer-metal/executor.rs:163`);**KV 预算:Qwen3 全静态 `total_pages` (`infer-cuda/executor.rs:280`),DSv4 `cudaMemGetInfo×0.9` (`dsv4.rs:804`)** | **vLLM `determine_num_available_blocks`:跑 profiling forward 测激活峰值再算 block 数**;SGLang `--mem-fraction-static`;TRT-LLM `kv_cache_free_gpu_mem_fraction` | **Gap ②** → 本批专 doc |
| **⑦ Warmup/Graph/JIT** | **CUDA decode graph 仅 B=1**(`DECODE_GRAPH_BATCH=1` `decode_graph_key.rs:9`),keyed on num_pages;TP/MoE 禁 graph (`executor.rs:127`);Metal MLX JIT | **vLLM `capture_model` 对一组 batch size 批量抓 graph**(`cudagraph_capture_sizes`),decode 时 pad 到桶;V1 piecewise cudagraph+torch.compile;SGLang `CudaGraphRunner` 多 bs | **Gap ③** → 本批专 doc |

---

## 2. 三个 gap 的取舍判断(verdict-first)

**Gap ①(读取路径:eager→mmap/streaming/GDS)** — ARLE 侧已被 `2026-06-04` doc 列为
load 优化 #1~#3 lever(pinned async H2D / 多线程 read / mmap),并带 gate。本批新增
[`2026-06-08-gap1-weight-load-streaming-mmap-survey.md`](2026-06-08-gap1-weight-load-streaming-mmap-survey.md)
把业界四套方案(safetensors mmap / RunAI streamer / tensorizer / fastsafetensors-GDS)
展开成架构+数字+适用性,供 lever 选型。**冷启动一次性成本**,不影响稳态 tok/s。

**Gap ②(KV 预算:静态/heuristic→profiling-run)** — 真实 gap。Qwen3 dense 路径**根本
没有自动预算**(用户手填 `total_pages`,填高即 runtime OOM、填低即浪费容量);DSv4 的
`free×0.9` 是 TRT-LLM 式 heuristic,**没测 forward 激活峰值**,0.9 是盲猜 margin。
vLLM 跑一次 profiling forward 实测峰值,精确且安全。详见
[`2026-06-08-gap2-kv-budget-profiling-run.md`](2026-06-08-gap2-kv-budget-profiling-run.md)。

**Gap ③(decode graph:仅 B=1→多 batch-size)** — coverage gap,但**收益高度存疑**。
ARLE graph 只覆盖 c=1;B≥2 并发 decode 完全无 graph(回退 eager launch)。但
[[feedback_b1_decode_gpu_bound_overhead_removal_wash]] 已证 B=1 是 GPU-bound、overhead
移除是 wash;DSv4 更是整条 graph 禁用(MoE host 路由+NCCL 不可捕获)。所以 Gap ③ **只对
Qwen3 dense CUDA 在中等并发(B≥2)有潜在意义**,且必须先用 nsys/CUDA-event 量出 B≥2 时
launch-overhead÷compute 比例才 license。详见
[`2026-06-08-gap3-multi-batch-cuda-graph-capture.md`](2026-06-08-gap3-multi-batch-cuda-graph-capture.md)。

---

## 3. 总体定位

ARLE 加载预处理覆盖了**正确性必需的全部环节**(格式/config/量化/分片/RoPE/KV 预算/
graph),工程路线偏"进程内即时、零离线 build 步",与 **SGLang 的轻量在线路线同侧**,
和 **TensorRT-LLM 的离线引擎编译路线**是两条道(后者把融合/调优/标定全前移到 build,
运行时只反序列化)。三个可对标 gap 里:**Gap ② 最实(且 Qwen3 路径是裸缺口),Gap ① 是
已知冷启动 lever 的业界选型补充,Gap ③ 收益存疑、需先量化**。

落地顺序建议:Gap ②(安全性+免手调,收益确定) > Gap ①(冷启动,lever 已 gated) >
Gap ③(先 measure 再决定是否做)。

---

## 关联文档
- [`2026-06-04-load-and-compile-optimization.md`](2026-06-04-load-and-compile-optimization.md) — ARLE 侧 load+compile 成本分析与 ranked lever(Gap ① 的 ARLE 半)。
- [`2026-06-05-build-compile-speed-optimization.md`](2026-06-05-build-compile-speed-optimization.md) — build/编译速度(正交于本文运行时加载)。
- 本批三个 gap 专调研 doc(见 §2 链接)。
