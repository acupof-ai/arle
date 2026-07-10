# Gap ① — 权重加载:eager-read → mmap / GPU-direct streaming(业界深survey）

研究日期 2026-06-08。配套主文档
[`2026-06-08-model-loading-preprocessing-arle-vs-industry.md`](2026-06-08-model-loading-preprocessing-arle-vs-industry.md)。

**本 doc 的分工**:ARLE 侧的成本分析与 ranked lever **已存在于**
[`2026-06-04-load-and-compile-optimization.md`](2026-06-04-load-and-compile-optimization.md) §A
(pinned async H2D #1 / 多线程 read #2 / mmap #3 / DSv4 FP8 repack #4),不在此重复。
本 doc 专做**业界四套权重加载方案的深 survey**——把那份 doc 里一行带过的 "Precedent:
RunAI…/vLLM…/safetensors…" 展开成架构 + 数字 + 适用边界,供 lever 选型与实现参考。

Evidence tier:✅ code-verified(ARLE) / 📖 upstream source-survey / ⚗️ needs experiment。

---

## 1. ARLE 现状(一句话,详见 2026-06-04 §A)

✅ CUDA 侧:`load_raw_from_shard` 对每个 shard 做 `fs::read()` 整文件入 host RAM,存进
**不淘汰的 `shard_cache`**(`infer-cuda/loader.rs:621-630`)→ 近似整模型在 RAM 复制一份;
每个 tensor 再 `view.data().to_vec()`(二次 host 拷)→ `clone_htod` **pageable 同步 H2D,
串行在 compute stream**(`loader.rs:378`),无 read↔copy 重叠、无 pinned 带宽,专用
`copy_stream`(`cuda-kernels/tensor.rs:181`)闲置。TP 各 rank 读**完整 shard** 再 host 切片
(`shard_slice.rs`)。Metal 侧已是 MLX mmap(`infer-metal/loader.rs:16`),且 M_e.13 实证
import≈35µs / disk≈100ms 不是瓶颈,**out of scope**。

主成本项:DSv4 ~19.6 GB/rank 冷加载。这是**一次性启动成本**,不影响稳态吞吐。

---

## 2. 业界四套方案(📖)

### 2.1 safetensors mmap(基线,最低风险)
- **机制**:`safetensors.safe_open(path, framework, device)` 把文件 `mmap` 进地址空间,
  header 是 JSON 描述每个 tensor 的 `(dtype, shape, data_offsets)`,zero-copy 切 tensor
  视图;按页 fault,**只有真正 `.to(device)` 的 tensor 才把那段页读进物理内存**。
- **对 TP 的关键收益**:配合 per-rank slice,各 rank 只 fault 自己分到的行/列区间页,
  **不把整 shard 读进 RAM**——直接干掉 ARLE 当前 "读全量再切" 的 RAM 复制。
- **vLLM/HF 默认走这条**;HF `from_pretrained(low_cpu_mem_usage=True)` = meta-device 初始化
  + safetensors mmap 增量填充,避免 "先建满精度张量再覆盖" 的双倍内存。
- **数字**:相对 `torch.load` 的 pickle 全量反序列化,mmap 省掉一次 host 全拷;冷盘首读
  仍受磁盘带宽限,热盘(page cache 命中)近乎免费。
- **适用 ARLE**:对应 2026-06-04 lever #3。Rust 侧 `memmap2` + `safetensors::SafeTensors::
  deserialize` 可直接吃 mmap 切片,去掉 `fs::read`+`to_vec` 两次拷 + 淘汰 `shard_cache`。

### 2.2 RunAI Model Streamer(GPU-direct 流式,vLLM `--load-format runai_streamer`)
- **机制**:多线程并发从对象存储/本地盘读权重 **chunk**,边读边经 pinned buffer
  `memcpy_htod_async` 直传 GPU,**read 与 H2D 流水重叠**;对 S3/网络盘尤其有效(隐藏
  网络延迟)。核心是"不等整文件落地,按块就传"。
- **数字(vLLM 公开)**:大模型加载 ~47s → ~7.53s(**~6×**),正是 2026-06-04 lever #1
  引用的 precedent。
- **适用 ARLE**:与 lever #1(pinned async H2D + copy_stream 重叠)同构。ARLE 已有
  `alloc_pinned`/`PinnedHostSlice`/`memcpy_htod_async` + `CudaPipelineFence`
  (`tensor.rs:216`)的原语,缺的是把 loader 的串行循环改成 "读下一个 tensor 的同时异步传
  上一个"。

### 2.3 CoreWeave tensorizer(序列化即流式,vLLM `--load-format tensorizer`)
- **机制**:把模型预序列化成 tensorizer 专有格式(可选加密),加载时**逐 tensor 流式**反
  序列化直传 GPU,支持从 S3/HTTP 直读,无需先落整文件到本地盘。强调"零本地落盘 + 按需
  tensor 粒度"。
- **代价**:需要一次**离线转换**把 HF safetensors 转成 tensorizer 格式(类似轻量 build 步)。
- **适用 ARLE**:与"零离线 build"的设计取向冲突(主文档 §3),除非部署在对象存储分发场景,
  否则收益不及 2.2。**优先级低于 mmap/streamer**。

### 2.4 fastsafetensors + GPU Direct Storage(GDS,vLLM `--load-format fastsafetensors`)
- **机制**:用 NVIDIA **cuFile/GDS** 让 NVMe ↔ GPU 显存 **DMA 直传,完全绕过 host CPU/RAM
  bounce buffer**;safetensors 布局天然适配(连续 tensor 段直接 DMA 到目标 device 偏移)。
- **数字**:在 NVMe + GDS-capable 平台上是理论最优路径(省掉 host staging 整段);但**强依赖
  硬件/驱动**(cuFile、对齐、文件系统支持),非 GDS 环境无收益甚至回退更慢。
- **适用 ARLE**:8×H20 pod 若挂 NVMe 且驱动支持 GDS,是 lever #1 之上的进一步上限;但
  **环境依赖重**,应在 mmap/streamer 落地并 gate 后再评估,不作首选。

### 2.5 多线程 read+deserialize(正交加速,所有方案可叠)
- vLLM `ThreadPoolExecutor` / SGLang `buffered_multi_thread_safetensors_weights_iterator`
  并发跨 shard 读+解析,吃满多核 IO/CPU。对应 2026-06-04 lever #2。
- ARLE 当前 `shard_cache` 是 `RefCell<HashMap>`(单线程),需改 `Arc<Mutex>` 或 per-thread。

---

## 3. vLLM `--load-format` 取值谱(📖,作为完整光谱参照)

`auto`(safetensors 优先,回退 .bin)· `safetensors` · `pt` · `npcache` · `dummy`(随机权重,
跑 perf 不读盘)· `tensorizer` · `runai_streamer` · `fastsafetensors` · `sharded_state`
(每 rank 预切好的分片 checkpoint,**load 期零切片**)· `gguf` · `bitsandbytes` ·
`mistral`。其中 `sharded_state` 值得注意:把 TP 切片**离线**做好,各 rank 直接 load 自己的
文件,完全消除 load-time 切片成本——是 ARLE "读全量再切" 的另一条解法(以离线预切换运行时
零切)。

---

## 4. 选型结论(对 ARLE)

| 方案 | 收益 | 风险/依赖 | 对 ARLE 优先级 |
|---|---|---|---|
| safetensors mmap + per-rank slice(2.1) | 干掉 RAM 复制 + 二次拷;TP 只 fault 自己片 | 低(纯 Rust `memmap2`) | **高**(= 2026-06-04 lever #3,且解 TP 全量读) |
| pinned async H2D + copy_stream 重叠(2.2 同构) | read↔H2D 流水,~6× precedent | 中(改 loader 循环为流水) | **高**(= lever #1) |
| 多线程 read(2.5) | 吃满多核 IO/CPU | 中(`shard_cache` 并发化) | 中(= lever #2) |
| sharded_state 离线预切(§3) | load 期零切片 | 中(需预切工具 + 分发) | 中(8×H20 固定拓扑下值得) |
| tensorizer(2.3) | 对象存储流式 | 需离线转换格式 | 低(与零-build 取向冲突) |
| fastsafetensors/GDS(2.4) | 绕 host 直 DMA,理论最优 | **重硬件/驱动依赖** | 低(mmap 落地后再评估) |

**净结论**:Gap ① 的最优落地是 **mmap(2.1)+ pinned async 流水(2.2)** 两条叠加,二者正是
2026-06-04 doc 的 lever #3+#1。本 survey 不改那份 doc 的 ranking,只补足业界方案的架构细节
与选型依据。

---

## 5. License-or-kill gate(⚗️,沿用 2026-06-04 §A 的 gate)

落地任一 lever 前,先上 **load-phase timer**(env-gated,逐 shard 打 read-µs /
deserialize-µs / h2d-µs,mirror `INFER_M_E13_TRACE`):
- 若 **H2D-bound** → 优先 2.2(pinned async)。
- 若 **IO/parse-bound** → 优先 2.5(多线程)+ 2.1(mmap 去双拷)。
- 若 **RAM-bound**(`shard_cache` 全量常驻)→ 优先 2.1(mmap 免常驻)。

一个 timer 把 "#1 vs #2 vs #3" 从 hypothesis 变 evidence,再选 lever。需 GPU(pod/colab)实测,
本 doc 不下"该做哪个"的最终结论——那是 timer 之后的事。

---

## 关联
- [`2026-06-04-load-and-compile-optimization.md`](2026-06-04-load-and-compile-optimization.md) §A — ARLE 侧成本分析 + ranked lever + gate(本 gap 的 ARLE 半)。
- [[reference_disabled_event_tracking_premature_buffer_free]] — 改 H2D/keepalive 时注意 DeviceContext 禁了 event-tracking,异步传中途的 host buffer 不能提前 drop。
- [[feedback_private_stream_needs_stream_wait]] — 用 `copy_stream` 异步传必须 `stream_wait` 回 compute stream,否则跨流 race。
