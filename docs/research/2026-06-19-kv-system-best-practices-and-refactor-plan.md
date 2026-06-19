# KV 系统最佳实践调研与重构方案

日期：2026-06-19
范围：整个 KV 系统，不只 L2 DRAM / L3 SSD。包含 host allocator、prefix index、backend device KV、model-specific side state、tier transport、quant/compression、观测和验证。
状态：主线 restore-boundary contract 已落地本地代码；未跑 pod / GPU bench。

## 结论

目标不是再加一个 “KV-tier” 小功能，而是把 KV 生命周期统一成一个系统：

1. HBM 里用固定 page/block pool，进程启动或模型加载时按预算预留大 arena；请求运行时只在 pool 内按 active pages 分配、引用计数和回收。不要每步 GPU malloc。
2. Prefix reuse 的元数据必须和 page/block 生命周期绑定。生产实现普遍用 block table + hash/radix index + refcount，而不是靠 slot 字段或临时 shadow buffer 推断。
3. `KvBatchDescriptor` 只能是执行 descriptor，不是生命周期真相；生命周期真相在 `HostPagedKvPool + RadixCache`，backend 只负责把 page ids 降到 FlashMLA/FlashInfer/MLX/DSv4 layout。
4. L2 DRAM 和 L3 SSD 只能是异步 readmission/write-back pipeline。同步 D2H/H2D、`Vec<u8>` payload、decode 路径阻塞式 SSD read 都不应成为最终形态。
5. Prefix attach 是全局 restore-boundary 规则：所有模型都必须证明 `KV pages + required side state` 足够恢复执行状态。pages-only 足够的后端可以 attach resident page table；不够的后端必须缩短前缀、走 whole-slot route 或 recompute。
6. 量化/压缩是 page format / cold-tier storage policy，不是 KV 生命周期权威。在线 correctness 和离线 bytes saving 必须分开验证。
7. 不够内存时，生产策略是 admission + eviction + readmission + fallback/recompute，不是让 attention/kernel 中途 OOM。

校准后的一句话目标：

不新建一套 KV 框架；把生命周期真相收敛到现有 `HostPagedKvPool + RadixCache`，backend 只做 page image、descriptor lowering、restore-boundary 检查。CUDA/Metal/HIP/Vulkan 的差异只体现在同一个后段接口返回多少个 leading prefix blocks 可恢复。

## 主线

主线只有一条：

`HostPagedKvPool / RadixCache lifecycle -> PrefixBlock restore-boundary mcheck -> mget/mset only for accepted leading blocks -> existing executor lowering -> metrics/async later`

这条线的顺序不能反过来：

1. 先把 `HostPagedKvPool + RadixCache` 定成唯一生命周期真相：谁拥有 page、谁 pin、谁 release、谁可以 evict，全部从这里回答。
2. `RadixCache` 输出统一的 `PrefixBlock::{ResidentPage, DemotedKey}`；core 不再区分模型。
3. backend 用 `reusable_prefix_blocks` 做 mcheck，只返回连续 leading 可恢复长度；不可恢复 tail 不发布、不 promote、不 attach。attention kernel 不消费 `DemotedKey`，promote 后必须先变成 resident page table。
4. page tier 只保留 batch 原语：`demote_prefix_pages` = mset，`promote_prefix_pages` = mget；是否能 mget 先由 mcheck 决定。
5. metrics、async L2、L3 SSD、remote、quant payload 账本都后置；没有主线 restore contract 前不加这些外围层。

对抗式看，主线最容易失败的点不是“KV 不够抽象”，而是抽象太早。当前最该避免的是在没有观测、没有 side-state contract、没有 lowering 边界之前，把 L2/L3/remote/quant 全塞进一个大接口。那会让每个模型都能编译，但没有一个模型能被证明正确。

## 非主线，先删除或禁止新增

1. 新 KV framework crate、`kv-transport` crate、通用 storage engine：等 L2 async 和 remote backend 都有收益证据后再说。
2. `CacheIndex` / `KV Authority` 并行对象：会和 `RadixCache` / `HostPagedKvPool` 争夺真相。
3. 在 `KvBatchDescriptor` 里塞 lifecycle / tier / kernel-specific 字段：它只能是一轮 forward 的只读执行 view。
4. L3 SSD / GDS / NIXL / Mooncake 先行：没有 L2 readmission 指标前，这是把慢路径复杂化。
5. 把 CUDA KIVI、DSv4 PackedBytes、Metal affine INT8 合成一个“通用量化 KV”接口：生命周期可以统一，payload layout 不能伪统一。
6. 任何 backend 只靠 KV pages 做 prefix attach：如果模型还依赖 compressed/indexer/GDR/draft/hidden 等 side state，必须缩短前缀、返回 0 reusable blocks 或走 whole-slot image。
7. graph capture / attention kernel 拥有 pages：kernel 只消费已经 resident 的 page table，不能触发生命周期状态变化。

## 主线验收

第一阶段只算完成到这里：

1. `HostPagedKvPool` / `RadixCache` invariant tests 能证明 page ref、pin、retain、demote、release 没有双重真相。
2. publish 和 attach 都走同一个 `reusable_prefix_blocks`，不可恢复 prefix 不进 radix，不先 promote 再丢。
3. Qwen dense 的 BF16/INT8/FP8、DSv4 PackedBytes、Metal affine INT8 只作为 format 账本存在，不引入通用量化接口。
4. 所有 backend 在 required side state 不完整时明确缩短前缀或返回 0 reusable blocks，不做假 attach。
5. 默认行为由 backend capability 决定；`infer-api` 不按模型名手动禁 prefix cache。

## 对抗式校准

1. **不建新 crate。** `crates/kv-native-sys` 继续只做 substrate；KV policy 不下沉进去。除非 L2 async 和 remote backend 都有测量收益，否则不抽 `kv-transport`。
2. **不建旁路 CacheIndex。** `RadixCache` 就是 prefix metadata owner；缺字段就扩 `Node`/entry state，不能再造一个并行 index。
3. **不建旁路 KV Authority。** `HostPagedKvPool` 是 host page lifecycle owner；缺 IO pin / page state 就补在这里或贴近它的 core 层，不让 scheduler 拼隐式状态。
4. **`KvBatchDescriptor` 只读。** 它描述一次 forward 已经 materialized 的 rows/pages；不能拥有 pages、不能触发 fetch、不能更新 tier。
5. **L2/L3 默认先保守。** GPU cache 已全命中的 workload 开 tier 只会变慢；L2/L3 只能在 prefix miss / preemption / warm-start / multi-replica workload 有 A/B 收益后扩大。
6. **全局不许“只靠 KV page”假 attach。** pages-only 足够就 attach resident page table；不够就必须证明 required side state 已恢复，否则返回更短前缀、0 reusable blocks 或走 whole-slot image。
7. **第一轮只做可证伪改造。** 先补 restore-boundary invariants；没有主线正确性前，不写 metrics/async pipeline。

## 外部最佳实践

### 1. vLLM / PagedAttention：page/block pool 是底座

PagedAttention 的核心不是“省一点内存”，而是把 KV cache 做成 OS virtual memory 类似的 block 管理：logical KV blocks 映射到 non-contiguous physical blocks，attention kernel 通过 block table 读。这样可以按实际 active blocks 分配物理容量，减少预留 max_seq_len × slots 的浪费。

vLLM 的 prefix caching 进一步把每个 KV block 的 identity 建成 hash：parent hash + block tokens + extra hashes。block manager 维护 free queue、hash table、refcount、eviction policy。被多个请求共享的 full blocks 通过 refcount 保护，写入 tail 时走 copy-on-write / 新 block。

ARLE 启发：

- page/block id 是一等公民，不能只是 slot 内部实现细节。
- full block 才能发布到 prefix cache；partial tail 默认不发布，append 前需要 COW 或独占证明。
- cache 命中后应该 attach page table，不该复制 KV 到另一个 contiguous slot。
- eviction 必须只选 refcount=0 的 block；active request 引用的 pages 不能被驱逐。

### 2. SGLang / HiCache：index、pool、storage connector 分层

SGLang 的 HiCache 方向是 GPU HBM、host memory、distributed storage 的层级 KV cache。设计重点不是单个 SSD 文件格式，而是：

- Radix/metadata 负责命中、位置、引用和策略；
- GPU memory pool 负责热 KV blocks；
- storage connector 负责 lower tier read/write；
- policy 控制 prefetch/write-back/eviction/backpressure；
- L3 可以接 Mooncake/NIXL/file-like backend，但 scheduler 不直接做 I/O。

ARLE 启发：

- `RadixCache` 不能只记录 resident/demoted 两态；最终要记录 span identity、tier location、pending transfer、version、format、bytes、refcount。
- scheduler 只能看 ready / pending / miss / recompute-advised，不能同步读 SSD 或 RDMA。
- fetch/store queue depth、page fault、readmission latency 是一等指标，不是 debug log。

### 3. LMCache：connector 化，KV 操作语义清楚

LMCache 把 KV cache 作为 vLLM 之外的 connector/engine 来做，强调 lookup、put、get、pin、transfer、remote/local backend 等接口。它的价值是边界清晰：scheduler/worker 不拥有所有 storage 细节，cache engine 也不直接替代 attention kernel。

ARLE 启发：

- KV 生命周期 API 要表达 “lookup/pin/materialize/release/evict”，而不是只给 backend 一个 `(page, key)` pair。
- lower tier 读写必须支持并发去重：多个请求命中同一个 cold span 时，应该共享一个 in-flight fetch。
- pin/unpin 是避免 eviction 与异步 I/O 竞争的基本能力。

### 4. TensorRT-LLM / production paged KV：capacity 和 scheduler 绑定

TensorRT-LLM 也把 KV cache manager、paged KV、block reuse、sliding window / max attention window 放在 runtime memory 管理里。生产含义是：KV capacity 不是某个 attention op 的局部配置，而是 admission、batching、reuse、eviction 共同依赖的预算。

ARLE 启发：

- page size、tokens per block、sliding window、dtype format 都必须进入统一预算计算。
- scheduler 选择 batch 时要看 pages_needed / free_pages / evictable_pages / pending_fetch，而不是只看 max_running_requests。
- 对 sliding-window 模型，discarded window pages 要尽早释放或转为不可复用，不能污染 prefix index。

### 5. NVIDIA Dynamo / KVBM：block lifecycle 要有状态机

Dynamo 的 KV block manager 把 KV blocks 当成有状态对象管理，配合 KV-aware routing 和跨 worker 复用。重点是 block state、事件、路由和 worker 生命周期解耦。

ARLE 启发：

- 一个 block/span 至少需要状态：free、active、sealed、cached、pinned、evicting、fetching、resident、dropped。
- 不能只用 `u64 tier_key` 表示 “在 tier 里”；缺 pending/failed/version/format 会让 readmission 和 cleanup 模糊。
- 未来多实例 / PD disagg / remote tier 需要 content identity 和 routing metadata，不能依赖进程本地 key。

### 6. NIXL / Mooncake / GDS：transport 是数据面，不是 policy

NIXL、Mooncake Transfer Engine、GDS 这类能力解决的是 registered memory、HBM/DRAM/SSD/NVMe-oF/RDMA/GDS 之间的异步搬运和批量非连续传输。它们不替代 cache policy。

ARLE 启发：

- `kv-native-sys` 应保持 substrate：mmap、shm、WAL、host arena、file/block I/O。
- policy 应留在 runtime/cache orchestrator；transport 只暴露 capabilities、register、batch transfer、poll/abort。
- GDS 只有在 T2 readmission 成为 wall-clock 热点且 payload 对齐时才值得做；否则会先制造文件格式复杂度。

### 7. vAttention：重要替代路线，但不是当前默认方向

vAttention 用 CUDA virtual memory management 让 KV 在虚拟地址上连续、物理页按需分配，从而避免 attention kernel 改成 paged API。它适合已有 contiguous attention kernels 且可接受 CUDA VMM 约束的系统。

ARLE 当前更适合 paged-native：

- CUDA 侧已有 `TokenKVPool`、FlashMLA/DeepGEMM/FlashInfer-style block table 路线；
- Metal/MLX 没有同样的 CUDA VMM 语义；
- DSv4 的 FlashMLA packed KV、DSA indexer、compressed side state 已经不是简单 contiguous KV。

vAttention 可作为未来 CUDA-only experiment，不应阻断当前 paged-KV 统一。

### 8. KV quant/compression：先 correctness，后 bytes

KVQuant/KIVI/FP8/INT8/NVFP4 等路线都证明 KV bytes 可以降；但生产系统通常把这当 page format 或 storage policy，而不是 cache lifecycle 本身。

ARLE 启发：

- online KV quant：先过 needle / long-context / same-config nondeterminism floor，再谈默认。
- persisted/cold-tier compression：可对 sealed pages 做 lossless/typed compression；不要放在 decode hot loop。
- final cache bytes 不是充分指标；要同时看 correctness、TTFT、ITL、H2D/D2H bytes、page fault rate。

## 本地事实层

### 当前 rewrite 栈的 KV seam

- `KvPool` 是 `KvQuery + KvAllocator + KvPrefixStore` 的组合，所有方法都用 host slot ids、page ids、token counts、logical positions 表达；backend device 类型不能穿过 seam。见 `crates/infer-seam/src/kv.rs:1`、`crates/infer-seam/src/kv.rs:11`。
- `KvQuery` 暴露 `page_size`、`free_pages`、`free_tokens`、`seq_len`、`slot_epoch`、`page_indices`、`page_indices_for_token_range`。见 `crates/infer-seam/src/kv_query.rs:7`、`crates/infer-seam/src/kv_query.rs:16`、`crates/infer-seam/src/kv_query.rs:25`、`crates/infer-seam/src/kv_query.rs:34`。
- `KvAllocator` 负责 append、detached pages、free_slot、truncate、migrate。见 `crates/infer-seam/src/allocator.rs:6`、`crates/infer-seam/src/allocator.rs:11`、`crates/infer-seam/src/allocator.rs:14`、`crates/infer-seam/src/allocator.rs:22`。
- `KvPrefixStore` 只表达 retain/release/attach retained pages。见 `crates/infer-seam/src/prefix_store.rs:6`、`crates/infer-seam/src/prefix_store.rs:11`、`crates/infer-seam/src/prefix_store.rs:20`。
- `KvBatchDescriptor` 是 host-only execution view，由 `ForwardPlan + KvPool` 生成，包含 rows、tokens、flat page ids；它要求 row append span 已经由 KV pool materialized。见 `crates/infer-seam/src/kv_batch.rs:1`、`crates/infer-seam/src/kv_batch.rs:14`、`crates/infer-seam/src/kv_batch.rs:57`、`crates/infer-seam/src/kv_batch.rs:164`。

判断：seam 方向正确，但现有 owner 还缺 lifecycle facts。现在 `KvPool` 能分配/attach pages，却不能表达 page state、tier location、pending fetch、format/version、pin lease。

### Host page owner

- `HostPagedKvPool` 是 backend-neutral host-side page bookkeeping；device KV buffers 和 backend-specific physical layouts 留在 executor 下。见 `crates/infer-seam/src/host_paged_kv_pool.rs:1`、`crates/infer-seam/src/host_paged_kv_pool.rs:14`。
- 它维护 `free`、per-slot pages、slot_len、slot_epoch、page_refs。见 `crates/infer-seam/src/host_paged_kv_pool.rs:19`、`crates/infer-seam/src/host_paged_kv_pool.rs:22`、`crates/infer-seam/src/host_paged_kv_pool.rs:24`、`crates/infer-seam/src/host_paged_kv_pool.rs:26`、`crates/infer-seam/src/host_paged_kv_pool.rs:30`。
- append 时按 token 数补 physical pages；free_slot 会释放未被 prefix retain 的 pages；attach_pages 只是把 retained pages 填到 slot。见 `crates/infer-seam/src/host_paged_kv_pool.rs:110`、`crates/infer-seam/src/host_paged_kv_pool.rs:152`、`crates/infer-seam/src/host_paged_kv_pool.rs:184`、`crates/infer-seam/src/host_paged_kv_pool.rs:207`。

判断：这是 “按 active pages 记账” 的正确方向，但它只是 HBM logical allocator，不是完整 KV 系统。

### Prefix index

- `RadixCache` 是 page-sized token blocks 的 host-side prefix cache；partial tail blocks 不发布。见 `crates/infer-core/src/radix.rs:1`、`crates/infer-core/src/radix.rs:3`。
- Node 现在只有 `page_id` 或 `tier_key`，外加 ref_count、last_access、parent、children、evicted。见 `crates/infer-core/src/radix.rs:57`、`crates/infer-core/src/radix.rs:72`、`crates/infer-core/src/radix.rs:75`、`crates/infer-core/src/radix.rs:76`、`crates/infer-core/src/radix.rs:79`。
- `tiered_longest_prefix_match` 可以穿过 demoted blocks，promote 后回到 resident attach。见 `crates/infer-core/src/radix.rs:200`、`crates/infer-core/src/radix.rs:206`。
- `insert` 只发布 full token blocks，并能 revive demoted node。见 `crates/infer-core/src/radix.rs:351`、`crates/infer-core/src/radix.rs:385`。

判断：当前 radix 是好的最小实现，但离生产级 prefix metadata 还差 span identity、format/version、tier location enum、pending transfer、pin lease、checksum/content hash、metrics。

### Engine choreography

- attach prefix 时，engine 先 clamp backend 可复用页，再 retain pages/refcount，再 `kv.attach_pages`。见 `crates/infer-core/src/prefix.rs:29`、`crates/infer-core/src/prefix.rs:46`、`crates/infer-core/src/prefix.rs:65`。
- append 分配失败时会通过 prefix cache reclaim pages。见 `crates/infer-core/src/prefix.rs:100`。
- publish prefix blocks 从 `kv.seq_len(slot)` 和 `kv.page_indices_for_token_range` 得到 sealed pages，再插入 radix。见 `crates/infer-core/src/prefix.rs:123`、`crates/infer-core/src/prefix.rs:134`、`crates/infer-core/src/prefix.rs:141`。
- T1 page tier 现在是 synchronous demote/promote hooks。见 `crates/infer-core/src/prefix.rs:196`、`crates/infer-core/src/prefix.rs:287`、`crates/infer-core/src/prefix.rs:325`、`crates/infer-core/src/prefix.rs:350`。
- Whole-slot route 存在，用于 page-less / side-state-heavy models：preemption 时 `demote_slot`，re-admission 时 `promote_slot`。见 `crates/infer-core/src/planner.rs:133`、`crates/infer-core/src/planner.rs:145`、`crates/infer-core/src/planner.rs:188`。
- 公开 stats 只有 host-tier counters，没有 page fault latency、copy bytes、queue depth、hit tier breakdown。见 `crates/infer-core/src/lib.rs:154`、`crates/infer-core/src/lib.rs:638`。

判断：engine 有正确骨架，但 tier route 现在是同步、页粒度状态贫弱，观测不足。

### CUDA physical pool

- `TokenKVPool` 是 CUDA 物理 page pool；它在 device 上为 K/V/scales/norms/working buffers 预留存储，并维护 `free_pages`、`page_indices`、`seq_lens`、slot_epochs、attach/ref counts。见 `crates/cuda-kernels/src/paged_kv.rs:26`、`crates/cuda-kernels/src/paged_kv.rs:38`、`crates/cuda-kernels/src/paged_kv.rs:81`、`crates/cuda-kernels/src/paged_kv.rs:84`、`crates/cuda-kernels/src/paged_kv.rs:87`、`crates/cuda-kernels/src/paged_kv.rs:93`。
- 注释明确 token-level sequence accounting + physical pages。见 `crates/cuda-kernels/src/paged_kv.rs:4`。
- 当前 BF16/INT8/FP8 page_size=16，TurboQuant page_size=1，PackedBytes page_size=64。见 `crates/cuda-kernels/src/paged_kv.rs:9`。
- `alloc_tokens` 按 active append 需求从 `free_pages` 取新 pages；`seq_lens` 是逻辑长度。见 `crates/cuda-kernels/src/paged_kv.rs:671`、`crates/cuda-kernels/src/paged_kv.rs:690`、`crates/cuda-kernels/src/paged_kv.rs:708`。
- rewrite Qwen path不直接用 device pool allocator，而是 host `CudaKvPool` 分配，executor 每步 `mirror_slot` 把 host page table 映射到 device pool。见 `crates/cuda-kernels/src/paged_kv.rs:741`、`crates/infer-cuda/src/executor.rs:471`、`crates/infer-cuda/src/executor.rs:696`、`crates/infer-cuda/src/executor.rs:731`。
- device pool 提供 D2H/H2D page image copy，但返回/消费 `Vec<u8>`/borrowed payload，并在 promote 末尾 sync。见 `crates/cuda-kernels/src/paged_kv.rs:851`、`crates/cuda-kernels/src/paged_kv.rs:935`、`crates/infer-cuda/src/executor.rs:606`、`crates/infer-cuda/src/executor.rs:623`。

判断：CUDA page pool 的 active-page 机制方向对，但 T1/T2 transfer 和 host/device owner 还没有完全统一；`seq_lens` 是 grounded plane，不能误认为整个 KV architecture 已完成。

### CUDA T1/T2 tier store

- CUDA `CudaKvTierStore` 是 two-level host store：T1 in-RAM map，T2 optional disk spill，payload 是 full `PagedKVPool` page image，不碰 device。见 `crates/infer-cuda/src/kv_tier.rs:1`、`crates/infer-cuda/src/kv_tier.rs:89`。
- T1 budget 由 available RAM + `split_host_tiers` 得到，T2 budget 由 disk free + policy 得到。见 `crates/infer-cuda/src/kv_tier.rs:63`、`crates/infer-cuda/src/kv_tier.rs:80`。
- insert 时 T1 满了会 spill coldest 到 disk；disk full 或 write fail 会拒绝。见 `crates/infer-cuda/src/kv_tier.rs:194`、`crates/infer-cuda/src/kv_tier.rs:225`、`crates/infer-cuda/src/kv_tier.rs:250`。
- read 会 touch T1，T2 read 到 `read_scratch`，不 rewarm；engine promote 后 drop key。见 `crates/infer-cuda/src/kv_tier.rs:269`。

判断：这是可用的局部 tier store，但还不是 production ideal：没有 async queue、没有 in-flight dedupe、没有 copy stream/event、没有 per-tier latency/bytes、没有 content identity，disk key 是进程本地 `u64`。

### Metal KV

- Metal KV pool 只是 `HostPagedKvPool` alias；MLX device KV/GDR 在 executor 内。见 `crates/infer-metal/src/kv_pool.rs:1`。
- Metal paged-prefix read 默认开，并有 hit/fallback counters；DFlash 明确禁用 prefix reuse，因为 prefix mirror 缺 target-hidden + draft state。见 `crates/infer-metal/src/executor.rs:158`、`crates/infer-metal/src/executor.rs:175`、`crates/infer-metal/src/executor.rs:527`。
- Metal SSD tier 是本地 `MetalSsdTier`，有 page/prefix record、budget、LRU、tier_to_logical、read_scratch。见 `crates/infer-metal/src/executor.rs:1350`、`crates/infer-metal/src/executor.rs:1377`。
- Metal store 可读写 prefix snapshot，并通过 `reusable_prefix_blocks` 限制只有有 snapshot 的 prefix boundary 可 attach。见 `crates/infer-metal/src/executor.rs:1893`、`crates/infer-metal/src/executor.rs:1947`、`crates/infer-metal/src/executor.rs:1960`。

判断：Metal 暴露出正确问题：KV pages 不等于可恢复执行状态。这个约束必须进入全局 KV 系统，而不是保留 Metal-only 经验。

### kv-native-sys

- `kv-native-sys` 是 POSIX persistence substrate：file/block I/O、WAL、mmap、shm、host arena。见 `crates/kv-native-sys/src/lib.rs:1`。
- host arena 是 anonymous mmap + bump pointer + free-list，支持 reserve/release/reset 和 reserved bytes。见 `crates/kv-native-sys/src/lib.rs:617`、`crates/kv-native-sys/src/lib.rs:657`、`crates/kv-native-sys/src/lib.rs:731`、`crates/kv-native-sys/src/lib.rs:748`、`crates/kv-native-sys/src/lib.rs:817`。

判断：它应该继续保持 substrate，不要塞 scheduler policy / CUDA stream / tier decision。

## Gap

### Gap 1：生命周期真相分散

现在至少有四个 truth surfaces：

- host `HostPagedKvPool`: slot pages / seq_len / refs；
- `RadixCache`: prefix block tree + resident/demoted key；
- CUDA `TokenKVPool`: device page table mirror + physical buffers；
- Metal/DSv4 executor side state：GDR/draft/compressed/indexer/FlashMLA packed KV。

这会导致两个问题：

- `KvBatchDescriptor` 很容易被误用成生命周期真相，但它只是一步 forward 的 view。
- prefix attach 只证明 KV pages 存在，不证明模型 side state 可恢复。

### Gap 2：tier state 太弱

`RadixCache` 的 `PrefixBlock::{ResidentPage, DemotedKey}` 能支撑最小 T1 promote，但不够支撑生产：

- 没有 `PendingFetch` / `PendingWrite`；
- 没有 source tier：T1/T2/local/remote；
- 没有 content fingerprint；
- 没有 payload format/version；
- 没有 concurrent fetch dedupe；
- 没有 TTL / soft pin / error backoff。

### Gap 3：T1/T2 搬运同步且 copy-heavy

CUDA 当前 demote 是 page-by-page `copy_pages_to_host` -> `Vec<u8>` -> T1/T2；promote 是 read payload -> `copy_pages_from_host` -> `ctx.sync()`。这保证正确，但不是最终性能形态。

最终应当：

- 批量 pages；
- 复用 pinned host regions；
- copy stream + CUDA events；
- read/write queue；
- H2D/D2H bytes 和 latency 可观测；
- SSD payload 对齐，为未来 GDS 留口。

### Gap 4：active pages 分配还没连上 admission

CUDA/host pool 已经按 active pages 分配 physical page ids，但 scheduler admission 还没有把下面这些量作为统一预算输入：

- free pages；
- evictable pages；
- retained pages；
- demoted pages；
- pending fetch target pages；
- max context expansion；
- model side-state per slot；
- tier transfer queue pressure。

所以 “不够怎么办” 还没有完全 production 化：现在能 reclaim/demote，但缺系统级 admission/eviction/readmission decision。

### Gap 5：旧文档和当前 rewrite 栈脱节

`docs/projects/tiered-kv-cache.md` 和 `docs/plans/tiered-kv-hicache-readmission.md` 里很多路径是 `infer/src/...` monolith 时代路径；当前 truth 在 `infer-*` crates。它们原则有用，但不能当实现事实。

## 删除后的设计

### 保留的 owner

- `HostPagedKvPool`：slot `seq_len`、slot page table、slot epoch、free pages、page refs。
- `RadixCache`：prefix token blocks、resident page、demoted key、refcount、LRU、dropped tier keys。
- executor/model code：device layout、format-specific lowering、side-state restore guard。
- `KvBatchDescriptor`：一次 forward 的只读 rows/pages/tokens view。

这已经够做第一轮。缺字段补到这些 owner 旁边；不再新建 manager、authority、adapter trait。

全局统一规则：prefix attach 只看一个问题，`KV pages + model required side state` 是否能恢复到同一个执行边界。Qwen dense 的 required side state 是空，所以 pages-only 成立；DSv4、Metal GDR、DFlash 只是 required side state 不为空的例子，不是特殊分支。

算子消费链路必须保持单向：`RadixCache` 只产出 `PrefixBlock`，storage tier 只保存 page payload，core promote 后得到 resident page id，executor 再把 page id lowering 成算子需要的 page table / block offsets。dense Qwen 的 TileLang/quant paged attention 消费 `PagedKVPool` 的 `kv_indices`/`kv_indptr`/`last_page_len`；DSv4 FlashMLA 当前仍要求 slot FlashMLA band 能由 table 映射到连续 block range，碎片化 page table 要等 Stage B 把 device table 直接传给 pack/index kernels 后才可打开 page-prefix reuse。

### 直接删除或禁止新增的抽象

| 删除项 | 不影响的原因 | 以后何时再加 |
| --- | --- | --- |
| `PagePhysicalState` 类型 | 当前只需要把 state 事实放进 `RadixCache`/tier stats/tests；单独类型没有行为 | async copy 已经落地，需要跨 owner 原子状态机 |
| `TierEntryId` newtype | 裸 `u64` 的问题是语义混乱，不是类型不够；先用命名、注释、metrics 收敛 | durable identity 跨进程/重启时 |
| `SnapshotCapability`/`PrefixSnapshotKind` enum | 现在已有 `reusable_prefix_blocks` 和 whole-slot route；不能恢复就返回 0 | 多个模型真的共享同一 restore 代码时 |
| `TransferOp`/`poll/abort` 接口 | 现有同步 copy 还没量到瓶颈；先加 copy metrics | L2 async A/B 证明需要队列和事件时 |
| backend adapter trait | executor 内已有 lowering 落点；trait 会变成一层转发 | 至少两个 backend 复用同一 lowering contract 时 |
| `kv-transport`/new KV crate | 现在只有本地 tier，policy 已在 runtime；新 crate 只是搬家 | remote backend 有实测收益时 |
| L4 remote / NIXL / Mooncake / EIC 设计 | 第一轮不碰 remote，写了也不会被验证 | 本地 L2/L3 已经有稳定 counters 后 |
| 通用量化 KV 接口 | CUDA KIVI、DSv4 PackedBytes、Metal affine 的 payload 不同；伪统一会害人 | 只有 format 账本先统一，payload 不统一 |

### 先删的逻辑

先删叙述和未来分支，不先删承载行为的代码。

| 先删 | 怎么删 | 不影响的原因 |
| --- | --- | --- |
| backend-specific restore 叙述 | 把 `DSv4/Metal fail-closed` 改成全局 restore-boundary 规则 | 规则本来全局，只是 required side state 不同 |
| `reusable_prefix_blocks` 的 Metal/GDR 特例注释 | 改成“所有 backend 都返回可恢复前缀长度” | 方法本身要保留；它是全局 guard |
| whole-slot route 的 DSv4-only 叙述 | 改成“page-less / side-state-heavy route；当前 CUDA 只有 DSv4 实现” | 代码仍只 dispatch DSv4，但概念不应特殊化 |
| `remote` / L4 / NIXL / Mooncake / EIC 的执行计划 | 从第一轮计划里删除，只保留外部调研来源 | 本轮不验证 remote，写计划会诱导新 crate |
| `TierEntryId` / durable location 计划 | 不加 newtype，只把 `u64 tier_key` 的语义写清楚 | 当前 key 是进程内临时 key，不是 durable id |
| async transfer 接口计划 | 不写 `TransferOp` / `poll` / `abort` | 主线 restore contract 还没有完成归因，不先扩外围层 |
| 通用量化接口计划 | 只留 format 账本，不建 trait | payload layout 不同，统一接口只会隐藏差异 |

### 现在不能删的逻辑

| 不能删 | 原因 |
| --- | --- |
| `reusable_prefix_blocks` | attach 前唯一的全局 restore-boundary clamp；删了会重新允许假 attach |
| `clamp_prefix_to_backend` | 把 radix 命中裁到 backend 可恢复边界；这是 correctness guard |
| `tiered_longest_prefix_match` / `promote_demoted_block` | page-tier readmission 的实际路径；删了 demoted prefix 只能重算 |
| `drain_dropped_tier_keys` | 防 tier store 泄漏；删了 promote/drop 后 key 可能残留 |
| `release_prefix_pages` hook | backend mirror/snapshot 清理点；默认 no-op 但 Metal 已使用 |
| whole-slot `demote_slot` / `promote_slot` route | page-less / side-state-heavy 模型的 preemption route；能统一命名，不能先删 |

### 物理事实放哪里

不做“物理层抽象”。事实直接落在现有结构：

| 事实 | 当前落点 | 第一轮补什么 |
| --- | --- | --- |
| page owner/ref/epoch | `HostPagedKvPool` | invariant tests |
| prefix resident/demoted/ref/LRU | `RadixCache` | demoted/pending/failure counters |
| copy bytes/latency | CUDA/Metal tier copy call site | metrics |
| payload format/page_size/scales | executor/model format code | markdown账本 + tests where cheap |
| restore-boundary safety | `reusable_prefix_blocks` / whole-slot route | required state 不完整就 return shorter prefix or 0 |

### Active pages 的正确语义

“按实际 active pages 分配”不是每个 token 动态调用 `cudaMalloc`。正确形态是：

1. 启动时按 budget 创建 HBM arena / device buffers，容量是可控上限。
2. 每个请求只拿实际需要的 physical pages，slot page table 记录 logical->physical。
3. prefix/shared pages 通过 refcount/pin 保留；append partial tail 前 COW 或证明独占。
4. 空闲 pages 回 free list；eviction 只移动/释放 refcount=0 的 sealed pages。
5. admission 使用 `free + evictable + promotable - pending` 计算是否接新请求。

因此，理想态是 “arena 预留 + active-page accounting”，不是 “max_seq_len × slots 全静态占用”，也不是 “热路径 per-page malloc/free”。

### Tier 命名只保留到 L3

命名建议：

- L1 HBM：attention 直接读的热 pages。
- L2 pinned DRAM：实例本地，异步 promote/demote buffer。
- L3 local SSD：实例本地 cold prefix / warm restart，异步。
- remote/object/PD store：删除出本轮方案。

规则：

- decode/prefill 新 KV 永远先写 L1。
- sealed full pages 才能 publish。
- L2/L3 hit 先变成 pending fetch，fetch complete 后 materialize 到 L1，再进入 runnable。
- L3 不应该阻塞 decode critical path；miss 或 queue backpressure 时应 recompute/fallback。
- L2/L3 的价值只在 prefix reuse、long session preemption、warm start、多副本/PD 工作负载上；GPU cache 已经全命中时，tier 可能是纯损耗。

### Attention kernel 对接

原则：scheduler 不对接 kernel。路径固定为：

`KvPool/RadixCache -> KvBatchDescriptor -> existing executor lowering -> kernel descriptor`

kernel 只吃已经 resident in L1 的 device pages。L2/L3 命中必须先 materialize 到 L1；不能让 attention kernel 同步 page fault、读 SSD、读 host map。

#### 统一输入

existing executor lowering 从 `KvBatchDescriptor` 和本地 state 拿这些字段：

- `mode`: prefill / decode / mixed；
- `slot` / `slot_epoch`；
- `seq_len` / `append_pos` / `append_len`；
- logical page ids in order；
- `page_size`；
- `format`；
- kernel-required side state：position、last_page_len、block table、sparse indices、snapshot boundary。

#### Kernel-specific lowering

| 模型/路径 | Kernel 看到的 KV | Lowering | 关键限制 |
| --- | --- | --- | --- |
| Qwen dense CUDA BF16 | `[page, kv_head, page_size, head_dim]` | pages -> `indptr/indices/last_page_len` | page_size=16，直接 paged attention |
| Qwen dense CUDA INT8/FP8 | quantized data + scales + BF16 work buffer | pages -> quantized decode metadata + fused dequant attention | KIVI K scale必须 ready；V scale per token/head |
| Qwen CUDA TQ | TQ packed indices + norms | 目前 page_size=1，走 TQ decode path | 没有 page_size=16 paged prefill；不能当通用 paged format |
| DSv4 Flash | FlashMLA PackedBytes + DSA/indexer/compressed side state | rows/start_pos -> DSA select -> FlashMLA sparse indices | `PackedBytes` 不是普通 K/V；必须走 DSv4 lowering |
| Metal full-attn | MLX KV arrays / INT8 affine groups | host page ids -> Metal page store / MLX session state | snapshot boundary 不完整就不能 attach |
| DFlash / GDR | KV + recurrent/draft/hidden state | prefix boundary -> snapshot restore | pages-only 不够 |

#### 对接规则

- 每个 kernel descriptor 必须由 executor/model lowering 构造；不要把 FlashMLA/FlashInfer/MLX 字段加进 `KvBatchDescriptor`。
- 每个 format 必须声明 `attention_read_mode`：
  - `Direct`: BF16 paged；
  - `FusedDequant`: INT8/FP8；
  - `PackedSpecial`: DSv4 PackedBytes；
  - `BackendOwned`: Metal MLX；
  - `ColdOnly`: compressed SSD payload。
- prefill 和 decode 可以用不同 kernel，但必须共享同一 host page lifecycle。
- 如果 kernel 需要 contiguous 临时 buffer，只能在 executor 内显式 materialize，且要计入 copy bytes/time；不能让 scheduler 以为它是零拷贝 paged path。
- graph capture 只缓存 kernel descriptor shape，不拥有 pages；slot_epoch/page table 变了必须重建或 fail closed。

### Page format 与量化矩阵

KV page format 必须是 metadata 的一部分：

- BF16；
- INT8/FP8 with scales；
- PackedBytes / FlashMLA MLA latent；
- future TQ/NVFP4；
- cold-tier compressed payload。

规则：

- hot attention kernel 支持的 format 才能 resident；
- cold-tier 可以 lossless compression；
- lossy quantization 只能在 correctness gate 后进入 online path；
- convert/pack/unpack 只能在 executor / 搬运边界，不散落在 scheduler。

#### 热路径格式

| Format | L1 attention | L2/L3 payload | 状态 |
| --- | --- | --- | --- |
| BF16 | direct paged read | raw page image | correctness fallback |
| CUDA INT8 KIVI | fused dequant attention | data + K static scales + V row/head scales | 可热路径；K scale 是 page metadata 的依赖 |
| CUDA FP8 E4M3 KIVI | fused dequant attention | data + K static scales + V row/head scales | 可热路径；Hopper/Ada 更自然，老卡要 gate |
| CUDA TQ2/3/4 | TQ decode kernel | packed indices + FP16 norms + signs | experimental；page_size=1，先不进通用 paged prefill |
| INT4 KIVI PoC | 不作为默认 hot path | packed nibbles + scales | control only；质量 gate 不够 |
| DSv4 PackedBytes | FlashMLA only | opaque 584B/token record | DSv4-only，不可当普通 K/V |
| Metal INT8 affine | MLX backend-owned | MLX triple snapshot / SSD record | Metal-only，不和 CUDA format 合并 |
| cold compressed | 不允许直接读 | lossless compressed block | fetch 后先 decompress/materialize |

#### 量化决策

- **默认参考**：BF16。任何新 format 先和 BF16 跑 correctness gate。
- **优先热路径**：CUDA INT8/FP8 KIVI。它们保持 page_size=16，能接 paged attention/fused dequant。
- **低 bit 路线**：TQ4 优先于 INT4 KIVI。INT4 KIVI 是失败控制组；TQ 有旋转和 norms，但需要补 page_size=16 prefill/metadata 才能进统一路径。
- **DSv4**：不要把 DSv4 FP8/FP4 weights 和 KV PackedBytes 混在一起。weights 走 DeepGEMM/MoE；KV 走 FlashMLA packed latent。
- **Metal**：INT8 是 MLX affine group storage，不复用 CUDA KIVI layout。
- **冷层压缩**：只做 lossless。解压后必须恢复为 hot kernel 支持的 page format。

#### 每个 format 只需要账本字段

- `bytes_per_page_hot`；
- `bytes_per_page_cold`；
- `page_size`；
- `scale_layout`；
- `attention_read_mode`；
- `prefill_supported`；
- `decode_supported`；
- `copy_payload_layout`；
- `correctness_gate`；
- `perf_gate`。

缺任何一项，不能设为默认。不要因此发明通用 format trait。

### 模型 side state

每个可 attach 的 prefix boundary 必须声明 restore contract：

- Qwen dense paged attention：KV pages 足够；promote 后通过 `mirror_slot -> PageMeta -> kv_indices/kv_indptr` 给 TileLang/quant paged attention 消费。
- Qwen hybrid / Metal GDR：需要 recurrent/conv state snapshot。
- DFlash：需要 target KV/GDR + target hidden + draft state。
- DSv4：需要 FlashMLA packed KV + DSA index key/cache + compressed CSA/HCA state + SW ring + rollback-compatible seq_len。

这是全局规则，不是 DSv4/Metal 特判。如果现有 executor/model code 无法证明 boundary 可恢复，`reusable_prefix_blocks` 必须返回更短前缀或 0。

## 删除式重构方案

### P0：restore-boundary contract 和不变量测试

先锁住 correctness 主线。

要做：

- 标 monolith 旧文档为 historical context。
- 为 `HostPagedKvPool` 增加/补齐 active-page invariant tests：alloc/free/retain/attach/truncate/COW。
- 为 `RadixCache` 增加 tier state invariant tests：resident/demoted/publish revive/drop keys/refcount。
- 为 `KvBatchDescriptor` 增加 “descriptor 不是 lifecycle owner” 的 tests：row page range 来自 materialized host pool。
- `PrefixBlock` 成为 core 和 backend 之间唯一的 prefix restore 描述。
- publish / attach / tier promote 前都先走 `reusable_prefix_blocks`。
- required side state 不完整的 backend 返回更短 leading count 或 0。

退出条件：

- CPU-only tests 过；
- CUDA/Metal 前门 typecheck 过；
- 文档列清楚每个 mutable buffer/state owner。

### P1：删隐式状态，不加新接口

目标：把 lifecycle 操作集中，不再让 scheduler / executor 直接拼多个隐式状态。

做法：

- 在现有 `HostPagedKvPool` / `RadixCache` 上收敛命名，不新建大 crate。
- 不引入 `PageLease` / `SpanLease` / `TransferOp`。
- `KvBatchDescriptor` 保持只读 view；构造前必须已 materialize。
- 对 `reusable_prefix_blocks` 增加 snapshot contract 注释和 tests。

删除/替换：

- 删除/收敛把裸 `u64 tier_key` 当完整 location 的调用路径；暂不加 `TierEntryId`。
- 删除 executor 里重复推断 host seq_len/device seq_len 的 silent branch；保留 loud preflight。

### P2：只做 admission 观测，不改策略

目标：不够内存时，在 admission 阶段解决，而不是 allocator/kernel 报 OOM。

做法：主线 contract 稳定后，再把 scheduler 已经能看到的 pages_needed/free/retained/demoted/pending 输出到 stats/log。不要先写 admission verdict enum。

退出条件：

- `/v1/stats`/metrics 能看到 free/retained/demoted/pending pages。

## 验证矩阵

### correctness

- prefix attach：resident hit、demoted hit、partial-tail COW、stale epoch invalidation。
- preemption：page route 和 whole-slot route。
- model side state：
  - Qwen dense pages-only；
  - Metal GDR snapshot boundary；
  - DFlash 禁用或完整 snapshot；
  - DSv4 compressed/indexer/SW/FlashMLA state。
- quant：needle ladder、long context、same-config nondeterminism floor。

### performance

第一轮只跑主线变量：

- baseline：tier off；
- restore-boundary guard：required side state 不完整时返回更短前缀或 0 reusable blocks。
- metrics-only：后置，新增 counters/copy timing 时再单独 A/B。

未来真的进入 L2/L3/quant/compression 时，才按单变量 A/B 加回对应项。

指标：

- TTFT / ITL / output tok/s；
- p50/p95/p99；
- prefix hit tokens/pages；
- L1/L2/L3 hit breakdown；
- page fault count；
- fetch wait ms；
- D2H/H2D bytes；
- SSD read/write bytes and latency（仅 L3 实验启用时）；
- queue depth/backpressure；
- HBM reserved vs active vs retained；
- CPU time and DRAM bandwidth。

### kill rules

- L2 readmission 命中后 wall-clock 不赢 recompute：kill 或调小 scope。
- L3 命中拉坏 p99：默认 off，只保留 warm-start/offline。
- compression 进入 hot loop 后 ITL 退：挪回 cold tier。
- quant correctness 不过：opt-in 保留，禁止 default。
- restore boundary 无法证明：`reusable_prefix_blocks` 返回更短前缀或 0，不能假 attach。

## 第一轮可执行改造

只做到 P0-P2，不碰 remote：

1. 文档标记 monolith-era KV docs 的当前有效/失效部分。
2. 增加 `HostPagedKvPool` / `RadixCache` / `KvBatchDescriptor` 的 lifecycle invariant tests。
3. 明确全局 restore boundary：每个 backend/model 写清 required side state；缺失时 `reusable_prefix_blocks` 返回更短前缀或 0。
4. publish / lookup / attach 都复用同一个 mcheck，不让不可恢复 tail 进入 radix 或 promote。
5. 收敛 `u64 tier_key` 的命名/注释：它只是本进程 tier key，不是 durable location；暂不包新类型。
6. 给现有 KV formats 补 `attention_read_mode / page_size / scale_layout / prefill_supported / decode_supported` 账本。
7. metrics/copy timing 作为下一轮外围层，主线完成后再做。

这轮结束后再决定是否进入 metrics 和 P3 异步 L2。原因很直接：主线 restore contract 不稳时就改 async transfer，会再次变成不可归因的大改。

## Sources

外部一手资料：

- vLLM PagedAttention paper: https://arxiv.org/abs/2309.06180
- vLLM prefix caching design: https://docs.vllm.ai/en/stable/design/prefix_caching/
- SGLang HiCache design: https://docs.sglang.ai/advanced_features/hicache_design.html
- LMCache architecture: https://docs.lmcache.ai/developer_guide/architecture.html
- TensorRT-LLM memory / KV cache docs: https://nvidia.github.io/TensorRT-LLM/reference/memory.html
- NVIDIA Dynamo KVBM design: https://docs.nvidia.com/dynamo/design-docs/component-design/kvbm-design
- NVIDIA Dynamo vLLM KV cache offloading: https://docs.nvidia.com/dynamo/backends/v-llm/kv-cache-offloading
- vAttention paper: https://arxiv.org/abs/2405.04437
- NIXL project/docs: https://github.com/ai-dynamo/nixl

本地主要事实：

- `docs/index.md:80`
- `docs/projects/tiered-kv-cache.md:1`
- `docs/projects/tiered-kv-runtime-flow.md:1`
- `docs/plans/2026-06-07-unified-batched-kvpool-abstraction.md:1`
- `docs/plans/2026-05-25-kv-storage-transport-library-design.md:1`
- `docs/plans/tiered-kv-hicache-readmission.md:1`
- `crates/infer-seam/src/kv.rs:1`
- `crates/infer-seam/src/kv_query.rs:7`
- `crates/infer-seam/src/allocator.rs:6`
- `crates/infer-seam/src/prefix_store.rs:6`
- `crates/infer-seam/src/kv_batch.rs:1`
- `crates/infer-seam/src/host_paged_kv_pool.rs:1`
- `crates/infer-core/src/radix.rs:1`
- `crates/infer-core/src/prefix.rs:1`
- `crates/infer-core/src/planner.rs:129`
- `crates/infer-cuda/src/kv_tier.rs:1`
- `crates/infer-cuda/src/executor.rs:464`
- `crates/cuda-kernels/src/paged_kv.rs:1`
- `crates/infer-metal/src/executor.rs:158`
- `crates/infer-metal/src/executor.rs:1350`
- `crates/kv-native-sys/src/lib.rs:1`
