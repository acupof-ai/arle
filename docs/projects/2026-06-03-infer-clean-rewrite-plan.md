# `infer/` 干净重写计划(壮士断腕)

**类型:** 完整重构思路(greenfield rewrite plan)。
**分支:** `arch/ideal-inference-engine`
**地基:** [`2026-06-03-ideal-inference-engine-architecture.md`](2026-06-03-ideal-inference-engine-architecture.md)(理想架构)
+ [`2026-06-03-backend-seam-redesign.md`](2026-06-03-backend-seam-redesign.md)(现状审计)
**驱动:** ckl —「infer 全部可重写,壮士断腕;太多 AI 垃圾代码」。

---

## 0. 第一原则:断哪只腕,保哪块肉

`infer/src` = **167,062 行**。"全部重写"若指**逐行从零**,等于**重新推导所有数值正确性**——
RoPE/attention/MoE 路由/FP8 KV/量化——把你修过的每个 bug 重新引入一遍(DSv4 long-ctx RoPE、
FP8 KV step-1 divergence、hybrid prefix downgrade、TileLang warp23 NaN…)。那不是断腕,是自杀。

**SOLID 的"壮士断腕"= 砍掉架构骨架的 AI 垃圾,保留并移植被测试锁住的数值与 kernel。**
垃圾是**骨架**(scheduler god-trait、双 scheduler、3 套 metrics、dup/dead/half-state);
肉是**数值**(`model/` 前向数学、`ops/` kernel 封装、radix、kv_tier、collectives、loaders)。

> 若你确实要连数值一起从零(完全不信任现有 model 数学),那是另一个量级更大/更险的工程,
> 需单独 license——本计划**默认移植数值、重写骨架**。这是我的强烈推荐。

### Keep / Port / Rewrite 边界(逐模块,grounded)

| 处置 | 模块(行数) | 理由 |
|---|---|---|
| **KEEP**(crates/,不动) | `cuda-kernels` 44k · `mlx-sys` 81k · `*-spec` · `deepep-sys` · `xgrammar-sys` · `kv-native-sys` · `autograd`/`train`(OPD,独立面) | kernel/桥/配置/训练,非 infer 骨架 |
| **PORT**(逻辑/数值保留,接口重接到新 seam) | `model/` 45k(前向数学)· `ops/` 8.7k · `prefix_cache` ~4k(radix 算法)· `kv_tier/` 7.9k · `distributed/` 3.4k(collectives)· `weight_loader` 2.7k · `quant`/`gguf`/`hf_hub`/`tokenizer`/`sampler`/`speculative` | 被测试锁住的正确性 + 实用工具;重写 = 纯风险 |
| **REWRITE**(AI 垃圾骨架,在理想架构上重建) | `scheduler/` 18k(god-trait/双 sched/half-state)· `backend/` 的 dispatch+bootstrap+MetalScheduler glue · `metrics/`+`metrics.rs` ~5k(3 套合 1)· `server_engine`+`http_server` 的 enum dispatch · `main`/`bin` 入口 | 这就是"AI 垃圾":重复、死代码、半成品、god-trait |
| **NET-NEW** | 5 个 seam 契约 + engine-core 骨架 | 理想架构的地基 |

粗算:**~100k 移植,~50-60k 重写,seam 净新增**。净删的真垃圾(dup/dead/half-state)叠加在重写里。

---

## 1. 目标 crate 图(用编译器强制"后端无关")

现状 `infer` 是**单 crate + cfg**,所以 scheduler 能 `use crate::backend::cuda::PagedKVPool`——
后端无关全靠自律。**重写的核心收益:拆 crate,让"engine-core 不能依赖任何后端"成为编译期事实。**

```
crates/
  infer-plan/     ← ForwardPlan, ForwardMode, Request/Response IR(纯数据, 零后端依赖)
  infer-seam/     ← traits: BackendExecutor · KvPool · Communicator · Sampler · GraphRunner · ModelArch
  infer-core/     ← engine core: scheduler · radix · admission · slot lifecycle · overlap loop · PP microbatch
                    依赖 {infer-plan, infer-seam} —— **编译期无法 mention CUDA/Metal**
  infer-models/   ← ModelArch 实现(Qwen3/35, DSv4):用 Communicator/KvPool 写层;依赖 {seam, *-spec}
  infer-cuda/     ← CudaExecutor · CudaKvPool · NcclCommunicator · CudaGraphRunner;wrap crates/cuda-kernels
  infer-metal/    ← MetalExecutor · MetalKvPool · MetalGraphRunner;wrap crates/mlx-sys
  infer-server/   ← frontend: HTTP/OpenAI · tokenize · detokenize · stream;依赖 {core, 选定 backend}
  infer/          ← 薄 re-export + bins(metal_serve/cuda serve);feature 选 backend
  (KEEP: cuda-kernels, mlx-sys, *-spec, deepep-sys, autograd, train, …)
```

依赖方向**单向向下**:`server → core → {plan, seam} ← {cuda, metal, models}`。
`infer-core` 对 `infer-cuda` 零依赖 → 加 HIP = 新增 `infer-hip` crate,core 不重编。
这是当前 cfg-单-crate **给不了**的保证,也是"AI 垃圾"最难自查的根源。

---

## 2. 五个契约(Rust 签名,地基——先钉死这个)

```rust
// infer-plan: 纯数据,后端无关
pub enum ForwardMode { Prefill, Decode, Mixed, Idle, TargetVerify, DraftExtend }
pub struct ForwardPlan {            // = SGLang ForwardBatch;ARLE LogicalServePlan 升格为此
    pub mode: ForwardMode,
    pub decode_rows: Vec<DecodeRow>,        // slot, last_token, kv_offset
    pub prefill_rows: Vec<PrefillRow>,      // slot, tokens, start_pos, total
    pub microbatch: Option<MicrobatchId>,   // PP
    pub spec: Option<SpecPlan>,
}

// infer-seam: 行为契约,窄。设备/并行/kernel 全在实现里。
pub trait KvPool {                  // 替代直持 PagedKVPool
    fn alloc(&mut self, slot: usize, tokens: usize) -> Result<()>;
    fn free_slot(&mut self, slot: usize);
    fn seq_len(&self, slot: usize) -> usize;
    fn page_indices(&self, slot: usize) -> &[u32];
    fn migrate(&mut self, slot: usize, range: Range<usize>) -> Result<()>;
    fn free_pages(&self) -> usize;  fn page_size(&self) -> usize;  /* ~14 方法, 已测绘 */
}
pub trait BackendExecutor {         // 替代 ModelForward 50-方法 god-trait 的执行片
    type Plan = ForwardPlan;
    fn execute(&mut self, plan: &ForwardPlan, kv: &mut dyn KvPool,
               comm: &dyn Communicator) -> Result<StepOutput>;   // prefill/decode/mixed 统一入口
    fn graph(&self) -> &dyn GraphRunner;                          // CUDA graph 能力
}
pub trait Communicator {            // TP/EP/PP 的 collectives;LayerCommunicator 升格
    fn all_reduce(&self, t: &mut DeviceTensor);                   // TP
    fn all_to_all(&self, send: &DeviceTensor, recv: &mut DeviceTensor); // EP (DeepEP)
    fn send_recv(&self, stage: StageId, t: &mut DeviceTensor);    // PP
    fn topology(&self) -> &Topology;                              // {tp,pp,ep,dp}+rank mesh
}
pub trait Sampler { fn sample(&mut self, logits: &DeviceVec, params: &[SamplingParams]) -> Vec<u32>; }
pub trait GraphRunner {             // padded 桶 + metadata 两段(SGLang 模式)
    fn capture(&mut self, bs: usize);
    fn replay(&mut self, bs: usize, meta_out_of_graph: &PlanMeta) -> Result<()>;
}

// infer-seam: 模型抽象,横跨设备
pub trait ModelArch {
    fn forward(&self, plan: &ForwardPlan, kv: &mut dyn KvPool,
               comm: &dyn Communicator, exec: &dyn LayerKernels) -> Result<Logits>;
}
```

**Overlap 是契约的一部分,不是事后补丁**(§理想架构 §4.2):engine-core 持 future-buffer,
`execute` 返回的 token 以**槽位索引**形式发布,下一步 plan 直接引用,不等回 host。对齐 SGLang
`FutureMap` / vLLM zero-overhead。ARLE 现有 `pending_decode`/`pending_prefill` async 逻辑**移植**进来,
但表达成显式 future 契约,不再绑在 god-trait。

---

## 3. 构建顺序(greenfield 自底向上,每步可独立验证)

并行老树继续服务,新树在 `crates/infer-*` 长出来,**parity-gate 通过前不删老树**(防"重写丢正确性")。

- **R0 契约层** — `infer-plan` + `infer-seam`。纯定义,零依赖。`cargo check` 守。
- **R1 engine-core + mock executor** — 把 scheduler/radix/admission/overlap/slot 逻辑**重写**到
  `infer-core`,泛型 `<E: BackendExecutor, K: KvPool>`。配一个 **CPU mock executor**(产假 token),
  让**连续批/radix/retract/chunked/overlap 全部可在 CPU 上单测**——这是现在做不到的(scheduler 焊死 GPU)。
- **R2 CudaExecutor(wrap,不重写 kernel)** — `infer-cuda` 实现 seam,内部调 `crates/cuda-kernels`
  现有 kernel + `PagedKVPool`(impl KvPool)+ NCCL(impl Communicator)。**kernel 一行不动**。
- **R3 models 移植** — `infer-models` 把 `model/qwen3·qwen35·dsv4` 前向**移植**到 `ModelArch`,
  collective 改走 `Communicator`,KV 改走 `KvPool`。**数值逻辑保留**,只换接口。
- **R4 frontend** — `infer-server` 重写 HTTP/tokenize/detokenize/stream,与 core 进程/线程分离
  (vLLM V1 模式),CPU 活不污染 GPU 循环。
- **R5 parity-gate + cutover** — 新树过门(§4)后,**一个 tranche** 删 `infer/src` 老骨架,
  `infer` 变薄 re-export + bins。no-half-state。
- **R6 MetalExecutor** — `infer-metal` 实现 seam,删 `MetalScheduler`(1.1k),Metal 挂共享 core。
  **此时一个 scheduler 服务 CUDA+Metal。**
- **R7 新轴** — DP-attention+EP(Qwen3.6-MoE)、PP microbatch(`scheduler_pp_mixin` 模式)、HIP
  (`infer-hip`,验证抽象)、disagg(远期)。

---

## 4. 正确性保全(重写最大的命门——SOLID 强制)

**重写丢正确性是头号死因。** 防线:**不删老树,直到新树逐项过 parity-gate。**

1. **黄金 parity 套件(删老树前必绿):**
   - `kv_precision_parity`(BF16 vs INT8/FP8/TQ4 轨迹)在新树通过;
   - `greedy_consistency`(scheduler vs 单请求数值漂移)通过;
   - `e2e` / `e2e_qwen35` 对 `infer/test_data/` JSON baseline 通过;
   - `bench_guidellm` TTFT/ITL/tok-s 在绑定 SLO shape 上**不回归**(H20 pod,CLAUDE.md 强制)。
2. **必保的硬赢行为(逐条带 experience 锚,移植时立回归测试,不靠记忆):**
   - DSv4 long-ctx 输出 inverse-RoPE(`arle_dsv4_output_inverse_rope_cuda`);
   - hybrid 模型 partial-prefix → MISS 降级;
   - chunked prefill 三 hit 模式(exact-full / prefix-of-cached / partial);
   - decode retract/requeue(sglang victim 启发式);
   - FP8 KV step-1 divergence 的已知规避(auto-default 路由);
   - TileLang warp23 NaN 的 `INFER_BYPASS_TILELANG_PREFILL` 路由;
   - wired-limit 自动 pin(Metal);prefill-cap-8 多 shape 默认。
3. **分支纪律:** 长寿分支 = drift 风险。R1–R5 尽快推到 parity-gate;每个 R 步独立 commit;
   热路径步在 pod 补 bench(pending-remote)。

---

## 5. 这次重写按构造消灭的 AI 垃圾(对症 ckl 的痛点)

| 垃圾(现状,已 grounded) | 重写后由构造消除 |
|---|---|
| `ModelForward` 50-方法 god-trait | 拆成 5 个窄 seam,各自单测 |
| 2 个 scheduler(cuda 13.7k + metal 1.1k) | 1 个 `infer-core`,后端是 executor |
| 3 套 metrics(死 SchedulerMetrics + stats EMA + ServerMetrics) | 1 套观测层 |
| `update_ema`×3 / decode fan-out×10 / 两份 readback / 两份 launch-plan | engine-core 重写,无重复 |
| 半成品 unified_scheduler(StepPlan→Logical 往返 + flag) | ForwardPlan 是唯一 IR,无 shadow |
| scheduler 直持 `PagedKVPool` | `KvPool` trait;编译期后端无关 |
| 死 spec 假-100% 接受率焊在 decode | spec 作为 ForwardMode,独立可测 |
| `prefix_cache.rs`(2114)+ `prefix_cache/`(2205)疑似重叠 | 移植时合一 |
| CPU 活与 GPU 循环耦合 | frontend 进程/线程分离 |

---

## 6. 风险与"它会怎么失败"(SOLID 自检)

- **R 把数值也重写了** → 必死(re-derive 数学)。**缓解:R3 是移植,parity-gate 守,§4.2 逐条回归。**
- **长寿分支 drift** → 老树同期还在改。**缓解:分支短;R5 尽快 cutover;只 cherry-pick 关键修复。**
- **overlap/async 微妙逻辑重写出 race** → decode 流水线损坏。**缓解:移植现有 async 证明过的逻辑,
  不重新发明;CPU mock executor 上单测 overlap 不变量。**
- **seam 抽早了(speculative shaping)** → 从 **CUDA↔Metal 两个真实后端共性**抽(已共享 ForwardPlan),
  HIP 验证;非 HIP-first 凭空设计([[feedback_no_speculative_interface_shaping]])。
- **crate 拆分撞 cuda-kernels extraction 治理** → 对齐 `docs/plans/cuda-kernel-crate-extraction.md`
  的 trip-wire;`infer-cuda` 是 wrap 层,不重复 prelude([[feedback_prelude_minimal]])。
- **范围爆炸(167k)** → 严守 keep/port/rewrite 边界;**只重写骨架**。每个 R 步可独立 ship + 验证。

---

## 7. 第一步(R0,可立即开)

`infer-plan` + `infer-seam` 两个新 crate:纯契约定义,零后端依赖,`cargo check` 即守,Mac 可验。
钉死 §2 的 5 个 trait + ForwardPlan——这是整座楼的地基,值得先就签名对齐一轮再铺代码。

> 待你确认两点再开 R0:**(a)** keep/port/rewrite 边界(§0)认可?默认**移植数值、重写骨架**;
> 若要连数值从零,范围/风险翻几倍,需另 license。**(b)** crate 拆分(§1)走还是先在单 crate 内
> 用模块边界过渡?我推荐**直接拆 crate**——编译期后端无关是这次重写最大的、单 crate 给不了的收益。
