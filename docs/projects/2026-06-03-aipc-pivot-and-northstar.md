# 北极星转向:AI PC 推理引擎(agent-workflow benchmark + OS 良民)

**类型:** 战略转向 + rewrite 重排序。叠加在
[`ideal-inference-engine-architecture.md`](2026-06-03-ideal-inference-engine-architecture.md)
与 [`infer-clean-rewrite-plan.md`](2026-06-03-infer-clean-rewrite-plan.md) 之上。
**分支:** `arch/ideal-inference-engine`
**驱动:** ckl —「新的未来是 AI PC 的 AI 推理引擎;benchmark 应该是 agent 的实行工作流,
并且不影响用户的操作系统的使用。」

---

## 0. 转向的本质:服务器引擎 → 个人设备引擎

主流引擎(vLLM/SGLang/TRT-LLM/Dynamo)都假设**专属硬件 + 吞吐最大化**。AI PC 反过来:
**共享设备 + 用户体验最大化**。三条公理因此重写:

1. **设备是借来的,不是占有的。** 引擎跑在用户**正在用**的机器上。占满 GPU/CPU/内存 =
   卡死用户的 OS = 产品死亡。**"不影响 OS 使用" 是硬约束,不是优化项。**
2. **指标是任务,不是 token。** benchmark = **agent 完成一个真实工作流**(多轮 + 工具调用 +
   代码编辑)的端到端表现 + **OS 影响**(前台是否流畅、内存是否还够、有没有热/电拖累),
   不是 guidellm 的 tok/s sweep。
3. **并发是 1,不是 N。** 单用户、单 agent、c=1 是常态([[feedback_metal_focus_c1_local]])。
   服务器的 DP/EP/TP/PP-at-scale、disaggregation 不是 AI PC 的主轴。

---

## 1. 架构含义:新增一个 seam,重排优先级

### 1.1 新一等公民:`ResourceGovernor`(OS 良民层)

理想架构(§4.3 五契约)**补第六个**:engine-core 在 admission 与 step 边界**咨询** governor。

```rust
pub trait ResourceGovernor {
    /// 现在可以再 admit 工作吗?(内存压力 / 前台活跃 / 电池 / 热)
    fn admission_gate(&self) -> AdmissionVerdict;     // Admit | Hold | ShedTo(n)
    /// 这一 tick GPU 给多少预算才不卡前台?(token / 时间)
    fn step_budget(&self) -> StepBudget;
    /// 该让路吗?(前台 app 抢资源 / 内存告警 / 降温)
    fn should_yield(&self) -> bool;
}
```

后端各自提供 OS 信号读取:Metal 读 macOS memory-pressure + wired-limit 余量 + 前台/电池;
CUDA(消费级)读 nvml 显存 + 是否独显/核显;AMD APU 读统一内存压力。**host 侧、契约干净**,
和现有 host-only seam 哲学一致。

这条直接对应已知教训:overlap scheduler **绝不能 busy-spin**(H5 cuEventQuery 2.71M/29s 那次)——
busy-spin 在服务器是浪费,在 AI PC 是**抢用户的核**。governor + 让路是它的架构归宿。

### 1.2 新北极星 benchmark:agent-workflow harness

替代 tok/s sweep。一套跑**代表性 agent 任务**(多轮工具调用 / 代码改 / 检索)的 harness,测:
- **任务维度**:端到端完成时延、每轮 TTFT(交互性命脉)、轮间 KV 复用命中率;
- **OS 影响维度**:峰值内存、前台响应代理指标(并行跑一个 UI/输入延迟探针)、CPU 争用、
  热/功耗。**"引擎跑时用户机器还能不能流畅用" 是 PASS/FAIL gate。**

agent 工作流的引擎侧含义(全部抬为一等):**多轮 session KV 复用**(radix + session cache)、
**低 TTFT**、**快速模型加载/切换**(AI PC 会换模型)、**on-device MoE**(Qwen3.6-A3B 的 experts
在单设备路由,非跨设备 EP)。

### 1.3 异构 AI PC 硅:后端无关比服务器更值钱

AI PC 硅是天然异构:**Apple Silicon(Metal)· 消费 NVIDIA(CUDA)· AMD APU(HIP)·
Intel NPU/XPU**。这正是后端无关 core 的最大价值场景——一个 engine-core,每种 PC 芯片一个薄
executor。**Metal 升为首要后端**(Apple Silicon = AI PC 的典范)。

---

## 2. rewrite 重排序(覆盖 infer-clean-rewrite-plan §R 序列)

| 步 | 原计划 | AI PC 重排后 |
|---|---|---|
| R0 contracts | ✅ | ✅ `322d9d76` |
| R1a engine loop | ✅ | ✅ `37359c14` |
| R1b-d | 港 admission/radix/chunked | **不变**(后端无关);admission 留 `ResourceGovernor` 钩子 |
| **R2** | CudaExecutor first | **MetalExecutor first**(本地验,Apple Silicon 主场) |
| **R2.5** | — | **新增 `ResourceGovernor` seam + Metal 实现**(OS 良民) |
| R3 | 港 model 数值 | 港 model 数值,**Qwen3.6-35B-A3B-4bit MoE 为 canonical** |
| **R4** | frontend | frontend + **agent-workflow bench harness**(新北极星) |
| R5 | parity-gate cutover | cutover gate = **agent-workflow bench + OS-impact** 通过(本地 Metal) |
| R6 | Metal | **CudaExecutor**(消费 NVIDIA;V100/H20 验) |
| R7 | DP-attention/EP, HIP, disagg | **HIP(AMD APU)· Intel NPU/XPU**;on-device MoE 路由。**服务器 DP/EP/TP-at-scale / disagg 显式推迟为可选 server track** |

**de-scope(AI PC 不追,留作可选 server 分支):** 跨节点 TP/PP、跨设备 EP、DP 副本扩缩、
disaggregated P/D。这些是服务器吞吐轴,不是个人设备体验轴。on-device 单设备 MoE 路由保留。

---

## 3. 不变的地基

R0/R1a 已证明的东西**全部仍成立**,只是服务对象换了:
- host-only seam(ForwardPlan/StepOutput/KvPool)—— 异构 PC 硅更需要它。
- overlap loop —— 现在多一层意义:不 busy-spin = 不抢用户的核。
- 后端无关 engine-core —— AI PC 异构硅是它的杀手场景。
- KV-中心 + radix —— 多轮 agent session 复用的命脉。

转向不推翻架构,只**补一个 seam(ResourceGovernor)、换一个 benchmark(agent-workflow)、
调一个优先级(Metal-first,server-parallelism 推迟)**。
