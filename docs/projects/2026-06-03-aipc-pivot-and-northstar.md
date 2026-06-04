# North Star Pivot: AI PC Inference Engine (agent-workflow benchmark + OS good citizen)

**Type:** Strategic pivot + rewrite reordering. Layered on top of
[`ideal-inference-engine-architecture.md`](2026-06-03-ideal-inference-engine-architecture.md)
and [`infer-clean-rewrite-plan.md`](2026-06-03-infer-clean-rewrite-plan.md).
**Branch:** `arch/ideal-inference-engine`
**Driver:** ckl — "The new future is the AI inference engine for AI PCs; the benchmark
should be the agent's actual working workflow, and it must not interfere with the
user's use of the operating system."

---

## 0. The essence of the pivot: server engine → personal-device engine

Mainstream engines (vLLM/SGLang/TRT-LLM/Dynamo) all assume **dedicated hardware +
throughput maximization**. The AI PC inverts this: **shared device + user-experience
maximization**. Three axioms are therefore rewritten:

1. **The device is borrowed, not owned.** The engine runs on the machine the user is
   **actively using**. Saturating GPU/CPU/memory = locking up the user's OS = product
   death. **"Do not interfere with OS use" is a hard constraint, not an optimization
   item.**
2. **The metric is the task, not the token.** benchmark = **end-to-end performance of an
   agent completing a real workflow** (multi-turn + tool calls + code editing) +
   **OS impact** (whether the foreground stays smooth, whether memory is still
   sufficient, whether there is thermal/power drag), not a guidellm tok/s sweep.
3. **Concurrency is 1, not N.** Single user, single agent, c=1 is the norm
   ([[feedback_metal_focus_c1_local]]). The server's DP/EP/TP/PP-at-scale and
   disaggregation are not the AI PC's main axis.

---

## 1. Architectural implications: add one seam, reorder priorities

### 1.1 New first-class citizen: `ResourceGovernor` (OS good-citizen layer)

The ideal architecture (§4.3 five contracts) **adds a sixth**: engine-core **consults**
the governor at admission and step boundaries.

```rust
pub trait ResourceGovernor {
    /// May we admit more work now? (memory pressure / foreground active / battery / thermal)
    fn admission_gate(&self) -> AdmissionVerdict;     // Admit | Hold | ShedTo(n)
    /// How much GPU budget this tick to keep the foreground responsive? (tokens / time)
    fn step_budget(&self) -> StepBudget;
    /// Should we yield? (foreground app contends / memory alarm / cooling)
    fn should_yield(&self) -> bool;
}
// Landed in crates/infer-seam (commit 6cd0afc5) with PermissiveGovernor default.
```

Each backend provides its own OS-signal reads: Metal reads macOS memory-pressure +
wired-limit headroom + foreground/battery; CUDA (consumer) reads nvml VRAM + whether it
is discrete/integrated GPU; AMD APU reads unified-memory pressure. **Host-side, clean
contract**, consistent with the existing host-only seam philosophy.

This maps directly to a known lesson: the overlap scheduler **must never busy-spin** (the
H5 cuEventQuery 2.71M/29s incident) — busy-spin is waste on a server, but on an AI PC it
**steals the user's cores**. governor + yielding is its architectural home.

### 1.2 New North Star benchmark: agent-workflow harness

Replaces the tok/s sweep. A harness that runs **representative agent tasks** (multi-turn
tool calls / code edits / retrieval), measuring:
- **Task dimension**: end-to-end completion latency, per-turn TTFT (the interactivity
  lifeline), cross-turn KV reuse hit rate;
- **OS-impact dimension**: peak memory, foreground-responsiveness proxy metric (run a
  UI/input-latency probe concurrently), CPU contention, thermal/power. **"Can the user's
  machine still be used smoothly while the engine runs" is a PASS/FAIL gate.**

The engine-side implications of agent workflows (all elevated to first-class):
**multi-turn session KV reuse** (radix + session cache), **low TTFT**, **fast model
load/switch** (an AI PC will switch models), **on-device MoE** (Qwen3.6-A3B's experts
routed on a single device, not cross-device EP).

### 1.3 Heterogeneous AI PC silicon: backend-agnosticism is worth more than on servers

AI PC silicon is inherently heterogeneous: **Apple Silicon (Metal) · consumer NVIDIA
(CUDA) · AMD APU (HIP) · Intel NPU/XPU**. This is precisely the highest-value scenario
for a backend-agnostic core — one engine-core, one thin executor per PC chip. **Metal is
promoted to the primary backend** (Apple Silicon = the archetype of the AI PC).

---

## 2. Rewrite reordering (overrides the infer-clean-rewrite-plan §R sequence)

| Step | Original plan | AI PC reorder |
|---|---|---|
| R0 contracts | ✅ | ✅ `322d9d76` |
| R1a engine loop | ✅ | ✅ `37359c14` |
| R1b-d | Port admission/radix/chunked | **Unchanged** (backend-agnostic); admission keeps the `ResourceGovernor` hook |
| **R2** | CudaExecutor first | **MetalExecutor first** (local validation, Apple Silicon home turf) |
| **R2.5** | — | **Add `ResourceGovernor` seam + Metal implementation** (OS good citizen) |
| R3 | Port model numerics | Port model numerics, **Qwen3.6-35B-A3B-4bit MoE as canonical** |
| **R4** | frontend | frontend + **agent-workflow bench harness** (new North Star) |
| R5 | parity-gate cutover | cutover gate = **agent-workflow bench + OS-impact** passing (local Metal) |
| R6 | Metal | **CudaExecutor** (consumer NVIDIA; V100/H20 validation) |
| R7 | DP-attention/EP, HIP, disagg | **HIP (AMD APU) · Intel NPU/XPU**; on-device MoE routing. **Server DP/EP/TP-at-scale / disagg explicitly deferred as an optional server track** |

**de-scope (AI PC does not pursue these; kept as an optional server branch):**
cross-node TP/PP, cross-device EP, DP-replica scale-out, disaggregated P/D. These are
server throughput axes, not personal-device experience axes. On-device single-device MoE
routing is retained.

---

## 3. Unchanged foundation

Everything R0/R1a already proved **still holds**, only the served target has changed:
- host-only seam (ForwardPlan/StepOutput/KvPool) — heterogeneous PC silicon needs it even
  more.
- overlap loop — now with one extra meaning: not busy-spinning = not stealing the user's
  cores.
- backend-agnostic engine-core — heterogeneous AI PC silicon is its killer scenario.
- KV-centric + radix — the lifeline of multi-turn agent session reuse.

The pivot does not overturn the architecture; it only **adds one seam (ResourceGovernor),
swaps one benchmark (agent-workflow), and adjusts one priority (Metal-first,
server-parallelism deferred)**.
