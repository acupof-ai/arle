# DSv4 decode: launch-bound attack plan (PDL chaining + per-step alloc/memset + fused epilogue)

> Status: Active | 2026-07-07 | evidence: `/host/kern141_decode2.nsys-rep` (07-03, TP=4/EP=4, #141-post, MTP-on)

**Verdict.** DSv4 B=1 decode is **launch-bound, not kernel-bound**. nsys (post-#141)
measures `cudaLaunchKernel` = **39.8% wall** + `cuStreamSynchronize` = **26.6% wall**
= **66% of wall is launch + idle-sync**, with **zero `cuGraphLaunch`** (no CUDA
graph on this path). ~1271 kernel launches/step × ~8458 steps = 10.75M launches.
Every per-kernel vectorization is therefore a wash (the GPU already finishes each
tiny kernel in µs, then waits ~3.6µs for the next launch). **The only levers that
move wall-clock attack the launch count / launch bubbles / per-step sync**, not
kernel compute. This reframes all prior "which kernel is scalar" work as
mis-targeted.

Prior-art guardrails: whole-step CUDA decode graph **re-killed 3×** (wall-neutral,
`errors/2026-06-10-dsv4-wholestep-graph-production-path-wash-rekill.md`), because a
graph removes launches but the wall is the *serial GPU chain + per-step ctx.sync*,
not the launch latency alone. PDL is a **different** mechanism (overlap, not
removal) and is **already partly in use** (`cuLaunchKernelEx` 1.64M calls, 7.8%
wall — the DeepGEMM lane). So this plan extends an existing, working mechanism
rather than re-attempting the killed graph.

---

## Measured baseline (the numbers every option is judged against)

CUDA API wall shares (`cuda_api_sum`):

| API | Calls | Wall % | Note |
|---|---:|---:|---|
| `cudaLaunchKernel` (bare `<<<>>>`) | 8.75M | **39.8%** | the target — no PDL, no graph |
| `cuStreamSynchronize` | 5,620 | 26.6% | per-step host barrier (ctx.sync) |
| `cuMemsetD8Async` | 2.41M | 9.1% | per-step zeroing |
| `cuLaunchKernelEx` (PDL) | 1.64M | 7.8% | DeepGEMM lane — already PDL |
| `cuMemAllocAsync`+`cuMemFreeAsync` | 12.2M | 7.7% | ~7 `HiddenStates::uninit`/layer |
| `cuMemcpyHtoDAsync` | 0.58M | 2.2% | per-step H2D |

~1271 launches/step (10.75M insts / 8458 steps). GPU-busy kernel top shares:
gemv_handwritten 16.6%, fp8_gemv_batch 9.6%, ncclAllReduce 8.6%, mhc_params 7.8%,
grouped_swiglu 7.6%, grouped_down 5.4%, deep_gemm ~13%, FlashMLA 3.1%. **All
kernels are already vectorized/licensed; none is the wall.**

---

## The path (one dependency-ordered chain, NOT a feature menu)

These are not independent options to pick from — they are **one launch-bound
病根 attacked in dependency order**. Per-step allocation must go FIRST because it
both wins on its own AND unblocks PDL (an `cuMemAllocAsync` interleaved in a kernel
chain breaks PDL's stream serialization). Each step is gated by the prior step's
measurement; do NOT stack them in one binary.

```
Step 0 (gate)  →  Step 1 (alloc pool)  →  Step 2 (PDL chain)  →  Step 3 (fused epilogue)
   nsys probe       removes the alloc         needs the clean         cleanup, secondary
   decides 2         that would break 2        no-alloc stream
```

### Step 0 — gate probe (one nsys run, decides whether Step 2 is real)
Confirm on a fresh TP=4 decode capture: (a) the MoE decode kernels are **NOT inside
a stream capture** (`cudaStreamGetCaptureInfo` appears 545K times — rule out that
this path is already captured, which would make PDL redundant), and (b) the
inter-launch gap between the 7 MoE kernels is nonzero (the bubble PDL would remove).
**If captured or gap≈0 → Step 2 is a wash, stop after Step 1.** This is the
license-or-kill for the PDL work; it costs one probe, not a rewrite.

### Step 1 — remove per-step alloc/memset (root; also unblocks Step 2)
`cuMemAllocAsync`+`cuMemFreeAsync` = 12.2M calls / 7.7% wall; `cuMemsetD8Async`
= 2.41M / 9.1% → **16.8% wall, the largest single API share**, larger than any
kernel. Root: ~7 `HiddenStates::uninit` allocs **per layer per token**
(`dsv4.rs:4917-5413`) + 3 tail `DeviceVec::zeros` + the alloc-per-token sampler
(`dsv4.rs:4479` → `ops.rs:432`), vs the zero-alloc scratch pool Qwen3.5 already uses.

Pre-allocate a per-layer decode scratch pool once (extend the existing
`MhcDecodeScratch` to hold the 7 per-layer HiddenStates), reuse across steps →
removes the 12.2M alloc/free + the memsets that zero fresh buffers. **This is also
the precondition for Step 2**: PDL serializes a *stream*; an `cuMemAllocAsync`
between two kernels forces a sync point that breaks the PDL chain. No clean
no-alloc stream → no PDL win.

Prior caveat (`docs/plans/2026-07-02-dsv4-6ms-token-plan.md:67-71`) flagged this as
"may wash — B=1 GPU-bound." That predates this API-table evidence showing
alloc+memset+free = 16.8% wall in a **launch-bound** (not GPU-bound) regime; the
07-03 data licenses re-testing. **A/B required.**

### Step 2 — PDL-chain the now-clean bare `<<<>>>` decode launches
Only meaningful after Step 0 passes AND Step 1 removes the inter-kernel allocs.
Mechanism: `cuLaunchKernelEx` + `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`
lets kernel N+1's CTAs run their prologue during kernel N's tail wave.
**Infra already exists** — `construct_launch_config(..., enable_pdl)` + `launch_kernel`
at `deepgemm_native.cu:485-538` (the DeepGEMM lane, `cuLaunchKernelEx` 1.64M calls
= 7.8% wall, already PDL). Extend it to the bare launches. Pure C++; Rust FFI
signatures unchanged; cudarc not involved.

Device handshake per `__global__`: `cudaGridDependencySynchronize()` before reading
a predecessor's output; `cudaTriggerProgrammaticLaunchCompletion()` after the last
global write.

The DSv4 decode MoE chain (`dsv4_moe_forward_decode_fp8`, `moe.rs:2958-3073`) is 7
bare `<<<>>>` launches/layer, two data-dependent sub-chains 1→2→3 and 4→5→6→7:

| # | kernel | launch site | `__global__` |
|---|---|---|---|
| 1 | `dsv4_count_local_experts` | moe.rs:2958 | dsv4_route.cu:455 |
| 2 | `dsv4_exclusive_scan_i32` | moe.rs:2967 | dsv4_route.cu:492 |
| 3 | `dsv4_pack_local_experts_with_slots` | moe.rs:2994 | dsv4_route.cu:804 |
| 4 | `dsv4_fp8_grouped_swiglu_decode` | moe.rs:3025 | dsv4_fp8_decode_moe.cu:112 (launch :332) |
| 5 | `dsv4_fp8_grouped_down_decode` | moe.rs:3042 | dsv4_fp8_decode_moe.cu:223 (launch :359) |
| 6 | `dsv4_scatter_all_route_slots` | moe.rs:3064 | dsv4_route.cu:1224 |
| 7 | `dsv4_combine_route_slot_outputs` | moe.rs:3073 | dsv4_route.cu:1298 |

Then extend across the attention block: `dsv4_prepare_q/k`, `dsv4_oproj_group_gather`,
`dsv4_fp8_kv_pack`, `dsv4_compressor_update` are bare launches interleaved with the
already-PDL DeepGEMM in `mla_attention_decode_graph` (`attention.rs`) — each breaks
the chain until converted, so converting them restores one unbroken PDL chain over
the whole per-layer forward.

### Step 3 — fuse residual-add + RMSNorm into one kernel (cleanup, secondary)
Per layer, allreduce → separate `add_batch` → separate `rms_norm_batch` = 3 launches
(`dsv4.rs:5209/5230/5243` attn; `5417/5537` MoE); allreduce is NCCL (`tp.rs:367`).
Fold `ops::add_batch` + `ops::rms_norm_batch` into one `fused_add_rmsnorm` kernel
(reduce stays NCCL), 2 launches → 1 at the GLM plain-residual sites (`hc_mult==1`).
Low-single-digit % (fusable part is only the residual+norm launches, not the reduce).
HPC-Ops's multimem/Lamport fused-allreduce is **blocked** by our multi-process TP
(needs single-process peer pointers / multicast VA / cudarc driver API) — do the
plain fused epilogue only, defer multimem.

## Verification protocol (binding, per bench spec)
- Gate on Step 0 before touching Step 2.
- Each step: same-binary same-session A/B, wall-clock ms/committed-token (MTP),
  needle gate (MoE non-determinism) before any default flip.
- Land `docs/experience/wins/` (or `errors/`) with Δ% per step.
- One variable per experiment — do NOT stack the steps in one binary.
