# Ideal Architecture for an Efficient, High-Performance, Evolvable Inference Engine

> **SINGLE SOURCE OF TRUTH for the rewrite.** This is the authoritative design + status doc.
> The siblings are subordinate: `backend-seam-redesign` (the SGLang survey that *seeded* the
> seam — historical), `infer-clean-rewrite-plan` (keep/port/rewrite boundary + correctness gate),
> `r3-metal-port-plan` / `r6-cuda-port-plan` (per-backend execution detail),
> `rewrite-verification-targets` (gate definitions), `aipc-pivot-and-northstar` (why AI-PC).
> They are archive/execution-detail, **not authority** — when they disagree with §6 here, §6 wins
> ([[feedback_docs_are_not_truth]]).

**Type:** Architecture north star (survey + ideal design) **+ executed-state record (§6)**.
**Branch:** `arch/ideal-inference-engine`
**Related:** [`2026-06-03-backend-seam-redesign.md`](2026-06-03-backend-seam-redesign.md) (the survey that seeded the seam — historical, not current state; current state is §6 here)
**Method:** `arle-upstream-runtime-scan`. SGLang from source (`sgl-project/sglang@3e681d7`),
vLLM/Dynamo from official docs, the rest from architectural knowledge — **all hypothesis-grade**, to be
license-or-kill'd against local bench/test before landing in ARLE. Sources in the final section.

---

## 0. Mental model: an inference engine is a dataflow pipeline, not a function

A request's lifecycle is a **stage pipeline**:

```
ingress → tokenize → schedule → forward(layer stages) → sample → detokenize → stream
  (CPU)     (CPU)      (CPU)         (GPU)              (GPU/CPU)   (CPU)      (CPU/net)
```

The whole architecture is shaped by three **fundamental tensions**; every mainstream engine's design is an answer to them:

1. **CPU↔GPU must overlap.** tokenize/schedule/sample-prep/detokenize are serial CPU micro-work;
   forward is the big GPU work. If CPU stages and GPU stages run serially, the GPU idles between each step.
   → Solution: **overlap scheduler** (CPU schedules step N+1 in parallel with the GPU running step N).
2. **GPU wants static shapes + large batches.** kernel launch overhead, CUDA graph, and cuBLAS tuning all favor fixed shapes
   and full batches. But requests arrive dynamically and vary in length. → Solution: **continuous batching + chunked prefill + static-shape
   graph (padding to buckets)**.
3. **Devices/parallelism evolve; scheduling logic should not be rewritten alongside.** NVIDIA/AMD/Intel/NPU, TP/PP/EP/DP,
   disaggregation — these are **orthogonal axes of change**. → Solution: **a backend-agnostic engine core +
   a narrow-contract seam** that confines device/parallel/kernel concerns behind a pluggable layer.

The ideal architecture = fixing these three answers into **layers**: the dataflow is a pipeline, execution is a layered seam, axes of change are plugins.

---

## 1. Mainstream engine survey

| Engine | scheduler↔backend separation | continuous batch / chunked prefill | TP / PP / EP / DP | CPU↔GPU overlap | CUDA graph | KV memory | device portability | disagg P/D |
|---|---|---|---|---|---|---|---|---|
| **vLLM V1** | ✅ process-level (API ‖ EngineCore ‖ Worker) | ✅ token-budget, no prefill/decode phase distinction | TP+PP(+EP MoE) | ✅ async, zero-overhead | ✅ piecewise + torch.compile | paged + prefix cache | platform abstraction: CUDA/ROCm/TPU/XPU/CPU | supported (KV connector) |
| **SGLang** | ✅ Scheduler / ModelRunner / AttentionBackend | ✅ chunked, RadixAttention | TP+PP(`scheduler_pp_mixin`)+EP(DeepEP)+**DP-attention** | ✅ overlap scheduler(`FutureMap`) | ✅ standard + piecewise + breakable + CPU graph | paged + **radix** + tiered | `AttentionBackend` 24 impls: NV/**AMD aiter·hip·wave**/Intel xpu·amx/NPU | supported |
| **TensorRT-LLM** | partial (Executor API + compiled engine) | ✅ in-flight batching | TP+PP+EP | ✅ | ✅ | paged + reuse | **NVIDIA only** (AOT compiled, per-shape profile) | supported |
| **NVIDIA Dynamo** | ✅✅ **above the orchestration layer** (backend = vLLM/SGLang/TRT-LLM) | delegated to backend | delegated + multi-node scale-out | — | — | **KVBM** + NIXL point-to-point KV movement | engine-agnostic | ✅✅ **core selling point** (PrefillRouter + KV-aware routing) |
| **DeepSpeed-FastGen** | partial | ✅ **Dynamic SplitFuse** (one of the origins of chunked prefill) | TP | partial | ✅ | blocked KV | NVIDIA | — |
| **Mooncake** | KV-centric architecture | ✅ | — | — | — | **KVStore pool** (separate prefill/decode/KV three pools) | — | ✅✅ disagg pioneer |
| **LMDeploy(TurboMind)** | partial | ✅ persistent batch | TP | ✅ | ✅ | blocked KV | mainly NVIDIA | — |
| **HF TGI** | weak | ✅ | TP | partial | partial | paged(FlashAttn) | NV/partial ROCm | — |
| **llama.cpp** | ✅ **ggml-backend interface** | weak (batch-oriented) | limited | — | partial | simple KV | ✅✅ CUDA/Metal/Vulkan/ROCm/CPU/SYCL | — |
| **MLC-LLM / TVM** | compilation-based | ✅ | limited | — | ✅(codegen) | paged | ✅✅ **compiler cross-hardware** (incl. WebGPU) | — |

**Each engine's architecture in one sentence:**

- **vLLM V1** = splits V0's single process into *API process ‖ EngineCore process ‖ Worker*; the scheduler degenerates into
  a `{request_id: num_tokens}` token-budget dict (no phase), buying maximal overlap of CPU work with the GPU core loop.
- **SGLang** = a device-agnostic Scheduler produces a `ForwardBatch`, the `AttentionBackend` ABC is the narrow kernel seam
  (24 impls spanning all hardware), `FutureMap` keeps scheduling running ahead of the GPU; DP-attention+EP is the standard answer for MoE.
- **TRT-LLM** = peak single-card/single-vendor performance, at the cost of AOT compilation + NVIDIA lock-in, the worst portability.
- **Dynamo** = builds no engine; builds the **dataflow orchestration layer above engines**: KV-aware routing + KVBM + NIXL split prefill/decode
  into independent worker pools, with KV moved point-to-point across VRAM. This is the ultimate expression of the "dataflow mindset".
- **llama.cpp / MLC** = two roads to portability: **hand-written backend interface (ggml)** vs **compiler codegen (TVM)**.

---

## 2. Six industry-converged patterns (just copy them)

Across these 10 engines, the design has converged. These six are the greatest common divisor of "efficient, high-performance, evolvable":

1. **Three-way process/loop split: Frontend ‖ EngineCore ‖ Workers.** API/tokenize/detokenize (CPU, GIL/lock-sensitive)
   are independent from the scheduler+execute core loop (vLLM V1 process separation, SGLang tokenizer_manager separation).
   A single process couples CPU work and the GPU loop = the GPU starves.
2. **Backend-agnostic scheduler + logical ForwardBatch IR; device-specific execution lives below a narrow seam.**
   The scheduler produces data (a plan), never touches kernels. Device diversity is stuffed into `AttentionBackend`+`KVCache` (SGLang)
   / platform+attn backend (vLLM). **One scheduler serves all hardware.**
3. **Token-budget continuous batching, dropping prefill/decode phases; chunked prefill unifies the two.**
   No more "prefill queue first, then decode queue"; instead each step gets a token budget, and prefill chunks and decode rows
   mix in the same forward (vLLM token-budget, SGLang chunked, DeepSpeed SplitFuse).
4. **Overlap scheduler: CPU schedules step N+1 ‖ GPU runs step N, decoupled via a future-token buffer.**
   SGLang `FutureMap`: the scheduler **publishes placeholder future tokens** for step N's output, and step N+1 reads directly
   from the `output_tokens_buf` index, **without waiting for results to return to host**. vLLM calls this "zero-overhead".
   This is the key mechanism that lands §0 tension ① — and the very substance of CPU/GPU natural parallelism.
5. **CUDA graph = padding to static buckets + two-stage metadata.** Shapes are fixed at capture time (batch buckets
   1/2/4/.../max, padded up); per-iter dynamic metadata goes **out-of-graph** (host op,
   `.item()`/`.cpu()` happen here), while recordable static GPU ops go **in-graph**. SGLang implements this as
   `init_forward_metadata_{out,in}_graph` two stages; vLLM uses **piecewise graph** (recording only the parts outside
   attention into the graph, leaving attention dynamic) + torch.compile.
6. **KV-centric memory: paged pool + radix prefix + tiering/offload + (new) disagg KV movement.**
   The KV cache is both the bulk of VRAM and a reuse goldmine. paged (fragmented allocation) + radix (cross-request prefix sharing, SGLang) +
   host/disk tiering + disaggregation (Dynamo KVBM/NIXL, Mooncake) manage KV as a first-class citizen.

---

## 3. Parallelism-axis mental model: which layer DP / TP / EP / PP each lives in

This is the precondition for "good support for DP/EP/TP" — they **are not one switch, but four orthogonal axes, at different layers**. The ideal architecture must
let them **compose orthogonally** (TP×EP×DP×PP) rather than entangle.

| Axis | what it splits | comm primitive | which layer it lives in | nature |
|---|---|---|---|---|
| **TP** tensor parallel | intra-layer: weight matrices split by column/row across cards | per-layer all-reduce / reduce-scatter | model linear/attn layers + `Communicator` | latency-sensitive, needs NVLink-class interconnect; KV is **replicated** within the TP group |
| **PP** pipeline parallel | inter-layer: split L layers into S stages, one card per stage | inter-stage P2P send/recv (activations) | **scheduler orchestrates microbatches** + inter-stage comm | throughput-oriented; introduces bubbles, scheduler must be microbatch-ified (SGLang `event_loop_pp`: async-send/sync-recv to squeeze bubbles) |
| **EP** expert parallel | MoE: experts split across cards, tokens routed to the card holding the expert | all-to-all dispatch/combine (DeepEP) | MoE layer + all-to-all `Communicator` | MoE-specific; load imbalance is the main enemy |
| **DP** data parallel | replicate the whole engine / **replicate attention** (DP-attention) | none between replicas (or within the EP group) | engine replica / attention layer | throughput; **DP-attention is the key for MoE**: attention goes DP (avoiding KV replication across TP), FFN/experts go EP |

**The most important one (the standard answer for DeepSeek/Qwen-MoE, SGLang `enable_dp_attention`):**
On large MoE, **attention = DP (each rank independent KV, not replicated across TP), experts = EP (all-to-all)**.
Pure TP replicates the KV cache N times on each TP rank, exploding VRAM; DP-attention + EP keeps KV at one copy/rank,
which is the path ARLE must take to run Qwen3.6-MoE.

**Architectural implication: the scheduler perceives parallelism only at the plan level** (knows how many ranks/microbatches there are, how tokens route),
**collectives (all-reduce/all-to-all/P2P) live in the `Communicator` below the seam**. The four-axis combination = a
**topology/mesh descriptor** (`{tp, pp, ep, dp}` + rank mapping) injected into the executor, from which the scheduler produces a
microbatch-ified plan.

---

## 4. The ideal architecture

### 4.1 Layered view (who depends on whom)

```
┌──────────────────────────────────────────────────────────────┐
│ Frontend          HTTP/OpenAI · tokenize · detokenize · stream │  CPU, async, separate process/thread pool
│                   (decoupled from the core loop; GIL/locks don't pollute the GPU loop) │
└───────────────────────────┬──────────────────────────────────┘
                            │ Request / StreamDelta
┌───────────────────────────▼──────────────────────────────────┐
│ Engine Core  (backend-agnostic, 1 copy, single writer)         │
│   admission · continuous batch · radix prefix · slot lifecycle · retract │
│   chunked-prefill policy · microbatch orchestration (PP)        │
│   ────────────────────────────────────────────────────────    │
│   produces ForwardPlan(= ForwardBatch IR): mode + row layout + KV indices │
└───────────────────────────┬──────────────────────────────────┘
        ForwardPlan(data contract)│         ▲ future tokens(overlap decoupling)
┌───────────────────────────▼─────────┴────────────────────────┐
│ Executor Seam(narrow contract, trait, compile-time generic)    │
│  • BackendExecutor: execute(plan) → prefill/decode/mixed        │
│  • KvPool:          alloc/free/page/migrate                     │
│  • Communicator:    all_reduce / all_to_all / p2p (TP·EP·PP)    │
│  • Sampler:         logits → tokens                             │
│  • GraphRunner:     capture/replay, padded buckets, two-stage metadata │
└──┬───────────────┬───────────────┬───────────────┬────────────┘
   │ CudaExecutor  │ MetalExecutor │ HipExecutor   │ …(NPU/XPU)   ← each ~1-2k lines, 0 scheduler
   │ +CudaKvPool   │ +MetalKvPool  │ +HipKvPool    │
┌──▼───────────────▼───────────────▼───────────────▼────────────┐
│ Kernels:  attention(flash/paged) · gemm · moe(all-to-all) · quant│  per-device, hand-written or codegen
└────────────────────────────────────────────────────────────────┘

Orthogonal injection: Topology{tp,pp,ep,dp}+rank mesh → determines Communicator wiring + plan microbatch-ification
Orthogonal injection: KV tier(host/disk/remote), disagg role(prefill-only/decode-only) → above KvPool
```

### 4.2 Pipeline view (timeline: how CPU naturally parallels with GPU)

In steady state, the engine core runs one step ahead of the GPU, decoupled by the future-buffer:

```
step:        N-1            N              N+1            N+2
GPU forward: [===== fwd N-1 =====][===== fwd N =====][===== fwd N+1 =====]
CPU core:        [sched N][              ][sched N+1][          ][sched N+2]
                     │ publish future(N)      │ resolve future(N) as input(N+1)
Frontend:    [detok N-2 ‖ tokraw N+1] [detok N-1 ‖ tokraw N+2] …  (another thread/process)

Key: sched N+1 does not wait for fwd N's tokens to return to host — it reads the output-slot index of fwd N in the future buffer.
      The GPU never idles "waiting for CPU scheduling". detokenize/tokenize add another overlap layer on the Frontend thread.
```

CUDA graph lands on the GPU forward lane: decode-step shapes are fixed → replay the captured graph (launch overhead → 0);
dynamic metadata (seq_lens, page table) is prepared by CPU stages outside capture, and the graph only reads static pointers.

### 4.3 Five contracts (the precise definition of the seam)

| Contract | Type | Role | Who implements |
|---|---|---|---|
| `ForwardPlan` + `ForwardMode{Prefill,Decode,Mixed,Idle,Verify,Draft}` | **data** | the sole bridge between engine core ↔ executor; carries token layout/positions/KV indices/spec | engine core produces, executor reads |
| `BackendExecutor` | **behavior (narrow)** | `execute_prefill/decode/mixed(plan, kv, comm)`; async launch/readback overlap implemented here | per backend |
| `KvPool` | **behavior** | `alloc/free/seq_len/page_indices/migrate`; paged/quant/tiered layout here | per backend |
| `Communicator` | **behavior** | `all_reduce`(TP)/`all_to_all`(EP)/`send_recv`(PP); topology-agnostic interface | per backend (NCCL/RCCL/MPI/Gloo) |
| `Sampler` + `GraphRunner` | **behavior** | sample logits→token; graph capture/replay + buckets + two-stage metadata | per backend |

`ModelArch` (layer definitions) spans above them: writing TP/EP collectives via `Communicator`, reading/writing KV via `KvPool`,
driven by `BackendExecutor`. Model and device are decoupled — adding a model doesn't touch the backend, adding a backend doesn't touch the model.

**Proven vs hypothesis (2026-06-03, sharp).** Only **`BackendExecutor` + `KvPool` are exercised by real
forward paths** (Metal MLX + CUDA BF16). `Communicator`, `Sampler`, `GraphRunner`, and `ModelArch` are
**defined but not yet driven by a real caller** (`infer-models` is a skeleton). Per
[[feedback_no_speculative_interface_shaping]] those four stay **hypothesis-grade** — do not lock their
signatures until a second real model / multi-GPU / graph caller forces them. The seam earned its keep on
the two-backend commonality; the rest is a sketch awaiting evidence.

---

## 5. Evolvability: each axis of change = one local plugin

| What you want to add | Where to change | Where **not** to change |
|---|---|---|
| New backend (HIP/ROCm) | `BackendExecutor`+`KvPool`+`Communicator`(RCCL) impls, ~1-2k lines | **not one line in scheduler / engine core** |
| New attention kernel (FlashMLA…) | one attention branch inside an executor | plan / scheduler / other kernels |
| New parallel axis (disagg P/D) | router + KV movement (NIXL-class) + two engine roles | plan IR / KvPool (already abstracted) |
| New model | `ModelArch` impl (using Communicator/KvPool) | backend / scheduler |
| New quant/KV dtype | `KvPool` variant | scheduler / executor trunk |
| New parallel combination (TP×EP×DP) | topology descriptor + Communicator wiring | scheduler trunk (only reads plan microbatch) |

This is "evolvable": change is absorbed by **contract boundaries**, the core loop stays stable. Dynamo goes one step further —
making the "engine" itself swappable and doing dataflow orchestration above it; that is ARLE's next abstraction layer once it matures (after the single-machine engine is solid).

---

## 6. Execution reality (what was actually built — supersedes the incremental L-plan)

**The incremental in-place plan below was abandoned.** The original adoption sequence (L0.1 ForwardPlan
normalization → L0.3 split the `ModelForward` god-trait in place → L1+L2 merge `scheduler/cuda` with
`MetalScheduler`) was superseded by a **greenfield clean rewrite** (壮士断腕): build fresh `crates/infer-*`
beside the legacy tree, prove them, then cut over. Don't refactor the 167k-LOC welded tree in place.

**What exists now (branch `arch/ideal-inference-engine`):**

| New crate | LOC | Status |
|---|--:|---|
| `infer-plan` (ForwardPlan/StepOutput/SamplingParams) | 180 | ✅ |
| `infer-seam` (BackendExecutor/KvPool + hypothesis contracts) | 235 | ✅ proven seam; rest hypothesis (§4.3) |
| `infer-core` (device-neutral scheduler/radix/overlap) | 2.1k | ✅ 18 CPU tests; **replaces 18.3k `scheduler/cuda` + the separate Metal scheduler** |
| `infer-metal` (real MLX Qwen forward) | 3.2k | ✅ **shipped** — 4-config bit-identical parity, 195.5 tok/s, TTFT 6→3, peak RSS 444 MiB |
| `infer-cuda` (clean BF16 Qwen forward) | 1.78k | ✅ local check+clippy green; **GPU parity pending-remote** (pod infra, not code) |
| `infer-models` / `infer-server` | 0.66k | skeleton / in-flight facade |

**The kernel-library lesson (the most expensive miss — encode it).** Port effort is set by the kernel
library's **abstraction level**, *not* the backend name. MLX exposes a **high-level** `qwen35_compiled`
forward → `infer-metal` ports thin (3.2k). `cuda-kernels` exposes only **low-level** ops (bare
attention/gemm/moe) → the CUDA forward orchestration is irreducible and had to be **written clean**
(deletion-refactor: a 25.6k coupled closure → 1.78k single canonical BF16 path), not wrapped. The original
"R2: wrap CUDA, not one kernel line changes" assumption was wrong because it ignored this.

**Honest framing (no self-deception).** `167k → 8.2k` is **misleading** — the new stack is incomplete (no
HTTP server, single CUDA BF16 path, partial model coverage). Cite the *fair, same-functionality* numbers:
scheduler **18.3k → 2.1k** (one device-neutral scheduler vs CUDA-welded + separate Metal); CUDA forward
**25.6k → 1.78k**; new-stack clone density **7.88%**. See
[`../experience/wins/2026-06-03-rewrite-unified-bench-report.md`](../experience/wins/2026-06-03-rewrite-unified-bench-report.md).

**Remaining (real, sequenced):**
1. **CUDA GPU parity** — open correctness gate; blocked on pod toolchain/network infra (use `RUSTUP_TOOLCHAIN=1.92.0` to skip the 1.95.0 download; sync repo + oniond model). Unblock recipe in the unified report.
2. **R5 cutover** — serving facade (OpenAI v1 over new `ServeHandle`, in flight) → flip 4 shallow consumers (`cli`/`agent`/`train`; `arle` bin has zero `infer::` refs) → delete `infer/src`. Final delete gated on (1). Consumer surface compiler-probe-verified: only `server_engine` + `sampler` + `hf_hub`.
3. **DP-attention + EP** (Qwen3.6-MoE), **PP microbatch**, **HIP** (validates abstraction), **disagg** — all post-cutover.

---

## 7. North star + anti-patterns

**North star:** one device-agnostic engine core; backends are thin plugins; CPU overlaps the GPU end-to-end (future-buffer);
static-shape graph; KV as a first-class citizen; DP/TP/EP/PP composed orthogonally (topology injection); the dataflow is an explicit stage pipeline.

**Anti-patterns (ARLE already stepped on or easily will, grounded):**
- ❌ one scheduler per backend (current state: cuda+metal, adding HIP = a third).
- ❌ god-trait seam (`ModelForward` 50 methods, mixing forward+KV+graph+NCCL+sample+spec).
- ❌ scheduler holding a device-concrete type (`PagedKVPool`).
- ❌ CPU stages serial with the GPU (no future-buffer → GPU waits on scheduling).
- ❌ dynamic shapes blocking graph capture (must pad to buckets + two-stage metadata).
- ❌ KV replicated across TP ranks (large MoE must use DP-attention).
- ❌ pre-shaping interfaces for hypothetical backends (extract the seam from **the commonality of the two real backends CUDA↔Metal**, validate with HIP;
  not HIP-first design out of thin air).

---

## 8. Module-design review against elegant-systems standards (2026-06-03, code-grounded)

**Standard applied** (Ousterhout, *A Philosophy of Software Design*): a module is good when it is
**deep** — a **narrow interface** hiding a **deep implementation**. Interface width = cognitive load +
information leakage. Plus classic cohesion (one reason to change per module) / low coupling. The test
isn't "is it split into crates" — it's "is each boundary narrow and does it hide its internals."

### Boundary clarity — clean cross-crate, two interface-width problems

| Contract | Width | Verdict |
|---|---|---|
| `BackendExecutor` (submit/poll/warmup) | 3 methods | **deep ✓** — a narrow interface over the *entire* MLX/CUDA forward. The model boundary; the rewrite's best module. |
| `KvPool` | **19 methods** | **too wide / leaky ✗** — exposes pages/epochs/retain/release/attach/detached/migrate. `infer-core`'s radix cache drives the page-level ops (`retain_pages`/`attach_pages`/`page_indices_for_token_range`, lib.rs:638-870) → paging mechanics leak into the scheduler. |
| `Communicator` (all_reduce/all_to_all/send_recv) | 3 methods | **right idea, wrong shape** — flat, no process-group / device-mesh; can't express TP×EP×PP composition (real NCCL keeps *hierarchical* communicators per parallel dimension). |

**Fix 1 — split `KvPool` by cohesion into two narrow traits.** `KvAllocator`
(`alloc`/`free_slot`/`seq_len`/`free_pages`/`free_tokens` — what the scheduler needs for admission) and
`KvPrefixStore` (`page_indices`/`attach`/`retain`/`release`/`detached` — what the radix cache needs).
Page-id vocabulary is intrinsic to a *host-side* radix cache, but isolating it stops the scheduler from
seeing paging internals. Drop `migrate` until KV tiering has a real caller (currently unused = speculative).

### Internal cohesion — `infer-cuda` mixes 4 concerns in one 1315-line file

`infer-cuda/src/model.rs` holds the forward (`CudaModel`/blocks), op wrappers
(`gemm_batch`/`rms_norm`/`*_attention`/`silu_mul`), the safetensors loader, and KV paging (`PageMeta`).
`infer-metal` already splits these (config/loader/mlx/qwen35/weights). **Fix 2:** split `model.rs` →
`model.rs` (forward) + `ops.rs` (kernel-call wrappers) + `loader.rs` (safetensors) + `kv.rs` (paging),
matching `infer-metal` + the flat-module convention. Same numerics, four cohesive files.

### Multi-card + communication — where each parallelism axis lands

| Axis | Where it lives | Seam fit |
|---|---|---|
| **TP** (tensor) | `all_reduce` inside the model forward, **below the executor seam**; executor owns the rank group | ✓ scheduler stays device/rank-neutral — the deep-module property pays off |
| **EP** (expert/MoE) | `all_to_all` dispatch/combine inside the MoE layer, **below the seam** (DeepEP-class) | ✓ hidden in the executor |
| **DP** (data) | **above the `Engine`** — N `Engine` instances + a router in `infer-server` | ✓ the `Engine` is the DP unit |
| **PP** (pipeline) | microbatches across stages + `send_recv` at boundaries | ✗ **the one real gap** |

**The single multi-card gap is PP.** TP/EP hide cleanly *below* the executor (adding them changes executor
internals, not the scheduler — exactly what the deep seam buys); DP sits *above* the `Engine`. PP is the
exception: the single `inflight: Option` + single executor can't express pipeline stages
(`microbatch: Option<u32>` is an unwired hook). PP needs the **core** to keep ≥2 in-flight microbatches in a
small ring and the executor to be stage-aware. Defer until a model needs PP — but that single-inflight
assumption is what to revisit then. **Fix 3 (when PP lands):** in-flight ring + stage-tagged plan;
**Fix 4:** `Communicator` carries a process-group / device-mesh before TP×EP×PP compose.

### Sampling — the abstraction that's currently *non-functional* for real workflows

`ForwardPlan` carries **no `SamplingParams`**; both executors hardcode `argmax`; the `Sampler` trait is
wired nowhere. The system can only greedy-decode → it cannot run a real agent at temperature>0 / with
stop sequences. **Fix 0 (highest priority, bites the c=1 north-star itself):** thread per-row
`SamplingParams` through `ForwardPlan`, executor calls `Sampler` (temp==0 → argmax fast-path preserves
parity). This is a prerequisite for the serving facade and the cutover.

### Priority (pre-cutover hygiene, by north-star bite)

0. **Sampling** (Fix 0) — only gap that breaks the actual agent workflow. Do first.
1. **Split `KvPool`** (Fix 1) + **split `infer-cuda/model.rs`** (Fix 2) — interface-width + cohesion; cheap, before more callers depend on the wide shapes.
2. **PP ring + `Communicator` mesh** (Fix 3/4) — defer to when a model needs PP; record the single-inflight assumption as the revisit point.

---

## Sources and method

- **SGLang** `sgl-project/sglang@3e681d7` (from source): `base_attn_backend.py` (24 attn backends,
  incl. AMD aiter/hip_radix/wave, Intel xpu/amx, NPU), `forward_batch_info.py` (ForwardMode),
  `mem_cache/memory_pool.py` (KVCache ABC), `managers/overlap_utils.py` (FutureMap),
  `scheduler_pp_mixin.py` (event_loop_pp), `server_args.py` (tp/dp/ep/pp/dp-attention),
  `model_executor/*cuda_graph*` (standard/piecewise/breakable).
- **vLLM V1** design blog: <https://vllm.ai/blog/2025-01-27-v1-alpha-release> (process separation,
  token-budget scheduler, zero-overhead overlap, piecewise CUDA graph + torch.compile,
  platform abstraction).
- **NVIDIA Dynamo** docs: <https://docs.nvidia.com/dynamo/> (disagg P/D, PrefillRouter,
  KVBM, NIXL point-to-point KV movement), NIXL background: <https://www.spheron.network/blog/nvidia-nixl-disaggregated-inference-guide/>.
- TRT-LLM / DeepSpeed-FastGen / Mooncake / LMDeploy / TGI / llama.cpp / MLC-LLM: architectural knowledge,
  **hypothesis-grade**, not read source-by-source.

**§8 module-design standards:** Ousterhout, *A Philosophy of Software Design* (deep modules / narrow
interface / information hiding / cohesion). Multi-card: vLLM & SGLang distributed docs (per-rank engine,
Megatron TP, DP-as-separate-engine), NCCL hierarchical-communicator + DeepEP/EP-API papers
(arxiv 2603.13606) for the process-group/mesh shape.

**Discipline (skill requirement):** the above survey is hypothesis-grade; before landing any L step in ARLE, license-or-kill with local
`bench_guidellm` + `greedy_consistency` + nsys on the binding SLO shape.
narrow-window share ≠ wall-clock impact.
