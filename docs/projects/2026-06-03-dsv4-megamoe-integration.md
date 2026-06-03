# DSv4 MegaMoE — integration into the clean rewrite seam

**Status:** design / research only. Implementation is Phase 2 (EP + DeepEP) and
Phase 3 (DeepGEMM) of the multi-GPU port roadmap, **gated on CUDA Phase-0
parity** (single-GPU `CudaModel` forward parity). This doc is the "what makes
mega-scale MoE fast + how we wire it cleanly *below* the seam" plan a strong
engineer can execute.

**Scope anchors**
- Branch `arch/ideal-inference-engine`; seam crates `infer-seam` / `infer-cuda`
  / `infer-models` / `infer-topo`.
- Cross-references:
  [`2026-06-03-multigpu-port-roadmap.md`](2026-06-03-multigpu-port-roadmap.md)
  Phases 2/3/4/5 and
  [`2026-06-03-ideal-inference-engine-architecture.md`](2026-06-03-ideal-inference-engine-architecture.md)
  §3 (parallelism-axis mental model), §4.3 (five contracts), §8 (Communicator
  hierarchy gap = Fix 4).
- Legacy reference to port numerics from: `infer/src/model/deepseek/*`,
  `infer/src/native_deepep.rs`, `crates/deepep-sys`, `crates/deepseek-spec`.

This is a design doc, not landed code. Every perf claim is cited to upstream
SGLang/DeepSeek sources; every "this repo does X" claim cites a file:line in the
legacy tree. Nothing here is benched — perf numbers are upstream-reported, and
the EP=8 verification shape (§C.6) is the evidence gate before any default flip.

---

## Part A — What makes mega-scale MoE fast (SGLang + DeepSeek kernels)

The MoE forward on a sharded model is a five-stage pipeline. SGLang names the
stages explicitly: **dispatch → pre-permute → core runner → post-permute →
combine** (`BaseDispatcher` / `MoeRunner` / `PermuteMethodPool`,
[SGLang Expert Parallelism docs][sgl-ep]). The cost lives in two places — the
all-to-all (dispatch/combine, comm-bound) and the grouped expert GEMM (core
runner, compute-bound). Every technique below attacks one of those two, or
overlaps them.

### A.1 DeepEP — the all-to-all dispatch/combine library

DeepEP is DeepSeek's EP communication library: "high-throughput and low-latency
all-to-all GPU kernels, also known as MoE dispatch and combine"
([DeepEP README][deepep]). It has two operating modes, and the mode is chosen by
serving phase:

| Mode | Phase | Why fast | Transport |
|---|---|---|---|
| **normal** (high-throughput) | prefill | maximizes BW on long token batches; "symbolic" output shapes (incompatible with CUDA Graph) | NVLink intranode + RDMA internode |
| **low-latency** (pure RDMA) | decode | minimal delay, **fixed pre-allocated shapes → CUDA-Graph-capturable**; pairs with masked grouped GEMM | pure RDMA |

Reported bandwidth: NVLink dispatch ~150+ GB/s, RDMA dispatch ~45+ GB/s
intranode; low-latency mode hits microsecond-class dispatch on H800
([DeepEP README][deepep]). The PD-disaggregation deployment uses **normal
dispatch on the prefill server, low-latency on the decode server**
([LMSYS large-scale EP blog][lmsys]).

Two fastness mechanisms beyond raw BW:

- **Double-buffered (ping-pong) low-latency overlap.** Decode dispatch/combine
  use ping-pong buffers so the next micro-step's RDMA overlaps the current
  step's compute ([DeepEP README][deepep]).
- **Hook-based comm/compute overlap using zero SMs.** A "receiving hook"
  interface runs RDMA traffic in the background *without occupying any SM*, so
  the all-to-all does not steal compute resources from the expert GEMM
  ([DeepEP README][deepep]). Newer DeepEP ("Normal-SMFree", `ElasticBuffer`)
  pushes this further — V3-class workloads drop SM usage from ~24 to 4–6 while
  matching or beating throughput ([DeepEP README][deepep]).

### A.2 DeepGEMM — the FP8 grouped expert GEMM (and "Mega MoE")

DeepGEMM is DeepSeek's FP8/FP4 GEMM library with **fine-grained (block) scaling**
([DeepGEMM repo][deepgemm]). The MoE-relevant pieces:

- **Grouped GEMM, M-axis only.** "Unlike CUTLASS grouped GEMMs, DeepGEMM groups
  only the M-axis, while N and K stay fixed" — tailored to MoE where all experts
  share `[N=intermediate, K=hidden]` shape ([DeepGEMM guide][dg-guide]). Two
  layouts:
  - **Contiguous** (`m_grouped_fp8_gemm_nt`) — prefill/training. Tokens for all
    experts concatenated along M; each expert segment aligned to the GEMM M
    block (`get_mk_alignment_for_contiguous_layout()`).
  - **Masked** (`m_grouped_fp8_gemm_nt_masked`) — decode. A mask tensor marks
    valid tokens per expert; **shapes are fixed so CUDA Graph survives** even
    though per-expert token counts vary per step. This is exactly what pairs
    with DeepEP low-latency dispatch ([DeepGEMM guide][dg-guide],
    [SGLang DeepSeek docs][sgl-ds]).
- **FP8 fine-grained scaling + 2-level accumulation.** E4M3 inputs, per-tile
  scale (one per ~128 K-columns), WGMMA accumulates in FP32, output cast to
  BF16. Recovers outlier accuracy without a BF16 fallback. SM90 uses FP32 scales
  in a TMA-aligned transposed layout; SM100 (Blackwell) uses packed UE8M0
  exponent-only scales ([DeepGEMM guide][dg-guide]).
- **"Mega MoE" — `fp8_fp4_mega_moe`** (April 2026). A *single fused mega-kernel*
  doing: EP dispatch → FP8×FP4 linear-1 → SwiGLU → FP8×FP4 linear-2 → EP combine,
  overlapping NVLink comm with tensor-core compute by treating dispatch/combine
  as symmetric scheduled work on shared (symmetric-memory) buffers
  ([DeepGEMM guide][dg-guide]). **Hard requirement: already-on-DeepEP +
  PyTorch ≥2.9 symmetric memory; FP8×FP4 only at launch** — "not a free win if
  deployed standalone." This is the literal "MegaMoE" of the task title: one
  kernel over all local experts *with* the comm fused in, rather than
  dispatch / per-expert GEMM / combine as separate launches.

### A.3 Expert-parallel layout + load balancing (EPLB)

- **EP sharding.** Expert weights are split across cards; tokens are routed
  (all-to-all) to the card that owns the selected expert ([arch §3][arch]). The
  enemy is **load imbalance** — a hot expert serializes its owner rank.
- **EPLB redundant experts.** SGLang/DeepSeek replicate hot experts: e.g. 256
  base experts + 32 redundant = a 288-expert pool, placing duplicates of
  frequently-hit experts on otherwise-idle ranks. Rebalance happens in three
  stages: system loading → async weight transfer → device-to-device copy.
  Reported **1.49× prefill / 2.54× decode** speedup from removing the imbalance
  tail ([LMSYS large-scale EP blog][lmsys]).
- **Shared expert.** DeepSeek MoE layers have routed experts *plus* a shared
  expert run for every token. The shared expert is dense (no routing) and is
  computed locally / overlapped with the routed all-to-all rather than going
  through dispatch ([SGLang DeepSeek docs][sgl-ds], arch §3).

### A.4 Two-batch overlap (TBO) — hide the all-to-all behind compute

TBO splits one batch into two micro-batches so **compute and all-to-all comm
overlap**: while micro-batch A's experts compute, micro-batch B's dispatch/combine
runs on the wire (and vice-versa). It also halves peak activation memory.
Reported **+27–35% prefill throughput**, and decode **17,552 vs 12,929 tok/s
(+35%)** ([LMSYS large-scale EP blog][lmsys]). Key implementation rule: submit
GPU compute *before* launching the CPU-blocking comm call, so the GPU never
idles waiting on the host ([LMSYS large-scale EP blog][lmsys]). The MoE all-to-all
overlapping with attention is the same idea applied across the layer boundary
(attention of step *t+1* overlaps experts of step *t*).

### A.5 Routing / topk + scatter-gather (the permute pipeline)

The dispatch needs the tokens grouped by destination expert; the combine needs
them un-grouped back to original positions. SGLang isolates this as the
**pre-permute / post-permute** stages around the core runner, registered via
`PermuteMethodPool` so a backend can swap a fused Triton permute kernel in
([SGLang Expert Parallelism docs][sgl-ep]). The fast path:

1. router GEMM → logits → topk (sigmoid/softmax scoring, optional bias for
   aux-loss-free balancing) → per-token expert ids + weights;
2. **permute / scatter** tokens into per-expert contiguous segments (for
   contiguous grouped GEMM) — DeepGEMM-paired permute is a custom Triton kernel
   ([SGLang DeepSeek docs][sgl-ds]);
3. grouped GEMM over all local experts (one kernel, not per-expert);
4. **unpermute / gather** + weighted reduce back to token order (the combine).

---

## Part B — DSv4 MoE in this repo (the legacy numerics to port)

### B.1 Architecture (deepseek-spec)

`DeepSeekV4Config` (`crates/deepseek-spec/src/v4.rs`) carries the MoE shape:
`n_routed_experts`, `n_shared_experts`, `num_experts_per_tok` (top-k),
`moe_intermediate_size`, `routed_scaling_factor`, `norm_topk_prob`,
`scoring_func`, `topk_method`. The local 1B init checkpoint
(`infer/models/dsv4-mini-1B-init/`) is the dev shape; the production V4 target is
DeepSeek-V4-Pro: **384 routed + 1 shared experts, top-6 per token**
([V4-Pro config / overview][v4pro]), versus the V3.2 family's 256+1, top-8
([SGLang DeepSeek docs][sgl-ds]). The repo's unit fixture uses 8 experts /
top-2 (`infer/src/model/deepseek/config.rs:271`), too small to expose EP
imbalance — keep it for numerics, not perf.

Routing modes (`crates/deepseek-spec/src/v4.rs`,
`DeepSeekV4MoeRoutingKind`): **LearnedBias** (gate logits + per-expert bias,
the aux-loss-free balancing variant) and **Hash** (tid→eid table). Scoring:
`softmax` / `sigmoid` / `sqrtsoftplus` (`mlp.rs:2051`). DSv4 also carries an
MTP (multi-token-prediction) head (`num_nextn_predict_layers`) and the
compressed/sparse + hybrid-compressed attention modes — orthogonal to MoE but
present in the same forward.

`Shard` enum (`crates/deepseek-spec/src/lib.rs`) already has the
`ExpertParallel { dim }` variant: "the expert axis is owned by EP/MoE-EP
placement rather than tensor-parallel slicing." This is the layout authority the
new `infer-topo` should consume.

### B.2 Router / gate

`mlp.rs:1858`+ and `:2105` (`ffi::dsv4_route_cuda`): a single CUDA kernel takes
gate logits (`gate_weight` GEMM, shape `n_routed_experts × hidden`), optional
`gate_bias` (LearnedBias) or `gate_tid2eid` (Hash), and emits per-token
`route_indices [tokens, topk]` + `route_weights [tokens, topk]`. It folds
scoring (softmax/sigmoid/sqrtsoftplus), top-k selection, optional `norm_topk_prob`,
and the `routed_scaling_factor` multiply into one launch. This is **port-clean
numerics** — a deterministic kernel with a JSON-comparable output.

### B.3 Two EP transports already exist in legacy

The legacy DSv4 has **two** all-to-all implementations (a half-state the rewrite
should converge to one):

1. **Hand-rolled NCCL path** — `mlp.rs` `forward_*_routed_gpu` using
   `comm.moe_all_gather_i32` (`:4095`) for the per-rank/per-expert count exchange
   and `comm.moe_reduce_scatter_bf16[_overlap]` (`:5058`–`:5221`) for the combine.
   Has an overlap variant (`moe_reduce_scatter_bf16_can_overlap`, `:5058`). Combine
   strategy is gated by `dsv4_reduce_scatter_combine_enabled()` (`:1485`) and
   `ARLE_DSV4_MOE_COMBINE` (`allgather` vs `reduce_scatter`, `:1431`).
2. **Native DeepEP path** — `forward_native_deepep_routed_gpu`
   (`mlp.rs:5355`). Calls `Buffer::dispatch` (`:5609`/`:5637`) then
   `Buffer::combine` (`:5889`/`:5912`) via `deepep-sys`. Dispatch params:
   `num_sms` default **20** (`dsv4_native_deepep_num_sms`, `:1215`; must be even),
   `nvl_chunked_send=6 / nvl_chunked_recv=256`. The combine passes the caller's
   **compute stream** so it does event-based `stream_wait` instead of host
   `cudaStreamSynchronize` (`deepep-sys/src/lib.rs` `CombineParams.compute_stream`).

The startup contract (`forward.rs:728`, `:890`) **hard-requires**
`ARLE_DSV4_MOE_BACKEND=native-deepep` and `ARLE_DSV4_EXPERT_BACKEND=deepgemm`
for the current ARLE target — the hand-rolled path is the debug/fallback lane.

### B.4 deepep-sys / native_deepep (the DeepEP wrapping)

`crates/deepep-sys` is a **torch-free** Rust binding (`src/lib.rs`). Build is
env-gated: `ARLE_DEEPEP_DIR` → `build.rs` nvcc-compiles DeepEP's
`csrc/kernels/{intranode,layout,runtime}.cu` + a thin C wrapper
(`csrc/deepep_buffer.{hpp,cpp}`) into a static archive; **unset → `deepep_stub`
cfg, every call returns `NotBuilt`** (so non-CUDA hosts still compile). Two facts
that bound the integration:

- **Intranode-only today.** The header (`deepep_buffer.hpp:2`) wraps
  `intranode::{barrier, notify_dispatch, dispatch, cached_notify_combine,
  combine}` and `layout.cu`. `build.rs` compiles `intranode.cu` + `layout.cu` +
  `runtime.cu` only, `sm_90` gencode. **No internode RDMA, no low-latency-mode
  kernels are wrapped.** `Buffer::sync` asserts `world_size ∈ [2, 8]`
  (`src/lib.rs`). → legacy EP = **single node, NVLink, ≤8 ranks** (normal
  dispatch only).
- **Handle exchange rides the existing EP NCCL group.** `NativeDeepEp::boot`
  (`native_deepep.rs`) all-gathers 64-byte IPC handles + device ids over
  `NcclGroup::all_gather_bytes`, then `Buffer::sync`. One `NativeDeepEp` per
  process behind `Arc<Mutex<Buffer>>`; forward grabs the lock per dispatch/combine.

### B.5 DeepGEMM expert GEMM (build-gated FFI)

The expert FFN is the DeepGEMM grouped GEMM, dispatched by weight format and a
runtime M-threshold (`mlp.rs:755` `dsv4_run_grouped_block_scaled_gemv`):

- FP8 path: `ffi::dsv4_fp8_grouped_gemm_batch_cuda` when
  `max_count >= dsv4_grouped_gemm_m_threshold()` (the contiguous/large-M layout),
  else `dsv4_fp8_grouped_gemv_batch_cuda` (the small-M / decode layout)
  (`mlp.rs:795`–`:830`). Same split for the fused gate+up "pair" kernel
  (`dsv4_run_grouped_block_scaled_gemv_pair`, `:863`) and the route-scatter
  variants (`:988`, `:1063`).
- FP4 path: `dsv4_fp4_grouped_gemv_batch_cuda` (`:830`).
- Grouped weight pointers (`[local_experts]` device-ptr arrays for w1/w2/w3 +
  block scales) are built once and cached (`dsv4_build_grouped_weight_ptrs`,
  `:512`; `dsv4_build_deepgemm_expert_cache`, `:706`). Format detected from the
  raw FP8/FP4 qweight + block-scale tensors (`dsv4_grouped_format`, `:313`).
- The M-threshold dispatch is the legacy stand-in for DeepGEMM's
  contiguous-vs-masked choice (large-M contiguous / small-M masked-class) — but
  it is **GEMV-shaped, not the masked-grouped-GEMM-under-CUDA-Graph** that
  upstream decode uses. The native DeepGEMM is build-gated behind
  `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1`; without it the bridge stub links and
  returns `CUDA_ERROR_NOT_SUPPORTED` (see memory:
  `feedback_deepgemm_build_flag_stub`).

**Numerics-port summary (B → C):** the router kernel, SwiGLU-clamped activation
(`ops::dsv4_swiglu_clamped_batch_into`, `mlp.rs:100`), shared-expert add
(`add_shared_expert_with_scratch_into`, `:1739`), and the FP8/FP4 grouped GEMM
FFI are **port-clean** (deterministic, JSON-comparable). The two EP transports
and the scratch-buffer choreography (`DeepseekNativeDeepEpRuntimeScratch`,
`state.rs:165` — ~25 device buffers) are **rebuild-clean** behind the new
`Communicator`.

---

## Part C — Integration design into the new seam

### C.1 Where each piece lands (the layer map)

Per arch §4.3 the seam has five contracts; MoE touches three of them. The MoE
forward lives entirely **below the executor seam** — engine-core/scheduler never
sees experts, dispatch, or FP8 (validated extensibility verdict,
[roadmap §"易于扩展"][roadmap]).

| SGLang/DeepSeek technique | Seam home | Contract |
|---|---|---|
| DeepEP dispatch/combine (A.1) | `Communicator` impl in `infer-cuda` | `Communicator::all_to_all` (+ low-latency variant, see C.4) |
| DeepGEMM grouped/masked GEMM (A.2) | `infer-cuda` ops variant (`ops.rs`) | called inside `ModelArch::forward`; no seam method |
| Mega MoE fused kernel (A.2) | `infer-cuda` ops variant (later) | a fused op the MoE `ModelArch` may call instead of dispatch/GEMM/combine |
| EP sharding + EPLB (A.3) | `infer-topo` (placement) + loader | topology descriptor → which experts this rank owns |
| Shared expert (A.3) | the MoE `ModelArch` (`infer-models`) | dense local compute, no collective |
| TBO / attention⊗MoE overlap (A.4) | the MoE `ModelArch` + executor microbatch | uses `Communicator` + plan microbatch hook |
| router topk + permute/unpermute (A.5) | `infer-cuda` ops (port `dsv4_route_cuda` + scatter) | inside `ModelArch::forward` |

The MoE block is a `ModelArch` (or a layer within the DSv4 `ModelArch`) that:
router → permute → `comm.all_to_all` (dispatch) → grouped GEMM (`infer-cuda`
ops) → SwiGLU → grouped GEMM → `comm.all_to_all` (combine) → unpermute + shared
expert. This is the five-stage SGLang pipeline (A.0) expressed against the seam.

### C.2 Port vs rebuild

**PORT (tested numerics, copy with parity tests vs JSON baselines):**
- `dsv4_route_cuda` router kernel (scoring + topk + bias/hash + scaling) — B.2.
- SwiGLU-clamped activation + shared-expert add — B.5.
- FP8/FP4 grouped GEMM + pair + route-scatter FFI kernels — B.5. These are
  `crates/cuda-kernels/csrc` kernels reached through `infer-cuda/ops.rs`.
- `deepep-sys` crate as-is (the C wrapper + `Buffer` lifecycle) — it is already
  torch-free and seam-agnostic; only its *caller* moves.
- The DSv4 config/spec (`deepseek-spec`) — `infer-topo` reads `ExpertParallel`
  from it.

**REBUILD (clean, behind the seam):**
- The two competing EP transports (B.3) collapse to **one**: a `Communicator`
  impl whose `all_to_all` wraps `deepep-sys` `Buffer::dispatch/combine`. Drop
  the hand-rolled NCCL all-gather/reduce-scatter lane and the
  `ARLE_DSV4_MOE_COMBINE` knob — converge on DeepEP (the startup contract
  already mandates `native-deepep`, B.3, so this just deletes the dead fallback).
  (Per `feedback_no_half_states` / first-principles: one canonical flow.)
- The ~25-buffer `DeepseekNativeDeepEpRuntimeScratch` choreography → a tidy MoE
  workspace owned by the `ModelArch`, allocated through the executor's allocator,
  not a model-private `Option<…>` grab-bag.
- The M-threshold GEMV dispatch (B.5) → an explicit **contiguous (prefill) vs
  masked (decode)** grouped-GEMM selection that mirrors DeepGEMM's two layouts
  (A.2) and is **CUDA-Graph-safe in the masked/decode path** (today's GEMV path
  is not the masked-under-graph shape).

### C.3 The seam extension that's actually required (Fix 4)

The blocker is recorded in arch §8 and roadmap §2: **`Communicator` is flat —
one implicit group, three methods** (`all_reduce` / `all_to_all` / `send_recv`,
`infer-seam/src/lib.rs:59`). TP *alone* works flat. **MegaMoE needs TP×EP
composition** (DP-attention + EP is the standard MoE answer, arch §3), which
needs *named process groups per axis* — the SGLang `parallel_state` / NCCL
hierarchical-communicator pattern. Concretely:

```rust
// infer-seam: Communicator gains a group constructor (Fix 4).
trait Communicator {
    type Tensor;
    /// Sub-communicator over a named rank subset (the EP group, the TP group).
    fn new_process_group(&self, ranks: &[u32], tag: &str) -> Self;  // device mesh
    fn all_reduce(&self, tensor: &mut Self::Tensor);                 // TP group
    fn all_to_all(&self, send: &Self::Tensor, recv: &mut Self::Tensor); // EP group
    fn send_recv(&self, stage: u32, tensor: &mut Self::Tensor);      // PP group
}
```

`all_to_all` must also carry the DeepEP layout metadata the kernel needs (the
per-rank / per-expert prefix matrices, `send_head`, `recv_src_idx` — `state.rs:165`),
not just `send`/`recv` tensors. Options: (a) widen `all_to_all` to a `MoeDispatch`
struct (dispatch returns a handle the combine consumes — mirrors DeepEP's
`handle`); or (b) keep `Communicator` thin and put the MoE-specific layout in an
`infer-cuda`-internal type, exposing only `dispatch(plan) -> Handle` /
`combine(Handle)`. **(b) is preferred** — it keeps the seam narrow (Ousterhout
deep-module) and matches the legacy split where `deepep-sys` already owns the
layout buffers. The EP group itself is the only thing that must cross the seam.

`infer-topo` owns the rank-mesh descriptor (`{tp, ep, dp, pp}` + rank coord,
the legacy `MultiAxisConfig`/`RankCoord`, `config.rs:50`) and hands each
`Communicator` its EP/TP rank subset. This is the orthogonal-injection point
from arch §4.1 ("Topology → Communicator wiring").

### C.4 Normal vs low-latency, prefill vs decode (the CUDA-Graph constraint)

Upstream pairs **normal dispatch + contiguous GEMM (prefill)** and
**low-latency dispatch + masked GEMM (decode, CUDA-Graph)** (A.1/A.2). Legacy
`deepep-sys` wraps **only normal/intranode** (B.4) and the decode GEMM is
GEMV-not-masked (B.5). So the clean rewrite needs, in order:

1. **Phase 2 (EP, intranode):** port the normal-dispatch DeepEP path under
   `Communicator::all_to_all` for *both* phases first (correctness before graph).
2. **Phase 3 (DeepGEMM):** add the masked grouped-GEMM decode layout so decode is
   CUDA-Graph-capturable (the `GraphRunner` seam already exists,
   `infer-seam/src/lib.rs`).
3. **Later (post-cutover, internode):** extend `deepep-sys`/`build.rs` to compile
   DeepEP's `internode.cu` + low-latency kernels for multi-node EP. Out of scope
   for the EP=8-single-node gate.

### C.5 TBO / overlap — where it hooks

TBO (A.4) needs the executor to keep two micro-batches in flight so MoE comm of
one overlaps compute of the other. Arch §8 flags exactly this: the executor's
single `inflight: Option` can't express microbatches (the `microbatch: Option<u32>`
plan field is an unwired hook). **TBO is the same structural gap as PP (Fix 3) —
defer it.** Two cheaper overlaps are available *without* the ring:
- DeepEP combine already overlaps via `compute_stream` event-wait (B.3) — port
  that.
- The hook-based zero-SM overlap (A.1) is intra-call and lands with the DeepEP
  port. Full TBO (split-batch) waits on the in-flight ring (Fix 3).

### C.6 EP=8 verification shape (the evidence gate)

Per roadmap Phase 2/3 each phase ends with an **H20 verification gate**, and per
the distilled lesson "SLO verdict must come from the SLO workload": no default
flip without multi-shape EP parity. On the 8×H20 pod (`~/bin/pod`, TP=8 trigger
`INFER_CUDA_DEVICES`, see memory `project_h20_pod_access`):

1. **EP=1 == single-rank:** DSv4 MoE on one GPU (grouped GEMM, no all-to-all) ==
   the legacy single-GPU forward, greedy-token parity vs JSON baseline.
2. **EP=8 mock == EP=1:** all-to-all over a mock/loopback `Communicator` (no NCCL)
   reproduces EP=1 logits bit-for-bit (isolates the permute/dispatch math from
   the wire).
3. **EP=8 real DeepEP == mock:** real `deepep-sys` intranode dispatch/combine over
   8×H20 NVLink == the mock (isolates the kernel from the math).
4. **Long-context needle + greedy parity** (roadmap Phase 4 gate) at the
   production prompt length — not a c=1 smoke shape. The legacy DSv4 long-context
   path is validated (memory `project_dsv4_compressed_attention_longctx_bug`); the
   EP port must hold that.
5. **Perf A/B** only after correctness: same-binary, same-shell, two env flips
   (`ARLE_DSV4_MOE_BACKEND` legacy hand-rolled vs DeepEP), side-by-side
   (lesson `wins/2026-05-27-dsv4-native-deepep-perf-ab`), `scripts/bench_guidellm.sh`
   per backend, Δ% vs baseline. wins/ entry per `feedback_bench_every_change`.

### C.7 Phased effort estimate

| Phase | Work | Effort | Gate |
|---|---|---|---|
| **0 (prereq)** | CUDA single-GPU `CudaModel` forward parity (roadmap Phase 1 TP done) | — | already sequenced |
| **2a** | DSv4 MoE `ModelArch` single-GPU: port router + SwiGLU + shared expert + grouped-GEMM ops into `infer-cuda`/`infer-models`; parity vs JSON | M (~1–1.5k LoC, mostly port) | EP=1 == legacy single-GPU |
| **2b** | `Communicator::all_to_all` wrapping `deepep-sys` (normal/intranode); `infer-topo` EP-group wiring; **Fix 4** (`new_process_group`) | M (seam extension + ~600 LoC) | EP=8 mock == EP=1, real == mock |
| **3** | DeepGEMM contiguous(prefill)/masked(decode) grouped GEMM; build-gated `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE`; CUDA-Graph decode | M | FP8-expert parity vs BF16 in tol; graph capture green |
| **4** | EPLB redundant-expert placement in `infer-topo`+loader; hot-expert replication | M-L | imbalance-tail removed; throughput A/B |
| **5 (later)** | Mega MoE fused `fp8_fp4_mega_moe` op; needs symmetric-memory + FP8×FP4 | L (depends on upstream maturity) | fused == staged within tol, perf win |
| **6 (later)** | TBO split-batch (= Fix 3 in-flight ring); internode low-latency DeepEP | L | post-cutover |

`Mega MoE` proper (Phase 5) is gated on (a) DeepEP + symmetric memory in the
clean stack and (b) FP8×FP4 weights — both downstream of Phases 2/3. Phases 2/3
deliver the staged MoE (dispatch / grouped-GEMM / combine as separate kernels)
which is the correct, CUDA-Graph-capturable baseline; the fused mega-kernel is a
later perf collapse of that pipeline.

---

## Integration verdict

**The new seam fits the staged MegaMoE pipeline cleanly, with one required
extension.** Dispatch/combine sit under `Communicator::all_to_all`, the grouped
GEMM is an `infer-cuda` ops variant inside `ModelArch::forward`, the shared
expert is dense local compute, EP placement is `infer-topo` — none of it touches
engine-core/scheduler (the deep-module property holds, as the roadmap already
validated). The **one thing that must change is Fix 4**: the flat 3-method
`Communicator` cannot express TP×EP composition; it needs `new_process_group`
(named per-axis groups / device mesh) before DP-attention+EP runs, and the EP
all-to-all needs to carry (or hide behind a `MoeDispatch` handle) the DeepEP
layout metadata. Everything else is port (tested numerics) or rebuild (collapse
the two legacy EP transports to one, tidy the scratch grab-bag, add the
masked/contiguous GEMM split).

**Caveats grounded in the legacy tree:** (1) `deepep-sys` is **intranode/normal
only** today — low-latency + internode RDMA are unwrapped, so the first EP=8 gate
is single-node NVLink, and CUDA-Graph decode waits on the masked GEMM (Phase 3).
(2) The fused **Mega MoE** kernel (`fp8_fp4_mega_moe`) needs symmetric memory +
FP8×FP4 and is a Phase-5 perf collapse, not the entry point. (3) TBO split-batch
is the **same in-flight-ring gap as PP (Fix 3)** — defer; the DeepEP
`compute_stream` event-overlap and zero-SM hook give partial overlap without it.

---

## Sources

[sgl-ep]: https://docs.sglang.io/advanced_features/expert_parallelism.html "SGLang — Expert Parallelism (dispatch→pre-permute→core runner→post-permute→combine; BaseDispatcher/MoeRunner/PermuteMethodPool; normal vs low_latency)"
[sgl-ds]: https://docs.sglang.io/basic_usage/deepseek_v3.html "SGLang — DeepSeek V3/V3.1/R1 Usage (DeepEP/DeepGEMM/EPLB; masked vs contiguous; shared expert; 256+1 top-8)"
[deepep]: https://github.com/deepseek-ai/DeepEP/blob/main/README.md "DeepEP README (normal vs low-latency, NVLink/RDMA BW, double-buffer ping-pong, zero-SM receiving hook, FP8 dispatch, ElasticBuffer/Normal-SMFree)"
[deepgemm]: https://github.com/deepseek-ai/DeepGEMM "DeepGEMM repo (FP8 fine-grained-scaling GEMM; M-grouped contiguous/masked; Mega MoE)"
[dg-guide]: https://agentpedia.codes/blog/deepgemm-guide "DeepGEMM Guide — Mega MoE (fp8_fp4_mega_moe), m_grouped_fp8_gemm_nt[_masked], UE8M0 SM100 scales, 2-level accumulation"
[lmsys]: https://www.lmsys.org/blog/2025-05-05-large-scale-ep/ "LMSYS — Large-Scale EP on 96 H100 (EPLB 1.49×/2.54×, TBO +27–35% / 17.5k vs 12.9k tok/s, PD disagg, DP-attention)"
[v4pro]: https://www.aimadetools.com/blog/deepseek-v4-pro-complete-guide/ "DeepSeek-V4-Pro overview (384 routed + 1 shared experts, top-6, 1.6T params)"
[arch]: 2026-06-03-ideal-inference-engine-architecture.md "ARLE ideal-architecture doc (§3 parallelism axes, §4.3 contracts, §8 Communicator hierarchy gap / Fix 4)"
[roadmap]: 2026-06-03-multigpu-port-roadmap.md "ARLE multi-GPU port roadmap (Phases 2 EP / 3 DeepGEMM / 4 DSv4 / 5 Communicator hierarchy)"
