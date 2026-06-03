# `infer/` Clean-Rewrite Plan (Amputate to Survive)

**Type:** Full refactor strategy (greenfield rewrite plan).
**Branch:** `arch/ideal-inference-engine`
**Foundation:** [`2026-06-03-ideal-inference-engine-architecture.md`](2026-06-03-ideal-inference-engine-architecture.md) (ideal architecture)
+ [`2026-06-03-backend-seam-redesign.md`](2026-06-03-backend-seam-redesign.md) (current-state audit)
**Driver:** ckl — "infer can be fully rewritten, amputate to survive; too much AI-garbage code."

---

## 0. First Principle: Which Limb to Amputate, Which Flesh to Keep

`infer/src` = **167,062 lines**. If "rewrite everything" means **line-by-line from zero**, that equals
**re-deriving all numerical correctness** — RoPE/attention/MoE routing/FP8 KV/quantization — reintroducing
every bug you already fixed (DSv4 long-ctx RoPE, FP8 KV step-1 divergence, hybrid prefix downgrade,
TileLang warp23 NaN…). That is not amputation, it is suicide.

**The SOLID "amputate to survive" = cut the AI-garbage of the architectural skeleton, keep and port the test-locked numerics and kernels.**
The garbage is the **skeleton** (scheduler god-trait, dual schedulers, 3 metrics stacks, dup/dead/half-state);
the flesh is the **numerics** (`model/` forward math, `ops/` kernel wrappers, radix, kv_tier, collectives, loaders).

> If you really want to start from zero on the numerics too (fully distrusting the existing model math), that is a
> separate engineering effort an order of magnitude larger/riskier, and needs its own license — this plan
> **defaults to porting the numerics and rewriting the skeleton**. This is my strong recommendation.

### Keep / Port / Rewrite Boundary (per-module, grounded)

| Disposition | Module (lines) | Rationale |
|---|---|---|
| **KEEP** (crates/, untouched) | `cuda-kernels` 44k · `mlx-sys` 81k · `*-spec` · `deepep-sys` · `xgrammar-sys` · `kv-native-sys` · `autograd`/`train` (OPD, independent surface) | kernels/bridges/config/training, not the infer skeleton |
| **PORT** (logic/numerics kept, interfaces re-wired to the new seam) | `model/` 45k (forward math) · `ops/` 8.7k · `prefix_cache` ~4k (radix algorithm) · `kv_tier/` 7.9k · `distributed/` 3.4k (collectives) · `weight_loader` 2.7k · `quant`/`gguf`/`hf_hub`/`tokenizer`/`sampler`/`speculative` | test-locked correctness + practical utilities; rewriting = pure risk |
| **REWRITE** (AI-garbage skeleton, rebuilt on the ideal architecture) | `scheduler/` 18k (god-trait/dual sched/half-state) · `backend/`'s dispatch+bootstrap+MetalScheduler glue · `metrics/`+`metrics.rs` ~5k (3 stacks merged into 1) · `server_engine`+`http_server`'s enum dispatch · `main`/`bin` entrypoints | this is the "AI garbage": duplication, dead code, half-states, god-trait |
| **NET-NEW** | 5 seam contracts + engine-core skeleton | the foundation of the ideal architecture |

Rough math: **~100k ported, ~50-60k rewritten, seams net-new**. The truly-deleted garbage (dup/dead/half-state) is folded into the rewrite.

---

## 1. Target Crate Graph (Use the Compiler to Enforce "Backend-Agnostic")

Today `infer` is a **single crate + cfg**, so the scheduler can `use crate::backend::cuda::PagedKVPool` —
backend-agnosticism relies entirely on self-discipline. **The core payoff of the rewrite: split crates so that "engine-core cannot depend on any backend" becomes a compile-time fact.**

```
crates/
  infer-plan/     ← ForwardPlan, ForwardMode, Request/Response IR (pure data, zero backend deps)
  infer-seam/     ← traits: BackendExecutor · KvPool · Communicator · Sampler · GraphRunner · ModelArch
  infer-core/     ← engine core: scheduler · radix · admission · slot lifecycle · overlap loop · PP microbatch
                    depends on {infer-plan, infer-seam} — **cannot mention CUDA/Metal at compile time**
  infer-models/   ← ModelArch impls (Qwen3/35, DSv4): layers written via Communicator/KvPool; deps {seam, *-spec}
  infer-cuda/     ← CudaExecutor · CudaKvPool · NcclCommunicator · CudaGraphRunner; wraps crates/cuda-kernels
  infer-metal/    ← MetalExecutor · MetalKvPool · MetalGraphRunner; wraps crates/mlx-sys
  infer-server/   ← frontend: HTTP/OpenAI · tokenize · detokenize · stream; deps {core, selected backend}
  infer/          ← thin re-export + bins (metal_serve / cuda serve); feature selects backend
  (KEEP: cuda-kernels, mlx-sys, *-spec, deepep-sys, autograd, train, …)
```

Dependency direction is **strictly downward**: `server → core → {plan, seam} ← {cuda, metal, models}`.
`infer-core` has zero dependency on `infer-cuda` → adding HIP = a new `infer-hip` crate, core does not recompile.
This is a guarantee the current cfg-single-crate **cannot give**, and the hardest-to-self-audit source of the "AI garbage."

---

## 2. The Five Contracts (Rust signatures, the foundation — nail this down first)

```rust
// infer-plan: pure data, backend-agnostic
pub enum ForwardMode { Prefill, Decode, Mixed, Idle, TargetVerify, DraftExtend }
pub struct ForwardPlan {            // = SGLang ForwardBatch; ARLE LogicalServePlan promoted to this
    pub mode: ForwardMode,
    pub decode_rows: Vec<DecodeRow>,        // slot, last_token, kv_offset
    pub prefill_rows: Vec<PrefillRow>,      // slot, tokens, start_pos, total
    pub microbatch: Option<MicrobatchId>,   // PP
    pub spec: Option<SpecPlan>,
}

// infer-seam: behavioral contracts, narrow. Device/parallel/kernel all live in impls.
pub trait KvPool {                  // replaces holding PagedKVPool directly
    fn alloc(&mut self, slot: usize, tokens: usize) -> Result<()>;
    fn free_slot(&mut self, slot: usize);
    fn seq_len(&self, slot: usize) -> usize;
    fn page_indices(&self, slot: usize) -> &[u32];
    fn migrate(&mut self, slot: usize, range: Range<usize>) -> Result<()>;
    fn free_pages(&self) -> usize;  fn page_size(&self) -> usize;  /* ~14 methods, mapped */
}
pub trait BackendExecutor {         // replaces the execution slice of the ModelForward 50-method god-trait
    type Plan = ForwardPlan;
    fn execute(&mut self, plan: &ForwardPlan, kv: &mut dyn KvPool,
               comm: &dyn Communicator) -> Result<StepOutput>;   // unified prefill/decode/mixed entry
    fn graph(&self) -> &dyn GraphRunner;                          // CUDA-graph capability
}
pub trait Communicator {            // TP/EP/PP collectives; LayerCommunicator promoted
    fn all_reduce(&self, t: &mut DeviceTensor);                   // TP
    fn all_to_all(&self, send: &DeviceTensor, recv: &mut DeviceTensor); // EP (DeepEP)
    fn send_recv(&self, stage: StageId, t: &mut DeviceTensor);    // PP
    fn topology(&self) -> &Topology;                              // {tp,pp,ep,dp}+rank mesh
}
pub trait Sampler { fn sample(&mut self, logits: &DeviceVec, params: &[SamplingParams]) -> Vec<u32>; }
pub trait GraphRunner {             // padded buckets + two-stage metadata (SGLang pattern)
    fn capture(&mut self, bs: usize);
    fn replay(&mut self, bs: usize, meta_out_of_graph: &PlanMeta) -> Result<()>;
}

// infer-seam: model abstraction, spanning devices
pub trait ModelArch {
    fn forward(&self, plan: &ForwardPlan, kv: &mut dyn KvPool,
               comm: &dyn Communicator, exec: &dyn LayerKernels) -> Result<Logits>;
}

// NOTE: illustrative sketch. The landed contracts are in crates/infer-seam (R0,
// commit 322d9d76) — KvPool grew to ~14 host methods, BackendExecutor uses
// submit/poll(PollResult) for overlap, and the lower-seam traits carry an
// associated device Tensor type. The ResourceGovernor seam (commit 6cd0afc5)
// is the AI-PC addition.
```

**Overlap is part of the contract, not an after-the-fact patch** (§ideal-architecture §4.2): engine-core holds a future-buffer,
the tokens returned by `execute` are published as **slot indices**, the next-step plan references them directly without waiting on the host round-trip. Aligns with SGLang
`FutureMap` / vLLM zero-overhead. ARLE's existing `pending_decode`/`pending_prefill` async logic is **ported** in,
but expressed as an explicit future contract, no longer bound to the god-trait.

---

## 3. Build Order (greenfield bottom-up, each step independently verifiable)

The old tree keeps serving in parallel; the new tree grows in `crates/infer-*`, and **the old tree is not deleted until the parity-gate passes** (guards against "rewrite loses correctness").

- **R0 contract layer** — `infer-plan` + `infer-seam`. Pure definitions, zero dependencies. Guarded by `cargo check`.
- **R1 engine-core + mock executor** — **rewrite** the scheduler/radix/admission/overlap/slot logic into
  `infer-core`, generic over `<E: BackendExecutor, K: KvPool>`. Pair it with a **CPU mock executor** (emits fake tokens),
  so that **continuous batching/radix/retract/chunked/overlap can all be unit-tested on CPU** — which is impossible today (the scheduler is welded to the GPU).
- **R2 CudaExecutor (wrap, do not rewrite kernels)** — `infer-cuda` implements the seam, internally calling the existing kernels in `crates/cuda-kernels`
  + `PagedKVPool` (impl KvPool) + NCCL (impl Communicator). **Not one kernel line changes.**
- **R3 models port** — `infer-models` **ports** the `model/qwen3·qwen35·dsv4` forwards onto `ModelArch`,
  routing collectives through `Communicator` and KV through `KvPool`. **Numerical logic preserved**, only the interface swapped.
- **R4 frontend** — `infer-server` rewrites HTTP/tokenize/detokenize/stream, process/thread-separated from core
  (vLLM V1 pattern), so CPU work does not pollute the GPU loop.
- **R5 parity-gate + cutover** — after the new tree clears the gate (§4), **in one tranche** delete the `infer/src` old skeleton,
  and `infer` becomes a thin re-export + bins. no-half-state.
- **R6 MetalExecutor** — `infer-metal` implements the seam, deletes `MetalScheduler` (1.1k), Metal hangs off the shared core.
  **At this point one scheduler serves CUDA+Metal.**
- **R7 new axes** — DP-attention+EP (Qwen3.6-MoE), PP microbatch (`scheduler_pp_mixin` pattern), HIP
  (`infer-hip`, validates the abstraction), disagg (long-term).

---

## 4. Correctness Preservation (the biggest fatal risk of a rewrite — SOLID-enforced)

**Losing correctness in a rewrite is the number-one cause of death.** Defense: **do not delete the old tree until the new tree clears the parity-gate item by item.**

1. **Golden parity suite (must be green before deleting the old tree):**
   - `kv_precision_parity` (BF16 vs INT8/FP8/TQ4 trajectory) passes on the new tree;
   - `greedy_consistency` (scheduler vs single-request numerical drift) passes;
   - `e2e` / `e2e_qwen35` pass against the `infer/test_data/` JSON baseline;
   - `bench_guidellm` TTFT/ITL/tok-s **does not regress** on the binding SLO shape (H20 pod, CLAUDE.md-mandated).
2. **Must-preserve hard-won behaviors (each with an experience anchor; add a regression test when porting, do not rely on memory):**
   - DSv4 long-ctx output inverse-RoPE (`arle_dsv4_output_inverse_rope_cuda`);
   - hybrid model partial-prefix → MISS downgrade;
   - chunked prefill three hit modes (exact-full / prefix-of-cached / partial);
   - decode retract/requeue (sglang victim heuristic);
   - the known workaround for FP8 KV step-1 divergence (auto-default routing);
   - the `INFER_BYPASS_TILELANG_PREFILL` routing for TileLang warp23 NaN;
   - wired-limit auto pin (Metal); prefill-cap-8 multi-shape default.
3. **Branch discipline:** a long-lived branch = drift risk. Push R1–R5 to the parity-gate as fast as possible; each R step is an independent commit;
   hot-path steps get bench coverage backfilled on the pod (pending-remote).

---

## 5. The AI Garbage This Rewrite Eliminates by Construction (targeting ckl's pain points)

| Garbage (current state, already grounded) | Eliminated by construction after rewrite |
|---|---|
| `ModelForward` 50-method god-trait | split into 5 narrow seams, each unit-tested |
| 2 schedulers (cuda 13.7k + metal 1.1k) | 1 `infer-core`, the backend is an executor |
| 3 metrics stacks (dead SchedulerMetrics + stats EMA + ServerMetrics) | 1 observability layer |
| `update_ema`×3 / decode fan-out×10 / two readbacks / two launch-plans | engine-core rewritten, no duplication |
| half-baked unified_scheduler (StepPlan→Logical round-trip + flag) | ForwardPlan is the sole IR, no shadow |
| scheduler directly holding `PagedKVPool` | `KvPool` trait; compile-time backend-agnostic |
| dead spec fake-100%-acceptance welded into decode | spec as a ForwardMode, independently testable |
| `prefix_cache.rs` (2114) + `prefix_cache/` (2205) suspected overlap | merged into one during port |
| CPU work coupled with the GPU loop | frontend process/thread separated |

---

## 6. Risks and "How It Would Fail" (SOLID self-check)

- **R rewrites the numerics too** → certain death (re-deriving math). **Mitigation: R3 is a port, the parity-gate guards, §4.2 regresses item by item.**
- **long-lived branch drift** → the old tree is still changing in the same period. **Mitigation: keep the branch short; cut over at R5 ASAP; only cherry-pick critical fixes.**
- **rewriting subtle overlap/async logic introduces a race** → decode pipeline corrupted. **Mitigation: port the existing async-proven logic,
  do not reinvent; unit-test overlap invariants on the CPU mock executor.**
- **seam abstracted too early (speculative shaping)** → abstract from the **commonality of the two real backends CUDA↔Metal** (ForwardPlan already shared),
  HIP validates; not HIP-first designed out of thin air ([[feedback_no_speculative_interface_shaping]]).
- **crate split collides with cuda-kernels extraction governance** → align with the trip-wires of `docs/plans/cuda-kernel-crate-extraction.md`;
  `infer-cuda` is the wrap layer, does not duplicate the prelude ([[feedback_prelude_minimal]]).
- **scope explosion (167k)** → strictly hold the keep/port/rewrite boundary; **rewrite only the skeleton**. Each R step ships + verifies independently.

---

## 7. First Step (R0, can start immediately)

`infer-plan` + `infer-seam`, two new crates: pure contract definitions, zero backend dependency, `cargo check` is the guard, verifiable on Mac.
Nail down §2's 5 traits + ForwardPlan — this is the foundation of the whole building, worth aligning on signatures for one round before laying down code.

> Pending your confirmation on two points before starting R0: **(a)** Do you accept the keep/port/rewrite boundary (§0)? Default is **port the numerics, rewrite the skeleton**;
> if you want the numerics from zero too, scope/risk multiply several-fold and needs a separate license. **(b)** Do we go with the crate split (§1), or transition first
> within the single crate using module boundaries? I recommend **splitting crates directly** — compile-time backend-agnosticism is the biggest payoff of this rewrite, one a single crate cannot give.
