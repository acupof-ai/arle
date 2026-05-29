# Backend Operator Library — FlashInfer-style `plan()`/`run()` as the unification of the four dispatch-governance gates

> Diagnosis: [`../reviews/2026-05-29-gpu-dispatch-governance-analysis.md`](../reviews/2026-05-29-gpu-dispatch-governance-analysis.md).
> Gate plan: [`gpu-dispatch-governance.md`](gpu-dispatch-governance.md).
> Policy primitive (landed): [`infer/src/dispatch_policy.rs`](../../infer/src/dispatch_policy.rs).
> Registry data (sibling, in progress): [`../reviews/kernel-registry.md`](../reviews/kernel-registry.md).
> Status: **approach-first artifact, awaiting sign-off.** Architectural, cross-cutting
> (>5 files). No runtime code lands until the approach is accepted; Phase 1 lands the
> trait *with* its first real consumer, never as an empty scaffold
> (`memory/feedback_no_speculative_interface_shaping`).

## The one-paragraph claim

The four gates (Declare · Observe · Assert · Govern) are not four mechanisms — they are
four *views* of one missing artifact: a **dispatch plan** computed before execution and
kept around. FlashInfer already proves the shape of that artifact. Its wrappers split
`plan(shape, dtype, …)` — done once, selects the kernel and sets up workspace — from
`run(tensors)` — executed every step against the cached plan. ARLE's resolvers
(`LinearKernelPlan::batched`, `linear.rs:124`; the `match kv_pool.format` at
`batch_decode.rs:1220`/`1354`; the TileLang head-config match at `attention.rs:1466`)
*are* `plan()` logic — they just inline the decision into the launch and discard it. This
doc proposes to lift those resolvers into an explicit `plan()` that returns a named,
inspectable plan object. Once that object exists, all four gates fall out of it for free:

| Gate | Falls out of the plan object as |
|---|---|
| **Declare** | `plan()` *is* the single resolver; the returned plan *is* "what runs on this GPU." |
| **Govern** | `plan()` is the one home where "select the *best* operator" lives — registry data feeds its branches. |
| **Observe** | the plan names the chosen kernel → that name is the counter label / log line. |
| **Assert** | `plan()` is a **pure function** → "does my kernel get selected for shape X?" is a GPU-free unit test. |

The killer property is the last one. Because `plan()` is pure over
`(op, shape, dtype, batch, quant, SKU, DispatchPolicy)`, the question that costs ARLE the
most — *"is my new path even reachable?"* — becomes `assert_eq!(plan(shape).kernel,
Expected)`: no GPU, no weights, no bench. That defuses the corpus's single most expensive
failure class (实现了但从没跑 / 链路不通): the ~17-commit c=4-never-on-paged-path miss
(`errors/2026-05-07-three-layer-audit-miss-c4-real-path-is-packed-batch.md`), DeepGEMM
built-cached-6GiB-never-branched (`errors/2026-05-27-b33-deepgemm-not-wired-on-native-deepep.md`),
default-OFF-forever flags. Each was a `plan()`-equivalence question answered three weeks
late by a 0-delta bench.

---

## 1. FlashInfer's pattern, mapped onto ARLE

FlashInfer exposes per-op-family **wrappers** (`BatchDecodeWithPagedKVCacheWrapper`,
`BatchPrefillWithRaggedKVCacheWrapper`, `BatchPrefillWithPagedKVCacheWrapper`, the GEMM
and `bmm`/grouped-GEMM helpers). Each wrapper has the same two-method life-cycle:

```
plan(qo_indptr, kv_indptr, num_heads, head_dim, page_size, dtype, …)   # once per shape-class
  → picks the kernel template (mask mode, head-dim specialization, split-KV scheduling),
    sizes + pins the workspace/scheduler metadata, returns a handle that remembers all of it.
run(q, paged_kv, …)                                                     # every step
  → launches the kernel the plan already chose, against the pinned workspace.
```

The split exists because kernel selection + workspace sizing is shape-dependent and
*expensive to redo per step*, while the launch is cheap and repeated. ARLE has exactly
this structure, scattered:

| FlashInfer | ARLE today (the implicit `plan()`) | ARLE today (the implicit `run()`) |
|---|---|---|
| `BatchDecode…Wrapper.plan()` | `match kv_pool.format` quant-arm select (`batch_decode.rs:1220`), attention-read arm (`batch_decode.rs:1354`), TileLang head-config match (`attention.rs:1466`), graph-vs-eager heuristic (`batch_decode.rs:969`) | the `tilelang_*_run_cuda` / `kv_quant::*` launches |
| `BatchPrefill…Wrapper.plan()` | `LinearKernelPlan::batched(..., Prefill)` (`linear.rs:124`), the paged/ragged prefill path select | the GEMM + paged-prefill kernel launches |
| GEMM / grouped-GEMM `plan()` | `LinearKernelPlan::{batched,decode}` (30 variants, `linear.rs:64-101`) | the `match plan { … }` launch dispatch (`linear.rs:544`) |
| `plan()` workspace pinning | the Marlin decode scratch arena (`MarlinDecodeScratch`, `linear.rs:360`), CUDA-graph capture cache | scratch reuse / graph replay |

The mapping is one-to-one. **ARLE already does FlashInfer's `plan()` — it just never
returns the plan.** The whole proposal is: stop discarding it.

### 1.1 Why `plan()` subsumes Govern + Declare

`plan()` is the *single* function where kernel choice happens. That makes it the only
place that needs to know "which kernel is best for this shape" — i.e. Govern's policy
lives in `plan()`'s branch conditions, fed by the registry (§5). And the *returned* plan
is the literal answer to "what runs on this GPU" — i.e. Declare's `ExecutionPlan`
descriptor (governance-plan Phase 2) becomes the tuple of per-op `plan()` outputs, not a
parallel re-derivation. One resolver, consulted by both gates.

### 1.2 Why the returned plan subsumes Observe + Assert

The plan object **names** the chosen kernel (an enum variant, e.g.
`AttnPlan::TileLangPagedDecodeHd256 { qo: 16, kv: 4 }`). That name is:

- the Observe counter label — `dispatch_kernel_total{op="attn_decode", variant="TileLangPagedDecodeHd256"}`
  (governance-plan Phase 1) increments with `plan.kernel_label()`, no new resolve;
- the Assert oracle — `assert_eq!(oplib.attention().plan(shape, sku, policy).kernel,
  AttnPlan::TileLangPagedDecodeHd256 { .. })` runs on CPU, no GPU.

### 1.3 The pure-`plan()` property, stated precisely

```
plan : (OpFamily, Shape, Dtype, Batch, QuantFormat, SkuCaps, &DispatchPolicy) → Plan
```

`plan()` touches **no device memory, launches no kernel, allocates no GPU buffer**. It
reads only: the operand *shapes/dtypes/quant tags* (host-side metadata already on
`DeviceMatrix` / `PagedKVPool` — `weight.weight_format()`, `weight.rows/cols`,
`kv_pool.format`), the SKU capability table, and the resolved `&DispatchPolicy`. Workspace
*sizing* is computed (a number); workspace *allocation* stays in `run()`/setup, outside
the pure core. Therefore `plan()` is unit-testable with hand-built shape/quant/SKU inputs
and a `DispatchPolicy` literal — exactly the GPU-free reachability test the corpus has
been missing. This is the same property `dispatch_policy.rs` already exercises: its
parsers are pure and unit-tested directly (`dispatch_policy.rs:101-148`); we extend that
discipline from *policy parsing* to *kernel selection*.

---

## 2. The trait surface

Backend-neutral, per-op-family, mirroring FlashInfer's wrapper-per-family rather than one
god-trait. Model code holds `&dyn OperatorLibrary` (or the concrete type behind the
`server_engine::LoadedInferenceEngine` enum) and **never names a `cuda_kernels` or
`mlx-sys` type** — backend isolation per [`backend/AGENTS.md`](../../infer/src/backend/AGENTS.md)
invariants 2–3 and root `AGENTS.md §Backend isolation`. The concrete `cudarc`/`mlx`
handles live entirely inside the impls.

```rust
// infer/src/oplib.rs — backend-neutral contract. NO cuda/mlx types in any signature.
pub trait OperatorLibrary {
    fn linear(&self)        -> &dyn LinearOps;         // GEMV/GEMM, Marlin, quantized, cuBLASLt
    fn grouped_linear(&self)-> &dyn GroupedLinearOps;  // MoE / grouped GEMM
    fn attention(&self)     -> &dyn AttentionOps;      // paged decode, paged/ragged prefill
    fn norm(&self)          -> &dyn NormOps;           // RMSNorm
    fn rope(&self)          -> &dyn RopeOps;
    fn sampling(&self)      -> &dyn SamplingOps;
    fn kv_quant(&self)      -> &dyn KvQuantOps;        // KVFormat quantize-on-write
    fn sku(&self) -> SkuCaps;                          // SM tier / AOT head-configs / chip
}
```

Each family is a FlashInfer-style wrapper with the `plan`/`run` split. The plan type is a
**backend-neutral enum that names the chosen kernel** — this enum is what gets
counted/logged/asserted (Observe/Assert) and is the per-family slice of Declare's
`ExecutionPlan`:

```rust
pub trait LinearOps {
    /// PURE. No device touch. The relocated LinearKernelPlan resolver.
    fn plan(&self, shape: LinearShape, w: WeightDesc, batch: usize,
            phase: LinearDispatchPhase, policy: &DispatchPolicy) -> LinearPlan;
    /// IMPURE. Launches exactly the kernel `plan` named, against `out`.
    fn run(&self, plan: &LinearPlan, x: &Hidden, w: &Weight, out: &mut Hidden) -> Result<()>;
}

/// The named dispatch artifact. Backend-neutral: it names the *logical* kernel,
/// not a device function pointer. CUDA maps it to an FFI symbol; Metal to an MLX op.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinearKernel {
    Bf16Gemv, Bf16GraphsafeGemm, Bf16CublasGemm,
    W2A16Gemv, W4A16Gemv, W8A16Gemv,  /* … the 30 variants of linear.rs:64-101 … */
    MarlinW4Gemm, MarlinW4A8Gemm, MarlinW4Hybrid, MarlinW4FP8Prefill,
    Dsv4Fp8BatchGemv, /* … */
    // Metal-resolvable subset reuses the same enum where semantics match;
    // Metal-only kernels (mlx quantized_matmul) get their own variants.
    MlxQuantizedMatmul { bits: u8, group_size: u8 }, MlxBf16Matmul, MlxGgufMatmul,
}

pub struct LinearPlan {
    pub kernel: LinearKernel,        // ← the counted/logged/asserted name
    pub workspace_bytes: usize,      // sized here, allocated in run()/setup
    pub graph_safe: bool,            // CUDA-graph capturable (graphsafe_batched_weight, linear.rs:324)
    pub fallback_from: Option<LinearKernel>, // Some(_) ⇒ a loud Observe fallback fired
}
```

The same `plan(pure) / run(impure) / Plan(named-enum)` triple repeats per family:

- **`AttentionOps`** — `plan(AttnShape{kind: Decode|PagedPrefill|RaggedPrefill, qo_heads,
  kv_heads, head_dim, page_size, kv_format}, sku, policy) -> AttnPlan`. `AttnPlan::kernel`
  folds today's three scattered decisions into one named value:
  `TileLangPagedDecodeHd256{qo,kv}` (the head-config match, `attention.rs:1466`),
  `Fp8FusedDecode` / `Int8FusedDecode` / `Int4FusedDecode` / `Bf16PagedDecode` /
  `TurboQuantDecode` (the attention-read `match kv_pool.format`, `batch_decode.rs:1354`),
  each carrying `graph_safe: bool` (the graph-vs-eager heuristic, `batch_decode.rs:969`).
  The unprecompiled-head-config hard-fail (`attention.rs:1470`) becomes a **`plan()`-time
  error**, surfaced by `explain-dispatch` before bench day instead of at first launch.
- **`KvQuantOps`** — `plan(kv_format, has_static_scales) -> KvQuantPlan`. Folds the
  quantize-on-write `match kv_pool.format` (`batch_decode.rs:1220`) + the KIVI
  per-channel-vs-per-row K branch (`batch_decode.rs:1256`). The decode KV path's two
  matches (`:1220` quantize, `:1354` attention) collapse into `kv_quant().plan()` +
  `attention().plan()` — same two decisions, now named and testable.
- **`GroupedLinearOps`** — MoE/grouped GEMM (DeepGEMM / grouped Marlin / per-expert
  cuBLAS). The DeepGEMM-never-branched miss is precisely a `GroupedLinearPlan.kernel`
  that the unit test would have caught.
- **`NormOps` / `RopeOps` / `SamplingOps`** — small surfaces, mostly one kernel each, but
  carried in the trait so model code routes *everything* through one library (no
  cfg-leak escape hatch) and the plan enum is exhaustive for Declare's per-step
  `ExecutionPlan`.

`SkuCaps` is the **SM/SKU tier resolver the analysis found entirely missing** (diagnosis
§2.2): `{ sm_major, sm_minor, aot_head_configs: &[(u8,u8)], chip: ChipClass }`. `plan()`
consults it; `explain-dispatch` prints it; the AOT hard-fail moves from a runtime launch
panic to a `plan()`-time check.

---

## 3. Two backend impls (sketch)

### 3.1 `CudaOpLib` — `#[cfg(feature = "cuda")]`, lives behind `LoadedInferenceEngine::Cuda`

Wraps the existing kernels; adds **no new kernel**. `plan()` is the relocated resolver;
`run()` is the relocated launch.

- `LinearOps::plan` = the moved body of `LinearKernelPlan::{batched,decode}`
  (`linear.rs:104-188`), reading `WeightFormat`, batch, phase, and `&DispatchPolicy`
  fields (`r4_w4a16_gemv_override`, `marlin_w4_fp8_prefill`, `hybrid_w4a8_prefill`,
  already centralized in `dispatch_policy.rs:48-64`). `run()` = the
  `match plan.kernel { … }` launch (`linear.rs:544`) dispatching to `ffi::marlin_*`,
  the `w{2,4,8}a16_gemv_batch` kernels, cuBLAS/cuBLASLt GEMM, the hand-rolled quantized
  GEMVs. The silent alignment-fail fallbacks (`linear.rs:139,165`, today `log::trace!`)
  become `plan.fallback_from = Some(MarlinW4Gemm)` → one loud Observe counter.
- `AttentionOps::run` = TileLang AOT (`tilelang_batch_decode_paged_hd256_*`,
  `attention.rs:1467-1469`), FlashMLA, the FP8/INT8/INT4 fused-decode kernels. `plan()`
  encodes the head-config + KV-format + graph decision.
- `GroupedLinearOps::run` = DeepGEMM / grouped Marlin.
- Workspace: `LinearPlan.workspace_bytes` drives the existing `MarlinDecodeScratch`
  arena sizing (`linear.rs:360`); `graph_safe` reuses `graphsafe_batched_weight`
  (`linear.rs:324`).

### 3.2 `MetalOpLib` — `#[cfg(feature = "metal")]`, behind `LoadedInferenceEngine::Metal`

Wraps MLX primitives (`infer/src/backend/metal/mlx.rs`): `quantized_matmul`
(`mlx.rs:651`), `gguf_quantized_matmul` (`mlx.rs:696`), `rms_norm` (`mlx.rs:823`),
`fast::rope` + `fast::scaled_dot_product_attention` (`crates/mlx-sys/src/mlx_bridge.cpp:1790,1806`).

**The opacity boundary, handled honestly.** Below `mx::eval`, MLX picks the actual Metal
GPU kernel — a black box with no hook (diagnosis §2.1; `feedback_mlx_async_eval_is_caller_thread`).
The plan abstraction does **not** pretend to see through it. `MetalOpLib::plan()` names
**the MLX op it will invoke** — `LinearKernel::MlxQuantizedMatmul{bits,group_size}` /
`MlxGgufMatmul` / `MlxBf16Matmul`, `AttnPlan::MlxFastSdpa` — not the GPU kernel MLX
chooses underneath. That is the correct granularity: it is the **last decision ARLE
controls**, so it is exactly what reachability assertions and counters should pin. This
matches governance-plan non-goal "Not building MLX kernel introspection" — Observe/Assert
stop at the Rust→MLX boundary, and the plan enum says so by construction.

Both impls satisfy the **same** `OperatorLibrary` trait; model code (`model/qwen35/*`)
calls `oplib.attention().plan(...)` then `.run(...)` and is backend-agnostic. The
CUDA-vs-Metal concrete type is selected once at the `server_engine::LoadedInferenceEngine`
enum arm, never cfg-leaked into model code.

---

## 4. Migration — incremental, deletion-style, non-speculative

**Hard constraints.** (1) The trait lands *with* its first real consumer, not as an empty
scaffold (`feedback_no_speculative_interface_shaping`: KvHandleRef/async-Verifier killed
for exactly this). (2) No parallel old+new path — the resolver is **moved**, not
duplicated (`feedback_no_half_states`). (3) Each op family is **behavior-preserving**:
`plan()` must select bit-identically what the old code selected, proven by a
plan-equivalence test, *before* any new kernel is wired.

### First family to migrate: **linear / GEMM** (`LinearKernelPlan`)

Argued, not asserted:
- **Worst scattered dispatch.** 30 variants across `batched`+`decode` (`linear.rs:64-188`),
  3 inline policy reads (now centralized in `dispatch_policy.rs`) still consumed at the
  resolve site, 2 silent `log::trace!` fallbacks (`linear.rs:139,165`). It is the densest
  implicit-`plan()` in the tree.
- **Clearest win.** It is already *named* as an enum and already *stringified* for one
  purpose (`linear.rs:544-576` maps every variant to a `&str`). That string map is a
  half-built Observe label with no counter behind it — the migration just gives it a
  home. Lowest distance from current state to a returned, counted, asserted plan.
- **Already half-relocated.** `dispatch_policy.rs` (landed) lifted the policy inputs out
  of the hot path; relocating the resolver that *consumes* them is the natural next
  tranche the policy module's own doc-comment anticipates ("the kernel-selection knobs in
  `ops/linear.rs` + `ops/attention.rs`", `dispatch_policy.rs:11-13`).
- Alternative considered: paged-decode attention first. Rejected for ordering — its plan
  folds three decisions (head-config + KV-format + graph) and touches the KV pool;
  linear is the cleaner first cut that proves the pattern before attention inherits it.

### Phase 1 — relocate `LinearKernelPlan` into `oplib::linear::plan()` (file-by-file)

| File | Delta |
|---|---|
| `infer/src/oplib.rs` *(new)* | Declare `OperatorLibrary` + `LinearOps` trait + `LinearKernel` enum + `LinearPlan` struct. **Only the `LinearOps` family is real this phase**; the other `fn …()` accessors are added one-per-migrated-family later, not stubbed empty now (non-speculative). |
| `infer/src/ops/linear.rs` | **Move** `LinearKernelPlan` enum + `batched`/`decode`/alignment helpers (`linear.rs:64-205`) into `CudaOpLib`'s `LinearOps::plan`. Rename the enum to the backend-neutral `LinearKernel`. **Delete** the old enum from `linear.rs` — no parallel copy. The `match plan { … }` launch (`:544`) becomes `LinearOps::run`. The `_kernel_label` string map (`:544-576`) is **deleted** and replaced by `LinearKernel`'s `Debug`/an explicit `kernel_label()` — one source of names, not two. |
| `infer/src/backend/cuda/bootstrap.rs` | Construct `CudaOpLib`; store on the engine; expose via `LoadedInferenceEngine::Cuda`. |
| `infer/src/model/qwen35/*` (+ qwen3 callers of `gemm_into`/`gemv`) | The call sites (`linear.rs:544`, `:732`, `:2305`) move to `oplib.linear().plan(...)` then `.run(...)`. First **real consumer** — satisfies "trait + consumer in the same tranche." |
| `infer/tests/oplib_linear_plan_equivalence.rs` *(new)* | **The behavior-preservation gate.** Parameterized over `{WeightFormat × batch ∈ {1,2,8,4096} × phase × DispatchPolicy permutations}`, asserts `oplib.linear().plan(...).kernel` equals the variant the pre-migration `LinearKernelPlan::{batched,decode}` returned for the same inputs. Pure, CPU-only, no GPU. This is the plan-equivalence test that licenses the move. |
| `infer/src/metrics.rs` + `metrics/render.rs` | (governance Phase 1 Observe) `dispatch_kernel_total{op="linear",variant=plan.kernel_label()}` incremented once in `run()`; `dispatch_fallback_total` when `plan.fallback_from.is_some()`. |

Phase 1 acceptance: (a) `oplib_linear_plan_equivalence` green — proves bit-identical
selection; (b) `cargo test --workspace` + clippy clean; (c) one `bench_guidellm.sh` run
per backend+model showing Δ ≈ 0 vs the pre-move baseline (relocation is behavior-preserving
→ wins entry per §Benchmarks; or `pending-remote` stub for CUDA-on-Mac); (d) `grep` shows
the old `LinearKernelPlan` enum and the `_kernel_label` map **gone** (deletion-style, no
half-state).

### Later phases (sketched, same recipe each)

2. **Paged-decode attention** — relocate the `attention.rs:1466` head-config match + the
   `batch_decode.rs:1354` KV-format match + the `:969` graph heuristic into
   `AttentionOps::plan`; the unprecompiled-head hard-fail moves to `plan()`-time.
   Plan-equivalence test mirrors Phase 1.
3. **KV quantize-on-write** — `batch_decode.rs:1220` match → `KvQuantOps::plan`.
4. **Grouped/MoE GEMM** — DeepGEMM/grouped-Marlin select → `GroupedLinearOps`. (The
   family whose absence caused the 6-GiB-never-branched miss.)
5. **Prefill (paged + ragged)** GEMM path, then **norm/rope/sampling** — small, last,
   completes the exhaustive per-step `ExecutionPlan` for Declare.
6. **`MetalOpLib`** implements each family in lockstep or trailing by one phase; the
   trait is proven on CUDA first, then Metal fills the same surface.

Each phase: trait family + real consumer + plan-equivalence test + Observe counter, in
one revertible tranche. Never a family ahead of its consumer.

---

## 5. Consuming `DispatchPolicy` and the kernel registry

### `DispatchPolicy` is `plan()`'s policy argument

`plan()`'s signature ends in `policy: &DispatchPolicy`. The landed
[`dispatch_policy.rs`](../../infer/src/dispatch_policy.rs) already resolves every
ops-layer knob once into one struct (`dispatch_policy.rs:48-95`); `plan()` reads its
fields (`r4_w4a16_gemv_override`, `marlin_w4_fp8_prefill`, `hybrid_w4a8_prefill`,
`tilelang_bf16_split_kv`, …) instead of `std::env::var`. This makes `plan()` pure *and*
deterministic for a given policy literal — the unit test passes a `DispatchPolicy { .. }`
by value and asserts the resulting kernel, with zero env coupling. `dispatch_policy.rs`
is the Declare-gate **policy** input; `oplib::*::plan()` is the Declare-gate **selection**
that consumes it. They compose: policy (parsed-once) → plan (resolved-per-shape) → named
kernel.

### The kernel registry is the *data* behind `plan()`'s selection

The sibling [`kernel-registry.md`](../reviews/kernel-registry.md) (governance Phase 4) is
one row per `(op, shape-class, SKU, quant)` → chosen kernel · impl type · roofline
position · best-known-alternative-and-why-unwired · owner. Today that knowledge lives in
one-off audit docs nobody consults at dispatch time (diagnosis §2.4). With `plan()` as the
single selection home, the registry becomes the *spec* for `plan()`'s branches: each row
maps to one `plan()` branch returning one `LinearKernel`/`AttnPlan` variant. The registry
answers "what is best"; `plan()` enacts it; the `plan().kernel` counter (Observe) confirms
the live path matches the registry's chosen row. The loop closes — Govern's "best kernel
exists but isn't wired" stops being silent, because a registry row whose `chosen` differs
from its `best-known` is a visible, owned diff against `plan()`'s actual branch.

---

## 6. Risks / non-goals

**Non-goals (explicit).**
- **Not a from-scratch kernel library.** Every `run()` calls an *existing* kernel —
  TileLang AOT, Marlin, FlashMLA, cuBLAS/cuBLASLt, hand-rolled quantized GEMVs, MLX
  `quantized_matmul`/`fast::sdpa`/`fast::rope`. `plan()` selects; it never computes. Zero
  new `.cu`/`.metal`.
- **Not MLX-internal introspection.** Metal plans name the MLX op invoked, not the GPU
  kernel MLX picks under `mx::eval`. The opacity boundary is preserved by design
  (§3.2; governance non-goal; `feedback_mlx_async_eval_is_caller_thread`).
- **Not a new decision layer.** `plan()` is the *relocation* of existing resolvers. If it
  grows logic the old `match` did not have, the migration failed — caught by the
  plan-equivalence test going red.
- **Not a default-behavior change.** Migration is behavior-preserving per family;
  wiring a *new/better* kernel into a `plan()` branch is separate, c-sweep-gated work
  (`errors/2026-05-25-axis2-mixed-default-kill.md`) that may only land *after* the
  family's plan-equivalence test is green and the registry row justifies it.

**Risks.**
- **Hidden state in the old resolver.** If `LinearKernelPlan::batched` secretly depended
  on something not in `(shape, weight, batch, phase, policy)`, the "pure `plan()`"
  premise breaks. *Mitigation:* the plan-equivalence test sweeps the full input
  cross-product; any divergence fails the move before it lands.
- **Trait churn dragging ahead of consumers.** *Mitigation:* the non-speculative rule —
  only the migrated family's accessor is real; unmigrated families are absent, not stubbed
  (`feedback_no_speculative_interface_shaping`).
- **`run()` plumbing cost (extra indirection on the hot path).** *Mitigation:* `plan()`
  is resolved per shape-class and cached (FlashInfer's whole point) — the per-step cost is
  `run(&plan)`, a `match` already present at `linear.rs:544`. Verify with the Phase 1
  Δ≈0 bench; kill if any family's relocation regresses > 0.5% per-token wall-clock.
- **Backend-isolation leak.** A `cudarc`/`mlx` type sneaking into a trait signature would
  re-create the bootstrap straddle (`backend/AGENTS.md` invariant 2). *Mitigation:*
  `cargo check -p infer --no-default-features --features cuda,no-cuda` in CI; the plan
  enums are backend-neutral by construction (they name logical kernels, not handles).

---

## 7. Why this is the unification, not a fifth gate

The governance plan ships four gates as separate interventions (counters, a policy
refactor, a test file + bench flag, a registry doc). They work — but they are four places
to keep in sync. This proposal observes that all four are *projections of one object*. Land
`plan()`-returns-a-named-plan once per op family and: Declare is the resolver, Govern is its
branch policy, the returned name is Observe's label, and the function's purity is Assert's
test harness. The four gates stop being four maintenance surfaces and become four reads of
the same artifact — which is the deletion-style convergence the analysis (§4) and
`backend/AGENTS.md` (refactor posture) both ask for. `dispatch_policy.rs` already took the
first step (policy → one struct); this takes the second (selection → one function returning
one named plan).

**Awaiting sign-off on the approach before any runtime code lands.**
