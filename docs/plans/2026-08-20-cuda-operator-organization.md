# CUDA operator organization across kernels and models

> Status: Proposed umbrella plan
>
> Applies to: CUDA serving for Qwen3 dense, Qwen3.5/3.6/3.8, DeepSeek V4,
> and GLM-5.2; OPD autograd CUDA; native CUDA, autograd NVRTC, vendored
> kernels, TileLang AOT, DeepGEMM JIT, FlashMLA, and collectives.
>
> First child plan:
> [Qwen CUDA quant-linear dispatch consolidation](2026-08-20-qwen-quant-linear-dispatch-consolidation.md).

## Decision

Organize the CUDA stack around five ownership layers:

```text
model execution
    -> model/operator policy
        -> typed operator-family launcher
            -> private FFI ABI
                -> native / vendored / AOT / JIT implementation
```

Aggregate shared launch mechanics, validation, workspace lifecycle, static
implementation identity, and evidence reporting. Keep model order, state,
fallback policy, shape policy, and performance thresholds inside the owning
model or operator family.

The hot path remains statically dispatched Rust and direct CUDA calls. This
plan prohibits trait objects, string lookup, registry parsing, heap allocation,
host readback, locks, and device synchronization during forward execution.

## Current state

Snapshot from the 2026-08-20 checkout:

| Surface | Size | Problem |
| --- | ---: | --- |
| `crates/infer-cuda/src/` | 61,587 Rust lines | Model policy, orchestration, scratch, and raw FFI calls overlap |
| `attention.rs` | 8,543 lines | Qwen paged attention, DSv4 MLA, preparation, routing, and launch code share one root |
| `loader.rs` | 6,709 lines | Format detection, upload, model assembly, repack, source retention, TP, and MoE loading overlap |
| `moe.rs` | 4,429 lines | Qwen and DSv4 policies share routing, scratch, transport, and kernel calls |
| `ops/quant_linear.rs` | 2,065 lines | Two entry routes and multiple weight-family policies overlap |
| `crates/cuda-kernels/src/ffi/` | 5,868 lines | ABI declarations are grouped, while many consumers still call them directly |
| `crates/cuda-kernels/csrc/` | 71 native CUDA source/header files | Source folders are already grouped by family and should remain so |
| `crates/autograd/src/` | 40,194 Rust lines | Training has separate policy, storage, compilation, and launch ownership |
| `crates/autograd/src/backend_cuda/kernels/` | 29 NVRTC CUDA source files | Forward, backward, optimizer, rollout, and bridge kernels compile into dtype-specific modules |

The native source layout already has useful family boundaries:

```text
attention/  comm/  elementwise/  gemm/  kv/  moe/
norm/       quant/ recurrent/    sampling/
```

Autograd has a second intentional source lane. Its 29 `include_str!` CUDA files
are concatenated and compiled by NVRTC for the current SM and tape dtype. They
remain autograd-owned unless an ABI and numerical contract is identical to an
already proven shared launcher.

The primary disorder is above those files. Raw ABI calls, provider selection,
model policy, scratch ownership, and counters are repeated in large consumer
modules.

## Scope

### CUDA model families

| Model route | Status | Operator policy owner |
| --- | --- | --- |
| Qwen3 dense | Supported | `infer-cuda` dense Qwen executor and shared operator families |
| Qwen3.5/3.6 hybrid dense/MoE | Supported | Qwen35 executor, attention, recurrent, MoE, and quant-linear families |
| Qwen3.8 mixed NVFP4/per-channel FP8 | Supported through Qwen35 | Qwen35 plus quant-linear storage and route policy |
| DeepSeek V4 | Supported | DSv4 executor and DSv4-specific attention/MoE policies |
| GLM-5.2 | DSv4 adapter, verification pending | DSv4 policy with GLM config adaptation |
| Qwen3-MoE public schema | Unsupported | Continue to fail during model classification |
| Gemma4/DiffusionGemma CUDA | Unsupported forward | Continue to fail before executor construction |

Unsupported models remain explicit. Operator cleanup cannot turn schema
similarity into model support.

### CUDA operator families

1. embedding and positional preparation;
2. normalization and elementwise operations;
3. dense and quantized linear projection;
4. paged, non-paged, full, linear, MLA, DSA, CSA, HCA, FA3, and FlashMLA
   attention;
5. KV addressing, packing, quantization, restore, and page preparation;
6. recurrent GDR, convolution, and FlashQLA paths;
7. MoE route, dispatch, grouped GEMM, activation, combine, DeepEP, and local
   fallback;
8. sampling and speculative draft/verify primitives;
9. TP/CP/EP collectives and custom all-reduce;
10. CUDA autograd forward, backward, optimizer, and OPD-specific fused paths.

Every production CUDA kernel belongs to one family. A kernel may serve multiple
models, but it has one launch owner and one ABI declaration.

## What already exists

| Existing truth/mechanism | Keep | Change |
| --- | --- | --- |
| `crates/cuda-kernels/kernels.toml` | TileLang AOT build matrix | Do not add runtime model policy |
| `operators/registry.toml` | Semantic operator and implementation IDs | Expand after code ownership is stable |
| `benchmarks/operators/optimal.json` | Qualified generated policy inputs | Preserve evidence and artifact binding |
| `scripts/reduce_operator_evidence.py` | Qualification gate | Continue rejecting synthetic identity |
| `KERNEL_BUILD_ID` and backend artifact identity | Build provenance | Keep outside per-step dispatch |
| `OperatorDispatchStats` | Request-boundary engagement | Extend by family without per-step allocation |
| `DeviceContext` and typed device buffers | Stream/device lifetime | Reuse in typed launchers |
| autograd `KernelCache` and NVRTC modules | Training kernel compilation and function lookup | Add source/flags/SM/NVRTC provenance and keep compilation outside steady-state forward |
| family FFI files | Private C ABI declarations | Stop direct use from model orchestration |
| `attention/`, `dsv4/`, `executor/` submodules | Existing decomposition | Move remaining root responsibilities into them |
| Qwen quant-linear child plan | First complete route-owner migration | Use as the migration template |

Three truth layers remain separate:

```text
operator legality     operators/registry.toml + generated static policy
measured evidence     benchmark JSON + numerical/model gates
artifact provenance   binary/kernel manifest + model revision
```

`kernels.toml` defines only the TileLang build set. Runtime legality remains in
the operator registry and generated static policy.

## Design principles

### Aggregate mechanisms

The following concerns are shared and should have one implementation:

- pointer acquisition and lifetime guards;
- shape, alignment, dtype, and buffer-length checks;
- checked host integer to CUDA ABI conversion;
- CUDA error conversion with operator and shape context;
- launch success/decline semantics;
- capture-safe workspace preparation and access;
- static implementation identifiers and post-success hit counters;
- warmup/preflight result caching;
- numerical harness conventions and evidence metadata;
- build and artifact identity.

### Preserve policy ownership

The following concerns remain model or family-specific:

- layer and operator order;
- KV/state mutation and rollback;
- prefill/decode/mixed/speculative phase selection;
- DeepGEMM, Marlin, FlashMLA, FA3, scalar, and fallback priority;
- `M`, context, TP/EP, SM, dtype, and checkpoint-dependent thresholds;
- source-weight retention and repack policy;
- exact scratch sizes and layout;
- numerical acceptance bounds for changed math;
- whether a route is graph-capturable.

Shared code owns how to launch. The model family owns when a launch is legal and
which implementation wins.

## Target architecture

### Runtime data flow

```text
ForwardPlan
   |
   v
model executor
   |  owns layer order, state, KV, TP/EP, speculative semantics
   v
operator policy
   |  static enum/match on phase, shape, dtype, SM, retained layout
   v
operator-family facade
   |  validates typed buffers, obtains pointers, uses prepared workspace
   v
cuda-kernels safe launcher
   |  one Rust function per CUDA ABI
   v
private ffi declaration
   |
   +-- native CUDA
   +-- vendored FlashMLA/FA3/Marlin/CUTLASS
   +-- TileLang AOT dispatch wrapper
   +-- DeepGEMM JIT cache
   `-- autograd NVRTC module when the operation is training-owned
```

### Load and warmup data flow

```text
checkpoint/config
   -> classify model and weight format
   -> validate source shape and byte length
   -> shard/fuse/repack
   -> decide source retention
   -> validate final storage state
   -> derive static legality/capability facts
   -> allocate maximum declared workspace
   -> preflight/JIT/warm selected implementations
   -> publish immutable weights + mutable per-slot state
```

Forward execution consumes this immutable result. Loader decisions never derive
from unrelated `Option` fields or environment variables.

### Crate ownership

| Layer | Ownership | Excluded ownership |
| --- | --- | --- |
| `cuda-kernels/csrc` | Device math and vendor integration | Model selection, serving policy |
| `cuda-kernels/src/ffi` | Private ABI declarations and generated AOT resolution | Public model-facing API |
| `cuda-kernels/src/<family>` | Typed checked launchers, device storage, family-local helpers | Qwen/DSv4 route priority |
| `infer-cuda/src/ops/<family>` | Serving route policy, family scratch, counters | Scheduler and HTTP |
| `infer-cuda/src/qwen*` | Qwen layer order, state, KV, speculative behavior | Raw CUDA ABI calls |
| `infer-cuda/src/dsv4*` | DSv4/GLM layer order, state, MLA/MoE/transport policy | Generic kernel registry runtime |
| `autograd/src/backend_cuda*` | Training route policy, gradients, tape-visible behavior | Serving model policy |
| `operators/` and `benchmarks/operators/` | Offline semantic truth and evidence | Per-step dispatch |

## Common code contracts

### 1. Typed safe launchers

Raw `ffi::*_cuda` calls become private to `cuda-kernels`. A family launcher:

1. accepts typed `CudaSlice`/view references and scalar dimensions;
2. validates sizes and alignment;
3. obtains device pointers and keeps guards alive;
4. performs checked ABI conversions;
5. invokes one FFI symbol;
6. converts the CUDA result with implementation and shape context.

Example shape:

```rust
pub fn launch_fp8_block_gemv(
    ctx: &DeviceContext,
    weight: &Fp8BlockWeight<'_>,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
    m: usize,
) -> Result<()>;
```

This is a concrete function. Do not introduce a `Kernel`, `Operator`, or
`Backend` trait for a single CUDA call.

### 2. Explicit engagement

Optional fast paths use one small value type:

```rust
#[must_use]
enum Engagement {
    Launched,
    Declined,
}
```

`Declined` means the caller owns a valid fallback. Errors mean the selected
implementation failed or storage is invalid. Do not encode these three states
as `bool`, `Option<Result<_>>`, or a missing counter.

Use this type only for optional routes. Mandatory launchers return `Result<()>`.

### 3. Static route decisions

Use enums and exhaustive `match` statements. Split each family into:

```text
select_route(metadata, phase, dynamic_shape) -> Route
launch_route(route, buffers, workspace)       -> Result
```

Selection is pure and CPU-testable. Launching owns device effects. Route enums
stay family-local because `Fp8Route`, `MlaRoute`, and `MoeTransportRoute` have
different legality.

Static facts should be computed at load or warmup. Dynamic facts such as live
rows, current context length, and accepted speculative depth remain per-step
inputs. Do not cache a decision across a dimension that changes.

### 4. Workspace lifecycle

Every family documents:

- maximum shape envelope;
- owning state object;
- allocation point;
- zero/reset requirement;
- capture lifetime;
- offload/reload behavior;
- concurrent stream assumptions.

Allowed lifecycle:

```text
load/warmup: ensure_capacity(max_envelope)
capture:     borrow fixed addresses
forward:     borrow/reset/launch
restore:     rebuild transient pointers, retain logical capacity
```

Forbidden lifecycle:

```text
forward -> allocate/grow -> synchronize/read back -> launch
```

There is no universal workspace trait. Families retain concrete scratch types
because layout and reset rules are part of their correctness contract.

### 5. Identity and counters

- implementation IDs are compile-time constants generated or checked against
  `operators/registry.toml`;
- the selected launcher increments its counter after successful submission;
- counters are relaxed atomics or existing device-independent counters;
- stats allocate names and vectors only at `/v1/stats` or another explicit
  request boundary;
- build identity, kernel bundle identity, model revision, and policy hash are
  reported together;
- component probes cannot mark an operator qualified.

### 6. Errors

Every error includes:

- semantic operator or family;
- implementation ID;
- model family when relevant;
- phase;
- `M/N/K` or attention/KV dimensions;
- SM and dtype when they affect legality;
- missing or invalid storage/workspace.

Errors are formatted only on failure. The success path performs no string
allocation.

## Model-specific boundaries

### Qwen3 dense

Reuse common embedding, norm, linear, paged attention, KV, sampling, and TP
launchers. Keep the dense paged-KV model flow and radix-page semantics in the
Qwen executor.

### Qwen3.5/3.6/3.8

Reuse the same dense operator launchers where buffer/layout contracts match.
Keep gated-delta recurrent state, periodic full attention, MoE/shared-expert
policy, DSpark/MTP state, LoRA merge, and whole-slot restore in Qwen35 modules.

The quant-linear child plan is the first route consolidation. It proves the
safe-launcher and family-policy split before attention or MoE migration.

### DeepSeek V4 and GLM-5.2

Share DSv4 launcher families and workspace mechanics. Keep model adapter values,
MLA/DSA/CSA/HCA order, MTP, DeepEP transport, expert topology, and KV band
policy inside DSv4. GLM verification remains a separate model gate even when it
uses identical launchers.

### OPD autograd

Reuse `cuda-kernels` typed launchers for an identical ABI and buffer contract.
Keep tape semantics, saved tensors, gradient accumulation, backward selection,
and optimizer behavior in `autograd`. Serving route enums do not cross into
training.

## Operator-family organization

### Dense and quantized linear

Follow the child plan. One route owner per weight family, one launcher per ABI,
load-time validation of retained representations, and identical M=1 behavior
across vector and batched entry points.

### Attention and KV

Target structure:

```text
infer-cuda attention facade
  +-- qwen_paged policy
  +-- qwen35_full policy
  +-- qwen35_linear policy
  +-- dsv4_mla policy
  +-- dsv4_dsa policy
  `-- shared metadata/address helpers

cuda-kernels launchers
  +-- paged attention
  +-- non-paged attention
  +-- FA3
  +-- FlashMLA
  +-- KV pack/quant
  `-- preparation kernels
```

Do not unify Qwen paged KV and DSv4 MLA latent state. Share page-index
translation only where the byte/address contract is identical.

Split `attention.rs` by moving existing coherent families into flat sibling
modules. The root becomes a facade and shared types. No compatibility wrappers
remain after each tranche.

### Recurrent

Group GDR, conv1d, and FlashQLA launch mechanics. Keep recurrent-state mutation,
chunk replay, and accepted-length rollback in Qwen35. Training backward launchers
remain separate from inference forward policy while sharing ABI wrappers where
identical.

### MoE

Separate these policies:

```text
Qwen35 local MoE
DSv4 local grouped MoE
DSv4 DeepEP normal-latency transport
DSv4 DeepEP low-latency transport
DSv4 W4AFP8/NVFP4 path
```

They may share routing primitives, pointer-table construction, activation
launchers, and checked grouped-GEMM wrappers. They do not share one fallback
ladder. Expert count, top-k, TP/EP ownership, scale layout, transport, and
capture rules remain explicit.

### Sampling and speculative execution

Share sampling kernels and typed logits/mask launchers. Keep request-feature
compatibility and accepted-chain state in each model executor. Speculative
compatibility is a semantic decision and cannot live in the kernel registry.

### Collectives

Keep `CollectiveBackend`, dtype, reduction operation, and direct NCCL/custom
all-reduce launchers shared. Model code owns placement and overlap. A generic
collective scheduler is outside this plan.

## File organization target

### `cuda-kernels`

Keep the existing `csrc/<family>` layout. In Rust:

```text
src/ffi/<family>.rs          private extern declarations
src/<family>.rs              typed safe launchers and family types
src/tensor.rs                device buffers and storage ownership
src/collective.rs            communication backend
src/ring_attention.rs        shared ring-attention implementation
```

Large repack/format-conversion methods may leave `tensor.rs` only when the move
produces one clear family owner. Do not create a generic tensor operation layer.

### `infer-cuda`

Target roots:

```text
ops.rs                       small common facade
ops/quant_linear*.rs         dense/quant route owners
attention.rs                 attention facade and shared types
attention/*.rs               existing semantic families
moe.rs                       MoE facade and shared types
moe/*.rs                     Qwen, DSv4 local, DeepEP, W4AFP8 policies
loader.rs                    common model detection/upload entry
qwen35_load.rs               Qwen35 assembly
dsv4/load.rs                 DSv4/GLM assembly
```

The goal is ownership clarity. File length is secondary evidence; cohesive
ownership determines acceptance. A long cohesive implementation may remain
long.

### `autograd`

Keep `backend_cuda/kernels/*.cu` and the existing dtype-module cache as the
training-owned provider lane. Group the compile catalog and Rust launch methods
by semantic family without creating one wrapper file per symbol. Replace a
direct launch only after the matching `cuda-kernels` typed launcher has shipped
and passed serving plus training parity. A cross-crate migration changes one
policy owner per commit.

The NVRTC artifact identity records the concatenated source hash, compile flags,
SM, tape dtype, CUDA driver, and NVRTC version. Module compilation and function
lookup finish during backend initialization or the declared dtype warmup. The
steady-state OPD forward/backward loop receives a prepared module and workspace.

## Performance invariants

The organizational migration has a zero-regression contract.

### Forbidden hot-path additions

- trait objects or virtual dispatch;
- `HashMap` or string-based implementation lookup;
- registry/JSON/TOML reads;
- heap allocation or format construction;
- mutexes, read/write locks, or new atomic coordination;
- host/device synchronization or D2H metadata reads;
- device capability queries after warmup;
- source repack or JIT compilation after the declared warmup boundary;
- additional CUDA launches, events, or stream waits;
- unstable workspace addresses under capture.

### Required equivalence receipt

For every structural tranche, compare baseline and candidate:

1. CUDA kernel symbol sequence and launch count by phase;
2. launch grid/block/shared-memory arguments;
3. input/output/storage addresses where fixed-address capture applies;
4. implementation hit counters;
5. allocated device bytes and workspace high-water marks;
6. host submit time and synchronization count;
7. numerical outputs according to the operation class;
8. model-level correctness;
9. end-to-end latency and throughput.

An organizational refactor should produce the same kernel work. Any changed
symbol, count, argument, or route is a behavioral change and moves into a
separate correctness/performance tranche.

### Acceptance thresholds

- address/copy/mask/index operations: bit-exact;
- pointwise operations: no finite/sign-class change and at most one output-dtype
  ULP from reference;
- GEMM/reduction/attention/recurrent: candidate reference-error metrics cannot
  worsen more than 5%; unchanged launch arguments should remain bit-identical;
- route engagement: exact counter equality for the same request;
- allocations and synchronizations: no increase;
- canonical workload: no unresolved negative median; deltas inside drift require
  at least three matched trials per arm and median plus range;
- failures, empty outputs, loops, and timeouts: zero.

## Testing matrix

### Host-only checks

- pure route selection for every implementation branch;
- shape/alignment/dtype/storage validation;
- unsupported model classification;
- operator registry ID uniqueness and source-path existence;
- generated policy freshness;
- no-cuda compilation of all touched routes.

Use one table-driven test per family. Avoid one test per wrapper.

### CUDA operator checks

| Family | Required operator gate |
| --- | --- |
| Linear/quant | FP32/FP64 reference, all production shapes, M sweep, three seeds, saturation/extrema |
| Attention | Reference attention, ragged rows, page tails, smallest/largest context, split/non-split |
| KV | Exact address/pack round trip, page boundary, quant extrema, restore |
| Recurrent | Reference forward/state, chunk boundary, replay, long sequence, backward where used |
| MoE | Route counts, expert offsets, long prefill, empty expert, max routed rows, scale semantics |
| Sampling | Deterministic seed, masks/bias/penalties/forced token, invalid distributions |
| Collectives | Per-rank parity, non-divisible shards, stream ordering, TP/CP/EP sizes |
| Autograd | Forward reference, finite-difference or analytic gradient, accumulation and optimizer step |

### Model gates

| Model/config | Required coverage |
| --- | --- |
| Qwen3 dense BF16 | Prefill, decode, paged KV, prefix reuse, c=1 and batched |
| Qwen3.6 FP8 | Hybrid attention, recurrent state, MoE, MTP/DSpark, FP8 KV where supported |
| Qwen3.8 NVFP4 | FP4 + per-channel FP8, source release, long-agent prefix reuse |
| Qwen W8A16 | Repacked/unrepacked shapes and untied quantized `lm_head` |
| DSv4 TP/EP | MLA/DSA, MoE, DeepEP/local fallback, MTP/DSpark, long prefill, concurrent decode |
| GLM-5.2 | Separate config/load/forward gate on the DSv4 implementation |
| OPD CUDA | Teacher logits, rollout, backward, optimizer, offload/reload, LoRA re-merge |

For runtime kernel-route changes, run `scripts/lever_gate.sh` and
`scripts/needle_gate.py temp` at 512/4096/16384/32768, three repetitions, on
the exact candidate binary. MoE non-determinism relaxes token identity at the
model layer only.

## Benchmark and trace contract

Every runtime tranche receives a dated `wins/` or `errors/` entry following
`docs/bench-and-trace-spec.md`.

Structural tranches use the archived pre-change binary as baseline. Keep model,
GPU, clocks, TP/EP, slots, KV dtype, flags, workload, seed, request order, and
output cap fixed.

Use the canonical 32K multi-turn agent workload through
`scripts/bench_throughput.py`. Report cold and warm slices, route counters,
prefix hits, KV residency, queue/preempt counts, output tok/s, req/s, TTFT,
ITL, errors, and incomplete outputs.

Trace only when launch receipts or wall metrics differ. A component profile
explains the difference; it cannot license an end-to-end claim.

## Failure modes

| Failure | Guard |
| --- | --- |
| Generic facade hides a model-specific legality rule | Family-local route enum and model-owned policy |
| Direct FFI call bypasses validation/counter | Private FFI plus registry source-path check |
| Shared wrapper adds allocation or synchronization | Hot-path invariant test and trace receipt |
| Load releases a representation still needed by fallback | Final storage validation and forced-decline test |
| Static route caches a dynamic row/context decision | Route input audit and boundary cases |
| Counter increments before submission succeeds | Increment after successful submission |
| Capture uses a reallocated workspace address | Warmup envelope and eager/captured address receipt |
| Registry claims an implementation absent from the binary | Verified manifest and generated-policy freshness gate |
| Component probe marks model qualification | Reducer requires real model E2E artifact |
| Shared launcher changes training numerics | Serving and autograd gates before deleting direct call |
| File split leaves old and new routes live | One complete family per tranche; delete old path |
| Unsupported model inherits a shared route accidentally | Classification and load failure tests |
| DeepGEMM JIT tail shape is missing from warmup | Production shape receipt and clean-cache tail test |
| W4AFP8 short smoke misses routed-row overflow | Long-prefill MoE gate beyond all block caps |

## Migration sequence

No big-bang rewrite. Each runtime tranche has one behavioral owner, touches at
most five files, removes its old path, and passes remote gates before the next
family starts.

### T0: Inventory and ownership map

Docs/dev tooling only.

- enumerate semantic operators, implementation IDs, source symbols, consumers,
  models, phases, SM tiers, capture support, workspace, and gates;
- mark registry rows as qualified, provisional, or unsupported;
- reject duplicate IDs, missing source paths, and model claims without gates;
- record current direct FFI call sites by family.
- inventory all autograd NVRTC sources, exported symbols, tape dtypes, compile
  options, module warmup points, and Rust launch consumers.

Exit: every production launch maps to one family and owner. No runtime change.

### T1: Safe-launcher boundary

Start with embedding, norm, and elementwise because they have simple buffer
contracts and no route policy.

- add typed launchers in existing `cuda-kernels` family modules;
- switch one consumer family at a time;
- compare symbol, arguments, launch count, outputs, and submit time;
- make migrated FFI declarations inaccessible to external consumers;
- delete direct calls after the last consumer migrates.

Exit: the pattern is proven without a performance or capture regression.

### T2: Qwen quant-linear pilot

Execute the child plan. This proves route selection, retained storage, optional
engagement, counters, and M=1/batched convergence.

Exit: all Qwen dense quantized projection routes have one owner.

### T3: Attention and KV

Order:

1. simple non-paged attention used by serving and autograd;
2. dense-Qwen paged attention and KV addressing;
3. Qwen35 full attention and recurrent preparation;
4. DSv4 MLA/DSA/CSA/HCA and FlashMLA;
5. quantized KV and restore paths.

Each subtranche moves one semantic family and keeps its model gate independent.

Exit: `attention.rs` is a facade/shared-types root; raw launch code belongs to
family modules and `cuda-kernels` launchers.

### T4: Recurrent

- consolidate GDR/conv1d/FlashQLA launch mechanics;
- retain Qwen state transitions and replay policy;
- migrate training forward/backward launchers only after inference parity;
- exercise long context and accepted-length rollback.

Exit: one launcher per recurrent ABI and separate serving/training policies.

### T5: MoE and transport

Order:

1. common routing and pointer-table mechanics;
2. Qwen35 local MoE;
3. DSv4 local grouped MoE;
4. W4AFP8/NVFP4 path;
5. DeepEP normal-latency;
6. DeepEP low-latency.

Long-prefill and concurrent gates are mandatory. Transport paths retain separate
fallback ladders.

Exit: `moe.rs` is a facade/shared-types root with explicit policy modules.

### T6: Sampling, speculative primitives, and collectives

Migrate typed launch mechanics. Keep compatibility decisions, sequence state,
and overlap placement in model executors.

Exit: direct FFI calls remain only where an explicit exception documents why.

### T7: Autograd reuse

First classify all 29 autograd NVRTC files as forward, backward, optimizer,
rollout, bridge, or layout support. Keep autograd-specific math and compilation
inside `autograd`. For each ABI already proven in serving:

- replace the matching autograd direct call with the typed launcher;
- preserve tape-visible dtype, saved state, stream, and gradient semantics;
- run forward, backward, accumulation, optimizer, and OPD E2E gates;
- delete the old launch path in the same tranche.

For retained NVRTC kernels:

- group the source catalog and exported symbol list by semantic family;
- verify every exported symbol has one Rust launch owner;
- bind the compiled module to source, flags, SM, dtype, driver, and NVRTC
  identity;
- compile every required dtype during initialization or declared warmup;
- reject any first-use compile or function lookup in steady-state OPD steps.

Exit: identical device operations share launchers; training-owned kernels have
one compile path, one launch owner, and complete artifact provenance.

### T8: Registry and evidence closure

- expand `operators/registry.toml` to every migrated semantic family;
- bind implementation IDs to source, provider, legality, fallback, model gates,
  artifact identity, and evidence;
- generate/check static IDs at build time;
- update `docs/index.md`, support matrix, and CHANGELOG on phase exits and
  accept-or-reject verdicts.

Exit: code ownership, runtime engagement, evidence, and artifact provenance
agree for every supported CUDA model.

## Commit and file limits

Each runtime commit follows this shape:

```text
1 family facade/policy file
1 cuda-kernels safe-launcher file
1 consumer/model file when required
1 focused test or harness file
1 dated experience report
```

Five files maximum. Mechanical file moves and behavior changes land in separate
commits unless separating them would leave duplicate live paths. No compatibility
adapter survives a tranche.

## Parallelization

T0 and T1 are sequential because they establish shared contracts. After T1:

| Lane | Work | Dependency |
| --- | --- | --- |
| A | quant-linear child plan | T1 launcher contract |
| B | attention/KV subtranches | T1 launcher contract |
| C | registry inventory and checks | T0 ownership IDs |

MoE waits for the common launch and evidence patterns. Autograd waits for the
matching serving launcher. Two lanes cannot edit the same root facade in
parallel; integrate one family before the next begins.

## Rollback

- archive the baseline binary and kernel manifest before every runtime tranche;
- one family moves per commit, so revert restores the previous direct path;
- rebuild from restored source and verify binary/kernel identity;
- rerun the operator and model gates;
- preserve KILL evidence in `docs/experience/errors/`;
- do not keep both paths behind a permanent flag after verdict.

## NOT in scope

- cross-backend CUDA/Metal/HIP/Vulkan operator traits;
- one runtime registry that dynamically chooses every kernel;
- rewriting vendor kernels into a common source language;
- merging TileLang build truth with runtime legality;
- changing model architecture or scheduler semantics;
- adding unsupported CUDA model families;
- changing performance thresholds during structural tranches;
- deleting a specialized kernel solely to reduce file count;
- reorganizing all autograd code before serving launchers are stable.

## Definition of done

1. Every production CUDA launch has one semantic family, one launch owner, one
   ABI declaration, and one implementation ID.
2. Model files contain layer/state orchestration and family-policy calls, with no
   raw FFI launch mechanics.
3. Shared mechanisms are implemented once in `cuda-kernels` or the owning
   `infer-cuda` family.
4. Model-specific legality, state, fallback, and thresholds remain explicit.
5. No new allocation, synchronization, dynamic dispatch, registry lookup,
   capability query, or launch appears in the forward hot path.
6. Structural tranches preserve launch symbols, arguments, counts, counters,
   memory, capture behavior, numerical outputs, and end-to-end performance.
7. Qwen3, Qwen35/Qwen3.8, DSv4, GLM, and OPD gates pass independently; all 29
   autograd NVRTC source files are classified and owned.
8. Unsupported model routes still fail before execution.
9. Operator legality, measured evidence, and artifact identity agree.
10. Old routes, temporary adapters, false qualification, and unexplained TODOs
    are deleted.

## Acceptance verdict

The umbrella plan is complete when T0-T8 close with per-family evidence. Each
tranche can PASS or KILL independently. A KILL preserves the current accepted
path and blocks the next dependent tranche until the mechanism is understood.

Organization alone carries no performance claim. The accepted result is a
lower-entropy codebase with identical device work and independently verified
model behavior.
