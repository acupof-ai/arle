# CUDA operator organization across kernels and models

> Status: Proposed
>
> Scope: all production CUDA operator paths for Qwen3, Qwen3.5/3.6/3.8,
> DeepSeek V4, GLM-5.2, and OPD autograd.
>
> First implementation unit:
> [Qwen CUDA quant-linear dispatch consolidation](2026-08-20-qwen-quant-linear-dispatch-consolidation.md).

## Decision

Use five ownership layers:

```text
model execution
    -> model/operator policy
        -> typed operator-family launcher
            -> private FFI ABI
                -> native / vendor / AOT / JIT / NVRTC provider
```

Aggregate launch mechanics, validation, workspace lifecycle, implementation
identity, and evidence reporting. Keep model order, mutable state, legality,
fallback priority, and performance thresholds in the owning model or operator
family.

The steady-state path remains statically dispatched Rust and direct CUDA calls.
Forward execution contains no trait objects, string lookup, registry parsing,
heap allocation, locks, host readback, JIT compilation, or new synchronization.

## Why this fits ARLE

The device source tree already has useful operator families. The disorder is in
the consumer layer, where model policy, raw ABI calls, scratch ownership,
provider selection, and counters overlap.

Snapshot from the 2026-08-20 checkout:

| Surface | Size | Current responsibility overlap |
| --- | ---: | --- |
| `crates/infer-cuda/src/` | 63,437 Rust lines | Model orchestration, route policy, scratch, and raw FFI calls |
| `attention.rs` | 8,543 lines | Qwen paged attention, DSv4 MLA, preparation, routing, and launches |
| `loader.rs` | 6,709 lines | Format detection, upload, assembly, repack, source retention, TP, and MoE |
| `moe.rs` | 4,429 lines | Qwen and DSv4 routing, scratch, transport |
| `ops/quant_linear.rs` | 2,120 lines | Two entry routes and several weight-family policies |
| `crates/cuda-kernels/src/ffi/` | 5,868 lines, 358 extern symbols (330 hand-written, 28 TileLang-generated) | Grouped ABI declarations with direct external consumers |
| `crates/cuda-kernels/src/*.rs` typed wrappers | 11,255 lines | Existing typed-launcher layer; 113 of 358 FFI symbols already wrapped |
| `crates/cuda-kernels/csrc/` | 72 `.cu`/`.cuh` files | Existing device-math family layout |
| `crates/autograd/src/backend_cuda/kernels/` | 29 CUDA files | Autograd-owned NVRTC forward, backward, optimizer, rollout, and bridge kernels |

The raw-FFI census behind this plan: ~243 call sites in 28 files consume 149
distinct symbols outside `cuda-kernels`; ~142 symbols have no typed wrapper.
Model modules (`attention.rs`, `qwen35*`, `dsv4/*`, `tp.rs`, `hc.rs`,
`loader.rs`) hold 163 of them; `ops/` holds 40. `infer-cuda/src/moe.rs` is the
reference consumer of the target pattern: zero raw FFI calls, all launches
through `cuda_kernels::moe` typed wrappers. Its remaining disorder is model-
policy mixing (Qwen and DSv4 in one file), not ABI access.

The correct unit of reuse is the launch mechanism. Model semantics remain
different:

- Qwen3 owns dense paged-KV execution;
- Qwen35 owns hybrid full attention, recurrent state, MoE, MTP/DSpark, and
  speculative rollback;
- DSv4 owns MLA/DSA/CSA/HCA, expert topology, and DeepEP policy;
- OPD owns tape-visible state, gradients, accumulation, and optimizer behavior.

A global `Operator` interface would need a large context object, optional
fields, or runtime type checks. Concrete family launchers preserve type safety
and keep the hot path visible to the compiler.

## Industry reference

The engines differ in implementation language and extension requirements. They
share three boundaries: semantic operator, provider implementation, and model
execution policy.

| Engine | Current structure | Applicable lesson | ARLE adaptation |
| --- | --- | --- | --- |
| vLLM | `CustomOp`, attention backends, quantization methods | Separate semantic operation from platform implementation | Use concrete Rust family APIs because ARLE has no PyTorch or out-of-tree dispatch requirement |
| SGLang | `kernels.ops.<family>`, `KernelSpec`, capability filtering, cached callable | Give every operator family one public home and track provider provenance separately | Resolve static facts during load/warmup and use enum dispatch in forward |
| TensorRT-LLM | compiled graph, plugins, direct CUDA launchers, prepared workspace, CUDA Graph | Finish selection and resource preparation before steady-state execution | Preserve ARLE's dynamic scheduler and model state while adopting the same runtime lifecycle |

Primary references:

- [vLLM CustomOp design](https://github.com/vllm-project/vllm/blob/main/docs/design/custom_op.md)
- [vLLM attention backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backend.py)
- [vLLM linear methods](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/linear.py)
- [SGLang unified kernel namespace](https://github.com/sgl-project/sglang/issues/29630)
- [SGLang kernel specification](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/spec.py)
- [SGLang deterministic selector](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/selector.py)
- [TensorRT-LLM model, compiler, and plugin structure](https://nvidia.github.io/TensorRT-LLM/0.18.2/architecture/core-concepts.html)
- [TensorRT-LLM runtime structure](https://nvidia.github.io/TensorRT-LLM/developer-guide/overview.html)

ARLE should copy the ownership boundaries and runtime lifecycle. Python class
registries and TensorRT whole-graph compilation solve different product
constraints.

## Scope

### Model routes

| Route | State | Policy owner |
| --- | --- | --- |
| Qwen3 dense | Supported | Dense Qwen executor and shared operator families |
| Qwen3.5/3.6 hybrid dense/MoE | Supported | Qwen35 executor, attention, recurrent, MoE, and quant-linear families |
| Qwen3.8 mixed NVFP4/per-channel FP8 | Supported through Qwen35 | Qwen35 plus quant-linear storage and route policy |
| DeepSeek V4 | Supported | DSv4 executor and DSv4-specific attention/MoE policies |
| GLM-5.2 | DSv4 adapter, verification pending | DSv4 policy with separate GLM load and model gates |
| OPD CUDA | Supported training path | Autograd CUDA backend and OPD orchestration |
| Qwen3-MoE public schema | Unsupported | Load-time classification failure remains |
| Gemma4/DiffusionGemma CUDA | Unsupported forward | Executor construction failure remains |

Schema similarity never grants model support. Each model route needs its own
load, correctness, and performance evidence.

### Operator families

Every production CUDA kernel maps to one of these families:

1. embedding and positional preparation;
2. normalization and elementwise operations;
3. dense and quantized linear projection;
4. paged, full, linear, MLA, DSA, CSA, HCA, FA3, and FlashMLA attention;
5. KV addressing, packing, quantization, restore, and page preparation;
6. recurrent GDR, convolution, and FlashQLA;
7. MoE routing, grouped GEMM, activation, combine, DeepEP, and local fallback;
8. sampling and speculative draft/verify primitives;
9. TP/CP/EP collectives and custom all-reduce;
10. autograd forward, backward, optimizer, rollout, and OPD fused operations.

A device kernel has one semantic family, one launch owner, and one ABI
declaration. Several models may reuse the same launcher.

## Existing foundations

| Existing mechanism | Role in this plan |
| --- | --- |
| `crates/cuda-kernels/csrc/<family>` | Keep the current device-source family layout |
| `crates/cuda-kernels/src/ffi/<family>.rs` | Keep private ABI declarations and generated AOT resolution |
| `DeviceContext` and typed device buffers | Reuse stream, device, pointer, and storage lifetimes |
| Typed launcher modules (`cuda-kernels/src/{moe,paged_kv,kv_quant,tensor,ring_attention,attention,collective}.rs`) | 11,255-line proven launcher layer; `infer-cuda/src/moe.rs` is the reference consumer | Extend to unwrapped families; T1 generalizes this pattern |
| `OperatorDispatchStats` | Extend request-boundary engagement reporting |
| `crates/cuda-kernels/kernels.toml` | Keep as TileLang AOT build truth |
| `operators/registry.toml` | Expand as semantic operator and implementation truth after ownership stabilizes |
| `benchmarks/operators/optimal.json` | Keep qualified generated policy inputs and artifact binding |
| `scripts/reduce_operator_evidence.py` | Keep the qualification gate |
| `KERNEL_BUILD_ID` and backend artifact identity | Keep build provenance outside per-step dispatch |
| autograd `KernelCache` | Keep NVRTC compilation and dtype-specific module ownership in autograd |

Three truth layers remain independent:

```text
operator legality     operators/registry.toml + generated static policy
measured evidence     benchmark JSON + numerical/model gates
artifact provenance   binary/kernel manifest + model revision
```

`kernels.toml` defines the TileLang build set. Runtime legality comes from the
operator registry and generated static policy.

## Target architecture

### Load and warmup

```text
checkpoint/config
   -> classify model and weight format
   -> validate source shape and byte length
   -> shard/fuse/repack
   -> decide source retention
   -> validate final storage state
   -> derive static route facts
   -> allocate maximum declared workspace
   -> preflight/JIT/warm selected providers
   -> publish immutable weights + mutable per-slot state
```

Loader decisions become explicit metadata. Forward execution never reconstructs
them from unrelated `Option` fields or environment variables.

### Forward

```text
ForwardPlan
   -> model executor
       owns layer order, state, KV, TP/EP, speculative semantics
   -> family policy
       selects a route from static metadata and dynamic shape
   -> typed family launcher
       validates views and borrows prepared workspace
   -> private FFI
   -> provider
       native | vendor | TileLang AOT | DeepGEMM JIT | autograd NVRTC
```

Static route facts are computed during load or warmup. Live row count, context
length, and accepted speculative depth remain explicit per-step inputs.

### Ownership

| Layer | Ownership | Excluded ownership |
| --- | --- | --- |
| `cuda-kernels/csrc` | Device math and vendor integration | Model selection and serving policy |
| `cuda-kernels/src/ffi` | Private C ABI declarations | Public model APIs |
| `cuda-kernels/src/<family>` | Typed launchers and family-local device helpers | Qwen and DSv4 route priority |
| `infer-cuda/src/ops/<family>` | Serving route policy, scratch, and counters | Scheduler and HTTP |
| `infer-cuda/src/qwen*` | Qwen layer order, KV, state, and speculative behavior | Raw CUDA ABI calls |
| `infer-cuda/src/dsv4*` | DSv4/GLM layer order, MLA, MoE, and transport policy | Runtime kernel registry |
| `autograd/src/backend_cuda*` | Training policy, gradients, tape state, and NVRTC provider | Serving model policy |
| `operators/` and `benchmarks/operators/` | Offline legality, evidence, and provenance | Per-step dispatch |

## Common contracts

### Typed launchers

Each CUDA ABI has one concrete Rust launcher. It:

1. accepts typed CUDA buffers/views and scalar dimensions;
2. checks size, alignment, dtype, and buffer length;
3. keeps pointer guards alive through submission;
4. converts host integers to ABI types with checked conversions;
5. invokes one FFI symbol;
6. reports implementation and shape context on failure.

Example:

```rust
pub fn launch_fp8_block_gemv(
    ctx: &DeviceContext,
    weight: &Fp8BlockWeight<'_>,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
    m: usize,
) -> Result<()>;
```

Mandatory launchers return `Result<()>`. Optional fast paths return:

```rust
#[must_use]
enum Engagement {
    Launched,
    Declined,
}
```

`Declined` requires a valid caller-owned fallback. An error means the selected
route failed or its storage contract is invalid.

### Route selection

Each family separates pure selection from device effects:

```text
select_route(metadata, phase, dynamic_shape) -> Route
launch_route(route, buffers, workspace)       -> Result
```

Routes use family-local enums and exhaustive `match`. `Fp8Route`, `MlaRoute`,
and `MoeTransportRoute` retain their own legality rules.

### Workspace lifecycle

Every family records its maximum shape, owner, allocation point, reset rule,
capture lifetime, restore behavior, and stream assumptions.

```text
load/warmup: ensure_capacity(max_envelope)
capture:     borrow fixed addresses
forward:     borrow/reset/launch
restore:     rebuild transient pointers and retain logical capacity
```

Forward-time workspace growth, D2H metadata reads, and synchronization are
release blockers. Scratch types remain concrete because layout and reset rules
are part of correctness.

### Identity and evidence

- implementation IDs are compile-time constants checked against
  `operators/registry.toml`;
- counters increment after successful submission;
- name/vector allocation happens at `/v1/stats` or another request boundary;
- build identity, kernel bundle, model revision, and policy hash are reported
  together;
- a component probe provides diagnostic evidence;
- model qualification requires the exact candidate binary and real model E2E
  evidence.

Autograd NVRTC identity includes concatenated source hash, compile flags, SM,
tape dtype, CUDA driver, and NVRTC version. Required dtype modules compile at
backend initialization or declared warmup.

## Family boundaries

| Family | Shared mechanism | Retained policy |
| --- | --- | --- |
| Linear/quant | Typed GEMM/GEMV/repack/quant launchers | Weight-format storage, M route, source retention, provider priority |
| Attention/KV | Metadata checks, address translation, attention and KV launchers | Qwen paged KV, Qwen35 recurrent/full split, DSv4 latent state and attention order |
| Recurrent | GDR, conv1d, and FlashQLA launch mechanics | State mutation, chunk replay, accepted-length rollback, backward semantics |
| MoE | Routing primitives, pointer tables, activation and grouped-GEMM launchers | Expert topology, top-k, scale layout, local/DeepEP fallback, capture rules |
| Sampling/speculative | Logits, mask, penalty, sample, draft/verify launchers | Request compatibility and accepted-chain state |
| Collectives | NCCL/custom launchers, dtype and reduction types | Placement, overlap, TP/CP/EP model policy |
| Autograd | Identical shared launchers where ABI and numerics match | Tape state, saved tensors, gradients, accumulation, optimizer, NVRTC-only math |

Training-owned CUDA files stay under `autograd`. Sharing requires an identical
ABI, stream contract, and numerical contract proven by serving and training
gates.

## File target

```text
crates/cuda-kernels/
  csrc/<family>/              device math and vendor integration
  src/ffi/<family>.rs         private ABI declarations
  src/<family>.rs             typed launchers and family types

crates/infer-cuda/src/
  ops.rs                      small common facade
  ops/<family>.rs             serving route owners
  attention.rs                facade and shared attention types
  attention/*.rs              semantic attention families
  moe.rs                      facade and shared MoE types
  moe/*.rs                    Qwen, DSv4, DeepEP, and W4AFP8 policies
  loader.rs                   common detection/upload entry
  qwen35_load.rs              Qwen35 assembly
  dsv4/load.rs                DSv4/GLM assembly
  graph.rs                    capture/re-bake machinery; owner of baked-pointer repair when workspace moves
  decode_graph.rs             decode capture keys and workspace-epoch invalidation

crates/autograd/src/backend_cuda/
  kernels.rs                  NVRTC module lifecycle and family catalogs
  kernels/*.cu                training-owned CUDA sources
  <family>.rs                 tape-visible launch policy
```

File length is secondary evidence. Cohesive ownership determines whether a
split is useful. A migration deletes the old entry in the same tranche.

## Execution plan

Each runtime tranche changes one semantic family, touches at most five files,
and has its own correctness and performance receipt.

T3, T5, and T7 are programs of named sub-tranches, not single tranches. Each
sub-migration (one attention family, one MoE route class, one autograd
catalog) gets its own exit condition and receipt. A family extraction may use
a pure mechanical-move commit — the five-file cap is relaxed for moves, line
count is not capped — followed by a behavioral commit. Shared private helpers
move with their largest consumer or into a family-shared module, named in the
sub-tranche plan.

| Phase | Work | Exit condition |
| --- | --- | --- |
| T0 Inventory | Map every production launch to semantic ID, provider, consumer, model, phase, SM, workspace, capture support, and gate | Every launch has one family and owner; registry gaps are explicit |
| T1 Launcher boundary | Extend the proven typed-launcher pattern (`cuda-kernels/src/moe.rs` wrappers, `infer-cuda/src/moe.rs` consumer) to embedding, norm, and elementwise; land the second registry+generated-policy binding here | Direct ABI calls removed for migrated operations; launch receipt is identical; registry schema proven on two families |
| T2 Qwen quant-linear | Execute the child plan across M=1 and batched entry points | One route owner per Qwen weight family; retained storage validates at load |
| T2L Loader decomposition | Split `loader.rs` into common detection/upload entry + `qwen35_load.rs` + `dsv4/load.rs`; make load/warmup pipeline metadata explicit | Common loader owns only detection/upload; model assembly in named owners; per-family final storage validation runs at publish |
| T3 Attention and KV | Migrate non-paged, Qwen paged, Qwen35 full/recurrent preparation, DSv4 MLA family, and quantized KV in that order | Root attention module contains facade/shared types; family modules own routes |
| T4 Recurrent | Consolidate GDR, conv1d, and FlashQLA launch mechanics | Serving state and training backward policies remain separate; ABI launchers are singular |
| T5 MoE and transport | Migrate common routing, Qwen local, DSv4 local, W4AFP8/NVFP4, DeepEP normal, then DeepEP low-latency | Long-prefill and concurrent gates pass for each route |
| T6 Sampling and collectives | Migrate typed launch mechanics | Compatibility, sequence state, placement, and overlap stay model-owned |
| T7 Autograd | Classify all 29 NVRTC sources; share proven launchers; organize retained compile/launch catalogs | Every exported symbol has one Rust owner and complete artifact identity |
| T8 Evidence closure | Bind legality, evidence, model gates, and artifact identity for all routes; registry schema and codegen already proven on two families in T1 | Runtime engagement, registry, evidence, and binary agree for all supported routes |

Dependencies:

```text
T0 -> T1 -> T2 -> T2L
          -> T3 -> T4
          -> T5
          -> T6

matching serving launcher -> T7
T2..T7, T2L complete      -> T8
```

T0 and T1 are sequential. After T1, independent families may use separate
lanes when they do not edit the same root facade; T2L may run parallel with
T3/T5/T6 (different root files). T7 follows the matching serving launcher. T8
closes after all runtime owners stabilize.

The registry schema and generated-policy codegen are generalized on the T1
family — the second binding after `qwen.fp8_dense_projection`. Each tranche
T2..T7 adds one registry binding for its family, so T8 is mechanical closure,
not construction.

## Verification

### Path coverage

```text
load
  +-- supported model/format -> validate -> prepare -> warm -> publish
  `-- unsupported/invalid    -> explicit load failure

forward
  -> select_route
      +-- mandatory route -> launch -> success/error
      `-- optional route  -> launched | declined -> valid fallback

capture
  -> fixed workspace/address -> replay with identical launch sequence

restore/offload
  -> rebuild transient pointers -> validate retained storage -> resume

OPD
  -> warm NVRTC dtype module -> forward -> backward -> accumulate -> optimizer
```

The smallest tests that close these branches are:

- one table-driven pure route test per family;
- one typed-launch validation test per ABI contract class;
- one CUDA numerical harness per changed operation family;
- one model gate per affected model route;
- one captured/eager parity gate for capture-capable paths;
- one OPD forward/backward/optimizer gate for shared training launchers.

### Operator gates

| Family | Required gate |
| --- | --- |
| Linear/quant | FP32/FP64 reference, production shapes, M sweep, three seeds, saturation/extrema |
| Attention | Reference attention, ragged rows, page tails, min/max context, split modes |
| KV | Exact address/pack round trip, page boundary, quant extrema, restore |
| Recurrent | Reference forward/state, chunk boundary, replay, long sequence, backward where used |
| MoE | Route counts, expert offsets, empty expert, max routed rows, long prefill, scale semantics |
| Sampling | Deterministic seed, mask, bias, penalties, forced token, invalid distribution |
| Collectives | Per-rank parity, non-divisible shards, stream ordering, TP/CP/EP sizes |
| Autograd | Forward reference, gradient reference, accumulation, optimizer step |

### Model gates

| Model/config | Required gate |
| --- | --- |
| Qwen3 dense BF16 | Prefill, decode, paged KV, prefix reuse, c=1 and batched |
| Qwen3.6 FP8 | Hybrid attention, recurrent state, MoE, MTP/DSpark, supported FP8 KV |
| Qwen3.8 NVFP4 | FP4 and per-channel FP8, source release, long-agent prefix reuse |
| Qwen W8A16 | Repacked/unrepacked shapes and untied quantized `lm_head` |
| DSv4 TP/EP | MLA/DSA, MoE, DeepEP/local fallback, MTP/DSpark, long prefill, concurrent decode |
| GLM-5.2 | Separate config, load, and forward gate on the DSv4 implementation |
| OPD CUDA | Teacher logits, rollout, backward, optimizer, offload/reload, LoRA re-merge |

Runtime route changes run `scripts/lever_gate.sh` and
`scripts/needle_gate.py temp` at 512/4096/16384/32768, three repetitions, on
the exact candidate binary.

Structural tranches use drift-band acceptance: at least three matched trials
per arm, median plus range, and an unresolved negative median blocks. The
strict allocation/sync/launch-count invariants apply to the launch-receipt
comparison, not to re-derived per-step measurements.

All CUDA gates are remote (H20); a Mac workstation runs `cargo check` only. A
tranche may claim local exit with typecheck plus a `pending-remote` report
stub per the bench spec; the remote verdict closes it. GLM-5.2 gates remain
pending-remote until the support matrix flips and do not block DSv4 tranche
exits on their own.

### Performance receipt

For every structural tranche, compare archived baseline and candidate:

1. kernel symbol sequence and launch count by phase;
2. grid, block, and shared-memory arguments;
3. fixed addresses used by capture;
4. implementation counters;
5. allocated bytes and workspace high-water marks;
6. host submit time and synchronization count;
7. operator numerical output;
8. model correctness;
9. end-to-end latency and throughput.

Acceptance:

- address, copy, mask, and index operations are bit-exact;
- pointwise output changes stay within one output-dtype ULP and preserve finite
  and sign classes;
- GEMM, reduction, attention, and recurrent reference error worsens by at most
  5%;
- route counters match exactly for the same request;
- allocation, synchronization, event, stream-wait, and launch counts do not
  increase;
- the canonical workload has no unresolved negative median after at least
  three matched trials per arm;
- errors, incomplete outputs, loops, and timeouts remain zero.

Every runtime tranche writes a dated `docs/experience/wins/` or `errors/` entry
under `docs/bench-and-trace-spec.md`. GPU kernel changes include measured
before/after CUDA profiling. A structural change with altered kernel work moves
to a separate behavioral tranche.

## Failure handling

| Failure | Prevention and detection |
| --- | --- |
| Generic facade hides model legality | Family-local route enums and model-owned policy |
| Consumer bypasses validation | Private FFI and source-path ownership check |
| Wrapper adds allocation or synchronization | Launch receipt and captured/eager trace |
| Loader releases required fallback storage | Final storage validation and forced-decline test |
| Static cache captures dynamic row/context state | Route-input audit and boundary tests |
| Counter increments before submission | Increment after successful launch submission |
| Workspace address changes under capture | Warmup envelope and address receipt |
| Registry claims an absent implementation | Binary manifest and generated-policy freshness check |
| Component probe grants model qualification | Evidence reducer requires real model E2E artifact |
| Shared launcher changes training numerics | Serving and OPD gates before old-path deletion |
| File split leaves two live routes | One complete family per tranche and same-commit deletion |
| Unsupported model reaches a shared route | Classification and load-failure tests |
| DeepGEMM misses a production tail shape | Clean-cache production-shape warmup test |
| W4AFP8 smoke misses routed-row overflow | Long-prefill MoE gate beyond block caps |

Rollback targets the current tranche: revert all of its commits and rebuild
against the archived baseline binary/kernel manifest. A latent failure in a
landed tranche is handled by reverting the dependent chain or fixing forward;
tranches are sequentially dependent, so reverting an earlier tranche alone
does not compile. Permanent dual paths and compatibility adapters are removed
after the verdict.

## Commit rules

Each runtime commit contains at most:

```text
1 family facade/policy file
1 cuda-kernels launcher file
1 model consumer file when required
1 focused test or harness
1 dated experience report
```

Mechanical moves and behavioral changes use separate commits whenever that
separation leaves one valid live path. Each commit compiles and closes its own
operator and model gates.

## NOT in scope

- a cross-backend CUDA/Metal/HIP/Vulkan operator trait;
- a runtime registry that dynamically selects every kernel;
- rewriting vendor kernels into one source language;
- combining TileLang build truth with runtime legality;
- changing scheduler or model architecture semantics;
- adding unsupported CUDA model families;
- changing performance thresholds during structural tranches;
- deleting specialized kernels to reduce file count;
- reorganizing unrelated autograd code;
- whole-graph compilation of ARLE's dynamic model execution.

## Completion criteria

1. Every production CUDA launch has one semantic family, launch owner, ABI
   declaration, and implementation ID.
2. Model modules own layer/state orchestration and contain no raw CUDA ABI
   calls (excluding `#[cfg(test)]` call sites and registered fn-pointer
   tables).
3. Shared launch mechanics exist once in `cuda-kernels` or the owning family.
4. Model legality, state, fallback, thresholds, and capture rules remain
   explicit.
5. Forward adds no allocation, synchronization, dynamic lookup, capability
   query, JIT compilation, or launch.
6. Structural tranches preserve kernel work, memory, numerics, capture behavior,
   and end-to-end performance.
7. Qwen3, Qwen35/Qwen3.8, DSv4, GLM, and OPD gates pass independently.
8. All 29 autograd NVRTC files are classified and owned.
9. Unsupported model routes fail before execution.
10. Operator legality, measured evidence, and artifact identity agree.
11. Old routes, temporary adapters, false qualification, and unexplained TODOs
    are removed.

Each tranche receives `PASS` or `KILL`. `KILL` preserves the last accepted path
and blocks dependent work until the mechanism is understood. Organization alone
carries no performance claim.
