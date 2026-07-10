# Ideal operator platform: truth, performance, development, and release

> Status: Active design - proposed 2026-07-10. This defines the target state
> and migration. It licenses no runtime change, benchmark claim, or default
> flip before the relevant tranche passes its gates.

## Verdict

ARLE needs one declarative operator platform, not one kernel compiler.

The ideal state is:

1. One logical operator graph defines semantics, ABI, implementations, legal
   shapes, state mutation, fallback, build inputs, and verification ownership.
2. The graph generates FFI, build work, runtime selection types, inventory, and
   documentation. Hand-maintained parallel tables are deleted.
3. Kernel compilation is an explicit producer operation. Normal Cargo builds
   only compile Rust and link exact, verified artifacts.
4. Operator benchmarks persist canonical JSON. Runtime winners are generated
   only from correctness-qualified, same-site measurements.
5. Kernel artifacts live in a content-addressed OCI registry. GitHub Releases
   contain formal product binaries and frozen offline bundles, not the
   high-frequency development cache.
6. A supported machine downloads a matching product and runs without Cargo,
   Python, TileLang, CMake, a C++ compiler, or nvcc.

TileLang, native CUDA, FlashMLA, DeepGEMM, MLX, HIP, and Vulkan remain eligible.
The measured winner owns each hardware and workload cell.

## Why the current shape is not the target

The 2026-07-10 tree already has useful pieces:

- 48 TileLang rows in `crates/cuda-kernels/kernels.toml`;
- 61 native CUDA translation units;
- 312 CUDA FFI declarations; the earlier 313 count included the ordinary Rust
  helper `ffi/nccl.rs::check`;
- per-SM TileLang source hashes and persistent local cache;
- T1 `sm_80/86/89/90`, Blackwell, sm70, and Metal release lanes;
- a zero-Python TileLang bundle consumer;
- a broader DSv4 prebuilt archive and manifest;
- `sccache` for Rust and nvcc;
- same-process resident A/B harnesses for important CUDA paths.

They do not form one system:

- `kernels.toml` describes only TileLang.
- Semantic operators, native helpers, provider kernels, FFI, runtime
  conditionals, and performance decisions are separate truth surfaces.
- Cargo `build.rs` can unexpectedly become a kernel compiler and toolchain
  installer boundary.
- `arle-kernels-latest.tar.gz --clobber` is mutable and cannot reproduce a
  historical checkout exactly.
- Release bundles and DSv4 fast-build archives use different protocols.
- DeepGEMM still compiles shapes at runtime.

The migration reuses proven code, then deletes these structural splits.

## M0 evidence already established

Detailed evidence and commands:
[`../reviews/2026-07-10-operator-platform-m0-readonly-audit.md`](../reviews/2026-07-10-operator-platform-m0-readonly-audit.md).

This is measured or source-traced evidence, not target-state inference:

- The 312 CUDA FFI declarations split across 13 domains. 256 have a Rust
  reference outside their declaration module; 53 have no runtime Rust caller,
  two are test-only, and one NCCL symbol is consumed by a same-module helper.
  Zero Rust references means `dead candidate`, not deletion license: C-to-C,
  dynamically loaded, stub, and provider paths still need archive/link/runtime
  evidence.
- The checked-in 61 CUDA TUs are not the actual build DAG. `build.rs` swaps
  FlashMLA/FA3 stubs and real shims, adds vendor TUs, and changes custom
  all-reduce membership by feature.
- Of 48 TileLang rows, static tracing finds eight default-reachable HD128/KV8
  BF16 rows, three doubly gated FlashQLA rows, and 37 rows with no current runtime
  resolver path. Runtime counters per product lane must confirm this before any
  deletion.
- Seven existing call structures are genuine composites: paged-attention
  prep/run/finalize, fused MHC pre+norm, Qwen gate+up, the three-stage DeepGEMM
  MoE pipeline, fused DSv4 Q/K preparation, fused DSA index/cache operations,
  and FlashMLA multi-source packing.
- CUDA dispatch is distributed across model/backend branches. Qwen dense and
  MoE use hard-coded row floors plus weight/preflight conditions; DSv4 fused
  projections and attention use separate default-on gates. There is no current
  central policy to translate mechanically.
- `/v1/stats` has no operator, policy, or artifact counters. The existing
  GuideLLM trace wrapper still parses legacy `key=value` fields from JSON, so a
  canonical evidence run cannot yet prove engagement through that consumer.
- DeepGEMM runtime identity includes more than `(M,N,K,SM)`: generated layout,
  GPU SM count, `DG_NUM_SMS`, DeepGEMM/CUTLASS sources, nvcc, host compiler, and
  flags affect output. The current JIT digest does not cover this full set.
- DeepEP has two different build products: a `cuda-kernels` external sidecar
  built with NVSHMEM disabled, and an in-process `deepep-sys` archive that may
  use NVSHMEM. Their liveness and ownership must be decided before unification.
- `ARLE_CUDA_KERNEL_SET=dsv4_flash` disables TileLang AOT but still compiles the
  recursively collected native CUDA TUs. It is not a native incremental build
  graph.
- On the local Apple Silicon development host, the canonical CUDA/no-CUDA check
  passed in 2.60 seconds after one crate change and 0.56 seconds hot, with an
  explicit `skipping CUDA/TileLang kernel compilation` result.
- On the current 8xH20 sm90 pod, recent release builds take 59-113 seconds even
  with warm state. The build environment records a prior forced TileLang kernel
  regeneration of 251 seconds versus 53 seconds from its persistent cache.
- The pod product dynamically requires libc, libstdc++, CUDA driver/runtime,
  cuBLAS/cuBLASLt, NCCL, OpenSSL, and system libraries. DeepGEMM also uses
  `dlopen` and runtime compiler subprocesses, which `ldd` alone cannot reveal.
- The pod is an 8xH20 NV18-connected host with two NUMA domains. This confirms
  that the distributed policy key must record realized topology and collective
  backend, not only requested TP/EP configuration.

## Non-negotiable decisions

### D1: One operator graph, multiple providers

"Unified operators" means one contract and selection system. It does not mean
rewriting a faster official kernel in TileLang.

### D2: Cargo never compiles GPU kernels implicitly

`cargo build` compiles checked-in generated Rust and verifies exact artifacts.
It does not generate registries or invoke Python, TileLang, nvcc, CMake, Ninja,
or provider JIT.

Kernel producers use an explicit command:

```text
cargo xtask kernels build ...
```

### D3: Content address inputs and outputs separately

- `build_id`: SHA-256 of one build-DAG node's canonical transitive inputs. A
  source checkout computes it before download or build.
- `bundle_digest`: OCI digest of the exact output manifest and layers.

Every graph node has its own build ID. A policy-only change therefore rebuilds
the policy and product nodes, not kernel nodes. The same `build_id` producing a
different `bundle_digest` is a reproducibility failure.

### D4: OCI is the kernel store; GitHub Release is the product store

GitHub Actions Artifact handles temporary PR outputs. GHCR stores immutable,
content-addressed kernel layers. GitHub Releases store tagged ARLE products and
an optional offline kernel bundle referenced by exact OCI digest.

### D5: No hidden runtime compilation

Every production implementation is AOT, packaged, or explicitly labeled
`compile-on-first-load`. Request execution never discovers a compiler.

### D6: Evidence selects; prose does not

Markdown can explain a decision. Only canonical benchmark JSON can generate an
eligible runtime winner.

### D7: Explicit composites, not a general graph compiler

The selection unit may be one semantic operator or one named composite such as
fused QKV, quantize-once projection groups, or a captured decode block. A
composite declares the exact operator sequence and boundary layouts it replaces.
ARLE does not add a general graph-rewrite engine: only explicitly registered,
benchmarked composites are eligible.

### D8: Legality includes resources and runtime compatibility

Shape and SM are insufficient. Every implementation declares workspace,
alignment, stream, async-lifetime, capture, reentrancy, determinism, and launch
requirements. Products bind the dynamic runtime ABI they were built and tested
against, including libc, CUDA runtime, and distributed libraries.

### D9: Evidence expires on load-bearing drift

A winner is valid only for its evidence compatibility key. Changes to kernel
bytes, layouts, runtime scheduling, driver/toolkit family, distributed topology,
or benchmark protocol invalidate the affected cells. There is no permanent
winner detached from the bytes and environment that proved it.

## Target repository shape

```text
operators/
  schema.toml
  semantic/
    attention.toml
    gemm.toml
    moe.toml
    recurrent.toml
    kv.toml
    quant.toml
    collective.toml
    sampling.toml
  implementations/
    cuda.toml
    metal.toml
    hip.toml
    vulkan.toml
  composites/
    cuda.toml
    metal.toml
  abi/
    cuda.toml
    metal.toml
  generated/
    build-graph.json
    inventory.json
    operator-inventory.md

benchmarks/operators/
  schema.json
  runs/
  index.json
  optimal.json

xtask/
  src/operators.rs
  src/kernels.rs
  src/artifacts.rs
  src/benchmarks.rs

crates/infer-plan/src/generated/operators.rs
crates/cuda-kernels/src/generated/{ffi,registry}.rs
crates/infer-metal/src/generated/registry.rs
```

`operators/semantic` is backend-neutral. Backend manifests contain provider,
ABI, layout, hardware, and build constraints. `kernels.toml` is migrated into
this graph and deleted; it does not survive as a second registry.

Generated Rust and JSON projections are checked in. `cargo xtask operators
generate --check` is the drift gate. Normal Cargo builds do not need a TOML
parser or generator.

## Operator definition model

Inventory uses two independent axes:

- role: `semantic endpoint`, `composite stage/helper`, `provider ABI/control`,
  `test/probe-only`, or `dead candidate`;
- liveness: `default`, `feature-gated`, `shape-gated`, or `unreachable`.

This keeps NCCL lifecycle/control symbols out of the semantic-operator set and
keeps a gated FlashQLA implementation distinct from an unreachable legacy GDR
entry. Only archive symbols, link maps, provider markers/preflights, and runtime
counters can promote a dead candidate to proven dead.

### Semantic operator

```toml
[[operator]]
id = "attention.paged_decode"
contract = "paged HND attention for one decode step"
inputs = ["q", "k_pages", "v_pages", "page_table", "seq_lens"]
outputs = ["output"]
owner = "attention"
correctness_gate = "needle+same-config-floor"
```

The ID is stable. Renaming it requires a schema migration because benchmark,
artifact, telemetry, and policy records reference it.

### Implementation

```toml
[[implementation]]
operator = "attention.paged_decode"
id = "cuda.tilelang.paged_decode_hd128_bf16"
provider = "tilelang"
entry = "tilelang_batch_decode_paged_hd128_run_cuda"
source = "crates/cuda-kernels/tools/tilelang/batch_decode_paged_hd128.py"
abi = "cuda.paged_attn_v1"
artifact_component = "cuda.tilelang_aot"
phases = ["decode"]
dtypes = ["bf16"]
layouts = ["hnd"]
head_dims = [128]
sms = [70, 80, 86, 89, 90, 100, 120]
fallback = "cuda.native.paged_decode_hd128_bf16"
writes = [{ buffer = "output", disposition = "overwritten" }]
workspace = "batch * heads * head_dim * sizeof(f32)"
alignment = 16
stream_semantics = "caller_stream"
async_lifetime = "inputs+output live until recorded completion event"
capture_safe = true
reentrant = true
determinism = "same-config floor"
dynamic_smem_max = 49152
runtime_compat = "cuda12"
```

Every implementation declares:

- semantic operator ID and stable implementation ID;
- provider, source, exported entry, and ABI;
- phase, dtype, layout, exact dimensions or bands, SMs, and feature gates;
- artifact component and explicit fallback;
- every written device buffer and exact state disposition;
- workspace formula, pointer alignment, stream/event ownership, and async
  lifetime;
- graph-capture safety, reentrancy, determinism, dynamic shared-memory, and
  cooperative-launch requirements;
- runtime compatibility and required weight/KV layout IDs;
- correctness gate and benchmark family;
- support status: `optimized`, `coverage-only`, `experimental`, or `disabled`.

### Composite

```toml
[[composite]]
id = "cuda.qwen35.fused_qkv_quant_once"
replaces = ["linear.q", "linear.kv"]
implementation = "cuda.deepgemm.fused_qkv_fp8"
requires_layouts = ["qwen35.wq_wkv.fp8_block128.v1"]
provides_layouts = ["qwen35.qkv.split_view.v1"]
workspace = "rows * hidden * sizeof(fp8)"
correctness_gate = "layer-logits+needle"
```

Composites are explicit call structures, not inferred rewrites. Their boundary
layouts, state writes, workspace, and correctness gate are part of the contract.
The policy compares a composite against the complete unfused sequence on
end-to-end wall time.

### ABI

ABI signatures live once under `operators/abi/`. The registry compiler emits:

- C declarations and dispatch wrappers;
- Rust extern declarations and typed function aliases;
- symbol allowlists for artifact verification;
- ABI version and digest for manifests.

Manual duplicate declarations are deleted.

### Registry validation

Generation fails on:

- duplicate or unknown IDs;
- missing source, ABI, symbol, gate, artifact component, or fallback;
- fallback cycles;
- unsupported shape or SM claims;
- a stateful implementation without complete `writes`;
- a missing resource, compatibility, or layout contract;
- a composite whose replacement sequence or boundary layouts do not resolve;
- overlapping shape/topology predicates without explicit priority;
- an optimized cell without canonical evidence;
- a generated projection diff;
- a live runtime selector that is not generated from the graph;
- an exported semantic symbol absent from the graph.

Internal helper kernels attach to an implementation. They do not become fake
semantic operators.

## Generated build graph

The registry compiler emits a normalized DAG:

```text
source + ABI + generator + toolchain + flags + target
  -> generated source
  -> object/cubin
  -> component archive
  -> backend bundle
  -> product
```

Each node has:

- deterministic inputs and output names;
- target triple, backend, SM, and feature set;
- toolchain container digest;
- expected exported symbols;
- dependencies and parallelism limits;
- cacheability and reproducibility policy.

The explicit producer executes this graph. Cargo only locates and verifies the
resulting backend bundle.

## Artifact identity and layout

### Build ID

Each graph node's `build_id` hashes a canonical sorted document containing only
its transitive inputs:

- artifact schema and ABI versions;
- committed source tree hashes;
- dirty patch hash;
- `Cargo.lock`, build requirements, and vendored provider revisions;
- registry and generated build-graph hashes required by that node;
- optimal-policy hash only for policy or product nodes that embed it;
- target triple, backend, SM list, PTX policy, features, and profile;
- Rust, C/C++, CUDA, nvcc, TileLang, linker, and provider versions;
- runtime compatibility ID: libc/libstdc++ ABI, CPU ISA floor, CUDA runtime,
  NCCL/NVSHMEM/DeepEP/OFED ABI where linked;
- every behavior-changing environment variable;
- producer container digest.

Dirty state participates in local cache identity. Remote publication rejects a
non-empty dirty hash. Kernel-node IDs remain stable across a policy-only change.

The producer records actual file reads, environment reads, and compiler argv
during M0. Only inputs proven to affect a graph node enter its build ID; absolute
checkout/OUT_DIR paths are normalized, not hashed. This replaces the current
prebuilt manifest, which both omits load-bearing toolchain/provider inputs and
hashes non-behavioral path/env values. Artifact harvesting uses the graph's exact
output path, never the newest matching Cargo `OUT_DIR` by mtime.

Producer builds set `SOURCE_DATE_EPOCH`, deterministic archive mode, stable
debug-prefix maps, deterministic strip, and fixed locale/timezone. A release
candidate is rebuilt by two clean producer jobs; equal build IDs with unequal
bundle digests fail before publication. M0 measures which current provider
outputs need normalization rather than assuming bit reproducibility.

TileLang and runtime JIT cache keys obey the same identity. Legacy TileLang
artifacts without a complete hash are rejected; installed package/patch bytes,
CUDA/CUTLASS headers, host compiler, and full compiler argv are covered. JIT
cache reuse across toolchain drift is forbidden.

### OCI artifact

Use an OCI index as an ARLE descriptor catalog, not a standard platform
selector:

```text
ghcr.io/cklxx/arle-kernels:<build-id>
  index
    manifest digest + io.arle.backend=cuda + io.arle.sm=80
    manifest digest + io.arle.backend=cuda + io.arle.sm=86
    manifest digest + io.arle.backend=cuda + io.arle.sm=89
    manifest digest + io.arle.backend=cuda + io.arle.sm=90
    manifest digest + io.arle.backend=cuda + io.arle.sm=100
    manifest digest + io.arle.backend=cuda + io.arle.sm=120
    manifest digest + io.arle.backend=cuda + io.arle.sm=70
```

OCI's standard platform tuple cannot distinguish CUDA SMs: all CUDA entries are
still `linux/amd64`. Generic container platform resolution is therefore not a
consumer API. `cargo xtask kernels fetch` reads the index descriptors, requires
one exact match on signed `io.arle.backend`, `io.arle.sm`, target triple, and
build-ID annotations, then pulls and verifies that descriptor digest. Zero or
multiple matches fail. Products that need a fat binary request the explicit SM
set and verify every descriptor.

Each platform manifest contains reusable layers:

```text
manifest.json
libkernels_cuda.a
libtilelang_kernels_aot.a
provider libraries
optional sidecars
symbol-manifest.json
licenses/
```

Layers deduplicate unchanged providers and per-SM outputs. Consumers pull only
the required backend/SM layers.

The manifest records:

- build ID and OCI bundle digest;
- source commit and producer workflow;
- target, toolchain, minimum driver, features, and provider revisions;
- runtime compatibility ID plus the complete dynamic-library SONAME allowlist;
- registry, ABI, and policy hashes;
- exact files, sizes, SHA-256 values, and symbols;
- correctness and performance evidence references;
- support status for every included implementation.

Extra, missing, or mismatched files fail verification.

### Local CAS

Exact OCI layers unpack under:

```text
~/.cache/arle/artifacts/sha256/<digest>/
```

The cache is immutable. A checkout stores only a small build-ID-to-digest
reference. `cargo clean` never deletes it. Garbage collection removes only
unreferenced digests after a configurable age.

Downloads write to a per-digest temporary directory under a per-digest lock,
verify and `fsync`, then atomically rename into CAS. Active link/load operations
hold a lease so GC cannot remove their digest. Disk-full or interrupted writes
leave no visible cache entry.

### GitHub Releases

A tagged ARLE release contains:

- lane-specific final `arle` products;
- product manifest with exact kernel OCI digest;
- checksums, licenses, and GitHub artifact attestation;
- optional frozen offline kernel bundle for air-gapped installation.

Release assets are immutable. `stable` and `canary` are signed channel
manifests, not overwritten binaries.

## Product lanes

| Lane | Native target | Contract |
| --- | --- | --- |
| CUDA T1 | Linux x86_64, sm80/86/89/90 | one product with native cubins |
| CUDA Blackwell | Linux x86_64, sm100/120 | separate native product |
| CUDA Volta | Linux x86_64, sm70 | separate legacy product |
| Metal | macOS arm64 | MLX/Metal product |
| HIP | pinned Linux/ROCm targets | created after backend ratification |
| Vulkan | target-specific AIPC targets | created after backend ratification |

PTX may provide forward reachability. It is `coverage-only` until correctness
and performance run on the new SM.

For a new machine class, CI builds and validates its native layer once, pushes
it by build ID, and every later consumer reuses it. "Any machine compiles once"
means once per exact backend, ISA, toolchain, and operator graph, not one native
binary for incompatible platforms.

Before model load, the product verifies CPU ISA, libc ABI, driver/runtime,
required SONAMEs, backend features, and for distributed lanes the
NCCL/NVSHMEM/OFED compatibility envelope. Packaging defines whether each
dependency is static, bundled with RPATH, or host-provided; undeclared dynamic
dependencies fail the release gate.

The dependency gate merges `readelf/lddtree` DT_NEEDED closure with a cold
runtime `LD_DEBUG=libs` plus `strace(openat,execve)` closure. This is required for
`dlopen` libraries and compiler/tool subprocesses that do not appear in `ldd`.
Sidecars are audited independently from the main binary.

Model-dependent prepacked weights are not kernel artifacts. They use a separate
private/local identity `(model_hash, layout_id, codec_version, backend)` and are
never uploaded to the public kernel registry. Operator selection verifies the
required layout ID before choosing an implementation.

## Developer interface

Add one Rust `xtask`; delete permanent orchestration duplication from shell
scripts.

```text
cargo xtask operators check
cargo xtask operators inventory
cargo xtask kernels status
cargo xtask kernels fetch [--build-id ID]
cargo xtask kernels build --changed --sm native
cargo xtask kernels test --changed
cargo xtask kernels bench OPERATOR
cargo xtask kernels push --channel dev
cargo xtask artifacts verify DIGEST
cargo xtask artifacts gc
```

Shell scripts may remain as compatibility shims for one release, then delete.

Every command prints:

- clean/dirty source state;
- affected semantic operators and implementation IDs;
- target lane and SM;
- build ID and resolved OCI digest;
- local and remote cache hits;
- exact subprocesses before execution;
- output evidence and manifest paths.

## Developer inner loop

### 1. Classify

The generated build graph maps the git diff to affected nodes.

| Change | Required work |
| --- | --- |
| Rust above a backend | normal Cargo build; no kernel work |
| CUDA Rust caller only | Mac `cuda,no-cuda` check; link exact cached bundle |
| one TileLang source | one kernel for the selected SM |
| one native CUDA TU | that TU and dependent archive |
| ABI or semantic contract | all dependent implementations and consumers |
| provider/toolchain pin | all nodes emitted by that provider/toolchain |
| optimal policy only | generated policy and Rust relink; no kernel compile |

### 2. Resolve artifacts

`cargo xtask kernels fetch` computes the build ID, checks local CAS, then pulls
the exact OCI manifest. It never asks for `latest`.

If the exact bundle is absent:

- print the missing build ID and affected nodes;
- offer the explicit producer command;
- do not install a toolchain or start compilation implicitly.

### 3. Build the delta

Producer mode requires an approved, pinned container or remote builder. Local
GPU development pins one SM and uses `release-fast`, persistent CAS, and
`sccache`.

Structural invariants:

- no-op: zero kernel compiler subprocesses;
- Rust-only edit: zero kernel compiler subprocesses;
- one TileLang implementation on one SM: one generator invocation;
- one native TU edit: only that TU misses compiler cache;
- a load-bearing input change always changes build ID;
- output bytes are verified before entering CAS.

### 4. Correctness

Run the smallest end-to-end gate that detects the changed logic. Stateful
operators enumerate every mutated buffer before implementation. Numerical
operators compare the correct reference for their contract; MoE
nondeterminism does not force false byte identity.

### 5. Performance

The component harness loads candidate and reference implementations in the same
binary and process, using the same buffers, stream, input, and warmup. A
developer override selects implementations only inside this harness; production
policy remains generated.

Persist dirty exploratory records locally. Only clean-commit reruns can change
canonical policy.

### 6. End-to-end gate

Any runtime winner change runs the production SLO shape, decoded-case
correctness, and same-binary A/B. Component speed alone never flips a default.

## Canonical performance evidence

### Storage

```text
benchmarks/operators/
  schema.json
  runs/<yyyy-mm-dd>-<run-id>.json
  index.json
  optimal.json
```

`runs/` is truth. `index.json`, `optimal.json`, Markdown, and README content are
generated projections.

Large traces remain external and are referenced by URI and checksum.

### Required record

```text
schema, run_id, timestamp, source_commit, build_id, bundle_digest
operator_id, candidate_id, reference_id
model_id, model_revision, layer_index, position_kind, position
backend, GPU SKU, SM, physical SM count, firmware, driver, toolkit
runtime_compat_id, provider versions, compiler argv digest
CPU ISA, world_size, TP, EP, rank_role, interconnect, collective policy
realized collective backend and fallback state
GPU clocks, power limit, temperature, MIG mode, ECC state
phase, dtype, layout, exact dimensions, concurrency, stream policy
objective profile, scheduler/policy hash, benchmark protocol version
warmup, samples, duration, timing method
latency samples, median, p95, GB/s or TFLOPS, peak ratio
correctness method, tolerance, decoded cases, result
raw artifact URI and checksum
```

### Policy reducer

A record can select a winner only when:

1. source and artifact identities are exact and clean;
2. candidate and reference pass correctness;
3. model position, shape, dtype, SM, binary, stream, and scheduler match;
4. duration and repetitions meet `bench-and-trace-spec.md`;
5. confidence separates the winner, otherwise the simpler implementation wins
   a declared tie;
6. the SLO-shaped end-to-end guard passes.

Each policy cell names an objective profile. Correctness and memory capacity are
hard constraints; the profile then orders TTFT, ITL, throughput, and artifact or
workspace cost lexicographically. The reducer never collapses unlike objectives
into an undocumented scalar score. Separate service profiles may select
different winners from the same evidence.

Shape bands are ordered, non-overlapping predicates with an explicit generic
fallback. Generation rejects overlaps, unreachable cells, uncovered supported
shapes, and a configured maximum cell count. Distributed predicates apply only
to implementations whose contract depends on topology.

The evidence compatibility key contains implementation bytes, boundary layout,
runtime compatibility, scheduler/policy, hardware/topology class, and benchmark
protocol. A change to any member marks only its dependent cells stale and queues
their revalidation; stale evidence remains historical but cannot select a
winner.

Historical Markdown/prose imports default to `INSUFFICIENT`. They become policy
evidence only when the tested artifact identity, exact cell, correctness record,
and raw measurement can be reconstructed.

The reducer emits `WINNER`, `TIE`, `INSUFFICIENT`, or `KILL`. It never
extrapolates a winner across SMs, dtypes, or shape bands.

## Runtime selection

`optimal.json` generates a static, allocation-free table keyed by:

```text
backend + hardware ISA + operator/composite + phase + dtype + layout
+ shape band + relevant execution/topology class
```

Fallback is fixed:

1. exact correctness-qualified winner;
2. registered same-hardware generic implementation;
3. registered `coverage-only` implementation;
4. `NOT_SUPPORTED` with the missing cell.

There is no silent cross-SM policy reuse.

Policy stays compiled into the product. A policy change relinks the product but
does not rebuild kernel nodes. This keeps startup and rollback simple; a remote
runtime policy service is deliberately out of scope.

At load, each backend records:

- selected implementation and reason;
- registry and policy hashes;
- build ID and OCI bundle digest.

`/v1/stats` exposes hit counts, fallback counts, and coverage-only counts. Logs
emit selection once, never per request.

Each operator family migrates atomically: generated selector in, manual selector
out in the same commit.

## CI

### Pull request

1. Generate and check registry, ABI, build graph, Rust sources, and inventory.
2. Classify affected operators and artifact nodes.
3. Run Mac/Linux no-toolchain Rust checks.
4. Build the minimal representative GPU delta when a kernel changes.
5. Run component correctness.
6. Run same-process A/B for implementation or policy changes.
7. Upload temporary OCI layout as a 14-day Actions Artifact.
8. Require canonical evidence or an explicit `pending-remote` record.
9. Never publish PR bytes to stable channels.

Fork and untrusted PR kernels run only on isolated ephemeral workers with no
registry, signing, or release credentials. Self-hosted production GPU runners
build protected clean commits only. Main always rebuilds publishable bytes; it
never promotes PR artifacts.

### Main

1. Compute the clean build ID.
2. Reuse an existing exact OCI digest or build missing graph nodes.
3. Build affected supported SM layers in pinned producer images.
4. Singleflight each build ID; publish through a temporary reference only after
   every layer verifies, then atomically create the immutable reference.
5. Verify symbols, ABI, checksums, provenance, and two-producer reproducibility.
6. Run a cold consumer with no Python kernel stack.
7. Run real-hardware correctness and representative SLO gates.
8. Push immutable OCI manifests.
9. Move `canary` only after required gates pass.

### Stable promotion

After the hardware matrix and soak, stable promotion moves a signed channel
manifest to the exact tested OCI digest. It never rebuilds bytes.

### Product release

Build final lane-specific products from the stable digest. Audit the process
tree from install through first correct request. Then attach immutable products,
offline bundles, checksums, licenses, and attestations to GitHub Release.

## Zero-local-compilation contract

An end-user install and first request spawn none of:

```text
cargo rustc cc c++ nvcc python tilelang cmake ninja
```

The audit also detects provider-specific compiler subprocesses. Driver loading
is allowed. PTX driver JIT is reported and marks the lane coverage-only.

Normal source developers who only change Rust also spawn no kernel compiler.

## Runtime JIT closure

DeepGEMM is the known violation. The current first cache lifetime costs about
3.7 seconds for documented Qwen shapes, and dense prefill warms additional
shapes at model load.

Close it by evidence:

1. record every reached `(operator,M,N,K,layout,dtype,SM)` and compile wall time;
2. separate bounded production shapes from genuinely dynamic shapes;
3. test generated object relocatability across identical producer/driver lanes;
4. package relocatable objects as OCI layers;
5. otherwise AOT bounded production shapes;
6. label remaining dynamic lanes `compile-on-first-load` and keep them out of
   the strict zero-compile product.

Arbitrary shapes, zero JIT, and globally optimal code cannot all be promised.
The strict product chooses bounded supported shapes, zero JIT, and measured
optimal implementations.

## Failure and rollback

| Failure | Behavior |
| --- | --- |
| exact build ID absent | print producer command; no silent compile |
| corrupt local layer | delete only that digest and refetch |
| ABI or symbol mismatch | fail before link/load |
| same build ID, new digest | fail reproducibility gate |
| unsupported hardware | coverage fallback or explicit `NOT_SUPPORTED` |
| correctness regression | remove candidate eligibility |
| performance regression | retain prior winner and write errors entry |
| bad canary | atomically restore previous channel digest |
| bad product | release prior verified OCI digest as a new product version |
| compromised producer | publish signed denylist, roll back channel, quarantine digest |

Immutable OCI manifests and release assets are never overwritten.

Revocation is separate signed trust metadata. It contains a monotonic sequence,
expiry, revoked build IDs and digests, reason, and replacement digest. Consumers
refresh and verify it before accepting a remote or local-CAS hit. A revoked
local digest is quarantined and cannot run. Air-gapped bundles carry a signed,
expiring trust snapshot; after expiry they require a newer snapshot or an
explicit unsafe override.

## Security and provenance

- Producer images, actions, compilers, and providers are pinned by digest.
- Publish permission exists only in protected main/release workflows.
- Offline root keys sign rotating channel and revocation metadata; producer
  credentials cannot remove a denylist entry.
- Every OCI manifest and product has checksums, source commit, workflow run,
  builder identity, SBOM, licenses, and artifact attestation.
- Consumers verify target, minimum driver, build ID, OCI digest, file allowlist,
  checksums, ABI, symbols, trust-metadata sequence, expiry, and revocation status
  before consulting or executing local CAS content.
- Benchmark records can cite only the digest of tested bytes.
- Sidecars follow the same rules as archives.

SBOM presence is not redistribution permission. Each packaged component carries
an SPDX ID, source/revision, notice files, and `redistributable=true|false` from
an reviewed allowlist. Product packaging fails on unknown or incompatible
licenses. macOS products are code-signed and notarized; Linux products record
the complete bundled and host-provided dynamic dependency set.

## Schema evolution

Operator, ABI, benchmark, policy, and artifact schemas each use an independent
integer version. Readers declare a supported inclusive range and fail closed on
newer incompatible input. A migration command rewrites old canonical data;
golden upgrade and downgrade-rejection fixtures gate every version bump. Product
rollback verifies that its embedded reader can consume the pinned registry,
policy, and artifact manifests.

## Migration DAG

```text
M0 baseline and inventory
  |
  +-> M1 operator graph and generator
        |
        +-> M2 explicit artifact producer -> M3 OCI/CAS dev and CI
        |                                     |
        |                                     +-> M6 product releases
        |
        +-> M4 canonical evidence -> M5 generated selection

M5 + M6 -> M7 runtime JIT closure
```

M1 follows M0 and emits the build DAG and stable IDs. M2 and M4 then proceed
independently. M3 needs M2's explicit build IDs. M5 needs M4's legality and
evidence. M6 needs verified OCI consumption. M7 starts only when telemetry
proves exact selection and artifact identity.

## Migration tranches

### M0: Measure and classify

M0 changes no selector, kernel, or default. M0c adds instrumentation and dev-tool
parsing only; because it touches runtime observability, it follows the normal
runtime verification and bench-entry rule.

#### M0a: Source, archive, and liveness truth

Deliver:

- role+liveness inventory for CUDA FFI, checked-in/vendor TUs, generated
  TileLang rows, DeepGEMM, FlashMLA/FA3, cuBLAS/cuBLASLt, NCCL, and both DeepEP
  products;
- Rust reference graph and explicit composite/call-structure boundaries;
- per-product `nm --defined-only` plus linker-map evidence mapping
  symbol -> object/provider -> build ID;
- marker/preflight distinction between real and stub providers;
- one-shot runtime counters across Qwen dense/hybrid, DSv4/GLM, TP/EP, and
  feature-gated lanes.

Exit:

- every exported symbol has role, liveness, owner, artifact node, and evidence;
- a zero-reference entry remains `dead candidate` until archive/link/runtime
  evidence agrees;
- both DeepEP products have an explicit KEEP/MERGE/DELETE verdict;
- every selectable single operator or composite has resource, layout, and
  compatibility ownership.

#### M0b: Build and artifact process truth

On a clean sm90 producer, measure at least:

```text
cold source/full cache off
warm exact no-op
Rust-only edit
one native CUDA TU
one TileLang source
ABI/operator graph change
prebuilt link-only
DeepEP intranode
DeepEP with NVSHMEM
T1 fat product
sm70 product
Blackwell product
```

The pinned M0 producer image includes `strace`, `lddtree`, `readelf`,
`diffoscope`, `cuobjdump`, `nm`, and `ar`. Missing audit tools fail preflight;
the current H20 image is known to lack `strace`, `lddtree`, and `diffoscope`.

For each case capture `time -v`, `strace -f -e process`, sccache statistics,
graph-node misses, wall/CPU/max-RSS, compiler subprocess counts, and output
bytes. Capture static and runtime dependency closure with
`readelf/lddtree` plus `LD_DEBUG=libs` and `strace(openat,execve)`.

Build the same release candidate in two clean checkouts at different absolute
paths. Compare archives/products with SHA-256 and `diffoscope`; scan strings for
checkout, OUT_DIR, temporary, CUDA, and provider absolute paths.

Exit:

- every `execve` and output file has one graph owner;
- current cache-key omissions and false inputs map to new build-ID fields or
  producer normalization;
- same build ID differences are fully attributed;
- link-only consumption launches no GPU compiler;
- actual output paths replace mtime-based OUT_DIR harvesting;
- dynamic, `dlopen`, sidecar, and runtime compiler dependencies all have owners.

#### M0c: Evidence plumbing trust

Define a host-only `OperatorDispatchStats` snapshot through the existing seam:

```text
infer-seam trait/default
  -> infer-core snapshot
  -> infer-server execution aggregation
  -> multiprocess WireStats
  -> HTTP schema and /v1/stats
```

Fix the GuideLLM service-trace consumer to parse the current JSON schema instead
of legacy `key=value` fields. A self-test sends a known request and requires raw
`/v1/stats`, trace JSONL, summary output, and an independent launch counter to
report the same delta.

Exit:

- no backend type leaks above the seam;
- selector/artifact/composite hit and fallback counters survive multiprocess
  aggregation;
- a deliberately selected implementation is proven by counter delta and one
  independent NVTX/nsys or launch-count source;
- canonical benchmark collection is blocked when the self-test fails.

#### M0d: Real GPU and JIT measurement

Required matrix:

- 1xH20: empty DeepGEMM JIT cache; model load/warmup; dense rows
  `{1,2,4,8,16,17,32}`; prefill remainder and speculative-verify shapes;
- 8xH20 TP8/EP8: production collective and DeepEP lanes, `nvidia-smi topo -m`,
  versions, realized collective backend/fallback, `comm_bench`, and resident
  composite A/B;
- at least one pre-Hopper lane (A100 sm80 or V100 sm70): prove fail-closed
  DeepGEMM and the real fallback, not only build coverage;
- one SLO-shaped run whose stats engagement agrees with nsys/NVTX launch count.

Cold JIT tracing records every compiler `execve`, source/digest, generated
layout, physical SM count, `DG_NUM_SMS`, compile wall, produced file, and whether
the first request still compiles after warmup.

Exit:

- every production JIT shape and compiler launch is enumerated;
- exact measured policy cells are separated from unmeasured fallbacks;
- hardware class includes GPU SKU/physical SM count and realized collective
  state;
- no prose-only historical result becomes a winner;
- no build-time or performance target is guessed beyond measured evidence.

### M1: Replace registries with the operator graph

Files:

- new `operators/**`;
- new `xtask/src/operators.rs`;
- generated Rust/JSON/Markdown projections;
- `crates/cuda-kernels/build.rs`;
- existing FFI and runtime selector modules.

Work:

- import current semantics, explicit composites, TileLang, native CUDA,
  provider, and Metal implementations;
- generate ABI, FFI, inventory, and build graph;
- model provider/TU/SM objects before composing fat-product unions; current
  archives are not assumed to be independently reusable per-SM layers;
- validate resource, layout, shape-band, and compatibility contracts;
- switch every consumer in the tranche;
- delete `crates/cuda-kernels/kernels.toml` and duplicate hand tables.

Exit:

- one logical graph owns all imported operators;
- generated output is deterministic;
- current runtime behavior is byte-identical;
- no parallel old/new registry remains.

### M2: Move kernel compilation out of Cargo

Files:

- new `xtask/src/kernels.rs`;
- slim `crates/cuda-kernels/build.rs`;
- slim `crates/deepep-sys/build.rs`;
- CUDA/TileLang/provider build helpers moved from `build.rs`;
- existing fast-build and artifact scripts removed after parity.

Work:

- execute the generated build DAG explicitly;
- move both the external DeepEP sidecar and in-process DeepEP/NVSHMEM archive
  compilation out of Cargo after M0a decides their ownership;
- use pinned producer images and per-node cache keys;
- normalize timestamps, paths, archives, and strip output for reproducibility;
- make Cargo only verify and link an exact local artifact;
- preserve per-SM hard failures and symbol validation.
- discover sidecars through the product manifest or executable-relative path;
  never bake an absolute Cargo OUT_DIR into the product.

Exit:

- normal Cargo never launches a GPU compiler;
- no-op and Rust-only builds launch zero kernel compiler processes;
- one TileLang edit on one SM launches one generator;
- two clean producer jobs agree on the release-candidate digest;
- old implicit compiler path is deleted.

### M3: OCI, local CAS, and developer flow

Files:

- new `xtask/src/artifacts.rs`;
- GHCR producer/consumer workflows;
- `.cargo/config.toml` xtask alias;
- `scripts/pod.sh` reduced to remote transport/process control;
- environment and contributor docs.

Work:

- push/pull exact OCI manifests;
- implement locked atomic CAS writes, active leases, and garbage collection;
- implement signed channel/revocation refresh, expiry, and CAS quarantine;
- provide status/fetch/build/test/push/verify commands;
- use Actions Artifact only for PR temporaries.

Exit:

- historical checkout fetches its exact digest;
- corrupt and missing fixtures fail clearly;
- revoked cached content is rejected before execution;
- dirty builds stay local;
- zero-Python cold consumer links and runs.

### M4: Canonical benchmark evidence

Files:

- new `benchmarks/operators/**`;
- new `xtask/src/benchmarks.rs`;
- existing component harnesses adapted to stable operator IDs.

Work:

- persist JSON from same-process candidate/reference A/B;
- validate same-site keys and decoded-case correctness;
- record environment/topology health and invalidate evidence on compatibility-key
  drift;
- generate bounded, non-overlapping shape/topology predicates;
- generate index, policy candidates, and docs;
- import one existing crossover without changing its verdict.

The first pilot is `qwen.fp8_dense_projection`, using the existing same-process
small-M probe. Import only its exact H20/model/shape cells. Add candidate versus
reference numerical comparison before it can become correctness-qualified; the
historical global `M >= 2` prose remains `INSUFFICIENT` outside measured cells.

Exit:

- WINNER/TIE/INSUFFICIENT/KILL fixtures pass;
- dirty or incomparable runs cannot select a winner;
- stale evidence and overlapping/uncovered predicates cannot select a winner;
- deleting generated Markdown loses no evidence.

### M5: Generated runtime selection

Files:

- generated `optimal.json` and Rust tables;
- backend operator dispatch modules;
- `/v1/stats` projection.

Work:

- migrate one complete family per commit;
- migrate its explicit composites with their complete unfused references;
- delete its manual selector in the same commit;
- export hit/fallback/coverage counts;
- run correctness and same-binary SLO A/B.

Exit per family:

- one selector remains;
- every selected cell has exact evidence or declared fallback status;
- topology-sensitive choices include the relevant execution class;
- stats prove the intended implementation ran;
- runtime change has its wins/errors entry.

### M6: Formal products and GitHub Releases

Files:

- release workflows and packaging;
- installer;
- install, support-matrix, and release-checklist docs.

Work:

- build lane-specific products from stable OCI digests;
- embed source, build, bundle, registry, ABI, and policy identities;
- verify the runtime compatibility envelope and dynamic dependency allowlist;
- enforce redistribution licenses and platform signing/notarization;
- audit clean-host install and first request;
- attach optional offline kernel bundles.

Exit:

- CUDA T1, Blackwell, sm70, and Metal products install on clean hosts;
- no compiler subprocess appears;
- online products enforce current revocations and offline products enforce their
  signed trust-snapshot expiry;
- unsupported hardware fails before model load;
- rollback uses previously tested bytes.

### M7: Eliminate hidden JIT

Files:

- DeepGEMM bridge and callers;
- operator graph;
- artifact producer;
- OCI manifest and product policy.

Work:

- measure actual compile shapes;
- license or kill relocatable packaging;
- AOT bounded production shapes;
- isolate explicitly dynamic products.

Exit:

- strict products contain no hidden JIT;
- request TTFT includes no compilation;
- AOT/package paths retain correctness and the measured winner.

## Verification matrix

| Gate | M1 | M2/M3 | M4/M5 | M6/M7 |
| --- | --- | --- | --- | --- |
| generator/schema fixtures | required | required | required | required |
| Mac CUDA no-toolchain check | required | required | required | required |
| generated drift check | required | required | required | required |
| ABI and symbol verification | required | required | required | required |
| zero-Python cold consume | no | required | guard | required |
| target-GPU correctness | behavior identity | required | required | required |
| same-process component A/B | no | guard | required | required |
| SLO-shaped end-to-end A/B | no | guard | required for policy | required |
| clean-host process audit | no | no | no | required |

Runtime changes create dated wins/errors entries. Schema, generated inventory,
artifact tooling, and docs are benchmark-exempt until they change generated
kernel bytes or runtime selection; commit bodies state the exemption.

## Completion criteria

The platform is complete when:

1. Every semantic operator, implementation, ABI, legal cell, fallback, evidence,
   selected winner, and containing OCI digest is mechanically queryable.
2. Explicit composites are compared against complete unfused sequences; resource,
   layout, compatibility, and topology contracts are machine-validated.
3. `kernels.toml`, duplicate FFI tables, mutable latest bundles, hand-written
   winner policy, and implicit kernel compilation are deleted.
4. Rust-only builds launch zero kernel compilers.
5. One implementation edit rebuilds only its affected graph nodes.
6. Main publishes reproducible immutable OCI artifacts; historical commits fetch exact
   digests.
7. Runtime selection comes only from current, correctness-qualified canonical
   JSON with bounded non-overlapping predicates.
8. `/v1/stats` proves the selected implementation and artifact identity.
9. Formal products pass runtime compatibility, license, signature, and
   zero-compiler gates before serving first output.
10. Rollback moves signed pointers to previously tested bytes.
11. Strict products contain no hidden runtime JIT.

## Self-review

- Operator counts are inventory, not proof of liveness. M0 must trace callers
  before deleting or registering symbols.
- OCI solves content addressing and layer reuse; GitHub Release remains the
  simpler user-facing product surface. Using Release as the high-frequency CAS
  would recreate mutable tags, retention pressure, and poor layer reuse.
- Moving compilation out of `build.rs` is a one-time structural cost but removes
  the permanent surprise-build boundary. Keeping both paths would violate the
  no-half-state rule.
- Build speed targets come after M0 measurement. Initial gates use exact
  subprocess counts because they isolate the mechanism.
- Static embedded policy is intentional: separate remote policy distribution
  would add another trust and rollback surface without reducing kernel builds.
- Explicit composites cover proven fusion/call-form wins without committing to
  a general graph optimizer.
- DeepGEMM object relocatability is unknown. M7 measures it; the plan does not
  assume packaging works.
- PTX is not native performance evidence. New SMs remain coverage-only until a
  real machine supplies correctness and wall-clock data.
