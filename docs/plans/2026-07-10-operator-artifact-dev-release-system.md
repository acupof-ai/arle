# Operator clarity, evidence, and exact release artifacts

> Status: Active - scope reduced after necessity review on 2026-07-10. This
> replaces the earlier global operator-platform/OCI-first design. No runtime
> selector or default changes before its family-specific correctness and SLO
> gates pass.

## Verdict

Build speed is not the primary problem.

Measured current behavior:

- exact hot H20 Cargo build: 0.408 seconds, zero `Compiling` lines;
- hot Mac CUDA/no-CUDA check: 0.56 seconds;
- common release increments: 59-113 seconds;
- cached TileLang regeneration: 53 seconds versus 251 seconds forced;
- current kernel Release bundle: 7.5 MB;
- v0.2.1 products: 9-12 MB CUDA and 42 MB Metal.

These numbers do not license an OCI platform, a remote build service, or moving
all GPU compilation out of Cargo. They do license fixing three narrower problems:

1. runtime-selectable operators and composites are difficult to inspect;
2. performance decisions are prose and scattered conditionals rather than
   correctness-qualified machine-readable evidence;
3. release builds consume mutable `latest` kernel artifacts and current cache
   identities omit load-bearing inputs.

The plan is one complete vertical slice first, then expand only when repeated
evidence proves the abstraction removes more complexity than it adds.

Detailed M0 evidence:
[`../reviews/2026-07-10-operator-platform-m0-readonly-audit.md`](../reviews/2026-07-10-operator-platform-m0-readonly-audit.md).

## Goal

- Make the selected implementation, fallback, evidence, and artifact identity
  queryable for migrated operator families.
- Persist same-site component and end-to-end evidence as canonical JSON.
- Prove runtime engagement through stats, not source inspection or an
  `enabled` log.
- Publish exact immutable GitHub Release bundles and consume them by ID.
- Keep current hot/no-op build speed and existing source-build capability.
- Ensure formal products do not compile inside request execution.

## Non-goals

- No OCI/GHCR artifact protocol.
- No new remote build farm.
- No general graph optimizer.
- No runtime remote policy service.
- No all-at-once rewrite of 351 declared CUDA FFI entries.
- No immediate migration of every backend or DSv4 path.
- No claim that 59-113 second builds are kernel-bound before process tracing.
- No automatic default flip without human review of canonical evidence.

## Current truth

### Inventory

The inspected CUDA surface contains:

```text
312 handwritten FFI declarations
39 generated TileLang FFI declarations
351 total declared Rust FFI entries for the inspected build shape
61 checked-in CUDA TUs
48 TileLang registry rows
```

TileLang static reachability:

```text
8 default attention rows
6 autograd GDR rows, opt-in and shape-gated
3 FlashQLA rows, build/runtime/shape-gated
31 rows with no current runtime caller
```

Static no-caller status is not deletion evidence. Each family still needs
archive/link/provider/runtime proof.

### Artifacts

The current prerelease `kernel-artifacts` has two 7,507,544-byte assets: one
hash-named bundle and one mutable `latest` copy. The hash-named mechanism is
already sufficient in scale; its identity and consumption contract need
hardening.

Current formal product assets are also small enough for GitHub Releases:

```text
CUDA T1:       11.2 MB
CUDA sm70:      9.4 MB
CUDA Blackwell: 10.0 MB
Metal arm64:   42.5 MB
```

GitHub Release remains the remote store until measured scale proves it
insufficient.

### Dispatch

Dispatch is currently family-local:

- Qwen FP8 dense uses hard-coded row floors and shape/preflight checks.
- Qwen MoE combines load-time weight format, row count, provider, and composite
  call form.
- DSv4 combines stateful composites, FlashMLA, DeepGEMM JIT, TP/EP, and
  collective fallback.

This is why the first family is Qwen FP8 dense, not DSv4.

## Necessity decisions

| Item | Decision | Reason |
| --- | --- | --- |
| Canonical benchmark JSON | KEEP now | Performance truth cannot remain prose |
| Explicit named composites | KEEP now | Real winners are often call forms, not one kernel |
| Family-local generated selector | KEEP now | Removes one scattered manual policy without global rewrite |
| Host-only hit/fallback stats | KEEP now | Required to prove engagement |
| Exact GitHub Release bundle | KEEP now | Current size/lanes fit Release easily |
| Complete cache/build identity | KEEP now | Existing omissions can reuse stale bytes |
| Product dependency/compile audit | KEEP now | `ldd` misses dlopen and runtime compiler paths |
| Minimal existing local cache | KEEP | Already solves `cargo clean` and checkout reuse |
| Full global operator graph | DEFER | Needs three different family proofs |
| Move GPU compilation out of Cargo | DEFER | Current timing does not attribute the wall to GPU compilation |
| OCI/per-SM layers/custom resolver | DELETE now | No measured scale or consumer need |
| Signed channels/revocation/TUF | DELETE now | GitHub Release plus checksums/attestation is sufficient |
| General CAS/GC/lease platform | DELETE now | Current artifacts are tiny |
| General graph optimizer | DELETE | Explicit composites cover the real requirement |
| Runtime remote policy | DELETE | Static product policy is simpler and sufficient |
| Every-release dual producer | DEFER | Run once to diagnose determinism; repeat only on trigger |
| Global strict zero model-load JIT | DEFER | Cold trace first; request-time compilation remains forbidden |

## Minimal data model

Use one `operators/registry.toml` for migrated families. It has three logical
record types, not a full workspace inventory database.

### Semantic or composite

```toml
[[semantic]]
id = "qwen.fp8_dense_projection"
kind = "composite"
inputs = ["activation", "fp8_weight", "block_scales"]
outputs = ["projection"]
reference = "cuda.qwen.fp8_gemv"
correctness_gate = "numeric+e2e"
```

### Implementation

```toml
[[implementation]]
id = "cuda.qwen.fp8_pack_deepgemm"
semantic = "qwen.fp8_dense_projection"
provider = "deepgemm"
source = "crates/infer-cuda/src/ops/quant_linear.rs"
legality = "sm90 && exact_shape_cell && preflight"
fallback = "cuda.qwen.fp8_gemv"
```

### ARLE-owned ABI

```toml
[[abi]]
id = "cuda.qwen_fp8_dense_v1"
owner = "arle"
```

The first slice references existing ABI rather than regenerating all FFI.
External NCCL/CUDA/cuBLAS/NVSHMEM/DeepEP ABI remains owned by upstream headers
or bindings.

Role and reachability are separate:

```text
role: semantic | composite stage/helper | provider control | test/probe
reachability: default | feature-gated | shape-gated | unreachable
```

Caller graphs, symbol/object maps, liveness, build DAG, and Markdown are
generated evidence, not additional handwritten truth.

`crates/cuda-kernels/kernels.toml` remains the TileLang build authority until a
later TileLang family migrates atomically. There is no long-lived mirror or
adapter.

## Canonical evidence

Store:

```text
benchmarks/operators/schema.json
benchmarks/operators/runs/<date>-<run-id>.json
benchmarks/operators/index.json
benchmarks/operators/optimal.json
```

Only `runs/*.json` is truth. Other files are generated.

Required fields:

```text
source commit, dirty state, product/bundle identity
semantic/composite ID, candidate, complete reference sequence
model revision, layer/position, exact M/N/K, dtype/layout
GPU SKU, SM, physical SM count, driver/toolkit/provider
world size/topology only when the family depends on them
warmup, samples, timing method, raw samples
numeric correctness and independent end-to-end gate
latency/throughput result and raw artifact checksum
```

Historical Markdown defaults to `INSUFFICIENT`. It becomes policy evidence only
when exact artifact, cell, correctness, and raw measurements are recoverable.

The reducer initially supports exact cells plus one generic fallback. It does
not implement an arbitrary predicate language.

Each policy cell names an objective profile. Correctness and memory are hard
constraints; the profile then orders the relevant SLO metrics. Unlike metrics
are never merged into an undocumented score.

## Runtime selector

The generated selector is family-local, checked in, and allocation-free. A
policy-only change relinks Rust but compiles no kernel.

Fallback:

1. exact correctness-qualified cell;
2. the existing family fallback;
3. clear unsupported error only where no correct fallback exists.

Unknown shapes do not inherit a measured winner.

Production policy stays embedded in the product. No remote policy service.

## Engagement proof

Add a host-only snapshot through the existing stats path:

```text
infer-seam
  -> infer-core
  -> infer-server execution
  -> multiprocess WireStats
  -> /v1/stats
```

First-slice stats contain only:

```text
policy hash
product/bundle identity
implementation ID -> hit count
fallback count
```

Do not use exact shape as a metrics label.

Keep `scripts/bench_throughput.py` aligned with the current JSON stats schema. A
self-test requires raw stats, trace JSONL, rendered summary, and an independent
launch counter to agree.

## Build policy

Keep Cargo GPU builds for now.

The current no-op is 0.408 seconds. Existing 59-113 second release increments
include Rust/LTO/relink work and have not been process-attributed. Moving all
provider compilation out of Cargo before M0 process traces would solve an
unproven problem.

Improve now:

- replace partial/FNV cache identities with SHA-256 over actual
  output-affecting source, ABI, provider/toolchain bytes, and compiler argv;
- do not hash absolute checkout paths or non-behavioral environment variables;
- reject legacy artifacts without complete identity;
- use per-ID lock, verified temporary output, and atomic rename;
- harvest exact declared outputs, never newest OUT_DIR by mtime;
- make DeepGEMM JIT cache identity include generated layout, GPU SKU/physical SM
  count, `DG_NUM_SMS`, sources, compiler, headers, and flags.

Move a provider compiler out of Cargo only when one measured trigger fires:

- a Rust-only edit launches that GPU compiler;
- kernel compilation is more than 20 seconds or 30% of a common incremental
  build;
- missing Python/nvcc/provider toolchain blocks a supported source developer;
- cache/toolchain failures exceed 5% over 20 builds.

Extract only the provider that crosses the trigger. Do not create a general
build farm.

## Exact GitHub Release artifacts

Publish:

```text
arle-kernels-<lane>-<build-id>.tar.gz
manifest.json
SHA256SUMS
GitHub build attestation
```

`build-id` covers only build inputs. The manifest records output SHA-256,
source commit, target lane, toolchain/provider versions, ABI/symbol allowlist,
and correctness status.

Rules:

- correctness status is the closed enum `not-run | passed | failed`;
- local packing may record `not-run`; publish and formal fetch require `passed`;
- passed evidence is immutable JSON bound to bundle ID, GPU-tested commit, and tested candidate archive SHA-256, with its own SHA-256 in the manifest;
- kernel build/pack success never publishes by itself; only the qualified workflow path publishes;
- a qualified bundle may serve a later descendant commit only when the exact bundle ID is unchanged;
- formal release checks only `release-blockers.json`, never historical docs;
- hash-named assets are immutable;
- CI and product release fetch only the exact expected ID;
- mutable `latest` is removed after one compatibility release;
- PR artifacts remain 14-day Actions Artifacts and are never promoted;
- official products embed or reference their exact kernel bundle ID.

No OCI, custom SM resolver, signed channel service, or remote CAS.

## Product compile contract

Formal products must pass:

- clean install and first correct request;
- DT_NEEDED plus declared `dlopen` dependency audit;
- no compiler subprocess during request execution;
- correct provider symbol/preflight markers;
- exact bundle and policy identity reporting.

DeepGEMM model-load JIT remains a measured question. First trace cold load,
warmup, prefill remainder, speculative verify, and the first request. If warmup
covers every production shape and startup stays inside its SLO, package/AOT is
not yet necessary. If a request compiles, or startup misses its SLO, package the
bounded shapes and use a correct generic fallback for unknown cells.

## Execution plan

### P0: Trust the evidence path

Files:

- `crates/infer-seam/src/lib.rs`
- `crates/infer-core/src/lib.rs`
- `crates/infer-server/src/execution.rs`
- `crates/infer-server/src/multiproc_relay.rs`
- `crates/infer-server/src/schema.rs`
- `scripts/bench_throughput.py`

Work:

- add minimal host-only operator dispatch stats;
- fix JSON stats parsing;
- add raw stats -> trace -> summary consistency test;
- cross-check one forced implementation against an independent launch count.

Exit:

- no backend type above the seam;
- multiprocess aggregation preserves counts;
- canonical evidence collection fails closed when engagement proof fails.

### P1: Qwen FP8 dense vertical slice

Files:

- new `operators/registry.toml`
- new `benchmarks/operators/schema.json`
- new `benchmarks/operators/runs/<run>.json`
- new small reducer/generator under `scripts/`
- `crates/infer-cuda/examples/fp8_smallm_gemm_probe.rs`
- `crates/infer-cuda/src/ops/quant_linear.rs`
- one generated family selector

Work:

- add numerical candidate/reference comparison to the existing same-process
  probe;
- persist exact H20/Qwen3.6 model/shape cells;
- keep historical M=2 prose `INSUFFICIENT` outside exact cells;
- generate the family selector and delete the replaced manual branch in the
  same commit;
- run same-binary E2E A/B and SLO gate.

Exit:

- M=1 fallback is unchanged;
- exact measured cells engage and pass correctness;
- unknown cells use the old correct fallback;
- stats agree with independent launch evidence;
- policy-only regeneration launches no kernel compiler.

### P2: Operator inventory and deletion proof

Files:

- new `scripts/operator_inventory.py`
- generated `docs/generated/operator-inventory.md`
- current FFI/provider/TileLang sources, changed only per proven family

Work:

- generate role+reachability inventory from source and registry;
- collect per-lane `nm`, link-map, marker/preflight, and runtime-counter evidence;
- delete dead entries one family at a time;
- remove declaration, wrapper/export, build row, implementation, test, and docs
  together.

Exit:

- no symbol is deleted from zero-reference evidence alone;
- GDR six rows remain registered as opt-in/shape-gated;
- no hard-coded generated-symbol count remains;
- every survivor has an owner and support state.

### P3: Exact cache and Release identity

Files:

- `crates/cuda-kernels/build.rs`
- `scripts/cuda_prebuilt_manifest.sh`
- `scripts/kernel_artifacts.sh`
- `.github/workflows/kernels-publish.yml`
- `.github/workflows/release.yml`

Work:

- complete cache/build identity and atomic writes;
- remove mtime-based artifact harvesting;
- publish and fetch exact immutable GitHub Release assets;
- stop release/CI consumption of `latest`;
- run one two-checkout reproducibility diagnosis to normalize timestamps and
  paths, without making it an every-release gate.

Exit:

- historical checkout fetches its exact bundle;
- stale/partial/legacy cache entries fail clearly;
- product reports exact source, bundle, and policy IDs;
- current 0.408-second no-op does not regress beyond noise.

### P4: Prove the schema on three archetypes

After P1, migrate:

1. Qwen MoE decode composite: workspace and multi-launch call form;
2. one collective family: world-size/topology/rank correctness.

Expand to a global operator graph only when all are true:

- leaf, composite, and topology-sensitive families fit without
  family-specific schema escapes;
- validator rejects dirty/mismatched evidence, overlap, hole, and missing E2E
  guard;
- selectors preserve fallback behavior and add no allocation/device sync;
- repeated clean runs do not flip winners;
- generator/validator code is smaller than the hand-maintained policy it
  deletes.

Only then consider migrating TileLang ABI/build rows and deleting
`kernels.toml` atomically.

### P5: Conditional build/JIT work

Run M0 process traces with a pinned image containing `strace`, `lddtree`, and
`diffoscope`.

- If a Cargo compiler-extraction trigger fires, extract only that provider into
  an explicit producer and keep `build.rs` link-only for it.
- If production JIT coverage is bounded and request/startup gates fail, package
  or AOT those exact shapes.
- Otherwise retain current source build and model-load warmup behavior.

P5 is not on the critical path until its trigger fires.

## Verification

| Change | Required gate |
| --- | --- |
| stats/parser | known-request counter consistency |
| evidence schema | valid/invalid golden fixtures |
| selector | numeric component parity + same-binary E2E/SLO |
| dead deletion | caller graph + archive/link + runtime zero-hit |
| cache identity | input mutation matrix + corruption fixtures |
| release bundle | clean exact-ID consumer |
| JIT policy | cold lifecycle compiler trace + first-request audit |

Runtime changes produce the required wins/errors entry. Docs and dev-only
inventory generation are benchmark-exempt until they change runtime selection
or generated kernel bytes.

## Stop conditions

Stop and reduce scope when:

- the first family requires a general predicate DSL;
- stats instrumentation affects hot-path performance;
- generated selector is larger or less readable than the deleted branch;
- exact Release bundle work grows into a registry service;
- build extraction begins before process tracing attributes the wall;
- a dead-code cleanup lacks runtime/archive proof.

## Completion

The necessary plan is complete when:

1. Qwen FP8 dense has canonical evidence, generated exact-cell selection, and
   trustworthy engagement stats.
2. Operator inventory distinguishes role and reachability without duplicating
   all FFI by hand.
3. Dead rows/symbols are removed only by family-level proof.
4. CI/releases fetch immutable exact kernel bundles, never `latest`.
5. Cache identities cannot reuse stale provider/toolchain output.
6. Formal request execution launches no compiler.
7. Current hot/no-op development speed is preserved.

Everything beyond these seven outcomes remains trigger-gated.
