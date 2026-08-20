# Qwen CUDA quant-linear dispatch consolidation

> Status: Proposed
>
> Parent plan:
> [CUDA operator organization across kernels and models](2026-08-20-cuda-operator-organization.md).
>
> Scope: Qwen dense quantized linear dispatch in `infer-cuda`. This plan does
> not change kernel math, serving APIs, model formats, or the backend seam.

## Decision

Keep the public `gemm_batch` and `gemv` entry points. Replace their parallel
quantized routing with one internal dispatch keyed by weight format and `M`.
Give each stored weight family one route owner and each CUDA ABI one launch
helper. Validate retained weight layouts when loading and fail before serving
if no route can consume them.

Do not add an operator registry, trait hierarchy, runtime graph, or generic
kernel planner. Duplicated local control flow is the complete problem.

## Problem

`crates/infer-cuda/src/ops/quant_linear.rs` is 2,065 lines and currently owns
four concerns at once:

1. route policy;
2. scratch allocation and profiling;
3. CUDA ABI launch code;
4. fallback ordering for every quantized format.

The same weight can enter through two independent paths:

```text
HiddenStates                         DeviceVec
    |                                   |
    v                                   v
gemm_batch()                         gemv()
    |                                   |
    +-- ordered try_* chain              +-- Marlin match
    +-- fallback format match            +-- fallback format match
```

This has already produced a correctness defect. W8A16 repack released
`qweight` and `qscales`; `gemm_batch` used the retained Marlin layout while
`gemv` attempted to read the released source for a quantized `lm_head`. The
same earlier defect class affected FP4 and FP8. See
`docs/experience/errors/2026-08-20-qwen-spec-budget-and-w8-lm-head.md` and
`docs/experience/wins/2026-08-20-marlin-source-freed-18gb.md`.

The storage contract is also implicit. `DeviceMatrix` exposes independent
`Option` fields for source weights, source scales, Marlin weights, Marlin
scales, and DeepGEMM metadata. Legal combinations depend on format, device,
shape, and whether prefill needs the source. Routing currently reconstructs
that contract from field presence.

## What already exists

| Existing mechanism | Location | Decision |
| --- | --- | --- |
| Public shape validation | `crates/infer-cuda/src/ops.rs:152-249` | Reuse unchanged |
| Explicit checkpoint format | `crates/cuda-kernels/src/tensor.rs:764-812` | Reuse as the top-level dispatch key |
| FP4 route policy | `crates/infer-cuda/src/ops/quant_linear.rs:978-1007` | Move with the FP4 family |
| FP8 route policy | `crates/infer-cuda/src/ops/quant_linear.rs:1244-1273` | Move with the FP8 family |
| Shared raw-pointer Marlin launchers | `crates/infer-cuda/src/ops/quant_linear.rs:903-1346` | Keep one launcher per ABI |
| Qwen operator hit counters | `crates/infer-cuda/src/ops/quant_linear.rs:176-242` | Preserve identifiers and semantics |
| FP8 and W8A16 numerical harnesses | `crates/infer-cuda/examples/marlin_{fp8,w8a16}_parity.rs` | Extend the existing harnesses |
| NVFP4 probe | `crates/infer-cuda/examples/marlin_fp4_probe.rs` | Reuse for FP4 route checks |
| Load-time Marlin preparation | `crates/infer-cuda/src/loader.rs:5718-5745` | Add post-preparation validation here |
| Source-release marker | `DeviceMatrix::quant_source_freed()` | Preserve LoRA and offload behavior |

## Goals

1. One route owner per stored weight family.
2. `gemm_batch` and `gemv` differ only in shape validation and buffer view
   construction.
3. Every retained-layout state is either consumable by a route or rejected at
   load with the tensor name, format, shape, and missing buffers.
4. No new allocation, host readback, device synchronization, or environment
   lookup in the forward hot path.
5. Kernel selection, operator counters, numerical output, CUDA graph behavior,
   and serving performance remain unchanged during the structural phase.
6. Later kernel tuning uses measured route shares and lands separately from the
   refactor.

## NOT in scope

- **MoE grouped GEMM routing.** It has different activation packing, expert
  metadata, and scale contracts. Sharing a registry would hide those costs.
- **DSv4 W4AFP8 kernels.** They are model-specific grouped operators and have a
  separate correctness gate.
- **Changing `DeviceMatrix` to a large storage enum.** This would touch every
  format, loader, LoRA path, snapshot path, and backend consumer. Local
  validation removes the current invalid states at much lower cost.
- **New CUDA kernels.** The first two phases reorganize existing routes and
  validate storage only.
- **Changing DeepGEMM/Marlin crossover thresholds.** Threshold changes require
  their own measured treatment and dated report.
- **Changing public stats identifiers.** Existing dashboards and experience
  entries depend on them.
- **A cross-backend operator abstraction.** Metal and CUDA have different
  execution and storage costs; this stays below the backend seam.

## Scope control

The complete plan reaches eight runtime or harness files across multiple
tranches. No commit touches more than five files. The work stays sequential
because the same route contract connects every tranche.

The smaller alternative is to add another shared raw Marlin helper and leave
the two fallback matches in place. It fixes the next missing Marlin arm but
retains the mechanism that caused the FP4, FP8, and W8A16 route drift. The
larger alternative is a typed `DeviceMatrix` storage rewrite across all
backends and formats. Its migration cost and blast radius exceed this local
problem. The phased route-owner design is the minimum complete solution.

## Required invariants

### Dispatch invariants

1. `WeightFormat` is the only top-level format discriminator.
2. Each format family owns the complete priority order from preferred kernel to
   terminal fallback.
3. A route may return `Declined` only when a later route can consume the same
   resident representation.
4. Once a source buffer is released, every supported `M` must select a retained
   layout.
5. A CUDA call increments its counter only after successful launch submission.
6. Route selection performs no allocation and launches no work.
7. Dynamic resource failure, such as unavailable scratch capacity, retains the
   current fallback behavior and never falls through to a released source.

### Storage invariants

| Format | Valid source layout | Valid retained layout | Required rule |
| --- | --- | --- | --- |
| `W8A16` | `qweight + qscales` | `marlin_packed + marlin_scales` | At least one complete pair; half-pairs rejected |
| `W4A16` | `qweight + qscales` | None in this plan | Source pair required |
| `Fp4E2M1Group` | `qweight_u8 + qscale_fp8 + scale_f32` | `marlin_packed + marlin_scales` | `fp4_deepgemm_sfb` requires the complete source triplet |
| `Fp8BlockScaled` | `qweight_u8 + scale_f32` | `marlin_packed + marlin_scales` for per-channel weights | DeepGEMM availability requires source retention |
| `Fp8PerShard` | `qweight_u8 + scale_f32` | None | Source pair required |
| `Dsv4Fp8BlockScaled` | `qweight + dsv4_scales` | Existing DeepGEMM cache outside this dispatcher | Source pair required for this fallback |
| `Dsv4Fp4BlockScaled` | `qweight + dsv4_scales` | Existing DeepGEMM cache outside this dispatcher | Source pair required for this fallback |
| `MarlinW4A8` | Existing Marlin W4A8 fields | Same | Preserve current validation and route |
| `W2A16` / `TurboQuant` | Existing format-specific fields | None | Preserve current validation and route |

The validator reports a load error. It does not repair state, synthesize a
fallback, or retain extra VRAM defensively.

### Hot-path invariants

- No `ctx.sync()`, D2H/H2D metadata copy, or device query in dispatch.
- Scratch remains preallocated or lazily created at the existing warmup sync
  point. Refactoring must not move first allocation into a captured step.
- Raw device-pointer guards remain live through the CUDA call.
- Every `usize -> i32` ABI conversion remains guarded by existing shape limits
  or becomes a checked conversion before launch.
- Empty `M` never launches a zero-grid kernel. The public caller must reject or
  skip it before quant dispatch.

## Target architecture

```text
ops::gemm_batch()                     ops::gemv()
  shape checks                          shape checks
  M = x.seq_len                         M = 1
         \                               /
          +---- quant_linear::run() ----+
                        |
                        v
              match weight.weight_format
          +-------------+--------------+-------------+
          |             |              |             |
          v             v              v             v
       fp8::run      fp4::run       int::run      existing rare
          |             |              |           format routes
          v             v              v
      one ordered   one ordered    one ordered
      route owner   route owner    route owner
          |             |              |
          +-------------+--------------+
                        |
                        v
             one launch helper per CUDA ABI
```

The internal call shape is deliberately small:

```rust
fn run(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    input: &CudaSlice<bf16>,
    output: &mut CudaSlice<bf16>,
    m: usize,
) -> Result<()>;
```

`HiddenStates` and `DeviceVec` already store the same `CudaSlice<bf16>` data.
The public wrappers retain their typed shape checks, then pass only the buffers
and `M`. Do not introduce an input trait or owned view type.

### Module ownership

Use flat sibling modules, matching the repository convention:

| File | Owns |
| --- | --- |
| `ops/quant_linear.rs` | Entry dispatch, shared Marlin scratch, profiling helper, stats aggregation, uncommon legacy formats |
| `ops/quant_linear_fp8.rs` | FP8 policy, DeepGEMM, Marlin, dequant-BF16, scalar/batched GEMV, FP8 counters |
| `ops/quant_linear_fp4.rs` | NVFP4 policy, widen-to-E4M3 DeepGEMM, Marlin, dequant-BF16, scalar/batched GEMV, FP4 counters |
| `ops/quant_linear_int.rs` | W8A16, W4A16, MarlinW4A8, W2A16, and TurboQuant routes and counters |

Do not create `mod.rs`. `quant_linear.rs` declares the three siblings with
explicit `#[path = "quant_linear_*.rs"]` attributes.

### Route ownership

#### FP8

```text
Fp8BlockScaled / Fp8PerShard
  |
  +-- eligible native/block/per-channel DeepGEMM? -> launch
  |
  +-- retained per-channel Marlin layout?         -> launch for every M
  |
  +-- source present and M above dequant floor?   -> dequant + BF16 GEMM
  |
  +-- source present?                             -> scalar/batched GEMV
  |
  `-- error: no consumable FP8 representation
```

Scratch reservation may decline DeepGEMM before launch. The next branch must
still have its required source. Loader validation must preserve a representation
for this dynamic fallback.

#### FP4

```text
Fp4E2M1Group
  |
  +-- source + sfb + measured prefill floor? -> widen E4M3 + DeepGEMM
  |
  +-- retained Marlin layout?                -> Marlin
  |
  +-- source + large M?                      -> dequant + BF16 GEMM
  |
  +-- source?                                -> scalar/batched GEMV
  |
  `-- error: no consumable FP4 representation
```

The `sfb` presence both enables DeepGEMM and pins the source. Preserve that
coupling until a separate design replaces it explicitly.

#### W8A16 and W4A16

```text
W8A16: retained Marlin -> source dequant/GEMV -> error
W4A16: source GEMV -> error
```

Delete `try_w4a16_dequant_bf16_gemm_batch`; it always returns `false` and
creates a route that does not exist. Keep W8A16 dequant only if the current
implementation can still engage for an unrepacked source; otherwise prove it
dead and delete it in the same tranche.

### Singular and batched GEMV ABIs

Some formats expose both a singular and batched CUDA ABI. Keep that distinction
inside the family launch helper:

```text
route = Gemv
  M == 1 and singular ABI is accepted -> singular ABI
  otherwise                            -> batched ABI(M)
```

Do not replace the singular ABI with `batch=1` from code inspection alone.
First require exact or reference-bounded numerical parity and an H20 timing
comparison. If the batched ABI is equal or faster, delete the singular route in
a separate measured commit. If it loses, retain both behind this one helper.

## Load and restore contract

Add `validate_qwen_quant_linear_storage(name, matrix)` in `infer-cuda`, close to
`marlin_repack_dense`. Call it after repack, DeepGEMM metadata preparation, and
source release.

Validation order:

```text
checkpoint load
  -> format/shape validation
  -> optional fuse
  -> optional Marlin repack
  -> optional DeepGEMM metadata preparation
  -> conditional source release
  -> validate final consumable representations
  -> publish DeviceMatrix to model weights
```

Also audit these mutation paths before landing:

1. LoRA merge and unmerge, especially `quant_source_freed()` callers;
2. weight offload and reload snapshots that rebuild Marlin layouts;
3. fused `gate_up` construction, which must happen before repack;
4. tied and untied `lm_head` loading;
5. TP shards whose local `N` or `K` cease to satisfy Marlin alignment.

The validator remains Qwen-local. Moving all `DeviceMatrix` variants into a
global typed-storage enum is deferred until at least one more consumer needs the
same state machine.

## Failure modes

| Failure | Prevention | Test | Visible result |
| --- | --- | --- | --- |
| Source released, fallback still selected | One family route plus final-state validator | Route matrix with source absent and Marlin present | Load fails or retained route runs; never a late missing-buffer error |
| Half-created Marlin pair | Reject `packed XOR scales` | Pure validator test | Tensor-named load error |
| DeepGEMM scratch declines after source release | Validator requires another consumable representation | Forced-decline route test | Marlin fallback or explicit error |
| `M=1` takes a different policy from batched `M=1` | Both wrappers call the same internal dispatcher | Counter/route equality test | Same implementation ID |
| Pre-sm80 device chooses Marlin | SM gate remains part of route and repack | Pure SM route cases | Source fallback |
| TP shard becomes unaligned | Repack declines without releasing source | Misaligned N/K cases | Source fallback with warning |
| FP4 `sfb` exists after source release | Validator rejects it | Invalid-state test | Load error |
| LoRA updates a source beside an active Marlin copy | Preserve current hard error for source-freed or dual-layout base | Existing LoRA merge tests plus runtime smoke | Clear merge error; no stale layout |
| Reload restores source but not retained layout | Audit snapshot rebuild and validate after restore | Offload/reload round-trip | Restored route matches pre-offload route |
| Counter moves before failed CUDA submission | Increment after `.result()?` | Injected/invalid launch harness where available | Failed launch does not count |
| Refactor adds lazy allocation inside capture | Keep scratch ownership and warmup unchanged | captured/eager route parity | No capture failure or per-step allocation |
| `usize` truncates at CUDA ABI | Checked conversion at boundary | oversized metadata unit case | Host error before launch |

Any state that would otherwise produce a silent fallback is a critical gap.
The route must either consume a verified representation or return a specific
error.

## Test plan

### Coverage map

```text
CODE PATHS

ops::gemm_batch / ops::gemv
  +-- [existing] dense BF16 bypass
  +-- [required] quantized wrappers produce identical route at M=1
  `-- quant_linear::run
      +-- [required] FP8
      |   +-- blocked DeepGEMM
      |   +-- per-channel DeepGEMM
      |   +-- Marlin with released source
      |   +-- dequant-BF16 fallback
      |   `-- scalar/batched GEMV fallback
      +-- [required] FP4
      |   +-- widen-E4M3 DeepGEMM
      |   +-- Marlin with released source
      |   +-- dequant-BF16 fallback
      |   `-- scalar/batched GEMV fallback
      +-- [required] W8A16
      |   +-- Marlin with released source, including M=1 lm_head
      |   `-- source fallback when repack declines
      +-- [required] W4A16 / W2A16 / TurboQuant retained behavior
      `-- [required] invalid storage returns a tensor-named error

RUNTIME FLOWS

Qwen prefill -> gemm_batch -> family route -> chosen kernel
Qwen decode  -> gemm_batch or gemv -> same family route -> chosen kernel
MTP/DSpark   -> M=1 gemm_batch -> same route as lm_head gemv
offload      -> snapshot -> reload/repack -> same route and output class
```

### Local checks

Add one table-driven unit test for route decisions and storage validation. It
must cover all branches above without a GPU. Do not add one test function per
format.

Run:

```bash
cargo fmt --check
CUDARC_CUDA_VERSION=12080 cargo check -p infer-cuda --release \
  --no-default-features --features cuda,no-cuda --tests
git diff --check
```

The macOS `cuda,no-cuda` test binary cannot link CUDA C symbols, so `cargo
check --tests` is the local gate. CUDA execution is a remote gate.

### Kernel numerical checks

Run the existing harnesses on one H20 with at least three deterministic seeds:

```bash
CUDA_HOME=/usr/local/cuda cargo build --release --features cuda \
  -p infer-cuda --example marlin_fp8_parity
CUDA_HOME=/usr/local/cuda cargo build --release --features cuda \
  -p infer-cuda --example marlin_w8a16_parity
```

Required shapes and `M` values:

- FP8: every production shape already listed by `marlin_fp8_parity`, with
  `M={1,2,4,16,64,256}`;
- W8A16: existing dense shapes plus an untied quantized `lm_head`, with
  `M={1,2,4,8,16,32}`;
- FP4: all Qwen NVFP4 dense shapes, with both aligned and repack-declined cases;
- boundary cases: zero values, signed small values, finite extrema, last output
  tile, misaligned `N`, misaligned `K`, and unsupported group size.

For GEMM or changed launch selection, record `max_abs`, `p99_abs`, RMSE, maximum
relative error with a stated near-zero floor, and cosine against FP32/FP64.
Candidate reference-error metrics may not worsen by more than 5% relative to
the accepted implementation.

### Runtime correctness checks

Use three checkpoint families:

1. Qwen3.6 27B block-scaled FP8;
2. Qwen3.8 27B mixed NVFP4/per-channel FP8;
3. Qwen3.5/3.6 W8A16 with an untied, quantized `lm_head`.

For each applicable checkpoint:

- one non-degenerate generation;
- prefill and decode engagement counters before/after the request;
- MTP/DSpark M=1 projections where the model supports them;
- eager and captured execution where capture is supported;
- zero request errors, empty outputs, loops, or missing usage;
- `scripts/lever_gate.sh` against the same-config baseline envelope;
- `python3 scripts/needle_gate.py temp` at 512/4096/16384/32768, three runs per
  length.

The W8A16 gate remains open until the checkpoint actually has an untied
quantized `lm_head`; a tied BF16 embedding does not exercise the route.

### Route and telemetry checks

Preserve all existing implementation IDs. For a fixed request and checkpoint,
the baseline and candidate must have the same total projection count and the
same route counts by implementation, except for an explicitly approved route
change in a later performance tranche.

Add no per-request allocation to telemetry. If M=1-specific evidence is needed,
prefer a standalone harness or profile label. Add a permanent counter only when
it answers an ongoing production question.

## Performance plan

The structural change predicts no speedup. Its acceptance condition is no
measurable regression and identical route engagement.

### Structural A/B

- Archive the pre-change binary and candidate binary.
- Same H20, clocks, model, TP/EP, slots, KV dtype, server flags, JSONL workload,
  request order, output cap, and seed.
- Use the canonical 32K multi-turn agent workload through
  `scripts/bench_throughput.py` at `c={1,4,8,16}` with at least 20 completed
  requests per reported point.
- Record cold and warm slices, prompt/completion token distributions, prefix
  hits, KV residency, queue/preempt counters, errors, output tok/s, req/s, TTFT
  p50/p99, and ITL p50/p99.
- If delta is within the documented drift band, run at least three trials per
  arm and report median plus range. Any unresolved negative sign blocks the
  refactor.

### Later kernel tuning

After consolidation, collect a full-run kernel ledger. Rank candidates by
actual event time separately for prefill, decode, mixed, and drain. Optimize
only the largest supported share.

Potential treatments, each requiring a separate commit and report:

1. delete singular GEMV ABIs when `batch=1` is numerically equivalent and
   faster or equal;
2. change DeepGEMM/Marlin/dequant floors for a named shape and workload;
3. fuse preparation only when trace evidence shows it on the request critical
   path;
4. remove a fallback only after route counters prove zero supported use and
   load validation proves a replacement.

A component-kernel gain is diagnostic evidence. Serving claims require the
canonical workload A/B.

## Implementation sequence

### Tranche 0: Baseline receipt

No code changes.

1. Record HEAD, binary hash, kernel build ID, model, GPU, driver/CUDA, flags,
   slot line, and KV capacity.
2. Capture `/v1/stats` implementation hits for one FP8, one NVFP4, and one W8A16
   request.
3. Archive the binary and raw outputs used by the later A/B.

Exit: baseline can be reproduced and its binary is still available.

### Tranche 0A: Reproducible numerical inputs

Commit: `test(cuda): cover quant-linear routes across fixed seeds`

Files:

1. `crates/infer-cuda/examples/marlin_fp8_parity.rs`;
2. `crates/infer-cuda/examples/marlin_w8a16_parity.rs`;
3. `crates/infer-cuda/examples/marlin_fp4_probe.rs`.

Work:

- accept or internally enumerate three fixed seeds;
- print the seed, shape, `M`, kernel build ID, and reference-error metrics;
- add repack-declined boundary shapes without changing production code;
- exit nonzero on the first acceptance violation.

This is dev-only harness work and is exempt from the runtime bench-entry gate.

Exit: the baseline and candidate can run byte-identical input matrices across
all affected routes.

### Tranche 1: One route owner per format

Commit: `refactor(cuda): consolidate Qwen quant-linear dispatch`

Files, maximum five including the required report:

1. `crates/infer-cuda/src/ops/quant_linear.rs`;
2. `crates/infer-cuda/src/ops/quant_linear_fp8.rs`;
3. `crates/infer-cuda/src/ops/quant_linear_fp4.rs`;
4. `crates/infer-cuda/src/ops/quant_linear_int.rs`;
5. one dated `docs/experience/errors/*pending-remote*.md` entry.

Work:

- introduce the common slice-plus-`M` internal entry;
- move format-specific code without changing CUDA arguments;
- make each family own its full route order;
- preserve counters and profiling labels;
- delete the repeated full-format fallback match and the permanently disabled
  W4 dequant arm;
- add the one table-driven route test.

Exit: local checks pass; no old and new dispatch paths coexist; remote
numerical, engagement, capture, and model gates pass; structural A/B has no
unresolved regression.

### Tranche 2: Reject invalid retained layouts at load

Commit: `fix(cuda): validate Qwen quant-linear storage states`

Files:

1. `crates/infer-cuda/src/loader.rs`;
2. the family module that owns validation predicates, only if needed;
3. a separate dated `docs/experience/errors/*pending-remote*.md` entry.

Work:

- validate complete source/retained pairs after final repack and release;
- include tensor name, format, shape, group size, and missing representation in
  errors;
- audit fuse, TP, tied/untied head, LoRA, and reload paths;
- add invalid-state cases to the existing table-driven test.

Exit: incompatible checkpoints fail during load; every accepted matrix has at
least one route for every reachable `M`; offload/reload restores the same route.

### Tranche 2A: Verification verdict

After the remote gates finish, update both dated reports with raw results and
add one `CHANGELOG.md` line linking the accept-or-reject verdict. This docs-only
commit contains no runtime change. Cut a release tag only if this work is
declared a named project phase exit.

### Tranche 3: Measured simplification or performance treatment

Commit only when Tranche 0/1 evidence identifies one treatment. Use
`perf(cuda)` for a measured gain or `refactor(cuda)` for a proven deletion.

One treatment per commit. Do not combine a crossover change, kernel ABI removal,
and storage refactor. Each runtime treatment receives its own dated result and
matched A/B.

Exit: PASS or KILL verdict recorded. A KILL removes the treatment and retains
the evidence.

## Rollback

Each tranche is independently revertible.

- Tranche 1 rollback restores the archived pre-change dispatcher and binary.
- Tranche 2 rollback restores permissive loading but must not remain deployed if
  it re-admits a demonstrated invalid state.
- Tranche 3 rollback restores the named accepted route threshold or ABI.

After rollback, rebuild from the restored source and rerun the model gate. Source
equality alone is not a rollback proof.

## Sequential execution

No parallel worktrees. All implementation steps touch the same quant-linear
module and route contract; parallel edits would create merge conflicts and make
behavioral equivalence harder to review.

## Engineering review summary

| Area | Result |
| --- | --- |
| Scope | Reduced to Qwen dense quant-linear routing; global storage rewrite and MoE remain outside scope |
| Architecture | Two confirmed issues: parallel entry routing and implicit retained-layout state |
| Code quality | Three confirmed issues: 2,065-line mixed-responsibility file, duplicated format matches, permanently disabled W4 dequant branch |
| Tests | One table-driven host test plus three existing CUDA harnesses and model-level gates cover the planned branches |
| Performance | Structural work has a no-regression contract; threshold and ABI tuning are separate measured treatments |
| Failure handling | Every listed silent failure receives load validation, a route test, or both |
| TODOs | No `TODOS.md` update; deferred items are trigger-gated in `NOT in scope` |
| Parallelization | Sequential implementation; every tranche shares the route contract |

No unresolved architectural decision remains. Implementation results may still
KILL a singular-ABI deletion or a later threshold treatment without affecting
the route consolidation.

## Definition of done

- `gemm_batch` and `gemv` call one internal quantized dispatcher.
- There is one top-level execution match on `WeightFormat`.
- Every weight family owns one ordered route and its complete terminal error.
- Every CUDA ABI has one launch helper; singular/batched variants are grouped in
  that helper.
- No route reads a buffer that load or reload may release.
- Invalid retained-layout combinations fail at load with tensor context.
- Existing operator IDs and route counts remain stable for the structural
  tranche.
- Local type checks, numerical harnesses, capture/eager checks, model gates, and
  canonical workload A/B pass.
- A dated experience entry contains Goal, Hypothesis, Parameters, Environment,
  Results, Problems, and Learnings.
- No temporary flags, duplicate old path, deferred compatibility adapter, or
  unexplained TODO remains.

## Acceptance verdict

Accept Tranches 1-2 when correctness, engagement, capture, storage validation,
and no-regression A/B all pass. This closes operator organization.

Performance work remains a separate measured loop. Accept each later treatment
only when its target wall-clock metric improves with aligned scheduling and no
correctness, latency, throughput, memory, or failure-rate regression.
