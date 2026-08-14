# cuda-oxide borrow assessment

Date: 2026-08-13. Upstream pinned at `17eb3cc5` (NVlabs/cuda-oxide, 2026-08-13).
Research note only — no runtime change, so no bench entry is required by the
bench gate.

## Question

What in cuda-oxide is worth borrowing for ARLE? cuda-oxide is NVlabs' custom
rustc codegen backend that compiles `#[kernel]` functions in pure Rust to PTX,
plus a host runtime (`cuda-core`, `cuda-async`). This note covers the host-side
patterns only; the compiler itself is assessed under "Rejected" below.

## Method

Every claim below was verified against source at the pinned commit, not the
README. ARLE-side costs are cited to in-tree files. The review was
adversarial: each candidate was stress-tested for what breaks when ported to
ARLE's stack (cudarc, graph-captured serving hot path, offline pod builds).

## Candidate 1 — event-gated deferred reclamation: adopt

### Mechanism (verified)

`crates/cuda-async/src/reclaim.rs` (~150 usable LOC): a process-wide
`Mutex<Vec<LimboEntry>>`, each entry pairing a completion gate (a CUDA event
recorded on the stream after the submitted work) with a type-erased payload.
`sweep()` runs on every future poll and drop: it queries each gate and drops
payloads whose event has passed; entries whose query fails stay parked.
`drain()` blocks for deterministic teardown; if the blocking wait fails, the
payload is leaked with a stderr message. Payloads drop outside the lock so a
payload's own drop may re-enter the limbo.

`crates/cuda-async/src/device_future.rs`: dropping an in-flight future never
cancels GPU work. It records an event on the stream and parks the result; only
if the event cannot be recorded does it fall back to synchronizing the stream,
and if that fails it leaks. Completion wakes the future via
`cuLaunchHostFunc` + `AtomicWaker`; the callback only sets an atomic and makes
no CUDA calls, as the driver requires.

`crates/cuda-async/src/device_box.rs`: `DeviceBox` owns device memory and
enqueues `cuMemFreeAsync` on a per-device deallocator stream at drop.

### Adversarial findings

1. The deallocator stream has no event-ordering against producer streams.
   Correctness rests on drop-after-completion (the owned-launch path) or a
   caller-side sync (the borrowed-launch path, stated as a safety contract in
   the type's docs). An adopter must keep that invariant explicitly.
2. The use-after-free hazard is specific to the stream-ordered allocator
   (`cuMemAllocAsync` pool): a freed block can be handed to the next
   allocation on the same stream while a kernel still writes it. ARLE uses
   this pool (`crates/cuda-kernels/src/tensor.rs:364`), so the hazard is live.
3. The pattern does not compose with stream capture: a `cuMemFreeAsync` on a
   non-captured deallocator stream executes immediately while another stream
   is captured. cuda-oxide has no graph support, so this never arises there.
   In ARLE the pattern belongs on eager paths only.

### ARLE current state

cudarc event tracking is deliberately disabled
(`crates/cuda-kernels/src/tensor.rs:353-362`) because automatic per-buffer
waits break graph capture; cross-stream ordering is owned explicitly by
`CudaPipelineFence`. The reclaim pattern is per-cancellation, not
per-buffer, so it does not reintroduce that problem.

Two sites pay full-device syncs today for exactly this hazard:

- `copy_bf16_device_ptr_to_local`
  (`crates/autograd/src/backend_cuda/handle.rs:449-463`): a
  compute-sanitizer-confirmed use-after-free (foreign allocator frees via
  `cuMemFreeAsync` while the D2D copy runs) is fixed with a full
  `context.synchronize()` per bridge call.
- `release_kv_pool` (`crates/infer-cuda/src/executor/qwen35.rs:898-918`):
  sync, drop (enqueues frees), sync again, then trim. Two full-device syncs
  per release.

Both are eager paths. An event-gated free replaces the syncs with an event
record plus deferred reclamation.

### Verdict

Adopt the pattern, implemented locally against cudarc's event API (~150 LOC),
at the two sites above. Keep it out of the captured serving hot path.

## Candidate 2 — launch contract: borrow the idea, not the machinery

### Mechanism (verified)

`#[launch_contract(domain = 1, block = (..), dynamic_shared = N,
min_compute_capability = (9, 0), requires = (a.len() >= n))]` on a kernel
(`crates/cuda-macros/src/lib.rs:576-729`). The `cuda_module` macro generates a
`prepare_*` method that checks indexing dimensionality, exact block shape, the
dynamic shared-memory envelope (exact or range, plus power-of-two alignment),
the device's compute capability, and the `requires` relations. The relations
are validated against the parameter list at macro-expansion time and emitted
as overflow-safe host-side checks. The raw launch stays `unsafe`; the prepared
launch is safe. The raw builder is inert and `!Send`, and the finalized launch
is immutable, with compile-fail tests pinning both properties
(`crates/cuda-async/src/launch.rs:39-58, 82-89`).

### Adversarial findings

1. The checks cover only what the author declares. A wrong contract gives a
   false sense of safety.
2. For ARLE's FFI kernels the checks are runtime checks. Nothing links the
   Rust launch site to the C kernel signature; that is inherent to FFI and a
   contract cannot fix it.
3. The machinery is tied to their proc-macro and embedded-PTX model. ARLE
   would reimplement the idea, not reuse the code.

### ARLE current state

`launch_1d` / `launch_rows` (`crates/autograd/src/backend_cuda/kernels.rs:421-498`)
hardcode 256 threads per block and zero dynamic shared memory. The kernel set
is 78 `__global__` definitions across 29 `.cu` files, reached through 58
distinct string-keyed lookups. Shape and alignment requirements live in
comments.

### Verdict

Borrow the `requires`/envelope idea: a small per-kernel descriptor declared at
registration and checked before launch. Low priority — the geometry is already
centralized in two helpers. The string-keyed lookup is the larger risk, and
the contract pattern does not address it.

## Candidate 3 — KernelFamily: pattern only, YAGNI today

### Mechanism (verified)

`crates/cuda-host/src/kernel_family.rs` (497 LOC). A const-generic family of
`N` compiled variants. Eligibility (`KernelProblem::validate`) is separated
from preference (`KernelSelector::select`). `Force(id)` overrides cache and
selector but still passes eligibility validation. Cache entries are treated as
untrusted and revalidated; selector output is revalidated; every selection
reports its provenance (override, cache, selector). Selection is
allocation-free (fixed-array compaction). The module has unit tests and one
real user in their tree (`gemm_sol_final`).

### Adversarial findings

1. One real user in their own tree. The design is clean but not yet
   load-bearing.
2. For ARLE's current selection sites — a handful of branches (decode versus
   prefill, head-dim variants, FP8 KV split-KV) — the framework is heavier
   than the problem.

### Verdict

Keep as a named pattern. Revisit when per-SM-tier kernel variants or a growing
variant count make the ad hoc selection a maintenance cost.

## Rejected — the compiler itself

### Verified facts

No CUDA Graph support exists in the tree at the pinned commit (grep for
`cuGraph`, stream capture, and graph APIs across `*.rs` and `*.md`: zero
hits). The project is alpha, requires a pinned nightly (2026-04-03) with
`rustc-dev`, a custom codegen backend, LLVM 21/22, and clang 21.

### Adversarial re-check

The strongest case for adoption is the autograd training kernels: about 4k
LOC of simple elementwise/reduce CUDA C, string-keyed FFI launches, and a
Rust-fluent team. The graph objection does not apply to the training path,
which is eager. Four facts still block adoption:

1. Offline pod build cost: a forked rustc toolchain plus LLVM 21/22 added to
   a build that already vendors a tilelang venv.
2. Correctness risk class: a miscompile in an alpha custom backend versus
   nvcc, on the correctness-critical OPD path.
3. No measured FFI tax. The training path is launch-bound (72% host idle);
   the fix is fewer launches (fusion, capture), not a different kernel
   language. cuda-oxide does not reduce launch count.
4. The kernels exist and work. A rewrite is pure cost.

What would change the answer: a measured marshalling cost that a
same-language kernel removes, or a sustained stream of new training kernels.
The cheap test then is a one-kernel pilot with a matched A/B. The LTOIR FFI
path (a Rust kernel calling CCCL C++ in the same translation unit) is the
hedge if partial adoption ever makes sense.

## Reference value (no adoption)

- `intrinsics/`: a generated catalog of PTX intrinsics where each entry
  carries its toolchain version, expected PTX text, PTX ISA documentation
  link, and evidence stage (lowered, validated, executed). This is the method
  to copy if ARLE ever tracks PTX intrinsic coverage from the Rust side.
- `crates/rustc-codegen-cuda/examples/gemm_sol_final/src/kernels.rs`
  (1304 lines): the cleanest public sm_100 hand-written kernel reference —
  CLC, tcgen05, and TMA multicast with the cta_group::2 barrier protocol
  documented in comments. Read it before writing Blackwell kernels.

## Recommendation summary

| Item | Decision | Action |
|------|----------|--------|
| Event-gated deferred reclamation | Adopt | Implement locally (~150 LOC), apply at the two sync-heavy sites |
| Launch contract | Borrow idea | Per-kernel envelope descriptor with `requires` checks; low priority |
| KernelFamily | Pattern only | Revisit when variant selection grows |
| Rust-to-PTX compiler | Reject | Re-evaluate on a measured FFI tax or a stream of new training kernels |
| Intrinsics catalog, gemm_sol_final | Reference | Use as documentation |

## Outcome (2026-08-14)

Site 1 (the bf16 teacher-logits bridge) shipped, but the implementation
diverged from the limbo design above. Reading the actual free path showed
that cudarc's `CudaSlice::drop` frees on the slice's own stream
(`cuMemFreeAsync(ptr, slice.stream)`), so the source buffer's free is already
stream-ordered. The correct fix is therefore cheaper than a limbo: enqueue the
D2D copy on the source stream (ordered after the producer lm_head GEMM),
record a completion event, and make the student stream wait on it. The
source's later free is ordered after the copy on the same stream. No host
sync, no limbo, ~40 LOC. A `src_stream == 0` fallback keeps the legacy sync
path for callers without a source stream. See
`docs/experience/wins/2026-08-14-bf16-bridge-event-ordered.md` (pending-remote
bench).

Site 2 (`release_kv_pool`) also shipped: the two full-context syncs are
replaced by a single event sync (record after drop, wait for the frees, then
trim). The event sync is cheaper than a context sync because it only waits
for the frees, not all queued work. See
`docs/research/2026-08-14-stream-ordered-gpu-memory-long-term.md` for the
long-term plan and the remaining sync sites.

The limbo pattern itself remains unbuilt. It solves a different problem
(cancelled futures) that ARLE does not have; the two sync sites were both
better addressed by stream ordering.

