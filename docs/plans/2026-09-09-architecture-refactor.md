# Architecture: stage boundaries the compiler enforces

Date: 2026-09-09 · Status: Plan, pending acceptance · Owner: ckl

## What this document decides

Three things, in this order:

1. Which boundary in the runtime is wrong, stated as a measurement rather than a preference.
2. What the boundary should be, and by what criterion it stays correct.
3. The sequence that gets there, where each step is verifiable on its own and leaves no half-state.

It does not decide model support, feature priority, or anything on
[`2026-08-24-roadmap.md`](2026-08-24-roadmap.md). Those rank work inside the
current structure; this changes the structure.

## 1. The complaint, and the mechanism behind each part

| Complaint | Mechanism |
|---|---|
| Rust iteration is slow | `infer-cuda` is 60,347 lines in one crate. A one-line change to host logic recompiles it and links 84,728 lines of C++ |
| Kernels are hard to work with | `arle kernel` exists — `crates/infer-cuda/src/kernel_bench.rs`, 316 lines — with zero references in `scripts/pre_push_checks.sh` or any file under `.github/workflows/` |
| The flow is not legible | No type names the CPU-to-GPU transition. `ForwardPlan` (`crates/infer-plan/src/lib.rs:85`) and `KvBatchDescriptor` (`crates/infer-seam/src/kv_batch.rs:16`) are both pure host; the transition happens afterwards inside `infer-cuda` with no name. 3,322 lines of host logic have drifted below the seam |

The three share one cause. There is no place above the device where step
construction is supposed to live, so it lives below it, and everything below it
carries the compile cost and the GPU-only verification cost of the device.

## 2. Where performance actually comes from

319 entries under `docs/experience/wins/`, bucketed by subject from the
filename. Counts, not magnitudes:

| Bucket | Entries |
|---|---:|
| kernel / quant | 63 |
| KV / cache | 44 |
| speculative decode | 30 |
| parallel / TP | 28 |
| scheduler | 25 |
| memory | 8 |
| launch / graph | 3 |
| unclassified | 115 |

The kernel bucket is the largest and its remaining margin is the smallest.
Three separate kernel-side items were closed as structural rather than
fixable in `2026-08-24-roadmap.md`: W4AFP8 GEMV restructuring (memory-bound at
M=1), M-split GEMV grid (+2–6%), and the NVFP4 c=1 gap against FP8 (structural
to the compute mix). One kernel-side item is open and large:
`2026-07-28-fa3-one-launch-per-layer` measured MoE expert kernels
(`dsv4_fp8_grouped_{down,swiglu}_decode`) at **53.9% of GPU time** after FA3
was fixed. Dense GEMV/GEMM is closing; MoE grouped kernels are not.

The launch/graph bucket has three entries and carries the largest verified
deltas in the tree, in both directions:

| Entry | Change | Measured |
|---|---|---|
| `2026-07-28-fa3-one-launch-per-layer` | one FA3 launch per layer instead of per row | ITL p50 c=16 94.61 → 59.51 ms (1.59×); decode 9.5 → 14.0 tok/s (1.47×); launches 34,212 → 3,049 over the same 35 s; step model `12.7 + 5.12·B` → `14.9 + 2.79·B`, so the per-row marginal fell 1.83× with a flat intercept. c=1 unchanged (16.03 → 16.04 ms) |
| `2026-08-23-dsv4-c1-decode-graph` | CUDA graph on the c=1 decode step | decode +21.3%, ITL p50 −9.0%, ITL p99 −63.7% |
| `2026-08-17-collectives-to-comm-stream` | NCCL moved to its own stream, fenced | **78.7 → 55–59 tok/s, a 25–30% regression.** Cause: 80 all-reduces × 2 fences = 160 `cuEventCreate`/`cuEventDestroy` per step, 3–5 ms/step of host overhead. A partial revert recovered to 63 tok/s, still −19% |

Three entries: two of the largest wins, and one of the largest self-inflicted
regressions. In all three the mechanism was host-side launch bookkeeping, not
device numerics — launch count, capture eligibility, event allocation. None of
them required reading a `.cu` file to diagnose.

**This is the layer with the most remaining margin and no boundary.** It has no
crate, no trait, and no gate. The regression entry is the strongest single
argument in this document: a 25–30% loss shipped through a layer that nothing
guards, and the cost was 160 host allocations per step.

## 3. Six structural findings

### 3.1 "Kernel" names three things that share one crate

| Layer | Size | Authored by | Source of regressions |
|---|---|---|---|
| Device compute — MMA loops, TileLang AOT, vendored FlashMLA / DeepGEMM / DeepEP | 84,728 lines C++ | mostly vendored | rarely |
| Per-kernel host adaptation — repack, scale lift, launcher, dispatch | 20,503 lines Rust FFI | ours | yes |
| Kernel selection | 17 dispatch matches | ours | yes |

All 17 dispatch matches are in `qwen35_load.rs`, `quant_format.rs`,
`loader.rs`, and `dsv4/load.rs` — the load path — and none depends on a device
return value. Kernel selection is a pure function of (model, layer, dtype,
quant kind, shape, SM tier). It is fully CPU-testable today and is not tested
on the CPU today.

### 3.2 Kernel-to-model binding is four stages written as one

A Marlin GEMM does not know which model calls it. What is model-bound is the
operator sequence and the weight layout requirement. The real relation is

```
model → operator sequence → weight layout requirement → kernel selection → launch
```

Only the last is device-side.

The first version of this section said the four stages are collapsed into
`crates/infer-cuda/src/executor/qwen35.rs`. A line-by-line read of all 3,529
lines (2026-09-09) showed that is wrong, and the claim was made from the
file's name and size without reading it. The executor file is ~80% outside
this axis; it delegates through one-line calls such as
`model.forward_tokens_recall(...)`. The stages actually live in the model
modules: `qwen35_forward.rs` (1,024), `qwen35_decode.rs` (383),
`qwen35_spec.rs` (648), `qwen35/dspark.rs` (1,994), `qwen35_load.rs` (3,287)
— about 13,500 lines.

What the executor file holds instead is a junction of six blocks, each of
which already has a destination named elsewhere in this plan:

| Block | Lines | Destination |
|---|---:|---|
| Speculative decode orchestration | ~1,180 | step scheduling, `infer-plan` |
| The `submit` tree | ~670 | step scheduling, `infer-plan` |
| KV / tier / sidecar lifecycle | ~535 | `infer-kvspace`, step 2 |
| Construction and weight setup | ~370 | weight axis, step 5 |
| Launch and OPD surface | ~290 | stays in `infer-cuda` |
| Device scheduling (graph) | ~245 | `device_sched`, step 4 |

Counted per block against `executor/qwen35.rs` at 3,531 lines, comments and
blank lines included. The KV/tier/sidecar block is the one that moved: it
holds `demote_slot` / `promote_slot` / `restore_recurrent_sidecar` and the
pool release/ensure pair as well as the sidecar machinery.

The consequence for the plan is in step 3, which is split in two: the
executor file disperses to homes that already exist, and `infer-model` takes
the model modules, which is where the four stages actually are.

### 3.3 Scheduling has four levels and three names

| Level | Where | Named |
|---|---|---|
| Request — continuous batching, admission, preemption | `infer-core` | yes |
| Step — prefill/decode/mixed, producing `ForwardPlan` | `infer-plan` | yes |
| KV — page/slot allocation, prefix match, tiering | `infer-seam` | yes, across six traits |
| **Device — stream assignment, graph capture, collective placement** | inside `infer-cuda` | **no** |

Only three files in the tree touch graph capture. Graph capture is a
step-scheduling constraint — this step's launch sequence must be isomorphic to
the last — and putting it inside the executor is what allowed
`errors/2026-08-01-decode-graph-flag-is-a-noop-under-paged-kv`: a flag that
silently did nothing because the constraint it depended on lived somewhere the
flag could not see.

Scheduling and kernels have exactly one real coupling: **batch shape decides
kernel selection** (M=1 GEMV, M≥64 GEMM, fixed shape for a captured decode).
The scheduler does not need to know a kernel; it needs to emit a shape. The
type that would carry "the shape is settled" does not exist, which is why host
logic keeps being written on the far side of the seam.

A seam audit of both backends' `submit` (2026-09-09) found a second coupling,
in the opposite direction. `submit` receives `&mut dyn KvPool` — a capability,
not data — and Qwen35 writes back through it: `alloc(slot, delta)` and
`truncate_slot(slot, target)`, at nine call sites on the speculative decode
path. The engine pre-budgets the full speculative chain (#197) and the backend
truncates the over-allocation once the draft model's actual chain length is
known.

The host KV pool therefore has two writers: the engine budgets, the backend
corrects. That is the same shape as the five shared-state failures found in
tooling this session — one mutable resource, several writers, no identity —
and it is the reason the seam cannot be closed by moving reads alone. The
correct end state is one writer: the backend reports the actual length, the
engine applies it. Chain length cannot move above the seam, because it is a
device result.

The same four-stage read of `executor/dsv4.rs` (650 lines) found the same
shape — the same delegation to a model module, the same graph path
(`try_graph_decode_c1`, 114 lines, against `try_graph_decode_paged`, 122), the
same submit tree. The boundary is therefore not overfitted to one model, which
was the open risk in deriving it from qwen35 alone.

DSv4 also carries a working prototype of the type this plan is built around:
`Dsv4DecodeBatch` (`crates/infer-cuda/src/dsv4/kv_contract.rs:14`) is a pure
host batch — slot ids, tokens, start positions, positions, rows — with a KV
view built from it. Qwen35 has no equivalent and rebuilds device-bound
`PageMeta` per call.

Two types are still missing, both on the request axis:

1. **Stage 2's output.** There is no host geometry descriptor.
   `build_prefill_geometry` returns `(PageMeta, Option<Qwen35CpPrefill>)`, both
   device-bound: the host arithmetic (slices, q/k positions, kv indices) is
   uploaded the moment it is computed, so it is never a value anything else can
   see or test. `Dsv4DecodeBatch` is the shape this type should take.
2. **Stage 4's output.** Nothing carries "the shape is settled".
   `dispatch_decode_rows` computes the batched/gate decision and matches on it
   immediately; the capturable test inside `try_graph_decode_paged` is inline.
   Kernel selection has no result type, which is why kernel choice and kernel
   launch cannot be tested apart.

The audit also found that the Metal backend never calls
`KvBatchDescriptor::from_plan` at all; it reads the pool directly
(`slot_epoch`, `seq_len`, `page_size`, `page_indices`). Two independent drift
paths, one cause.

### 3.4 KV is four roles inside one trait group

`KvPool = KvQuery + KvAllocator + KvPrefixStore` (`crates/infer-seam/src/kv.rs:16`),
plus `KvPageTier`, `KvSlotTier`, `DeviceKvFit` as capability traits. Six traits
for one object, because KV is simultaneously:

| Role | Change rate | Needs a GPU to verify |
|---|---|---|
| Capacity resource — how many pages remain | near-static | no |
| Content index — prefix match, radix | active; hybrid recurrent sidecar is current work | no |
| Kernel operand — page table layout the attention kernel reads | tracks the kernel | yes |
| Tiered storage — L2/L3 | new | partly |

Grouping them by "all about KV" puts a near-static concern, an actively
changing one, and a device-bound one behind the same trait. Three of the four
never need a GPU.

Hybrid models already falsified the assumption underneath the grouping —
"KV is the attention KV". Recurrent state is currently carried as a sidecar
bound to the radix block
([`design-theses` note 1](2026-09-02-design-theses.md)). The general type is
sequence state; attention KV is one implementation, recurrent state another,
and paging is an implementation detail of the first.

### 3.5 Fusion is a gate problem, not an architecture problem

41 distinct `fused_*` symbols in `crates/cuda-kernels/`, no fusion mechanism —
each is a hand-written `.cu`. A fusion framework needs an operator graph, and
the tree has none (zero `enum Op` / `struct OpGraph` / `trait Op` definitions).
Building one means rewriting the executors, and it would buy backend plurality
we do not have: 2 backends, against SGLang's 28 attention backends and vLLM's
161-method `Platform` interface.

What is actually missing is the ability to A/B one fused kernel by itself. That
is a gate, and `arle kernel` is 90% of it (see §6, step 0).

Cross-operator **launch merging** is a different thing and belongs to device
scheduling — `fa3-one-launch-per-layer` deleted 31,163 launches without
changing any numerics.

### 3.6 TP is a configuration dimension; PP does not exist

32 files reference NCCL or collectives. Zero reference pipeline parallelism.

No PP requirement has appeared — the measured configuration is 2×H20 TP=8 on a
167 GB model — so PP is not built. The architecture must still not exclude it,
and the only thing that currently would is the assumption "one step = one
full-model forward", baked into `StepOutput`. A step is therefore defined at a
**stage boundary**, which is identical to a model boundary when there is one
stage and costs nothing.

Collectives are scheduling objects, not kernels. The evidence is the
regression in §2: moving NCCL to its own stream cost 25–30%, and the entire
cost was host event allocation. Collectives contend for streams, SMs, and host
time; they do not contend for numerics. They belong on the device-scheduling
interface rather than scattered through operator implementations.

TP leaks upward in exactly three places: weight sharding into the loader, rank
into the KV budget (`tier budget ÷ world`, still pending on the roadmap), and
vocabulary into sampling (`lm_head` vocab-parallel, implemented and deleted at
). Three leaks means TP cannot be sealed inside a backend. It is a
global configuration dimension like dtype: **world and rank are
construction-time parameters, never runtime queries.**

## 4. The criterion

A boundary holds only if the compiler holds it.

> **Criterion: does this crate compile and test on a machine with no CUDA
> toolchain — with `cuda-kernels` absent from `cargo tree`, not merely behind a
> feature gate.**

The criterion used in the first pass of this work was "does this code call a
device function", and it is too weak. It is true only of raw cudarc symbols.
Two ways code passes it and still cannot move:

- Code that holds a device type without calling a device operation.
- A pure host state machine reachable only through an `impl` on a device-owning
  struct. `a0` found both in `spec_decode.rs` and `prefix_pool.rs`: the state
  machines are pure host, but they hang off `impl Dsv4CudaExecutor`.

And one way code fails it that is real and must stay below:
`set_dsv4_verify_frozen` is a host `AtomicBool` that device code reads.

The dependency-graph criterion has none of this ambiguity, is checked by
`cargo tree`, and cannot be satisfied by discipline alone.

## 5. How the target was derived

Five passes. Each one is recorded with what it rejected, because the rejected
versions are the ones that will be proposed again.

**Pass 1 — split by pipeline stage and add the missing `DeviceBatch` type.**
Rejected as incomplete: it names the CPU-to-GPU transition but leaves device
scheduling unnamed, leaves KV's four roles fused, and leaves kernel selection
in the same crate as kernel implementation.

**Pass 2 — split by change rate × verification cost.** Rejected: "change rate"
is a judgement, not a type. It cannot be checked, so it erodes. It was the
implicit criterion that produced the current state.

**Pass 3 — split by whether it compiles without a CUDA toolchain.** Kept. It is
binary, compiler-enforced, and cannot be argued with. It reframes the target:
put as much as possible in crates where `cuda-kernels` does not appear in the
dependency graph.

**Pass 4 — check for over-correction.** If everything host-side moved up,
`infer-cuda` would be roughly its 1,403 device-symbol lines plus device
scheduling. That is not reachable in one pass and not desirable: code that
holds device types has to stay. The target is therefore not a line count. It is
that **new code defaults to above the line, and existing code moves up when a
gate needs it to.**

**Pass 5 — check against the stated core.** The core is kernel execution plus
its scheduling. The target leaves kernel execution below, moves all scheduling
above except device scheduling, and gives device scheduling a name for the
first time — which §2 says is where the margin is. Kernel-to-model binding is
handled by §3.2's four-stage split, which is what makes `qwen35.rs` divisible.

## 6. Target structure

Request axis, per step:

```
infer-protocol   protocol, tokenizer, sampling parameters          no cuda
infer-plan       request scheduling → ForwardPlan                  no cuda    exists
infer-kvspace    capacity + content index + layout description     no cuda    new
infer-model      model → op sequence → kernel selection            no cuda    new
                 → DeviceBatch                                                new type
───────────── above this line, cargo tree contains no cuda-kernels ─────────────
infer-cuda::device_sched   streams, graph capture, collective placement   new module
infer-cuda                 kernel launch
cuda-kernels               vendored and hand-written .cu
```

Weight axis, load time, disjoint from the request axis:

```
Checkpoint → quant-format → QuantSpec → weight-layout (repack plan) → ResidentWeights
              no cuda        no cuda     no cuda                       cuda, copy only
```

The second axis explains why `quant_format.rs` (629 lines) is the cleanest
extraction available: it was never on the request pipeline, so nothing about
step construction constrains it.

**Two new crates, not five.** Device scheduling starts as a module and a trait
inside `infer-cuda`. It needs device types, so extracting it now would yield a
trait and nothing else; it gets extracted when it is stable, or never.

**Crate boundaries follow stages; modules inside may follow models.** A
dsv4-specific byte codec belongs in a `dsv4` module inside `infer-kvspace`, not
in a `dsv4-host` crate. A crate named for a model family commits the tree to a
sibling for every other family, which is the per-model duplication that
`feedback_unified_abstraction_not_per_model` records.

The direct evidence for this rule is `SpecKind`: it is referenced by exactly
two files, `executor/qwen35.rs` and `executor/spec_decode.rs`. One qwen-family
model and one deepseek-family model share the type, so it is a step-scheduling
concept and belongs in `infer-plan`, whose only dependencies are `serde` and
`thiserror`.

## 7. What is deliberately not built

Recorded here so it is not reopened without new evidence:

| Not built | Reason |
|---|---|
| Operator graph | Needed only by a fusion framework; building it means rewriting both executors |
| Fusion framework | Buys backend plurality we do not have (2 backends, not 28). The real gap is a per-kernel A/B gate |
| Pipeline parallelism | No requirement has appeared. Not excluded: step boundary = stage boundary |
| Plugin / backend registry | 2 backends. `BackendExecutor` is 20 methods, 17 with default bodies; a registry adds indirection to a two-element set |
| Splitting `kv-native-sys` | It is already at its boundary |
| Rewrite to C or C++ | Examined and rejected. The host side of an inference engine can be pure C — `ggml-quants.c` uses zero C++ features — but GPU backends cannot, and llama.cpp upstream is 78% C++ by bytes. The proposal's value was diagnostic: it surfaced that the layering, not the language, is what makes iteration slow |
| Type versioning across the seam | One repository, one release train |

## 8. Order

Every step is verifiable alone and leaves no parallel old/new path.

### Step 0a — lift the CPU reference out of the cuda crate, and gate it

`kernel_bench.rs` has its device boundary at `DeviceContext::new()`, line 276.
Everything before it is CPU: argument validation (255), registry lookup (268),
unknown-name bail (274). `cpu_ref_fp4` (line 62) is a pure host dequantize-and-
GEMM.

That function is the differential oracle — the thing a kernel is checked
against — and it is trapped inside the crate that requires a GPU to build. Lift
it and the registry surface to where they compile without cuda, and gate them
in `pre_push_checks.sh`.

This also connects two tracks that are currently separate: the same oracle
checks a kernel's numerics and checks `quant_format`'s parsing of the same
format.

Gate: `cargo test` on a Mac; breaking the reference turns it red.

### Step 0b — wire the device half into the pod flow

`arle kernel fp4-gemv` against `marlin-fp4-gemm` at M=1, into `test_pod_flow.sh`
and `pod-remote-run.sh`.

The whole command must not go into a CPU CI lane. `crates/cli/src/lib.rs:173`
makes `arle kernel` print "requires a cuda build" and exit non-zero under
`not(feature = "cuda")`, so a CPU lane would either fail always or be skipped
always — a gate arm that never runs
(`feedback_gate_arm_that_errors_stops_gating_silently`).

Gate: pod run with a matched A/B, and the wins entry that has been pending.

### Step 1 — `DeviceBatch` on the seam

Define it in `infer-seam`; `submit` takes it instead of `ForwardPlan` plus
`KvBatchDescriptor`.

This moves no code. Its whole value is that the next host function someone
writes inside `infer-cuda` will visibly have its input in a type that lives on
the seam, which makes the drift legible at review time instead of at line 3,322.

Gate: existing seam tests, plus: `submit` no longer names `dyn KvPool`. The
write path narrows to a two-method `KvSlotAccounting` trait rather than keeping
the whole pool, so the residue is countable and Step 1b has a definite target.

### Step 1b — one writer for the host KV pool

Replace the narrowed write capability with returned data: `submit` yields the
adjustment, or it rides on `PollResult`. `PollResult` is the better candidate
because the final accepted chain length is only known after poll anyway, which
merges two corrections into one.

Separate from step 1 because it changes speculative-decode KV accounting, where
an error is KV corruption or a premature free. It is the only step in this plan
that moves logic rather than types.

Gate: needle gate ×3, plus `spec_parity.py` — token-exact equality between the
speculative and greedy arms.

### Step 2 — `infer-kvspace`

Capacity accounting, prefix matching, layout description — three of KV's four
roles. The kernel-operand role stays in `infer-cuda`.

Second because KV has the clearest existing boundary and because
`prefix_pool.rs` is already under judgement.

Gate: `cargo tree -p infer-kvspace` contains no `cuda-kernels`; contract tests
run against both the fake and the real adapter.

### Step 3a — disperse the executor file

`executor/qwen35.rs`'s six blocks each go to a destination that already exists
(the table in §3.2): speculative-decode orchestration and the submit tree to
`infer-plan`, KV/tier/sidecar to `infer-kvspace`, construction to the weight
axis, device scheduling to `device_sched`. About 2,800 of 3,531 lines leave;
550-650 stay.

No new crate. This is the easy 80%, and it follows steps 1, 2 and 5, because
each block's destination has to exist before the block can move.

The dominant obstacle is not the block boundaries but the functions inside
them: the recurring shape is host arithmetic and a device call fused in one
body — `build_prefill_geometry` computes slices and positions and uploads
them, `mirror_host_slot` does shard arithmetic and a device write,
`try_graph_decode_paged` does kernel selection, graph capture and launch.
Each is split along the boundary; the boundary itself does not move.

The blocks move one at a time, cheapest first, one pull request each:

1. **KV / tier / sidecar to `infer-kvspace`.** Its host and device halves are
   already separated — the two-phase commit is in the seam, sidecar
   serialization is off the hot path, and only four calls touch the device
   (`snapshot_recurrent`, `restore_recurrent_from_snapshot`, `mirror_slot`,
   `swap_out` / `swap_in_image`).
2. **Device scheduling to `device_sched`,** together with graph invalidation,
   which has to be stated as one policy rather than left where it is. There
   are six sites of two kinds: five whole-graph (`self.decode_graph = None`),
   one inside `try_graph_decode_paged` itself and four in the launch/OPD
   block; and one per-slot rebuild in `submit_prefill_row`, which replaces a
   single slot's `CudaGraphState` and clears its bake while the rest of the
   graph stands. Construction and the KV block touch neither.
3. **The submit tree and speculative-decode orchestration last.** These two
   are fused: the spec row handlers are the decode path, and
   `dispatch_decode_rows` is the only seam between them. The fusion sits in
   the row handlers, not in `spec_verify_forward` — that function is already
   an orchestration layer that arranges rows and calls `verify_logits`, with
   no accept or rollback arithmetic in it.

Gate: no-cuda build of each destination crate; needle gate ×3.

### Step 3b — `infer-model`

The model modules' host halves — `qwen35_forward.rs`, `qwen35_decode.rs`,
`qwen35_spec.rs`, `qwen35/dspark.rs`, about 4,000 lines — split by §3.2's four
stages. This is where the stages actually are, and it is the real work.

Blocked on the two missing types named in §3.2. Without a host geometry
descriptor and a settled-shape descriptor there is nothing for stages 2 and 4
to produce, and the split would cut through the middle of every function that
computes host arithmetic and uploads it in the same breath.

Gate: no-cuda build; needle gate ×3 against the baseline envelope.

### Step 4 — `device_sched`

Stream assignment, graph capture eligibility, collective placement, as a module
and a trait inside `infer-cuda`. Last, because its gate is on a GPU.

Gate: pod window, matched A/B. The `collectives-to-comm-stream` regression is
its regression test — the same change must be reproducible and its host event
count observable before anything else moves.

### Step 5 — weight axis

`quant_format` extraction, then weight layout. Orthogonal to 1–4; runs in
parallel. Gate: the oracle from step 0a.

Step 0a is the only one that should not wait. Without it, no later step can
demonstrate it did not break a kernel.

## 9. Exit

- `cargo tree` for `infer-plan`, `infer-kvspace`, and `infer-model` contains no
  `cuda-kernels`.
- `arle kernel`'s host half runs in pre-push; its device half runs in the pod
  flow.
- `DeviceBatch` is the only type crossing the seam for a step.
- Each step lands its own CHANGELOG line and its wins or errors entry.

## 10. What this document does not settle

- Whether MoE grouped decode kernels (53.9% of GPU time) are a kernel problem
  or a launch problem. Both are plausible and the measurement is not taken.
- Where sequence state (§3.4) is defined once hybrid models are more than a
  sidecar. `infer-kvspace` is the location; the type is not designed.
- Whether `infer-protocol` is worth separating from `infer-server`. Listed in
  the target for completeness; no evidence yet that it is a boundary.
