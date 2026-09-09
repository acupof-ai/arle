# The backend seam is a cost contract

Design note 3 of 5 ([plan](../plans/2026-09-02-design-theses.md)). CUDA, Metal,
and the CPU smoke path. Date: 2026-09-09.

## Problem

An inference runtime serves more than one backend — here CUDA and Metal, with a
CPU path for machines with no accelerator — and features land continuously:
prefix reuse, tiered KV, OPD weight residency, multimodal generation. Every
feature is wanted on one backend first. The structural problem is keeping the
other backend's *compile* honest: either the other backend silently lacks the
behavior, or the scheduler grows a per-backend branch and the backends stop
being interchangeable. The workload that exposes the design is the engine
itself: one `Engine<E, K>` in `infer-core` drives every backend, and a new
backend must be addable without a scheduler change.

## Standard practice and where it fails

The standard shapes are a worker / model-runner hierarchy (vLLM's Worker and
ModelRunner) or a backend op registry. In both, a new feature is a new method
with a default implementation, and the default is usually empty. That is what
this codebase had: `BackendExecutor` grew to 49 methods, 46 defaulted, many to
`{}`. A new feature added a method, the other backend compiled, and it silently
lacked the behavior. The failure that named the pattern was the
sampling-penalty drop: penalties vanished on the backend that had not
implemented the method, and nothing at compile time or load time said so. A
second failure mode was the silent no-op flag: `--kv-oversubscription` on a
backend without a slot tier parsed, ran, and did nothing.

## The design

**The seam is the engine's cost contract.** `BackendExecutor`
([`infer-seam/src/lib.rs:339`](../../crates/infer-seam/src/lib.rs)) is
submit/poll plus three things: `step_limits()` (one struct of per-step cost
parameters, replacing six limit methods), `stats()`, and capability accessors
that default to `None`. The engine plans from costs and calls only capabilities
that are present; an accessor returning `None` is a written opt-out, and the
engine substitutes the documented inert behavior at the call site.

**Capability traits carry zero default bodies.** `PrefixReuse`, `KvPageTier`,
`KvSlotTier`, `DeviceKvFit`, `WeightResidency`, `MultimodalGenerate` have no
default implementations. A backend that returns `Some` for an accessor is
compiler-forced to implement the whole capability; one that lacks it returns
`None`. A runtime flag requesting an absent capability fails at engine
construction with the flag named — that is where the `--kv-oversubscription`
silent no-op went to die.

**KV is one host-only trait with three faces.** `KvPool`
([`kv.rs:16`](../../crates/infer-seam/src/kv.rs)) is `KvQuery` + `KvAllocator`
+ `KvPrefixStore`: what pages exist, hand me pages, store prefix content. The
engine's KV accounting is host-side page ids and lengths, so the same
`HostPagedKvPool` serves every backend, and serves as the CPU smoke path: the
CPU backend is the feature-free placeholder `MetalExecutor`
([`infer-metal/src/executor.rs:272`](../../crates/infer-metal/src/executor.rs))
over `HostPagedKvPool`, wired as `Cpu(ServeInferenceEngine<MetalExecutor,
HostPagedKvPool>)` ([`infer-api/src/loaded.rs:524`](../../crates/infer-api/src/loaded.rs)).

**A family whose decode is not submit/poll shaped forks the loop.** The
precedent is `BufferedDiffusionExecutor`
([`infer-seam/src/diffusion_executor.rs`](../../crates/infer-seam/src/diffusion_executor.rs)):
a block-diffusion model keeps its own per-slot generation buffer behind the
same submit/poll surface, and the engine still sees one token per step. The
trait is not bent to fit the family; the family gets its own executor.

**Why two traits suffice, and what would justify a third.** Everything the
scheduler plans against is one of two kinds: a *cost* (a number it plans from —
token, slot, or byte budgets) or a *capability* (a behavior it may call). Costs
are fields on `StepLimits`; capabilities are accessors on `BackendExecutor` or
traits behind one. Growth in either kind adds a field or an accessor, not a
trait. `KvPool` is the second trait because KV is the one resource the engine
admits against that is not the executor: the scheduler allocates pages, so page
ownership is a contract the engine holds, not a capability it asks for. A third
trait is justified only by a second *resource the engine admits against* — a new
axis of admission, the way KV capacity is one today. A new behavior is not
that, and neither is a new model family: behaviors go behind accessors,
families fork the loop. The test is "would the engine's admission logic plan
against this?" If no, it does not get a trait.

## The failure

The 49-method trait was the failure, and it shipped for months before the
2026-08-14 refactor contracted it to 15
([wins 2026-08-14](../experience/wins/2026-08-14-seam-cost-contract-refactor.md)).
The refactor was zero-behavior-change by construction: every call site's `None`
arm reproduces the deleted default verbatim, audited per site, and the one
intended behavior change was the `--kv-oversubscription` load-time failure
above. The sampling-penalty drop that motivated the rule was never bisected to
a single commit — the trait shape allowed it, and the shape is what was fixed.

## The number

Two numbers. The seam is 15 methods today, down from 49; the capability traits
carry 9/11/3/1/5/2 methods with no defaults, every one compiler-checked on the
backends that claim it. The reproduction for this note: `cargo test -p
infer-core --release` — 16 engine tests, 16 passed, 0 failed, run 2026-09-09 on
this Mac with no accelerator present. (The plan's printed command carries
`--features cpu,no-cuda`, which this crate does not have; the device-neutral CI
lane runs the same tests with no feature flags.) The engine tests exercise the
production host pool, not a test double — the planner tests build
`HostPagedKvPool::new(4, pages, 16)` ([`planner.rs:600`](../../crates/infer-core/src/planner.rs))
behind a `MockExecutor`.

## What would be done differently

Start from the cost contract on day one. The 49-method trait grew method by
method, each addition reasonable at the time; the rule "costs are struct
fields, capabilities are accessors, absence fails at load" was written down
only after the sampling-penalty drop. The one remaining core-trait method that
returns data rather than opting out — `model_stop_token_ids` defaulting to an
empty vec ([`lib.rs:363`](../../crates/infer-seam/src/lib.rs)) — would be an
accessor too, so that "no model defaults" is a written `None` the same way "no
prefix reuse" is.
