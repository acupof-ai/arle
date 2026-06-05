# Elegant Rust for ARLE — borrow-and-optimize

**Date:** 2026-06-05. **Provenance:** 5-agent research workflow (Rust API
Guidelines · exemplar codebases · error/ownership/type patterns · a scan of the
ARLE infer crates) → synthesis. **Status:** advisory. Six load-bearing claims
were spot-checked against source and confirmed real (`infer-topo::TopoError` /
`infer-moe::MoeError` String-wrappers, `train/src/control.rs:44 kind: String`,
`execution.rs deliver_completions` per-tick `c.clone()`, `infer-plan::ForwardMode`
spec-decode variants, **zero `#[non_exhaustive]` in any in-scope crate**). Exact
line numbers may drift — symbols are the anchor; verify against source before
editing ([[feedback_docs_are_not_truth]]). `infer-cuda` / `cuda-kernels` were
out of scope (Codex's live DSv4 tree).

> Apply order is the §2 borrow-list ROI ranking. Each landed item is its own
> small commit; hot-path items (§4) need a bench entry per the bench contract.

---

## 1. Verdict

For a perf-sensitive inference runtime, "elegant Rust" is not aesthetic polish — it is **converting runtime invariants into compile errors at zero hot-path cost**. Every top pattern (newtype IDs, sealed traits, typed enums, RAII guards) takes a bug class ARLE already fights in production — wrong-data-fed-to-path (`feedback_validate_comparison_inputs_before_bug.md`), slot/buffer reuse-after-free (`reference_disabled_event_tracking_premature_buffer_free.md`) — and makes it unrepresentable. The internal scan confirms ARLE's *newest* crates (`infer-core`, `infer-seam`, `infer-plan`, `infer-server`) are already strongly idiomatic: `RequestHandle(u64)` / `SessionId(Arc<str>)` newtypes, the `KvPool: KvQuery + KvAllocator + KvPrefixStore` sub-trait split with blanket impl, pervasive `#[must_use]`. The divergences are **maturity-of-contract issues concentrated in the older `train`/`chat`/`agent` crates plus two cross-cutting gaps** — not codebase-wide rot, and not beginner-grade problems. The single highest-leverage move is **propagating `infer-core`'s conventions backward** into the legacy crates, plus one near-zero-churn workspace-wide fix: `#[non_exhaustive]` is **absent from every in-scope crate** despite ~20 carrying `thiserror` enums and several public cross-crate seam enums.

The two highest-correctness-ROI items are the **stringly-typed error/record types** (where a downstream `.contains()` or `.get("magic_key")` will eventually cause a silent behavior bug); the two highest-perf-ROI items are **borrow-before-clone on per-tick paths** and **RAII guards on KV-block leases** (the documented premature-free footgun).

---

## 2. The borrow-list (prioritized)

| Pattern | Elegant form (1-line) | Anti-pattern it kills | ARLE site to apply | ROI | Risk |
|---|---|---|---|---|---|
| **Structured error enums** | `#[derive(thiserror::Error)] enum` with variants + `#[from]` | stringly errors / `.contains()` matching | `infer-topo/src/error.rs:TopoError`, `infer-moe/src/error.rs:MoeError` | **high** | low — `Display` stays byte-identical |
| **`FromStr::Err` = typed enum** | `type Err = ArgError` not `String` | String-as-error, defeats `?`-into-typed | `train/src/cli_args.rs:105 BackendChoice`, `:150`, `train/src/model_family.rs:14` | **high** | trivial — `ArgError` already exists in same file |
| **Closed-set enum over `String` kind** | `enum RecordKind` + `#[serde(rename_all)]` | stringly dispatch, `_ => {}` swallows typos | `train/src/control.rs:44/:233 kind`, `chat/src/lib.rs:279 role`, `train/src/teacher_infer.rs:529 dtype` | **high** | low — wire format unchanged |
| **`#[non_exhaustive]` on seam enums** | one attribute line | SemVer-fragile public enum, additive break | `infer-plan::{ForwardMode,FinishReason}`, `infer-seam::AdmissionVerdict`, `infer-topo::ParallelLinearKind`, `qwen3-spec::{Shard,RopeScalingConfig}` | **high** | none — purely additive |
| **Typed struct over parallel maps** | named `Option<f32>` fields + free-form extras map | schemaless `BTreeMap<String,_>` `.get("loss")` | `train/src/control.rs:50-54 TrainingRecord`, `infer-api/src/types.rs:256,270` | med-high | low — split known from dynamic |
| **Borrow / move-out, not clone** | `take_completed(h) -> Option<_>` (move out of map) | `.clone()` to appease borrowck on per-tick path | `infer-server/src/execution.rs:955-959 deliver_completions` | med-high (perf) | **needs bench entry per §0** |
| **RAII Drop guard + `#[must_use]`** | `KvLease` with `Drop`, bind to `let _g =` | leak-on-error-path, ignored handle | (general) KV-block leases, NVTX scopes | med-high (correctness) | med — `Drop` not async, timing |
| **Generational slot key** | `SlotKey { idx, gen }` over bare `usize` | ABA / stale-key aliases live KV | (general) scheduler slot table; mirror `slab`/`slotmap` | med-high | med — adds `.get()` friction |
| **Pass struct, not 11 `Option` args** | `req.into_sampling()` spreading `..default()` | long param list + `#[allow(too_many_arguments)]` | `infer-server/src/schema.rs:138-166 sampling_params` (+ dup wrappers 51-65/101-115) | med | low — request struct already holds fields |
| **Shared error enum + `#[from]`** | one `CliError` with `#[from] ArgError` | copy-paste error enums + `Custom(String)` | `train/src/bin/*` per-binary `CliError` | med | low — doc says they overlap |
| **Flags struct over `&[(&str,bool)]`** | `struct StatusFlags { started, finished }` | bool-soup / boolean trap | `train/src/control.rs:183 record_status` | med | trivial |
| **`chunks_exact` over index math** | `gate_logits.chunks_exact(n).map(...).collect()` | manual `t*n..(t+1)*n` slicing, off-by-one | `infer-moe/src/route.rs:278 route` | low | trivial |
| **Sealed backend trait** | supertrait `private::Sealed` bound | "please don't implement" doc comment | (general) `server_engine::InferenceEngine` | low-med | none — 2 known impls |
| **`cfg_cuda!` macro over scattered `#[cfg]`** | one declarative gate per backend | `#[cfg]` sprawl, `--all-features` drift | (general) backend gating across crates | low-med | low |
| **`Option<NonZeroU32>` niche IDs** | 4-byte `Option<Idx>`, never-zero invariant | `0`/`u32::MAX`-as-sentinel, 8-byte `Option` | (general) KV-block/slot index tables | low-med | med — `.get()`/`-1` friction |

---

## 3. Top 5 deep-dives

### 3.1 Structured error enums — kill the `String`-wrapper "error"

ARLE has *two* `thiserror`-named types whose entire payload is one opaque `String`, with hand-rolled `bail!` macros. The doc comment is the smell itself: *"Messages are ported verbatim from the legacy `bail!` strings so substring-matching callers behave identically."* Substring-matching callers = `if err.contains(...)`, the canonical brittle anti-pattern.

```rust
// before — crates/infer-topo/src/error.rs
pub struct TopoError(pub(crate) String);
bail!("rank ({rank}) must be < world_size ({world_size})");
// caller is forced into: if e.message().contains("world_size") { ... }  ← stringly

// after — structured, matchable, Display-identical via thiserror
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum TopoError {
    #[error("rank ({rank}) must be < world_size ({world_size})")]
    RankOutOfRange { rank: usize, world_size: usize },
    #[error("world_size must be >= 1")]
    ZeroWorldSize,
}
```

**Why better.** Callers branch on a *variant* (e.g. `RankOutOfRange` → clamp-and-retry vs propagate), per C-GOOD-ERR / C-CALLER-CONTROL. The `#[error("...")]` string keeps `Display` byte-identical, so existing log output is unchanged — only the *matching* surface improves.

**Tradeoff.** `thiserror` is a pure proc-macro — generates exactly the impls you'd hand-write, **zero runtime cost**. Boundary rule: `thiserror` at library seams (`infer-core`, `infer-cuda`, `infer-metal`, `infer-topo`, `infer-moe`); `anyhow` *only* at the binary edge (`arle`, `chat`, `agent` glue) — never in a library's public signature. **Non-obvious nuance:** watch `size_of::<Result<T,E>>()` on the decode/prefill return path — a fat variant pessimizes the *success* path (`clippy::result_large_err`); box the rare-large variant (`Cuda(Box<CudaError>)`).

### 3.2 Closed-set enum over stringly `kind`/`role`/`dtype`

A semantically-closed set carried as `String` and re-parsed at every match. The `_ => {}` arm silently swallows typos; the field can be constructed with any string.

```rust
// before — crates/train/src/control.rs:44 / :233
pub kind: String,   // "run_start" | "run_end" | "status" | "metric"
match record.kind.as_str() { "run_start" => guard.started = true, _ => {} }

// after
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordKind { RunStart, RunEnd, Status, Metric }
match record.kind { RecordKind::RunStart => guard.started = true, _ => {} }
```

**Why better.** The match becomes **exhaustive at compile time** — add a variant and the compiler flags every unhandled arm, which the current `_ => {}` actively hides. `#[serde(rename_all)]` keeps wire bytes identical. The `dtype` case (`teacher_infer.rs:529`, matching `"f32"/"bf16"`) is most load-bearing: a typo'd `"bf16"` today silently falls to the error arm instead of being unrepresentable.

**Tradeoff.** Enum is `Copy`, smaller than `String`, no per-record alloc, no re-parse. Apply only to *closed* vocabularies (`kind`/`role`/`dtype` all are); keep `String` for genuinely user-extensible tags.

### 3.3 Borrow / move-out, not clone, on the per-tick delivery path

`deliver_completions` clones every `CompletedRequest` (each carrying two `Vec<u32>`) on every delivery scan, because the borrow of `engine` can't outlive the `pending.remove`. The design (`Streamers = Rc<RefCell<…>>`, single-thread) is *correct*, but the clone-per-scan is avoidable and runs every tick over all pending handles — O(pending) `Vec` clones per scheduler tick.

```rust
// before — crates/infer-server/src/execution.rs:955-959
pending.keys().copied()
    .filter_map(|h| engine.completed(h).map(|c| (h, c.clone())))  // clones 2× Vec<u32> per scan
// after — engine exposes move-out; delivery drains instead of clones
pending.keys().copied()
    .filter_map(|h| engine.take_completed(h).map(|c| (h, c)))      // moves out of `completed` map
```

**Why better.** "Borrow, don't clone" is core idiomatic Rust; moving out of an owning map is the zero-copy form. An accidental clone on a per-token/per-tick path is the single biggest *perf* lever — it silently moves the bench numbers §0 obsesses over.

**Tradeoff — hypothesis, not a free win.** Verify the contract first: if a completion must stay in `completed` for a later `completed(handle)` lookup, moving it out breaks that. Per §0 and `feedback_bench_every_change.md`, this needs a bench entry before landing — flag as hypothesis until a matched A/B confirms the delta.

### 3.4 RAII Drop guard + `#[must_use]` for KV-block leases

ARLE's continuous-batching scheduler leases KV blocks; the error path is exactly where blocks leak. A `Drop` guard makes leak-on-error *structurally impossible* — cleanup runs on early `?` and on panic.

```rust
#[must_use = "leasing a KV block without binding it leaks the block"]
struct KvLease { block: BlockIdx, pool: *mut KvPool }
impl Drop for KvLease {
    fn drop(&mut self) { unsafe { (*self.pool).release(self.block); } }  // runs on ? / panic
}
let _g = nvtx_push("decode");   // _g, NOT _ — `let _ =` drops at the `;`, defeating the guard
```

**Why better.** `std`'s `MutexGuard`/`File` are the exemplars (C-MUST-USE). `#[must_use]` turns "you discarded the lease" into a *compile warning* instead of a heisenbug. The `let _g =` vs `let _ =` distinction is load-bearing: `let _ =` drops immediately, closing the range at the semicolon.

**Tradeoff.** `Drop` can't be `async` and can't return errors — for KV-tier flush that must `await` or can fail-report, use explicit `commit()`/`close()` + `#[must_use]`. Critically, `reference_disabled_event_tracking_premature_buffer_free.md` is the *counter-case*: when `DeviceContext` disables cudarc event-tracking and frees at Rust last-use, automatic `Drop` timing is exactly what bites — there the deliberate substitute is the forward-level keepalive `Vec`. So: RAII `Drop` for the *control-plane* lease/scope lifecycle; explicit keepalive for the *GPU-async* buffer lifecycle where free must land at a precise sync point.

### 3.5 `#[non_exhaustive]` on cross-crate seam enums

`grep` shows **zero** `#[non_exhaustive]` in any in-scope crate, against public enums *designed as evolving runtime contracts*: `infer-plan::ForwardMode` already grew spec-decode variants (`TargetVerify`/`DraftExtend`); `RopeScalingConfig` grows as models land. Without the attribute, every downstream `match` must be exhaustive, so adding a variant is a breaking change across all crates.

```rust
#[non_exhaustive]
pub enum ForwardMode { Prefill, Decode, Mixed, Idle, TargetVerify, DraftExtend }
// downstream is forced to write a `_ =>` arm now, so a future variant compiles cleanly
```

**Why better.** RFC 2008 / C-STRUCT-PRIVATE: lets a seam enum gain variants as a non-breaking change. Highest *clarity-per-churn* item — one line on ~7 enums, directly serving the multi-crate seam architecture `infer-plan`/`infer-seam` exists to provide. The std/`http`/`tokio`/`serde` pattern for "this will grow."

**Tradeoff / when NOT.** It forces a `_` arm even on your own re-exports, which can *hide a genuinely-missing variant* you'd want flagged. Apply per-enum judgment: yes on open contracts (`ForwardMode`, `RopeScalingConfig`, `AdmissionVerdict`); **no** on enums where you *want* the compiler to force every backend to handle every case — e.g. a `Dtype { F32, Bf16, Fp8 }` where a new dtype *must* break every kernel match (the attribute would hide that real breakage). `FinishReason` is a judgment call.

---

## 4. Hot-path caveats — elegant patterns that are WRONG on the decode/forward path

- **`.clone()` / `.to_owned()` of tensors, KV blocks, or token `Vec`s.** A stray clone in the inner loop is a heap copy per token and silently corrupts bench numbers. Weights `&self` (immutable, `Arc`-shared, clone = one atomic add, never a data copy); per-request mutable state in `State`. Use `Arc<T>` *not* `Arc<Mutex<T>>` for read-mostly weights.
- **`fn foo() -> Vec<T>` per tick** allocates on the scheduler hot path. Return `impl Iterator<Item=…>` (lazy, zero-alloc, swappable impl) — `iter().enumerate().filter(...).map(SlotId)` compiles to the hand loop with bounds checks elided. **Caveat-on-the-caveat:** `impl Iterator` return can't be named/stored and leaks `Send`/`Sync` implicitly — across an async/thread boundary use a named type or `Box<dyn Iterator>`.
- **`_into` discipline must be extended, not dropped.** `infer/src/ops/AGENTS.md` already mandates caller-supplied-buffer `*_into` variants (std `Read::read`, polars/ndarray idiom). In a per-token loop, returns-fresh ops are death-by-allocation; offer both, returns-fresh implemented on top of `_into`.
- **`Cow<'a, T>` only after profiling.** Right for prompt/template normalization (99% pass-through, alloc only on rare rewrite). Wrong as a reflexive "maybe I'll mutate" hedge — infects signatures with a lifetime and forces `.into_owned()` at consumers.
- **`Drop` for GPU-async buffer free.** Where free must land at a precise CUDA sync point, automatic `Drop` timing is the footgun (`reference_disabled_event_tracking_premature_buffer_free.md`) — use the explicit forward-level keepalive `Vec`.
- **`thiserror` fat variants** inflate `Result<T,E>` and pessimize the *success* return on every hot-loop `Ok`. Box the rare-large variant; watch `clippy::result_large_err`.
- **`unsafe impl Send` without a written `// SAFETY:`.** The unavoidable FFI raw ptr (MLX/cudarc handle) needs `Send` to cross the scheduler — but document the invariant, or it's how the FFI session races happen (`feedback_ffi_session_owns_data.md`).

---

## 5. Explicitly NOT worth it

- **Const generics for model dims.** Model `hidden_size` is *runtime* config (Qwen3.5-0.8B vs Qwen3.6-35B). `const N` forces per-model monomorphization — absurd, plus icache/compile bloat and forced dynamic-dispatch escape hatches. Const generics fit *intra-kernel* fixed quantities only (SIMD lane width, fixed `head_dim`, tile dim).
- **Typestate everywhere.** Right for a *linear, statically-known* lifecycle where "forgot a required step" is catastrophic (scheduler built without a KV pool; quant path missing scale tensors). Wrong when state is *data-determined at runtime* (which expert a token routes to — a value, not a type) or when you must store heterogeneous-state objects in one `Vec`. Don't typestate a 2-field config.
- **`uom` / 12 hand-rolled arithmetic newtypes.** Newtype the IDs that get *confused* (token vs block vs byte offset — the `[16]-vs-11111` class). Don't newtype a quantity used in one function with no same-typed sibling; don't half-implement `Add` (a bare `u32` beats a broken `Add`).
- **salsa incremental-recompute, sled lock-free log, `crossbeam-epoch`.** Heavyweight, transferable only with a *measured* contention/incremental-recompile problem. Adopting speculatively violates `feedback_no_speculative_interface_shaping.md`.
- **Sealing pure-data or open-extension traits.** Seal the backend `InferenceEngine` (closed `{Cuda, Metal}` universe, want additive trait methods). Do *not* seal a tokenizer/model-registry plugin trait you *want* third parties to implement.
- **`cfg_cuda!` macro for a handful of lines.** Introduce it only once a feature gates ≥several items across ≥several files (tokio's threshold).

---

## 6. Sources

- **Rust API Guidelines** (rust-lang.github.io/api-guidelines/) — C-NEWTYPE, C-CUSTOM-TYPE, C-SEALED, C-CONV / C-CONV-TRAITS, C-GOOD-ERR, C-CALLER-CONTROL, C-COMMON-TRAITS, C-SEND-SYNC, C-DEBUG / C-DEBUG-NONEMPTY, C-MUST-USE, C-STRUCT-PRIVATE, C-BUILDER.
- **RFC 2008** `#[non_exhaustive]`; **RFC 2000** const generics.
- **dtolnay** — `thiserror` (library errors), `anyhow` (binary errors); `bon` / `typed-builder`.
- **Alexis King**, "Parse, don't validate" (2019).
- **Exemplar codebases:** rust-analyzer `la-arena` (`Idx<T>`); `id-arena` (`#![forbid(unsafe_code)]`); cranelift `entity_impl!`; `slab` (free-list slot map); `slotmap` / `generational-arena` (ABA-safe keys); `bytes` (`&'static Vtable` zero-copy); tokio `src/macros/cfg.rs` (`cfg_*!`); `bitflags`; rustls `ConfigBuilder<State>` (typestate); ripgrep / BurntSushi (typed errors + perf-without-`unsafe`); serde (zero-copy `#[serde(borrow)]`, `flatten`).
- **Rust Performance Book** + Jon Gjengset, *Rust for Rustaceans* — `Arc<Mutex<_>>` smell; borrow-before-clone.
- **clippy:** `result_large_err`, `too_many_arguments`.
- **ARLE memory:** `feedback_validate_comparison_inputs_before_bug.md`, `reference_disabled_event_tracking_premature_buffer_free.md`, `feedback_ffi_session_owns_data.md`, `feedback_no_speculative_interface_shaping.md`, `feedback_bench_every_change.md`.
- **ARLE symbols:** `infer-topo/src/error.rs:TopoError`, `infer-moe/src/error.rs:MoeError`, `train/src/cli_args.rs:105/:150 BackendChoice`+`ArgError`, `train/src/model_family.rs:14`, `train/src/control.rs:44/:50-54/:183/:206-235/:233`, `chat/src/lib.rs:279`, `train/src/teacher_infer.rs:529`, `infer-api/src/types.rs:256,270`, `infer-server/src/schema.rs:138-166`+wrappers `:51-65`/`:101-115`, `infer-server/src/execution.rs:955-959 deliver_completions`, `infer-moe/src/route.rs:278 route`, `infer-plan::{ForwardMode,FinishReason}`, `infer-seam::AdmissionVerdict`+`KvPool` sub-trait split, `infer-topo::ParallelLinearKind`, `qwen3-spec::{Shard,RopeScalingConfig}`, `infer-core::{RequestHandle,SessionId,BlockId}`.