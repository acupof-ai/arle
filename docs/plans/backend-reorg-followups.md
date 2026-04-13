# Round 2 backend reorg — follow-up index

Round 2 (April 2026) collapsed the `infer/src/` root from 39 top-level `.rs` files down to 26 and moved every backend-related module under `infer/src/backend/{cuda,metal,cpu,runtime}/`. The reorg was kept strictly structural so every commit stayed reviewable and bisectable. Three debts were consciously deferred; this document is the index tracking each one to closure.

| # | Item | Owner | Status | Plan |
|---|---|---|---|---|
| F1 | Split `backend/metal.rs` (1766 lines) into topical submodules | codex (local Mac) | **in progress** | [`backend-metal-split.md`](backend-metal-split.md) |
| F2 | Audit `backend/cuda/graph_pool.rs` — deliberate scaffold or dead code? | ckl (decision), remote CUDA host (if wire-in) | **awaiting decision** | §F2 below |
| F3 | Document `--features cuda,no-cuda` type-check invocation | me | **done** (commit `4b493c8`) | — |
| Round 3 | Extract `backend`/`ops`/`model`/`scheduler` into `crates/infer-engine` | remote Linux CUDA host | **queued** (starts after F1 lands) | [`cuda-crate-extraction.md`](cuda-crate-extraction.md) |

## Ordering & coordination

```
Round 2 (DONE, pushed)
     ↓
F1 (codex, local Mac)  ──┐
     ↓                    │
     ↓                    F2 decision (ckl, any time — does not block F1)
     ↓                    │
Round 3 (remote CUDA) ←──┘
     ↓
    ship
```

**F1 and Round 3 must be sequential**, not parallel. Round 3 Layer 1 moves `backend/metal/*` wholesale into `crates/infer-engine/`; if F1 is in flight at the same time, every `pub mod` line in `backend/metal.rs` collides and the cross-crate visibility decisions have to be redone. F1 first, Round 3 second.

**F2 can run in parallel** with either F1 or Round 3 because it touches only `backend/cuda/graph_pool.rs` (independent module, no internal consumers). But if Round 3 starts before F2 is decided, the decision has to be re-made in the new location (`crates/infer-engine/src/backend/cuda/graph_pool.rs`). Cleaner to close F2 before Round 3 ships.

---

## F1 · Split `backend/metal.rs`

Owner: codex on local Mac.
Plan doc: [`backend-metal-split.md`](backend-metal-split.md) (self-contained, codex-executable).
Prerequisite: commit `7a876e1` or later (Round 2 housekeeping).

**What it does (summary).** Takes the 1766-line `infer/src/backend/metal.rs` and breaks it into five topical submodules: `weights.rs` (~780 lines, weight types + loading), `generate.rs` (~320, generation loop + KV pool guard), `forward.rs` (~185, per-step graph builders), `ops.rs` (~75, linear / eval / extend_kv_cache), `sampling.rs` (~80, sampling helpers). `metal.rs` shrinks to ~440 lines and becomes a thin facade holding only the struct, the `impl InferenceBackend`/`impl StreamingInferenceBackend` trait blocks, and submodule declarations.

**Exit criteria.** Five atomic commits (one per extract pass) on `main` from `7a876e1` baseline. Each commit passes:
- `cargo check -p infer --no-default-features --features metal`
- `cargo check -p infer --no-default-features --features cuda,no-cuda`
- `cargo test -p infer --no-default-features --features metal --lib -- --test-threads 1`
- `cargo clippy -p infer --no-default-features --features metal -- -D warnings`
- `cargo check -p infer --no-default-features --features metal --bin metal_bench --bin metal_request --bin metal_serve`

No line-count ranges exceeded (see §2 in the plan doc). Zero semantic changes — this is pure relocation.

When F1 lands, update the status row in the table above with the final commit SHA and move on to Round 3 (or F2 if it hasn't been decided yet).

---

## F2 · `backend/cuda/graph_pool.rs` audit

Owner: ckl must make the call; execution depends on the call.
Location: `infer/src/backend/cuda/graph_pool.rs` (441 lines).

### Evidence

**History** (`git log --follow`):

| Commit | Date | Message |
|---|---|---|
| `a95de77` | 2026-03-31 | feat(graph): CUDA graph batch pool — CPU tracking + GPU capture stub |
| `8fbbac7` | 2026-03-31 | chore(polish): code quality, docs, and CI |
| `ad4fda8` | 2026-03-31 | refactor: rename pegainfer → infer |
| `9f64ddf` | 2026-04-13 | refactor(cuda): build backend/cuda house and relocate CUDA plumbing |

**Author:** ckl (same as everything else in this repo). This is not legacy from a departed contributor — it is **your own scaffold**, written 2026-03-31 with explicit intent:

> "CUDA graph batch pool for multi-request decode. POOL_BATCH_SIZES [1,2,4,8,16,32,64,128], pad_to_pool_size(), GraphPool with GraphCaptureState lifecycle (Uncaptured→Capturing→Ready/Failed), dispatch() → DispatchDecision (Graph{padded,actual,padding} | Eager{actual}), warmup_schedule(), capture_decode_graph() GPU stub (todo! GPU required). 14 unit tests for padding math, pool lifecycle, dispatch logic."

**Public surface** (10 items):

```rust
pub const POOL_BATCH_SIZES: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128];
pub const MAX_GRAPH_BATCH_SIZE: usize = 128;
pub fn pad_to_pool_size(n: usize) -> Option<usize>;
pub fn is_graph_eligible(n: usize) -> bool;
pub fn padding_slots(n: usize) -> usize;
pub enum GraphCaptureState { Uncaptured, Capturing, Ready, Failed }
pub struct GraphPool { ... }                    // lifecycle tracker
pub enum DispatchDecision { Graph{..}, Eager{..} }
pub fn warmup_schedule(max_batch: usize) -> Vec<usize>;
pub fn capture_decode_graph<F>(...)             // GPU stub: todo!("GPU required: ...")
```

**Current call graph:** exhaustive grep across `infer/`, `crates/*/src/`, `src/`, `infer/tests/`, `infer/benches/`, `infer/src/bin/` for every one of the 10 public names — **zero hits** outside the file itself. The only references are in the module's own doc comments and 14 unit tests. The scaffold was never wired into `scheduler/` or any decoder.

**Key observation.** The existing `src/scheduler/cuda/` warmup code (see `scheduler/cuda/core.rs`) speaks of CUDA Graph warmup for batch sizes 1–32 and the scheduler already emits warmup batches — but it talks directly to `backend/cuda/ffi` via cudarc, not through `GraphPool`. So `graph_pool` is a **parallel, more-abstracted** CUDA-graph lifecycle tracker that never replaced the direct path.

### The three decisions

#### Option A · **Wire it in** (finish the integration)

**When this is right:** If the scheduler's current CUDA-graph handling is ad-hoc / duplicated / hard to reason about, and `GraphPool`'s `Uncaptured→Capturing→Ready/Failed` state machine would make the scheduler cleaner.

**Work required:**
1. Remote CUDA host task — cannot be done locally (no nvcc, no real graphs).
2. Replace `capture_decode_graph`'s `todo!()` body with the real cudarc calls that exist in `scheduler/cuda/core.rs`.
3. Refactor `scheduler/cuda/decode.rs` to go through `GraphPool::dispatch()` instead of directly deciding Graph vs. Eager based on batch size.
4. Add a `GraphPool` member to the `Scheduler` struct.
5. Verify warmup still compiles every pool size (1..128).
6. Throughput regression check via `scripts/bench_throughput_sweep.py` — this is a hot-path change, any 1% regression is a deal-breaker and means revert.

**Estimated commit count:** 3–5 (scaffold → integration → warmup polish → benchmark snapshot).

**Risk:** medium-high. Scheduler's CUDA graph handling is on the decode hot path. Integration bugs would manifest as incorrect tokens, not crashes, and bisecting an accuracy regression is expensive.

#### Option B · **Park it** (explicit hibernation)

**When this is right:** If the scheduler direction is good as-is and `graph_pool` was a design exploration that proved unnecessary.

**Work required:**
1. Add a module-level `TODO(ckl, 2026-Q3)` comment at the top of `graph_pool.rs` explaining: "Parked scaffold from 2026-03-31. Intended as a more-abstracted lifecycle tracker over the scheduler's direct CUDA-graph handling. Not wired in. Revisit when {X condition}."
2. Downgrade module visibility from `pub mod graph_pool` (in `backend/cuda.rs`) to `pub(crate) mod graph_pool` so external consumers can't accidentally depend on it.
3. Add `#[allow(dead_code)]` at the module level so clippy stops reminding us every build.
4. Update `docs/projects/cuda-graph-pool.md` (NEW) with the above + a 1-paragraph plan for the eventual wire-in.

**Estimated commit count:** 1 (atomic `chore(cuda): park graph_pool with hibernation TODO`).

**Risk:** zero. This is a docs + visibility change only.

#### Option C · **Delete it**

**When this is right:** If you are confident `graph_pool`'s abstraction is wrong (i.e., the real scheduler integration will never look like this, either because cudarc's API has evolved past it or because the `GraphCaptureState` state machine is the wrong shape).

**Work required:**
1. `git rm infer/src/backend/cuda/graph_pool.rs`
2. Remove `pub mod graph_pool;` from `infer/src/backend/cuda.rs`
3. Any references to `GraphPool` etc. in doc comments elsewhere (none expected — already grep-verified).
4. Add the deletion rationale to `docs/experience/errors/2026-MM-DD-graph-pool-dead-scaffold.md` (NEW) — following the postmortem template in `CLAUDE.md`.

**Estimated commit count:** 1 (`chore(cuda): drop unused graph_pool scaffold`).

**Risk:** low once the decision is made. Irreversible unless reverted from history, but trivial to recover from git history if the wire-in direction returns.

### Recommendation

**Option B (Park it)** is the default recommendation unless you have an active plan to wire this up within the next sprint. Reasons:

1. 14 unit tests + the lifecycle state machine encode design thinking that would be expensive to re-derive.
2. The `pad_to_pool_size` / `warmup_schedule` utilities are small and correct — worth keeping in the toolbox even if the larger `GraphPool` abstraction is rethought later.
3. Deleting (C) destroys 3-hours-of-thought in exchange for zero immediate benefit — violates the "don't leave debt" directive by trading one kind of debt (unused code) for another (re-derivation cost).
4. Wiring (A) is a hot-path change and should be done when someone is specifically assigned to the scheduler area, not as a tail-cleanup of a structural refactor round.

If you haven't made the call by the time Round 3 is ready to ship, **default to Option B** and defer the decision to the next scheduler-focused sprint. Mark F2 as "parked" in the status table above and move on.

---

## F3 · `cargo check --features cuda,no-cuda` documentation

**Status: done.** Closed in commit `4b493c8` (`docs: update architecture map for backend/ reorg + log follow-ups`).

`infer/CLAUDE.md` Tests section now documents the invocation:
```bash
cargo check -p infer --no-default-features --features cuda,no-cuda
```
— which enables rustc type-checking of `#[cfg(feature = "cuda")]`-gated code while the `no-cuda` feature makes `build.rs` skip nvcc, allowing Darwin / CI-without-nvcc hosts to validate CUDA-only refactors without a real CUDA toolchain.

This was the single most useful undocumented trick discovered during Round 2. It unblocked the CUDA relocation commit (`42f5d8a`) from being purely-mechanical-and-unverified to being rustc-verified.

---

## Round 3 · Crate extraction

Owner: remote Linux CUDA host.
Plan doc: [`cuda-crate-extraction.md`](cuda-crate-extraction.md) (self-contained, remote-executable).
Prerequisites: Round 2 housekeeping merged ✓, F1 merged, ideally F2 decided (but not strictly required).

**What it does (summary).** Moves `backend/`, `ops/`, `model/`, `scheduler/`, `weight_loader.rs`, `gguf.rs`, `hf_hub.rs`, and several utility modules from the monolithic `infer` crate into `crates/infer-engine`. Secondary moves: `metrics.rs` / `trace_reporter.rs` / `logging.rs` → `crates/infer-observability`; optionally `sampler.rs` / `tokenizer.rs` / `error.rs` → `crates/infer-core`. The `infer` crate shrinks from ~44 000 lines to ~8 000 and becomes the thin http + bin shell it should always have been.

**Why remote.** Layers 1–4 of Round 3 touch CUDA hot paths. Darwin can only verify rustc type-check via `cargo check --features cuda,no-cuda`; it cannot run the real nvcc build, cannot run e2e tests, and cannot run throughput benchmarks. A refactor of this size needs real CUDA integration validation — that's only available on the remote Linux host. Do not attempt Round 3 on Darwin except to prototype.

**Exit criteria.** Six commits (Layer 0 feature prep + Layers 1–5) on `main`, each atomic, each verified locally on Darwin via the rustc-only matrix AND on the remote CUDA host via:
- `CUDA_HOME=/usr/local/cuda cargo build --release`
- `cargo test --release --lib`
- `PEGAINFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release --test e2e`
- `cargo test --release --test e2e_qwen35`
- `scripts/bench_throughput_sweep.py --label round3-layer-N-after` with ≤ 1% throughput delta vs. the pre-Round-3 baseline snapshot.

**Coordination note.** Round 3 is a large refactor — 5 atomic layers, potentially 500+ files touched across layers. Do not start without:
1. F1 merged (confirmed via the F1 status row above).
2. F2 decision made OR an explicit note saying "F2 defer, graph_pool carries over".
3. A pre-Round-3 throughput sweep snapshot committed to `docs/experience/wins/YYYY-MM-DD-bench-pre-round3.md` — so regression detection is possible.
4. A clean `origin/main` — no other PRs open that would conflict with the crate boundary.

---

## Template for closing this document

When all four items are done:

1. Update each status row with the final commit SHA.
2. Move this file to `docs/archives/backend-reorg-followups.md` — it's historical at that point.
3. Write a single win entry at `docs/experience/wins/YYYY-MM-DD-backend-reorg-complete.md` summarising the four-part journey: Round 2 structural → F1 metal split → F2 graph_pool decision → Round 3 crate extraction.

Until all four are done, this file stays at its current path as the live index.
