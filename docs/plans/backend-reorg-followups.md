# Round 2 backend reorg — follow-up index

Round 2 (April 2026) collapsed the `infer/src/` root from 39 top-level `.rs` files down to 26 and moved every backend-related module under `infer/src/backend/{cuda,metal,cpu,runtime}/`. The reorg was kept strictly structural so every commit stayed reviewable and bisectable. Three debts were consciously deferred; this document is the index tracking each one to closure.

| # | Item | Owner | Status | Plan |
|---|---|---|---|---|
| F1 | Split `backend/metal.rs` (1766 lines) into topical submodules | codex (local Mac) | **done** — commits `64c0baa` / `32875b2` / `19a433d` / `f59238c` | §F1 below |
| F2 | Audit `graph_pool.rs` — deliberate scaffold or dead code? (now at `crates/cuda-kernels/src/graph_pool.rs` post `a4e12f5`) | ckl (decision), remote CUDA host (if wire-in) | **parked** (`Option B`) | §F2 below |
| F3 | Document `--features cuda,no-cuda` type-check invocation | me | **done** (commit `4b493c8`) | — |
| Round 3 | Extracted-runtime split | remote Linux CUDA host | **reverted** (2026-04-15, Route-A) | [`../archives/cuda-crate-extraction.md`](../archives/cuda-crate-extraction.md) |

## Ordering & coordination

```
Round 2 (DONE, pushed)
     ↓
F1 (codex, local Mac)  ──┐
     ↓                    │
     ↓                    F2 decision (ckl, any time — does not block F1)
     ↓                    │
Round 3 (reverted by Route A) ←──┘
     ↓
    ship
```

**F1 and Round 3 were sequential**, not parallel. That dependency no longer
matters for current work because Route A reverted the extracted-runtime split
before it ever shipped.

**F2 can still run in parallel** with follow-up backend work because it touches
only `graph_pool.rs` (independent module, no internal consumers; post
2026-04-15 the file lives at `crates/cuda-kernels/src/graph_pool.rs`).

---

## F1 · Split `backend/metal.rs`

Owner: codex on local Mac.
Plan doc: retired — the original `docs/plans/backend-metal-split.md` was
archived after execution and then pruned 2026-04-15 once every reference
from active docs was replaced with commit refs. Historical execution rationale
is recoverable via `git log` on the commits below if needed.
Prerequisite: commit `7a876e1` or later (Round 2 housekeeping).

Status: done on 2026-04-13.
Final style checkpoint: `19a433d` (`style(metal): rustfmt post-split + line count verification`).

Final layout:

```
infer/src/backend/metal.rs
infer/src/backend/metal/weights.rs
infer/src/backend/metal/generate.rs
infer/src/backend/metal/forward.rs
infer/src/backend/metal/ops.rs
infer/src/backend/metal/sampling.rs
```

Final line counts:
- `metal.rs`: 505
- `weights.rs`: 679
- `generate.rs`: 307
- `forward.rs`: 182
- `ops.rs`: 62
- `sampling.rs`: 62

Local Metal regression benchmark against a pre-split baseline on `mlx-community/Qwen3.5-4B-MLX-4bit`
(`prompt=128`, `generation=128`, `warmup=1`, `runs=2`) passed:
- `prompt_tps`: `705.6 -> 726.3`
- `generation_tps`: `77.0 -> 80.1`
- `ttft_ms`: `181.4 -> 176.3`

**What it does (summary).** Takes the 1766-line `infer/src/backend/metal.rs` and breaks it into five topical submodules: `weights.rs` (~780 lines, weight types + loading), `generate.rs` (~320, generation loop + KV pool guard), `forward.rs` (~185, per-step graph builders), `ops.rs` (~75, linear / eval / extend_kv_cache), `sampling.rs` (~80, sampling helpers). `metal.rs` shrinks to ~440 lines and becomes a thin facade holding only the struct, the `impl InferenceBackend`/`impl StreamingInferenceBackend` trait blocks, and submodule declarations.

**Exit criteria.** Five atomic commits (one per extract pass) on `main` from `7a876e1` baseline. Each commit passes:
- `cargo check -p infer --no-default-features --features metal`
- `cargo check -p infer --no-default-features --features cuda,no-cuda`
- `cargo test -p infer --no-default-features --features metal --lib -- --test-threads 1`
- `cargo clippy -p infer --no-default-features --features metal -- -D warnings`
- `cargo check -p infer --no-default-features --features metal --bin metal_bench --bin metal_request --bin metal_serve`

No line-count ranges exceeded. Zero semantic changes — this was pure relocation.

When F1 lands, update the status row in the table above with the final commit SHA and move on to Round 3 (or F2 if it hasn't been decided yet).

---

## F2 · `graph_pool.rs` audit

Owner: ckl must make the call; execution depends on the call.
Location: `crates/cuda-kernels/src/graph_pool.rs` (~441 lines, post
`a4e12f5 refactor(cuda): extract cuda-kernels api`). Analysis below
predates the 2026-04-15 kernel-crate extraction and refers to the file's
former location `infer/src/backend/cuda/graph_pool.rs`; the decision text
is preserved for historical record.

Status: parked via Option B. `graph_pool.rs` now lives inside
`cuda-kernels` as an internal scaffold, explicitly marked
hibernating until scheduler work resumes. Because the file moved to the
kernel crate, Option B's visibility-downgrade action item is no longer
directly applicable — the kernel crate's `prelude.rs` / `lib.rs`
re-export surface is the new lever.

### Evidence

**History** (`git log --follow`):

| Commit | Date | Message |
|---|---|---|
| `a95de77` | 2026-03-31 | feat(graph): CUDA graph batch pool — CPU tracking + GPU capture stub |
| `8fbbac7` | 2026-03-31 | chore(polish): code quality, docs, and CI |
| `ad4fda8` | 2026-03-31 | refactor: rename infer → infer |
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

> **Reversal note (2026-04-15):** Route A executed the opposite move. The
> experimental extracted-runtime split was rolled back and the four
> pseudo-independent crates were folded back into `infer`. The historical plan
> is archived at [`../archives/cuda-crate-extraction.md`](../archives/cuda-crate-extraction.md).

No further action remains under this heading. Any future runtime topology work
should start from the current in-tree `infer` layout, not from the archived
Round 3 split plan.

---

## Template for closing this document

When all four items are done:

1. Update each status row with the final commit SHA.
2. Move this file to `docs/archives/backend-reorg-followups.md` — it's historical at that point.
3. Write a single win entry at `docs/experience/wins/YYYY-MM-DD-backend-reorg-complete.md` summarising the four-part journey: Round 2 structural → F1 metal split → F2 graph_pool decision → Round 3 crate extraction.

Until all four are done, this file stays at its current path as the live index.
