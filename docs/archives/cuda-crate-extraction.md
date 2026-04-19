> **ARCHIVED 2026-04-15 — Superseded by Route-A refactor.** The four-crate split
> this plan described (`infer-core` / `infer-engine` / `infer-observability` /
> `infer-policy`) was folded back into `infer` after it became clear the split
> did not achieve real independence. See `docs/codebase-map.md` for the current
> layout. This file is retained for history only.

# Round 3 · Extract `backend`/`ops`/`model`/`scheduler` into `crates/infer-engine`

**Owner:** remote Linux CUDA host (the only environment where nvcc + real GPU tests can run)
**Prerequisite commits:**
- `7a876e1` (Round 2 housekeeping) — backend/{cuda,metal,cpu,runtime} layout in place.
- All F1 commits (Metal split) — Metal must be split first so `backend/metal/*` is a clean set of topical files, not one 1766-line monolith. F1 is tracked in `../plans/backend-reorg-followups.md` §F1; the original plan doc `backend-metal-split.md` was pruned 2026-04-15 after execution.
- A clean working tree on `main`, current with `origin/main`.

**Status:** design complete, not yet executed

**Do not start until F1 is merged.** F1 restructures the internals of `backend/metal/`. If Round 3 runs in parallel, every layer-1 commit will collide with F1 on `backend/metal.rs` and on visibility changes inside `backend/metal/`.

---

## 1. Why

The workspace already has eight shell crates under `crates/` (`infer-core`, `infer-engine`, `chat`, `infer-policy`, `infer-observability`, `agent`, `cli`, `tools`), four of which the `infer` crate already depends on (`infer-core`, `chat`, `infer-policy`, `infer-observability`). The direction "break `infer` single-crate into a set of focused crates" is the path the project already committed to.

But the **compute core** — `backend/`, `ops/`, `model/`, `scheduler/`, `weight_loader.rs` — still lives inside the monolithic `infer` crate (44 k+ lines of Rust). This means:
- `cargo check -p infer` rebuilds the entire engine for every single `http_server.rs` edit.
- `cudarc` / `cuda` feature flags leak into the http / server layer, forcing every consumer to re-compile cuda-dependent artifacts even when only serving changed.
- There is no way to run `cargo test -p infer-engine` on just the compute core without dragging in http + tokio + axum.
- The eight empty shells sit as "unfinished plans" rather than organised decomposition.

Round 3 finishes the split. After Round 3:

```
infer-core              (types, traits, sampler, tokenizer, error)          — pure Rust, no GPU
  ↑
infer-engine            (backend/, ops/, model/, scheduler/, weight_loader) — cudarc + metal + mlx-sys
  ↑
infer                   (http_server/, server_engine, metrics, trace, bins) — axum + tokio
```

Three crates with one-way dependency, clear responsibility. `cargo check -p infer-engine` runs without dragging in http. `--features cuda` scopes cleanly to `infer-engine`. `infer` becomes a thin server shell over `infer-engine`.

---

## 2. Why remote

Round 3 touches CUDA-heavy code paths: `backend/cuda/*` (which compiles with real nvcc), `ops/*` (which links to CUDA kernels via `crate::backend::cuda::ffi`), `model/*` (which uses CUDA tensors), `scheduler/*` (which owns the CUDA paged-KV pool). A refactor that touches these but can only be verified with `cargo check --features cuda,no-cuda` (rustc-only, skips nvcc) is **not adequately verified**. The full compile chain, integration tests, and throughput benchmarks need a real CUDA host.

The remote Linux CUDA host should be the place where:
- `CUDA_HOME=/usr/local/cuda cargo build --release` runs (full nvcc + linking);
- `cargo test --release` runs the CUDA unit tests;
- `cargo test --release --test e2e` runs end-to-end greedy regression against JSON baselines in `infer/test_data/`;
- `cargo test --release --test e2e_qwen35` runs Qwen3.5 regression;
- `scripts/bench_throughput_sweep.py` runs the standard throughput sweep for before/after snapshots.

If any of these fail after a Round 3 commit, the commit is not done — revert or follow-up immediately.

---

## 3. Target topology (detailed)

### 3.1 `crates/infer-core` — pure types

**Absorbs from `infer`:**
- `src/error.rs`
- `src/sampler.rs` (the `SamplingParams` struct is a pure data type; the actual sampling kernels stay in `backend/cuda` or `backend/metal`)
- `src/tokenizer.rs` (the `Tokenizer` wrapper is a thin trait around `tokenizers::Tokenizer`; no GPU dependency)

**Keeps its existing 119 lines** (request id, inference mode, event kinds — already used by scheduler policy).

**Crates it can depend on:** `anyhow`, `serde`, `log`, `half`, `thiserror`, `tokenizers`, `rand`.
**Crates it MUST NOT depend on:** `cudarc`, `mlx-sys`, `infer`, `infer-engine`, `axum`, `tokio`.

### 3.2 `crates/infer-engine` — compute core

**Absorbs from `infer`:**
- `src/backend.rs` + `src/backend/` (everything: cuda, metal, cpu, runtime)
- `src/ops.rs` + `src/ops/`
- `src/model.rs` + `src/model/`
- `src/scheduler.rs` + `src/scheduler/`
- `src/weight_loader.rs`
- `src/model_registry.rs`
- `src/gguf.rs`
- `src/hf_hub.rs`
- `src/block_manager.rs`
- `src/memory_planner.rs`
- `src/prefix_cache.rs`
- `src/quant.rs`
- `src/request_handle.rs`
- `src/speculative.rs`
- `src/tensor_parallel.rs`
- `src/kv_tier.rs` + `src/kv_tier/`

**Features it owns:** `cuda`, `metal`, `cpu`, `no-cuda`, `rdma-nixl`, `rdma-nixl-real`. The `infer` crate's `cuda`/`metal`/`cpu` features become forwarders that enable `infer-engine/cuda` etc.

**Crates it depends on:** everything `infer` currently depends on for compute (`cudarc`, `mlx-sys`, `safetensors`, `memmap2`, `half`, `ndarray`, `log`, `anyhow`, `hf-hub`, `nixl-sys`, `fastrace`) + `infer-core` + `infer-policy` (scheduler uses its admission / chunking policies) + `infer-observability` (scheduler emits events).

**Crates it MUST NOT depend on:** `axum`, `tokio` (outside of `sync::mpsc`), `infer`, anything http-related.

### 3.3 `crates/infer-observability` — metrics + traces

**Absorbs from `infer`:**
- `src/metrics.rs`
- `src/trace_reporter.rs`
- `src/logging.rs`

**Keeps its existing 55 lines** (`EngineEvent`, `EventSink`, `NoopEventSink`).

### 3.4 `infer` — http + server + bin

**What stays:**
- `src/main.rs` + `src/bin/*` — entry points
- `src/http_server.rs` + `src/http_server/` — axum handlers
- `src/server_engine.rs` — single-request façade (re-exports + thin adapter over `infer_engine::scheduler::Scheduler`)
- `src/lib.rs` — now a short re-export shell that makes `infer::backend::*`, `infer::model::*`, `infer::ops::*` etc. still resolvable for downstream tooling (optional: the `pub use` block may simplify migration for external crates)

**What gets re-exported from infer-engine:**
```rust
// infer/src/lib.rs (post-Round-3)
pub use infer_engine::{backend, model, ops, scheduler};
pub use infer_engine::server_engine;
// ... or whatever the final cut decides
```
Note: re-exports are normally a "backwards-compat hack" (CLAUDE.md forbids them), **but** in this case they are a **crate boundary fix**, not an in-crate hack — external tests/benches that say `use infer::backend::cuda::tensor::DeviceVec` continue to work without a sweeping rewrite. The re-exports should eventually be deleted in a later cleanup pass once `infer_engine::...` paths are adopted throughout the workspace, but **not in Round 3**.

### 3.5 Shells to delete (dead shells, no path forward)

Audit `crates/tools` (empty — no `infer` dependency, no consumers), `crates/agent`, and `crates/cli`. If any have no source files beyond a stub `lib.rs` and no dependents, delete them in the final Round 3 commit with a `chore(crates): drop unused shells` message. Leave them alone if they have anything non-trivial.

---

## 4. Execution order (layered)

Round 3 cannot be incremental in the simple "one module at a time" sense — moving `ops/` out of `infer` while `model/` still lives in `infer` and references `crate::ops::*` would break compilation. Each layer must move a **dependency-closed cluster** and update every caller in the same commit.

Five commits, each one a layer. Within a commit you may do the work in several `perl -i -pe` passes for readability but must land as one atomic commit.

### Layer 0 · Feature plumbing prep

**Commit:** `chore(crates): mirror cuda/metal/cpu features on infer-engine`

Before moving code, make `infer-engine`'s `Cargo.toml` feature-forward-capable so that `infer/Cargo.toml`'s `cuda` feature becomes `["infer-engine/cuda"]` and similarly for metal/cpu/no-cuda. This commit alone should compile cleanly because the features currently do nothing in infer-engine.

Touch:
- `crates/infer-engine/Cargo.toml` — add `[features]` block mirroring `infer`'s, plus `cudarc`, `mlx-sys`, `memmap2`, `libc` as optional deps (not yet used).
- No `.rs` file changes.

Verify: `cargo check --workspace --no-default-features --features cuda,no-cuda`, `cargo check --workspace --features metal`.

### Layer 1 · Move `backend/` to infer-engine

**Commit:** `refactor(crates): move backend/ into infer-engine`

The biggest commit of Round 3. Moves `backend.rs` + `backend/cuda/` + `backend/metal/` + `backend/cpu.rs` + `backend/runtime.rs` into `crates/infer-engine/src/backend{,.rs}/`, and updates every caller of `crate::backend::*` inside the not-yet-moved modules (ops/, model/, scheduler/, server_engine, etc.) to use `infer_engine::backend::*`.

Detailed steps:

1. `mkdir -p crates/infer-engine/src/backend/cuda crates/infer-engine/src/backend/metal`
2. `git mv infer/src/backend.rs crates/infer-engine/src/backend.rs`
3. `git mv infer/src/backend/cuda.rs crates/infer-engine/src/backend/cuda.rs`
4. `git mv infer/src/backend/cuda/* crates/infer-engine/src/backend/cuda/` (one `git mv` per file)
5. `git mv infer/src/backend/metal.rs crates/infer-engine/src/backend/metal.rs`
6. `git mv infer/src/backend/metal/* crates/infer-engine/src/backend/metal/` (one per file)
7. `git mv infer/src/backend/cpu.rs crates/infer-engine/src/backend/cpu.rs`
8. `git mv infer/src/backend/runtime.rs crates/infer-engine/src/backend/runtime.rs`
9. Update `crates/infer-engine/Cargo.toml`: add real `cudarc`, `mlx-sys`, `memmap2`, `libc`, `hf-hub`, `tokio`, `anyhow`, `log`, `tokenizers`, `infer-core`, `infer-policy`, `infer-observability` dependencies as needed.
10. Update `crates/infer-engine/src/lib.rs`:
    ```rust
    pub mod backend;
    ```
11. Remove `pub mod backend;` from `infer/src/lib.rs`.
12. Add `infer-engine = { path = "../crates/infer-engine" }` to `infer/Cargo.toml` (already present — confirm).
13. Bulk-update every `crate::backend::*` in the still-in-infer files to `infer_engine::backend::*`:
    ```bash
    perl -i -pe 's/\bcrate::backend::/infer_engine::backend::/g' \
      $(find infer/src -name '*.rs' -not -path '*/backend/*')
    ```
    Then hand-fix `infer/src/lib.rs` which may now need `pub use infer_engine::backend;` to keep external consumers' `infer::backend::...` paths working (see §3.4 re-export rule).
14. Inside the moved files, references that were `crate::...` (infer-internal) may now need to be changed. The tricky cases:
    - `crate::sampler::SamplingParams` — sampler hasn't moved yet in this layer. The moved `backend/metal.rs` uses it via `use crate::sampler::SamplingParams;`. That `crate::sampler` now means `infer_engine::sampler`, which doesn't exist — it still lives in infer. **Option A:** move `sampler.rs` as part of Layer 1 too (add it to the move list). **Option B:** change the reference to `infer::sampler::SamplingParams` (reverse dep — breaks the one-way direction). **Option A is correct.** Add `sampler.rs` to the Layer-1 move list.
    - Similarly `crate::tokenizer::Tokenizer` — add `tokenizer.rs` to Layer 1.
    - `crate::hf_hub::*` — add to Layer 1.
    - `crate::error::*` — add to Layer 1 (or push to infer-core, but it's fine to start in infer-engine and promote later).
    - `crate::request_handle::*` used by backend::runtime — add to Layer 1.
15. Update external callers in `infer/src/bin/*`, `crates/infer-engine/src/lib.rs` re-exports, and `crates/infer-*` dependent crates.

**Verification:**
- `cargo check -p infer-engine --no-default-features --features cuda,no-cuda`
- `cargo check -p infer-engine --no-default-features --features metal`
- `cargo check -p infer --no-default-features --features cuda,no-cuda`
- `cargo check -p infer --no-default-features --features metal`
- `cargo check --workspace --features metal`
- **On Linux CUDA host:** `CUDA_HOME=/usr/local/cuda cargo build --release`
- **On Linux CUDA host:** `cargo test --release --lib` (unit tests)

If end-to-end tests exist at the crate boundary, run them. If not, move to Layer 2.

### Layer 2 · Move `ops/` to infer-engine

**Commit:** `refactor(crates): move ops/ into infer-engine`

Moves `ops.rs` + `ops/` into `infer-engine`. Much smaller than Layer 1 because all its cross-crate callers (model, scheduler) will be moved in Layer 3 — so Layer 2 only needs to update callers that stay in infer (a short list: mostly tests and probably nothing else).

1. `git mv infer/src/ops.rs crates/infer-engine/src/ops.rs`
2. `git mv infer/src/ops/* crates/infer-engine/src/ops/`
3. Add `pub mod ops;` to `crates/infer-engine/src/lib.rs`.
4. Remove `pub mod ops;` from `infer/src/lib.rs`.
5. Inside moved files, `crate::backend::cuda::*` works directly (both are now inside infer-engine).
6. Update the few remaining `crate::ops::*` references in infer: `infer/src/ops/tests.rs` is currently inside ops/ and will move with it; check whether there are other references (`grep -n "crate::ops::" infer/src/`).

**Verification:** same suite as Layer 1.

### Layer 3 · Move `model/` + `weight_loader.rs` + loaders

**Commit:** `refactor(crates): move model/ and weight_loader into infer-engine`

1. `git mv infer/src/model.rs crates/infer-engine/src/model.rs`
2. `git mv infer/src/model/* crates/infer-engine/src/model/`
3. `git mv infer/src/weight_loader.rs crates/infer-engine/src/weight_loader.rs`
4. `git mv infer/src/gguf.rs crates/infer-engine/src/gguf.rs`
5. `git mv infer/src/model_registry.rs crates/infer-engine/src/model_registry.rs`
6. `git mv infer/src/quant.rs crates/infer-engine/src/quant.rs` (if it's weight quant, belongs in infer-engine)
7. Add `pub mod model; pub mod weight_loader; pub mod gguf; pub mod model_registry; pub mod quant;` to `crates/infer-engine/src/lib.rs`.
8. Remove those `pub mod`s from `infer/src/lib.rs`.
9. Update any `crate::model::*` / `crate::weight_loader::*` etc. in infer's still-present files (scheduler will move in Layer 4, so it still references `crate::model::*` which now means `infer_engine::model::*`).

**Verification:** same suite.

### Layer 4 · Move `scheduler/` + KV machinery

**Commit:** `refactor(crates): move scheduler + kv_tier into infer-engine`

1. `git mv infer/src/scheduler.rs crates/infer-engine/src/scheduler.rs`
2. `git mv infer/src/scheduler/* crates/infer-engine/src/scheduler/`
3. `git mv infer/src/kv_tier.rs crates/infer-engine/src/kv_tier.rs`
4. `git mv infer/src/kv_tier/* crates/infer-engine/src/kv_tier/`
5. `git mv infer/src/block_manager.rs crates/infer-engine/src/block_manager.rs`
6. `git mv infer/src/memory_planner.rs crates/infer-engine/src/memory_planner.rs`
7. `git mv infer/src/prefix_cache.rs crates/infer-engine/src/prefix_cache.rs`
8. `git mv infer/src/request_handle.rs crates/infer-engine/src/request_handle.rs` (if not already moved in Layer 1)
9. `git mv infer/src/speculative.rs crates/infer-engine/src/speculative.rs`
10. `git mv infer/src/tensor_parallel.rs crates/infer-engine/src/tensor_parallel.rs`
11. Register them in `crates/infer-engine/src/lib.rs`, remove from `infer/src/lib.rs`.
12. Update `infer/src/http_server.rs`, `infer/src/server_engine.rs`, `infer/src/bin/*`: `crate::scheduler::*` → `infer_engine::scheduler::*`.

**Verification:** same suite. After this layer `infer-engine` should be functionally complete.

### Layer 5 · Observability split + infer thinning

**Commit 1:** `refactor(crates): move metrics + trace into infer-observability`

1. `git mv infer/src/metrics.rs crates/infer-observability/src/metrics.rs`
2. `git mv infer/src/trace_reporter.rs crates/infer-observability/src/trace_reporter.rs`
3. `git mv infer/src/logging.rs crates/infer-observability/src/logging.rs`
4. Register in `crates/infer-observability/src/lib.rs`.
5. Update callers in infer and infer-engine.

**Commit 2:** `refactor(crates): thin infer crate to http + bin`

1. Audit `infer/src/lib.rs` — it should now be short. Either (a) delete remaining `pub mod`s that only re-export infer-engine, or (b) replace with `pub use infer_engine::*;` block.
2. Clean up `infer/Cargo.toml` — remove now-unused dependencies (`cudarc`, `mlx-sys`, `safetensors`, etc. are only needed through `infer-engine`).
3. Expected state: `infer/src/` contains `main.rs`, `bin/`, `http_server.rs`, `http_server/`, `server_engine.rs`, `lib.rs`, possibly `error.rs` stub. Line count target for `infer/src/` total: **< 8000 lines** (down from current 44000+).

**Commit 3 (optional):** `chore(crates): drop unused tools/agent/cli shells`

Only if §3.5's audit confirms they have no dependents. Conservative default: leave them.

---

## 5. Verification matrix (per layer + final)

Every layer must pass this matrix before its commit lands:

### Darwin (Mac, dev machine)
```bash
cargo check -p infer --no-default-features --features metal
cargo check -p infer --no-default-features --features no-cuda
cargo check -p infer --no-default-features --features cuda,no-cuda
cargo check -p infer-engine --no-default-features --features metal
cargo check -p infer-engine --no-default-features --features no-cuda
cargo check -p infer-engine --no-default-features --features cuda,no-cuda
cargo check --workspace --no-default-features --features metal
cargo test -p infer-engine --no-default-features --features metal --lib
cargo test -p infer --no-default-features --features metal --lib
cargo clippy --workspace --no-default-features --features metal -- -D warnings
```

### Linux CUDA host (REAL validation)
```bash
CUDA_HOME=/usr/local/cuda cargo build --release
CUDA_HOME=/usr/local/cuda cargo test --release --lib
INFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release --test e2e
cargo test --release --test e2e_qwen35

# Throughput regression check (per docs/experience/wins/ convention):
scripts/bench_throughput_sweep.py --label round3-layer-N-after
# Diff against docs/experience/wins/YYYY-MM-DD-bench-pre-round3.md
```

If a layer ships and the throughput sweep shows >5% regression on any key metric, **stop**. The layer's commit should be reverted immediately and the root cause investigated. A pure refactor should show ≤1% noise.

### CI workflow (.github/workflows/*.yml)

- `metal-ci.yml` — path filters need updating (see Layer 5). Currently globs `infer/src/backend/**`; after Round 3 the Metal backend lives at `crates/infer-engine/src/backend/metal/**`. Update the path filter or the CI job will stop triggering.
- Any CUDA CI workflow similarly needs path updates.

---

## 6. Pitfall list (things that will surprise you)

1. **Cross-crate visibility.** Items that were `pub(crate)` inside `infer` become accessible only within `infer-engine` after the move. External access from `infer` requires upgrading to `pub`. Expect several `unreachable_pub` warnings after Layer 1; each is a hint that the item in question needs `pub(crate)` → `pub` for cross-crate use. Don't blanket-upgrade — each one is a boundary decision.

2. **`scheduler` uses `infer_policy`, `infer_observability`, `infer_core`** already (it's the bootstrap for the "use the shells" direction). These edges already work; Round 3 does not break them.

3. **`cudarc` feature propagation.** `infer` currently says:
   ```toml
   cuda = ["dep:cudarc", "dep:memmap2"]
   ```
   After Layer 1, cudarc and memmap2 become optional deps of `infer-engine`. The `infer` feature must forward:
   ```toml
   cuda = ["infer-engine/cuda"]
   ```
   And `infer-engine/Cargo.toml` must declare its own optional `cudarc` / `memmap2` with the same version constraints. **Version mismatch between the two crates will cause linker errors that blame the wrong crate**.

4. **`build.rs` ownership.** `infer/build.rs` currently compiles CUDA C kernels, calls nvcc, handles FlashInfer headers, etc. After Layer 1, `infer-engine` owns the backend, so `build.rs` should move to `crates/infer-engine/build.rs`. `infer` crate then loses its build script entirely. Or — alternative — keep a thin `infer/build.rs` that just forwards to `infer-engine`'s (ugly; don't do this).
   - **Recommendation:** move `infer/build.rs` → `crates/infer-engine/build.rs` as the FIRST action of Layer 1.
   - `infer/csrc/cuda/**` — should it move too? Technically csrc belongs with the crate that compiles it. So yes: `git mv infer/csrc crates/infer-engine/csrc`.
   - `infer/tools/triton/**` — same question, same answer: move with the crate.
   - `infer/mlx-sys/**` — it's already a separate sub-crate with its own `Cargo.toml`. Its path dep in infer needs to be updated to `crates/infer-engine` dependency.

5. **`mlx-sys` path relocation.** `infer-engine` needs `mlx-sys = { path = "..." }`. Currently it's at `infer/mlx-sys`. After Round 3, move it to either `crates/infer-engine/mlx-sys` (nested) or promote to `crates/mlx-sys` (flat). The flat layout is cleaner — move it to `crates/mlx-sys`.

6. **`test_data/` location.** `infer/test_data/*.json` are golden baselines for e2e tests. The tests that read them are currently in `infer/tests/`. If tests move with the crate that owns them (they should), then `test_data/` moves too: `crates/infer-engine/test_data/`. But `infer/tests/` may also contain tests specific to the http layer that should stay in `infer/tests/`. Audit each test file in Layer 3 or 4 and split.

7. **`cargo test --workspace --features metal` runs tests across all crates.** If there's a test that was conditional on `feature = "cuda"` in `infer/tests/` and it still lives there after the move, ensure `infer/Cargo.toml` still defines the `cuda` feature as a forwarder. Otherwise `cargo test --features cuda` on the workspace will silently skip the test.

8. **Git rename detection across crate boundaries.** `git log --follow` on a moved file may break because the file jumped between directories too far. Mitigation: do each move in a single commit where the file content is byte-identical to its previous revision. If you also edit the file in the same commit, git sees it as delete + add. To keep history: first commit the move alone, then a second commit edits the file. (Two-step per file for the most important history-preservation cases; one-step for the bulk.)

9. **Workspace Cargo.lock churn.** Every `cargo check` on a restructured workspace regenerates Cargo.lock. Expect to see Cargo.lock diffs in every Round 3 commit; include them.

10. **`cargo fmt` formatting after bulk perl sweeps.** Expect formatting drift, same as Round 2 Commit 3. Add a `style(crates): rustfmt after Layer N extract` commit after each layer if needed, matching the Round 2 pattern in commit `bbd8693`.

---

## 7. Rollback / abort strategy

If any layer ships and CUDA bench regresses or e2e baseline mismatches, revert the offending layer commit and investigate:

```bash
git revert <commit-sha>      # creates a new commit undoing the layer
```

Do **not** use `git reset --hard` or force-push. Round 3 layers are merges of substantial work; treat them as permanent history. If a layer is half-broken, revert + fix-forward, don't erase.

Each layer is a bisect boundary. Use `git bisect` aggressively if an e2e regression appears and you're not sure which layer introduced it.

---

## 8. What "done" looks like

- `infer/src/` line count < 8 000 (down from ~44 000).
- `crates/infer-engine/src/` line count ~36 000.
- `crates/infer-core/src/` line count ~1 000.
- `crates/infer-observability/src/` line count ~500.
- `cargo check -p infer-engine --features cuda,no-cuda` green on Darwin.
- `cargo build --release --features cuda` green on Linux CUDA host.
- `cargo test --release` green on Linux CUDA host (unit + e2e).
- `scripts/bench_throughput_sweep.py` shows ≤1% noise vs. pre-Round-3 baseline.
- `.github/workflows/*.yml` path filters all updated.
- `docs/plans/backend-reorg-followups.md` updated to mark Round 3 as closed out with the final commit SHA.
- Round 3 gets a win entry: `docs/experience/wins/YYYY-MM-DD-round3-crate-extraction.md`.

---

## 9. Out of scope for Round 3

- **Do not rename** any existing item in-situ (e.g. `Scheduler` → `CudaScheduler`). Pure relocation only.
- **Do not add** new public APIs, new traits, or new enum variants.
- **Do not split** `backend/metal.rs` — F1 is the split. Round 3 assumes F1 is done and just moves the already-split files wholesale.
- **Do not rewrite** CUDA kernels or introduce new kernels — `csrc/cuda/` is relocated, not edited.
- **Do not delete** `cuda_graph_pool` dead code even though Round 2 noted it has zero consumers — that's a separate F2 follow-up.
- **Do not touch** `agent_infer/` Python package, `src/` top-level agent binary (except to update its `use infer::X` imports if the infer re-exports are insufficient), or any Python scripts.
