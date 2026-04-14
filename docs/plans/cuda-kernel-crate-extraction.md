# `infer-cuda-kernels` extraction blueprint

**Status:** queued, not yet triggered. Execute when one of the trip wires in §2 fires.
**Owner of the trigger decision:** ckl.
**Owner of the execution:** Codex on the remote Linux + CUDA host (Darwin can rustc-verify but not nvcc-link).
**Prerequisites:** all currently-landed Route-A and CUDA-internal hygiene commits (`d902090`, `909e8cc`, `d3136ba`, `26c8f39`, `efcc991`).

This is the forward-looking plan for migrating `infer/src/backend/cuda/**` plus
`infer/csrc/cuda/**` plus `infer/tools/triton/**` plus the relevant build.rs
slice into a standalone `crates/infer-cuda-kernels/` crate. The plan exists so
that when the trigger fires, the work is **mechanical, not architectural** —
no re-debate, no scope creep, no Route-A 2.0.

---

## 1 · Why we did not extract yet

The 2026-04-15 strategic discussion (Claude + Codex independent reads) chose
**option A (monolithic `infer`)** with three internal hygiene moves
(committed as `26c8f39` and `efcc991`) instead of immediate extraction. The
reasoning, in one paragraph:

> Today's `bootstrap.rs` reaches into `crate::model::*`, `crate::scheduler::*`,
> `crate::tokenizer::Tokenizer`, and `crate::model_registry::*` to load and
> dispatch model-specific CUDA engines. Extracting `backend/cuda/` now would
> force `Tokenizer` / `KVCacheDtype` / `ModelType` / `Qwen3Model` /
> `Qwen35Model` / `GLM4Model` to become `pub` cross-crate (they are currently
> `pub(crate)`), and `bootstrap.rs::load_*_components` would straddle the new
> crate boundary. That is the exact failure mode of the Route-A four-shell
> split. Doing it before the consumer signals are real would relocate files
> without delivering decoupling.

The internal hygiene moves that ARE landed:

1. **`infer/src/backend/cuda/ffi.rs` split into 10 domain submodules** — touching
   one binding no longer invalidates the whole 1500-line file.
2. **`infer/src/backend/cuda/prelude.rs` as proto-API contract** — 7 stable
   types that ~31 model/ops files import through; the prelude is intentionally
   *narrow* so it can graduate to a real public API surface without churn.
3. **Triton `cargo:rerun-if-changed` auto-derived** from `read_dir(tools/triton)`
   so the build script can't drift from the actual kernel set.
4. **Dead Triton kernels deleted** (`flash_attention_prefill`, `attention_decode`,
   `attention_reduce`, single-token `embedding`).

These moves make the eventual extraction cheaper because the seams are already
where they need to be. The work below assumes those seams.

---

## 2 · Trip wires (any one of these → execute)

These are not hypotheticals. They are all on `ROADMAP.md::Missing` and the
project is committed to following SGLang/vLLM in capability:

| # | Trigger | Why it forces extraction |
|---|---|---|
| **T1** | **Parallel kernel build configs producing incompatible `.a` artifacts** | E.g. when FA-3 H100 work introduces sm_90-only kernel paths that conflict with sm_80 fallbacks, or when a vendored FlashInfer fork needs a different include search path. A single `infer/build.rs` cannot ship two different `libkernels_cuda.a` flavors. Extraction lets each artifact live in its own crate (or feature-gated within one). |
| **T2** | **Adding NCCL tensor parallel communication** | `ROADMAP.md` §"Missing" explicitly calls this out. NCCL pulls in `libnccl` linkage + multi-GPU coordination kernels. Today the build script already has feature gates piling up (`cuda`, `no-cuda`, `metal`, `cpu`, `rdma-nixl`, `rdma-nixl-real`); adding `nccl` on top will start producing combinatorial build matrices that don't fit cleanly under one crate's feature flags. |
| **T3** | **FlashAttention-3 (H100 / sm_90) prefill** | `ROADMAP.md` §1.2. FA-3 needs warp specialization + async pipeline. The current `prefill_attention.cu` + `flashinfer_single_prefill` path is sm_80-tuned. Adding FA-3 means **two parallel attention kernel implementations selected by SM target** — i.e., trip wire T1 follows. |
| **T4** | **MLA attention (DeepSeek-V2/V3/R1) + DeepSeek MoE + FP8 GEMM (sm_90)** | `ROADMAP.md` §1.3 + §1.4. This is essentially a second compute backend: new attention algorithm (latent KV compression), new GEMM family (FP8 E4M3), new model family. ~30+ new kernel files. The `csrc/cuda/` tree doubles in size, and the parallel kernel set forces T1 / T2 anyway. |
| **T5** | **Speculative decoding GPU integration** | `ROADMAP.md` §"Missing". Today `infer/src/speculative.rs` and `infer/src/speculative/cuda.rs` are CPU stubs. Real GPU integration means a draft kernel surface that gets called from inside the scheduler hot loop. The existing scheduler→backend→cuda boundary will need cleanup before this lands cleanly. |
| **T6** | **A second team / external consumer starts owning only the kernel layer** | E.g. another inference project wants to reuse `infer-cuda-kernels` directly without pulling in the agent runtime. The "two direct consumers" admission criterion from `docs/archives/art-grade-architecture-for-long-agent-infer.md` §六 is met. |

**Decision rule:** if **any one** of T1–T6 enters in-flight implementation
status (i.e., a real PR opens against it), execute this blueprint *as part of
that PR's preparation*, not as a separate refactor pass. The extraction must
land **before** the new work starts, so the new code is written against the
extracted crate from day one.

---

## 3 · Target topology

```
agent-infer/                        ← workspace root (unchanged)
├── crates/
│   ├── infer-agent/                ← unchanged
│   ├── infer-chat/                 ← unchanged
│   ├── infer-cli/                  ← unchanged
│   ├── infer-cuda-kernels/         ← NEW
│   │   ├── Cargo.toml              ← cudarc + cc + flashinfer detection;
│   │   │                              `cuda` and `no-cuda` features mirror infer's
│   │   ├── build.rs                ← lifted from infer/build.rs (nvcc + Triton AOT)
│   │   ├── csrc/cuda/              ← lifted from infer/csrc/cuda/
│   │   ├── tools/triton/           ← lifted from infer/tools/triton/
│   │   └── src/
│   │       ├── lib.rs              ← `pub use ffi::*;` + `pub use prelude::*;`
│   │       ├── ffi.rs              ← parent (type aliases + mod tree)
│   │       ├── ffi/{attention,gemm,kv,norm,quant,sampling,
│   │       │       embedding,elementwise,recurrent,misc}.rs
│   │       ├── tensor.rs           ← DeviceContext / DeviceVec / DeviceMatrix
│   │       │                          / RawDevicePtr / HiddenStates
│   │       ├── paged_kv.rs         ← PagedKVPool / TokenKVPool
│   │       ├── flashinfer.rs       ← FlashInferDecodeMetadata / workspace
│   │       ├── graph_pool.rs       ← parked F2 scaffold (still pub(crate))
│   │       └── prelude.rs          ← THE public API surface (see §4)
│   ├── infer-tools/                ← unchanged
│   └── mlx-sys/                    ← unchanged
├── infer/                          ← thin runtime + HTTP shell
│   ├── build.rs                    ← stripped of nvcc + Triton; only the
│   │                                  pure-Rust feature pass-through
│   ├── csrc/                       ← (gone — moved to infer-cuda-kernels)
│   ├── tools/                      ← (gone — moved to infer-cuda-kernels)
│   └── src/
│       ├── backend/
│       │   ├── cuda/               ← gone, except ONE thin façade file
│       │   └── cuda.rs             ← `pub use infer_cuda_kernels::*;` re-export
│       ├── backend/cuda/bootstrap.rs   ← STAYS in infer (uses model + scheduler)
│       ├── model/                  ← unchanged (imports via infer_cuda_kernels::prelude)
│       ├── ops/                    ← unchanged (imports via infer_cuda_kernels::ffi)
│       ├── scheduler/              ← unchanged
│       ├── server_engine.rs        ← unchanged
│       └── …                       ← rest of infer untouched
└── …
```

**Key invariant:** `bootstrap.rs` stays inside `infer` because it needs
`crate::model::*` and `crate::scheduler::*`. It calls into
`infer_cuda_kernels` from the `infer` side. The dependency edge is
**`infer → infer-cuda-kernels`, never the reverse.**

---

## 4 · The public API surface

The `infer-cuda-kernels::prelude` module is the **only** thing `infer`'s
model/ops/scheduler code is allowed to import from. Today's `prelude.rs`
already pre-stages this contract:

```rust
pub use flashinfer::FlashInferDecodeMetadata;
pub use paged_kv::PagedKVPool;
pub use tensor::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates, RawDevicePtr};
```

After extraction, those `pub(crate)` become `pub` (real cross-crate `pub`).
That's the ENTIRE diff for the public surface. **No new symbols become public.**

`TokenKVPool` is intentionally NOT in the prelude (used by only 2 prefill
files + `model.rs`). After extraction, those three call sites import
`infer_cuda_kernels::TokenKVPool` directly from the crate root, not via the
prelude. This keeps the prelude narrow on purpose.

The `ffi` submodule tree is also `pub` after extraction. The 10 domain
submodules already exist as a clean partition; they become the
`infer_cuda_kernels::ffi::{attention, gemm, kv, …}` namespace tree.

---

## 5 · Step-by-step execution (when a trip wire fires)

This is the mechanical part. Designed to be a **single PR**, completable in
one focused day on a CUDA-equipped host.

### 5.1 — Pre-flight (Darwin or CUDA host)

```
git checkout -b extract/infer-cuda-kernels
cargo fmt --all -- --check
cargo check -p infer --no-default-features --features cuda,no-cuda
cargo check --workspace --no-default-features --features cpu,no-cuda
cargo check --workspace --no-default-features --features metal
```

Capture before-snapshot benchmarks per CLAUDE.md `Benchmark Rules`:
`docs/experience/wins/YYYY-MM-DD-bench-pre-cuda-extraction.md`.

### 5.2 — Move files

```
git mv infer/csrc/cuda                      crates/infer-cuda-kernels/csrc/cuda
git mv infer/tools/triton                   crates/infer-cuda-kernels/tools/triton
git mv infer/src/backend/cuda/ffi.rs        crates/infer-cuda-kernels/src/ffi.rs
git mv infer/src/backend/cuda/ffi           crates/infer-cuda-kernels/src/ffi
git mv infer/src/backend/cuda/tensor.rs     crates/infer-cuda-kernels/src/tensor.rs
git mv infer/src/backend/cuda/paged_kv.rs   crates/infer-cuda-kernels/src/paged_kv.rs
git mv infer/src/backend/cuda/flashinfer.rs crates/infer-cuda-kernels/src/flashinfer.rs
git mv infer/src/backend/cuda/graph_pool.rs crates/infer-cuda-kernels/src/graph_pool.rs
git mv infer/src/backend/cuda/prelude.rs    crates/infer-cuda-kernels/src/prelude.rs
```

`bootstrap.rs` stays at `infer/src/backend/cuda/bootstrap.rs`.

### 5.3 — New `crates/infer-cuda-kernels/Cargo.toml`

Mirror `infer/Cargo.toml`'s CUDA deps:

```toml
[package]
name = "infer-cuda-kernels"
version = "0.1.0"
edition = "2024"

[features]
cuda = ["dep:cudarc", "dep:memmap2"]
no-cuda = []

[dependencies]
anyhow = "1.0"
half = { version = "2.4", features = ["num-traits"] }
log = "0.4"
cudarc = { version = "0.18", features = ["cuda-version-from-build-system", "cublas", "cuda-12080", "f16"], optional = true }
memmap2 = { version = "0.9", optional = true }

[build-dependencies]
cc = "1.0"

[lints]
workspace = true
```

### 5.4 — Lift `build.rs`

Move the nvcc + Triton AOT pass from `infer/build.rs` into
`crates/infer-cuda-kernels/build.rs`. The FlashInfer detection helper
(`find_flashinfer_include`) goes with it. Strip the corresponding lines from
`infer/build.rs`. The `infer/build.rs` becomes a small no-op (or is deleted
entirely if no other build-time work remains).

### 5.5 — Update `infer/Cargo.toml`

```toml
[dependencies]
infer-cuda-kernels = { path = "../crates/infer-cuda-kernels", default-features = false, optional = true }

[features]
cuda = ["dep:cudarc", "dep:memmap2", "infer-cuda-kernels/cuda"]
no-cuda = ["infer-cuda-kernels/no-cuda"]
```

`cudarc` and `memmap2` stay in `infer/Cargo.toml` because `bootstrap.rs` and
the rest of the runtime still use them. They are also pulled in transitively
through `infer-cuda-kernels`, but Cargo will dedupe.

### 5.6 — Re-pub the prelude exports

In each of `crates/infer-cuda-kernels/src/{tensor,paged_kv,flashinfer,graph_pool}.rs`,
change `pub(crate)` → `pub` on the items listed in the prelude (and `TokenKVPool`).
Everything else stays `pub(crate)` to preserve encapsulation.

In `crates/infer-cuda-kernels/src/lib.rs`:

```rust
pub mod ffi;
pub mod flashinfer;
pub mod graph_pool;
pub mod paged_kv;
pub mod prelude;
pub mod tensor;
```

### 5.7 — Update `infer/src/backend/cuda.rs`

Replace the existing module declarations with a thin re-export shim:

```rust
//! Re-exports of the `infer-cuda-kernels` crate so existing
//! `crate::backend::cuda::…` paths continue to resolve.
pub use infer_cuda_kernels::*;

pub mod bootstrap;  // stays in infer because it pulls model/ + scheduler/
```

This single re-export means **zero call site changes** in `infer/src/model/**`,
`infer/src/ops/**`, `infer/src/scheduler/**`, `infer/src/server_engine.rs`,
`infer/src/speculative/**`. Every `use crate::backend::cuda::prelude::{…}` and
every `use crate::backend::cuda::ffi::xxx` keeps resolving via the re-export.

### 5.8 — Verify

```
cargo fmt --all
cargo check -p infer --no-default-features --features cuda,no-cuda
cargo check -p infer-cuda-kernels --no-default-features --features cuda,no-cuda
cargo check --workspace --no-default-features --features cpu,no-cuda
cargo check --workspace --no-default-features --features metal
cargo test --workspace --release --no-default-features --features cpu,no-cuda --lib
cargo test --workspace --release --no-default-features --features metal --lib
cargo clippy --workspace --no-default-features --features cpu,no-cuda -- -D warnings
cargo clippy --workspace --no-default-features --features metal -- -D warnings
```

On the CUDA host:

```
CUDA_HOME=/usr/local/cuda cargo build --release
cargo test --release --lib
PEGAINFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release --test e2e
cargo test --release --test e2e_qwen35
scripts/bench_throughput_sweep.py --label post-cuda-extraction
```

After-snapshot must show **≤1% throughput delta** vs the before-snapshot
captured in §5.1. Anything more is a regression and the PR reverts.

### 5.9 — One-commit landing

Per CLAUDE.md PR discipline, the entire extraction lands as **one commit**
because it is a pure structural move with no behavior change. The trigger
work (T1–T6) lands as a **separate commit** on top, written against the
new crate boundary.

Commit message scope: `refactor(crates):` (the new "crates extraction" scope
mirrors the Route-A revert commit `d902090`).

---

## 6 · What stays in `infer` (and why)

| File / module | Why it stays |
|---|---|
| `infer/src/backend/cuda/bootstrap.rs` | Reaches into `crate::model::*`, `crate::scheduler::*`, `crate::tokenizer::Tokenizer`, `crate::model_registry::*`. These are application-level concerns, not kernel concerns. |
| `infer/src/backend/cuda.rs` (shim) | Provides the `pub use infer_cuda_kernels::*;` re-export so 60+ existing call sites don't need to change. |
| `infer/src/model/**` | Models are application logic. They consume the kernel crate via `infer_cuda_kernels::prelude` (transitively through `crate::backend::cuda::prelude`). |
| `infer/src/ops/**` | Ops are the Rust-side kernel invocation layer. They live next to model code because they're tightly coupled to model layouts. |
| `infer/src/scheduler/cuda/**` | Scheduler is application logic. It uses `PagedKVPool` and `DeviceContext` from the prelude. |
| `infer/src/server_engine.rs` | Hosts the `InferenceEngine` trait + `LoadedInferenceEngine` enum. Backend-agnostic by design. |

The split's **load-bearing principle**: a file goes in `infer-cuda-kernels`
**only if it knows nothing about models or schedulers**. A file stays in
`infer` if it reaches for `Tokenizer`, model-specific weights, or scheduler
state. Bootstrap is the one and only place where the two layers meet.

---

## 7 · Anti-goals (things this plan refuses to do)

- **No `infer-ops` extraction.** Ops are tightly coupled to model data layouts
  and would force `crate::model::common::*` to become `pub` cross-crate. Keep
  them next to the models that own the layouts.
- **No `infer-scheduler-core` extraction.** The CUDA scheduler reaches into
  `PagedKVPool`, `FlashInferDecodeMetadata`, model-specific `Qwen3Model` /
  `Qwen35Model` / `GLM4Model` types in bootstrap. Splitting it out would
  re-create the bootstrap straddle problem.
- **No `infer-runtime-api` trait crate.** Already covered by
  `infer::server_engine::InferenceEngine`. Putting it in a separate crate
  with no second consumer would re-create the Route-A failure mode.
- **No `infer-cuda` (Rust-only) AND `infer-cuda-kernels` (native-only) split.**
  One crate, both layers. The Rust types and the kernels they wrap belong
  together — splitting them creates a `*-sys` boundary with one consumer.

---

## 8 · Reverse-direction safety net

If after extraction we discover that the throughput regressed, that the
build matrix got worse on Darwin, or that the prelude had to grow uncontrollably
to support model code that suddenly needs a previously-private symbol, the
revert path is:

1. `git mv` everything back (one commit).
2. Update `infer/Cargo.toml`, `infer/build.rs`, `infer/src/backend/cuda.rs`
   to the pre-extraction shape.
3. Delete `crates/infer-cuda-kernels/`.
4. Write a `docs/experience/errors/YYYY-MM-DD-cuda-extraction-revert.md`
   postmortem documenting which prediction failed.

The Route-A revert (`d902090`) is the precedent. We will not be afraid to
revert if reality rejects the plan.

---

## 9 · Cross-references

- `docs/architecture.md` — current workspace topology + Future Evolution section
  pointing here.
- `docs/codebase-map.md` — current `infer/` tree layout.
- `docs/archives/art-grade-architecture-for-long-agent-infer.md` — the
  ambitious 8-crate split that Route-A reverted; §六 governance rules and §七
  acceptance criteria still inform the trip wire criteria above.
- `docs/archives/cuda-crate-extraction.md` — the original (overly ambitious)
  Round-3 extraction plan that bundled `backend + ops + model + scheduler`
  together. Superseded by this narrower kernel-only blueprint.
- `infer/src/backend/cuda/prelude.rs` — the proto-API contract that becomes
  the public surface of the extracted crate.
- `ROADMAP.md` §"Missing" — the source of trip wires T1–T6.
