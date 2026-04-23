# `cuda-kernels` extraction blueprint

**Status:** ✅ **executed** — landed in `a4e12f5 refactor(cuda): extract cuda-kernels api`
(with fmt follow-up `f81e2d5`). Kept as a historical reference for the trip
wires and decision rationale; §3 Target topology has been updated to match
what actually landed. For the current workspace shape see
[`../architecture.md`](../architecture.md) §"Kernel-Crate Extraction".
**Original trigger decision owner:** ckl.
**Execution:** landed through a single focused PR on the CUDA-equipped host.
**Prerequisites (all landed):** Route-A revert + CUDA-internal hygiene moves
(`d902090`, `909e8cc`, `d3136ba`, `26c8f39`, `efcc991`).

This was the forward-looking plan for migrating `infer/src/backend/cuda/**`
plus `infer/csrc/cuda/**` plus `infer/tools/triton/**` plus the relevant
build.rs slice into a standalone `crates/cuda-kernels/` crate. The plan
was designed so the execution was **mechanical, not architectural** — no
re-debate, no scope creep, no Route-A 2.0. That property held: the extraction
landed in one commit, the before/after benchmarks stayed within the ≤1%
delta bar, and the §5 procedure below matches what was actually run (with
one notable divergence — see the §3 note about `csrc/cuda/` being flattened
out during the move).

---

## 1 · Why we pre-staged before extracting

The 2026-04-15 strategic discussion (Claude + Codex independent reads) first
chose **option A (monolithic `infer`)** with three internal hygiene moves
(committed as `26c8f39` and `efcc991`) instead of jumping straight to
extraction. The
reasoning, in one paragraph:

> Today's `bootstrap.rs` reaches into `crate::model::*`, `crate::scheduler::*`,
> `crate::tokenizer::Tokenizer`, and `crate::model_registry::*` to load and
> dispatch model-specific CUDA engines. Extracting `backend/cuda/` now would
> force `Tokenizer` / `KVCacheDtype` / `ModelType` / `Qwen3Model` /
> `Qwen35Model` to become `pub` cross-crate (it is currently
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

These moves made the subsequent extraction cheap because the seams were
already where they needed to be. The §5 procedure below assumes those seams
and they held during execution — the `pub(crate) → pub` flip in `prelude.rs`
was a one-line change per type, exactly as §4 predicted.

---

## 2 · Trip wires (any one of these → execute)

These are not hypotheticals. They are all on `ROADMAP.md::Missing` and the
project is committed to following SGLang/vLLM in capability:

| # | Trigger | Why it forces extraction |
|---|---|---|
| **T1** | **Parallel kernel build configs producing incompatible `.a` artifacts** | E.g. when FA-3 H100 work introduces sm_90-only kernel paths that conflict with sm_80 fallbacks, or when a vendored FlashInfer fork needs a different include search path. A single `infer/build.rs` cannot ship two different `libkernels_cuda.a` flavors. Extraction lets each artifact live in its own crate (or feature-gated within one). |
| **T2** | **Adding NCCL tensor parallel communication** | `ROADMAP.md` §"Missing" explicitly calls this out. NCCL pulls in `libnccl` linkage + multi-GPU coordination kernels. Today the build script already has feature gates piling up (`cuda`, `no-cuda`, `metal`, `cpu`, `rdma-nixl`, `rdma-nixl-real`); adding `nccl` on top will start producing combinatorial build matrices that don't fit cleanly under one crate's feature flags. |
| **T3** | **FlashAttention-3 (H100 / sm_90) prefill** | `ROADMAP.md` §1.2. FA-3 needs warp specialization + async pipeline. The current `prefill_attention.cu` + `flashinfer_single_prefill` path is sm_80-tuned. Adding FA-3 means **two parallel attention kernel implementations selected by SM target** — i.e., trip wire T1 follows. |
| **T4** | **MLA attention (DeepSeek-V2/V3/R1) + DeepSeek MoE + FP8 GEMM (sm_90)** | `ROADMAP.md` §1.3 + §1.4. This is essentially a second compute backend: new attention algorithm (latent KV compression), new GEMM family (FP8 E4M3), new model family. ~30+ new kernel files. The `crates/cuda-kernels/csrc/` tree doubles in size, and the parallel kernel set forces T1 / T2 anyway. |
| **T5** | **Speculative decoding GPU integration** | `ROADMAP.md` §"Missing". Today `infer/src/speculative.rs` and `infer/src/speculative/cuda.rs` are CPU stubs. Real GPU integration means a draft kernel surface that gets called from inside the scheduler hot loop. The existing scheduler→backend→cuda boundary will need cleanup before this lands cleanly. |
| **T6** | **A second team / external consumer starts owning only the kernel layer** | E.g. another inference project wants to reuse `cuda-kernels` directly without pulling in the agent runtime. The "two direct consumers" admission criterion from `docs/archives/art-grade-architecture-for-long-agent-infer.md` §六 is met. |

**Decision rule:** if **any one** of T1–T6 enters in-flight implementation
status (i.e., a real PR opens against it), execute this blueprint *as part of
that PR's preparation*, not as a separate refactor pass. The extraction must
land **before** the new work starts, so the new code is written against the
extracted crate from day one.

---

## 3 · Target topology (as actually landed)

> **Divergence from the pre-extraction draft of this section.** The earlier
> draft kept an `infer/csrc/cuda/` → `crates/cuda-kernels/csrc/cuda/`
> rename during the move. At execution time we flattened that — the `cuda/`
> subdirectory was dropped and the domain folders (`attention/`, `gemm/`,
> `kv/`, `misc/`, `quant/`) became direct children of `csrc/`. Everything
> else below matches the final layout.

```
agent-infer/                        ← workspace root (unchanged)
├── crates/
│   ├── agent/                ← unchanged
│   ├── chat/                 ← unchanged
│   ├── cli/                  ← unchanged
│   ├── cuda-kernels/         ← shipped
│   │   ├── Cargo.toml              ← cudarc + cc + flashinfer detection;
│   │   │                              `cuda` and `no-cuda` features mirror infer's
│   │   ├── build.rs                ← lifted from infer/build.rs (nvcc + Triton AOT)
│   │   ├── csrc/                   ← lifted from infer/csrc/cuda/ (flattened)
│   │   │   ├── attention/          ← flashinfer_*, fused_attention, prefill_attention, …
│   │   │   ├── gemm/               ← gemv, quantized_gemv, marlin_*, turboquant_weight_gemv
│   │   │   ├── kv/                 ← paged_kv_append, kv_cache_to_paged, kv_quant, scatter_kv
│   │   │   ├── quant/              ← turboquant, turboquant_fast, dtype_convert
│   │   │   ├── misc/               ← norm, sampling, pos_enc, conv1d, gdr, fused_mlp, …
│   │   │   └── common.cuh
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
│   │       ├── graph_pool.rs
│   │       ├── kv_quant.rs / kv_turboquant.rs / kv_types.rs / turboquant_state.rs
│   │       └── prelude.rs          ← THE public API surface (see §4)
│   ├── tools/                ← unchanged
│   └── mlx-sys/                    ← unchanged
├── infer/                          ← thin runtime + HTTP shell
│   ├── build.rs                    ← stripped of nvcc + Triton; only the
│   │                                  pure-Rust feature pass-through
│   ├── csrc/                       ← (gone — moved to cuda-kernels)
│   ├── tools/                      ← (gone — moved to cuda-kernels)
│   └── src/
│       ├── backend/
│       │   ├── cuda.rs             ← `pub use infer_cuda_kernels::*;` re-export shim
│       │   └── cuda/
│       │       └── bootstrap.rs    ← STAYS in infer (uses model + scheduler)
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
**`infer → cuda-kernels`, never the reverse.**

---

## 4 · The public API surface

The `cuda-kernels::prelude` module is the **only** thing `infer`'s
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
git checkout -b extract/cuda-kernels
cargo fmt --all -- --check
cargo check -p infer --no-default-features --features cuda,no-cuda
cargo check --workspace --no-default-features --features cpu,no-cuda
cargo check --workspace --no-default-features --features metal
```

Capture before-snapshot benchmarks per CLAUDE.md `Benchmark Rules`:
`docs/experience/wins/YYYY-MM-DD-bench-pre-cuda-extraction.md`.

### 5.2 — Move files (what actually ran)

The actual extraction flattened the `csrc/cuda/` subdirectory during the
move, so the domain folders (`attention/`, `gemm/`, `kv/`, `misc/`,
`quant/`) ended up as direct children of `csrc/` inside the new crate:

```
git mv infer/csrc/cuda/attention            crates/cuda-kernels/csrc/attention
git mv infer/csrc/cuda/gemm                 crates/cuda-kernels/csrc/gemm
git mv infer/csrc/cuda/kv                   crates/cuda-kernels/csrc/kv
git mv infer/csrc/cuda/misc                 crates/cuda-kernels/csrc/misc
git mv infer/csrc/cuda/quant                crates/cuda-kernels/csrc/quant
git mv infer/csrc/cuda/common.cuh           crates/cuda-kernels/csrc/common.cuh
git mv infer/tools/triton                   crates/cuda-kernels/tools/triton
git mv infer/src/backend/cuda/ffi.rs        crates/cuda-kernels/src/ffi.rs
git mv infer/src/backend/cuda/ffi           crates/cuda-kernels/src/ffi
git mv infer/src/backend/cuda/tensor.rs     crates/cuda-kernels/src/tensor.rs
git mv infer/src/backend/cuda/paged_kv.rs   crates/cuda-kernels/src/paged_kv.rs
git mv infer/src/backend/cuda/flashinfer.rs crates/cuda-kernels/src/flashinfer.rs
git mv infer/src/backend/cuda/graph_pool.rs crates/cuda-kernels/src/graph_pool.rs
git mv infer/src/backend/cuda/prelude.rs    crates/cuda-kernels/src/prelude.rs
```

`bootstrap.rs` stays at `infer/src/backend/cuda/bootstrap.rs`.

### 5.3 — New `crates/cuda-kernels/Cargo.toml`

Mirror `infer/Cargo.toml`'s CUDA deps:

```toml
[package]
name = "cuda-kernels"
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
`crates/cuda-kernels/build.rs`. The FlashInfer detection helper
(`find_flashinfer_include`) goes with it. Strip the corresponding lines from
`infer/build.rs`. The `infer/build.rs` becomes a small no-op (or is deleted
entirely if no other build-time work remains).

### 5.5 — Update `infer/Cargo.toml`

```toml
[dependencies]
cuda-kernels = { path = "../crates/cuda-kernels", default-features = false, optional = true }

[features]
cuda = ["dep:cudarc", "dep:memmap2", "cuda-kernels/cuda"]
no-cuda = ["cuda-kernels/no-cuda"]
```

`cudarc` and `memmap2` stay in `infer/Cargo.toml` because `bootstrap.rs` and
the rest of the runtime still use them. They are also pulled in transitively
through `cuda-kernels`, but Cargo will dedupe.

### 5.6 — Re-pub the prelude exports

In each of `crates/cuda-kernels/src/{tensor,paged_kv,flashinfer,graph_pool}.rs`,
change `pub(crate)` → `pub` on the items listed in the prelude (and `TokenKVPool`).
Everything else stays `pub(crate)` to preserve encapsulation.

In `crates/cuda-kernels/src/lib.rs`:

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
//! Re-exports of the `cuda-kernels` crate so existing
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
cargo check -p cuda-kernels --no-default-features --features cuda,no-cuda
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
INFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release --test e2e
cargo test --release --test e2e_qwen35
scripts/bench_guidellm.sh post-cuda-extraction
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

The split's **load-bearing principle**: a file goes in `cuda-kernels`
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
  `Qwen35Model` types in bootstrap. Splitting it out would
  re-create the bootstrap straddle problem.
- **No `infer-runtime-api` trait crate.** Already covered by
  `infer::server_engine::InferenceEngine`. Putting it in a separate crate
  with no second consumer would re-create the Route-A failure mode.
- **No `infer-cuda` (Rust-only) AND `cuda-kernels` (native-only) split.**
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
3. Delete `crates/cuda-kernels/`.
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
