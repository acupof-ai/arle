# F1 · Split `infer/src/backend/metal.rs` into topical submodules

**Owner:** codex (local Mac, `--features metal`)
**Prerequisite commit:** `7a876e1` (Round 2 housekeeping) — this plan assumes the tree is at or after that commit.
**Status:** ready to execute
**Round 3 note:** the old extracted-runtime plan was reverted by the 2026-04-15 Route-A refactor, so this warning is historical only.

---

## 1. Why

`infer/src/backend/metal.rs` is **1766 lines**. Round 2 moved it from the flat `src/metal_backend.rs` position into `src/backend/metal.rs`, but the contents were left untouched because a split is internally risky and was worth doing in its own commit for clean `git blame` attribution.

One file currently mixes six logical concerns:

1. The `MetalBackend` struct definition + its `impl`, `impl Default`, `impl InferenceBackend`, `impl StreamingInferenceBackend`, and the `run_with_metal_panic_boundary` helper — the public façade.
2. Weight-type declarations (`WeightTensor`, `AttentionInputProjection`, `MlpInputProjection`, `StandardMetalWeights`, `MetalWeights`, `StandardMetalLayerWeights`).
3. Weight loading / merging logic (`load_qwen3_metal_weights`, `build_qwen3_cpp_model`, `merge_quantized_projection_rows`).
4. The autoregressive generation loop (`metal_generate` + its KV-pool lifecycle guard).
5. Per-step forward computation (`build_forward_graph`, `rust_transformer_layer`).
6. Low-level tensor primitives (`linear`, `metal_async_eval`, `clear_metal_cache`, `extend_kv_cache`) and sampling helpers.

Mixing these makes `git blame` noisy, pushes each cognitive load above what a reviewer can hold, and forces every future PR that touches one concern to re-read the entire file to confirm it isn't breaking another.

Goal: finish the Round 2 reorganisation by pushing each concern into its own file under `backend/metal/`. Net behavioural change: **zero**. Net file count change: +5 (metal.rs shrinks from 1766 to ~440 lines). This is a pure refactor.

---

## 2. Target layout

```
infer/src/backend/metal.rs                   ← ~440 lines (facade)
infer/src/backend/metal/
  ├── config.rs                              ← unchanged (existed before Round 2)
  ├── loader.rs                              ← unchanged
  ├── gdr.rs                                 ← unchanged (Round 2 relocation)
  ├── kv_pool.rs                             ← unchanged (Round 2 relocation)
  ├── prefix_cache.rs                        ← unchanged (Round 2 relocation)
  ├── scheduler.rs                           ← unchanged (Round 2 relocation)
  ├── mlx.rs                                 ← unchanged (Round 2 relocation)
  ├── qwen35.rs                              ← unchanged
  ├── weights.rs       ★ NEW  ~780 lines
  ├── generate.rs      ★ NEW  ~320 lines
  ├── forward.rs       ★ NEW  ~185 lines
  ├── ops.rs           ★ NEW  ~75  lines
  └── sampling.rs      ★ NEW  ~80  lines
```

Every new file inherits the same `#[cfg(feature = "metal")]` gating pattern that the moved items already have on their individual declarations. There is no new cfg gate at the submodule level; the internal per-item cfgs stay in place.

---

## 3. Exact function → file assignment

Line numbers below reference `backend/metal.rs` **as of commit `7a876e1`**. If the file has drifted by the time you start (check `git log --oneline infer/src/backend/metal.rs`), re-derive the boundaries with `grep -n "^pub fn\|^fn \|^pub(crate) fn\|^impl\|^pub struct\|^struct \|^pub enum\|^enum " infer/src/backend/metal.rs` first and update the numbers before cutting.

### 3.1 `metal.rs` (stays — becomes the thin facade, ~440 lines)

Keep **only** the following, in roughly this order:

| Lines (current) | Item |
|---|---|
| 1 – 87 | Module docs, all `use` imports, `KV_CACHE_CHUNK` / `METAL_KV_POOL_REQUEST_ID` / `BENCHMARK_PROMPT_CHUNK` constants (see §3.7 for which consts move) |
| 88 – 101 | `pub struct MetalBackend { ... }` |
| 302 – 466 | `impl MetalBackend { ... }` — `new`, `generate_stream`, `generate_from_token_ids`, `generate_from_token_ids_with_callback`, `benchmark_prompt_ids` |
| 468 – 492 | `run_with_metal_panic_boundary` + `panic_message` (both cfg variants) |
| 494 – 498 | `impl Default for MetalBackend` |
| 500 – 505 | `unsafe impl Send for MetalBackend` (cfg-gated) |
| 507 – 593 | `impl InferenceBackend for MetalBackend` |
| 595 – 608 | `impl StreamingInferenceBackend for MetalBackend` |

Add at the top of the file, below the existing `pub mod config / loader / qwen35 / gdr / kv_pool / prefix_cache / scheduler / mlx` declarations:

```rust
#[cfg(feature = "metal")]
pub mod forward;
#[cfg(feature = "metal")]
pub mod generate;
#[cfg(feature = "metal")]
pub mod ops;
#[cfg(feature = "metal")]
pub mod sampling;
#[cfg(feature = "metal")]
pub mod weights;
```

(Alphabetical; rustfmt's import-group ordering will not reorder `pub mod` declarations but the file reads better this way.)

Replace the old free-function references at existing call sites inside the retained code with `self::generate::metal_generate`, `self::weights::load_qwen3_metal_weights`, etc. There are a handful of such call sites inside `impl MetalBackend::load()` (around line 558) and inside `generate_from_token_ids_with_callback()` (around line 382). Leave everything else untouched.

### 3.2 `backend/metal/weights.rs` (NEW, ~780 lines)

All weight types + all weight-loading logic.

| Source lines | Item | New visibility |
|---|---|---|
| 103 – 125 | `pub enum WeightTensor` | `pub` → `pub` (unchanged, used by other submodules) |
| 127 – 159 | `impl WeightTensor` | unchanged |
| 161 – 217 | `pub enum AttentionInputProjection` + `impl` (including `kv_dtype` and `project`) | `pub` |
| 219 – 260 | `pub enum MlpInputProjection` + `impl project` | `pub` |
| 262 – 276 | `struct StandardMetalWeights` | `pub(super)` |
| 278 – 282 | `enum MetalWeights` | `pub(super)` |
| 284 – 300 | `struct StandardMetalLayerWeights` | `pub(super)` |
| 1205 – 1319 | `fn merge_quantized_projection_rows` | `pub(super) fn` |
| 1326 – 1443 | `fn load_qwen3_metal_weights` | `pub(super) fn` |
| 1444 – 1766 | `fn build_qwen3_cpp_model` | module-private (`fn`), only called by `load_qwen3_metal_weights` |

Required imports at the top:
```rust
use anyhow::{Context, Result};
use std::path::Path;

use super::config::{MetalModelConfig, QuantConfig};
use super::loader::{
    TensorMap, load_embed_tokens_from_tensors, load_proj_from_tensors, load_tensor_map,
    tensor_get, tie_lm_head_from_embed_tokens,
};
use super::mlx::{Dtype, MlxArray, concatenate_axis, eval};
use super::qwen35;
```

All items stay `#[cfg(feature = "metal")]` exactly as they are in the source file.

### 3.3 `backend/metal/generate.rs` (NEW, ~320 lines)

The generation loop and its KV-pool guard.

| Source lines | Item | New visibility |
|---|---|---|
| 637 – 640 | `KV_CACHE_CHUNK` + `METAL_KV_POOL_REQUEST_ID` consts | `pub(super) const` |
| 643 – 654 | `struct MetalGenerateOutput` | `pub(super) struct` |
| 656 – 661 | `fn metal_kv_pool_enabled` | `pub(super) fn` |
| 663 – 667 | `struct MetalKvPoolRequestCleanup` | module-private |
| 669 – 692 | `impl MetalKvPoolRequestCleanup` + `impl Drop` | module-private |
| 694 – 702 | `fn metal_kv_pool_flag_is_truthy` | module-private |
| 703 – 910 | `fn metal_generate` | `pub(super) fn` |

Required imports:
```rust
use std::time::Instant;

use anyhow::{Context, Result};

use super::config::MetalModelConfig;
use super::forward::build_forward_graph;
use super::kv_pool::MetalKVPool;
use super::mlx::{Dtype, MlxArray, argmax, eval};
use super::ops::{clear_metal_cache, metal_async_eval};
use super::sampling::{gpu_sample_token, validate_metal_sampling_params};
use super::weights::{MetalWeights, StandardMetalWeights};
use crate::sampler::SamplingParams;
```

The actual import set will depend on what `metal_generate` uses. When extracting, start with the above and let rustc tell you what else is needed — add imports one at a time until it compiles.

### 3.4 `backend/metal/forward.rs` (NEW, ~185 lines)

Per-step forward computation. Both functions are called by `metal_generate`.

| Source lines | Item | New visibility |
|---|---|---|
| 952 – 1015 | `fn build_forward_graph` | `pub(super) fn` |
| 1016 – 1121 | `fn rust_transformer_layer` | `pub(super) fn` |

Required imports:
```rust
use super::config::MetalModelConfig;
use super::mlx::{MlxArray, ...};  // fill in as needed
use super::ops::{extend_kv_cache, linear};
use super::weights::{StandardMetalLayerWeights, StandardMetalWeights};
```

### 3.5 `backend/metal/ops.rs` (NEW, ~75 lines)

Low-level tensor primitives shared by forward.rs, generate.rs, weights.rs, and sampling.rs.

| Source lines | Item | New visibility |
|---|---|---|
| 913 – 917 | `fn metal_async_eval` | `pub(super) fn` |
| 919 – 923 | `fn clear_metal_cache` | `pub(super) fn` |
| 924 – 951 | `fn extend_kv_cache` | `pub(super) fn` |
| 1185 – 1202 | `fn linear` | `pub(super) fn` |

Required imports:
```rust
use anyhow::Result;

use super::mlx::{MlxArray, async_eval, clear_cache, matmul, quantized_matmul};
use super::weights::WeightTensor;
```

### 3.6 `backend/metal/sampling.rs` (NEW, ~80 lines)

Sampling parameter validation + per-step token selection.

| Source lines | Item | New visibility |
|---|---|---|
| 1122 – 1157 | `fn validate_metal_sampling_params` | `pub(super) fn` |
| 1158 – 1169 | `fn gpu_sample_token` | `pub(super) fn` |
| 1170 – 1174 | `fn greedy_sample_token` | module-private |
| 1175 – 1184 | `fn categorical_sample_token` | module-private |

Required imports:
```rust
use anyhow::{Result, bail};

use super::mlx::{MlxArray, argmax, categorical, multiply};
use crate::sampler::SamplingParams;
```

### 3.7 Constant placement decisions

- `KV_CACHE_CHUNK: i32 = 256` — used only inside `metal_generate` and `extend_kv_cache`. **Move to `generate.rs`** as `pub(super) const`. Re-export into ops.rs via `use super::generate::KV_CACHE_CHUNK` if `extend_kv_cache` needs it (it does, for chunk-aligned allocation math).
- `METAL_KV_POOL_REQUEST_ID: usize = 0` — used only inside `metal_generate`. **Move to `generate.rs`**, module-private.
- `BENCHMARK_PROMPT_CHUNK: &str = " benchmark throughput"` — used only inside `impl MetalBackend::benchmark_prompt_ids`. **Stays in `metal.rs`**.

---

## 4. Execution order (suggested commit sequence)

Break into 5 commits so each one can be bisected. Each commit must compile and pass `cargo check -p infer --no-default-features --features metal` plus the Metal unit test suite (`cargo test -p infer --no-default-features --features metal --lib -- --test-threads 1`).

### Commit 1: `refactor(metal): extract weights module`

1. Create `infer/src/backend/metal/weights.rs` with the items listed in §3.2.
2. Add `pub mod weights;` to `infer/src/backend/metal.rs`.
3. In `metal.rs`, update the call sites that reference `WeightTensor`, `AttentionInputProjection`, etc. to use `self::weights::*` (or leave the bare names and add `use self::weights::*;` — match the adjacent style).
4. Delete the extracted items from `metal.rs`.
5. Verify: `cargo check --features metal`, `cargo test --features metal --lib`.
6. Commit message: `refactor(metal): extract weight types and loaders into backend/metal/weights.rs`.

### Commit 2: `refactor(metal): extract ops + sampling primitives`

1. Create `infer/src/backend/metal/ops.rs` and `infer/src/backend/metal/sampling.rs` with the items from §3.5 and §3.6.
2. Add `pub mod ops;` and `pub mod sampling;` to `metal.rs`.
3. Update any remaining `metal.rs` code that calls these functions to use `self::ops::*` / `self::sampling::*`.
4. Delete the extracted items from `metal.rs`.
5. Verify: `cargo check --features metal`, `cargo test --features metal --lib`.
6. Commit message: `refactor(metal): extract linear/eval/sampling helpers into submodules`.

### Commit 3: `refactor(metal): extract forward graph builders`

1. Create `infer/src/backend/metal/forward.rs` with `build_forward_graph` and `rust_transformer_layer`.
2. Add `pub mod forward;` to `metal.rs`.
3. Update `metal.rs` callers (the only caller is `metal_generate` which is still in metal.rs at this point — it will reference `self::forward::build_forward_graph`).
4. Delete the extracted items from `metal.rs`.
5. Verify: `cargo check --features metal`, `cargo test --features metal --lib`.
6. Commit message: `refactor(metal): extract per-step forward graph into backend/metal/forward.rs`.

### Commit 4: `refactor(metal): extract generate loop`

1. Create `infer/src/backend/metal/generate.rs` with the generate loop and its KV-pool guard (§3.3).
2. Move `KV_CACHE_CHUNK` and `METAL_KV_POOL_REQUEST_ID` into `generate.rs`; ensure `ops.rs::extend_kv_cache` reaches the constant via `use super::generate::KV_CACHE_CHUNK`.
3. Add `pub mod generate;` to `metal.rs`.
4. Update `impl MetalBackend::generate_from_token_ids_with_callback` in metal.rs to call `self::generate::metal_generate(...)` instead of the now-gone free fn.
5. Delete the extracted items from `metal.rs`.
6. Verify: `cargo check --features metal`, `cargo test --features metal --lib`.
7. Commit message: `refactor(metal): extract generation loop into backend/metal/generate.rs`.

### Commit 5: `style(metal): rustfmt and final line-count assertion`

1. Run `cargo fmt --all`.
2. Verify line counts roughly match the targets in §2. If a file is >600 lines consider whether it should split further (ask first — don't over-refine).
3. Run the full verification suite in §5.
4. Commit message: `style(metal): rustfmt post-split + line count verification`.

If any commit's scope turns out bigger or smaller than expected, **adjust but don't skip verification between commits**. Each commit must stand on its own with `cargo check --features metal` green.

---

## 5. Verification

Run after every commit and before declaring F1 done:

```bash
# 1. Structural type-check for the three feature combinations we care about
cargo check -p infer --no-default-features --features metal
cargo check -p infer --no-default-features --features no-cuda
cargo check -p infer --no-default-features --features cuda,no-cuda  # rustc-only CUDA path

# 2. Metal unit tests (actually exercises the code — don't skip this one)
cargo test -p infer --no-default-features --features metal --lib -- --test-threads 1

# 3. Bins that depend on Metal must still link
cargo check -p infer --no-default-features --features metal --bin metal_bench
cargo check -p infer --no-default-features --features metal --bin metal_request
cargo check -p infer --no-default-features --features metal --bin metal_serve

# 4. Full-workspace sanity (catches any accidentally-public item that downstream crates reference)
cargo check --workspace --no-default-features --features metal

# 5. Clippy pedantic (project lints against warnings)
cargo clippy -p infer --no-default-features --features metal -- -D warnings

# 6. Line count assertion
wc -l infer/src/backend/metal.rs infer/src/backend/metal/weights.rs \
      infer/src/backend/metal/generate.rs infer/src/backend/metal/forward.rs \
      infer/src/backend/metal/ops.rs infer/src/backend/metal/sampling.rs
```

Expected line count ranges (rough):

| File | Target | Acceptable range |
|---|---|---|
| `metal.rs` | 440 | 380 – 520 |
| `weights.rs` | 780 | 700 – 850 |
| `generate.rs` | 320 | 280 – 380 |
| `forward.rs` | 185 | 160 – 230 |
| `ops.rs` | 75 | 60 – 120 |
| `sampling.rs` | 80 | 65 – 110 |

If `weights.rs` exceeds 850 lines or `metal.rs` exceeds 520 lines, reconsider the split — probably something leaked into the wrong file.

---

## 6. Constraints & pitfalls

1. **Do not rename any existing file under `backend/metal/`** (e.g. don't touch `config.rs`, `loader.rs`, `qwen35.rs`, `gdr.rs`, `kv_pool.rs`, `prefix_cache.rs`, `scheduler.rs`, `mlx.rs`). Those are from Round 2 and have external call sites that depend on their current names.
2. **Do not change any item's cfg gating.** Every item that carries `#[cfg(feature = "metal")]` today must carry the same attribute after the split. Do not "simplify" or "hoist" the cfg to the module level — the existing per-item granularity is load-bearing for the `no-cuda` build which compiles the module but with most items removed.
3. **Do not touch `backend.rs` (the trait file), `lib.rs`, or anything under `backend/cuda/`, `backend/cpu.rs`, `backend/runtime.rs`, `ops/`, `model/`, `scheduler/`, `http_server/`, `server_engine.rs`, `weight_loader.rs`.** F1 is contained to `backend/metal*`. Any diff touching code outside that scope is out of bounds.
4. **Do not modify `crates/mlx-sys/`.** It has in-progress work from another contributor.
5. **Do not change item signatures, add new public APIs, or "fix" naming opportunistically.** This is a pure move. Even tempting cleanups like shortening `metal_generate` to `generate` must wait for a follow-up.
6. **Follow flat layout.** `backend/metal.rs` + `backend/metal/{weights,generate,forward,ops,sampling}.rs`. No `mod.rs`. (This repo enforces it in CLAUDE.md.)
7. **Preserve git history.** Use `git mv` when creating new files out of extracted content if you first create empty files and `git mv` them — but in practice the new files are created via `Write` (new content), and the deletions from metal.rs are modifications. Git's rename detection will NOT kick in here because the new files are < 50% similar to the old metal.rs. That's fine; history is preserved at the commit level because each Commit 1–4 pairs an insertion with its corresponding deletion in a single commit, and `git log -L` can follow a function through the move.
8. **Commitizen format** for every commit: `refactor(metal): …` or `style(metal): …`.
9. **Never force-push.** If a commit is wrong, make a new commit that fixes it, don't amend. The project's rule is new-commit-over-amend.
10. **Do not skip hooks.** The cargo-fmt PostToolUse hook in this repo will rewrite imports after each edit. After each commit, run `git status` — if a file shows as `M` that you didn't mean to touch, investigate (usually it's the formatter and re-staging is fine; see `~/.claude/.../memory/feedback_git_mv_with_fmt_hook.md` for the pattern).

---

## 7. What "done" looks like

A sequence of 5 commits (numbered in §4) on top of `7a876e1` or whatever `main` is at F1 start. `infer/src/backend/metal.rs` reduced from 1766 to ~440 lines. Five new files under `infer/src/backend/metal/`. All verification in §5 green.

After the final commit, add an entry to `docs/plans/backend-reorg-followups.md` closing out F1 with the final commit SHA, and optionally add a postmortem note to `docs/experience/wins/YYYY-MM-DD-metal-split.md` if anything interesting came up.

**Do not push to origin**. Let the human running you decide whether to push — they may want to bundle this with other work or run remote CUDA tests first.
