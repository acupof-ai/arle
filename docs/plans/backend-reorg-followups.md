# Round 2 backend reorg — deferred follow-ups

Round 2 (April 2026) moved all backend-related code under `infer/src/backend/{cuda,metal,cpu,runtime}/`. The reorg was kept purely structural so every commit remained reviewable and bisectable. Two known debts were consciously deferred; tracking them here so the next pass picks them up cleanly.

## F1 · Split `backend/metal.rs` (1756 lines)

**Current state.** `infer/src/backend/metal.rs` is a fat facade — it both declares the 8 `backend/metal/*` submodules and hosts the main `MetalBackend` struct, its entire `InferenceBackend` + `StreamingInferenceBackend` implementations, all the free helper functions (`linear`, `rms_norm_last_dim`, `gpu_sample_token`, `extend_kv_cache`, `merge_quantized_projection_rows`, etc.), and the `WeightTensor` enum. A single file at 1756 lines that mixes six logical concerns.

**Why we didn't split it in Round 2.** A split is an internal re-arrangement of a single file — risk-disjoint from the file relocation Round 2 was doing. Mixing them in one commit would (a) make the diff unreviewable (every rename + every move looks like a rewrite), (b) risk lost semantics during the cluster boundary choice, and (c) prevent clean bisecting if a regression slips in. Round 2 stays mechanical; the split gets its own round.

**Proposed split.** Target: each new file ≤ ~400 lines, no semantic change, `cargo test` green on metal.

```
backend/metal.rs                ← thin facade: trait impls, submodule declarations, struct definition
backend/metal/
  ├── weights.rs                ← `WeightTensor` enum + quantized projection helpers
  ├── ops.rs                    ← `linear`, `rms_norm_last_dim`, `gpu_sample_token`
  ├── kv_cache.rs               ← `extend_kv_cache`, related layout helpers
  ├── generate.rs               ← generation loop, sampling dispatch, early stop
  └── (existing files unchanged: config, loader, gdr, kv_pool, mlx, prefix_cache, qwen35, scheduler)
```

**Exit criteria.**
1. `cargo check -p infer --no-default-features --features metal` green.
2. `cargo test -p infer --no-default-features --features metal --lib --release -- --test-threads 1` matches pre-split snapshot (same pass/fail, same timings ± noise).
3. `git log --follow` on `backend/metal.rs` still traces back through the split (use `git mv` + minimal rewrites so rename detection fires).
4. `metal_bench` + `metal_request` bins still compile and link.

## F2 · Audit `backend/cuda/graph_pool.rs` for dead code

**Current state.** Every public item in `backend/cuda/graph_pool.rs` (`GraphPool`, `POOL_BATCH_SIZES`, `pad_to_pool_size`, `is_graph_eligible`, `padding_slots`, `warmup_schedule`, `MAX_GRAPH_BATCH_SIZE`) has **zero consumers anywhere in the tree** — the only references are the file's own doc comments and its own unit tests. The file is 441 lines.

**How we confirmed.** During Round 2's CUDA relocation we grep-verified:
- No `crate::cuda_graph_pool::*` or `crate::backend::cuda::graph_pool::*` reference from any other source file.
- No `infer::cuda_graph_pool::*` or `infer::backend::cuda::graph_pool::*` reference from any bin / test / bench / workspace crate.
- No name-based import (`use …::GraphPool` or similar).

**Options.**
- **Delete the module** if it turns out to be legacy from an abandoned pre-FlashInfer batching path.
- **Wire it in** if the scheduler was always supposed to use it and nobody finished the integration.
- **Keep it** if it's a CPU-testable scaffold for a planned future feature — in which case add a `TODO(owner, deadline)` comment explaining the intent so the next reader doesn't go through the same archaeology.

**Process.** Open a lightweight investigation commit (`git blame` the original author, check `ROADMAP.md` / `docs/projects/` for CUDA-graph references). Decide one of the three options in a single follow-up PR. Do **not** fold this into the metal split — they are unrelated concerns.

## F3 · CUDA type-checking on non-CUDA hosts

**Discovery.** `cargo check -p infer --no-default-features --features cuda,no-cuda` engages rustc on all `#[cfg(feature = "cuda")]`-gated Rust code (because `cuda` is enabled) while still skipping nvcc compilation (because `no-cuda` is enabled in `build.rs`). This was not documented anywhere but is the right invocation for Darwin / Linux-without-nvcc developers to validate CUDA-only refactors.

**Action.** Add the invocation to `infer/CLAUDE.md` under the Tests section so future refactors don't have to re-discover it. One-line change; not worth its own commit, bundle with the next docs update.
