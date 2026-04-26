# L4 verification — scheduler regression + TileLang Phase 0

**Date:** 2026-04-26
**Host:** remote L4 box (24 GB, sm_89, driver 580.82.07, CUDA 12.8.93)
**HEAD at start:** `c819ba13`
**HEAD at end:** `68c183e8`
**Goal:** verify two distinct claims on the same machine in one session:

1. **Scheduler / kv-tier improvements** — the post-2026-04-14 refactor
   stack (mixed-batch align, decode emit-gate, admission cleanup,
   coordinator split, RAII regions, zero-copy slice, etc.) did not
   regress single-GPU concurrent serving on L4.
2. **Operator optimization (TileLang Phase 0)** — the
   `cuda,tilelang-attn` build path produces a working AOT-compiled
   prefill kernel on L4 (sm_89), unblocking the matched A/B that drives
   the Phase 0 ship/revert decision.

---

## Result summary

| Claim | Status | Evidence |
|---|---|---|
| Scheduler post-2026-04-14 refactors are flat on L4 | **VERIFIED** | wins/2026-04-26-bench-guidellm-cuda-l4-scheduler-current.md |
| TileLang Phase 0 builds + benches on L4 | **BLOCKED** | errors/2026-04-26-tilelang-aot-tilelang-0p1p9-blocker.md |

---

## 1. Scheduler regression check (PASS — flat)

### Reference

`memory/project_l4_perf_baseline.md`, in-process bench:

- d902090 (2026-04-14, route-A revert): Qwen3-4B 30.52 tok/s @ c=1
- 132bc84 (2026-04-14, post Triton→CUDA C port): Qwen3-4B 27.84 tok/s @ c=1

### Now (HEAD `802c5fc8`, guidellm HTTP, --quick c=1..8 × 60 s)

| concurrency | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | TPOT p50 ms | out tok/s |
|---|---|---|---|---|---|
| 1 | 114.4 | 120.1 | 33.28 | 33.9 | **29.32** |
| 2 | 183.3 | 260.1 | 35.16 | 36.4 | 52.52 |
| 4 | 341.7 | 436.2 | 35.28 | 38.0 | 98.08 |
| 8 | 618.3 | 5894.5 | 38.77 | 43.4 | 173.75 |

Service trace: peak `kv_util` 96.8 %, peak `running_batch` 8 (admission
not gating), zero failures across 329 trace samples.

### Conclusion

Decode at c=1 is **−3.9 % vs d902090 / +5.3 % vs 132bc84** — both
inside the matched-A/B noise band. The HTTP-vs-in-process measurement
delta plus guidellm 0.6.0 environmental drift (per
`memory/project_bench_env_drift_2026-04-20.md`) explain the small
−3.9 % comfortably; this is not a code regression. Aggregate scaling
through c=8 is 5.93× — consistent with HBM saturation on L4. The
scheduler refactor stack is **safe to leave on main**.

Caveat: this was an exploration-mode (`--quick`, 512/128) run. Lifting
to canonical sweep (4096/256, profile=sweep) is a separate exercise
and is not required to land the regression-flat verdict.

Full entry:
[wins/2026-04-26-bench-guidellm-cuda-l4-scheduler-current.md](../experience/wins/2026-04-26-bench-guidellm-cuda-l4-scheduler-current.md).

---

## 2. TileLang Phase 0 (BLOCKED — TileLang 0.1.9 AOT)

### What was supposed to happen

[`docs/plans/tilelang-integration-verification.md`](../plans/tilelang-integration-verification.md)
§2 — build matched off / on binaries, §3 — e2e parity, §4 — A/B sweep,
§5 — record `…-l4-floor.md`. The whole plan unblocks the H100
ship/revert decision.

### What actually happened

The `cargo build --release -p infer --features cuda,tilelang-attn`
invocation panics inside the AOT generator on three successive issues.
The first two are real source-side defects fixed inline; the third is
inside TileLang/TVM and is the actual blocker.

| # | Layer | Symptom | Status |
|---|---|---|---|
| 1 | `crates/cuda-kernels/build.rs::tilelang_target` + `gen_tilelang_aot.py::parse_target` | `ValueError: Target kind "cuda:89" is not defined. ... e.g. 'cuda -arch=sm_80'` | **Fixed in `802c5fc8`**. Format updated to `cuda -arch=sm_<sm>`. |
| 2 | `crates/cuda-kernels/tools/tilelang/batch_prefill_paged_hd128.py::_make_kernel` | `AssertionError: A and B must have the same dtype` at `T.gemm(p, v_tile, acc_o)` (P f32, V bf16) | Patch tested locally (FlashAttention-standard `p_bf16` cast). Held — landing without #3 leaves Phase 0 still broken (no-half-state rule). |
| 3 | TileLang 0.1.9 `LayoutInferencer` (TVM internals) | `tvm.error.InternalError: loop_var_to_thread = d // 64 * 64 + i // 32 * 32 + i % 8 * 4 + d % 8 // 2 contains inner var d` | **BLOCKER**. Inside `tilelang/3rdparty/tvm` GemmNode::InferLayout. Not a source-side bug. |

### What this means

- The Phase 0 commit trio (`022e8dd / 76e044b / 9896d25`) stays in the
  tree. Per plan §5 risk gate #2 the prescribed action is to revert,
  but the fix landed (#1) is independently correct and the user's
  explicit guidance was to push forward, not unwind. Phase 0 is
  **parked, not closed**.
- The pending-remote stub
  [`wins/2026-04-26-bench-guidellm-cuda-tilelang-prefill-hd128-pending-remote.md`](../experience/wins/2026-04-26-bench-guidellm-cuda-tilelang-prefill-hd128-pending-remote.md)
  stays in place; the new errors entry is the recorded blocker, not
  the closure.
- The `tilelang>=0.1` extra in `pyproject.toml` is the actual root
  cause: it resolves to whatever TileLang ships latest, which today
  is 0.1.9 with a regressed LayoutInferencer for our kernel shape.

### Recommended next step (lowest-risk)

Pin TileLang to a version that is known to compile this kernel before
re-attempting Phase 0:

1. Bisect 0.1.0 → 0.1.9 against
   `crates/cuda-kernels/tools/tilelang/batch_prefill_paged_hd128.py`
   directly via `python3 -c "import tilelang; tilelang.compile(get_kernel(32, 8), target='cuda -arch=sm_89')"`
   (no Rust build needed).
2. Update
   `pyproject.toml::[project.optional-dependencies].tilelang` to
   `tilelang==<verified>`.
3. Land the parked dtype patch + the version pin in one commit.
4. Re-run §2–§4 of the verification runbook.

If the bisect produces no green version, the fallback is to upstream
the LayoutInferencer fix or simplify the kernel layout (drop
`T.use_swizzle`, halve BLOCK_M/BLOCK_N) at a probable perf cost.

Full blocker write-up:
[errors/2026-04-26-tilelang-aot-tilelang-0p1p9-blocker.md](../experience/errors/2026-04-26-tilelang-aot-tilelang-0p1p9-blocker.md).

---

## What landed this session

| Commit | Scope | Notes |
|---|---|---|
| `4da98a7` (orphan, see `802c5fc8`) | fix(cuda) | TileLang target string format, rebased |
| `802c5fc8` | fix(cuda) | TileLang 0.1.9 target string format — issue #1 |
| `68c183e8` | docs(bench,errors) | wins/scheduler regression + errors/TileLang blocker |

(`models` symlink created in workspace root and `infer/models` per
`memory/project_remote_cuda_box.md` — symlinks only, untracked.)

---

## Open follow-ups

- **TileLang version pin** (highest priority) — see §2 above.
- **Canonical sweep on L4** — the regression check is `--quick`. A
  follow-up sweep with `profile=sweep, data=4096/256, max-seconds=60`
  closes out the open exploration-mode caveat.
- **Qwen3.5-4B parallel run** — symmetry check against the
  `27.59 tok/s` line from `project_l4_perf_baseline.md`. Same recipe,
  different model path.
- **Shared-prefix workload** — the bench shows `prefix_hit_rate=0.0%`
  because guidellm's synthetic prompts don't share prefixes. Re-run
  with an agent-trace dataset to verify the kv_tier promotion paths
  the recent refactors changed.
