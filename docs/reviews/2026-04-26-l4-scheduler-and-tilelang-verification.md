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
| Scheduler post-2026-04-14 refactors are flat on L4 | **VERIFIED — flat** | [`wins/2026-04-26-bench-guidellm-cuda-l4-scheduler-current.md`](../experience/wins/2026-04-26-bench-guidellm-cuda-l4-scheduler-current.md) |
| Mixed-batch refactors close prior c=4/c=8 backlog | **VERIFIED — wins** | [`wins/2026-04-26-bench-guidellm-cuda-l4-mixed-batch-vs-f98ca92.md`](../experience/wins/2026-04-26-bench-guidellm-cuda-l4-mixed-batch-vs-f98ca92.md) |
| TileLang Phase 0 builds + benches on L4 | **VERIFIED — functional, L4 floor** | [`wins/2026-04-26-bench-guidellm-cuda-l4-tilelang-prefill-hd128-floor.md`](../experience/wins/2026-04-26-bench-guidellm-cuda-l4-tilelang-prefill-hd128-floor.md); historical blocker chain in [`errors/2026-04-26-tilelang-aot-tilelang-0p1p9-blocker.md`](../experience/errors/2026-04-26-tilelang-aot-tilelang-0p1p9-blocker.md) |

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

## 1b. Mixed-batch (PASS — wins on c=4/c=8/c=16)

Re-ran the same workload as the 2026-04-22 SGLang comparison
(commit `f98ca92`, identical flags: `--num-slots 16 --max-seq-len 4608
--mem-fraction-static 0.94 --chunked-prefill-size 4096
--max-prefill-tokens 16384`). The post-2026-04-22 mixed-batch
refactors (`f526e10b` align with sglang, `27ba7308` decode emit-gate,
`df2d3e8e` workspace budget align) deliver:

| conc | TTFT p50 ms then | TTFT p50 ms now | Δ TTFT | tok/s then | tok/s now | Δ tok/s |
|---|---:|---:|---:|---:|---:|---:|
|  1 |   739.9 |    719.3 |  −2.8% | 26.59 |   26.56 |  −0.1% |
|  2 |  1485.0 |   1518.6 |  +2.3% | 41.59 |   45.21 |  +8.7% |
|  4 | 14556.7 |   2354.4 | **−83.8%** | 36.70 |   53.31 | **+45.3%** |
|  8 | 15403.7 |   3838.0 | **−75.1%** | 57.71 |   66.94 |  +16.0% |
| 16 | 15405.9 |  16356.9 |  +6.2% | 45.08 |   65.92 | **+46.2%** |

The c=4/c=8 TTFT collapse is the mixed-batch fingerprint — prefill no
longer starves decode while the running batch is held. Throughput at
c=16 closes ~15 percentage points of the prior SGLang gap (45 → 66
tok/s vs SGLang 137 tok/s; gap −67% → −52%).

Full entry:
[`wins/2026-04-26-bench-guidellm-cuda-l4-mixed-batch-vs-f98ca92.md`](../experience/wins/2026-04-26-bench-guidellm-cuda-l4-mixed-batch-vs-f98ca92.md).

---

## 2. TileLang Phase 0 (PASS — functional, L4 floor)

### What was supposed to happen

[`docs/plans/tilelang-integration-verification.md`](../plans/tilelang-integration-verification.md)
§2 — build matched off / on binaries, §3 — e2e parity, §4 — A/B sweep,
§5 — record `…-l4-floor.md`. The whole plan unblocks the H100
ship/revert decision.

### What actually happened

A four-issue chain, all closed.

| # | Layer | Symptom | Fix |
|---|---|---|---|
| 1 | `build.rs::tilelang_target` + `gen_tilelang_aot.py::parse_target` | `ValueError: Target kind "cuda:89" is not defined. ... e.g. 'cuda -arch=sm_80'` | `802c5fc8` — switch to `cuda -arch=sm_<sm>`. |
| 2 | `batch_prefill_paged_hd128.py::_make_kernel` | `AssertionError: A and B must have the same dtype` at `T.gemm(p, v_tile, acc_o)` (P f32, V bf16) | `4d9c65f0` — narrow P to bf16 via `T.copy(p, p_bf16)`; add `policy=T.GemmWarpPolicy.FullRow` to both gemms; hoist alpha rescale to 2D `T.Parallel(BLOCK_M, HEAD_DIM)`. |
| 3 | `gen_tilelang_aot.py` cubin probe | TileLang 0.1.9 emits a TVM-FFI `.so` rather than a raw cubin; `compiled.cubin_path` is `None` on cold-cache compile | `2a4ff6ce` — pull `adapter.device_kernel_source` (in-memory CUDA source) and nvcc to a raw cubin against TileLang's bundled `tl_templates/cuda` + `cutlass/include`. |
| 4 | C wrapper `cuLaunchKernel` | `Triton Error [CUDA]: an illegal memory access was encountered` — kernel uses `extern __shared__ buf_dyn_shmem[]` with ~48 KB; sm_89's default cap is 48 KB | `2a4ff6ce` — parse `dyn_shmem_bytes` from `host_kernel_source`, lift via `cuFuncSetAttribute(..., MAX_DYNAMIC_SHARED_SIZE_BYTES, ...)`, pass same value as `cuLaunchKernel`'s `sharedMemBytes`. |

### What this means

- The Phase 0 commit trio (`022e8dd / 76e044b / 9896d25`) stays in the
  tree (now operational, not parked).
- The pending-remote stub
  [`wins/2026-04-26-bench-guidellm-cuda-tilelang-prefill-hd128-pending-remote.md`](../experience/wins/2026-04-26-bench-guidellm-cuda-tilelang-prefill-hd128-pending-remote.md)
  stays in place — H100 is still required for the §5 ship/revert
  decision per plan §0.
- The `tilelang>=0.1` extra in `pyproject.toml` should still be
  pinned to `==0.1.9` so future ABI drift doesn't silently re-break
  Phase 0.

---

## What landed this session

| Commit | Scope | Notes |
|---|---|---|
| `802c5fc8` | fix(cuda) | TileLang 0.1.9 target string format — issue #1 |
| `68c183e8` | docs(bench,errors) | wins/scheduler regression + errors/TileLang blocker |
| `1f99c087` | docs(reviews) | this verification report (initial draft) |
| `4d9c65f0` | fix(cuda) | TileLang FlashAttention-2 layout alignment — issue #2 |
| `c5836a9a` | docs(errors) | errors entry update with bisect outcome + remaining work |
| `310305b7` | docs(bench) | wins entry: mixed-batch refactors close c=4/c=8 backlog |
| `2a4ff6ce` | feat(cuda) | TileLang AOT works end-to-end on TileLang 0.1.9 — issues #3 + #4 |
| `b4942807` | docs(bench) | wins entry: TileLang AOT prefill HD128 L4 floor |

(`models` symlink created in workspace root and `infer/models` per
`memory/project_remote_cuda_box.md` — symlinks only, untracked.)

---

## Open follow-ups

- **H100 spike for the §5 decision** — required to drive
  ship/revert per `tilelang-integration.md` §5; L4 is floor-only.
- **Pin `tilelang==0.1.9`** in `pyproject.toml` so the ABI we just
  wrote against can't drift.
- **Phase 1 (decode HD128/HD256)** — only relevant if H100 confirms
  Phase 0 wins; propagate the `T.int32` scalar pattern to the
  decode kernels.
- **Canonical sweep on L4** — the runs here use exploration-mode
  concurrent profiles to match prior baselines exactly. Lifting to
  the full `sweep` profile (synchronous + throughput legs) closes
  the open exploration-mode caveat on the wins/ entries.
- **Qwen3.5-4B parallel run** — symmetry check against the
  `27.59 tok/s` line from `project_l4_perf_baseline.md`.
- **Shared-prefix workload** — the bench shows `prefix_hit_rate=0.0%`
  because guidellm's synthetic prompts don't share prefixes. Re-run
  with an agent-trace dataset to verify the kv_tier promotion paths
  the recent refactors changed.
