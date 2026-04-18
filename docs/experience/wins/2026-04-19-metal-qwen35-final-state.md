# Metal Qwen3.5-4B-4bit decode optimization — final state at /loop ceiling

**Date**: 2026-04-19
**Machine**: Apple M4 Max (40 GPU cores, ~400 GB/s UMA)
**Model**: `mlx-community/Qwen3.5-4B-MLX-4bit` (24 GDR + 8 full-attn layers)
**Latest commit on hot path**: `e97d52a` — Iter 7 C++ session API (per-step FFI cost-amortized single-request decode)

## Context

Caps the multi-week Metal Qwen3.5 concurrent-decode optimization arc that
started at the 145 tok/s c=8 ceiling (`2026-04-18-metal-qwen35-concurrent-decode-ceiling.md`).
After 11 numbered iterations across `project_metal_qwen35_optimization.md`
this entry records the *terminal /loop-reachable state* so future sessions
don't redo the same investigations.

## Final ceilings (re-baselined this session)

| path | tok/s | notes |
|---|---|---|
| HTTP c=1 (`metal_serve` → `bench_throughput.py`) | **73.1** | per-stream ITL p50 = 12.9 ms |
| HTTP c=2 | 134.3 | 1.84× scaling, near-linear |
| HTTP c=4 | **162.5** | concurrent ceiling reached |
| HTTP c=8 | 162.9 | flat — adds nothing past c=4 |
| StepDriver tight loop (per-step FFI, no HTTP) | 76.5 | matches `metal_bench --use-step-driver` |
| `cpp_model.generate` (full C++ tight loop) | **84.1** | matches mlx_lm 84.4 — at parity |

**Key headline numbers:**
- c=1 ceiling for the in-process single-request path is **84 tok/s = mlx_lm parity** ✓
- c=1 HTTP path costs **~13% relative to the in-process ceiling** (73.1 vs 84.1)
- c=8 ceiling is now **163 tok/s, NOT 145 as previously reported** — prior memory's "regression to 136.9" was thermal/noise

## What's been ruled out (so future sessions skip these)

| Hypothesis | Outcome | Evidence |
|---|---|---|
| Cross-step double-buffer in `decode_token` | **Landed as P0c** (`5593448`/`aa11906`) — +2-4% | memory line 28 |
| Per-step `from_slice_i32` token alloc reuse | **Tried as P0e, REVERTED** — within noise | memory line 39 |
| mlx_lm has a Qwen3.5-specific trick we missed | **No.** Read `mlx_lm/models/qwen3_5.py` (531 LOC). No `@mx.compile`, no async_eval, no special cache. Uses `ArraysCache(size=2)` — structurally identical to our `Vec<MlxArray>` | memory line 40 |
| HTTP per-token streaming overhead (mpsc, tokio wake) | **Measured: 16 µs/tok = 0.12% of tpot.** Not the bottleneck | memory Iter 9 |
| C++ session API would close the c=1 gap | **Partial.** Iter 7 saved 0.5 ms/step but HTTP scheduler tick now dominates the 13% gap | memory Iter 7 |
| Intermediate-preservation in `qwen35_compiled_step_session` (mirror `qwen35_compiled_generate` pattern) | **Tried, FALSIFIED.** No improvement; possibly slight regression. The "5ms/step" comment in `qwen35_compiled_generate` is historical — pre-async_eval | memory Iter 10 |
| c=8 regressed from 145 → 136.9 | **NOISE.** Fresh re-bench (Iter 11) shows 162.9, BETTER than original 145 | memory Iter 11 |
| Scheduler tick fast-paths (`refresh_waiting_prefix_hits`, `reap_closed_clients`) | **Sub-1% contribution.** 78 ticks/sec × ~5 µs/call. Not worth a patch | this entry |

## Sized ceilings for remaining unattempted work

| Path | Ceiling | Cost |
|---|---|---|
| Eliminate ALL HTTP scheduler-tick overhead (~530 µs/tok) | c=1 jumps 73 → 76.5 tok/s = **+4.8%** | "half-state" architectural debt — fast paths in `scheduler.step()` parallel to normal path |
| Close StepDriver↔generate gap (76.5 → 84) | **+9.0% c=1** | Already 2-strike-failed (P0e, intermediate preservation). Likely needs MLX upstream changes |
| GDR kernel batch dispatch (the c≥4 wall, 6.1 ms/row) | Up to **+50% c=8** if reduced to 3 ms/row | Out of /loop scope. Requires Xcode Metal capture on M-series GPU + likely a new Metal kernel |

## Rule

When the Metal Qwen3.5 optimization "/loop" is re-fired with the **same stale
prompt** that was last updated before the e97d52a session API landed, treat
this entry as the project's terminal state. Three correct responses:

1. **Ship & close.** Land a final commit updating any project docs, then
   archive the project.
2. **Open a fresh non-/loop session** focused on Xcode Metal capture of the
   GDR kernel — the ONLY remaining lever with non-trivial upside.
3. **Update the /loop prompt** with a different option set (longer contexts,
   prefix cache hit-rate, multi-tenant queueing strategies, etc.) — the
   original 5-option list has been fully explored.

Do not attempt structural changes to `Qwen35StepDriver::decode_token` or
`qwen35_compiled_step_session` to close the c=1 gap — that problem has
2-strike-failed and the gap is now bounded by GDR + MLP kernel time, not
FFI/scheduler overhead.
