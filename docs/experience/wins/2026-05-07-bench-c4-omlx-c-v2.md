# Bench — c=4 oMLX-C v2 (mask eval removed) — 2026-05-07

## Goal

Find and remove the sync barrier inside oMLX-C v1's pipelined path so
the pipelining win actually materializes. v1 had `sample_us` drop from
4438→0μs but `prep_us` regressed 159→4233μs — the wait migrated, not
disappeared.

## Hypothesis (going in)

Some C++ FFI inside `step_batch_packed` is doing an internal
`mlx::eval` that blocks behind the prev call's async_eval-pending GPU
work. Find via finer-grained timing.

## What we found

The barrier wasn't in C++ — it was inside `build_varlen_decode_mask`
([`infer/src/backend/metal/mlx.rs:884`](../../../infer/src/backend/metal/mlx.rs)),
a Rust helper that built the additive SDPA mask. The function ended
with a defensive `eval(&[&mask])` (added in commit `a78364000` 2026-04-16,
the original varlen scaffolding). That eval was harmless when the GPU
queue was idle (~50μs in legacy) but **blocks for 4ms when the prev
call's async_eval is still in flight**, because MLX serializes eval
calls on the same stream.

Fix: drop the `eval(&[&mask])`. The mask is consumed downstream by
`step_batch_packed`'s lazy graph; there's no host-side reader. Tests
that need host values (the unit test at `mlx.rs:1300`) call eval
themselves.

## Params

- Binary: `target/release/metal_serve` rebuilt with mask eval removed
- Model: `models/Qwen3.5-0.8B-MLX-4bit`
- Workload: `/tmp/c4_smoke.sh` — 4 concurrent /v1/chat/completions
  POSTs, max_tokens=64, temperature=0.0
- Same as
  [`2026-05-07-bench-c4-phase-timing-baseline.md`](2026-05-07-bench-c4-phase-timing-baseline.md)
  for direct comparability

## Results — 4-way comparison at batch=4

| Phase | Baseline | oMLX-C v1 | v2-legacy (mask fix only) | v2-pipelined (mask fix + INFER_OMLX_C=1) |
|---|---:|---:|---:|---:|
| prep avg | 159μs | 4233μs ⚠️ | **7μs** | **14μs** |
| build_graph | 421 | 463 | 444 | 560 |
| async_kick avg | 1563 | 1614 | 1640 | **6683** ⚠️ |
| async_kick p50 | — | — | 1598 | **4836** |
| sample avg | 4438μs | 0 ✅ | 4138 | **8μs** ✅ |
| pool_dw | 0 | 0 | 0 | 1 |
| total avg | 6583 | 6312 | 6231 | 7269 |
| **total p50** | **6409** | 6230 | 6239 | **5437** |

→ **oMLX-C v2 pipelined p50: 5437μs vs baseline 6409μs = 15.2% ITL
reduction at c=4.**

## Problems / observations

1. **The bottleneck moves with each fix.**
   - Baseline: `sample` was 4.4ms (sync wait at end)
   - v1 pipelined: `sample` 0, `prep` became 4.2ms (mask eval inside Rust prep)
   - v2 pipelined: `prep` 14μs, but `async_kick` jumped to 4.8ms p50
2. **Why does `async_kick` regress?** The `async_eval(eval_refs)`
   call in v2 takes ~3-5ms instead of ~1.5ms. Most likely cause:
   the prev call's async_eval-pending graph is still being encoded
   on MLX's stream, and submitting the new graph forces a wait for
   the prev encoding to commit. So the wait migrated INSIDE async_eval.
3. **Heavy tail.** v2 pipelined avg=7269 vs p50=5437 — distribution is
   heavy-tailed. A few calls take 10ms+. Probably MLX command-buffer
   batching cycles aligning unfavorably for some calls.
4. **Legacy path also benefits modestly.** v2-legacy total p50=6239 vs
   baseline p50=6409 (~3% improvement). The mask eval fix is a
   universal small win; pipelining is what unlocks the 15%.

## Why we don't flip default to ON yet

CLAUDE.md `feedback_matched_ab_for_small_bench_effects.md` requires
matched A/B in ≥2 sessions for effects ≤10%. **15% on p50 IS over the
threshold**, but the avg-vs-p50 spread (7269 vs 5437) suggests an
outlier-driven worst case that could be a regression in some
workloads. Conservative default: keep `INFER_OMLX_C` as opt-in via
env var. Flipping default ON requires:

1. Reproduce the 15% across two sessions on cold start (not just one)
2. Investigate the heavy tail (some pipelined calls take 10ms+ —
   likely MLX command-buffer alignment)
3. Confirm no regression at c=1 (single-stream, no batching)

## Learnings

1. **Defensive `eval` calls are perf landmines for pipelined paths.**
   Any `eval(&[&array])` on a small intermediate is fast when GPU is
   idle but BLOCKS BEHIND THE WHOLE PENDING STREAM when GPU is busy
   from a prior async_eval. Audit all `eval(&[…])` sites for
   "is this caller in a pipelined hot path?" Rule added: search for
   `eval\(&\[` in `infer/src/backend/metal/` quarterly.
2. **The C++ FFI was innocent.** I expected `step_batch_packed`'s
   internals to be the culprit; turned out to be a same-file Rust
   helper. The finer-grained timing (`mask_us=4000`,
   `rope_us=2`...) localized this in one bench cycle.
3. **Each fix moves the bottleneck rather than removing the
   underlying GPU time.** The forward+sample kernels themselves take
   ~6ms at c=4. Pipelining can hide some of that behind host work,
   but not all. The 15% reduction is roughly the "host work between
   calls" fraction we successfully overlapped.
4. **Approach this as a multi-session iteration.** Each tick should
   move the bottleneck once; document where it landed; let the next
   tick attack the new top consumer. v2's `async_kick` heavy tail is
   the v3 target.

## What worked / Rule

- Inserted `metal_phase_timing_pipelined_prep_breakdown` log line
  with 5 sub-phase deltas (clear/ensure/take/mask/rope). One bench
  cycle pinpointed mask as 4000μs vs everything else <30μs.
- Same `INFER_PHASE_TIMING=1` env gate; same path probe. Zero
  production cost when off.
- Comparing v2-legacy alongside v2-pipelined isolated "fix benefit"
  (legacy +3%) from "pipelining benefit" (pipelined +15%).

## Rule

When optimizing a pipelined path, ALWAYS bench BOTH the pipelined
path and the legacy path on the same fix. The fix may help legacy too
(or only legacy — false signal). Report deltas for both paths
side-by-side.

## Next

- **oMLX-C v3 — investigate `async_kick` heavy tail** (avg 6683 vs
  p50 4836). Most likely: MLX command-buffer commit cycle. Tools:
  Xcode Metal capture or MLX env-flag trace.
- **Eliminate the `prev_sampled` shape-check `eval` if any**: verify
  no other defensive eval sites in the pipelined path.
- **Flip default ON after** v3 closes the heavy tail OR after a
  matched A/B reproduces 15%+ across 2 sessions.
- **Investigate c=1 single-stream impact** to confirm no regression
  off-batched workloads.

## References

- Phase 1 baseline:
  [`2026-05-07-bench-c4-phase-timing-baseline.md`](2026-05-07-bench-c4-phase-timing-baseline.md)
- v1 prior iteration:
  [`2026-05-07-bench-c4-omlx-c-v1.md`](2026-05-07-bench-c4-omlx-c-v1.md)
- Design plan:
  [`docs/plans/M_e1-omlx-c-multi-step-pipelining.md`](../../plans/M_e1-omlx-c-multi-step-pipelining.md)
- Original mask eval commit (defensive):
  `a78364000` (2026-04-16, varlen scaffolding Phase 1)
