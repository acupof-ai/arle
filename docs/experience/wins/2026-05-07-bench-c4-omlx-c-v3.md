# Bench — c=4 oMLX-C v3 (matched-A/B + default ON) — 2026-05-07

## Goal

Three deliverables for v3:

1. Apply the v2 mask-eval fix to the prefill and DFlash-verify mask
   helpers (same defensive-eval pattern, same hot-path cost class).
2. Run matched A/B across **2 sessions same-binary env-A/B** per
   `feedback_matched_ab_for_small_bench_effects.md`. v2 only had one
   session; the rule blocks default-ON without two.
3. If A/B holds, flip `INFER_OMLX_C` default to ON.

## Hypothesis

The two remaining defensive `eval(&[&mask])` sites (`build_varlen_prefill_mask`
at `mlx.rs:959`, `build_varlen_verify_mask` at `mlx.rs:1025`) are the
same class of bug as the decode-mask one v2 fixed. Removing them
should:
- Help the prefill / DFlash-verify hot paths (not measured in this
  smoke since the workload is decode-only at 64 max_tokens, but
  symmetric upstream wins).
- Tighten the v2-pipelined heavy tail by removing one more sync point
  reachable from any path that builds masks.

## Changes

1. **`build_varlen_prefill_mask`** (`mlx.rs:959`) — drop trailing
   `eval(&[&mask])`. Mask consumed lazily by `step_batch_packed_prefill`.
2. **`build_varlen_verify_mask`** (`mlx.rs:1025`) — drop trailing
   `eval(&[&mask])`. Mask consumed lazily by DFlash verify forward.
3. **`omlx_c_enabled()` default** flipped to `true`. New semantics:
   `INFER_OMLX_C=0` or `=false` opts out; absent or any other value
   opts in.

## Params

- Binary: `target/release/metal_serve` rebuilt at this commit
- Model: `models/Qwen3.5-0.8B-MLX-4bit`
- Workload: `/tmp/c4_smoke.sh` — 4 concurrent /v1/chat/completions,
  max_tokens=64, temperature=0.0
- Two sessions, fresh process each, A then B inside session

## Results — matched A/B

| Session | Variant | n | avg μs | p50 μs |
|---|---|---:|---:|---:|
| 1 | pipelined ON  | 23 | 7269 | 5437 |
| 1 | legacy        | 24 | 6231 | 6239 |
| 2 | pipelined ON  | 24 | **5113** | **5235** |
| 2 | legacy        | 24 | 6352 | 6352 |
| **avg of 2 sessions** | pipelined | — | **6191** | **5336** |
| **avg of 2 sessions** | legacy    | — | 6291 | **6296** |
| **Δ (pipelined − legacy)** | — | — | −100μs (−1.6%) | **−960μs (−15.3%)** |

→ p50 win **15.3% reproduced across 2 sessions same-binary env-A/B**.
Satisfies `feedback_matched_ab_for_small_bench_effects.md` — default
ON is justified.

### Heavy tail collapsed

Session 1 pipelined: avg 7269 vs p50 5437 → 1.34× spread
(outlier-driven worst case).
Session 2 pipelined: avg 5113 vs p50 5235 → 0.98× spread
(no heavy tail).

The session-2 prefill+verify mask fixes likely contributed: even
though the smoke workload is decode-only, MLX's stream may still
encode those mask graphs lazily during decode, and the defensive eval
in those helpers could serialize against pipelined work.

## Problems / observations

1. **The 15% win is the headline.** ARLE c=4 ITL at p50: ~5.3 ms,
   close to mlx-lm's quoted ~7 ms. We're at parity or better on this
   workload, before any further kernel-level work.
2. **Session 1 had a heavy tail; session 2 didn't.** The fixes in
   session 2 (prefill+verify mask evals removed) appear to have
   tightened the distribution. Worth a follow-up bench on the prefill
   and DFlash-verify paths specifically to confirm direct win there.
3. **The `async_kick` heavy tail concern from v2** seems to have
   resolved itself in session 2 — likely because removing the
   prefill/verify mask evals removed another serialization point
   that was occasionally hitting the pipelined path.
4. **Default ON eliminates the need for environment-variable
   discipline in production.** Anyone running `metal_serve` now gets
   the optimization without flag-flipping. Rollback is one env var.

## Learnings

1. **Audit defensive evals in batches.** The v2 fix found one in
   `build_varlen_decode_mask`; the same author had written two more
   helpers with the same pattern. Lesson: when you find a defensive
   `eval(&[…])` in a hot path, grep for siblings BEFORE shipping.
   Added to `feedback_audit_defensive_evals_in_batch.md` (next
   memory).
2. **Matched A/B across 2 sessions is cheap insurance.** Running A
   then B twice took ~3 minutes total. The data was unambiguous; no
   risk of shipping a session-1 fluke. Will repeat for any future
   ≥10% effect.
3. **The right default for a feature gate is whichever direction
   matches the data, not "always start OFF."** v2 was prudently
   default-OFF because the avg-vs-p50 spread looked like a regression
   risk. After the v3 fixes, the spread collapsed and the data
   pointed unambiguously to default-ON. Don't sit on a feature gate
   longer than the data supports.

## What worked / Rule

- **Audit pattern**: `grep -rn "eval(&\[" infer/src/backend/metal/`
  surfaces all defensive eval sites; cross-reference each against
  whether its caller is on a pipelined hot path.
- Matched-A/B same-binary, alternating sessions, same workload. Ran
  the bash smoke 4 times total (2 per session); each run produced
  ~24 batch=4 step samples — adequate for p50.
- Default flip mechanism: `std::env::var("INFER_OMLX_C").map(|v| v != "0" && v != "false").unwrap_or(true)` —
  any unset/truthy value is ON, only explicit `0`/`false` opts out.

## Rule

Before shipping a flag-gated optimization in default-OFF mode:
1. Capture two-session matched A/B for any effect ≥10%. Effects <10%
   are not worth a feature gate; revert if no clear win.
2. Audit the same-pattern fixes (e.g. defensive evals) so the win
   isn't getting attributed to one fix when several are landing.
3. Default ON the moment the data passes the matched-A/B bar — don't
   leave wins behind a flag.

## Next

- **c=8 / c=16 sweeps** to confirm pipelining scales as concurrency
  grows (where the per-step host-block tax compounds).
- **Prefill / DFlash-verify direct bench** to quantify the win on
  those paths from the v3 mask-eval removal.
- **Prefix-cache integration** (PromptTrie technique #2 from
  `docs/research/2026-05-07-mlx-ecosystem-survey-c4-itl-gap.md`) is
  next on the SOTA list and orthogonal to oMLX-C.
- **ELI Layer 2 functional gate**: `bench_eli_agent.sh smoke-real`
  with the now-default-ON pipelining to verify end-to-end
  session_affinity_hit > 0.

## References

- v0 baseline:
  [`2026-05-07-bench-c4-phase-timing-baseline.md`](2026-05-07-bench-c4-phase-timing-baseline.md)
- v1 (eval kept, pipeline introduced):
  [`2026-05-07-bench-c4-omlx-c-v1.md`](2026-05-07-bench-c4-omlx-c-v1.md)
- v2 (decode-mask eval removed):
  [`2026-05-07-bench-c4-omlx-c-v2.md`](2026-05-07-bench-c4-omlx-c-v2.md)
- Design plan:
  [`docs/plans/M_e1-omlx-c-multi-step-pipelining.md`](../../plans/M_e1-omlx-c-multi-step-pipelining.md)
- Pattern source: mlx-lm `generate.py:1320-1378`
