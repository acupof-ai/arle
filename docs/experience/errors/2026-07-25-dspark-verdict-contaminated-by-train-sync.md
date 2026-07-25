# DSpark's −50% at c=16 was a default-off training sidecar's stream sync (#183)

> Status: gate landed; clean re-measure DONE on `6aa4ca6d1` (#183+#184). The
> per-step-sync collapse is GONE, but DSpark c=1 came back at **+5.0%**, not the
> predicted ~+63% — so a second effect remains and the #128 verdict is still
> open (see §Re-measure).

## Context

First run in which DSv4 DSpark actually took the spec path, counter-proven:
`/v1/stats.spec_decode` went `available:false, chains:0` → `chains 64, drafted
320, accepted 176, accept_rate 0.55` after one greedy request; the no-spec arm
stayed `available:false` throughout. The old `--dspark-max-prompt-tokens 64`
routing gate no longer exists, so the `docs/baselines.md` "not triggered" row is
finally superseded.

Matched A/B, DSv4-Flash-FP8, 4×H20 TP=4/EP=4, 128-in/128-out, 60 s/point, seed
20260416, 0 errors at all 8 points:

| c | no-spec out tok/s | DSpark out tok/s | Δ | committed tok/step |
|---|---:|---:|---:|---:|
| 1 | 42.7 | 43.3 | +1.4% | 2.48 |
| 4 | 81.4 | 58.5 | −28.1% | 2.53 |
| 8 | 139.8 | 73.7 | −47.3% | 2.54 |
| 16 | 174.4 | 87.3 | −49.9% | 2.45 |

Acceptance is healthy and flat across concurrency (accept_rate 0.29–0.31, 2.45–
2.54 committed tokens per verify step, block 5, `conf_threshold 0` so every
chain drafts full width). The draft head works. The step does not.

## Root Cause

`aec71ef16` (2026-07-21 16:13, "wire DSv4-Flash into train sidecar") added the
`capture_dspark_experience{,_hidden}` call sites to both DSpark verify lanes
(`executor/dsv4.rs:1843`, `:2140`) — **unconditionally**. Each does two
vocab-wide bf16 D2H copies, two host bf16→f32 conversions of ~1.4M elements, and
two full `ctx.sync()`, per verify step, before any shape guard runs. The only
consumer is the `--dspark-train` sidecar, which is default-off.

**This is a regression against a measured win, not a first-measurement
disappointment.** `591772a43` (2026-07-20 18:55) measured DSpark **c=1 +63.8%**
(63.7 vs 38.9 tok/s,
[win](../wins/2026-07-20-dspark-sliding-window-c1-win-c8-regress.md)).
`aec71ef16` lands 21 h later — after that measurement and after the 07-21
batched-verify work — and `git log -S"capture_dspark_experience"` confirms it is
the commit that introduced the call sites. Today's c=1 is **+1.4%**.

Magnitude checks out, so the attribution is not just ordering: 07-20 c=1 63.7
tok/s at ~2.5 tok/step ⇒ 39 ms/step; today 43.3 tok/s at 2.48 tok/step ⇒ 57
ms/step. The +18 ms/step gap is the right size for two syncs plus 1.4M host
element conversions. At c≥4 the same per-step sync serializes the whole TP=4
NCCL batch, giving the monotone −28 / −47 / −50%.

**Why it sat unmeasured for 4 days:** the two attempts in between were both
blocked, not negative — 07-21's DSpark arm OOM'd on a stale-memory GPU, and the
07-19 baselines row had DSpark never triggering. A blocked arm reads the same as
a quiet one on a dashboard.

## Fix

`capturing()` = `BUFFER.get().is_some()`. The sidecar is the only thing that
initializes the global buffer (`spawn_dspark_train_sidecar` →
`engine.dspark_experience_buffer()` → `buffer()`), so its presence *is* the
`--dspark-train` signal — no second flag to drift out of sync. Both capture
entry points bail on `!capturing()` before touching device memory.

The clean re-measure must use `bench-prompts-64.jsonl` + `max_tokens 256` — the
07-20 dataset — so the prediction is falsifiable: c=1 should return to ~63 tok/s.
Anything materially short of that means a second cause is still in the path.

## Re-measure (2026-07-25, `6aa4ca6d1`, #183 gate + #184 scratch)

Clean 6-arm run, DSv4-Flash-FP8 4×H20 TP=4 GPUs 4-7, dataset
`dspark_natural_128in_128out.jsonl`, **max_tokens 128**, 60 s/point, 0 errors.
Raw: pod `bench-output/2026-07-25-R2-dsv4-*`, verified vs `result.json`.

| c | no-spec | DSpark | Δ | (contaminated 07-25) |
|---|---:|---:|---:|---:|
| 1 | 42.39 | **44.52** | **+5.0%** | +1.4% |
| 4 | 79.46 | 61.00 | −23.2% | −28.1% |
| 8 | 136.44 | 76.57 | −43.9% | −47.3% |
| 16 | 174.46 | 90.66 | −48.0% | −49.9% |

**The sync collapse is real and fixed** — but the curve is only mildly better,
NOT restored to the 07-20 +63.8%. Two reasons the c≥4 arm is still net-negative,
neither of which the gate addresses:

1. **Prediction MISSED, honestly.** §Fix predicted c=1 ~63 tok/s on the *07-20
   dataset* (`bench-prompts-64` / max_tokens 256). This run used the 128/128
   natural set, a **different fingerprint** — so +5.0% vs +63.8% is not a clean
   falsification, but it is NOT the predicted return either. The 07-20 number is
   not reproduced; whether that gap is the dataset or a second cause is
   **undecided** — a same-dataset (256-out) rerun is still owed before #128.
2. **Slot count, not just sync.** #184 cut DSpark per_slot 645→607MB (22 slots)
   vs no-spec's 338MB (59 slots). At c≥4 the DSpark arm runs fewer concurrent
   slots than no-spec — a real capacity gap the sync gate does not touch. The
   arms still differ in more than speculation.

**Verdict: DSpark is a c=1 / low-concurrency win** (+5% DSv4, +57.5% on 27B
single-GPU where c=1 is decode-bound), net-negative at c≥4 on this shape. #128
accept-or-kill still needs the same-dataset 256-out rerun to close reason (1).

## Rule (open items above)

- **A feature-wiring commit is a perf change to every path it touches.** No bench
  entry was cut for `aec71ef16` because it read as plumbing for a default-off
  flag; it silently added a stream sync to the hot loop. The gate is "does this
  execute on the default path", not "is the feature on by default".
- **A hot-path producer for a default-off consumer must be gated on the consumer,
  not on its own arguments.** The #176 guard checked shapes and dropped the push
  — but the D2H and the syncs ran *before* the guard, so the cost was paid on
  every step to build data that was then thrown away. Cheap-check-first is not
  optional when the check is "is anyone listening".
- **Derive the gate from existing state instead of adding a flag.** A `OnceLock`
  that only one caller initializes is already the enable bit.
- **A spec-decode verdict needs the arms to differ only in speculation.** Here
  the DSpark arm also differed in "pays a stream sync per step" and in slot count
  (22 vs 32, #184). Acceptance was measured correctly; wall-clock was not
  measuring what its label claimed.
