# Agent-OPD training round — per-stage ms/% breakdown (measured, one instrumented round)

## Context

First precise end-to-end profile of an in-process `train agent-opd` round.
8×H20 GPU 1, 27B FP8 student (zero-copy shared base), 4 tasks × 2 samples,
max-turns 5, max-tokens 512, writeback-cap 4, LoRA r16 qv, temp 0.7, 1 round +
round-0 eval. Instrumentation: `ARLE_AOPD_PROFILE=1` (commit `6d5ba119b`,
`crates/train/src/aopd_profile.rs`), zero-cost when off. RUN_EXIT=0.

## The breakdown (round wall = 173523.7 ms)

| stage | kind | calls | total ms | ms/call | %round |
|---|---|---|---|---|---|
| **rollout_decode** | GPU | 8 | 70112 | 8764 | **40.4%** |
| **writeback** (fwd+bwd+opt) | GPU | 4 | 57463 | **14366** | **33.1%** |
| **eval** (held-out) | GPU | 1 | 39868 | 39868 | **23.0%** |
| score_pytest | WALL | 8 | 5713 | 714 | 3.3% |
| sandbox_overview | WALL | 4 | 202 | 50 | 0.1% |
| sync_lora | GPU | 1 | 60 | 60 | 0.0% |
| boot/reset/diff/tool_exec/save | WALL/DISK | — | ~88 | — | ~0% |
| (untimed) | — | — | 17 | — | 0% |

**Three GPU stages = 96.5% of round wall.** Everything else is noise.

## Structure (measured)

- **Fully serial on the engine lock** — rollout / writeback / eval never overlap.
- Trigger counts/round: rollouts = tasks×samples = 8 → pytest 8; **writeback = 4**
  (= trained_pairs, capped) → **backward 4, optimizer 4**; sync_lora 1; eval 1
  (4 eval tasks internally); saves 1.
- **No teacher-model forward exists in agent-OPD** — the profiler confirms
  `score_pytest` IS the reward (execution gate), not a logprob teacher pass. It
  is on-policy rejection-sampling self-distillation: reward = pytest exit-0,
  target = the student's own passing trajectory via one masked-CE step.
- DAG: `kv_ensure → [per-task: boot → overview → (per-sample: reset → rollout →
  diff → pytest)] → writeback(all accepted) → sync_lora → eval → saves`. Each
  segment is `&mut`-exclusive, cannot overlap.
- GPU-idle = 3.5% provable/round (pytest 3.3%); ~15-18% coarse if eval's internal
  pytest is counted. Idle is corpus-dependent: synthetic pytest 0.7 s; real
  SWE-Pro repos run seconds-to-minutes, where pytest-overlap starts to pay.

## Optimization levers (ranked by measured %, cheapest-first)

1. **eval 23% —降频.** Held-out eval runs a full eval set every round; `eval_every>1`
   removes it from most rounds. Zero risk, cheapest, scales with round count.
2. **rollout_decode 40% — concurrent rollout uses plain-batched, not dspark.**
   Multi-sample rollout is concurrent (C≈samples); plain-batched beats dspark
   1.5-3.5× at C≥4 ([concurrency re-measure](2026-07-10-dspark-concurrency-derisk-kill.md)).
   `agent_opd_curve.sh` already gates dspark off at SAMPLES>1; verify the
   in-process engine actually batches the sample group.
3. **writeback 33% (14.4 s/call) — the training-side wall.** 27B masked-CE
   fwd+bwd+opt with grad-checkpointing. Largest single-call cost; biggest but
   highest-effort lever (touches loss/checkpoint path, not inference).
4. **pytest-overlap — DEFER.** Not worth it at synthetic 0.7 s pytest; revisit on
   real SWE-Pro repos where score_pytest dominates the host wall.

## Rule

- Profile the whole round before optimizing a phase: the intuition-ranked
  "rollout is the bottleneck" was only 40% — eval (23%, trivially cut) and
  writeback (33%, training-side) together outweigh it. Optimize by measured %,
  not by which stage feels slow.
