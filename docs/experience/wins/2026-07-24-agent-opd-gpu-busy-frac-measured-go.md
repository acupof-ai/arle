# agent-OPD rollout is idle-bound: gpu_busy_frac 0.30–0.34 — GO on concurrent mega-rollout

## Context

The online-RL acceleration survey
([research](../../research/2026-07-23-online-rl-acceleration.md)) gated its
biggest lever — collapsing serial group-rollouts into one concurrent
mega-rollout — on one number: the GPU-active fraction inside the rollout wall.
The `gpu_busy_frac` timer (engine-forward wall via `ENGINE_FORWARD_BUSY_MICROS`,
emitted per kind=group metrics row) landed 2026-07-23; two serve bugs then
blocked every measurement attempt (relay-ack teardown + fatal device-pool
exhaustion — [errors entry](../errors/2026-07-23-agent-opd-mtp-lockstep-single-gpu-teardown.md),
fixed in `e4ac039dc` + `a9d0c5412`).

Valid run: `busytimer-s4`, HEAD `0a42841ad`, 1×H20, ThinkingCap-Qwen3.6-27B-FP8,
`SMOKE=1 SPEC=off SAMPLES=4`. Hard gate passed: completion_tokens > 0 every
group, 0 teardowns, 0 `#164 backstop` engagements, 16/16 sessions exit 0 inside
the 600 s CC wall.

## What Worked

| task | gpu_busy_frac | gpu_busy_secs | rollout_secs | completion_tokens | rewards |
|---|---|---|---|---|---|
| paging-count | 0.2998 | 125.2 | 417.4 | 1851 | 4×1.0 |
| billing-roundhours | 0.3312 | 139.8 | 422.0 | 2156 | 4×1.0 |
| sales-peak | 0.3369 | 135.9 | 403.4 | 1632 | 4×1.0 |
| qty-clamp | 0.3271 | 135.1 | 413.1 | 1841 | 4×1.0 |

- **Verdict: GO.** The GPU forwards ~135 s of each ~410 s group — **~2/3 of the
  rollout wall is idle** on CC-side latency (turn processing, tool exec,
  request prep). Corroboration: 8-way concurrency measured 0.29 — doubling
  concurrency did not lift occupancy, so many more overlapped groups fit before
  the GPU saturates.
- **Case-audited, not trusted:** the 16/16 passes are real — textbook-correct
  one-line fixes (e.g. `total // per_page` → ceil-div), non-empty
  `fail_to_pass` hidden tests A/B-verified fail→pass (`2 passed in 0.02s`).
  Post-LoRA-sync groups (behavior_version 1/2) still 4/4 — merge does not
  corrupt the policy. Variance-weighted task selection skipped all always-pass
  tasks in round 1 as designed.
- **Reward-bearing ratio on SMOKE = 0.0** (all 4 toy tasks p=1.0 → all groups
  zero-variance). The ratio instrument works; a meaningful number needs the
  comfort-band corpus, not the smoke corpus.
- **Prompt-bloat root found:** every CC request carries ~31K tokens because
  `claude -p` walks up from the task workdir (under `/host/arle-build`) and
  ingests the repo's `CLAUDE.md` agent contract. Staging sandboxes outside the
  repo cuts most of the 31K/request — turn 1 measured 178–245 s vs 21–80 s for
  turns 2–4, dominated by exactly this prefill.

## Rule

Every agent-OPD run now self-reports `gpu_busy_frac` per group — gate any
conclusion on `completion_tokens > 0` first. For rollout-concurrency decisions,
measure occupancy at two concurrencies: flat busy-frac under doubled C = idle-
bound (scale C), rising = approaching GPU-bound. Stage CC task workdirs outside
any repo carrying a `CLAUDE.md`, or the agent contract silently becomes ~31K of
per-request prefill.
