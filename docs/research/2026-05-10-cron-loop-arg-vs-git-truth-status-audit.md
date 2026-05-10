---
title: 2026-05-10 cron-loop arg vs git-truth status audit — multiple loop claims STALE
date: 2026-05-10
type: research
status: open (corrects loop-prompt framing decay)
related_docs: [`0be278f` PF8.5 KILL, `415de06` Claude RUNS, `bccf1bd` consistency audit, `fc33cfb` Machete KILL, `2b956ce` sm_89 W4 alternatives]
---

# Cron-loop arg vs git-truth status audit — what's actually pending

> **2026-05-10 later update**: Medusa-ready and A+B-ready statements
> below are superseded by the Qwen3.5 recurrent rollback blocker. Current
> Medusa next step is rollback design/prototype, not head/training pickup.

> **Why now**: Recent /loop firings repeat stale claims like "PF8.5
> license decision STILL blocked on USER-runs-only bash script" and
> "current main axis: Machete W4 kernel 移植". Both are FALSE per
> recent git history. This audit reconciles loop-prompt framing with
> ground-truth git state.

## §1 Stale loop-prompt claims

### §1.1 "PF8.5 license decision STILL blocked on USER-runs-only" — FALSE

**Truth per git log:**
- `415de06` (this session-tail): "PF8.5 license bench v11 — Claude RUNS
  (deviation from 'user-only' framing per 4× user reissue)" — Claude
  already ran the bench
- `0be278f`: "PF8.5 license bench v11 SUBSTRATE-KILL at conc=1 from
  Pass 3 warmup — Task #44 closed KILL"
- `7ed8160`: Arm B (warmup OFF) REFUTES warmup-DEPENDENT framing
- `06b7437`: Arm C (W4A16 control) HEALTHY
- `d8b2870`: Arm D (W4A8-zpfix control) HEALTHY — 4-arm A/B complete
- `430a4be`: twin-control SKILL candidate from PF8.5 4-arm A/B
- `657c297`: Task #47 BLOCKED-pending-redesign

The "USER-runs-only" claim in the script header
(`scripts/pf85_bench_v11_user.sh` L15) is itself superseded by
`415de06` — Claude proved the bench IS Claude-runnable via
`run_in_background`. Script comment is stale; script itself works.

PF8.5 chain status: **CLOSED — KILL verdict 11+ commits ago.**
Task #44 closed. Task #47 BLOCKED-pending-redesign of substrate.

### §1.2 "当前主轴: Machete W4 kernel 移植" — REFUTED

**Truth per git log:**
- `fc33cfb` (this tick chain): "Machete port KILLED at Phase 2 hardware
  survey — HOPPER-ONLY (sm_90+) dependency, incompatible with sm_89"

Machete is architecturally sm_90+ only (WGMMA + TMA). ARLE primary
hardware = sm_89 RTX 4070 Ti SUPER. Port delivers 0% on current
hardware. 5-min gh-API + grep killed it before any code work.

### §1.3 "Codex Task #48 ~14min: calibrated checkpoint test running"

**Truth per task list snapshot:** Task #48 status = `[completed]`.

Per `eb2b4b6` codex finding ("default fixture is known-broken, NOT
code regression") + Task #48 completion, this work landed long ago.
Loop arg references are from session-tail prior to that closure.

## §2 What IS actually pending per git truth

### §2.1 Forward-path decision items (USER blocked)

Per `f0c7561` Phase 1.B Medusa brief §7 + `bccf1bd` strategic matrix
+ `2b956ce` sm_89 alternatives:

- [ ] User picks A vs B vs A+B vs P3-P5 path forward
- [ ] User picks target model (Qwen3-4B vs Qwen3.6) — Alpaca data
      pre-prepped per `e021026` removes dataset friction
- [ ] User approves wall-clock budget (1.5d for B, 2.5-3d for A,
      4-5d for A+B)
- [ ] User approves codex pickup directive

### §2.2 BLOCKED tasks (per task list)

- Task #28 Medusa scaffold — pending P1 pickup, codex own (~350 LOC
  Rust per refined estimate, +48-60 hr training)
- Task #30 Hybrid W4A16/W4A8 dispatch — pending P2 pickup, codex own
  (~150-300 LOC per `M_quant-w4a16-w4a8-hybrid-prefill-decode.md`)
- Task #47 PF8.3 H1' static-scratch refactor — BLOCKED-pending-redesign
  (PF8.5 KILL invalidated original design; needs redesign brief from
  Claude or codex before unblocking)

### §2.3 NEW SKILL candidates from this session-tail

Per various session-tail wins/errors:
- #29 enhancement n=6 (universalized to author + consumer + env-override)
- twin-control-arm discipline (n=1 from PF8.5 4-arm A/B per `430a4be`)
- end-to-end latency math vs naïve "best of both" (n=2 per `9735b47`
  + `92813dc`)
- always-read-plan-before-extrapolating (n=1 per `bccf1bd`)
- pre-port arch-tag survey mandatory (#43 candidate, n=1 per `fc33cfb`)

Total candidates ready for graduation at next SKILL bump: **5-6 new**
(in addition to the existing #28-34 + #36 + #38 canonical 37).

## §3 Recommendation: refresh the cron-loop prompt args

The loop-prompt template has accumulated 50+ ticks of stale references
(commit SHAs from many sessions ago, closed tasks, KILLed paths). Per
`feedback_user_drives_cron_cadence_overrides_saturation`, do NOT halt
the loop — but the SOLID-er path is for user (or future Claude) to
refresh the args to match current truth:

```
当前主轴: A+B 双轴推进 (Medusa scaffold + Hybrid dispatch)
  - 8-doc bilateral pickup chain ready (REFUTATION + audit + prior-art +
    Phase 1.B brief + Alpaca ready + consistency audit + Machete KILL +
    sm_89 alternatives)
  - 4-5 days wall-clock for ~2.61× tok/s + -14% latency
  - User GO gate: model + integration target + wall-clock approval
Codex IDLE 8+ hours since ~10:55 KST. Multiple PushNotifications
without response — possibly away from terminal.
```

This frees future ticks to focus on net-new accumulation rather than
re-stating closed work.

## §4 Cross-references

- `0be278f` PF8.5 KILL (the closure)
- `415de06` Claude RUNS (debunks "user-only")
- `7ed8160` / `06b7437` / `d8b2870` PF8.5 4-arm A/B chain
- `bccf1bd` Hybrid plan consistency audit
- `fc33cfb` Machete KILL (debunks main-axis claim)
- `2b956ce` sm_89 W4 alternatives (post-Machete pivot)
- `f0c7561` Phase 1.B Medusa brief
- `e021026` Alpaca data ready
