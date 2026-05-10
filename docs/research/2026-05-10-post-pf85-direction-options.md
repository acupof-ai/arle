---
title: Post-PF8.5 KILL — direction options for user decision
date: 2026-05-10
type: research
status: open (decision-needed; awaits user direction)
related_tasks: [#28 (Medusa, P1 pickup), #30 (Hybrid W4A16/W4A8 dispatch, P2 pickup), #47 (PF8.3 H1' redesign, blocked)]
---

# Post-PF8.5 — direction options

> **Purpose**: PF8.5 license bench v11 returned KILL (5878 kernel
> failures, see `0be278f`). Subsequent 3-arm A/B (Arms B/C/D) bounded
> the bug to PF8.3 substrate specifically. PF8 chain CLOSED-KILL.
> User has fired /loop ~10 times since then without explicit direction.
> This doc crystallizes the ~3 forward options to make decision easy.

## §1 Current state (post-PF8.5 chain)

- **Task #44 PF8 chain**: completed (KILL — `0be278f` + `7ed8160` +
  `06b7437` + `d8b2870`)
- **Task #47 PF8.3 H1' refactor**: BLOCKED-pending-redesign (per
  `657c297` plan update — original design empirically broken per
  Arm B refute)
- **Codex**: IDLE since ~10:55 KST (5hr+, frozen tmux at "Worked
  26m 51s" awaiting `/review`)
- **Pickup queue P1 PIVOT**: from #47 H1' → #28 Medusa
- **Perf floor for next pickup** (Medusa or otherwise):
  - TTFT ≤ 54.2 ms (W4A8-zpfix Arm D, the best-TTFT non-PF8 W4 path)
  - ITL ≤ 5.8 ms (W4A16-marlin-zpfix Arm C, the best-ITL W4 path)
  - tok/s ≥ 799 (W4A16 Arm C)

## §2 Three forward options

### §2.1 Option A: Pickup Task #28 Medusa scaffold (current P1)

**Scope**: ~500 LOC codex-side scaffold + 1 week training run for
Medusa heads. Per `63769be` Medusa Alpaca cross-link, HF auth blocker
is bypassable via Alpaca dataset (ungated).

**Acceptance gate**: Medusa must improve tok/s by ≥ 2× at acceptance
≥ 70% over W4A16 baseline (799 → ≥ 1600 tok/s) to justify the ~1
week training cost.

**Risk**: training data + scaffolding + acceptance rate all need to
align; Medusa is a real model with real failure modes.

**Time to first measurement**: 2-3 days (scaffold + training cycle).

### §2.2 Option B: Pickup Task #30 Hybrid W4A16/W4A8 dispatch (P2)

**Scope**: Phase 1-3 substrate (no current scaffold). Per task name
"Hybrid W4A16/W4A8 dispatch Phase 1-3 substrate" — would build the
runtime-side dispatch logic for picking W4A16 vs W4A8 path per layer
or per workload phase.

**Acceptance gate**: per Arm C+D data, hybrid would target
W4A8-prefill (54.2ms TTFT) + W4A16-decode (5.8ms ITL) =
**theoretically combines best of both**. Net improvement: -18%
TTFT vs W4A16-only + -50% ITL vs W4A8-only. Math not yet validated.

**Risk**: dispatch overhead at layer boundaries could eat the win;
needs measured A/B not just paper math.

**Time to first measurement**: ~5 days (Phase 1 substrate scaffold +
codex pickup + Claude bench cycle).

### §2.3 Option C: PF8.3 H1' redesign (Task #47 BLOCKED, broader scope)

**Scope**: Per `657c297` BLOCKED-pending-redesign plan, the original
"make MarlinScratch default-on for PF8" approach is empirically
broken (Arm B refute). H1' v2 needs ground-up workspace allocation
redesign — no per-call cudaMalloc, possibly pool-allocated buffers
sized at startup.

**Acceptance gate**: per `657c297` new criteria:
- OOM-regression A/B at conc=4 4k W4A16 (per da7f5a2)
- TTFT/tok-s regression A/B at same workload (per d09623a)
- PF8 conc=1 zero kernel failures over 60s
- Match/improve W4A8 perf bar (54.2ms / 11.9ms / 409 tok/s)

**Risk**: high — PF8 has been broken for ~weeks across multiple
refactor attempts; redesign may not converge. ~2-4 weeks.

**Why this isn't currently P1**: PF8.5 KILL came back 5878 failures
even at conc=1 with warmup OFF, suggesting the per-call alloc bug
isn't superficial. Medusa (Option A) gives more confident win path.

## §3 Recommendation matrix

| User priority | Recommended option | Time-to-result |
|---|---|---|
| Maximum tok/s (throughput) | **A (Medusa)** | 2-3 days |
| Maximum TTFT/ITL (latency) | **B (Hybrid)** | ~5 days |
| Resurrect PF8 path (W4A8 substitute) | **C (H1' v2)** | 2-4 weeks |
| **Default if no preference** | **A (Medusa)** | clearest pickup |

## §4 What Claude can do RIGHT NOW (without user decision)

If user wants to pre-empt their decision time:
- **Scaffold Medusa Phase 1.A dataset prep** (Alpaca download +
  tokenize, ~1-2 hr CPU work, no GPU; per `63769be`)
- **Read Task #30 Hybrid dispatch source** (current Linear dispatch
  in `linear.rs:2064-2095` per `1ba06f0`) to estimate Phase 1
  scope
- **Draft Option B Phase 1 brief** for codex pickup readiness

These don't commit to a decision, just reduce time-to-execution
for whichever option the user picks.

## §5 Status

**Open — awaits user direction**. Saturation acknowledged: 9+ idle
ticks since PF8.5 KILL with multiple PushNotifications dispatched.
This doc supersedes the implicit "blocked on user direction" status
in the loop prompts with explicit options + recommendation matrix.

## §6 Cross-references

- `0be278f` PF8.5 SUBSTRATE-KILL errors entry
- `7ed8160` Arm B refute (warmup-INDEPENDENT)
- `06b7437` Arm C control (W4A16 HEALTHY)
- `d8b2870` Arm D control (W4A8 HEALTHY) + perf comparison
- `657c297` H1' plan BLOCKED-pending-redesign
- `a7f913b` pickup queue P1 pivot to Medusa
- `63769be` Medusa Alpaca HF auth bypass
- SKILL `kernel-optimization` v1.15.0 (37 canonical anti-patterns)
