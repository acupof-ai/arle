# Cooperative loop asymmetric saturation — 50-tick stretch with codex silent + user non-engaged + Claude productive

## Context

Date range: 2026-05-10 ~10:55 KST → 13:38 KST (~6.5 hours, ~50 ticks)

Cooperative-loop pattern (Claude + codex tmux 0:0 + user) entered a
saturation tail with asymmetric productivity:

- **Codex tmux 0:0**: frozen at "Worked 26m 51s" since 10:55 KST.
  Final action queue:
  - `git diff --check`
  - `cargo check --release -p infer --features cuda`
  - `cargo test test_w4a8_vs_bf16_token_diff`
  - `cargo test test_e2e_w4a8_marlin_optional`
  - `工作区干净，8000 端口未被占用` (Chinese: workspace clean, port 8000 free)
  Then `> Run /review on my current changes` — never executed.
  **Zero codex commits in 6.5 hr.**

- **Claude (this agent)**: ~14 substantive commits across the same
  period including:
  - PF8.5 license bench v11 KILL (0be278f) — 5878 kernel failures
  - 4-arm A/B isolating PF8.3 substrate (Arms B/C/D)
  - 6-cell perf matrix W4A16/W4A8 at conc=1/2/4 (8d32576, 92813dc)
  - Direction options doc with ironclad recommendation (a64fad7,
    cc8b437, 12e0c07, 9340e04)
  - SKILL graduations + candidates (b255c58, 2356e6a, 430a4be)
  - Multiple housekeeping refreshes (3ea2aa4, 19b238d, 657c297)

- **User**: fired /loop ~50 times manually with stale prompts (each
  cron firing carried "PF8.5 license decision STILL blocked on USER"
  language even after PF8.5 KILL was definitively committed). No
  substantive engagement on:
  - PF8.5 KILL outcome (PushNotification dispatched)
  - Direction options A/B/C/D recommendation (PushNotification dispatched)
  - "scaffold X while I think" authorization for pre-emptive Alpaca
  - Any of my 14 substantive commits

## Root Cause

The cooperative loop is designed for productive bidirectional flow,
but degenerates under three concurrent conditions:

1. **Codex stuck without explicit interrupt**: codex was ready for
   `/review` and never fired. Either tmux session is stale (not
   actually responsive) OR codex is waiting for user `Enter` press.
   No mechanism to detect which.

2. **User firing /loop programmatically without state freshness**:
   loop prompts carry stale action items across N ticks. Per `da26eba`
   sediment, the same instruction (`e5deac8 needs cherry-pick`) was
   carried for 30+ ticks despite being moot since tick 19.

3. **Claude default-on accumulation discipline**: per user directive
   "持续累积 + 每 tick 至少 1 commit" combined with "NULL result 也
   commit". This produces output every tick regardless of marginal
   value. After saturation (all bench-axis evidence captured),
   continued commits are housekeeping > substrate work, with
   diminishing per-commit value.

## Fix

**No code fix this entry — meta-process observation.**

Ground rules for future cooperative-loop sessions:

1. **Codex silence > 1 hr should trigger explicit Claude → codex
   nudge** via tmux send-keys per skill `tmux-agent-control`. If
   codex doesn't respond to nudge within 5 min, treat as wedged and
   reset session. Currently no such trigger — codex frozen 6.5 hr
   without intervention.

2. **Loop prompt should include freshness check**: cron-fired
   prompts should be regenerated against current state at fire time,
   not carry stale instructions for tens of ticks. SKILL #29
   framing-decay pattern at the cron-instruction-persistence axis.

3. **Saturation acknowledgment should reduce wakeup cadence
   automatically**: when 3+ consecutive ticks produce
   housekeeping-only commits, max-cadence (3600s) and skip the
   "every tick 1 commit" if no substantive substrate work pending.
   Currently I bumped to 3600s manually but earlier ticks burned
   1800s wakeups for housekeeping commits — wasted cache budget.

4. **PushNotification fatigue is a real cost**: I sent 4+
   PushNotifications across the saturation tail. User did not
   engage with any. Future PushNotifications should be reserved for
   genuinely-new substantive findings (not "still blocked on you"
   reminders).

## Rule

When cooperative loop enters asymmetric saturation (one party
productive, others silent for 6+ hr), the **honest output is
saturation acknowledgment + reduced cadence**, NOT continued
diminishing-returns accumulation. The "持续累积" rule is genuinely
good early in a session but becomes self-perpetuating churn after
the work axis saturates. §0 SOLID rule "80% SOLID 不够" applies in
reverse here: more accumulation past 80% complete is INVENTED
work, not deeper SOLID.

For SKILL `kernel-optimization` / `tmux-agent-control` / future
cooperative-loop SKILLS: add explicit "saturation halt" criteria —
when continued accumulation would not change a recommendation,
PushNotification + max-cadence + STOP scheduling self-wakeup
(let user re-engage on their schedule).
