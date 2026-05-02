---
name: tmux-agent-control
description: Use this skill when the user asks to inspect, send a task to, queue work for, interrupt, or otherwise drive another coding-agent CLI (Codex / Claude Code / aider / ...) running inside a tmux session. Triggers include "看看 tmux 中 codex", "给 codex 发任务 X", "send X to codex", "推进 codex", "打断 codex / pivot", "check what the other agent is doing". Covers session discovery, single-line vs multi-line submit semantics (Enter vs Enter Enter), queue-vs-immediate behavior, scrollback capture, and don't-talk-to-yourself safety.
version: 1.1.0
---

# tmux-agent-control

Driving another coding-agent CLI (commonly Codex) that lives in a tmux session. Per CLAUDE.md §Delegation, this is the **only working execution path** for handing work to Codex from this project — the in-process subagent (`codex:codex-rescue`) and `mcp__openmax__execute_with_codex` both hang. The Codex review-via-Bash path (`codex review --uncommitted`) is unaffected and not what this skill covers.

## Quick reference

```bash
# 1. Discover
tmux ls

# 2. Read state (last screen)
tmux capture-pane -t <s>:<w> -p | tail -80

# 2b. Read state with history (for what scrolled off)
tmux capture-pane -t <s>:<w> -S -200 -p | tail -200

# 3a. Send single-line message
tmux send-keys -t <s>:<w> '继续' Enter

# 3b. Send multi-line message (note Enter Enter)
tmux send-keys -t <s>:<w> 'line one
line two' Enter Enter

# 4. Verify (always)
sleep 2 && tmux capture-pane -t <s>:<w> -p | tail -30

# 5. Interrupt (only if user asks)
tmux send-keys -t <s>:<w> Escape
```

## When to use

- "看看 tmux 里 codex 在干嘛" / "check what codex is doing"
- "给 codex 发任务 X" / "send X to codex" / "queue X for codex"
- "推进 codex" / "nudge codex" / "cron-loop 推进"
- "打断 codex" / "interrupt codex and tell it to pivot"
- Any time **another agent CLI is already running in tmux** and the user wants Claude to drive it.

If no agent is running yet and the user wants one started, that's out of scope — ask the user to start it themselves; tmux send-keys to launch a fresh CLI is fragile.

## Workflow

### 1. Discover sessions — never assume the name

```bash
tmux ls
```

Possible outputs and how to read them:
- `c: 1 windows ...` — CLAUDE.md historical convention put Codex at `c:0`. Sometimes still true, sometimes not.
- `1: 1 windows ...` — recent sessions have placed Codex at `1:0`. Don't assume.
- Multiple sessions — capture each and identify by banner.

**Identification by banner** (`tmux capture-pane -t <s>:<w> -p | tail -10`):
- Codex: footer shows `gpt-5.x ...` or model name + cwd; prompt char `›`.
- Claude Code (the one you're running in): banner shows `Claude Code v...`; prompt char `❯`. **Never send-keys here** — you'd be talking to yourself, creating a feedback loop where your tmux input arrives as a user message.
- aider / other: identify by banner before sending.

### 2. Read state before acting

```bash
tmux capture-pane -t <session>:<window> -p | tail -80
```

Look for these state markers:

| Marker | Meaning | Action |
|--------|---------|--------|
| `Working (Nm Ns ...)` | Agent mid-turn | New messages will queue, not interrupt |
| `Messages to be submitted after next tool call ↳ <text>` | Confirmed your message landed in the queue | Done — don't re-send |
| Empty input area + idle prompt (`›` / `❯`) | Idle | Send will execute immediately |
| Your text visible in input area, no `Working` | Submit failed (single Enter on multi-line) | Send `Enter Enter` to force submission |
| `Conversation interrupted` | Result of recent Escape | Input area may have stray text |

If the visible screen isn't enough (long Plan/Explored block scrolled off), use `-S -200` to grab history:

```bash
tmux capture-pane -t <s>:<w> -S -200 -p | tail -200
```

### 3. Send the message

**Single-line (the common case):**

```bash
tmux send-keys -t <session>:<window> '勇闯世界第一' Enter
```

**Multi-line (with embedded `\n` or a long directive):**

```bash
tmux send-keys -t <session>:<window> '请按以下顺序推进：
1. 跑 W3 bench
2. 把结果写进 wins/
3. 报告 Δ%' Enter Enter
```

Why the double Enter on multi-line: Codex CLI input area is multi-line by design. A single Enter inserts a newline *within* the prompt; only a second Enter on an empty line submits. Single-line sends submit fine with one Enter — they don't have intra-prompt newlines for the first Enter to "use".

If `capture-pane` after send shows the directive sitting in the input area with no `Working` status, the message didn't submit:

```bash
tmux send-keys -t <session>:<window> Enter Enter   # force submit
```

### 4. Verify (every send)

```bash
sleep 2 && tmux capture-pane -t <session>:<window> -p | tail -30
```

Three valid outcomes — anything else means re-try:

1. **Executing now**: `Working (Ns ...)` appeared, input area is empty.
2. **Queued behind current work**: banner reads `Messages to be submitted after next tool call ↳ <your text>`.
3. **Idle agent picked it up**: input area cleared and a new turn started in the transcript.

### 5. Interrupt (only when user asks for pivot)

```bash
tmux send-keys -t <session>:<window> Escape
```

Notes:
- Escape interrupts the current action; transcript shows `Conversation interrupted`.
- A partial directive may be left in the input area afterwards — capture-pane and clean up if needed.
- `Ctrl+U` / `Ctrl+D` / `Ctrl+C` do **not** reliably clear the input area. Cleanest recovery: let any queued stale message submit, then send a new directive that overrides.

## Common patterns

### Queue work behind a busy agent

If the agent is `Working`, just send — it'll queue and the next-tool-call boundary will flush it. Confirm via the `Messages to be submitted after next tool call` banner. Don't interrupt to "send sooner" unless the user explicitly wants the current direction abandoned.

### Periodic nudge / cron-loop 推进

The phrase "cron-loop 推进 codex" = on a schedule, send a brief directive (`继续` / `下一步` / a specific action) to keep it moving. Compose with the `loop` skill — `loop` owns the schedule, this skill owns the send-keys mechanics. Each tick: capture-pane → assess → send minimal nudge → verify.

### Pivot mid-task

User wants the other agent to drop its current direction:
1. `Escape` to interrupt.
2. `capture-pane` to confirm `Conversation interrupted` and check for stray input-area text.
3. Send the new directive with the appropriate Enter / Enter Enter.

### Status check without sending

User just asks "what's codex doing?" — only run discovery + capture-pane. Do not send anything. Summarize the current Plan / Explored / Working state and stop.

## Real example (from this repo's history)

User: "看看 tmux 中 codex 并且给它发任务 勇闯世界第一"

```bash
tmux ls
# → c: ... (Claude Code), 1: ... (Codex)

tmux capture-pane -t 1:0 -p | tail -80
# → gpt-5.5 banner, "Working (1m 04s)", Plan with 4 steps loaded

tmux send-keys -t 1:0 "勇闯世界第一" Enter

sleep 2 && tmux capture-pane -t 1:0 -p | tail -30
# → "Messages to be submitted after next tool call ↳ 勇闯世界第一"
# Confirmed queued. Done.
```

Reported back to user: which session, what Codex was doing (1m04s into A0 task), confirmation the new directive landed in the queue.

## Anti-patterns

- **Don't use `codex:codex-rescue` or `mcp__openmax__execute_with_codex` for execution.** They hang. Tmux is the path. (CLAUDE.md §Delegation.)
- **Don't assume the session name.** Always `tmux ls` first; the `c:0` convention is historical, not guaranteed.
- **Don't send to a Claude Code session.** Identify the banner before sending — sending to your own session creates a feedback loop.
- **Don't fire-and-forget.** Always `capture-pane` after sending to confirm it submitted vs. stuck in input area vs. queued.
- **Don't batch unrelated directives into one multi-line send** to "save round-trips" — the agent treats it as one prompt and may interleave the work confusingly. One intent per send.
- **Don't escape to "send faster".** Escape interrupts the current work; queueing is almost always the right move when the agent is mid-turn.

## Related

- `CLAUDE.md` §Delegation — authoritative on why tmux (not the in-process subagent) is the Codex execution path.
- `memory/feedback_codex_tmux_double_enter.md` — origin of the Enter-vs-Enter-Enter rule (2026-04-30 incident).
- `memory/feedback_claude_executes_codex_reviews.md` — current inverted-delegation preference (Claude implements, Codex reviews); overrides CLAUDE.md default until user reverses.
- `loop` skill — for scheduled periodic nudges.
