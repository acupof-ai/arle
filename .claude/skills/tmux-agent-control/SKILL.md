---
name: tmux-agent-control
description: Use this skill when ckl asks to inspect, queue work for, interrupt, spawn, replace, or otherwise drive another coding-agent CLI running inside tmux. Covers ARLE's known-safe tmux path for Codex/Claude Code delegation, including session discovery, capture-pane status checks, Enter semantics, long-brief buffer paste, queue-vs-immediate behavior, and don't-send-to-yourself safety.
version: 1.3.0
---

# tmux-agent-control

Driving another coding-agent CLI (commonly Codex) that lives in a tmux session. Per CLAUDE.md §Delegation, this is the **only working execution path** for handing work to Codex from this project — the in-process subagent (`codex:codex-rescue`) and `mcp__openmax__execute_with_codex` both hang. The Codex review-via-Bash path (`codex review --uncommitted`) is unaffected and not what this skill covers.

## Quick reference

```bash
# 1. Discover (sessions AND window indices — the latter is NOT always 0)
tmux ls
tmux list-windows -a

# 2. Read state (last screen)
tmux capture-pane -t <s>:<w> -p | tail -80

# 2b. Read state with history (for what scrolled off)
tmux capture-pane -t <s>:<w> -S -200 -p | tail -200

# 3a. Send single-line message
tmux send-keys -t <s>:<w> '继续' Enter

# 3b. Send short multi-line message (≤5 lines, no special chars)
tmux send-keys -t <s>:<w> 'line one
line two' Enter Enter

# 3c. Send a LONG brief (>5 lines or special chars) via tmux buffer
#     send-keys with embedded newlines fails for long input — tmux throws
#     repeated "not in a mode" errors and drops keys. Use the buffer instead.
cat > /tmp/brief.txt <<'EOF'
Track X — full directive, can be 30+ lines, no escaping needed.
EOF
tmux load-buffer -b brief /tmp/brief.txt
tmux paste-buffer -b brief -t <s>:<w>
sleep 1 && tmux send-keys -t <s>:<w> Enter Enter Enter Enter

# 3d. Spawn a peer Claude Code / Codex
tmux new-session -d -s 6 -c /path/to/repo 'claude --allow-dangerously-skip-permissions'
tmux new-session -d -s 7 -c /path/to/repo 'codex --dangerously-bypass-approvals-and-sandbox'
sleep 5 && tmux capture-pane -t 6:1 -p | tail -10   # confirm banner

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

If no agent is running yet and the user wants one started, see "Spawn a peer agent" in Common patterns — `tmux new-session -d ... 'claude --allow-dangerously-skip-permissions'` is reliable.

## Workflow

### 1. Discover sessions — never assume the name

```bash
tmux ls
tmux list-windows -a    # always do this too — window indices vary
```

Possible outputs and how to read them:
- `c: 1 windows ...` — CLAUDE.md historical convention put Codex at `c:0`. Sometimes still true, sometimes not.
- `1: 1 windows ...` — recent sessions have placed Codex at `1:0`. Don't assume.
- Multiple sessions — capture each and identify by banner.

**Window indices are not always `:0`.** If `tmux capture-pane -t 1:0 -p` returns `can't find window: 0`, run `tmux list-windows -a` — `1:1: ssh* ...` means window index 1, not 0. Use `1:1` going forward. Modern Claude Code / Codex sessions started via `tmux new-session` typically land on window `:1`.

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
| `Working (Nm Ns ...)` (Codex) / `Beboppin'…` / `Elucidating…` / `Cooking…` (Claude Code) | Agent mid-turn | New messages will queue, not interrupt |
| `Messages to be submitted after next tool call ↳ <text>` | Confirmed your message landed in the queue | Done — don't re-send |
| Empty input area + idle prompt (`›` / `❯`) | Idle | Send will execute immediately |
| Your text visible in input area, no `Working` | Submit failed (single Enter on multi-line) | Send `Enter Enter` to force submission |
| `[Pasted text #N +M lines]` placeholder (Claude Code) | Long paste/buffer landed but NOT submitted | Send `Enter Enter Enter Enter` to submit |
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

**Short multi-line (≤5 lines, no special chars):**

```bash
tmux send-keys -t <session>:<window> '请按以下顺序推进：
1. 跑 W3 bench
2. 把结果写进 wins/
3. 报告 Δ%' Enter Enter
```

Why the double Enter on multi-line: Codex/Claude Code input area is multi-line by design. A single Enter inserts a newline *within* the prompt; only a second Enter on an empty line submits. Single-line sends submit fine with one Enter — they don't have intra-prompt newlines for the first Enter to "use".

**Long brief (>~5 lines, special chars, or full task directive) via tmux buffer.** `send-keys` with embedded newlines fails for long input — tmux's key parser chokes and emits a flood of `not in a mode` errors, dropping keys silently. The reliable path is to load the message into a tmux buffer and paste it:

```bash
# 1. Write the brief (Write tool or heredoc — no escaping needed inside)
cat > /tmp/brief.txt <<'EOF'
Track X — full directive here, can be 30+ lines, with backticks, quotes,
slashes, whatever. The buffer ships it verbatim.

  - bullet one
  - bullet two
EOF

# 2. Load + paste + pump 4 Enters to submit
tmux load-buffer -b brief /tmp/brief.txt
tmux paste-buffer -b brief -t <session>:<window>
sleep 1 && tmux send-keys -t <session>:<window> Enter Enter Enter Enter
```

Notes on the buffer recipe:
- The buffer name (`-b brief`) is arbitrary; just keep it scoped per send so concurrent buffers don't collide.
- **Always pump 4 Enters after pasting.** 2 is rarely enough for a long paste; Claude Code in particular shows `[Pasted text #N +M lines]` and needs 4 to actually submit.
- Verify with capture-pane: agent should show `Working` (executing) or `Messages to be submitted after next tool call` (queued). If still showing `[Pasted text #N +M lines]` placeholder or the raw directive in the input area, pump 2 more Enters.

If `capture-pane` after send shows the directive sitting in the input area, OR Claude Code shows a `[Pasted text #N +M lines]` placeholder, with no `Working` status, the message didn't submit:

```bash
tmux send-keys -t <session>:<window> Enter Enter   # force submit
```

**Long messages need more Enters.** For directives over ~200 chars / wrapping ≥3 visual lines, the original "Enter Enter" recipe is sometimes not enough — observed needing **4–6 Enters total** before Codex submits. If a recheck shows the message + multiple blank lines + footer (no `Working`, no `Create a plan?` hint), keep sending 2-Enter bursts and re-checking until `Working` appears or the input area clears to the `Implement {feature}` placeholder. Cap at ~10 total Enters before assuming the session is wedged and escalating (Escape + investigate).

**The `Create a plan?  shift + tab use Plan mode   esc dismiss` hint.** For long directives, Codex sometimes interrupts with this hint instead of starting work. Two valid responses:
- **Want plan-first execution** (codex drafts a plan, you review next tick before code lands): `tmux send-keys -t <s>:<w> "S-Tab"` (Shift+Tab) — enters Plan mode.
- **Want direct execution** (the directive already has enough detail): `tmux send-keys -t <s>:<w> Escape` to dismiss the hint, then `Enter Enter Enter Enter` to actually submit (Escape only dismisses the offer; the directive stays in the input area until you Enter through it). Note: when dismissing, Escape here does NOT interrupt work because no work has started yet — different semantics from Escape during `Working`.

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

### Spawn a peer agent

User wants you to start a fresh peer agent (typically a peer Claude Code or Codex) to take on parallel work. Do it via `tmux new-session`:

```bash
# Peer Claude Code (bypassing permission prompts, in the right cwd)
tmux new-session -d -s 6 -c /path/to/code/agent-infer 'claude --allow-dangerously-skip-permissions'

# Peer Codex (YOLO mode)
tmux new-session -d -s 7 -c /path/to/code/some-other-repo 'codex --dangerously-bypass-approvals-and-sandbox'

# Wait ~5s for startup, then verify the banner came up clean (Claude Code v..., or OpenAI Codex v...)
sleep 5 && tmux capture-pane -t 6:1 -p | tail -10
```

Notes:
- Pick a `-s` (session name) not already in `tmux ls`.
- The new window will land at index `:1` (use `tmux list-windows -a` to confirm).
- Set `-c <cwd>` to where the agent should operate. For Claude Code this also seeds its CLAUDE.md context.
- After spawn, send the brief via the §3 Long-brief buffer recipe and verify `Working` appears.

### Replace one agent with another (quota / limit handoff)

Common when an agent (typically Codex) hits its weekly limit and the user wants the same work moved to a fresh agent:

1. **Spawn the replacement** (recipe above) in the right cwd. Verify the banner.
2. **Send the same brief via buffer** to the replacement. Adjust paths if the new cwd differs (e.g. drop absolute prefixes if the new cwd is now the target repo).
3. **Stand down the original**: `tmux send-keys -t <orig>:<w> Escape` to interrupt, then send a one-liner explaining the handoff so it doesn't sit at "Conversation interrupted" expecting input. Example: `Track X has been handed off to a fresh agent on session N. Stay idle, do nothing further.`
4. **Verify both**: capture-pane on the replacement (should show `Working`); capture-pane on the original (should be at idle prompt with no pending work).

## Anti-patterns

- **Don't use `codex:codex-rescue` or `mcp__openmax__execute_with_codex` for execution.** They hang. Tmux is the path. (CLAUDE.md §Delegation.)
- **Don't assume the session name.** Always `tmux ls` first; the `c:0` convention is historical, not guaranteed.
- **Don't send to a Claude Code session.** Identify the banner before sending — sending to your own session creates a feedback loop.
- **Don't fire-and-forget.** Always `capture-pane` after sending to confirm it submitted vs. stuck in input area vs. queued.
- **Don't batch unrelated directives into one multi-line send** to "save round-trips" — the agent treats it as one prompt and may interleave the work confusingly. One intent per send.
- **Don't escape to "send faster".** Escape interrupts the current work; queueing is almost always the right move when the agent is mid-turn.
- **Don't `send-keys` long multi-line briefs directly.** For directives over ~5 lines or with backticks/quotes/colons, use the §3 Long-brief buffer recipe (`load-buffer` + `paste-buffer`). Direct `send-keys` triggers a flood of `not in a mode` errors and silently drops keys.
- **Don't assume window index 0.** Run `tmux list-windows -a` first; many sessions live at `:1`. `can't find window: 0` is the classic symptom of guessing.

## Related

- `CLAUDE.md` §Delegation — authoritative on why tmux (not the in-process subagent) is the Codex execution path.
- `memory/feedback_codex_tmux_double_enter.md` — origin of the Enter-vs-Enter-Enter rule (2026-04-30 incident).
- `loop` skill — for scheduled periodic nudges.
