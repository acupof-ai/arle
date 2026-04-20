# CLI agent mode must not fabricate tool turns

## Context

In CLI agent mode, a natural-language question like "本地有什么文件" triggered a
shell tool call before the model produced any tool request, then returned a
templated follow-up line as if the assistant had naturally decided to inspect
the filesystem.

## Root Cause

`AgentSession::run_turn` had a fast path that called
`recover_tool_calls_from_user_request` before the first model turn. The builtin
tool policy also special-cased directory-listing requests and synthesized a
fixed "Listed the current directory above." response after the shell command
finished.

That made the interaction look model-authored even though the model never chose
the tool or wrote the reply.

## Fix

Require the model to request tools explicitly. Keep deterministic recovery only
for malformed model drafts, not raw user text, and remove canned shell
follow-up text for directory listings.

## Rule

In agent mode, never fabricate a tool call or assistant reply directly from the
user request when that would make the turn appear model-authored.
