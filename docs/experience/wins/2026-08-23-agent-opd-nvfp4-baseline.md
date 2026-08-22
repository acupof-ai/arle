# Agent-OPD baseline on an NVFP4 27B: the harness works, the model does not act

> Status: Baseline recorded

## Context

First agent RL run to reach real rollouts. Three infrastructure faults had to
clear first, and each one produced an all-zero reward that reads exactly like a
capability result:

1. No `claude` binary on the pod — 968 rollouts died in `spawn`, `wall=0.0s`.
2. Unsolicited Anthropic `thinking` blocks aborted CC's stream — `wall≈40s`,
   `turns=Some(1..2)`, zero tokens counted. See
   `errors/2026-08-23-anthropic-thinking-blocks-abort-claude-code.md`.
3. `--max-update-seq` at its 23000 default silently skipped every trajectory —
   a CC prompt is ~24k tokens, so even a scored rollout never reached training.

## Setup

`ThinkingCap-Qwen3.6-27B-NVFP4`, `--share-frozen-base` (256 borrowed
projections), `--lora-target-set attention-qv`, `--max-update-seq 131072`,
218 security-filtered localized swe-smith tasks, 4 samples per task, 1×H20.

## Result

16 rollouts: `edited=false` 16/16, `passed=false` 16/16, `SKIP trajectory` 0.
Turns are 1 (7), 2 (4), 4 (1) — the model stops after announcing intent.

Running the same agent prompt by hand returns 42 tokens:

> Let me quickly find the relevant code.

and ends the turn. `is_error=false`, `terminal_reason=completed` — a clean
finish, not a fault.

## The tool path is not the cause

A direct `/v1/messages` call with one tool defined returns a well-formed call:

```
stop_reason: tool_use
{"type":"tool_use","name":"Read","input":{"file_path":"src/main.py"}}
```

So the model emits tool calls and the server surfaces them. What fails is
sustaining them: given a 42k-token agent context and a multi-step instruction,
the model narrates one sentence instead of acting. Given a one-line "call the
tool now", it acts.

## Rule

Prove the harness completes one turn before reading a score as capability.
`turns`, `output_tokens`, and `terminal_reason` separate a broken pipe from a
weak model; all three were needed here, and each fault looked identical from
the reward alone.
