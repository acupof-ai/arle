# Agent Trajectory Export

> Status: **Phase 1 (v1) shipped 2026-04-29.** v2 (token IDs +
> `response_mask`) deferred — see "v2 follow-up plan" below.

ARLE is now an RL-training-first runtime: every agent turn must be
replayable so we can run reward models, GRPO, GSPO, and friends
against the same trajectories the live REPL produced. This doc is
the source of truth for the on-disk JSONL schema and the migration
policy that gates schema bumps.

The CLI flag is `--trace <path>` (writes one JSONL record per agent
turn). `--trace-prompts off` blanks the per-sub-turn `prompt_text`
field (prompts dominate trace size and can leak operator data).

---

## v2 schema (shipped — token layer live; supersedes v1)

v1 (`schema_version: 1`) records are no longer emitted. v2 is fully
backwards-readable for code that ignores `tokens`. The bump per the
"format change" rule below: when `tokens` started populating with
`Some(TokensRecord)` instead of `null`, the version moved.

```json
{
  "schema_version": 2,
  "ts": "2026-04-29T10:50:00Z",
  "turn_id": "<uuid v4>",
  "model_id": "Qwen3.6-35B-A3B-4bit",
  "backend": "metal",
  "user_input": "<verbatim user input string>",
  "messages": [
    {"role": "user", "content": "<text>"},
    {"role": "assistant", "content": [
      {"type": "text", "text": "<assistant reasoning text or empty>"},
      {"type": "tool_use", "id": "tu_0_0", "name": "shell", "input": {"command": "ls"}}
    ]},
    {"role": "tool", "tool_use_id": "tu_0_0", "content": "<tool stdout>", "result_truncated": true}
  ],
  "sub_turns": [
    {
      "index": 0,
      "prompt_text": "<full ChatML sent to engine, or null when --trace-prompts off>",
      "completion_text": "<raw text the engine returned, including <tool_call> XML>",
      "usage": {"prompt_tokens": 1234, "completion_tokens": 32},
      "ttft_ms": 420,
      "decode_secs": 0.8,
      "finish_reason": "stop"
    }
  ],
  "tokens": null,
  "result": {
    "text": "<final assistant text returned to user, possibly empty>",
    "terminal_state": "stop",
    "total_prompt_tokens": 12340,
    "total_completion_tokens": 286,
    "wall_secs": 202.9
  }
}
```

### Field-by-field rationale

- **`schema_version: 1`** — literal `i32` constant
  (`agent::TRAJECTORY_SCHEMA_VERSION`). Never magic-numbered in JSON.
  Increments only on rename/remove (see migration policy).
- **`ts`** — ISO-8601 UTC seconds, RFC-3339 form, no fractional
  seconds. Captured at write time by the CLI's `format_iso8601_utc`
  helper — no `chrono` dependency, deterministic to the second.
- **`turn_id`** — UUID v4. Unlike `tool_use_id` (which is
  deterministic), the per-turn id is intentionally random so
  trajectories from independent runs cannot collide in a shared
  store. Generated in the trace writer, not the agent loop, so
  re-running the same input twice produces two distinct turn ids.
- **`model_id` / `backend`** — pulled from the live engine and
  backend label (`metal` / `cuda` / `cpu`). Pinned in the record so
  the trajectory survives engine swaps mid-shard.
- **`user_input`** — verbatim trimmed user string. Recovery hooks
  (e.g. shell-listing-from-Chinese) read this; RL also uses it as
  the prompt anchor.
- **`messages`** — Anthropic-style. User and tool messages carry a
  plain string `content`; assistant messages always carry a content
  block array even when there's only text — keeps the consumer's
  invariant uniform.
- **`messages[].tool_use_id`** — deterministic
  `tu_<sub_turn_index>_<call_index_within_subturn>`. Stable across
  re-runs given the same input, so trajectories diff cleanly.
- **`messages[].result_truncated`** — set on `role: tool` messages.
  Mirrors `tools::ToolExecutionMetadata::truncated`, which is `true`
  iff the tool result string ends with the
  `\n... (output truncated)` marker emitted by
  `crates/tools/src/lib.rs::collect_output`.
- **`sub_turns[]`** — one entry per `complete_stream` invocation.
  Crucially, the recovered-user-request branch (deterministic
  policy hooks) does NOT call the engine and does NOT emit a
  sub-turn; the assistant message it synthesizes still appears in
  `messages` so RL doesn't lose the "what did the agent decide"
  signal.
- **`sub_turns[].prompt_text`** — full ChatML prompt the engine saw,
  exactly as `format_prompt(&messages, tools)` returned it. `null`
  when `--trace-prompts off`. Trace consumers that re-tokenize
  should refuse to run when this is null and `--trace-prompts on`
  was expected.
- **`sub_turns[].ttft_ms`** — per-sub-turn TTFT, in milliseconds.
  Measured from the `complete_stream` call site to the first
  non-empty delta. NOT the per-turn TTFT we already surface as
  `AgentTurnResult::time_to_first_token`; that metric covers the
  whole turn, this one zooms into the individual sub-call.
- **`sub_turns[].decode_secs`** — wall-clock seconds for this
  sub-turn (entire `complete_stream` duration).
- **`sub_turns[].finish_reason`** — `"stop"` / `"length"`. Mapped
  from `infer::server_engine::FinishReason` via
  `finish_reason_to_str`.
- **`tokens`** — JSON `null` in v1. Reserved for v2's
  `{prompt_ids, response_ids, response_mask}`.
- **`result.terminal_state`** — one of:
  - `stop`: tool_calls empty AND content non-empty → loop's normal
    "model produced final answer" exit.
  - `max_turns`: max_turns hit before final answer.
  - `empty_no_progress`: tool_calls empty AND `content.trim()`
    empty (the bug surface we caught reviewing the REPL last week).
  - `policy_short_circuit`:
    `tool_policy.finalize_after_tool_execution` returned `Some(_)`.
- **`result.wall_secs`** — total elapsed seconds for the turn,
  same monotonic anchor as `time_to_first_token`.

---

## Why dual-layer (token IDs deferred)

The user reviewed three industry approaches before approving v1:

- **OpenInference (Arize)** — message-level capture for
  observability dashboards. No token IDs, no logprobs. Optimized
  for "what did the agent do", not "what did the model see at the
  token level". v1 borrows this layer.
- **verl `AgentLoopOutput`** — token-aligned dump used to drive
  GRPO/GSPO directly off cached trajectories.
  `prompt_ids`/`response_ids`/`response_mask` carried at the
  trajectory level so the trainer can recompute logprobs without
  re-tokenizing. This is what v2 will mirror.
- **Anthropic content blocks** — assistant messages are always
  content-block arrays, with `tool_use` blocks correlated by id to
  a following `tool` message. v1 adopts this verbatim — it's the
  only one of the three that survives mid-turn tool-use without
  ambiguity, and it's what every modern hosted SDK now emits.

Token IDs require a non-trivial engine change (see v2 plan), and
v1 is enough for the human-readable RL trace surface and for
classical reward-model evaluation. Shipping v1 first decouples the
trace-format decision from the engine API decision.

---

## v2 follow-up plan

When v2 lands, `schema_version` bumps to `2` and `tokens` populates
with the verl-shaped object:

```json
"tokens": {
  "prompt_ids": [1, 2, 3, ...],
  "response_ids": [4, 5, 6, ...],
  "response_mask": [1, 1, 0, 1, ...]
}
```

Engine-side changes required:

1. **`infer::server_engine::CompletionStreamDelta`** gains
   `token_ids: Vec<u32>` — the per-delta token IDs the backend
   emitted. Today the trait only ships `text_delta`; backends
   detokenize internally.
2. **Backends Metal/CUDA/CPU** each populate the new field.
   - Metal: `infer/src/backend/metal/runtime.rs` decode path
     already has `token_id` in scope; surface it through the
     stream sender.
   - CUDA: `infer/src/backend/cuda/scheduler.rs` similar — the
     scheduler hands us `Vec<u32>` already.
   - CPU: `infer/src/backend/cpu/runtime.rs` mirrors CUDA.
3. **Agent loop** accumulates `prompt_ids` from the first
   sub-turn's tokenized prompt (one tokenization, not per
   sub-turn — re-tokenization would let drift creep in) and
   `response_ids` + `response_mask` across all engine sub-turns.
   Tool result tokens get `response_mask = 0`; assistant tokens
   get `response_mask = 1`.
4. **`AgentTurnResult`** gains `tokens: Option<TokensRecord>`,
   wired through `run_turn_inner` analogous to v1's `messages` /
   `sub_turns` plumbing.

The CLI's `--trace` plumbing does NOT change in v2 — only the
record body. Consumers gated on `schema_version >= 2` opt into the
new fields.

---

## Migration policy

- **Additive fields don't bump.** Adding optional or
  always-present fields whose absence is recoverable (e.g.
  `model_revision`, `latency_breakdown`) keeps the version at the
  current major. Consumers must tolerate unknown fields.
- **Renames or removals bump the version.** If we ever rename
  `terminal_state` or remove `tool_use_id`, that's a major bump.
- **Semantic changes bump.** If `ttft_ms` ever switches to
  microseconds, that's a major bump even though the field stayed
  the same name.
- **Phase boundary bump = automatic.** v1 → v2 is the first such
  transition; the existence of `tokens: null` in v1 is the only
  concession to the future shape.

A consumer should branch on `schema_version` first thing after
parsing the JSON envelope.

---

## Where the code lives

| Concern | Path |
|---|---|
| Schema constant + types | `crates/agent/src/lib.rs` (`TRAJECTORY_SCHEMA_VERSION`, `TrajectoryMessage`, `ContentBlock`, `SubTurnRecord`, `TerminalState`, `AgentTurnResult` extension) |
| Tool metadata | `crates/tools/src/lib.rs` (`ToolExecutionMetadata`, `execute_tool_call_with_metadata`, `TOOL_RESULT_TRUNCATION_MARKER`) |
| Trait hook | `crates/agent/src/lib.rs::ToolExecutor::execute_with_metadata` (default impl synthesizes neutral metadata; `BuiltinToolExecutor` overrides) |
| CLI flags | `crates/cli/src/args.rs` (`Args::trace`, `Args::trace_prompts`, `TracePromptsMode`) |
| Writer | `crates/cli/src/trace.rs` (`TraceWriter`, `AgentTrajectoryRecord`, `TrajectoryResult`) |
| REPL wiring | `crates/cli/src/repl.rs` (`run_repl`, `run_one_shot`, `run_agent_turn`) |
| Plumbing | `crates/cli/src/lib.rs::run_impl` opens the writer and threads `Option<&TraceWriter>` through both run paths |

---

## IO failure policy

Trace IO failures are **never** propagated to the REPL turn.
`TraceWriter::write_turn` logs at `warn` level and drops the
record. The run is the source of truth; the trace is a best-effort
sidecar. If the disk fills mid-run, the agent keeps responding and
the operator sees a warn line per dropped record.

The one exception: opening the trace file. We surface that error
on startup so an operator who typo'd a path doesn't run for an
hour before noticing nothing was written.

---

## Determinism notes

- `tool_use_id` IS deterministic: same input produces the same
  `tu_<sub>_<call>` ids across re-runs. This is required for
  trajectory diff workflows.
- `turn_id` is NOT deterministic: it's a fresh UUID v4 per write.
  This is required for de-duplication in shared trajectory
  stores.
- `ts` is wall-clock time of the write call, not the turn start.
  Wall-clock duration of the turn is in `result.wall_secs`.

---

## Reference: industry approaches

| Source | Layer | Notes |
|---|---|---|
| OpenInference | message-level | Used by Arize / Phoenix. Drop-in for observability; no token IDs. |
| verl `AgentLoopOutput` | token-level | Drives GRPO/GSPO. Fields: `prompt_ids`, `response_ids`, `response_mask`. |
| Anthropic content blocks | message-level | `assistant.content[]` with `text` + `tool_use` blocks; `tool_use_id` correlates a later `tool` message. |

ARLE v1 = OpenInference + Anthropic. v2 layers verl on top.
