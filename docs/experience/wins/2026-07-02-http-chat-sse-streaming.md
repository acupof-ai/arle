# Chat SSE streaming shipped — /v1/chat/completions stream=true (closes #79)

> Status: Shipped — `30ad158f`, 2026-07-02.

## Context

`stream=true` on `/v1/chat/completions` had been 400-stubbed since the R5
rewrite tranche ("deferred in R5 tranche 2", `schema.rs`), while
`/v1/completions` SSE already worked (guidellm consumes it). OpenAI-compat
clients that default to streaming failed outright, and chat ITL was
unmeasurable per-token. Issue #79.

## What Worked

- Mirrored the proven `/v1/completions` SSE block in `coordinator.rs`
  (streaming_submit → bounded chunk channel → guard-in-task → `data: [DONE]`),
  chat-specific parts only: `chatcmpl-` ids, role on the first delta,
  `chat.completion.chunk` shape.
- `StreamingReasoningSplitter` (`sse_util.rs`) — incremental twin of the
  non-streaming `split_reasoning`: reasoning until the first `</think>`
  (dropped; opener/closer may span decode-batch boundaries via a held-back
  prefix buffer), content after; thinking-off is byte-identical passthrough
  with zero scans. Truncated thinking flushes pending as reasoning.
- Multimodal chat + stream=true fails closed (400) — explicit, not silent.

## Evidence

- 48 unit tests green in `infer-server` (8 new: chunk shape, passthrough,
  closer/opener straddling pushes, truncation, opener strip); clippy clean;
  `infer-api` consumer compiles (cpu,no-cuda).
- Live contract smoke, local Metal serve (`Qwen3.5-0.8B-MLX-4bit`, M4 Pro,
  `INFER_TEST_MODEL_PATH` small-model exception — HTTP-layer change,
  backend-agnostic):
  - thinking off: role on first delta → content deltas → empty-delta
    `finish_reason:stop` → `[DONE]`.
  - thinking on, tight budget: all `reasoning_content`, truncation finishes
    clean (`length`) — matches non-streaming policy.
  - thinking on, 500 tokens: 121 reasoning frames → `</think>` swallowed
    (0 occurrences on the wire) → content frames → `stop` → `[DONE]`.
  - non-stream chat + completions SSE: unchanged (regression smoke).

## Bench note (explicit deviation)

No guidellm run for this entry: the non-stream hot path gains only one
`request.stream` branch check before the existing code (byte-identical
behavior verified by the regression smoke), and the canonical Metal model
was in use by a parallel session (no-concurrent-Metal-loads rule). Chat SSE
*unlocks* per-token chat ITL for future guidellm runs — the first canonical
sweep that uses it becomes the missing baseline.

## Rule

When a deferred surface has a proven sibling (completions SSE), the ship
path is mirror-the-sibling + an incremental adapter for the one semantic
difference (reasoning split) — not a redesign. The splitter's held-back
prefix buffer is the reusable pattern for any marker that can straddle
streaming chunk boundaries.
