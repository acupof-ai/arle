# Agent-Workload API Extensions

**Status:** Draft — opened 2026-05-03 by Track B
**Owner:** ckl
**Drives:** [`projects/2026-05-02-agent-load-mission-expansion.md`](../projects/2026-05-02-agent-load-mission-expansion.md)
W3/W4 mission, plus Eli/nexil interop on Apple-Silicon Metal.
**Related:** [`plans/2026-05-02-agent-load-bench-spec.md`](2026-05-02-agent-load-bench-spec.md)
locks the workload contract; this doc proposes the API surface that
contract sits on top of. [`infer/src/http_server/AGENTS.md`](../../infer/src/http_server/AGENTS.md)
governs the wire format. [`scripts/bench_eli_agent.sh`](../../scripts/bench_eli_agent.sh)
is the harness that exercises the proposed extensions end-to-end against
[`/path/to/code/eli`](https://github.com/cklxx/eli).

This is a measurement+API contract, not an implementation. Code lands in
follow-up commits gated on the open questions in §6.

---

## 1. Goal

Define the *minimum* set of HTTP-surface extensions agent-infer needs so
that an agent host (Eli via `nexil`, or any OpenAI-compatible client) can:

1. **Stream tool-call deltas** through `/v1/chat/completions` — the single
   biggest hard-blocker today.
2. **Verify session routing** end-to-end without polling `/v1/stats`.
3. **Hint KV retention and admission priority** so a 128-session W4 trace
   does not silently lose its 8k prefixes between warmup and resume.
4. **Drive prefix-cache-aware batching** so interactive agent traffic is
   not starved by background batch traffic on a shared backend.

Required properties for every extension:

- **Stock-OpenAI clients are unaffected.** Unknown fields are tolerated;
  echoed extension fields use namespaced keys (`prefix_reuse`, `agent`,
  `cache_strategy`) and serialize only when the request asks for them.
- **No second mapping layer.** Every new field plumbs through
  `RequestExecutionOptions` once and lands in `IncomingRequest`
  (`AGENTS.md` invariant 3).
- **Per-request observability is opt-in but free when on.** Counters
  already live in `latest_request_cache` / `session_cache`
  (`infer/src/metrics/render.rs:820`); echoing them in the response is
  cheap.
- **Backwards-compat with existing benches.** `bench_agent_trace.py`,
  `bench_guidellm.sh`, and the `bench-agent-load-*` wins entries do not
  need to change to keep working.

Non-goals for this plan:

- No model / scheduler change — admission and retention work belong in
  follow-up plans (`tiered-kv-hicache-readmission.md`).
- No new transport (no gRPC, no WebSocket).
- No auth / tenancy model — the bearer-token check stays as-is.
- No multi-model routing — `served_model_id` validation stays as-is.

---

## 2. Gap Analysis — what nexil expects vs what http_server exposes

Anchored in [`/path/to/code/eli/crates/nexil/src/clients/chat.rs`](../../../eli/crates/nexil/src/clients/chat.rs)
and `crates/nexil/src/providers/openai.rs`. nexil treats agent-infer as an
OpenAI provider, so the gap is everything nexil sends that we drop and
everything nexil expects that we never send.

| # | nexil expectation | agent-infer today | Gap |
|---|-------------------|-------------------|-----|
| G1 | Stream deltas may carry `delta.tool_calls[]` with `id`, `index`, `function.name`, `function.arguments` chunks. `ToolCallAssembler` exists specifically to re-assemble them. | `ChatCompletionRequest::validate` rejects `stream=true` whenever `tools` is non-empty (`openai_v1.rs:872`). Non-streaming responses do emit `tool_calls` correctly. | **P0 hard-block.** Any Eli streaming turn that exposes tools fails before the model runs. |
| G2 | Final stream chunk carries `usage` when `stream_options.include_usage=true` (already supported for chat). Clients infer cache reuse only by polling `/v1/stats`. | We emit `usage` correctly, but never echo per-request `matched_prefix_tokens` / `resume_prefill_tokens` / `session_affinity_hit` to the client. Clients must scrape `/v1/stats?format=json`. | **P0 observability gap.** A client cannot prove session affinity without out-of-band polling — racy under concurrent traffic. |
| G3 | nexil sends `tool_choice` (auto / none / required / specific) and `response_format` (json_schema) on every `tool_calls` request. | `ChatCompletionRequest` has no `tool_choice` and no `response_format` — `serde(deny_unknown_fields)` will reject the request outright. | **P0 hard-block** for nexil tool-loop default path. |
| G4 | nexil treats `session_id` as an opaque routing hint; it does not consume any retention contract from the server. | Routing exists, but `matched_prefix_tokens` collapses from 8256 (1-session control) to 32 (128-session W4) the moment we run out of slots — see [`errors/2026-05-02-bench-agent-load-a3-tool-resume-gate-miss.md`](../experience/errors/2026-05-02-bench-agent-load-a3-tool-resume-gate-miss.md). | **P1.** No way for a client to declare "this session is interactive, retain its KV until I disconnect". The W4 mission gate stays red until either the runtime grows retention, or the API lets the client buy it explicitly. |
| G5 | nexil dispatches concurrent agent turns alongside batch traffic (eval, distill, capture). | Scheduler treats all `IncomingRequest`s as equal-priority FIFO. Interactive traffic loses to batch when both contend for the same `--num-slots`. | **P1.** Without an `agent.kind` or `priority` knob, an agent host on a shared deployment cannot keep p99 TTFT bounded. |
| G6 | Eli stores conversation tape per `session_id` and may ask "is this session still resident?" before re-prompting. | No `GET /v1/agent/sessions/{id}` exists; clients must scrape `/v1/stats?format=json` and look up the session in the `sessions` map. | **P1.** Workable today via stats scraping; cleaner endpoint deferred. |
| G7 | OpenAI clients are tolerant of unknown response fields but agent-infer rejects unknown *request* fields (`#[serde(deny_unknown_fields)]`). | This is the right default — but it means *every* extension we propose must land here before any client can send it. | **Process item.** Each P0/P1 below either adds a field with `#[serde(default)]` or relaxes `deny_unknown_fields` — pick one canonical path. |

`/v1/stats` and per-session aggregates are already complete enough for
the harness side; the request/response surface is where the gaps live.

---

## 3. Proposed Extensions

Each extension lists: **wire schema** (JSON), **rationale**, **stock-OpenAI
behavior**, and **internal plumbing**. Stock-OpenAI behavior is the test:
a request from the official `openai` Python client with no extension
fields must keep producing identical bytes on the wire. An extension
field that is `None` is omitted from responses
(`#[serde(skip_serializing_if = "Option::is_none")]`).

### 3.1 [P0] Tool-call streaming on `/v1/chat/completions` (G1)

**Schema (request):** unchanged. Drop the
`stream + tools = error` validation
(`openai_v1.rs:872`, `:1184`).

**Schema (stream chunk):** extend `ChatStreamChoice.delta` to mirror
OpenAI's published delta:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion.chunk",
  "created": 1714672900,
  "model": "Qwen3-4B",
  "choices": [{
    "index": 0,
    "delta": {
      "tool_calls": [{
        "index": 0,
        "id": "call_abc",
        "type": "function",
        "function": { "name": "lookup_doc", "arguments": "{\"q\":" }
      }]
    },
    "finish_reason": null
  }]
}
```

Closing chunk uses `finish_reason: "tool_calls"`. `id` is emitted on the
first delta for that tool_call only; subsequent deltas only add to
`function.arguments`.

**Rationale:** This is the only schema OpenAI publishes. Matching it
means nexil's `ToolCallAssembler` works untouched, and curl-level
benches (guidellm tool-call mode, vLLM's `--tool-call-parser` mode)
match without per-engine adapters.

**Stock-OpenAI behavior:** Identical to today when `tools=[]`. When
tools are present and `stream=true`, behavior matches openai-python's
contract verbatim.

**Internal plumbing:** Add a streaming tool-call extractor that pairs
with the existing `openai_parse_tool_calls(...)` path — same parser, but
incremental. Two implementation options:

1. *Greedy parse on text deltas:* buffer text until the parser yields a
   complete tool-call, then emit. Simpler but adds latency on first tool.
2. *Streaming parser:* port the assembler logic from
   `crates/nexil/src/clients/chat.rs` server-side so we emit deltas as
   the model produces tokens. Lower latency, more state.

Default to option 1 for the P0 land — it preserves correctness without
new state machines, and the lost latency on tool-only turns is bounded
(typically 1–2 deltas vs ~30 for content). Option 2 is a P1 follow-up
once we have a baseline.

**Acceptance:** the bench harness in §5 runs Eli's tool-loop end-to-end
without falling back to non-streaming.

### 3.2 [P0] `tool_choice` and `response_format` passthrough (G3)

**Schema (request):**

```json
{
  "tool_choice": "auto" | "none" | "required" |
                 { "type": "function", "function": { "name": "lookup_doc" } },
  "response_format": { "type": "text" } |
                     { "type": "json_object" } |
                     { "type": "json_schema", "json_schema": { ... } }
}
```

**Rationale:** nexil sends both on every tool-loop turn. Today they are
silently rejected by `deny_unknown_fields`. Even if the runtime cannot
honor `json_schema` constrained decoding yet, accepting the field and
falling back to plain JSON is strictly better than 400-ing the request.

**Stock-OpenAI behavior:** Identical when omitted. When present,
`tool_choice` shapes the chat-template render; `response_format=text`
is the existing default.

**Internal plumbing:** add the fields to `ChatCompletionRequest`,
default `None`, plumb `tool_choice` into `chat::openai_messages_to_prompt`
(it already understands the field per Qwen's chat template). For
`response_format` in P0: accept and remember; honor only `text` and
`json_object` (the latter just sets a "JSON-only" stop heuristic). A
dedicated structured-decoding plan covers `json_schema` separately.

**Acceptance:** A nexil tool-loop turn with `tool_choice="auto"` and
`response_format={type:"json_object"}` returns a valid response; missing
support for `json_schema` returns a deterministic 422 with a
`response_format.json_schema` parameter pointer.

### 3.3 [P0] Per-request prefix-reuse echo (G2)

**Schema (non-streaming response, additive):**

```json
{
  "id": "chatcmpl-...",
  "usage": {
    "prompt_tokens": 8571,
    "completion_tokens": 256,
    "total_tokens": 8827,
    "prefix_reuse": {
      "matched_prefix_tokens": 8256,
      "resume_prefill_tokens": 323,
      "session_affinity_hit": true,
      "prefix_hit_rate": 0.962,
      "prefix_skip_rate": 0.962
    }
  }
}
```

**Schema (streaming):** appears only on the final usage chunk emitted
when `stream_options.include_usage=true`. No per-content-chunk overhead.

**Rationale:** The exact numbers we measure in `/v1/stats?format=json`'s
`last_request` field. Echoing them under `usage.prefix_reuse` lets a
client verify per-request what we today only expose globally — closes the
race where two concurrent sessions overwrite `last_request`.

**Stock-OpenAI behavior:** Identical when the request omits the
opt-in flag. Behind a request-level switch:

```json
{ "include_prefix_reuse": true }
```

Default is `false`; when `false`, the field is omitted entirely. This
keeps openai-python's strict response-shape validators happy (they
typically tolerate extras under `usage`, but we won't gamble on it).

**Internal plumbing:** Reuse `LatestRequestCache::record(...)`'s inputs.
Pass the per-request snapshot through `RequestExecutionOptions` →
`IncomingRequest` → completion result, then serialize when the request
opted in.

**Acceptance:** With `include_prefix_reuse=true`, two concurrent W4
sessions show distinct `prefix_reuse.matched_prefix_tokens` numbers in
their respective responses, even when their windows overlap.

### 3.4 [P1] KV-reuse hints — `cache_strategy` (G4)

**Schema (request, additive):**

```json
{
  "session_id": "agent-019",
  "cache_strategy": {
    "policy": "session_pin" | "prefer_prefix" | "ephemeral",
    "retention_tokens": 8192,
    "ttl_seconds": 600
  }
}
```

| policy | meaning |
|---|---|
| `session_pin` | Treat this session's KV as eviction-resistant. Scheduler may evict only when no `ephemeral` traffic can be evicted first. |
| `prefer_prefix` | Default. Best-effort prefix reuse (current behavior). |
| `ephemeral` | Opt-in eviction-first. Bench traffic, distractor cold sessions, oneshot scripts. |

**Rationale:** The W4 5/2 entry was unambiguous: A1 + A2 give us correct
session lookup, but the runtime evicts the warm 8k prefix before resume
can use it. The runtime fix lives elsewhere (`tiered-kv-hicache-readmission.md`),
but the API has to give the *client* a way to declare which sessions
matter, otherwise the runtime cannot tell apart Eli's interactive REPL
session from a 128-cold-distractor sweep.

**Stock-OpenAI behavior:** Identical when omitted. `policy=prefer_prefix`
is the default and matches today's behavior; `session_pin` and
`ephemeral` are new behaviors that only activate on opt-in.

**Internal plumbing:** Lands as `Option<CacheStrategy>` on
`RequestExecutionOptions`. Scheduler's prefix-admission path
(`infer/src/scheduler/`, owned by Track A — coordinate before editing)
reads it from `IncomingRequest`. **HTTP layer alone is not enough** — the
admission policy change is a scheduler edit. P1 lands the API surface
behind a feature flag, then a scheduler PR honors it.

**Acceptance:** Re-run the W4 5/2 trace with `cache_strategy=session_pin`
on the warmup turns. `matched_prefix_tokens` rises above ~7k on at
least 90% of resume turns; `prefix_hit_rate` matches but
`avoided_prefill_ratio` rises from 0.4% toward the W4-H1 0.90 gate.

### 3.5 [P1] Prefix-cache-aware batching — `agent` envelope (G5)

**Schema (request, additive):**

```json
{
  "agent": {
    "kind": "interactive" | "batch" | "oneshot",
    "priority": 1
  }
}
```

`priority` is a small integer (default 0, range -10..=10). Higher wins.

**Rationale:** A shared deployment runs Eli (interactive), nightly
distill (batch), and a debug REPL turn (oneshot) against the same
backend. Without a hint, all three queue equally. The smallest workable
shape is a 3-class kind plus a tie-breaker integer; both are widely
understood (vLLM's `priority`, SGLang's `priority`).

**Stock-OpenAI behavior:** Identical when omitted. `kind=interactive`
preempts `kind=batch` only when the scheduler's batch slice is
oversubscribed; under light load every kind admits in FIFO.

**Internal plumbing:** Same path as `cache_strategy`. Scheduler change
lives in Track A; HTTP-only PR adds the field and stashes it on the
request. Until the scheduler honors it, the field is observational.

**Acceptance:** A bench scenario that mixes Eli interactive turns
(`kind=interactive`) with a 128-cold-session sweep (`kind=batch`) keeps
Eli's TTFT p99 within 2× of solo-Eli TTFT p99, where today it degrades
unboundedly.

### 3.6 [P1] Session inspect & evict — `/v1/agent/sessions/{id}` (G6)

**Routes:**

```
GET    /v1/agent/sessions          → { sessions: [...] }
GET    /v1/agent/sessions/{id}     → { session_id, prefix_lookups, ... }
DELETE /v1/agent/sessions/{id}     → 204
```

**Rationale:** Today a client polls `/v1/stats?format=json` and grovels
through the `sessions` map. A scoped resource is cleaner and lets us
expose per-session retention without bloating the global stats payload.
`DELETE` lets a client release a long-lived agent that is going away
(end of REPL, gateway disconnect) so the runtime can reclaim the KV
without waiting for an LRU pass.

**Stock-OpenAI behavior:** New routes; OpenAI clients never call them.

**Internal plumbing:** Read from `metrics.session_cache`; `DELETE` calls
into the scheduler's session-evict path (must coordinate with Track A).

### 3.7 [P1] Streaming parser for tool-call deltas (G1 follow-up)

The P0 land in §3.1 batches tool-call emission until the parser sees a
complete call. P1 ports the streaming assembler so we emit `function.name`
on the first chunk and `function.arguments` chunk-by-chunk, matching
openai-python's `delta.tool_calls[i].function.arguments` semantics.

**Acceptance:** TTFT-to-first-tool-call delta on a single tool turn drops
from "complete tool" → "first chunk of tool name", measured by the §5
harness's `ttft_tool_first_arg_chunk` metric.

---

## 4. Phasing

P0 lands in a single tranche; the binary is shipped before any P1 work
begins. Each P1 lands independently behind its own feature surface.

### P0 — unblock nexil today

| Slice | What | Acceptance |
|---|---|---|
| P0.1 | §3.1 streaming tool-call deltas (greedy parse) | nexil tool-loop streams without falling back to non-streaming |
| P0.2 | §3.2 `tool_choice` + `response_format` passthrough | `deny_unknown_fields` no longer rejects nexil's standard tool-loop body |
| P0.3 | §3.3 `usage.prefix_reuse` opt-in echo | client confirms session affinity from response, no `/v1/stats` poll |
| P0.x | §5 harness ([`scripts/bench_eli_agent.sh`](../../scripts/bench_eli_agent.sh)) replays a trace through Eli → infer Metal | guidellm-shape JSON drops into `bench-output/<date>-bench-eli-agent-<label>/` |

P0 does not need any scheduler changes. All edits are in
`infer/src/http_server/` plus the chat-template wiring in `crates/chat`.

### P1 — agent-workload differentiation

| Slice | What | Coordinates with |
|---|---|---|
| P1.1 | §3.4 `cache_strategy` (API surface only) | Track A scheduler plan for retention/admission |
| P1.2 | §3.5 `agent.kind` + `priority` (API surface only) | Track A scheduler plan |
| P1.3 | §3.6 `/v1/agent/sessions/*` routes | Track A scheduler eviction hook |
| P1.4 | §3.7 streaming tool-call assembler (delta-grain) | none |
| P1.5 | scheduler honors §3.4 + §3.5 hints | Track A — separate plan |

P1 lands the API surface independently of the scheduler honoring it.
That keeps the API stable while the runtime work iterates.

---

## 5. Bench harness ties

The plan above is validated by [`scripts/bench_eli_agent.sh`](../../scripts/bench_eli_agent.sh)
(opened in this same tranche). The harness:

- Boots `infer` with the Metal backend at a fixed port and model.
- Boots Eli configured to point its `nexil` provider at that infer.
- Replays a fixed 4-session trace where each session has 2–3 turns and at
  least one tool call (per the W3/W4 protocol — single-host scaled-down
  replica of `agent-w3-short-multiturn` plus an `agent-w4-tool-resume`
  flavor).
- Captures guidellm-shape metrics (TTFT, ITL, tok/s, req/s) and the new
  `usage.prefix_reuse` echo.
- Snapshots `/v1/stats` before/during/after, matching the bench-and-trace
  spec.

Real numbers ship as a `wins/` entry only after the P0 land. Until then
the harness ships scaffold-only (per the bench-and-trace-spec
exemption — pure tooling additions don't need a bench entry).

---

## 6. Open questions

For ckl to resolve before the P0 implementation tranche opens:

1. **`response_format=json_schema`** — accept + ignore, or 422? Tracking
   structured decoding properly is a separate plan; the choice here is
   purely about whether to fail fast.
2. **`include_prefix_reuse=true` default** — leave opt-in, or flip the
   default once we are happy that openai-python's response validators
   don't trip on extra `usage` fields? Conservative call: opt-in until
   we have a clean nexil run.
3. **`cache_strategy.policy=session_pin` collision with `--num-slots`** —
   what does the runtime do when more sessions request `session_pin` than
   we have slots? Reject the new request, downgrade silently to
   `prefer_prefix`, or evict an older `session_pin` LRU? A failure mode
   here is silent degradation.
4. **`agent.priority` ceiling** — bound to a small integer range so
   clients cannot starve each other (e.g. `[-10, 10]`)? Pick the bound
   in §3.5 or leave it open until a real consumer wants it.
5. **`/v1/agent/sessions/{id}` DELETE auth** — same bearer token, or a
   dedicated admin token? Needed before the route lands.
6. **Chat-template support for `tool_choice="required"`** — does Qwen3
   already render that case correctly, or do we need a shim in
   `crates/chat`?
7. **Track A coordination** — confirm with the Track A driver that the
   API-only P1 lands (`cache_strategy`, `agent.kind`,
   `/v1/agent/sessions/*` routes) do not conflict with the scheduler
   work in flight. If they do, defer P1.5 explicitly to the scheduler
   plan.

---

## 7. Out of scope (intentional)

- **Multi-tenant isolation** — `session_id` is a routing hint, not a
  tenancy boundary. Auth/quota lives elsewhere.
- **Beam search / `n>1` sampling** — current API rejects `n>1`; not
  changing that here.
- **Multimodal input** — text-only validators stay (`openai_v1.rs:248`).
- **Speculative-decode overrides** — `SpecConfig` already exists; not
  reshaping it.
- **Train control plane** — the `/v1/train/*` proxy is owned by the
  train workstream.
