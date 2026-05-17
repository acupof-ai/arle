# Eli ↔ ARLE Integration Runbook

[Eli](https://github.com/cklxx/eli) is a hook-first AI agent framework
in Rust (single binary, append-only tape, governed self-evolution). Its
LLM layer (`nexil`) is provider-agnostic and ships first-party adapters
for OpenAI- and Anthropic-shaped APIs.

Because ARLE already exposes an OpenAI-compatible HTTP server
(`infer/src/http_server/openai_v1.rs`), the integration is layered.
Pick the layer that matches what you need.

## Layer 1 — Zero-code: `nexil` OpenAI adapter pointed at ARLE

This is the recommended starting point. Eli's nexil layer does not
need to know anything ARLE-specific.

1. Run ARLE's OpenAI-compatible server (Metal example):

   ```bash
   cargo run --release --no-default-features --features metal -- \
     serve --backend metal --model models/Qwen3.5-0.8B-MLX-4bit \
     --host 127.0.0.1 --port 8080
   ```

2. Point Eli's nexil at ARLE. In Eli's provider config (see
   `eli/crates/nexil/src/core/provider_registry.rs` for the schema —
   typically a TOML file under `~/.eli/`):

   ```toml
   [providers.arle-local]
   type = "openai"
   api_base = "http://127.0.0.1:8080/v1"
   api_key_env = "ARLE_API_KEY"   # any value; ARLE does not enforce
   default_model = "Qwen3.5-0.8B"
   ```

3. Run `eli chat --provider arle-local`. nexil will call
   `POST {api_base}/chat/completions` (and `/responses` if used) — the
   same shape ARLE's `openai_v1.rs` already serves.

**What this gives you:** `eli chat`, `eli run "..."`, and `eli gateway`
all routed through ARLE on whatever backend ARLE was started with
(CUDA / Metal). Nothing else changes.

**What this does not give you:** ARLE-specific extensions — session
affinity (`session_id`), prefix-cache hints, `/v1/stats` introspection,
streaming `/v1/responses` parity quirks. Use Layer 2 for those.

## Layer 2 — Native `nexil` provider for ARLE-specific extensions

Add a new adapter in Eli alongside the existing
`crates/nexil/src/providers/{openai,anthropic}.rs`. Suggested file:
`crates/nexil/src/providers/arle.rs` implementing `ProviderAdapter`.

The adapter URL routing is identical to OpenAI; the value-add is:

- Pass-through of `session_id` so ARLE's session-affinity slot tracking
  ([`infer/src/scheduler/AGENTS.md`](../../infer/src/scheduler/AGENTS.md)
  invariant 9) stays warm across an Eli session.
- Optional `X-ARLE-Prefix-Hint` header for shared-prefix agentic loops
  (depends on the radix prefix cache landing on Metal — see
  `2026-05-07-metal-world-first-gap-analysis.md` Tier A #1).
- `/v1/stats` polling for live envelope/ITL telemetry surfaced into
  Eli's tape.
- Honors ARLE's reasoning-effort knobs without remapping through the
  generic OpenAI `reasoning_effort` field.

Lives entirely in the Eli repo; no `arle-` crate dependency required.
Pure HTTP.

## Layer 3 — Embedded ARLE as an in-process backend

Heaviest integration. Useful when Eli is deployed alongside ARLE on
the same Apple Silicon host and you want to skip the loopback HTTP
hop. Two options:

- **`arle-embed` thin façade crate.** Exports
  `arle_embed::serve_in_process(config) -> Engine` so Eli can
  `cargo add arle-embed --features metal` and call it directly. Avoids
  pulling Eli into the ARLE workspace.
- **Sidecar IPC.** Eli's existing sidecar pattern (already used for
  Feishu/Slack/Discord) runs ARLE as a child process and speaks JSONL
  over stdio. No additional crate, but it loses session affinity
  unless we replicate Layer 2 hints over the sidecar protocol.

Recommend deferring Layer 3 until Layer 1 + 2 are battle-tested in
production.

## Open questions

- Should `arle serve` accept an `--eli-compat` flag that pins
  model-name strings to a stable shape Eli's tape expects? (low cost,
  high diagnosability)
- Should ARLE expose `/v1/responses` SSE deltas matching OpenAI's
  Responses API event taxonomy? (already partially in
  `openai_v1.rs`; verify under Eli's tool-call workload).
- Authoritative location for the integration tests: ARLE side or Eli
  side? The Eli side is simpler because Eli already has provider
  conformance suites.

## Cross-references

- ARLE OpenAI v1 surface:
  [`infer/src/http_server/AGENTS.md`](../../infer/src/http_server/AGENTS.md).
- Eli provider crate: `cklxx/eli/crates/nexil/src/providers/`.
- Eli runtime entry points: `eli/AGENTS.md`,
  `eli/docs/ARCHITECTURE_LANDSCAPE.md`.
