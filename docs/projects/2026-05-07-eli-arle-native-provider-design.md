# Eli ↔ ARLE Native `nexil` Provider Design — 2026-05-07

This entry is the *concrete design* sequel to
[`docs/resources/eli-integration.md`](../resources/eli-integration.md).
The runbook covered Layer 1 (zero-code OpenAI adapter pointed at ARLE);
this entry locks the design for **Layer 2 — native `nexil`
`ArleAdapter`** so it can be implemented in atomic commits without
re-discovery.

Authoritative source paths cited below live in
`/path/to/code/eli/`. Read each citation when implementing.

## 1. Eli `nexil` architectural facts that bind us

- **`ProviderAdapter` is intentionally minimal** — `crates/nexil/src/adapter.rs:7`
  exposes only `build_request_url` + `build_request_body`. No init, no
  streaming hook, no telemetry hook, no per-call header hook.
- **Adapter dispatch is keyed by `TransportKind`, not provider name** —
  `crates/nexil/src/providers/mod.rs:7`. `&'static dyn ProviderAdapter`
  constants (`OPENAI_ADAPTER`, `ANTHROPIC_ADAPTER`) cover
  Completion/Responses and Messages respectively. There is no
  per-provider adapter slot today.
- **`ProviderConfig.custom_headers` is dead code** —
  `crates/nexil/src/core/provider_registry.rs:46` stores it; the field
  is set in tests (`provider_registry.rs:194`, `llm/tests.rs:915`) but
  **never read by `client_registry.rs::build_client`**
  (`crates/nexil/src/core/client_registry.rs:113`). Per-provider
  headers are only applied for anthropic and codex JWT inline. This
  is a load-bearing gap for any header-based extension.
- **Streaming is wholly central** — `clients/chat.rs::forward_sse_events`
  (`chat.rs:333`) + `SseDecoder` (`clients/parsing/sse.rs`) +
  `parser_for_transport` per chunk. Adapters never see SSE bytes.
- **Tool schema translation is shared** —
  `LLMCore::convert_single_tool` and `convert_tools_for_responses`
  (`request_builder.rs:139`, `261`). For Completion the tools array
  is passed through unchanged.
- **No nexil-level integration tests exist.** `crates/nexil/` has no
  `tests/` directory; convention is inline `#[cfg(test)]` modules.
- **`AGENTS.md` rules** (`/path/to/code/eli/AGENTS.md:21`):
  ≤15-line function bodies, no compatibility shims, no comments
  restating code, ">3 files = approach-first". This design touches
  six files, so the design itself satisfies the gate.

## 2. Wiring posture — dispatcher overlay

Three viable wirings; ranked:

1. **Overlay trait at `adapter_for_transport` (recommended).** Keep
   `OpenAIAdapter` as the Completion adapter; add `ArleAdapter` that
   *delegates* to it for body-shaping then post-processes. Pick adapter
   by **(provider_name, transport)**, not transport alone.
   Requires changing `providers/mod.rs::adapter_for_transport` to take
   `(provider_name, transport)`.
2. **Config-only ARLE.** Insufficient: `extra_headers` is dropped by
   `decide_responses_kwargs` (`request_builder.rs:206`); kwargs cannot
   reach outgoing HTTP headers.
3. **Fix `custom_headers` plumbing in `client_registry::build_client`
   first**, then ARLE is config-only. Cleaner long-term but mixes
   cross-cutting refactor into the same diff.

Sequence the work as **(1) then (3) in two commits**. Layer-1 thin
runbook stays unchanged for users who don't need extensions.

## 3. Module sketch — `crates/nexil/src/providers/arle.rs`

```rust
use std::sync::atomic::AtomicU64;
use serde_json::Value;

use crate::adapter::ProviderAdapter;
use crate::clients::parsing::TransportKind;
use crate::core::errors::{ConduitError, ErrorKind};
use crate::core::request_builder::TransportCallRequest;
use crate::providers::openai::OPENAI_ADAPTER;

pub static ARLE_ADAPTER: ArleAdapter = ArleAdapter {
    stats_poll_count: AtomicU64::new(0),
};

pub struct ArleAdapter {
    stats_poll_count: AtomicU64, // optional observability surface
}

impl ProviderAdapter for ArleAdapter {
    fn build_request_url(&self, api_base: &str, transport: TransportKind) -> String {
        OPENAI_ADAPTER.build_request_url(api_base, transport)
    }

    fn build_request_body(
        &self,
        request: &TransportCallRequest,
        transport: TransportKind,
    ) -> Result<Value, ConduitError> {
        if transport != TransportKind::Completion {
            return Err(ConduitError::new(
                ErrorKind::Config,
                "arle adapter only supports completion transport",
            ));
        }
        let mut body = OPENAI_ADAPTER.build_request_body(request, transport)?;
        Self::inject_arle_extensions(&mut body, request);
        Ok(body)
    }
}

impl ArleAdapter {
    fn inject_arle_extensions(body: &mut Value, req: &TransportCallRequest) {
        let Some(obj) = body.as_object_mut() else { return };
        if let Some(sid) = req.kwargs.get("session_id").cloned() {
            obj.entry("session_id").or_insert(sid);
        }
        if let Some(hint) = req.kwargs.get("prefix_hint").cloned() {
            obj.entry("prefix_hint").or_insert(hint);
        }
    }
}
```

Decisions justified:

- **Inherit `OpenAIAdapter::build_completion_body` verbatim** — ARLE
  serves the OpenAI Chat Completions wire format; duplicating
  body-shaping creates drift.
- **Reject Responses/Messages** — ARLE serves Completion only.
- **Atomic counter on the static**, not per-call state — adapters are
  `Send + Sync` singletons. Used by stats polling tooling, not request
  hot path.

## 4. Extension placement table

| Extension | Cleanest injection point | Why |
|---|---|---|
| `session_id` body field | `inject_arle_extensions`, reading `kwargs` | TransportCallRequest has no `session_id` field; kwargs is the existing escape hatch (already preserved by `decide_kwargs_for_provider`). ARLE accepts `session_id` in body per `infer/src/http_server/openai_v1.rs`. **Body field, not header** — sidesteps the `custom_headers` gap. |
| `prefix_hint` body field | Same as above | ARLE-native; OpenAI ignores unknown fields, safe to leave in kwargs even when it leaks. |
| `/v1/stats` polling | Sibling helper `crates/nexil/src/providers/arle_stats.rs` exposing `poll_stats(api_base, client) -> ArleStats` | Adapters only shape requests; they don't own a Tokio runtime. Telemetry is a separate module called by eli on its own cadence. |
| `X-Session-ID` HTTP header | Defer until Wiring (3) lands `custom_headers` plumbing | Body field covers Layer-1 use cases; header form is an optimization. |

## 5. Dispatcher change

`providers/mod.rs`:

```rust
pub fn adapter_for_request(
    provider_name: &str,
    transport: TransportKind,
) -> &'static dyn ProviderAdapter {
    use crate::core::provider_policies::normalized_provider_name;
    if transport == TransportKind::Completion
        && normalized_provider_name(provider_name) == "local"
        && provider_name.eq_ignore_ascii_case("arle")
    {
        return &arle::ARLE_ADAPTER;
    }
    adapter_for_transport(transport)
}
```

A new alias `"arle"` is added to `provider_policies::provider_alias`
(line 39) — **not collapsed onto `local`** so the dispatcher can
distinguish ARLE from generic OpenAI-compat servers. Justify this
deliberate split in the commit body.

Update call sites: `request_builder.rs:296-321` and
`execution.rs:323-333`. Keep the old `adapter_for_transport` for
forward compatibility while the rest of the codebase migrates.

## 6. Init signature

No async constructor. Provider registered via builder:

```rust
pub fn register_arle(reg: &mut ProviderRegistry, api_base: impl Into<String>) {
    reg.register("arle", ProviderConfig::new(api_base, ApiFormat::Completion));
}
```

Stats polling, when added, gets its own `ArleStatsPoller::spawn(handle, api_base)`
returning a `JoinHandle` — owned by eli, not nexil.

## 7. Integration test sketch

First file under `crates/nexil/tests/`. Use **axum** (already
transitive via `reqwest`) as the fake server; do not add
`wiremock`/`mockito` deps without ckl approval.

Three tests:

1. **Streaming chat completion with session + prefix hint.** Fake ARLE
   server asserts request body carries `session_id` and `prefix_hint`,
   emits two SSE deltas + `[DONE]`, client collects to `"hello"`.
2. **Negative scope test.** Same code path with provider = `local`
   (not `arle`); assert `session_id` does NOT appear in the body —
   proves dispatcher overlay is correctly scoped.
3. **Non-streaming request body shape.** Same body shape lands on
   `/v1/chat/completions` without SSE handling.

## 8. Ship plan

Files to touch (six total, exactly at the >3 approach-first gate):

1. `crates/nexil/src/providers/arle.rs` (new, ~50 lines)
2. `crates/nexil/src/providers/mod.rs` (dispatcher overlay)
3. `crates/nexil/src/core/provider_policies.rs:39` (alias)
4. `crates/nexil/src/core/provider_registry.rs:50` (built-in entry)
5. `crates/nexil/src/core/request_builder.rs:296-321` + `core/execution.rs:323-333` (call-site updates)
6. `crates/nexil/tests/arle_provider.rs` (new)

Land as **two commits in eli**:

- Commit A — `feat(nexil): native ArleAdapter with session/prefix
  body extensions` (files 1–5, the three tests in file 6).
- Commit B — `fix(nexil): plumb ProviderConfig.custom_headers through
  build_client` (separate cross-cutting fix; unblocks header form of
  ARLE extensions).

Each commit lands its own integration tests and is independently
revertable.

## 9. Cross-references

- ARLE OpenAI v1 surface:
  [`infer/src/http_server/AGENTS.md`](../../infer/src/http_server/AGENTS.md).
- Eli provider crate: `cklxx/eli/crates/nexil/src/providers/`.
- Eli runtime entry points: `eli/AGENTS.md`,
  `eli/docs/ARCHITECTURE_LANDSCAPE.md`.
- Layer 1 runbook (zero-code config approach):
  [`docs/resources/eli-integration.md`](../resources/eli-integration.md).
- Tier-A acceptance gates from gap analysis:
  [`2026-05-07-metal-world-first-strategy.md`](2026-05-07-metal-world-first-strategy.md).
