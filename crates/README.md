# Workspace crates

This folder hosts the extracted workspace crates around `infer`:

- `infer-agent`: reusable CLI-agent session/control-plane logic.
- `infer-chat`: shared ChatML/tool-call protocol and OpenAI chat surface types.
- `infer-cli`: reusable REPL/CLI flow for the `agent-infer` binary.
- `infer-core`: shared domain types.
- `infer-engine`: runtime adapter boundary for backend-loaded engines; it must stay control-plane agnostic.
- `infer-observability`: shared event schema + sink trait.
- `infer-policy`: pluggable admission/chunking policy traits; runtime wiring is still pending.
- `infer-tools`: reusable builtin tool definitions and sandboxed execution.

`agent-infer` now depends on `infer-cli`, while `infer-cli` depends on
`infer-engine`, `infer-agent`, `infer-chat`, and `infer-tools`. The root package
is reduced to a thin binary wrapper.

This split is still an interim Phase 1 boundary, not the final atomized workspace. The `infer-engine` -> control-plane back edge has been removed; the remaining work is deeper scheduler/KV/runtime extraction plus fuller policy wiring in later phases.
