# Workspace crates

This folder hosts the extracted workspace crates around `infer`:

- `infer-agent`: reusable CLI-agent session/control-plane logic.
- `infer-chat`: shared ChatML/tool-call protocol and OpenAI chat surface types.
- `infer-cli`: reusable REPL/CLI flow for the `agent-infer` binary.
- `infer-core`: shared domain types.
- `infer-engine`: adapter layer that exposes backend-loaded engines to control-plane crates.
- `infer-observability`: shared event schema + sink trait.
- `infer-policy`: pluggable admission/chunking policy traits.
- `infer-tools`: reusable builtin tool definitions and sandboxed execution.

`agent-infer` now depends on `infer-cli`, while `infer-cli` depends on
`infer-engine`, `infer-agent`, `infer-chat`, and `infer-tools`. The root package
is reduced to a thin binary wrapper.
