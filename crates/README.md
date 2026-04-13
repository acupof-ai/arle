# Workspace atomic crates

This folder hosts phase-1 extracted crates from `infer`:

- `infer-agent`: reusable CLI-agent session/control-plane logic.
- `infer-chat`: shared ChatML/tool-call protocol and OpenAI chat surface types.
- `infer-core`: shared domain types.
- `infer-observability`: shared event schema + sink trait.
- `infer-policy`: pluggable admission/chunking policy traits.
- `infer-tools`: reusable builtin tool definitions and sandboxed execution.

`agent-infer` now depends on `infer-agent`, `infer-chat`, and `infer-tools`
directly so the root crate stays focused on CLI wiring and backend selection.
