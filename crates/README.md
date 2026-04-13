# Workspace atomic crates

This folder hosts phase-1 extracted crates from `infer`:

- `infer-core`: shared domain types.
- `infer-policy`: pluggable admission/chunking policy traits.
- `infer-observability`: shared event schema + sink trait.

`infer` currently re-exports these crates to keep compatibility while we migrate internals incrementally.
