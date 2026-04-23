# Route A GPU Validation Prompt

Use this checklist when validating the 2026-04-15 Route-A refactor + the
follow-up consolidation pass on a CUDA host after Darwin verification has
landed.

## Goal

Confirm that (a) folding the abandoned runtime split back into `infer` and
(b) collapsing the duplicate `agent_engine` facade into `server_engine` only
changed package/module structure and naming, not build behavior or runtime
semantics.

## Commands

Run these from the repo root, in order:

```bash
cargo fmt --all -- --check
cargo check --workspace --no-default-features --features no-cuda
cargo check -p infer --no-default-features --features cuda,no-cuda
cargo check --workspace --no-default-features --features metal
CUDA_HOME=/usr/local/cuda cargo build --release
cargo test --release --lib
INFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release --test e2e
cargo test --release --test e2e_qwen35
scripts/bench_guidellm.sh post-route-a-cuda
```

## What to inspect

- `infer/src/lib.rs` has no legacy `#[path]` redirects.
- `infer/src/agent_engine.rs` does **not** exist (deleted; its responsibilities
  collapsed into `infer/src/server_engine.rs`).
- `infer/src/types.rs`, `infer/src/events.rs`, `infer/src/scheduler/policy.rs`
  exist as folded-in modules.
- `infer/src/server_engine.rs` exposes the canonical surface: `InferenceEngine`
  trait, `LoadedInferenceEngine` enum (CUDA + Metal + CPU variants),
  `CompletionRequest`, `CompletionOutput`, `TokenUsage`,
  `CompletionStreamDelta`, `ModelInferenceEngine<M>`, and the
  `Qwen3InferenceEngine` / `Qwen35InferenceEngine` aliases.
- `Cargo.toml` workspace members do not list `infer-core`, `infer-engine`,
  `infer-observability`, or `infer-policy`.
- `crates/` contains `agent`, `chat`, `cli`,
  `tools`, `mlx-sys`, `cuda-kernels` (added by the
  2026-04-15 `a4e12f5` kernel-crate extraction — the CUDA kernel
  Rust layer now lives here rather than under `infer/src/backend/cuda/`),
  and `README.md`.

## Final sweep

```bash
grep -rn 'AgentEngine\|AgentComplete\|LoadedAgentEngine\|RealServerEngine\|GenericServerEngine\|LoadedServerEngine\|\bServerEngine\b\|\bCompleteRequest\b\|\bCompleteOutput\b\|\bEngineOptions\b\|\bStreamDelta\b\|ProtocolChatMessage\|ProtocolToolCall\|ProtocolToolDefinition\|protocol_messages_to_prompt\|parse_protocol_tool_calls\|init_default_logging' --include='*.rs' --include='*.toml' .
```

Expected results:

- zero `.rs` hits
- zero `.toml` hits
- `.md` hits only in `docs/archives/cuda-crate-extraction.md` and
  `docs/experience/wins/2026-04-14-kv-quant-audit.md`
