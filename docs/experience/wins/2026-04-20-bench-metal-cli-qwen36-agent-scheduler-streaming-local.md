# Metal CLI Qwen3.6 agent path uses scheduler + visible streaming, 2026-04-20

## Goal

- Diagnosis + regression-check: confirm the local CLI / REPL Metal path for `mlx-community/Qwen3.6-35B-A3B-4bit` no longer uses the slow direct `BackendInferenceEngine<MetalBackend>` generate path, and confirm agent mode now surfaces visible streaming text + live tool trace output.

## Hypothesis

- Switching `LoadedInferenceEngine::Metal` to the scheduler-backed request handle should make the CLI log `Metal scheduler runtime started` instead of entering the direct `C++ full generate` path that previously produced the user-reported `prefill 26.9 tok/s / decode 7.9 tok/s` shape.
- Agent-mode callbacks should stream user-visible text inline and print tool execution traces as they happen, rather than buffering the whole turn until completion.

## Command

Baseline log reported by the user before this fix:

```text
2026-04-20T21:47:04.473195+08:00   INFO infer::backend::metal::qwen35: qwen35.rs:1080 Metal forward path: C++ full generate (all in C++)
2026-04-20T21:47:19.980471+08:00   INFO infer::backend::metal::qwen35: qwen35.rs:1105   prefill 147 tokens (26.9 tok/s, 5467.5ms) decode 79 tokens (7.9 tok/s, 10039.6ms)
```

Local post-fix CLI smoke 1 (visible text streaming):

```bash
env RUST_LOG=info sh -c "printf 'Use the python tool to compute 7 * 8. After the tool returns, answer with just the integer.\n/exit\n' | target/release/agent-infer --model-path '/path/to/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46' --tools --max-turns 4 --max-tokens 96"
```

Local post-fix CLI smoke 2 (live tool trace):

```bash
env RUST_LOG=info sh -c "printf 'Use the python tool to compute 123 * 456. Do not do the math mentally. If a tool is needed, emit only valid <tool_call> blocks and no other text first. After the tool returns, answer with just the integer.\n/exit\n' | target/release/agent-infer --model-path '/path/to/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46' --tools --max-turns 4 --max-tokens 256"
```

## Environment

- **Backend:** metal
- **Model:** `mlx-community/Qwen3.6-35B-A3B-4bit`
- **Hardware:** local Apple Silicon workstation
- **Commit:** `8e43451` + local uncommitted diff
- **Feature set:** `cargo build --release --workspace --no-default-features --features metal,no-cuda`
- **Non-default flags / env vars:** `RUST_LOG=info`
- **Entry point:** `target/release/agent-infer` CLI / REPL, not `metal_serve`

## Results

| check | result |
|---|---|
| CLI startup path | `infer::backend::metal::runtime: Metal scheduler runtime started` |
| Direct path regression | no `Metal forward path: C++ full generate (all in C++)` line in either post-fix CLI smoke |
| Agent visible text | smoke 1 printed assistant text inline during the turn instead of only after completion |
| Agent live tool trace | smoke 2 printed `[tool: python] {"code":"print(123 * 456)"}` and inline result `56088` during the same turn |
| Turn timing observed locally | smoke 1 printed `(10.7s)`; smoke 2 printed `(8.6s)` |

Representative post-fix startup log:

```text
2026-04-20T22:23:10.583549+08:00   INFO infer::backend::metal::runtime: runtime.rs:245 Metal live prefix cache enabled for Qwen3.5 snapshot replay: block_size=16, max_cached_tokens=8192
2026-04-20T22:23:10.583561+08:00   INFO infer::backend::metal::runtime: runtime.rs:737 Metal scheduler runtime started
```

Representative live tool trace:

```text
2026-04-20T22:23:19.047519+08:00   INFO agent: lib.rs:475 Recovered tool call(s) via deterministic extraction
2026-04-20T22:23:19.049963+08:00   INFO tools: lib.rs:827 Executing python snippet (sandbox-exec, 16 chars)

[tool: python] {"code":"print(123 * 456)"}
56088
```

## Problems

- This is a CLI / REPL local smoke, not a canonical `guidellm` HTTP sweep. The HTTP serving path was already scheduler-backed before this change, so rerunning `scripts/bench_guidellm.sh` here would not measure the code path that changed.
- The post-fix smoke is not prompt-identical to the user-reported slow baseline, so there is no honest apples-to-apples throughput delta row for the direct-path numbers.
- `cargo clippy --release --workspace --no-default-features --features metal,no-cuda -- -D warnings` still fails in unrelated pre-existing code at `crates/autograd/src/backend.rs:38` (`clippy::arc_with_non_send_sync`), outside this diff.

## Learnings

- The slow Qwen3.6 Metal CLI behavior came from the wrong engine wrapper, not from the scheduler runtime itself: `LoadedInferenceEngine::Metal` must use the request-handle / scheduler path to match `metal_serve`.
- Agent-mode streaming needs two layers: raw token deltas from the engine, plus a small visible-text filter that strips protocol-only blocks like `<tool_call>` and `<think>` before printing to the terminal.
- Live tool trace rendering belongs at tool-execution time, not after the entire agent turn returns, or the CLI still feels buffered even when the backend is streaming.

## Δ vs baseline

- **Baseline:** user-reported direct-path CLI log from 2026-04-20 21:47:04 +08:00
- **Delta summary:** direct CLI path changed from `BackendInferenceEngine<MetalBackend>` to scheduler-backed request submission; user-visible agent mode now streams text/tool events inline.

| metric | baseline | now | Δ |
|---|---|---|---|
| Metal CLI path | direct `C++ full generate` | `Metal scheduler runtime started` | fixed route |
| Agent terminal behavior | buffered final answer only | visible text + live tool trace | fixed UX |
| Prefill / decode tok-s | `26.9 / 7.9 tok/s` on user log | n/a in this non-like-for-like smoke | not comparable |

## Artefacts

- Terminal logs embedded in this entry; no separate `bench-output/` artefact directory because this turn changed the local CLI runtime path rather than the HTTP benchmark target.

## Notes

- Code changes for this diagnosis:
  - `infer/src/server_engine.rs`
  - `crates/agent/src/lib.rs`
  - `crates/cli/src/repl.rs`
- Validation run:
  - `cargo check --release --workspace --no-default-features --features metal,no-cuda`
  - `cargo test --release --workspace --no-default-features --features metal,no-cuda`
