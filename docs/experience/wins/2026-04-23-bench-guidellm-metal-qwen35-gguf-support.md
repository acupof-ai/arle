# Metal Qwen3.5 GGUF support pending canonical bench

## Goal

- Land end-to-end Metal GGUF support for dense Qwen3.5 models, validated on
  the smallest practical Qwen3.5 GGUF checkpoint, while keeping the existing
  CUDA GGUF path type-check clean.

## Hypothesis

- Reusing the shared Rust GGUF parser and Qwen3.5 tensor-reorder logic should
  let Metal load Q4_K GGUF checkpoints via load-time BF16 dequant without
  creating a second format-specific loader stack.

## Command

```bash
scripts/bench_guidellm.sh metal-qwen35-0p8b-gguf \
  --target http://127.0.0.1:8010 \
  --model models/Qwen3.5-0.8B-GGUF \
  --processor Qwen/Qwen3.5-0.8B
```

Invoked via: `scripts/bench_guidellm.sh metal-qwen35-0p8b-gguf --target http://127.0.0.1:8010 --model models/Qwen3.5-0.8B-GGUF --processor Qwen/Qwen3.5-0.8B`

## Environment

- **Backend:** metal
- **Model:** `models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf`
- **Hardware:** Apple M4 Pro, 48 GB RAM
- **Commit:** `620a488`
- **Feature set:** `cargo build --release -p infer --no-default-features --features metal,no-cuda`
- **Non-default flags / env vars:** none
- **Server launch:** `./target/release/metal_serve --model-path models/Qwen3.5-0.8B-GGUF --port 8010 --warmup 0`

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh metal-qwen35-0p8b-gguf`

## Results — sweep headline table

| rate (req/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| pending-local-rerun | pending-local-rerun | pending-local-rerun | pending-local-rerun | pending-local-rerun | pending-local-rerun | pending-local-rerun |

## Problems

- The feature itself is locally verified, but the canonical `guidellm` sweep
  was interrupted after the server path was validated because the locked
  4096-in/256-out sweep rapidly pushed the tiny 0.8B GGUF checkpoint into
  scheduler-capacity rejection (`Scheduler at capacity`) and exceeded the turn
  budget before producing a complete headline table.
- This entry therefore remains `pending-local-rerun`.

## Learnings

- Metal should share the same GGUF format layer as CUDA: tensor-name lookup,
  GGUF path detection, and Qwen3.5 V-head reorder logic now live in
  `infer/src/gguf.rs` instead of leaking through the CUDA-only loader module.
- The single-request and scheduler-backed Metal entrypoints must stay on one
  loader path; once `MetalBackend::load()` understood GGUF, rebuilding
  `metal_serve` immediately brought the runtime path along because the
  scheduler already delegates to that backend load step.
- The canonical sweep for this checkpoint should be rerun with the same locked
  params as a follow-up benchmark task, not guessed from the smoke request.

## Δ vs baseline

- **Baseline:** first Metal GGUF Qwen3.5 snapshot in this repo

## Artefacts

- Raw: `pending-local-rerun`
- CSV: `pending-local-rerun`
- HTML: `pending-local-rerun`
- Service trace: `bench-output/2026-04-23-metal-qwen35-0p8b-gguf/service_stats_trace.jsonl` (partial run)
- Server log: partial `metal_serve` tty log in the local session on 2026-04-23

## Notes

- Local verification completed:
  - `cargo check -p infer --no-default-features --features metal,no-cuda`
  - `cargo check -p infer --no-default-features --features cuda,no-cuda`
  - `cargo clippy -p infer --no-default-features --features metal,no-cuda -- -D warnings`
  - `cargo build --release -p infer --no-default-features --features metal,no-cuda --bin metal_request`
  - `cargo build --release -p infer --no-default-features --features metal,no-cuda --bin metal_serve`
  - `./target/release/metal_request --model models/Qwen3.5-0.8B-GGUF --prompt 'Hello' --raw-prompt --warmup 0 --max-new-tokens 8`
- Smoke result:
  - Load succeeded via GGUF + runtime-asset fallback.
  - Prompt tokens: `1`
  - Output tokens: `8`
  - TTFT: `128.8 ms`
  - Gen TPS: `136.1 tok/s`
- Release binary sizes after enabling `strip = "symbols"`, `lto = "thin"`,
  `codegen-units = 1`:
  - `target/release/metal_request`: ~19 MB
  - `target/release/metal_serve`: ~21 MB
  - `target/release/agent-infer`: ~25 MB
- CUDA/no-cuda `cargo clippy -D warnings` still fails in unrelated pre-existing
  modules outside this GGUF change; CUDA compile coverage for this work is the
  passing `cargo check -p infer --no-default-features --features cuda,no-cuda`.
