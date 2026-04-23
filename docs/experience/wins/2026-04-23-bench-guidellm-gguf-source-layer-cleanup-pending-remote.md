# GGUF source-layer cleanup pending remote verification

## Goal

- Regression-check the GGUF source-layer cleanup that removes duplicated model
  resolution / tokenizer / runtime-assets fallback logic across CPU, Metal,
  and CUDA bootstrap paths.

## Hypothesis

- Centralizing model-source resolution should not change safetensors behavior
  and should preserve existing backend entry semantics; any remaining GGUF
  failures should localize to weight/config correctness rather than path
  discovery glue.

## Params

- Label: `gguf-source-layer-cleanup`
- Planned command: `scripts/bench_guidellm.sh gguf-source-layer-cleanup`
- Planned regression-check target: canonical CUDA bench host on a supported
  Qwen model
- Feature set: pending remote host selection

## Env

- Local change date: `2026-04-23`
- Local machine: `Apple M4 Pro / macOS 26.3.1(a)`
- Remote benchmark pending because this workspace is not the canonical CUDA
  bench host

## Results

- Status: `pending-remote`
- Local verification covers:
  - `cargo check -p infer --no-default-features --features metal,no-cuda`
  - `cargo check -p infer --no-default-features --features cuda,no-cuda`
  - `cargo check -p infer --no-default-features --features cpu,no-cuda`
  - `cargo test -p infer --no-default-features --features metal,no-cuda model_source -- --nocapture`
  - `cargo test -p infer --no-default-features --features cpu,no-cuda cpu_backend_ -- --nocapture`
  - `AGENT_INFER_GDR_METAL_KERNEL=0 ./target/debug/metal_request --model models/Qwen3.5-0.8B --prompt 'Hello' --raw-prompt --warmup 0 --max-new-tokens 1`
  - `AGENT_INFER_GDR_METAL_KERNEL=0 ./target/debug/metal_request --model models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf --prompt 'Hello' --raw-prompt --warmup 0 --max-new-tokens 1`

## Problems

- No remote `guidellm` run has been executed yet for this cleanup, so
  throughput / latency regression-check data are still pending.
- Local Metal validation shows the source-layer refactor did not regress the
  safetensors path (`Hello -> ","` still holds), but `Qwen3.5-0.8B-Q4_K_M.gguf`
  remains incorrect from the first generated token (`Hello -> "ois"`), which
  localizes the remaining bug to the GGUF weight / dequant path rather than
  model-source discovery.

## Learnings

- GGUF path discovery, runtime-assets fallback, and tokenizer extraction are
  cross-backend concerns and should exist exactly once.
- Once the entry glue is unified, the remaining correctness surface becomes
  much easier to isolate: backend-safe safetensors behavior stayed intact,
  while GGUF failures now point directly at tensor transforms / dequant logic.
