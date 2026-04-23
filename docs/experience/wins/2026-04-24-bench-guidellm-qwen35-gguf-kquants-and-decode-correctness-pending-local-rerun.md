# Qwen3.5 GGUF K-quants + decode-loop correctness pending local rerun

## Goal

- Regression-check the shared GGUF K-quants fix and the Qwen3.5 Metal decode
  loop correction, with the concrete acceptance bar that local Metal GGUF
  generation matches `llama.cpp` on the same checkpoint.

## Hypothesis

- Fixing the shared `get_scale_min_k4` layout once across Rust/CUDA/scripts and
  materializing sampled tokens before feeding them back into the Qwen3.5 Metal
  step loop should restore correct GGUF text generation without regressing the
  safetensors path.

## Params

- Label: `metal-qwen35-0p8b-gguf-kquants-fix`
- Planned canonical command:
  `GUIDELLM__MP_CONTEXT_TYPE=forkserver ./scripts/bench_guidellm.sh metal-qwen35-0p8b-gguf-kquants-fix --target http://127.0.0.1:8010 --model models/Qwen3.5-0.8B-GGUF --processor Qwen/Qwen3.5-0.8B`
- Canonical profile intent: `prompt_tokens=4096,output_tokens=256`

## Env

- Local change date: `2026-04-24`
- Commit base: `d9c40e4`
- Local machine: `Apple M4 Pro / macOS 26.3.1(a)`
- Backend: `metal`
- Model under test: `models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf`

## Results

- Status: `pending-local-rerun`
- Local verification covers:
  - `cargo test -p infer --no-default-features --features metal,no-cuda gguf::tests::test_decode_scale_min_k4_matches_ggml_layout -- --exact`
  - `cargo clippy -p infer --no-default-features --features metal,no-cuda -- -D warnings`
  - `cargo check -p infer --no-default-features --features cpu,no-cuda`
  - `cargo check -p infer --tests --no-default-features --features cuda,no-cuda`
  - `cargo test -p infer --no-default-features --features metal,no-cuda --lib backend::metal::qwen35::tests::compare_qwen35_0p8b_gguf_incremental_decode_vs_full_replay -- --ignored --exact --nocapture`
  - `./target/release/metal_request --model models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf --prompt 'Hello' --raw-prompt --warmup 0 --max-new-tokens 8 --temperature 0 --top-k 1`
  - `/opt/homebrew/bin/llama-completion -m models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf -p 'Hello' -n 8 --temp 0 --top-k 1 --seed 0 --no-display-prompt --no-warmup -no-cnv`
  - `./target/release/metal_request --model models/Qwen3.5-0.8B --prompt 'Hello' --raw-prompt --warmup 0 --max-new-tokens 4 --temperature 0 --top-k 1`
- Observed local correctness result:
  - Metal GGUF: `Hello -> ",\n\nI am writing a Python script"`
  - `llama.cpp` on the same GGUF: `Hello -> ",\n\nI am writing a Python script"`
  - Metal safetensors smoke still runs: `Hello -> ", I have a"`

## Problems

- The canonical `guidellm` sweep on this tiny `0.8B` checkpoint still did not
  produce a usable headline table. The run reached request submission capacity
  under the locked `4096/256` profile before emitting the normal benchmark
  artefacts.
- The output directory
  `bench-output/2026-04-24-metal-qwen35-0p8b-gguf-kquants-fix/` contains setup
  logs and service traces, but no completed `benchmarks.json/csv/html` set.

## Learnings

- GGUF K-quants bit packing must have a single canonical implementation across
  all surfaces: shared Rust dequant, CUDA native kernel, synthetic CUDA tests,
  and offline conversion scripts.
- For MLX/Metal decode loops, sampled token arrays are not safe to reuse as
  token inputs until they have been materialized. The safe pattern is:
  materialize token -> read token id -> build next step.
- Once the shared loader math was corrected, the remaining text corruption was
  not in GGUF weights at all; it was a decode-loop scheduling bug in the Metal
  Qwen3.5 runtime.

## Artefacts

- Partial bench directory:
  `bench-output/2026-04-24-metal-qwen35-0p8b-gguf-kquants-fix/`
- Present files:
  - `command.txt`
  - `guidellm.log`
  - `service_stats_before.txt`
  - `service_stats_trace.jsonl`

## Notes

- What changed in code since the prior pending GGUF source-layer cleanup entry:
  shared `Q4_K/Q5_K` decode fixed to match `ggml`, CUDA native `q4k` decode
  aligned to the same layout, and Qwen3.5 Metal decode now materializes
  sampled tokens before the next step.
- Follow-up: rerun the canonical `guidellm` sweep on a larger supported model
  or with a local profile that does not trivially saturate the 0.8B server
  admission limit, then append a normal headline table entry.
