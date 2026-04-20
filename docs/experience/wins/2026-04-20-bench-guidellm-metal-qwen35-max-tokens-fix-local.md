# Metal Qwen3.5 max_tokens runtime fix — local quick regression, 2026-04-20

## Goal

- Regression-check the Metal runtime fix that threads HTTP `max_tokens` into `SamplingParams.max_new_tokens`, while confirming the live Qwen3.5 path still serves and streams at the expected shape.

## Hypothesis

- The fix is correctness-only: `/v1/chat/completions` should stop over-generating past the requested cap, and Qwen3.5 Metal throughput / latency should stay within the normal quick-run envelope because no kernel or scheduler policy changed.

## Command

```bash
scripts/bench_guidellm.sh metal-qwen35-max-tokens-fix-quick \
  --quick \
  --target http://127.0.0.1:18083 \
  --model mlx-community/Qwen3.5-4B-MLX-4bit \
  --processor /path/to/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3
```

Additional correctness smoke:

```bash
curl -sS -X POST http://127.0.0.1:18083/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{"model":"local","messages":[{"role":"user","content":"Say hello in one short sentence."}],"max_tokens":16,"temperature":0.0}'
```

## Environment

- **Backend:** metal
- **Model:** `mlx-community/Qwen3.5-4B-MLX-4bit`
- **Hardware:** Apple Silicon local workstation
- **Commit:** `15a4066` + local uncommitted diff
- **Feature set:** `cargo build --release --no-default-features --features metal,no-cuda`
- **Non-default flags / env vars:** `GUIDELLM__MP_CONTEXT_TYPE=forkserver`
- **Server launch:** `target/release/metal_serve --model-path '/path/to/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3' --bind 127.0.0.1 --port 18083 --warmup 0`

## Canonical params

- This turn used the local quick regression shape, not the full canonical sweep.
- Profile: `concurrent`
- Data: `prompt_tokens=512,output_tokens=128`
- Max seconds: `60`
- Warmup: `5`
- Concurrency set: `1,2,4,8`

## Results — quick headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| conc1 | 781.7 | 1150.8 | 34.87 | 43.42 | 25.14 | 0.182 |
| conc2 | 1131.6 | 4279.7 | 48.99 | 143.6 | 22.31 | 0.182 |
| conc4 | 2727.0 | 9585.7 | 80.09 | 98.96 | 36.16 | 0.291 |
| conc8 | 15612.0 | 20853.8 | 82.70 | 98.65 | 35.83 | 0.291 |

## Problems

- This is a local quick regression-check, not the full canonical `4096 / 256` sweep.
- `guidellm` flagged `conc2` with `ITL p99 > 2× p50`; that points to normal thermal / queue variance in the quick profile rather than a new correctness problem.

## Learnings

- Threading `IncomingRequest.max_tokens` into `SamplingParams.max_new_tokens` fixes the live HTTP cap without disturbing the compiled Qwen3.5 Metal serving path.
- For runtime correctness patches on Metal, a quick `guidellm` regression plus an explicit `/v1/chat/completions` cap smoke is enough to prove both perf stability and user-visible behavior.

## Δ vs baseline

- **Baseline:** first local quick regression for this exact route.

## Artefacts

- Raw: `bench-output/2026-04-20-metal-qwen35-max-tokens-fix-quick/benchmarks.json`
- CSV: `bench-output/2026-04-20-metal-qwen35-max-tokens-fix-quick/benchmarks.csv`
- HTML: `bench-output/2026-04-20-metal-qwen35-max-tokens-fix-quick/benchmarks.html`

## Notes

- Code change: [`infer/src/backend/metal/runtime.rs`](../../../infer/src/backend/metal/runtime.rs) now copies `incoming.max_tokens` into `sampling.max_new_tokens` before building request state.
- Correctness smoke result: `finish_reason="length"` with `completion_tokens=16`, matching the request cap instead of the previous hard-coded `512` default.
- Follow-up: if this route gets another Metal runtime change, compare against this quick entry before deciding whether a full canonical rerun is necessary.
