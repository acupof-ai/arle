# Gemma4 CLI Image Input + VLM Smoke Bench

## Context

Added CLI image input for the Gemma4 VLM path:

- One-shot: `arle --model-path /tmp/arle-gemma4-e2b-it-4bit run --prompt "..." --image <path-or-url>`
- REPL: `/image <path-or-url>` attaches a local path or HTTP(S) image to the next chat turn.

The implementation uses the same Gemma4 image preprocessing and soft-token marker expansion as `/v1/chat/completions`; non-VLM backends fail closed instead of silently dropping images.

## What Worked

Correctness smoke used a network JPEG URL:

```bash
./target/release/arle \
  --model-path /tmp/arle-gemma4-e2b-it-4bit \
  --max-tokens 32 \
  run \
  --prompt "What animal is in this image? Answer in one short sentence." \
  --image https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg \
  --json
```

Output:

```json
{
  "model_id": "arle-gemma4-e2b-it-4bit",
  "backend": "metal-gemma4",
  "text": "The image shows a **cat**.",
  "prompt_tokens": 280,
  "completion_tokens": 8,
  "total_tokens": 288,
  "image_count": 1
}
```

## VLM Waterline

This is a controlled single-image, single-request HTTP microbench, not a guidellm SLO sweep. guidellm is text-only for the current harness, so this entry is a VLM smoke/perf waterline and cannot license a default/performance claim beyond this shape.

Environment:

- Backend: Metal
- Model: `/tmp/arle-gemma4-e2b-it-4bit`
- Hardware: Apple M4 Pro, 48 GiB unified memory
- Metal toolchain: `Apple metal version 32023.883`
- Base commit before this change: `b26f0bc5`
- Feature set: `cargo build --release --no-default-features --features metal,no-cuda`
- Server: `./target/release/arle --model-path /tmp/arle-gemma4-e2b-it-4bit serve --backend metal --port 8027 --bind 127.0.0.1`

Single image prompt:

- Image: Wikimedia `Cat03.jpg`, 279603 bytes, embedded as OpenAI data URL.
- Prompt: `What animal is in this image? Answer in one short sentence.`
- Max output: 32 tokens
- Warmup: 1 request
- Timed requests: 10, serial concurrency=1

Results:

| Shape | Prompt tokens | Output tokens | Avg latency | P50 latency | P95 latency | Output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 image short answer | 280 | 8 | 1.042s | 1.036s | 1.070s | `The image shows a **cat**.` |
| text-only short answer | 15 | 4 | 0.297s | 0.298s | 0.300s | `**Cat**` |

Interpretation for this shape: one-image VLM adds about 0.74s over the text-only short-answer baseline on this M4 Pro run. The image path is stable and correct for the cat smoke, but this is not a throughput or multi-image benchmark.

## Verification

```bash
cargo test -p infer-server --release --lib
cargo test -p infer-api --release --no-default-features --features metal,no-cuda --lib
cargo test -p cli --release --no-default-features --features metal,no-cuda
cargo build --release --no-default-features --features metal,no-cuda
cargo clippy -p infer-api --release --no-default-features --features metal,no-cuda --lib -- -D warnings
cargo clippy -p cli --release --no-default-features --features metal,no-cuda -- -D warnings
git diff --check
```

## Rule

CLI multimodal support must share the server-side preprocessing/rendering path and fail closed on backends without image soft-token support. A VLM perf claim needs a VLM-shaped harness; text-only guidellm does not substitute for image-prefill evidence.
