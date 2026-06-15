# Gemma4 Metal E2B 4bit Smoke

## Context

Gemma4 4bit MLX checkpoints were previously detected only as "not Qwen" and
fell through to the Qwen Metal loader, which failed before any Gemma4-specific
weight mapping could run. The target smallest smoke model was
`mlx-community/gemma-4-e2b-it-4bit`.

## What Worked

- `gemma-spec` now exposes the Gemma4 PLE and KV-share contract used by E2B/E4B:
  local/global attention dimensions, shared-KV donor lookup, and double-wide MLP
  sizing for the shared suffix.
- `infer-metal` now has a separate Gemma4 route with the `language_model.model`
  weight prefix, PLE weight registration, KV-share donor ids, and a C++/MLX
  greedy causal-generate entry.
- `infer-api` routes normal Gemma4 to `metal-gemma4` instead of the Qwen or
  DiffusionGemma routes.
- `/v1/chat/completions` now accepts OpenAI content part arrays at the schema
  boundary. Media parts fail closed because the vision/audio soft-token bridge
  is not wired yet; text parts continue through the checkpoint Jinja renderer.

## Verification

```bash
cargo test -p gemma-spec --release
cargo test -p infer-server --release --lib
cargo check -p infer-api --release --no-default-features --features metal,no-cuda --lib
cargo build --release --no-default-features --features metal,no-cuda
hf download mlx-community/gemma-4-e2b-it-4bit --local-dir /tmp/arle-gemma4-e2b-it-4bit
./target/release/arle serve --backend metal \
  --model-path /tmp/arle-gemma4-e2b-it-4bit \
  --port 8017 --low-impact --max-total-tokens 4096 \
  --max-prompt-tokens 2048 --total-pages 512
curl -sS http://127.0.0.1:8017/v1/models
curl -sS http://127.0.0.1:8017/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello in one short sentence."}],"max_tokens":8,"temperature":0}'
curl -sS http://127.0.0.1:8017/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":[{"type":"text","text":"what is in this image?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}],"max_tokens":4}'
curl -L --fail -o /tmp/arle-gemma4-test-cat.jpg \
  https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg
IMG=$(base64 < /tmp/arle-gemma4-test-cat.jpg | tr -d '\n')
jq -n --arg url "data:image/jpeg;base64,$IMG" \
  '{messages:[{role:"user",content:[{type:"text",text:"What is in this image? Answer briefly."},{type:"image_url",image_url:{url:$url}}]}],max_tokens:32,temperature:0}' \
  | curl -sS -i http://127.0.0.1:8017/v1/chat/completions \
      -H 'content-type: application/json' --data-binary @-
```

Results:

- `gemma-spec`: 4 tests passed.
- `infer-server`: 30 tests passed.
- Metal `infer-api` typecheck passed.
- Release Metal build passed.
- `/v1/models` returned `arle-gemma4-e2b-it-4bit`.
- Text chat returned `Hello!` with `prompt_tokens=16`, `completion_tokens=3`.
- Image content request returned the explicit 400:
  `image/audio/video content parts are parsed, but VLM soft-token embeddings are not wired yet`.
- A real network image probe used
  `https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg`
  (`/tmp/arle-gemma4-test-cat.jpg`, JPEG 1600x1598, 276K). Sent as a
  `data:image/jpeg;base64,...` OpenAI content part, it returned the same 400.

No guidellm benchmark was run: this tranche is model-onboarding and correctness
wiring, not a performance default flip. A canonical Gemma4 Metal sweep remains
pending after the real VLM soft-token bridge lands.

## Rule

New model-family routing must fail closed at every unsupported capability
boundary. A VLM checkpoint is not a VLM route until image/audio bytes are turned
into the model's soft-token embeddings; rendering `<|image|>` alone is not
evidence of VLM support.
