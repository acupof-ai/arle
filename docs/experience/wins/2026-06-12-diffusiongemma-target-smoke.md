# DiffusionGemma Target Smoke After Download

## Context

After the initial DiffusionGemma bridge landed, the target checkpoint was not
cached locally, so only compile and structural tests were available. The model
was then downloaded through the Hugging Face CLI into the default cache:

`~/.cache/huggingface/hub/models--mlx-community--diffusiongemma-26B-A4B-it-4bit/snapshots/252183330817f96e9cba0b20cc400b2947a575cf`

The downloaded snapshot is about 15 GiB in the HF cache and contains four
safetensor shards plus a standalone `chat_template.jinja`. `tokenizer_config.json`
still has `chat_template=null`, so ARLE must treat the external template file as
part of the checkpoint contract.

## What Worked

- `infer-server::OpenAiTokenizer` now resolves `chat_template.jinja` next to the
  tokenizer after checking inline `tokenizer_config.json` templates and before
  falling back to builtins / ChatML.
- `infer-api` now uses the normal tokenizer loader for Metal DiffusionGemma
  instead of forcing chat to fail closed.
- The real downloaded DiffusionGemma template renders under minijinja for a
  minimal user message.
- A low-impact Metal server loaded the downloaded checkpoint and answered both
  OpenAI completion surfaces.

Smoke command:

```bash
target/release/arle serve \
  --backend metal \
  --model-path mlx-community/diffusiongemma-26B-A4B-it-4bit \
  --port 8019 \
  --low-impact \
  --num-slots 1 \
  --total-pages 8 \
  --page-size 256 \
  --max-prompt-tokens 256 \
  --max-total-tokens 512 \
  --chunked-prefill-size 256
```

Readiness:

```json
{"object":"list","data":[{"id":"diffusiongemma-26B-A4B-it-4bit","object":"model","created":1781253971,"owned_by":"arle"}]}
```

Completions smoke:

```bash
curl -sS --max-time 180 http://127.0.0.1:8019/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model":"diffusiongemma-26B-A4B-it-4bit","prompt":"Hello","max_tokens":4,"temperature":0}'
```

```json
{"id":"cmpl-2ca8237304eb458886e84aabb3900a2b","object":"text_completion","created":1781253991,"model":"diffusiongemma-26B-A4B-it-4bit","choices":[{"text":"    ","index":0,"logprobs":null,"finish_reason":"length"}],"usage":{"prompt_tokens":1,"completion_tokens":4,"total_tokens":5}}
```

Chat smoke:

```bash
curl -sS --max-time 180 http://127.0.0.1:8019/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"diffusiongemma-26B-A4B-it-4bit","messages":[{"role":"user","content":"Say hi"}],"max_tokens":4,"temperature":0}'
```

```json
{"id":"chatcmpl-80cb2c2dc7bf45e9adfdae5c04dea574","object":"chat.completion","created":1781254012,"model":"diffusiongemma-26B-A4B-it-4bit","choices":[{"index":0,"message":{"role":"assistant","content":"hi!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":15,"completion_tokens":3,"total_tokens":18}}
```

## Verification

```bash
hf download mlx-community/diffusiongemma-26B-A4B-it-4bit --repo-type model
find ~/.cache/huggingface/hub/models--mlx-community--diffusiongemma-26B-A4B-it-4bit -name '*.incomplete' -print
cargo fmt --check
git diff --check
cargo test -p infer-server --release tokenizer::tests
cargo test -p infer-server --release real_checkpoint_tests::real_diffusion_gemma_external_template_renders_if_cached -- --nocapture
cargo test -p infer-api --release --no-default-features --features metal,no-cuda --lib
cargo clippy -p infer-server -p infer-api --release --no-default-features --features metal,no-cuda -- -D warnings
```

No throughput benchmark was run in this tranche. The smoke proves the downloaded
checkpoint loads and executes small completion/chat requests; production support
still needs a real prompt ladder, longer generations, memory pressure logging,
and a guidellm/latency snapshot.

## Rule

HF checkpoints may split chat templates into `chat_template.jinja` even when
`tokenizer_config.json` has `chat_template=null`. The OpenAI facade must resolve
that file before falling back to a builtin or ChatML renderer.
