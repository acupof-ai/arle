# DiffusionGemma CLI Direct Chat Path

## Context

`mlx-community/diffusiongemma-26B-A4B-it-4bit` is a block-diffusion chat model,
not an autoregressive ChatML tool agent. The CLI REPL previously always used
the agent loop, which rendered the prompt with internal `<|im_start|>` ChatML
and installed a `<|im_end|>` stop string. On DiffusionGemma this could finish
with one special/stop token and no visible text.

## What Worked

- `infer-api::InferenceEngine` now exposes `render_chat_prompt`; the default
  remains ChatML, while `ServeInferenceEngine` delegates to the checkpoint
  `OpenAiTokenizer` chat template.
- `LoadedInferenceEngine` forwards `render_chat_prompt` to the concrete backend
  engine. Without this dispatch hop, CLI callers held behind the enum fell back
  to the trait's default ChatML renderer even though `ServeInferenceEngine` had
  the right checkpoint template.
- CLI `metal-diffusion-gemma` runs in direct chat mode: no tools, no agent
  protocol repair loop, and no ChatML stop string.
- The REPL banner shows `Mode: chat`, spinner text is `denoising...`, and empty
  replies report finish reason plus completion-token count.
- `--max-tokens auto` now uses DiffusionGemma `generation_config.max_new_tokens`
  before falling back to context length, so the CLI displays the generation cap
  instead of the 262144-token context window.

## Verification

```bash
cargo fmt --all --check
cargo check -p infer-api --release --no-default-features --features cpu,no-cuda --lib
cargo check -p cli --release --no-default-features --features cpu,no-cuda
cargo test -p cli --release --no-default-features --features cpu,no-cuda diffusion_backend_uses_direct_chat_template_path -- --nocapture
cargo test -p cli --release --no-default-features --features cpu,no-cuda resolve_tests -- --nocapture
cargo test -p cli --release --no-default-features --features cpu,no-cuda -- --nocapture
cargo test -p infer-server --release real_checkpoint_tests::real_diffusion_gemma_external_template_renders_if_cached -- --nocapture
cargo check -p cli --release --no-default-features --features metal,no-cuda
cargo build --release --no-default-features --features metal,no-cuda
```

Same-binary HTTP control (`arle serve`, explicit 22 GiB memory budget because
the default serve budget was below DiffusionGemma's fixed 21 GiB requirement):

```bash
curl -fsS http://127.0.0.1:8131/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"diffusiongemma-26B-A4B-it-4bit","messages":[{"role":"user","content":"Say hi"}],"max_tokens":32,"temperature":0}'
```

returned:

```text
content="Hi! How can I help you today?"
usage={"prompt_tokens":15,"completion_tokens":10,"total_tokens":25}
```

The original Chinese prompt:

```bash
curl -fsS http://127.0.0.1:8131/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"diffusiongemma-26B-A4B-it-4bit","messages":[{"role":"user","content":"你好呢"}],"max_tokens":32,"temperature":0}'
```

returned:

```text
content="你好！很高兴见到你。请问有什么我可以帮你的吗？或者我们只是随便聊聊天？"
usage={"prompt_tokens":15,"completion_tokens":23,"total_tokens":38}
```

CLI one-shot after rebuilding the actual root binary:

```bash
ARLE_DIFFUSION_TRACE=1 ./target/release/arle \
  --model-path mlx-community/diffusiongemma-26B-A4B-it-4bit \
  --max-tokens 4 \
  --temperature 0 \
  --trace /tmp/dg_cli_trace.jsonl \
  --non-interactive run --prompt 'Say hi' --no-tools
```

returned visible text and the same prompt-token count as HTTP:

```text
Hi! How can
diffusion generate complete: prompt_tokens=15 generated_tokens=4 blocks=1 denoise_steps=48 forced_commits=1 adaptive_commits=0 finish=Length
prompt_text="<bos><|turn>user\nSay hi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
```

Original REPL-shape smoke:

```bash
ARLE_DIFFUSION_TRACE=1 ./target/release/arle \
  --model-path mlx-community/diffusiongemma-26B-A4B-it-4bit \
  --max-tokens 32 \
  --temperature 0 \
  --non-interactive
```

then `你好呢` in a TTY printed:

```text
Mode: chat
Tools available: 0
diffusion generate complete: prompt_tokens=15 generated_tokens=23 blocks=1 denoise_steps=11 forced_commits=0 adaptive_commits=1 finish=Stop
你好！很高兴见到你。请问有什么我可以帮你的吗？或者我们只是随便聊聊天？
out 23 tok / 3.0s · 7.7 tok/s
```

No guidellm run was taken because this tranche changes CLI prompt rendering and
operator display, not a performance default or hot-path kernel.

## Rule

Do not route block-diffusion chat checkpoints through the autoregressive tool
agent prompt path. The model's checkpoint chat template owns the prompt format,
and the diffusion config owns EOS/stop behavior.
